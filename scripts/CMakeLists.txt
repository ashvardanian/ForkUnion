# Shared logic for setting target properties for tests and examples
function (set_target_properties_for_forkunion_script target_name)
    target_link_libraries(${target_name} PRIVATE forkunion_header)

    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # This warning annoys us by reminding, that before GCC 4.6 the ABI for passing objects aligned to ≥32-bytes was
        # different... but that GCC version was released in 2014, over a decade ago.
        target_compile_options(${target_name} PRIVATE -Wno-psabi)
    endif ()

    # In release mode, enable optimizations
    if (CMAKE_BUILD_TYPE STREQUAL "Release")
        message(STATUS "Enabling optimizations for ${target_name}")
        target_compile_options(
            ${target_name} PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3 -march=native -mtune=native>
                                   $<$<CXX_COMPILER_ID:MSVC>:/O2 /Ob3>
        )
    endif ()

    # if we're not in Release mode
    if (NOT CMAKE_BUILD_TYPE STREQUAL "Release")
        # if we're on GCC/Clang-compatible compiler and not in Release, turn on TSAN here too
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
            OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
            OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"
        )
            # Using thread sanitizers caused all kinds of stalls on NUMA systems
            # ~~~
            # message(STATUS "Enabling ThreadSanitizer for ${target_name}")
            # target_compile_options(${target_name} PRIVATE -fsanitize=undefined,thread)
            # target_link_options(${target_name} PRIVATE -fsanitize=undefined,thread)
            # ~~~
            target_compile_options(${target_name} PRIVATE -g -O0)
            target_compile_options(${target_name} PRIVATE -fno-omit-frame-pointer)
            target_link_options(${target_name} PRIVATE)
        endif ()
    endif ()
endfunction ()

# Test targets for C++17 & C++20
set(TEST_SOURCES test.cpp)
set(CXX_STANDARDS 17 20 23)
foreach (STD IN LISTS CXX_STANDARDS)
    # Derive a unique target name
    set(TGT forkunion_test_cpp${STD})

    # Create the executable
    add_executable(${TGT} ${TEST_SOURCES})
    set_target_properties(
        ${TGT}
        PROPERTIES CXX_STANDARD ${STD}
                   CXX_STANDARD_REQUIRED ON
                   CXX_EXTENSIONS OFF
                   RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )

    # register it as a CTest test
    add_test(NAME ${TGT} COMMAND ${TGT})
    set_target_properties_for_forkunion_script(${TGT})

    # test.cpp uses no GNU extensions, so -pedantic-errors rejects a C++20 language feature that slips into a
    # lower standard - the conformance check, on the comprehensive source, not a separate lean TU. Only the
    # C++ tests: the C tests use nested functions and blocks, and the benchmarks pull Taskflow.
    target_compile_options(${TGT} PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-pedantic-errors>)

    # No `-latomic` here: `forkunion_header` carries it, probed, wherever the target needs it.
endforeach ()

enable_language(C)

# Builds one `test.c` C-API test target, sharing the C11 / output-dir / libatomic wiring. Callers add the one flag that
# distinguishes their variant (GNU nested functions, Clang blocks).
function (add_forkunion_c_test target extensions)
    add_executable(${target} test.c)
    set_target_properties(
        ${target}
        PROPERTIES C_STANDARD 11
                   C_STANDARD_REQUIRED ON
                   C_EXTENSIONS ${extensions}
                   RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    target_link_libraries(${target} PRIVATE forkunion_static)
    add_test(NAME ${target} COMMAND ${target})
endfunction ()

if (CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    message(STATUS "Skipping C11 C API tests on MSVC due to missing <stdatomic.h> support")
else ()
    add_forkunion_c_test(forkunion_test_c11 OFF)
endif ()

# GCC nested functions extension test
if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    add_forkunion_c_test(forkunion_test_c_gcc_nested ON)
endif ()

# Clang blocks extension test
if (CMAKE_C_COMPILER_ID STREQUAL "Clang" OR CMAKE_C_COMPILER_ID STREQUAL "AppleClang")
    add_forkunion_c_test(forkunion_test_c_clang_blocks ON)
    target_compile_options(forkunion_test_c_clang_blocks PRIVATE -fblocks)
    # Apple's `libSystem` ships the Blocks runtime; other UNIX needs it linked explicitly. The bare name resolves like
    # any library, rather than passing an opaque `-l` flag CMake cannot reason about.
    if (UNIX AND NOT APPLE)
        target_link_libraries(forkunion_test_c_clang_blocks PRIVATE BlocksRuntime)
    endif ()
endif ()

# Taskflow: a third comparison backend for the C++ benchmarks, fetched into the build tree so no submodule is vendored.
# `FIND_PACKAGE_ARGS` lets a system install satisfy configure without a clone.
find_package(Threads REQUIRED)
include(FetchContent)
set(TF_BUILD_TESTS
    OFF
    CACHE BOOL "" FORCE
)
set(TF_BUILD_EXAMPLES
    OFF
    CACHE BOOL "" FORCE
)
set(TF_BUILD_PROFILER
    OFF
    CACHE BOOL "" FORCE
)
# Taskflow is header-only, so configuring it buys nothing but its `install()` rules - which are not ours to run: they
# shipped 71 of its headers and a `taskflow.pc` into ForkUnion's own prefix, so a package built from a top-level
# checkout carried Taskflow's headers under ForkUnion's name. `EXCLUDE_FROM_ALL` does not help - it excludes targets
# from `ALL`, not `install()`.
#
# So: find a system copy first, and otherwise populate the source without ever configuring it. The sentinel
# `SOURCE_SUBDIR` names a directory holding no `CMakeLists.txt`, which is what makes `MakeAvailable` skip the
# `add_subdirectory` - and it replaces the deprecated single-argument `FetchContent_Populate`. Either way the benchmarks
# below just link `Taskflow::Taskflow`.
find_package(Taskflow QUIET)
if (NOT TARGET Taskflow::Taskflow)
    FetchContent_Declare(
        taskflow
        GIT_REPOSITORY https://github.com/taskflow/taskflow.git
        GIT_TAG v4.1.0
        SOURCE_SUBDIR forkunion-populate-only
    )
    FetchContent_MakeAvailable(taskflow)
    add_library(Taskflow::Taskflow INTERFACE IMPORTED)
    target_include_directories(Taskflow::Taskflow SYSTEM INTERFACE ${taskflow_SOURCE_DIR})
endif ()

# Both demo benchmarks share identical wiring - C++ for clear state ownership, but linking the pre-compiled C ABI so the
# runtime busy-wait waiter resolves at runtime - the whole point, since AArch64 `WFET` is unreachable through the
# compile-time C++ `preferred_yield_t`.
function (add_forkunion_benchmark target source)
    add_executable(${target} ${source})
    # C++20 for the benchmark harness (the library itself stays C++17); Taskflow v4 requires C++20 concepts.
    set_target_properties(
        ${target}
        PROPERTIES CXX_STANDARD 20
                   CXX_STANDARD_REQUIRED ON
                   CXX_EXTENSIONS OFF
                   RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    target_link_libraries(${target} PRIVATE forkunion_static)

    if (CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(
            ${target} PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3 -march=native -mtune=native>
        )
    endif ()

    # Strict IEEE math, deliberately: the N-body kernel hand-reassociates its reduction into the same eight lanes as the
    # Rust and Zig kernels, so no `-ffast-math` is wanted - and Clang's default `-ffp-contract=on` would fuse
    # multiply-adds Rust and Zig keep separate, diverging the bits.
    target_compile_options(${target} PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-ffp-contract=off>)

    # OpenMP: native on GCC, Homebrew `libomp` on Apple/Clang.
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(STATUS "Enabling OpenMP for ${target} on GCC")
        target_compile_options(${target} PRIVATE -fopenmp)
        target_link_options(${target} PRIVATE -fopenmp)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        find_program(BREW_EXECUTABLE brew)
        if (BREW_EXECUTABLE)
            execute_process(
                COMMAND ${BREW_EXECUTABLE} --prefix libomp
                OUTPUT_VARIABLE LIBOMP_PREFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
            )
            if (LIBOMP_PREFIX AND EXISTS "${LIBOMP_PREFIX}/include/omp.h")
                message(STATUS "Enabling OpenMP for ${target} on Clang with Homebrew libomp")
                target_include_directories(${target} PRIVATE ${LIBOMP_PREFIX}/include)
                target_link_directories(${target} PRIVATE ${LIBOMP_PREFIX}/lib)
                target_compile_options(${target} PRIVATE -Xclang -fopenmp)
                target_link_libraries(${target} PRIVATE omp)
            else ()
                message(STATUS "Homebrew libomp not found - OpenMP disabled for ${target}")
            endif ()
        else ()
            message(STATUS "Homebrew not found - OpenMP disabled for ${target}")
        endif ()
    endif ()

    # Taskflow: header-only, so linking the interface target only adds the include path and a threads dependency.
    target_link_libraries(${target} PRIVATE Taskflow::Taskflow Threads::Threads)
endfunction ()

add_forkunion_benchmark(forkunion_nbody nbody.cpp)
# Connected components: exact integer work; the CSR is replicated per memory domain.
add_forkunion_benchmark(forkunion_propagation propagation.cpp)
