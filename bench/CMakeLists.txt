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
    # C++20 for the benchmark harness, as for the library; Taskflow v4 requires C++20 concepts.
    set_target_properties(
        ${target}
        PROPERTIES CXX_STANDARD 20
                   CXX_STANDARD_REQUIRED ON
                   CXX_EXTENSIONS OFF
                   RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    target_link_libraries(${target} PRIVATE forkunion_static)

    # `harness.hpp` reads the environment through `std::getenv`, which MSVC flags in favour of `_dupenv_s`.
    target_compile_definitions(${target} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>)

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
