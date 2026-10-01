# Shared logic for setting target properties for tests
function (set_target_properties_for_forkunion_test target_name)
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

# Test targets for C++20 & C++23
set(TEST_SOURCES main.cpp)
set(CXX_STANDARDS 20 23)
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
    set_target_properties_for_forkunion_test(${TGT})

    # `check_atomic_ref` admits each reference through `runtime_capabilities()`, so the tests take every buildable rung.
    target_compile_definitions(${TGT} PRIVATE ${fu_compile_definitions_})

    # main.cpp uses no GNU extensions, so -pedantic-errors rejects a C++23 language feature that slips into the C++20
    # build - the conformance check, on the comprehensive source, not a separate lean TU. Only the C++ tests: the C
    # tests use nested functions and blocks, and the benchmarks pull Taskflow.
    target_compile_options(${TGT} PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-pedantic-errors>)

    # Correctness only where the option asks - the cross jobs, whose emulator cannot run the throughput suite in time.
    # `main.c` carries no stress cases, so only the C++ targets take the define.
    if (FORKUNION_TEST_SKIP_STRESS)
        target_compile_definitions(${TGT} PRIVATE FORKUNION_TEST_SKIP_STRESS_=1)
    endif ()

    # No `-latomic` here: `forkunion_header` carries it, probed, wherever the target needs it.
endforeach ()

enable_language(C)

# Builds one `main.c` C-API test target, sharing the C11 / output-dir / libatomic wiring. Callers add the one flag that
# distinguishes their variant (GNU nested functions, Clang blocks).
function (add_forkunion_c_test target extensions)
    add_executable(${target} main.c)
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
