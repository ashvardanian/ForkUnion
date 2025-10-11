# Shared logic for setting target properties for tests and examples
function (set_target_properties_for_fork_union_script target_name)
    target_link_libraries(${target_name} PRIVATE fork_union)

    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # This warning annoys us by reminding, that before GCC 4.6 the ABI for passing objects aligned to ≥32-bytes was
        # different... but that GCC version was released in 2014, over a decade ago.
        target_compile_options(${target_name} PRIVATE -Wno-psabi)
    endif ()

    # In release mode, enable optimizations
    if (CMAKE_BUILD_TYPE STREQUAL "Release")
        message(STATUS "Enabling optimizations for ${target_name}")
        target_compile_options(
            ${target_name}
            PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3 -march=native -mtune=native>
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
    set(TGT fork_union_test_cpp${STD})

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
    set_target_properties_for_fork_union_script(${TGT})

    # Link against `libatomic` for Linux toolchains that might need it
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        if (UNIX AND NOT APPLE)
            target_link_libraries(${TGT} PRIVATE -latomic)
        endif ()
    endif ()

endforeach ()

enable_language(C)

# Standard C11 test
if (CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    message(STATUS "Skipping C11 C API tests on MSVC due to missing <stdatomic.h> support")
else ()
    add_executable(fork_union_test_c11 test.c)
    set_target_properties(
        fork_union_test_c11
        PROPERTIES C_STANDARD 11
                   C_STANDARD_REQUIRED ON
                   C_EXTENSIONS OFF
                   RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    target_link_libraries(fork_union_test_c11 PRIVATE fork_union_static)
    add_test(NAME fork_union_test_c11 COMMAND fork_union_test_c11)

    if (CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID STREQUAL "Clang")
        if (UNIX AND NOT APPLE)
            target_link_libraries(fork_union_test_c11 PRIVATE -latomic)
        endif ()
    endif ()
endif ()

# GCC nested functions extension test
if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    add_executable(fork_union_test_c_gcc_nested test.c)
    set_target_properties(
        fork_union_test_c_gcc_nested
        PROPERTIES C_STANDARD 11
                   C_STANDARD_REQUIRED ON
                   C_EXTENSIONS ON
                   RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    target_link_libraries(fork_union_test_c_gcc_nested PRIVATE fork_union_static)
    add_test(NAME fork_union_test_c_gcc_nested COMMAND fork_union_test_c_gcc_nested)

    if (UNIX AND NOT APPLE)
        target_link_libraries(fork_union_test_c_gcc_nested PRIVATE -latomic)
    endif ()
endif ()

# Clang blocks extension test
if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
    add_executable(fork_union_test_c_clang_blocks test.c)
    set_target_properties(
        fork_union_test_c_clang_blocks
        PROPERTIES C_STANDARD 11
                   C_STANDARD_REQUIRED ON
                   C_EXTENSIONS ON
                   RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    target_compile_options(fork_union_test_c_clang_blocks PRIVATE -fblocks)
    target_link_libraries(fork_union_test_c_clang_blocks PRIVATE fork_union_static)
    add_test(NAME fork_union_test_c_clang_blocks COMMAND fork_union_test_c_clang_blocks)

    if (UNIX AND NOT APPLE)
        target_link_libraries(fork_union_test_c_clang_blocks PRIVATE -latomic -lBlocksRuntime)
    else ()
        target_link_libraries(fork_union_test_c_clang_blocks PRIVATE -lBlocksRuntime)
    endif ()
endif ()

# Include the N-body benchmark
add_executable(fork_union_nbody nbody.cpp)
target_link_libraries(fork_union_nbody PRIVATE fork_union)
set_target_properties(
    fork_union_nbody
    PROPERTIES CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF
)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "Enabling OpenMP for fork_union_nbody on GCC")
    target_compile_options(fork_union_nbody PRIVATE -fopenmp -flto -ffast-math)
    target_link_options(fork_union_nbody PRIVATE -fopenmp)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    find_program(BREW_EXECUTABLE brew)
    if (BREW_EXECUTABLE)
        execute_process(
            COMMAND ${BREW_EXECUTABLE} --prefix libomp
            OUTPUT_VARIABLE LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if (LIBOMP_PREFIX AND EXISTS "${LIBOMP_PREFIX}/include/omp.h")
            message(STATUS "Enabling OpenMP for fork_union_nbody on Clang with Homebrew libomp")
            target_include_directories(fork_union_nbody PRIVATE ${LIBOMP_PREFIX}/include)
            target_link_directories(fork_union_nbody PRIVATE ${LIBOMP_PREFIX}/lib)
            target_compile_options(fork_union_nbody PRIVATE -Xclang -fopenmp)
            target_link_libraries(fork_union_nbody PRIVATE omp)
        else ()
            message(STATUS "Homebrew libomp not found - OpenMP disabled for fork_union_nbody")
        endif ()
    else ()
        message(STATUS "Homebrew not found - OpenMP disabled for fork_union_nbody")
    endif ()
endif ()
set_target_properties_for_fork_union_script(fork_union_nbody)
