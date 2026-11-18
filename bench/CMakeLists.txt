# Taskflow: a third comparison backend for the C++ benchmarks, fetched into the build tree so no submodule is vendored.
# `FIND_PACKAGE_ARGS` lets a system install satisfy configure without a clone.
find_package(Threads REQUIRED)
include(FetchContent)
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

# OpenMP: native on GCC, Homebrew `libomp` on Apple/Clang host builds, looked up once for every benchmark.
# Cross builds skip Homebrew, whose `libomp` only links for the host, as iOS builds showed.
set(forkunion_libomp_prefix_)
if (CMAKE_CXX_COMPILER_ID MATCHES "^(AppleClang|Clang)$" AND NOT CMAKE_CROSSCOMPILING)
    find_program(FORKUNION_BREW_EXECUTABLE brew)
    if (FORKUNION_BREW_EXECUTABLE)
        execute_process(
            COMMAND ${FORKUNION_BREW_EXECUTABLE} --prefix libomp
            OUTPUT_VARIABLE forkunion_brew_libomp_
            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
        )
        if (forkunion_brew_libomp_ AND EXISTS "${forkunion_brew_libomp_}/include/omp.h")
            set(forkunion_libomp_prefix_ ${forkunion_brew_libomp_})
        endif ()
    endif ()
endif ()

# Both demo benchmarks share identical wiring and pick their waiter at compile time through `preferred_yield_t`.
function (forkunion_add_benchmark target_ source_)
    add_executable(${target_} ${source_})
    # C++20 for the benchmark harness, as for the library; Taskflow v4 requires C++20 concepts.
    set_target_properties(
        ${target_}
        PROPERTIES CXX_STANDARD 20
                   CXX_STANDARD_REQUIRED ON
                   CXX_EXTENSIONS OFF
    )
    target_link_libraries(${target_} PRIVATE forkunion_static forkunion_warnings)

    # `harness.hpp` reads the environment through `std::getenv`, which MSVC flags in favour of `_dupenv_s`.
    target_compile_definitions(${target_} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>)

    # Strict IEEE math, deliberately: the N-body kernel hand-reassociates its reduction into the same eight lanes as the
    # Rust and Zig kernels, so no `-ffast-math` is wanted - and Clang's default `-ffp-contract=on` would fuse
    # multiply-adds Rust and Zig keep separate, diverging the bits.
    target_compile_options(${target_} PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-ffp-contract=off>)

    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(STATUS "Enabling OpenMP for ${target_} on GCC")
        target_compile_options(${target_} PRIVATE -fopenmp)
        target_link_options(${target_} PRIVATE -fopenmp)
    elseif (forkunion_libomp_prefix_)
        message(STATUS "Enabling OpenMP for ${target_} on Clang with Homebrew libomp")
        target_include_directories(${target_} PRIVATE ${forkunion_libomp_prefix_}/include)
        target_link_directories(${target_} PRIVATE ${forkunion_libomp_prefix_}/lib)
        target_compile_options(${target_} PRIVATE -Xclang -fopenmp)
        target_link_libraries(${target_} PRIVATE omp)
    else ()
        message(STATUS "OpenMP disabled for ${target_}: neither GCC nor Clang with Homebrew libomp")
    endif ()

    # Taskflow: header-only, so linking the interface target only adds the include path and a threads dependency.
    target_link_libraries(${target_} PRIVATE Taskflow::Taskflow Threads::Threads)
endfunction ()

forkunion_add_benchmark(forkunion_nbody nbody.cpp)
# Connected components: exact integer work; the CSR is replicated per memory domain.
forkunion_add_benchmark(forkunion_propagation propagation.cpp)
