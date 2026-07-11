cmake_minimum_required(VERSION 3.14)

# Set modern CMake policies for better compatibility
if (POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW) # option() honors normal variables
endif ()
if (POLICY CMP0079)
    cmake_policy(SET CMP0079 NEW) # target_link_libraries() allows use with targets in other directories
endif ()
if (POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW) # MSVC runtime library flags are selected by an abstraction
endif ()

# Enable the in-repo Git hooks. Git will not auto-run hooks shipped inside a clone (that would let any cloned repo
# execute code), so point `core.hooksPath` at `.githooks` here - configuring the build arms the `pre-commit` checks.
# Skipped when there is no `.git` (tarball / export builds).
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND git config core.hooksPath .githooks
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE _fu_hooks_status
        OUTPUT_QUIET ERROR_QUIET
    )
    if (_fu_hooks_status EQUAL 0)
        message(STATUS "forkunion: Git hooks enabled (core.hooksPath -> .githooks)")
    endif ()
endif ()

project(
    forkunion
    VERSION 2.3.1
    DESCRIPTION "Low-latency OpenMP-style NUMA-aware cross-platform fine-grained parallelism library"
    LANGUAGES CXX
)

# Control strict interface warnings for consumers
option(FORKUNION_STRICT "Propagate strict warnings (including -Werror) to consumers" OFF)
option(FORKUNION_PORTABLE "Force every optional capability off, leaving the STL pool" OFF)

# Each kernel facility the library can use is a tri-state: AUTO, ON, or OFF.
#
# AUTO is the default and defines *nothing*. The header derives it, from the platform and from whether <numa.h> is there
# to include. That keeps the derivation rules in exactly one place, instead of spread across CMakeLists.txt, build.rs,
# and build.zig, where they would drift.
#
# ON and OFF pass `-DFU_WITH_<CAP>=1` or `=0`, overriding the header. An impossible ON - pinning on Apple, say - is
# rejected by an `#error` in `types.hpp`, not by a check duplicated here. The four coarse facilities a build
# realistically toggles, each a tri-state cache entry. The finer ones - cache sharing, HMAT metrics, thread QoS,
# scheduling class - are AUTO-only: the header derives them, and only `FORKUNION_PORTABLE` forces them off. Rust and Zig
# expose this same set.
set(FORKUNION_WITH_TOPOLOGY
    "AUTO"
    CACHE STRING "Enumerate cores and domains: AUTO, ON, or OFF"
)
set(FORKUNION_WITH_PLACE_MEMORY_ON_DOMAIN
    "AUTO"
    CACHE STRING "Place pages on a memory domain: AUTO, ON, or OFF"
)
set(FORKUNION_WITH_PLACE_HUGE_PAGES_ON_DOMAIN
    "AUTO"
    CACHE STRING "Request larger pages: AUTO, ON, or OFF"
)
set(FORKUNION_WITH_PLACE_THREADS_BY_AFFINITY
    "AUTO"
    CACHE STRING "Bind threads to cores: AUTO, ON, or OFF"
)
set_property(CACHE FORKUNION_WITH_TOPOLOGY PROPERTY STRINGS "AUTO;ON;OFF")
set_property(CACHE FORKUNION_WITH_PLACE_MEMORY_ON_DOMAIN PROPERTY STRINGS "AUTO;ON;OFF")
set_property(CACHE FORKUNION_WITH_PLACE_HUGE_PAGES_ON_DOMAIN PROPERTY STRINGS "AUTO;ON;OFF")
set_property(CACHE FORKUNION_WITH_PLACE_THREADS_BY_AFFINITY PROPERTY STRINGS "AUTO;ON;OFF")

# Every optional capability macro, coarse and fine - the set `FORKUNION_PORTABLE` zeroes.
set(FORKUNION_OPTIONAL_MACROS
    FU_WITH_TOPOLOGY FU_WITH_PLACE_THREADS_BY_AFFINITY FU_WITH_PLACE_THREADS_BY_CORE_CLASS
    FU_WITH_RESCHEDULE_THREADS_BY_CLASS FU_WITH_PLACE_MEMORY_ON_DOMAIN FU_WITH_PLACE_HUGE_PAGES_ON_DOMAIN
)

# Enforce C++17 as the minimum standard for the project.
set(PROJECT_CXX_STANDARD 17)
set(PROJECT_CXX_EXTENSIONS OFF)
set(PROJECT_CXX_STANDARD_REQUIRED ON)

# Header-only interface library
add_library(forkunion INTERFACE)
target_include_directories(
    forkunion INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
)

# Set C++17 requirement and features properly for library consumers
target_compile_features(forkunion INTERFACE cxx_std_17)
set_target_properties(
    forkunion
    PROPERTIES CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF
)

# Strictest possible compilation flags with fatal warnings
if (FORKUNION_STRICT)
    target_compile_options(
        forkunion
        INTERFACE # GCC/Clang: Maximum warnings + treat warnings as errors + security hardening
                  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall
                  -Wextra
                  -Wpedantic
                  -Werror
                  -Wconversion
                  -Wcast-qual
                  -Wcast-align
                  -Wunused
                  -Wno-unused-parameter
                  -Wno-unknown-pragmas
                  -Wno-sign-conversion
                  -Wno-unused-function
                  -Wshadow
                  -Wnon-virtual-dtor
                  -Wold-style-cast
                  -Woverloaded-virtual
                  -Wsign-promo
                  -Wno-null-dereference
                  -Wdouble-promotion>
                  $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<CONFIG:Release,RelWithDebInfo>>:-Wno-unused-variable>
                  # Additional GCC-specific warnings
                  $<$<CXX_COMPILER_ID:GNU>:-Wmisleading-indentation
                  -Wduplicated-cond
                  -Wduplicated-branches
                  -Wlogical-op
                  -Wno-null-dereference
                  -Wuseless-cast>
                  # Additional Clang-specific warnings
                  $<$<CXX_COMPILER_ID:Clang>:-Wmost
                  -Wthread-safety
                  -Wthread-safety-negative>
                  # MSVC: Maximum warnings + treat warnings as errors
                  $<$<CXX_COMPILER_ID:MSVC>:/W4
                  /WX
                  /permissive-
                  /w14242
                  /w14254
                  /w14263
                  /w14265
                  /w14287
                  /we4289
                  /w14296
                  /w14311
                  /w14545
                  /w14546
                  /w14547
                  /w14549
                  /w14555
                  /w14619
                  /w14640
                  /w14826
                  /w14905
                  /w14906
                  /w14928>
                  $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release,RelWithDebInfo>>:/wd4101>
    )
endif ()

# Pre-compiled libraries built from `c/forkunion.cpp`
add_library(forkunion_dynamic SHARED c/forkunion.cpp)
add_library(forkunion_static STATIC c/forkunion.cpp)

# Prefer C++20 for library builds
set_target_properties(
    forkunion_dynamic forkunion_static
    PROPERTIES CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF
)

# Re-use the public interface of the header-only target
target_link_libraries(forkunion_dynamic PUBLIC forkunion)
target_link_libraries(forkunion_static PUBLIC forkunion)

# Probe compiler support for -fcf-protection=full (depends on compiler+arch)
include(CheckCXXCompilerFlag)
set(HAS_CFP FALSE)
check_cxx_compiler_flag("-fcf-protection=full" HAS_CFP)

# Security hardening flags for compiled libraries
target_compile_options(
    forkunion_dynamic
    PRIVATE # Stack protection and buffer overflow detection
            $<$<CXX_COMPILER_ID:GNU,Clang>:-fstack-protector-strong
            -D_FORTIFY_SOURCE=2>
            # Control flow integrity and stack clash protection
            $<$<AND:$<CXX_COMPILER_ID:GNU>,$<BOOL:${HAS_CFP}>>:-fcf-protection=full>
            $<$<CXX_COMPILER_ID:GNU>:-fstack-clash-protection>
            $<$<AND:$<CXX_COMPILER_ID:Clang>,$<BOOL:${HAS_CFP}>>:-fcf-protection=full>
            # MSVC security features
            $<$<CXX_COMPILER_ID:MSVC>:/GS
            /guard:cf>
)
target_compile_options(
    forkunion_static
    PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-fstack-protector-strong
            -D_FORTIFY_SOURCE=2>
            $<$<AND:$<CXX_COMPILER_ID:GNU>,$<BOOL:${HAS_CFP}>>:-fcf-protection=full>
            $<$<CXX_COMPILER_ID:GNU>:-fstack-clash-protection>
            $<$<AND:$<CXX_COMPILER_ID:Clang>,$<BOOL:${HAS_CFP}>>:-fcf-protection=full>
            $<$<CXX_COMPILER_ID:MSVC>:/GS
            /guard:cf>
)

# Hardened linking flags
target_link_options(
    forkunion_dynamic
    PRIVATE
    # Enable RELRO, stack canaries, and NX bit
    $<$<PLATFORM_ID:Linux>:-Wl,-z,relro,-z,now,-z,noexecstack>
    # MSVC hardened linking
    $<$<CXX_COMPILER_ID:MSVC>:/DYNAMICBASE
    /NXCOMPAT
    /guard:cf>
)
target_link_options(
    forkunion_static PRIVATE $<$<PLATFORM_ID:Linux>:-Wl,-z,relro,-z,now,-z,noexecstack>
    $<$<CXX_COMPILER_ID:MSVC>:/guard:cf>
)

# Set the output directory for all executables and libraries - on Windows requires more boilerplate:
# https://stackoverflow.com/a/25328001
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR})

# Also set library output directories to avoid nested config directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR})

# Static analysis tools
find_program(CPPCHECK_EXECUTABLE cppcheck)
if (CPPCHECK_EXECUTABLE)
    add_custom_target(
        cppcheck
        COMMAND
            ${CPPCHECK_EXECUTABLE} --enable=all --std=c++17 --verbose --quiet --error-exitcode=1
            --suppress=missingIncludeSystem --suppress=unusedFunction --suppress=unmatchedSuppression
            --suppress=ConfigurationNotChecked --suppress=knownConditionTrueFalse --suppress=shadowFunction
            --suppress=shadowVariable --suppress=useStlAlgorithm --suppress=noExplicitConstructor
            -I${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/c
        COMMENT "Running cppcheck static analysis"
    )
endif ()

find_program(CLANG_TIDY_EXECUTABLE clang-tidy)
if (CLANG_TIDY_EXECUTABLE)
    add_custom_target(
        clang-tidy
        COMMAND
            ${CLANG_TIDY_EXECUTABLE} --config-file=${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy --quiet
            ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/c/*.cpp --
            -I${CMAKE_CURRENT_SOURCE_DIR}/include -std=c++17
        COMMENT "Running clang-tidy static analysis"
    )
endif ()

# Tests & benchmarking scripts
include(CTest)

option(FORKUNION_BUILD_TESTS "Build forkunion tests" ON)

if (BUILD_TESTING AND FORKUNION_BUILD_TESTS)
    enable_testing()
    add_subdirectory(scripts)
endif ()

# Install rules
include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
    TARGETS forkunion
    EXPORT forkunionTargets
    INCLUDES
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(
    TARGETS forkunion_dynamic forkunion_static
    EXPORT forkunionTargets # same export set
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # .so / .dylib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} # .a
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # .dll on Windows
)

# Export config files for find_package
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/forkunionConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    cmake/forkunionConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/forkunionConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/forkunion
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/forkunionConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/forkunionConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/forkunion
)

install(
    EXPORT forkunionTargets
    NAMESPACE forkunion::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/forkunion
)

# Turn the tri-states into compile definitions, and link whatever they imply.
#
# Note the asymmetry: we pass definitions only for the capabilities a caller pinned, but we link libnuma whenever it is
# present. Over-linking a library the code never calls costs a DT_NEEDED entry that `--as-needed` drops; under-linking
# one the header decided to `#include` costs a link error the caller cannot diagnose.
find_package(Threads)
if (Threads_FOUND)
    target_link_libraries(forkunion INTERFACE Threads::Threads)
endif ()

set(NUMA_FOUND FALSE)
if (UNIX AND NOT APPLE)
    find_library(NUMA_LIBRARY NAMES numa)
    find_path(NUMA_INCLUDE_DIR NAMES numa.h)
    if (NUMA_LIBRARY AND NUMA_INCLUDE_DIR)
        set(NUMA_FOUND TRUE)
        target_link_libraries(forkunion INTERFACE ${NUMA_LIBRARY})
        message(STATUS "libNUMA found: ${NUMA_LIBRARY}")
    else ()
        message(STATUS "libNUMA not found - the header will lower the capabilities that need it")
    endif ()
endif ()

# Translate one tri-state to its macro. AUTO says nothing and lets the header decide; ON and OFF force the macro to 1 or
# 0. Knows nothing of the environment - the libnuma check is its own concern, below.
function (forkunion_apply mode macro)
    if (mode STREQUAL "ON")
        target_compile_definitions(forkunion INTERFACE ${macro}=1)
    elseif (mode STREQUAL "OFF")
        target_compile_definitions(forkunion INTERFACE ${macro}=0)
    elseif (NOT mode STREQUAL "AUTO")
        message(FATAL_ERROR "${macro} tri-state must be AUTO, ON, or OFF - got '${mode}'")
    endif ()
endfunction ()

# `FORKUNION_PORTABLE` is the umbrella off-switch: it zeroes every optional capability and so cannot coexist with a
# forced-on one. Prerequisites between capabilities (huge pages need NUMA, which needs topology) are the header's
# business, enforced by its `#error`s - never re-encoded here.
if (FORKUNION_PORTABLE)
    if (FORKUNION_WITH_TOPOLOGY STREQUAL "ON"
        OR FORKUNION_WITH_PLACE_MEMORY_ON_DOMAIN STREQUAL "ON"
        OR FORKUNION_WITH_PLACE_HUGE_PAGES_ON_DOMAIN STREQUAL "ON"
        OR FORKUNION_WITH_PLACE_THREADS_BY_AFFINITY STREQUAL "ON"
    )
        message(FATAL_ERROR "FORKUNION_PORTABLE is ON, but a FORKUNION_WITH_* capability is forced ON")
    endif ()
    foreach (macro IN LISTS FORKUNION_OPTIONAL_MACROS)
        target_compile_definitions(forkunion INTERFACE ${macro}=0)
    endforeach ()
else ()
    # Forcing a libnuma-backed capability on without libnuma fails here, with a sentence, rather than at link time with
    # a list of missing symbols. AUTO is fine - the header derives it off when libnuma is absent.
    if (UNIX
        AND NOT APPLE
        AND NOT NUMA_FOUND
        AND (FORKUNION_WITH_TOPOLOGY STREQUAL "ON"
             OR FORKUNION_WITH_PLACE_MEMORY_ON_DOMAIN STREQUAL "ON"
             OR FORKUNION_WITH_PLACE_HUGE_PAGES_ON_DOMAIN STREQUAL "ON")
    )
        message(FATAL_ERROR "topology/numa-memory/huge-pages forced ON, but libnuma was not found")
    endif ()

    forkunion_apply("${FORKUNION_WITH_TOPOLOGY}" FU_WITH_TOPOLOGY)
    forkunion_apply("${FORKUNION_WITH_PLACE_MEMORY_ON_DOMAIN}" FU_WITH_PLACE_MEMORY_ON_DOMAIN)
    forkunion_apply("${FORKUNION_WITH_PLACE_HUGE_PAGES_ON_DOMAIN}" FU_WITH_PLACE_HUGE_PAGES_ON_DOMAIN)
    forkunion_apply("${FORKUNION_WITH_PLACE_THREADS_BY_AFFINITY}" FU_WITH_PLACE_THREADS_BY_AFFINITY)
endif ()
