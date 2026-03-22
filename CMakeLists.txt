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

project(
    fork_union
    VERSION 2.3.1
    DESCRIPTION "Low-latency OpenMP-style NUMA-aware cross-platform fine-grained parallelism library"
    LANGUAGES CXX
)

# Control strict interface warnings for consumers
option(FORK_UNION_STRICT "Propagate strict warnings (including -Werror) to consumers" OFF)

# Control NUMA enablement: AUTO, ON, OFF
set(FORK_UNION_ENABLE_NUMA
    "AUTO"
    CACHE STRING "NUMA support mode: AUTO, ON, or OFF"
)
set_property(CACHE FORK_UNION_ENABLE_NUMA PROPERTY STRINGS "AUTO;ON;OFF")

# Enforce C++17 as the minimum standard for the project.
set(PROJECT_CXX_STANDARD 17)
set(PROJECT_CXX_EXTENSIONS OFF)
set(PROJECT_CXX_STANDARD_REQUIRED ON)

# Header-only interface library
add_library(fork_union INTERFACE)
target_include_directories(
    fork_union INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
)

# Set C++17 requirement and features properly for library consumers
target_compile_features(fork_union INTERFACE cxx_std_17)
set_target_properties(
    fork_union
    PROPERTIES CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF
)

# Strictest possible compilation flags with fatal warnings
if (FORK_UNION_STRICT)
    target_compile_options(
        fork_union
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

# Pre-compiled libraries built from `c/lib.cpp`
add_library(fork_union_dynamic SHARED c/lib.cpp)
add_library(fork_union_static STATIC c/lib.cpp)

# Prefer C++20 for library builds
set_target_properties(
    fork_union_dynamic fork_union_static
    PROPERTIES CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF
)

# Re-use the public interface of the header-only target
target_link_libraries(fork_union_dynamic PUBLIC fork_union)
target_link_libraries(fork_union_static PUBLIC fork_union)

# Probe compiler support for -fcf-protection=full (depends on compiler+arch)
include(CheckCXXCompilerFlag)
set(HAS_CFP FALSE)
check_cxx_compiler_flag("-fcf-protection=full" HAS_CFP)

# Security hardening flags for compiled libraries
target_compile_options(
    fork_union_dynamic
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
    fork_union_static
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
    fork_union_dynamic
    PRIVATE
    # Enable RELRO, stack canaries, and NX bit
    $<$<PLATFORM_ID:Linux>:-Wl,-z,relro,-z,now,-z,noexecstack>
    # MSVC hardened linking
    $<$<CXX_COMPILER_ID:MSVC>:/DYNAMICBASE
    /NXCOMPAT
    /guard:cf>
)
target_link_options(
    fork_union_static PRIVATE $<$<PLATFORM_ID:Linux>:-Wl,-z,relro,-z,now,-z,noexecstack>
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
            -I${CMAKE_CURRENT_SOURCE_DIR}/include -DFU_ENABLE_NUMA=1 ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/c
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

option(FORK_UNION_BUILD_TESTS "Build fork_union tests" ON)

if (BUILD_TESTING AND FORK_UNION_BUILD_TESTS)
    enable_testing()
    add_subdirectory(scripts)
endif ()

# Install rules
include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
    TARGETS fork_union
    EXPORT fork_unionTargets
    INCLUDES
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(
    TARGETS fork_union_dynamic fork_union_static
    EXPORT fork_unionTargets # same export set
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # .so / .dylib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} # .a
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # .dll on Windows
)

# Export config files for find_package
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/fork_unionConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    cmake/fork_unionConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/fork_unionConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fork_union
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/fork_unionConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/fork_unionConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fork_union
)

install(
    EXPORT fork_unionTargets
    NAMESPACE fork_union::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fork_union
)

# NUMA selection logic controlled by FORK_UNION_ENABLE_NUMA
if (UNIX AND NOT APPLE) # Linux
    # Find POSIX threads library
    find_package(Threads)
    # Find libNUMA
    find_library(NUMA_LIBRARY NAMES numa)
    set(NUMA_FOUND FALSE)
    if (NUMA_LIBRARY)
        set(NUMA_FOUND TRUE)
        message(STATUS "libNUMA found: ${NUMA_LIBRARY}")
    else ()
        message(STATUS "libNUMA not found - NUMA-dependent features unavailable")
    endif ()

    if (FORK_UNION_ENABLE_NUMA STREQUAL "ON")
        if (Threads_FOUND AND NUMA_FOUND)
            target_link_libraries(fork_union INTERFACE ${NUMA_LIBRARY} Threads::Threads)
            target_compile_definitions(fork_union INTERFACE FU_ENABLE_NUMA=1)
        else ()
            message(FATAL_ERROR "FORK_UNION_ENABLE_NUMA=ON, but Threads or libnuma not found")
        endif ()
    elseif (FORK_UNION_ENABLE_NUMA STREQUAL "AUTO")
        if (Threads_FOUND AND NUMA_FOUND)
            target_link_libraries(fork_union INTERFACE ${NUMA_LIBRARY} Threads::Threads)
            target_compile_definitions(fork_union INTERFACE FU_ENABLE_NUMA=1)
        else ()
            target_compile_definitions(fork_union INTERFACE FU_ENABLE_NUMA=0)
        endif ()
    else () # OFF
        target_compile_definitions(fork_union INTERFACE FU_ENABLE_NUMA=0)
    endif ()
else () # Non-Linux platforms
    target_compile_definitions(fork_union INTERFACE FU_ENABLE_NUMA=0)
endif ()
