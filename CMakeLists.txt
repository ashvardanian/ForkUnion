cmake_minimum_required(VERSION 3.21)

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
    VERSION 3.0.3
    DESCRIPTION
        "Low-latency NUMA-aware fork-join thread-pool with zero allocations, syscalls, CAS, or false-sharing on the hot path"
    LANGUAGES CXX
)

# Control strict interface warnings for consumers
option(FORKUNION_STRICT "Propagate strict warnings (including -Werror) to consumers" OFF)
option(FORKUNION_PORTABLE "Force every optional capability off, leaving the STL pool" OFF)

# Each kernel facility the library can use is a tri-state: AUTO, ON, or OFF.
#
# AUTO is the default and defines *nothing*. The header derives it from the platform alone - no longer from whether
# <numa.h> happened to be installed here, which let a build host's packages decide a run host's capabilities. That keeps
# the derivation rules in exactly one place, instead of spread across CMakeLists.txt, build.rs, and build.zig, where
# they would drift.
#
# ON and OFF pass `-DFU_WITH_<CAP>=1` or `=0`, overriding the header. An impossible ON - pinning on Apple, say - is
# rejected by an `#error` in `types.hpp`, not by a check duplicated here. The four coarse facilities a build
# realistically toggles, each a tri-state cache entry. The finer ones - cache sharing, thread QoS, scheduling class -
# are AUTO-only: the header derives them, and only `FORKUNION_PORTABLE` forces them off. Rust and Zig expose this same
# set.
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

# Header-only interface library
add_library(forkunion_header INTERFACE)
target_include_directories(
    forkunion_header INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
)

# C++17 is the floor consumers must clear, not a ceiling: `cxx_std_17` raises a consumer below it up to 17 and leaves
# one already at 20/23 alone, so the C++20 fast paths - concepts, the `atomic_ref` monitor, `std::popcount` - stay
# available to whoever asks for them. No `CXX_STANDARD` property here - it governs only a target's own compilation, and
# an INTERFACE library has none, so it would be a no-op that reads like a cap. The pre-compiled libraries below pin
# themselves to 20.
target_compile_features(forkunion_header INTERFACE cxx_std_17)

# Strictest possible compilation flags with fatal warnings
if (FORKUNION_STRICT)
    target_compile_options(
        forkunion_header
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

# Pre-compiled libraries built from `c/forkunion.cpp` - the same translation unit twice, differing only in linkage.
add_library(forkunion_shared SHARED c/forkunion.cpp)
add_library(forkunion_static STATIC c/forkunion.cpp)

# What a consumer picks between, spelled as what they get:
#
# forkunion::header - the C++ core alone. Capabilities are baked in by the `FU_WITH_*` macros and `preferred_yield_t` at
# compile time, so the binary is only as capable as the machine that built it. forkunion::static - the C ABI,
# dispatching on `fu_runtime_capabilities()`, linked into the consumer's binary. forkunion::shared - the same C ABI, one
# copy per process rather than one per consumer.
#
# The `static`/`shared` split is linkage, not dispatch: both run the same runtime-dispatched TU above, and a process
# that links two static copies gets two pools and two topologies. `EXPORT_NAME` carries these spellings into the
# installed package, while the raw target names stay prefixed - they are global, and `static` alone would not be.
set_target_properties(forkunion_header PROPERTIES EXPORT_NAME header)
set_target_properties(forkunion_static PROPERTIES EXPORT_NAME static)
set_target_properties(forkunion_shared PROPERTIES EXPORT_NAME shared)

# Mirror the exported spellings in-tree, so `add_subdirectory` and `find_package` consumers write the same line - and so
# a typo fails to configure instead of quietly becoming a `-lforkunion` system lookup.
add_library(forkunion::header ALIAS forkunion_header)
add_library(forkunion::static ALIAS forkunion_static)
add_library(forkunion::shared ALIAS forkunion_shared)

# Prefer C++20 for library builds
set_target_properties(
    forkunion_shared forkunion_static
    PROPERTIES CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF
)

# `SOVERSION` carries the major alone - the only break the C ABI admits, matching `SameMajorVersion` below.
set_target_properties(forkunion_shared PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})

# Hide everything `forkunion.h` does not declare. On its own this exports zero symbols; the visibility pragma in
# `c/forkunion.cpp` is the other half. Static too - same translation unit, and a consumer may link it into their own
# shared object.
set_target_properties(
    forkunion_shared forkunion_static PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
)

# No `__declspec(dllexport)` in this source, so MSVC built a DLL with no import library and an unusable
# `forkunion::shared`. Let CMake write the module definition file instead.
set_target_properties(forkunion_shared PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

# Re-use the public interface of the header-only target
target_link_libraries(forkunion_shared PUBLIC forkunion_header)
target_link_libraries(forkunion_static PUBLIC forkunion_header)

# Probe compiler support for -fcf-protection=full (depends on compiler+arch)
include(CheckCXXCompilerFlag)
set(HAS_CFP FALSE)
check_cxx_compiler_flag("-fcf-protection=full" HAS_CFP)

# Security hardening flags for compiled libraries
target_compile_options(
    forkunion_shared
    PRIVATE # Stack protection and buffer overflow detection
            $<$<CXX_COMPILER_ID:GNU,Clang>:-fstack-protector-strong>
            # `_FORTIFY_SOURCE` needs `-O`; glibc `#warning`s otherwise, fatal under FORKUNION_STRICT. Allowlist the
            # optimized configs, since an empty `CMAKE_BUILD_TYPE` is not Debug yet emits no `-O`. The `-U` disarms
            # distributions that predefine it, whose redefinition would warn on the same path.
            $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<CONFIG:Release,RelWithDebInfo,MinSizeRel>>:-U_FORTIFY_SOURCE>
            $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<CONFIG:Release,RelWithDebInfo,MinSizeRel>>:-D_FORTIFY_SOURCE=2>
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
    PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-fstack-protector-strong>
            # Same reasoning as the shared target above.
            $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<CONFIG:Release,RelWithDebInfo,MinSizeRel>>:-U_FORTIFY_SOURCE>
            $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<CONFIG:Release,RelWithDebInfo,MinSizeRel>>:-D_FORTIFY_SOURCE=2>
            $<$<AND:$<CXX_COMPILER_ID:GNU>,$<BOOL:${HAS_CFP}>>:-fcf-protection=full>
            $<$<CXX_COMPILER_ID:GNU>:-fstack-clash-protection>
            $<$<AND:$<CXX_COMPILER_ID:Clang>,$<BOOL:${HAS_CFP}>>:-fcf-protection=full>
            $<$<CXX_COMPILER_ID:MSVC>:/GS
            /guard:cf>
)

# Hardened linking flags
target_link_options(
    forkunion_shared
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

# Default ON only when ForkUnion is the top-level project, so an `add_subdirectory` consumer does not inherit the
# `scripts/` targets - and does not have to remember to switch them off.
option(FORKUNION_BUILD_TESTS "Build forkunion tests" ${PROJECT_IS_TOP_LEVEL})
option(FORKUNION_TEST_SKIP_STRESS "Run only the tests' correctness suite, skipping the slow throughput/stress cases"
       OFF
)

if (BUILD_TESTING AND FORKUNION_BUILD_TESTS)
    enable_testing()
    add_subdirectory(scripts)
endif ()

# Install rules. Default ON only when ForkUnion is the top-level project: an `add_subdirectory` consumer installing
# itself would otherwise drop ForkUnion's headers, libraries, and CMake package into *its* prefix, colliding with a real
# `forkunion` package. That is the documented path - `FetchContent_MakeAvailable` adds the subdirectory plainly, without
# the `EXCLUDE_FROM_ALL` that would suppress these on newer CMake. A parent that does want ForkUnion packaged alongside
# it sets `FORKUNION_INSTALL` first; `CMP0077` above lets that normal variable win.
option(FORKUNION_INSTALL "Install ForkUnion's headers, libraries, and CMake package" ${PROJECT_IS_TOP_LEVEL})

if (FORKUNION_INSTALL)
    include(GNUInstallDirs)
    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    # No `INCLUDES DESTINATION` here: `target_include_directories` above already declares the same
    # `$<INSTALL_INTERFACE:include>`, and naming it twice exports the path twice.
    install(TARGETS forkunion_header EXPORT forkunionTargets)
    install(
        TARGETS forkunion_shared forkunion_static
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
        COMPATIBILITY SameMajorVersion
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
endif ()

# Turn the tri-states into compile definitions. Nothing left to link for them: the topology comes from sysfs and the
# placement calls `mbind` by syscall number, so the C library is the only dependency. `find_library(numa)` also baked
# this host's absolute `libnuma.so` path into the installed package, where another host's loader cannot follow it.
find_package(Threads)
if (Threads_FOUND)
    target_link_libraries(forkunion_header INTERFACE Threads::Threads)
endif ()

# Does `std::atomic` need libatomic on this target? x86-64 inlines the 64-bit operations; 32-bit Arm resolves them
# through the library, and the pools use `std::atomic<std::uint64_t>`. Probe the target rather than guess by platform -
# LLVM's `CheckAtomic` pattern - and carry the answer on the header, so every consumer inherits it: the tests, the
# compiled libraries, and a downstream `find_package` alike, none restating `-latomic`. The bare name `atomic` becomes
# `-latomic` at link and stays relocatable, unlike the absolute path `find_library` would bake into the package.
include(CheckCXXSourceCompiles)
check_cxx_source_compiles(
    "#include <atomic>
     #include <cstdint>
     // Mirror the pools' atomic footprint so the probe demands libatomic on exactly the targets whose
     // build emits a libcall: i686 for the 64-bit read-modify-write, RISC-V for the sub-word one it has
     // no native instruction for, and clang-with-libstdc++ for the `is_lock_free` it does not fold. A
     // load-only probe inlines on every one of them, under-reports, and the link fails downstream.
     struct probe {
         std::atomic<std::uint64_t> wide;
         std::atomic<std::uint16_t> half;
         std::atomic<std::uint8_t> byte;
         bool lock_free() const noexcept { return wide.is_lock_free() && half.is_lock_free() && byte.is_lock_free(); }
     };
     probe value;
     int main() {
         value.wide.fetch_add(1);
         value.half.fetch_add(1);
         value.byte.fetch_add(1);
         return static_cast<int>(value.wide.load()) + value.lock_free();
     }"
    FORKUNION_ATOMICS_ARE_BUILTIN
)
if (NOT FORKUNION_ATOMICS_ARE_BUILTIN)
    target_link_libraries(forkunion_header INTERFACE atomic)
endif ()

# A `.pc` for the consumers CMake never reaches. Must follow the two probes above: `configure_file` substitutes now,
# unlike `install(EXPORT)`, which writes at generate time and so tolerates a link interface declared later. Spelling the
# prefix against `${pcfiledir}` keeps it correct after `cmake --install --prefix` moves the tree. Under the same option
# as the rules above, which is also what brought `GNUInstallDirs` in - without it these paths resolve empty.
if (FORKUNION_INSTALL)
    file(RELATIVE_PATH FORKUNION_PC_PREFIX "${CMAKE_INSTALL_FULL_LIBDIR}/pkgconfig" "${CMAKE_INSTALL_PREFIX}")
    string(REGEX REPLACE "/+$" "" FORKUNION_PC_PREFIX "${FORKUNION_PC_PREFIX}")
    set(FORKUNION_PC_PREFIX "\${pcfiledir}/${FORKUNION_PC_PREFIX}")

    # `GNUInstallDirs` returns a relative path unless a distro forced an absolute one; only the former may follow a
    # prefix.
    if (IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
        set(FORKUNION_PC_LIBDIR "${CMAKE_INSTALL_LIBDIR}")
    else ()
        set(FORKUNION_PC_LIBDIR "\${exec_prefix}/${CMAKE_INSTALL_LIBDIR}")
    endif ()
    if (IS_ABSOLUTE "${CMAKE_INSTALL_INCLUDEDIR}")
        set(FORKUNION_PC_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}")
    else ()
        set(FORKUNION_PC_INCLUDEDIR "\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
    endif ()

    # A C caller linking the archive has to name the C++ runtime itself, so read it off the compiler rather than the
    # platform. These go in `Libs.private`, emitted only for `--static`; `Requires.private` would name pkg-config
    # modules, and none of the three ship one.
    set(FORKUNION_PC_PRIVATE "")
    foreach (fu_implicit_library IN LISTS CMAKE_CXX_IMPLICIT_LINK_LIBRARIES)
        if (fu_implicit_library MATCHES "^(stdc\\+\\+|c\\+\\+)$")
            list(APPEND FORKUNION_PC_PRIVATE "-l${fu_implicit_library}")
        endif ()
    endforeach ()
    if (CMAKE_THREAD_LIBS_INIT)
        list(APPEND FORKUNION_PC_PRIVATE "${CMAKE_THREAD_LIBS_INIT}")
    endif ()
    if (NOT FORKUNION_ATOMICS_ARE_BUILTIN)
        list(APPEND FORKUNION_PC_PRIVATE "-latomic")
    endif ()
    list(JOIN FORKUNION_PC_PRIVATE " " FORKUNION_PC_PRIVATE)

    configure_file(cmake/forkunion.pc.in "${CMAKE_CURRENT_BINARY_DIR}/forkunion.pc" @ONLY)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/forkunion.pc" DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endif ()

# Translate one tri-state to its macro. AUTO says nothing and lets the header decide; ON and OFF force the macro to 1 or
# 0. Knows nothing of the environment: there is no longer an environment to know.
function (forkunion_apply mode macro)
    if (mode STREQUAL "ON")
        target_compile_definitions(forkunion_header INTERFACE ${macro}=1)
    elseif (mode STREQUAL "OFF")
        target_compile_definitions(forkunion_header INTERFACE ${macro}=0)
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
        target_compile_definitions(forkunion_header INTERFACE ${macro}=0)
    endforeach ()
else ()
    # No environment check precedes these. Forcing topology or placement ON used to need libnuma on the build host, and
    # so needed a sentence here rather than a wall of missing symbols at link time; now it needs a Linux kernel, which
    # the header checks for itself. A capability the platform cannot honour still stops at an `#error` in `types.hpp`.
    forkunion_apply("${FORKUNION_WITH_TOPOLOGY}" FU_WITH_TOPOLOGY)
    forkunion_apply("${FORKUNION_WITH_PLACE_MEMORY_ON_DOMAIN}" FU_WITH_PLACE_MEMORY_ON_DOMAIN)
    forkunion_apply("${FORKUNION_WITH_PLACE_HUGE_PAGES_ON_DOMAIN}" FU_WITH_PLACE_HUGE_PAGES_ON_DOMAIN)
    forkunion_apply("${FORKUNION_WITH_PLACE_THREADS_BY_AFFINITY}" FU_WITH_PLACE_THREADS_BY_AFFINITY)
endif ()
