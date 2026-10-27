# cmake/forkunion_probes.cmake - which capabilities of `capabilities_t` the toolchain compiles
#
# One `forkunion_add_cpu_capability` row per instruction-level capability. Each probes `probes/<capability>.cpp`, the
# exact emission the header uses for it - inline assembly for the raw encodings, an assembler that takes
# `.arch_extension` for LSE and RCpc, the intrinsic where MSVC has one - into the cached
# `forkunion_target_<capability>_compiles`, caches `forkunion_target_<capability>_flags` empty, as no unit flag is
# needed, and defines its `FORKUNION_TARGET_<CAPABILITY>` macro on the target below. A probe fails off its own
# architecture, so every row runs everywhere. `-D FORKUNION_TARGET_<CAPABILITY>=0` leaves a capability out.
# `probes/README.md` has the table.
#
# A compiled capability is not one the unit's own flags enable, nor one the CPU runs. `forkunion_header` carries no
# macro, so its consumers read the compilation target's promise in `types.hpp`, and the runtime picks among the compiled
# ones.
include_guard(GLOBAL)

# Linked by the units that call CPU capability paths and pick them at runtime: each macro is 1 where the toolchain
# compiles the capability, else 0. Published as `forkunion::cpu_capabilities_compiled` for projects building ForkUnion
# in their own tree, like USearch, and never installed, as an installed header meets other toolchains.
add_library(forkunion_cpu_capabilities_compiled INTERFACE)
add_library(forkunion::cpu_capabilities_compiled ALIAS forkunion_cpu_capabilities_compiled)

# Probes one capability, named `capability_name_` in its probe and cache variable, and defines `capability_macro_` from
# the verdict. Tried as Release, without any target's flags, so a sanitized configuration does not fail the probes for
# want of its runtime. ThreadSanitizer sees no inline assembly, so a `thread` build leaves out the rows marked
# `HIDDEN_FROM_THREAD_SANITIZER`: the instruction-set atomics and the rungs above them.
function (forkunion_add_cpu_capability capability_name_ capability_macro_)
    cmake_parse_arguments(PARSE_ARGV 2 arg "HIDDEN_FROM_THREAD_SANITIZER" "" "")
    set(forkunion_target_${capability_name_}_flags
        ""
        CACHE INTERNAL "Flags enabling ${capability_macro_} across a unit"
    )
    if (NOT DEFINED forkunion_target_${capability_name_}_compiles)
        set(CMAKE_TRY_COMPILE_CONFIGURATION "Release")
        try_compile(
            forkunion_target_${capability_name_}_compiles ${CMAKE_BINARY_DIR}/forkunion_probes
            ${PROJECT_SOURCE_DIR}/probes/${capability_name_}.cpp
        )
    endif ()
    message(
        STATUS "Performing ISA probe ${capability_macro_} - compiles: ${forkunion_target_${capability_name_}_compiles}"
    )
    set(capability_enabled_ ${forkunion_target_${capability_name_}_compiles})
    if (DEFINED ${capability_macro_} AND NOT ${capability_macro_})
        set(capability_enabled_ FALSE)
    endif ()
    if (arg_HIDDEN_FROM_THREAD_SANITIZER AND FORKUNION_BUILD_SANITIZERS STREQUAL "thread")
        set(capability_enabled_ FALSE)
    endif ()
    target_compile_definitions(
        forkunion_cpu_capabilities_compiled INTERFACE "${capability_macro_}=$<BOOL:${capability_enabled_}>"
    )
endfunction ()

forkunion_add_cpu_capability(x86_pause FORKUNION_TARGET_X86_PAUSE)
forkunion_add_cpu_capability(x86_tpause FORKUNION_TARGET_X86_TPAUSE)
forkunion_add_cpu_capability(x86_cldemote FORKUNION_TARGET_X86_CLDEMOTE)
forkunion_add_cpu_capability(x86_cmpccxadd FORKUNION_TARGET_X86_CMPCCXADD HIDDEN_FROM_THREAD_SANITIZER)
forkunion_add_cpu_capability(x86_raoint FORKUNION_TARGET_X86_RAOINT HIDDEN_FROM_THREAD_SANITIZER)
forkunion_add_cpu_capability(arm64_yield FORKUNION_TARGET_ARM64_YIELD)
forkunion_add_cpu_capability(arm64_wfet FORKUNION_TARGET_ARM64_WFET)
forkunion_add_cpu_capability(arm64_dc_cvac FORKUNION_TARGET_ARM64_DC_CVAC)
forkunion_add_cpu_capability(arm64_prfm FORKUNION_TARGET_ARM64_PRFM)
forkunion_add_cpu_capability(arm64_lse FORKUNION_TARGET_ARM64_LSE HIDDEN_FROM_THREAD_SANITIZER)
forkunion_add_cpu_capability(arm64_rcpc FORKUNION_TARGET_ARM64_RCPC HIDDEN_FROM_THREAD_SANITIZER)
forkunion_add_cpu_capability(riscv64_pause FORKUNION_TARGET_RISCV64_PAUSE)
forkunion_add_cpu_capability(riscv64_wrs FORKUNION_TARGET_RISCV64_WRS)
forkunion_add_cpu_capability(riscv64_zicbom FORKUNION_TARGET_RISCV64_ZICBOM)
forkunion_add_cpu_capability(riscv64_zicbop FORKUNION_TARGET_RISCV64_ZICBOP)
forkunion_add_cpu_capability(riscv64_atomic FORKUNION_TARGET_RISCV64_ATOMIC HIDDEN_FROM_THREAD_SANITIZER)
forkunion_add_cpu_capability(riscv64_zacas FORKUNION_TARGET_RISCV64_ZACAS HIDDEN_FROM_THREAD_SANITIZER)
forkunion_add_cpu_capability(riscv64_zabha FORKUNION_TARGET_RISCV64_ZABHA HIDDEN_FROM_THREAD_SANITIZER)
