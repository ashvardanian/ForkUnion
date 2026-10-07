# cmake/fu_isa_probe.cmake - which capabilities of `capabilities_t` the toolchain compiles
#
# One `fu_cpu_capability_` row per instruction-level capability. Each probes `probes/<capability>.cpp`, the exact
# emission the header uses for it - inline assembly for the raw encodings, an assembler that takes `.arch_extension` for
# LSE and RCpc, the intrinsic where MSVC has one - into the cached `fu_target_<capability>_compiles`, and defines its
# `FORKUNION_TARGET_<CAPABILITY>` macro on the target below. A probe fails off its own architecture, so every row runs
# everywhere. `-D FORKUNION_TARGET_<CAPABILITY>=0` leaves a capability out. `probes/README.md` has the table.
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
# want of its runtime.
function (fu_cpu_capability_ capability_name_ capability_macro_)
    if (NOT DEFINED fu_target_${capability_name_}_compiles)
        set(CMAKE_TRY_COMPILE_CONFIGURATION "Release")
        try_compile(
            fu_target_${capability_name_}_compiles ${CMAKE_BINARY_DIR}/fu_probes
            ${PROJECT_SOURCE_DIR}/probes/${capability_name_}.cpp
        )
    endif ()
    message(STATUS "Performing ISA probe ${capability_macro_} - compiles: ${fu_target_${capability_name_}_compiles}")
    set(capability_enabled_ ${fu_target_${capability_name_}_compiles})
    if (DEFINED ${capability_macro_} AND NOT ${capability_macro_})
        set(capability_enabled_ FALSE)
    endif ()
    target_compile_definitions(
        forkunion_cpu_capabilities_compiled INTERFACE "${capability_macro_}=$<BOOL:${capability_enabled_}>"
    )
endfunction ()

fu_cpu_capability_(x86_pause FORKUNION_TARGET_X86_PAUSE)
fu_cpu_capability_(x86_tpause FORKUNION_TARGET_X86_TPAUSE)
fu_cpu_capability_(x86_cldemote FORKUNION_TARGET_X86_CLDEMOTE)
fu_cpu_capability_(x86_cmpccxadd FORKUNION_TARGET_X86_CMPCCXADD)
fu_cpu_capability_(x86_raoint FORKUNION_TARGET_X86_RAOINT)
fu_cpu_capability_(arm64_yield FORKUNION_TARGET_ARM64_YIELD)
fu_cpu_capability_(arm64_wfet FORKUNION_TARGET_ARM64_WFET)
fu_cpu_capability_(arm64_dc_cvac FORKUNION_TARGET_ARM64_DC_CVAC)
fu_cpu_capability_(arm64_lse FORKUNION_TARGET_ARM64_LSE)
fu_cpu_capability_(arm64_rcpc FORKUNION_TARGET_ARM64_RCPC)
fu_cpu_capability_(risc5_pause FORKUNION_TARGET_RISC5_PAUSE)
fu_cpu_capability_(risc5_wrs FORKUNION_TARGET_RISC5_WRS)
fu_cpu_capability_(risc5_zicbom FORKUNION_TARGET_RISC5_ZICBOM)
fu_cpu_capability_(risc5_atomic FORKUNION_TARGET_RISC5_ATOMIC)
fu_cpu_capability_(risc5_zacas FORKUNION_TARGET_RISC5_ZACAS)
