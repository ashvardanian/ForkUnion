# cmake/fu_isa_probe.cmake - shared probe infrastructure
#
# One compile probe per instruction-level capability bit of `capabilities_t`, each the exact emission the header uses
# for that bit, so the verdict says whether this toolchain can build that path: inline assembly for the raw encodings,
# an assembler that takes `.arch_extension` for LSE and RCpc, the intrinsic where MSVC has one. The per-architecture
# files (fu_x86_isa_probes.cmake and kin) call `fu_isa_probe_()` per bit and `fu_build_isa_defs_()` once to collect
# `FU_TARGET_<BIT>=0/1`. `probes/README.md` has the table.
include(CheckSourceCompiles)

# Save and restore the probe state around a file's rows, and try-compile as Release so a sanitized configuration does
# not fail the probes for want of its runtime.
macro (fu_isa_probes_begin_)
    set(fu_saved_required_flags_ "${CMAKE_REQUIRED_FLAGS}")
    set(fu_saved_try_compile_config_ "${CMAKE_TRY_COMPILE_CONFIGURATION}")
    set(CMAKE_REQUIRED_FLAGS "")
    set(CMAKE_TRY_COMPILE_CONFIGURATION "Release")
endmacro ()
macro (fu_isa_probes_end_)
    set(CMAKE_REQUIRED_FLAGS "${fu_saved_required_flags_}")
    set(CMAKE_TRY_COMPILE_CONFIGURATION "${fu_saved_try_compile_config_}")
endmacro ()

# Can the toolchain build this bit's path? The verdict is cached beside the flags it was reached with - none, since
# every path is a raw encoding, a mnemonic, or an intrinsic - for consumers composing units out of several bits.
macro (fu_isa_probe_ var_ probe_file_)
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/${probe_file_}" fu_probe_source_)
    check_source_compiles(CXX "${fu_probe_source_}" ${var_}_compiles)
    set(${var_}_flags
        ""
        CACHE INTERNAL ""
    )
endmacro ()

# Collects one architecture's verdicts into `fu_<arch_prefix_>_compile_defs_`.
function (fu_build_isa_defs_ arch_prefix_ arch_display_ bit_list_)
    set(compile_defs_ "")
    foreach (bit_ IN LISTS bit_list_)
        string(TOLOWER "${bit_}" bit_lower_)
        if (fu_target_${bit_lower_}_compiles)
            list(APPEND compile_defs_ "FU_TARGET_${bit_}=1")
        else ()
            list(APPEND compile_defs_ "FU_TARGET_${bit_}=0")
        endif ()
    endforeach ()
    message(STATUS "${arch_display_} capability probes: ${compile_defs_}")
    set(fu_${arch_prefix_}_compile_defs_
        "${compile_defs_}"
        PARENT_SCOPE
    )
endfunction ()
