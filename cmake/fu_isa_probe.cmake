# cmake/fu_isa_probe.cmake - shared probe infrastructure
#
# One compile probe per instruction-level capability bit of `capabilities_t`, each the exact emission the header uses
# for that bit, so the verdict says whether this toolchain can build that path: inline assembly for the raw encodings,
# an assembler that takes `.arch_extension` for LSE and RCpc, the intrinsic where MSVC has one. The per-architecture
# files (fu_x86_isa_probes.cmake and kin) call `fu_instruction_set_probe_()` per bit and
# `fu_build_instruction_set_definitions_()` once to append their verdicts to the one cached `fu_compile_definitions_`
# list of `FORKUNION_TARGET_<BIT>=0/1`. The check cache is the knob - a preset `-D fu_target_<bit>_compiles=0` skips
# that probe and every reader follows it. `probes/README.md` has the table.
include_guard(GLOBAL)
include(CheckSourceCompiles)

set(fu_compile_definitions_
    ""
    CACHE INTERNAL ""
)

# Can the toolchain build this bit's path? Tried as Release with no inherited flags, so a sanitized configuration does
# not fail the probes for want of its runtime.
function (fu_instruction_set_probe_ variable_ probe_file_)
    set(CMAKE_TRY_COMPILE_CONFIGURATION "Release")
    set(CMAKE_REQUIRED_FLAGS "")
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/${probe_file_}" probe_source_)
    check_source_compiles(CXX "${probe_source_}" ${variable_}_compiles)
    set(${variable_}_flags
        ""
        CACHE INTERNAL ""
    )
endfunction ()

# Appends one architecture's `FORKUNION_TARGET_<BIT>=0/1` verdicts to the cached list the compiled libraries and the
# tests read.
function (fu_build_instruction_set_definitions_ architecture_name_ capability_names_)
    set(compile_definitions_ "")
    foreach (capability_ IN LISTS capability_names_)
        string(TOLOWER "${capability_}" capability_lowercase_)
        if (fu_target_${capability_lowercase_}_compiles)
            list(APPEND compile_definitions_ "FORKUNION_TARGET_${capability_}=1")
        else ()
            list(APPEND compile_definitions_ "FORKUNION_TARGET_${capability_}=0")
        endif ()
    endforeach ()
    list(JOIN compile_definitions_ " " summary_)
    message(STATUS "${architecture_name_} compile verdicts: ${summary_}")
    list(APPEND fu_compile_definitions_ ${compile_definitions_})
    set(fu_compile_definitions_
        "${fu_compile_definitions_}"
        CACHE INTERNAL ""
    )
endfunction ()
