# The CPU ForkUnion's own units are tuned for.

# FORKUNION_MARCH raises the floor of ForkUnion's own units, its tests and benchmarks included: an ISA level like
# `x86-64-v3`, `native` for this host, or a CPU name, also when cross-compiling. The runtime still picks each
# instruction-level path by detection. The probes compile at `forkunion_unit_arch_flags_`.
set(forkunion_unit_arch_flags_)
if (FORKUNION_MARCH STREQUAL "native" AND CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "FORKUNION_MARCH=native needs the host as the target; name a CPU instead")
elseif (EMSCRIPTEN OR CMAKE_SYSTEM_PROCESSOR MATCHES "^wasm")
    # WebAssembly tiers like `v128` name SIMD levels that ForkUnion has no units for, so they add no flags.
elseif (FORKUNION_MARCH)
    message(WARNING "FORKUNION_MARCH=${FORKUNION_MARCH}: baking specific instructions; result may not run elsewhere")
    if (MSVC AND FORKUNION_MARCH STREQUAL "native")
        # MSVC has no `/arch:native`, and no capability row needs unit flags, so the toolchain default stays.
    elseif (MSVC)
        message(
            FATAL_ERROR "MSVC takes FORKUNION_MARCH=native only; pass an /arch: value such as AVX2 or armv8.1 through "
                        "CMAKE_CXX_FLAGS instead of `${FORKUNION_MARCH}`"
        )
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|AARCH64|arm64|ARM64")
        # AArch64 names CPUs through `-mcpu`, like `neoverse-v2`, as `-march` takes only architecture versions; Apple
        # Clang's `-march=native` also advertises just a subset of host features.
        set(forkunion_unit_arch_flags_ -mcpu=${FORKUNION_MARCH})
    else ()
        set(forkunion_unit_arch_flags_ -march=${FORKUNION_MARCH})
    endif ()
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:${forkunion_unit_arch_flags_}>")
endif ()
