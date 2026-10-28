# The CPU ForkUnion's own units are tuned for.

# FORKUNION_TARGET_ARCH tunes ForkUnion's own units, its tests and benchmarks included, for one CPU: `native` for this
# host, or a CPU name, also when cross-compiling. The runtime still picks each instruction-level path by detection.
if (FORKUNION_TARGET_ARCH STREQUAL "native" AND CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "FORKUNION_TARGET_ARCH=native needs the host as the target; name a CPU instead")
elseif (EMSCRIPTEN OR CMAKE_SYSTEM_PROCESSOR MATCHES "^wasm")
    # WebAssembly tiers like `v128` name SIMD levels that ForkUnion has no units for, so they add no flags.
elseif (FORKUNION_TARGET_ARCH)
    message(
        WARNING
            "FORKUNION_TARGET_ARCH=${FORKUNION_TARGET_ARCH}: baking specific instructions; result may not run elsewhere"
    )
    if (MSVC AND FORKUNION_TARGET_ARCH STREQUAL "native")
        # MSVC has no `/arch:native`, and no capability row needs unit flags, so the toolchain default stays.
    elseif (MSVC)
        message(
            FATAL_ERROR
                "MSVC takes FORKUNION_TARGET_ARCH=native only; pass an /arch: value such as AVX2 or armv8.1 through "
                "CMAKE_CXX_FLAGS instead of `${FORKUNION_TARGET_ARCH}`"
        )
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|AARCH64|arm64|ARM64")
        # AArch64 names CPUs through `-mcpu`, like `neoverse-v2`, as `-march` takes only architecture versions; Apple
        # Clang's `-march=native` also advertises just a subset of host features.
        add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-mcpu=${FORKUNION_TARGET_ARCH}>")
    else ()
        add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-march=${FORKUNION_TARGET_ARCH}>")
    endif ()
endif ()
