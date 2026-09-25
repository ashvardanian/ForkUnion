# WebAssembly via Emscripten: 32-bit, one thread, the module any page and any runtime loads.
#
# ~~~
#   git clone https://github.com/emscripten-core/emsdk && ./emsdk/emsdk install latest
#   ./emsdk/emsdk activate latest && source ./emsdk/emsdk_env.sh
#   emcmake cmake -B build_wasm32_emscripten -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm32-emscripten.cmake
#   cmake --build build_wasm32_emscripten
# ~~~
#
# `emcmake` already points CMake at Emscripten's own toolchain file and at the SDK's `node` as the emulator, so this one
# chains to it and adds only what ForkUnion needs. It sees a toolchain file or an emulator of yours only as the single
# argument `-DCMAKE_TOOLCHAIN_FILE=…` or `-DCMAKE_CROSSCOMPILING_EMULATOR=…`; spelled with a space, its own wins.
#
# Without `-pthread` a module has one thread and no shared memory, so `FORKUNION_WITH_SHARED_MEMORY` derives 0: a
# consumer spawns one caller-inclusive thread and never constructs a `std::thread`, which Emscripten would abort on. The
# pool tests spawn more, so this shape compiles them without running them; the threaded module is the wasm64 file.

if (NOT DEFINED ENV{EMSDK})
    message(FATAL_ERROR "EMSDK is unset; source the SDK's `emsdk_env.sh` before configuring.")
endif ()

# Emscripten's own toolchain file names the processor `x86` unless this is set before it loads.
set(EMSCRIPTEN_SYSTEM_PROCESSOR wasm32)
include("$ENV{EMSDK}/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake")

set(CMAKE_EXE_LINKER_FLAGS_INIT "-sALLOW_MEMORY_GROWTH")
