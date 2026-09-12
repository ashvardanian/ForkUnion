# WebAssembly via Emscripten: 64-bit addressing, one shared memory, a Web Worker per pool thread.
#
# ~~~
#   git clone https://github.com/emscripten-core/emsdk && ./emsdk/emsdk install latest
#   ./emsdk/emsdk activate latest && source ./emsdk/emsdk_env.sh
#   emcmake cmake -B build_wasm64_emscripten -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm64-emscripten.cmake
#   cmake --build build_wasm64_emscripten
#   ctest --test-dir build_wasm64_emscripten
# ~~~
#
# `emcmake` already points CMake at Emscripten's own toolchain file and at the SDK's `node` as the emulator, so this one
# chains to it and adds only what ForkUnion needs. It sees a toolchain file or an emulator of yours only as the single
# argument `-DCMAKE_TOOLCHAIN_FILE=…` or `-DCMAKE_CROSSCOMPILING_EMULATOR=…`; spelled with a space, its own wins.
#
# `-pthread` defines `__EMSCRIPTEN_PTHREADS__`, so `FU_WITH_SHARED_MEMORY` derives 1 and the pools are real. The width
# and the threads feature are whole-module choices, so both are fixed here: a page serves this module only when it is
# cross-origin isolated, and Node runs it from 24 on, where memory64 is on by default. The single-threaded module is the
# wasm32 file.

if (NOT DEFINED ENV{EMSDK})
    message(FATAL_ERROR "EMSDK is unset; source the SDK's `emsdk_env.sh` before configuring.")
endif ()

# Emscripten's own toolchain file reads the width off these flags as it loads, so they precede the include.
set(CMAKE_C_FLAGS
    "-pthread -sMEMORY64"
    CACHE STRING "Flags used by the C compiler during all build types."
)
set(CMAKE_CXX_FLAGS
    "-pthread -sMEMORY64"
    CACHE STRING "Flags used by the CXX compiler during all build types."
)
set(EMSCRIPTEN_SYSTEM_PROCESSOR wasm64)
include("$ENV{EMSDK}/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake")

# The main thread is proxied to a worker, so a blocking join never stalls the host's event loop.
set(CMAKE_EXE_LINKER_FLAGS_INIT "-pthread -sMEMORY64 -sPROXY_TO_PTHREAD -sALLOW_MEMORY_GROWTH -sEXIT_RUNTIME=1")
