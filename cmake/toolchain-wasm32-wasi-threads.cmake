# WebAssembly via the WASI SDK: `wasm32-wasip1-threads`, a thread per pool worker over one shared memory.
#
# ~~~
#   export WASI_SDK_PATH=/opt/wasi-sdk-34.0-x86_64-linux
#   cmake -B build_wasm32_wasi_threads -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm32-wasi-threads.cmake
#   cmake --build build_wasm32_wasi_threads
#   ctest --test-dir build_wasm32_wasi_threads
# ~~~
#
# The SDK ships the toolchain file naming its compilers, `-pthread` and the imported shared memory, so this one includes
# it and adds the triple and the runtime `ctest` drives: Wasmtime with its threads proposal and its WASI threads imports
# on, the two switches a bare `wasmtime` leaves off. `-pthread` defines `_REENTRANT`, so `FU_WITH_SHARED_MEMORY` derives
# 1 and the pools are real.

if (NOT DEFINED ENV{WASI_SDK_PATH})
    message(FATAL_ERROR "WASI_SDK_PATH is unset; point it at an unpacked wasi-sdk release before configuring.")
endif ()

set(WASI_SDK_PREFIX "$ENV{WASI_SDK_PATH}")
include("${WASI_SDK_PREFIX}/share/cmake/wasi-sdk-pthread.cmake")

# The SDK file names `wasm32-wasi-threads`, but its sysroot keeps the libraries under `wasm32-wasip1-threads` alone.
set(CMAKE_C_COMPILER_TARGET wasm32-wasip1-threads)
set(CMAKE_CXX_COMPILER_TARGET wasm32-wasip1-threads)
set(CMAKE_ASM_COMPILER_TARGET wasm32-wasip1-threads)
# The default `noeh` libc++abi has no `__cxa_throw`, so exceptions stay off.
string(APPEND CMAKE_CXX_FLAGS " -fno-exceptions")
# A shared memory stops at its initial size unless given a ceiling, and every spawned thread allocates its stack there.
string(APPEND CMAKE_EXE_LINKER_FLAGS " -Wl,--max-memory=2147483648")

find_program(FU_WASMTIME_ wasmtime REQUIRED PATHS "$ENV{HOME}/.wasmtime/bin")
set(CMAKE_CROSSCOMPILING_EMULATOR "${FU_WASMTIME_};run;-W;threads=y;-S;threads=y")
