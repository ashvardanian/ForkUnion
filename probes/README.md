# Toolchain probes

One probe per instruction-level capability bit of `capabilities_t`, each emitting exactly what the header emits for that bit: a mnemonic, a raw encoding, or the intrinsic MSVC reaches it through.
The per-architecture `cmake/fu_<arch>_isa_probes.cmake` files compile them at configure time and publish the verdicts on `forkunion::header` as `FU_TARGET_<BIT>=0/1`, build interface only, so a consumer building against the headers compiles the same paths the pre-compiled libraries do.
A consumer without the probes gets the same answer the header derives on its own from the architecture and the compiler.

The verdict answers one question: can this toolchain build this bit's path.
Every extension instruction is a raw encoding, so those probes fail only where `__asm__` is missing.
The LSE and RCpc mnemonics ride on `.arch_extension`, which every assembler of the last decade takes.
The RISC-V base atomics are the A extension's own mnemonics, so that probe also needs the extension in `-march`, as every `rv64gc` build has.
Whether the CPU has the instruction is the runtime's question, answered by `cpu_capabilities()`, never by a probe.

The OS-level bits - owned threads, topology, placements, transparent huge pages - have no probe: the `FU_WITH_*` tri-states and the runtime answer those.

| bit | probe | needs |
|---|---|---|
| `x86_pause`, `x86_tpause`, `x86_cldemote` | encodings, or `_mm_pause`, `_umonitor`/`_umwait`, `_mm_cldemote`/`_m_prefetchw` | nothing |
| `x86_cmpccxadd`, `x86_raoint` | raw bytes | inline assembly |
| `arm64_yield` | `yield`, or `__yield` | nothing |
| `arm64_wfet`, `risc5_pause`, `risc5_wrs`, `risc5_zicbom`, `risc5_zacas` | raw words | inline assembly |
| `arm64_dc_cvac` | `dc cvac` | inline assembly |
| `risc5_atomic` | `amoswap`, `lr` and `sc` | inline assembly and the A extension in `-march` |
| `arm64_lse`, `arm64_rcpc` | `.arch_extension` and a mnemonic | inline assembly |
