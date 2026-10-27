# Toolchain probes

One probe per instruction-level capability bit of `capabilities_t`, each emitting exactly what the header emits for that bit: a mnemonic, a raw encoding, or the intrinsic MSVC reaches it through.
Each row of `cmake/fu_isa_probe.cmake` compiles one at configure time into the cached `fu_target_<capability>_compiles` and defines `FORKUNION_TARGET_<CAPABILITY>=0/1` on `forkunion::cpu_capabilities_compiled`, which the compiled libraries and the tests link privately, never `forkunion::header`.
A consumer's own unit derives each bit in `types.hpp` - what the toolchain builds when it sets `FORKUNION_RUNTIME_DISPATCH`, what the compilation target promises otherwise.

The verdict answers one question: can this toolchain build this bit's path.
Every extension instruction is a raw encoding, so those probes fail only where `__asm__` is missing.
The LSE and RCpc mnemonics ride on `.arch_extension`, which every assembler of the last decade takes.
MSVC has no inline assembly and spells both rungs as intrinsics instead, so those two probes carry a second arm.
`__ldapr32` needs no flag, while the LSE arithmetic goes through `_Interlocked*`, which stays inline only under `/arch:armv8.1`, so the LSE probe asks for that flag through `__ARM_FEATURE_ATOMICS`, the same macro `FORKUNION_HAS_ARM64_ATOMIC_INTRINSICS_` keys the header's definitions on.
A baseline MSVC therefore answers LSE 0 and RCpc 1, and the demotion in `types.hpp` takes the child rung down with its parent.
The RISC-V base atomics are the A extension's own mnemonics, so that probe also needs the extension in `-march`, as every `rv64gc` build has.
Whether the CPU has the instruction is the runtime's question, answered by `cpu_capabilities()`, never by a probe.

The OS-level bits - owned threads, topology, placements, transparent huge pages - have no probe: the `FORKUNION_WITH_*` tri-states and the runtime answer those.

Where a probe cell offers a second spelling, it is the intrinsic MSVC reaches the bit through.

| bit             | probe                                                  | needs                          |
| :-------------- | :----------------------------------------------------- | :----------------------------- |
| `x86_pause`     | `pause`, or `_mm_pause`                                | nothing                        |
| `x86_tpause`    | `umonitor` and `umwait` bytes, or intrinsics           | nothing                        |
| `x86_cldemote`  | `cldemote` bytes, or `_mm_cldemote` and `_m_prefetchw` | nothing                        |
| `x86_cmpccxadd` | `cmpbexadd` bytes                                      | inline assembly                |
| `x86_raoint`    | `aadd` bytes                                           | inline assembly                |
| `arm64_yield`   | `yield`, or `__yield`                                  | nothing                        |
| `arm64_prfm`    | `prfm pldl2keep`, or `__prefetch2`                     | nothing                        |
| `arm64_rcpc`    | `.arch_extension rcpc` and `ldapr`, or `__ldapr32`     | nothing                        |
| `arm64_lse`     | `.arch_extension lse` and `swp`, or `__swp32`          | `/arch:armv8.1` on MSVC        |
| `arm64_wfet`    | `wfet` word                                            | inline assembly                |
| `arm64_dc_cvac` | `dc cvac`                                              | inline assembly                |
| `risc5_atomic`  | `amoswap`, `lr` and `sc`                               | inline assembly, A in `-march` |
| `risc5_pause`   | `pause` word                                           | inline assembly                |
| `risc5_wrs`     | `wrs.sto` word                                         | inline assembly                |
| `risc5_zicbom`  | `cbo.clean` word                                       | inline assembly                |
| `risc5_zicbop`  | `prefetch.r` word                                      | inline assembly                |
| `risc5_zacas`   | `amocas.w` word                                        | inline assembly                |
| `risc5_zabha`   | `amoadd.b` word                                        | inline assembly                |
