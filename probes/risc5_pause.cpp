/**
 *  @file probes/risc5_pause.cpp
 *  @author Ash Vardanian
 *  @date September 7, 2026
 *  @brief ForkUnion probe: the raw @c Zihintpause word, `FENCE W, 0`.
 */
#if !(defined(__riscv) && __riscv_xlen == 64)
#error "64-bit RISC-V only"
#endif
int main() {
    __asm__ __volatile__(".4byte 0x0100000f");
    return 0;
}
