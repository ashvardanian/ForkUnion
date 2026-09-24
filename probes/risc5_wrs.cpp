/**
 *  @file probes/risc5_wrs.cpp
 *  @author Ash Vardanian
 *  @date September 7, 2026
 *  @brief ForkUnion probe: the raw `WRS.STO` word.
 */
#if !(defined(__riscv) && __riscv_xlen == 64)
#error "64-bit RISC-V only"
#endif
int main() {
    __asm__ __volatile__(".4byte 0x01d00073" ::: "memory");
    return 0;
}
