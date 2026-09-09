/**
 *  @brief ForkUnion probe: the raw `WRS.STO` word.
 *  @author Ash Vardanian
 *  @file probes/risc5_wrs.cpp
 *  @date September 7, 2026
 */
#if !(defined(__riscv) && __riscv_xlen == 64)
#error "64-bit RISC-V only"
#endif
int main() {
    __asm__ __volatile__(".4byte 0x01d00073" ::: "memory");
    return 0;
}
