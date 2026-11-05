/**
 *  @file probes/riscv64_wrs.cpp
 *  @author Ash Vardanian
 *  @date September 7, 2026
 *  @brief ForkUnion probe: the @c LR.W that arms it, then the raw @c WRS.STO word.
 */
#if !(defined(__riscv) && __riscv_xlen == 64)
#error "64-bit RISC-V only"
#endif
int main() {
    unsigned value = 0, reserved;
    __asm__ __volatile__("lr.w %0, (%1)\n.4byte 0x01d00073" : "=&r"(reserved) : "r"(&value) : "memory");
    return 0;
}
