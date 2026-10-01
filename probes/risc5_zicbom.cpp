/**
 *  @file probes/risc5_zicbom.cpp
 *  @author Ash Vardanian
 *  @date September 7, 2026
 *  @brief ForkUnion probe: the raw `cbo.clean` word.
 */
#if !(defined(__riscv) && __riscv_xlen == 64)
#error "64-bit RISC-V only"
#endif
int main() {
    int word = 0;
    register void const *address __asm__("a0") = &word;
    __asm__ __volatile__(".4byte 0x0015200f" ::"r"(address) : "memory");
    return word;
}
