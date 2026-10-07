/**
 *  @file probes/risc5_zicbop.cpp
 *  @author Ash Vardanian
 *  @date October 7, 2026
 *  @brief ForkUnion probe: the raw `prefetch.r` word.
 */
#if !(defined(__riscv) && __riscv_xlen == 64)
#error "64-bit RISC-V only"
#endif
int main() {
    int word = 0;
    register void const *address __asm__("a0") = &word;
    __asm__ __volatile__(".4byte 0x00156013" ::"r"(address) : "memory");
    return word;
}
