/**
 *  @brief ForkUnion probe: the A extension - an `amoswap.w` and an `lr.w`/`sc.w` loop.
 *  @author Ash Vardanian
 *  @file probes/risc5_atomic.cpp
 *  @date September 9, 2026
 */
#if !(defined(__riscv) && __riscv_xlen == 64)
#error "64-bit RISC-V only"
#endif
#include <cstdint> // `std::uint32_t`
int main() {
    std::uint32_t word = 0, desired = 1, swapped;
    __asm__ __volatile__("amoswap.w.aqrl %0, %2, (%1)" : "=r"(swapped) : "r"(&word), "r"(desired) : "memory");
    std::int64_t const wanted = 1;
    std::int64_t observed, failed;
    __asm__ __volatile__("1:\n\t"
                         "lr.w.aqrl %[observed], (%[word])\n\t"
                         "bne %[observed], %[wanted], 2f\n\t"
                         "sc.w.rl %[failed], %[desired], (%[word])\n\t"
                         "bnez %[failed], 1b\n"
                         "2:"
                         : [observed] "=&r"(observed), [failed] "=&r"(failed)
                         : [word] "r"(&word), [wanted] "r"(wanted), [desired] "r"(desired)
                         : "memory");
    return static_cast<int>(swapped + observed);
}
