/**
 *  @file probes/ppc64_amo.cpp
 *  @author Ash Vardanian
 *  @date November 12, 2026
 *  @brief ForkUnion probe: the ISA 3.0 store atomic @c stwat, named under <tt>.machine power9</tt>.
 */
#if !(defined(__powerpc64__) || defined(__ppc64__) || defined(_ARCH_PPC64))
#error "64-bit Power only"
#endif
#include <cstdint> // `std::uint32_t`
int main() {
    std::uint32_t word = 0, operand = 1;
    __asm__ __volatile__(".machine push\n\t.machine power9\n\tstwat %1, %0, 0\n\t.machine pop"
                         :
                         : "b"(&word), "r"(operand)
                         : "memory"); // ? `stwat` with function code 0, store add
    return static_cast<int>(word != operand);
}
