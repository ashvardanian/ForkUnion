/**
 *  @file probes/risc5_zabha.cpp
 *  @author Ash Vardanian
 *  @date October 26, 2026
 *  @brief ForkUnion probe: an `amoadd.b` word with its registers pinned, its instruction a template
 *      immediate - inline assembly alone.
 */
#if !(defined(__riscv) && __riscv_xlen == 64)
#error "64-bit RISC-V only"
#endif
#include <cstdint> // `std::uint8_t`, `std::uint32_t`, `std::uint64_t`
template <std::uint32_t instruction_>
std::uint64_t amo(void *word, std::uint64_t operand) {
    register std::uint64_t observed __asm__("a0") = 0;
    register void *address __asm__("a1") = word;
    register std::uint64_t value __asm__("a2") = operand;
    __asm__ __volatile__(".4byte %3" : "+r"(observed) : "r"(address), "r"(value), "i"(instruction_) : "memory");
    return observed;
}
int main() {
    std::uint8_t byte = 0;
    return static_cast<int>(amo<0x06c5852f>(&byte, 1)); // ? `amoadd.b.aqrl a0, a2, (a1)`
}
