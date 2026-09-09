/**
 *  @brief ForkUnion probe: the `amocas.w` word with its registers pinned - inline assembly alone.
 *  @author Ash Vardanian
 *  @file probes/risc5_zacas.cpp
 *  @date September 7, 2026
 */
#if !(defined(__riscv) && __riscv_xlen == 64)
#error "64-bit RISC-V only"
#endif
#include <cstdint> // `std::uint32_t`
int main() {
    std::uint32_t word = 0, desired = 1;
    register std::int64_t observed __asm__("a0") = 0;
    register std::uint32_t *address __asm__("a1") = &word;
    register std::uint32_t value __asm__("a2") = desired;
    __asm__ __volatile__(".4byte 0x2ec5a52f"
                         : "+r"(observed)
                         : "r"(address), "r"(value)
                         : "memory"); // ? `amocas.w.aqrl a0, a2, (a1)`
    return static_cast<int>(observed);
}
