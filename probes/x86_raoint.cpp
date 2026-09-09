/**
 *  @brief ForkUnion probe: the RAO-INT `aadd` as raw bytes - inline assembly alone.
 *  @author Ash Vardanian
 *  @file probes/x86_raoint.cpp
 *  @date September 7, 2026
 */
#if !(defined(__x86_64__) || defined(_M_X64))
#error "x86-64 only"
#endif
#include <cstdint> // `std::uint32_t`
int main() {
    std::uint32_t word = 0, operand = 1;
    __asm__ __volatile__(".byte 0x0f, 0x38, 0xfc, 0x08"
                         :
                         : "c"(operand), "a"(&word)
                         : "memory"); // ? `aadd %ecx, (%rax)`
    return static_cast<int>(word);
}
