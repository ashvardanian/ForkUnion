/**
 *  @brief ForkUnion probe: the CMPCCXADD conditional add as raw bytes - inline assembly alone.
 *  @author Ash Vardanian
 *  @file probes/x86_cmpccxadd.cpp
 *  @date September 7, 2026
 */
#if !(defined(__x86_64__) || defined(_M_X64))
#error "x86-64 only"
#endif
#include <cstdint> // `std::uint32_t`
int main() {
    std::uint32_t word = 0, bound = 1, addend = 1;
    __asm__ __volatile__(".byte 0xc4, 0xe2, 0x69, 0xe6, 0x08" // ? `cmpbexadd %edx, %ecx, (%rax)`
                         : "+c"(bound)
                         : "d"(addend), "a"(&word)
                         : "memory", "cc");
    return static_cast<int>(bound);
}
