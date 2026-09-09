/**
 *  @brief ForkUnion probe: the LSE read-modify-writes under `.arch_extension lse`.
 *  @author Ash Vardanian
 *  @file probes/arm64_lse.cpp
 *  @date September 7, 2026
 */
#if !(defined(__aarch64__) || defined(_M_ARM64))
#error "AArch64 only"
#endif
#include <cstdint> // `std::uint32_t`
int main() {
    std::uint32_t word = 0, desired = 1, observed;
    __asm__ __volatile__(".arch_extension lse\n\tswp %w1, %w0, [%2]"
                         : "=r"(observed)
                         : "r"(desired), "r"(&word)
                         : "memory");
    return static_cast<int>(observed);
}
