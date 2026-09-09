/**
 *  @brief ForkUnion probe: the RCpc acquiring loads under `.arch_extension rcpc`.
 *  @author Ash Vardanian
 *  @file probes/arm64_rcpc.cpp
 *  @date September 7, 2026
 */
#if !(defined(__aarch64__) || defined(_M_ARM64))
#error "AArch64 only"
#endif
#include <cstdint> // `std::uint32_t`
int main() {
    std::uint32_t word = 0, value;
    __asm__ __volatile__(".arch_extension rcpc\n\tldapr %w0, [%1]" : "=r"(value) : "r"(&word) : "memory");
    return static_cast<int>(value);
}
