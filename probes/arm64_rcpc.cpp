/**
 *  @file probes/arm64_rcpc.cpp
 *  @author Ash Vardanian
 *  @date September 7, 2026
 *  @brief ForkUnion probe: the RCpc acquiring loads, as a mnemonic or as an intrinsic.
 */
#if !(defined(__aarch64__) || defined(_M_ARM64))
#error "AArch64 only"
#endif
#include <cstdint> // `std::uint32_t`

#if !(defined(__GNUC__) || defined(__clang__))
#include <intrin.h> // `__ldapr32`
#endif

int main() {
    std::uint32_t word = 0, value;
#if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__(".arch_extension rcpc\n\tldapr %w0, [%1]" : "=r"(value) : "r"(&word) : "memory");
#else
    value = __ldapr32(&word);
#endif
    return static_cast<int>(value);
}
