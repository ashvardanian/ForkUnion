/**
 *  @file probes/arm64_lse.cpp
 *  @author Ash Vardanian
 *  @date September 7, 2026
 *  @brief ForkUnion probe: the LSE read-modify-writes, as a mnemonic or as an intrinsic.
 */
#if !(defined(__aarch64__) || defined(_M_ARM64))
#error "AArch64 only"
#endif
#include <cstdint> // `std::uint32_t`

#if !(defined(__GNUC__) || defined(__clang__))
#include <intrin.h> // `__swp32`
#endif

int main() {
    std::uint32_t word = 0, desired = 1, observed;
#if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__(".arch_extension lse\n\tswp %w1, %w0, [%2]"
                         : "=r"(observed)
                         : "r"(desired), "r"(&word)
                         : "memory");
#elif defined(__ARM_FEATURE_ATOMICS)
    observed = __swp32(&word, desired);
#else
#error "MSVC out-of-lines the `_Interlocked*` half to the CRT without `/arch:armv8.1`"
#endif
    return static_cast<int>(observed);
}
