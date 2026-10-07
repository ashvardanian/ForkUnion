/**
 *  @file probes/arm64_prfm.cpp
 *  @author Ash Vardanian
 *  @date October 7, 2026
 *  @brief ForkUnion probe: the `PRFM PLDL2KEEP` read prefetch, or @c __prefetch2.
 */
#if !(defined(__aarch64__) || defined(_M_ARM64))
#error "AArch64 only"
#endif
#if !(defined(__GNUC__) || defined(__clang__))
#include <intrin.h> // `__prefetch2`
#endif
int main() {
    int word = 0;
#if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("prfm pldl2keep, [%0]" ::"r"(&word) : "memory");
#else
    __prefetch2(&word, 0x02);
#endif
    return word;
}
