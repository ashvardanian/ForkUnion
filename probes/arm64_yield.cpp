/**
 *  @file probes/arm64_yield.cpp
 *  @author Ash Vardanian
 *  @date September 7, 2026
 *  @brief ForkUnion probe: the @c YIELD hint, or @c __yield.
 */
#if !(defined(__aarch64__) || defined(_M_ARM64))
#error "AArch64 only"
#endif
#if !(defined(__GNUC__) || defined(__clang__))
#include <intrin.h> // `__yield`
#endif
int main() {
#if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("yield");
#else
    __yield();
#endif
    return 0;
}
