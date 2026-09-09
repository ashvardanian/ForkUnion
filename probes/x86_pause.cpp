/**
 *  @brief ForkUnion probe: the `PAUSE` spin hint, or `_mm_pause`.
 *  @author Ash Vardanian
 *  @file probes/x86_pause.cpp
 *  @date September 7, 2026
 */
#if !(defined(__x86_64__) || defined(_M_X64))
#error "x86-64 only"
#endif
#include <immintrin.h> // `_mm_pause`
int main() {
#if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("pause");
#else
    _mm_pause();
#endif
    return 0;
}
