/**
 *  @brief ForkUnion probe: the `CLDEMOTE` and `PREFETCHW` encodings, or `_mm_cldemote` and `_m_prefetchw`.
 *  @author Ash Vardanian
 *  @file probes/x86_cldemote.cpp
 *  @date September 7, 2026
 */
#if !(defined(__x86_64__) || defined(_M_X64))
#error "x86-64 only"
#endif
#include <immintrin.h> // `_mm_cldemote`
#if !(defined(__GNUC__) || defined(__clang__))
#include <intrin.h> // `_m_prefetchw`
#endif
int main() {
    int word = 0;
#if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__(".byte 0x0f, 0x1c, 0x00" ::"a"(&word) : "memory");
    __asm__ __volatile__(".byte 0x0f, 0x0d, 0x08" ::"a"(&word) : "memory");
#else
    _mm_cldemote(&word);
    _m_prefetchw(&word);
#endif
    return word;
}
