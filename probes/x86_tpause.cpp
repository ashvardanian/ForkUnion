/**
 *  @brief ForkUnion probe: the `UMONITOR` & `UMWAIT` encodings behind `TPAUSE`, or their `<immintrin.h>` intrinsics.
 *  @author Ash Vardanian
 *  @file probes/x86_tpause.cpp
 *  @date September 7, 2026
 */
#if !(defined(__x86_64__) || defined(_M_X64))
#error "x86-64 only"
#endif
#include <cstdint>     // `std::uint32_t`
#include <immintrin.h> // `_umonitor`, `_umwait`
int main() {
    std::uint32_t word = 0;
    std::uint64_t const deadline = 0;
#if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__(".byte 0xf3, 0x0f, 0xae, 0xf0" : : "a"(&word) : "memory");
    __asm__ __volatile__(".byte 0xf2, 0x0f, 0xae, 0xf1"
                         :
                         : "a"(static_cast<std::uint32_t>(deadline)), "d"(static_cast<std::uint32_t>(deadline >> 32)),
                           "c"(1u)
                         : "cc", "memory");
#else
    _umonitor(&word);
    (void)_umwait(1u, deadline);
#endif
    return static_cast<int>(word);
}
