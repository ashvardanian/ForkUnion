/**
 *  @brief ForkUnion probe: the raw `WFET` encoding, `D5031000`.
 *  @author Ash Vardanian
 *  @file probes/arm64_wfet.cpp
 *  @date September 7, 2026
 */
#if !(defined(__aarch64__) || defined(_M_ARM64))
#error "AArch64 only"
#endif
#include <cstdint> // `std::uint64_t`
int main() {
    std::uint64_t const deadline = 0;
    __asm__ __volatile__(    //
        "mov x0, %0\n"       // move the deadline to x0
        ".inst 0xD5031000\n" // wfet x0
        :
        : "r"(deadline)
        : "x0", "memory", "cc");
    return 0;
}
