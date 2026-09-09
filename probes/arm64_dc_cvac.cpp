/**
 *  @brief ForkUnion probe: the `DC CVAC` clean to the coherency point.
 *  @author Ash Vardanian
 *  @file probes/arm64_dc_cvac.cpp
 *  @date September 7, 2026
 */
#if !(defined(__aarch64__) || defined(_M_ARM64))
#error "AArch64 only"
#endif
int main() {
    int word = 0;
    __asm__ __volatile__("dc cvac, %0" ::"r"(&word) : "memory");
    return word;
}
