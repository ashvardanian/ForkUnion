/**
 *  @file probes/ppc64_pause.cpp
 *  @author Ash Vardanian
 *  @date November 12, 2026
 *  @brief ForkUnion probe: the ISA 3.1 @c pause_short, <tt>wait 2,0</tt>, as a raw word.
 */
#if !(defined(__powerpc64__) || defined(__ppc64__) || defined(_ARCH_PPC64))
#error "64-bit Power only"
#endif
int main() {
    __asm__ __volatile__(".4byte 0x7c40003c");
    return 0;
}
