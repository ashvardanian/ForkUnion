/**
 *  @file probes/ppc64_yield.cpp
 *  @author Ash Vardanian
 *  @date November 12, 2026
 *  @brief ForkUnion probe: the Power thread-priority hints, <tt>or 1,1,1</tt> then <tt>or 2,2,2</tt>.
 */
#if !(defined(__powerpc64__) || defined(__ppc64__) || defined(_ARCH_PPC64))
#error "64-bit Power only"
#endif
int main() {
    __asm__ __volatile__("or 1,1,1\n\tor 2,2,2");
    return 0;
}
