/**
 *  @file capabilities.hpp
 *  @brief CPU/RAM capability probing and the hardware-friendly busy-wait yields.
 *  @note Included by `<forkunion.hpp>`; not meant to be included on its own.
 */
#pragma once
#include "types.hpp"

namespace ashvardanian {
namespace forkunion {

#if FU_WITH_ASM_YIELDS_ // We need inline assembly support

#if FU_DETECT_ARCH_ARM64_

struct arm64_yield_t {
    inline void operator()() const noexcept { __asm__ __volatile__("yield"); }
};

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a")
#endif

/**
 *  @brief On AArch64 uses the `WFET` instruction to "Wait For Event (Timed)".
 *
 *  Places the core into light sleep mode, waiting for an event to wake it up,
 *  or the timeout to expire.
 *
 *  @note The WFET instruction is @b manually encoded using `.inst` to avoid
 *  compiler-specific target attributes that may not be recognized on all
 *  ARM64 platforms - using @b `target("arch=armv8-a+wfxt")` breaks compilation
 *  on Apple Clang. Compiler feature detection like `__ARM_FEATURE_NEON`, but
 *  for `WFxT` is not available at the time of writing.
 *
 *  Runtime detection via `capability_arm64_wfet_k` ensures this is only used when `FEAT_WFxT` is actually available.
 */
struct arm64_wfet_t {
    inline void operator()() const noexcept {
        std::uint64_t cntfrq_el0, cntvct_el0;
        // Read the timer frequency (ticks per second)
        __asm__ __volatile__("mrs %0, CNTFRQ_EL0" : "=r"(cntfrq_el0));
        // Convert one micro-second to timer ticks
        std::uint64_t const ticks_per_us = cntfrq_el0 / 1'000'000;
        // Fetch current counter value and build the deadline
        __asm__ __volatile__("mrs %0, CNTVCT_EL0" : "=r"(cntvct_el0));
        std::uint64_t const deadline = cntvct_el0 + ticks_per_us;
        // We want to enter a timed wait as `WFET <Xt>`, but Clang 15 doesn't recognize it yet.
        //
        //      __asm__ __volatile__("wfet %x0\n\t" : : "r"(deadline) : "memory", "cc");
        //
        // So instead, we can encode the instruction manually as `D50310XX`,
        // where XX encodes the lower bits of Xt - the deadline register number.
        __asm__ __volatile__(    //
            "mov x0, %0\n"       // move the deadline to x0
            ".inst 0xD5031000\n" // wfet x0
            :
            : "r"(deadline)
            : "x0", "memory", "cc");
    }
};

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // FU_DETECT_ARCH_ARM64_

#if FU_DETECT_ARCH_X86_64_

struct x86_pause_t {
    inline void operator()() const noexcept { __asm__ __volatile__("pause"); }
};

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("waitpkg"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("waitpkg")
#endif

/**
 *  @brief On x86 uses the `TPAUSE` instruction to yield for 1 microsecond if `WAITPKG` is supported.
 *
 *  There are several newer ways to yield on x86, but they may require different privileges:
 *  - `MONITOR` and `MWAIT` in SSE - used for power management, require RING 0 privilege.
 *  - `UMONITOR` and `UMWAIT` in `WAITPKG` - are the user-space variants.
 *  - `MWAITX` in `MONITORX` ISA on AMD - used for power management, requires RING 0 privilege.
 *  - `TPAUSE` in `WAITPKG` - time-based pause instruction, available in RING 3.
 */
struct x86_tpause_t {
    inline void operator()() const noexcept {
        constexpr std::uint64_t cycles_per_us = 3ull * 1000ull; // ? Around 3K cycles per microsecond
        constexpr std::uint32_t sleep_level = 0;                // ? The deepest "C0.2" state

        // Now we need to fetch the current time in cycles, add a delay, and sleep until that time is reached.
        // Using intrinsics from `<x86intrin.h>` it may look like:
        //
        //      std::uint64_t const deadline = __rdtsc() + cycles_per_us;
        //      _tpause(sleep_level, deadline);
        //
        // To avoid includes, using inline Assembly:
        std::uint32_t rdtsc_lo, rdtsc_hi;
        __asm__ __volatile__("rdtsc" : "=a"(rdtsc_lo), "=d"(rdtsc_hi));
        std::uint64_t const deadline = ((static_cast<std::uint64_t>(rdtsc_hi) << 32) | rdtsc_lo) + cycles_per_us;
        std::uint32_t const deadline_lo = static_cast<std::uint32_t>(deadline);
        std::uint32_t const deadline_hi = static_cast<std::uint32_t>(deadline >> 32);
        __asm__ __volatile__(               //
            "mov    %[lo], %%eax\n\t"       // deadline_lo
            "mov    %[hi], %%edx\n\t"       // deadline_hi
            ".byte  0x66, 0x0F, 0xAE, 0xF3" // TPAUSE EBX
            :
            : [lo] "r"(deadline_lo), [hi] "r"(deadline_hi), "b"(sleep_level)
            : "eax", "edx", "memory", "cc");
    }
};

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // FU_DETECT_ARCH_X86_64_

#if FU_DETECT_ARCH_RISC5_

struct risc5_pause_t {
    inline void operator()() const noexcept { __asm__ __volatile__("pause"); }
};

#endif // FU_DETECT_ARCH_RISC5_

#endif

/**
 *  @brief Represents the CPU capabilities for hardware-friendly yielding.
 *  @note Combine with @b `ram_capabilities()` to get the full set of library capabilities.
 */
inline capabilities_t cpu_capabilities() noexcept {
    capabilities_t caps = capabilities_unknown_k;

#if FU_DETECT_ARCH_X86_64_

    // Check for basic PAUSE instruction support (always available on x86-64)
    caps = static_cast<capabilities_t>(caps | capability_x86_pause_k);

#if FU_WITH_ASM_YIELDS_ // We use inline assembly - unavailable in MSVC
    // CPUID to check for WAITPKG support (TPAUSE instruction)
    std::uint32_t eax, __attribute__((unused)) ebx, ecx, __attribute__((unused)) edx;

    // CPUID leaf 7, sub-leaf 0 for structured extended feature flags
    eax = 7, ecx = 0;
    __asm__ __volatile__("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(eax), "c"(ecx) : "memory");

    // WAITPKG is bit 5 in ECX
    if (ecx & (1u << 5)) caps = static_cast<capabilities_t>(caps | capability_x86_tpause_k);
    fu_unused_(ebx);
    fu_unused_(edx);
#endif

#elif FU_DETECT_ARCH_ARM64_

    // Basic YIELD is always available on AArch64
    caps = static_cast<capabilities_t>(caps | capability_arm64_yield_k);

    // Use sysctl to check for WFET support on Apple platforms
#if defined(__APPLE__)
    int wfet_support = 0;
    size_t size = sizeof(wfet_support);
    if (sysctlbyname("hw.optional.arm.FEAT_WFxT", &wfet_support, &size, NULL, 0) == 0 && wfet_support)
        caps = static_cast<capabilities_t>(caps | capability_arm64_wfet_k);
#elif FU_WITH_ASM_YIELDS_ // We use inline assembly - unavailable in MSVC
    // On non-Apple ARM systems, try to read the system register
    // Note: This may fail on some systems where userspace access is restricted
    std::uint64_t id_aa64isar2_el0 = 0;
    __asm__ __volatile__("mrs %0, ID_AA64ISAR2_EL0" : "=r"(id_aa64isar2_el0) : : "memory");
    // WFET is bits [3:0], value 2 indicates WFET support
    std::uint64_t const wfet_field = id_aa64isar2_el0 & 0xF;
    if (wfet_field >= 2) caps = static_cast<capabilities_t>(caps | capability_arm64_wfet_k);
#endif

#elif FU_DETECT_ARCH_RISC5_

    // Basic PAUSE is available on RISC-V with Zihintpause extension
    // For now, we assume it's available if we're on RISC-V
    caps = static_cast<capabilities_t>(caps | capability_risc5_pause_k);

#endif

    return caps;
}

/**
 *  @brief Represents the memory-system capabilities, retrieved from the Linux Sysfs.
 *  @note Combine with @b `cpu_capabilities()` to get the full set of library capabilities.
 */
inline capabilities_t ram_capabilities() noexcept {
    capabilities_t caps = capabilities_unknown_k;

#if FU_WITH_NUMA_MEMORY
    // Check for NUMA support
    if (::numa_available() >= 0) caps = static_cast<capabilities_t>(caps | capability_numa_aware_k);

    // Check for huge pages support - simplest method is checking if the global directory exists
    {
        DIR *hugepages_dir = ::opendir("/sys/kernel/mm/hugepages");
        if (hugepages_dir) {
            caps = static_cast<capabilities_t>(caps | capability_huge_pages_k);
            ::closedir(hugepages_dir);
        }
    }

    // Check for transparent huge pages
    {
        FILE *thp_enabled = ::fopen("/sys/kernel/mm/transparent_hugepage/enabled", "r");
        if (thp_enabled) {
            char thp_status[64];
            if (::fgets(thp_status, sizeof(thp_status), thp_enabled))
                // THP is enabled if we see "[always]" or "[madvise]" in the output
                if (::strstr(thp_status, "[always]") || ::strstr(thp_status, "[madvise]"))
                    // THP is available and enabled - huge pages capability confirmed
                    caps = static_cast<capabilities_t>(caps | capability_huge_pages_transparent_k);
            ::fclose(thp_enabled);
        }
    }

#endif // FU_WITH_HUGE_PAGES

    return caps;
}

#pragma region - Compile-Time Capabilities

/**
 *  @brief Which kernel facilities this translation unit was compiled to use, one bit per `FU_WITH_*`.
 *  @sa `runtime_capabilities` for what the machine underneath turned out to offer.
 *
 *  Consult it before reaching for a domain-aware API. Without `capability_comptime_colocated_pools_k`
 *  there is no `colocated_pool` to spawn and no `linux_numa_allocator` to construct, and a caller has
 *  no other way to tell that apart from a machine that merely has one compute domain.
 */
constexpr capabilities_t comptime_capabilities() noexcept {
    // Each term is widened to `unsigned` before the `?:`, because GCC's `-Wextra` rightly objects to
    // a conditional whose arms are an enumerator and a plain `0`.
    return static_cast<capabilities_t>(                                                                   //
        (FU_WITH_THREADS ? static_cast<unsigned>(capability_comptime_threads_k) : 0u) |                   //
        (FU_WITH_TOPOLOGY ? static_cast<unsigned>(capability_comptime_topology_k) : 0u) |                 //
        (FU_WITH_TOPOLOGY_CACHES ? static_cast<unsigned>(capability_comptime_topology_caches_k) : 0u) |   //
        (FU_WITH_TOPOLOGY_METRICS ? static_cast<unsigned>(capability_comptime_topology_metrics_k) : 0u) | //
        (FU_WITH_THREAD_PINNING ? static_cast<unsigned>(capability_comptime_thread_pinning_k) : 0u) |     //
        (FU_WITH_THREAD_QOS ? static_cast<unsigned>(capability_comptime_thread_qos_k) : 0u) |             //
        (FU_WITH_THREAD_SCHED_CLASS ? static_cast<unsigned>(capability_comptime_thread_sched_class_k) : 0u) |
        (FU_WITH_NUMA_MEMORY ? static_cast<unsigned>(capability_comptime_numa_memory_k) : 0u) | //
        (FU_WITH_HUGE_PAGES ? static_cast<unsigned>(capability_comptime_huge_pages_k) : 0u) |   //
        (FU_WITH_COLOCATED_POOLS ? static_cast<unsigned>(capability_comptime_colocated_pools_k) : 0u));
}

/**
 *  @brief Which features this machine turned out to offer, probing the CPU and the memory system.
 *  @sa `comptime_capabilities` for what this build is able to ask for in the first place.
 */
inline capabilities_t runtime_capabilities() noexcept {
    return static_cast<capabilities_t>(cpu_capabilities() | ram_capabilities());
}

#pragma endregion - Compile - Time Capabilities

} // namespace forkunion
} // namespace ashvardanian
