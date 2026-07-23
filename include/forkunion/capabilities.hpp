/**
 *  @file capabilities.hpp
 *  @brief CPU/RAM capability probing and the hardware-friendly busy-wait yields.
 *  @note Included by `<forkunion.hpp>`; not meant to be included on its own.
 */
#pragma once
#include "types.hpp"

#if FU_DETECT_ARCH_X86_64_
#include <chrono> // `std::chrono::steady_clock` for the one-shot TSC calibration
#endif

/*  Where inline assembly is unavailable - MSVC - the same instructions are reached through intrinsics. */
#if !FU_DETECT_INLINE_ASM_SUPPORT_ && FU_DETECT_ARCH_X86_64_
#include <intrin.h>    // `__rdtsc`, `__cpuidex`
#include <immintrin.h> // `_mm_pause`, `_umonitor`, `_umwait`
#elif !FU_DETECT_INLINE_ASM_SUPPORT_ && FU_DETECT_ARCH_ARM64_
#include <intrin.h> // `__yield`
#endif

/*  Runtime `Zawrs` detection on Linux RISC-V goes through the `riscv_hwprobe` syscall, but only when
 *  this kernel's headers actually define it. Without them we fall back to the compile-time
 *  `__riscv_zawrs` macro, and claim nothing if neither is available. */
#if FU_DETECT_ARCH_RISC5_ && FU_ON_LINUX && __has_include(<asm/hwprobe.h>) && \
    __has_include(<sys/syscall.h>) && __has_include(<unistd.h>)
#include <asm/hwprobe.h> // `riscv_hwprobe`, `RISCV_HWPROBE_KEY_IMA_EXT_0`, `RISCV_HWPROBE_EXT_ZAWRS`
#include <sys/syscall.h> // `SYS_riscv_hwprobe`
#include <unistd.h>      // `syscall`
#define FU_DETECT_RISCV_HWPROBE_ 1
#endif

namespace ashvardanian {
namespace forkunion {

/** @brief The address of a waited word - a `std::atomic` object or a bare `std::atomic_ref`-owned slot.
 *      A monitored waiter needs only the address and the observed bit pattern, so both forms route here. */
template <typename value_type_>
inline void const *watched_address(std::atomic<value_type_> const &watched) noexcept {
    return &watched;
}
template <typename value_type_>
inline void const *watched_address(value_type_ const *watched) noexcept {
    return watched;
}

#if FU_DETECT_ARCH_X86_64_

/** @brief On x86, hints a spin-wait so the core neither burns issue slots nor trips memory-order speculation. */
struct x86_pause_t {
    static constexpr capabilities_t capability_k = capability_x86_pause_k;
    /** @brief Any waited word - a `std::atomic` object or a bare address - the hint watches nothing. */
    template <typename watched_type_, typename value_type_, typename thread_index_type_,
              typename bound_type_ = wait_capped_t>
    inline void operator()(watched_type_ const &, value_type_, thread_index_type_, bound_type_ = {}) const noexcept {
#if FU_DETECT_INLINE_ASM_SUPPORT_
        __asm__ __volatile__("pause");
#else
        _mm_pause();
#endif
    }
};

/** @brief All four registers of one `CPUID` invocation. @sa `cpuid`, the one home of both toolchain idioms. */
struct cpuid_registers_t {
    std::uint32_t eax, ebx, ecx, edx;
};

/**
 *  @brief Issues one `CPUID` for @p leaf and @p subleaf, via inline assembly or MSVC's `__cpuidex`.
 *  @note Reports for the @b executing core; on hybrid parts, pin before asking per-core questions.
 */
inline cpuid_registers_t cpuid(std::uint32_t const leaf, std::uint32_t const subleaf) noexcept {
    cpuid_registers_t r;
#if FU_DETECT_INLINE_ASM_SUPPORT_
    __asm__ __volatile__("cpuid" : "=a"(r.eax), "=b"(r.ebx), "=c"(r.ecx), "=d"(r.edx) : "a"(leaf), "c"(subleaf));
#else
    int regs[4];
    __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
    r = {static_cast<std::uint32_t>(regs[0]), static_cast<std::uint32_t>(regs[1]), static_cast<std::uint32_t>(regs[2]),
         static_cast<std::uint32_t>(regs[3])};
#endif
    return r;
}

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("waitpkg"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("waitpkg")
#endif

/**
 *  @brief The TSC rate in cycles per microsecond, detected once from `CPUID.15h` or by calibration.
 *
 *  `x86_tpause_t` turns a microsecond into a TSC deadline, and the invariant TSC does @b not tick at
 *  the core's frequency - it tracks the nominal base clock, which differs from part to part. Leaf
 *  `0x15` reports it exactly as `crystal_hz * numerator / denominator`, but many CPUs leave the
 *  crystal field zero, so we then time a short `RDTSC` span against the steady clock.
 */
inline std::uint64_t x86_detect_tsc_cycles_per_micro() noexcept {
    // Ask leaf 0x15 for the TSC-to-crystal ratio: EAX holds the denominator, EBX the numerator,
    // and ECX the crystal frequency in Hz - the exact fields of libc's
    // `__get_cpuid(0x15, &denominator, &numerator, &crystal_hz, &unused)`.
    cpuid_registers_t const leaf15 = cpuid(0x15u, 0);
    std::uint32_t const denominator = leaf15.eax, numerator = leaf15.ebx, crystal_hz = leaf15.ecx;
    if (denominator != 0 && numerator != 0 && crystal_hz != 0) {
        std::uint64_t const tsc_hz = static_cast<std::uint64_t>(crystal_hz) * numerator / denominator;
        std::uint64_t const cycles_per_us = tsc_hz / 1'000'000ull;
        if (cycles_per_us != 0) return cycles_per_us;
    }

    // The leaf was blank, as on many parts: measure how many TSC cycles pass over a fixed wall span.
    auto const started_at = std::chrono::steady_clock::now();
#if FU_DETECT_INLINE_ASM_SUPPORT_
    std::uint32_t start_lo, start_hi, end_lo, end_hi;
    __asm__ __volatile__("rdtsc" : "=a"(start_lo), "=d"(start_hi));
    while (std::chrono::steady_clock::now() - started_at < std::chrono::milliseconds(2)) {}
    __asm__ __volatile__("rdtsc" : "=a"(end_lo), "=d"(end_hi));
    std::uint64_t const start_cycles = (static_cast<std::uint64_t>(start_hi) << 32) | start_lo;
    std::uint64_t const end_cycles = (static_cast<std::uint64_t>(end_hi) << 32) | end_lo;
#else
    std::uint64_t const start_cycles = __rdtsc();
    while (std::chrono::steady_clock::now() - started_at < std::chrono::milliseconds(2)) {}
    std::uint64_t const end_cycles = __rdtsc();
#endif
    std::uint64_t const elapsed_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started_at).count());

    // Guess 3 GHz only if the calibration itself degenerated - it never should.
    if (elapsed_ns == 0) return 3'000ull;
    std::uint64_t const cycles_per_us = (end_cycles - start_cycles) * 1'000ull / elapsed_ns;
    return cycles_per_us != 0 ? cycles_per_us : 3'000ull;
}

/** @brief Memoizes `x86_detect_tsc_cycles_per_micro`; the rate is fixed for the life of the process. */
inline std::uint64_t x86_tsc_cycles_per_micro() noexcept {
    static std::uint64_t const cycles_per_us = x86_detect_tsc_cycles_per_micro();
    return cycles_per_us;
}

/** @brief `UMWAIT` sleep-depth control: bit 0 = 1 selects the shallow, fast-waking C0.1 state. */
inline constexpr std::uint32_t x86_umwait_shallow_c01_k = 1;
/** @brief `UMWAIT` sleep-depth control: bit 0 = 0 selects the deeper C0.2 state - slower to wake,
 *      but ceding more of the shared core's pipeline resources to the SMT sibling meanwhile. */
inline constexpr std::uint32_t x86_umwait_deeper_c02_k = 0;

/** @brief Reads the time-stamp counter, via inline assembly or MSVC's `__rdtsc`. */
inline std::uint64_t x86_now_tsc() noexcept {
#if FU_DETECT_INLINE_ASM_SUPPORT_
    std::uint32_t rdtsc_lo, rdtsc_hi;
    __asm__ __volatile__("rdtsc" : "=a"(rdtsc_lo), "=d"(rdtsc_hi));
    return (static_cast<std::uint64_t>(rdtsc_hi) << 32) | rdtsc_lo;
#else
    return __rdtsc();
#endif
}

/** @brief Arms this core's address-range monitor on the line holding @p watched_address.
 *      Where inline assembly is available the UMONITOR opcode is hand-encoded so no header is
 *      pulled in; MSVC has no inline assembly and instead calls the `<immintrin.h>` intrinsic
 *      the encoding stands in for - `_umonitor(const_cast<void *>(watched_address))`. */
inline void x86_arm_address(void const *watched_address) noexcept {
#if FU_DETECT_INLINE_ASM_SUPPORT_
    // Hand-encoding UMONITOR r64 as `F3 0F AE /6` with the address in RAX:
    __asm__ __volatile__(".byte 0xf3, 0x0f, 0xae, 0xf0" : : "a"(watched_address) : "memory");
#else
    _umonitor(const_cast<void *>(watched_address));
#endif
}

/**
 *  @brief Arms this core's address-range monitor on @p watched and reports whether to enter the wait.
 *  @retval true if the monitor is armed and @p watched still holds @p observed - proceed to wait.
 *  @retval false if @p watched already moved - the caller must re-check.
 */
template <typename value_type_>
inline bool x86_arm_monitor(std::atomic<value_type_> const &watched, value_type_ const observed) noexcept {
    x86_arm_address(&watched);
    // A normal load does not disarm the monitor, so re-check: if the word already moved, don't wait.
    return watched.load(std::memory_order_acquire) == observed;
}

/** @brief Same, for a word owned through `std::atomic_ref` rather than a `std::atomic` object. */
template <typename value_type_>
inline bool x86_arm_monitor(value_type_ const *watched, value_type_ const observed) noexcept {
    x86_arm_address(watched);
    // Acquire-load the bare word: `std::atomic_ref` where it exists, else the compiler's own load,
    // since C++17 has no portable `atomic_ref` and this waiter already needs GCC/Clang's opcodes.
#if defined(__cpp_lib_atomic_ref)
    value_type_ const current =
        std::atomic_ref<value_type_>(*const_cast<value_type_ *>(watched)).load(std::memory_order_acquire);
#elif defined(__GNUC__) || defined(__clang__)
    value_type_ const current = __atomic_load_n(watched, __ATOMIC_ACQUIRE);
#else
    value_type_ const current = *static_cast<value_type_ const volatile *>(watched);
    std::atomic_thread_fence(std::memory_order_acquire); // ? The monitor re-check tolerates a stale read
#endif
    return current == observed;
}

/** @brief Sleeps in @p sleep_state until @p deadline as a TSC value, an interrupt, or a store to the
 *      monitored line. Inline assembly hand-encodes the opcode to avoid an include, while MSVC calls
 *      the `<immintrin.h>` intrinsic - in pseudo-code, `_umwait(sleep_state, deadline)`. */
inline void x86_umwait_until(std::uint64_t const deadline, std::uint32_t const sleep_state) noexcept {
#if FU_DETECT_INLINE_ASM_SUPPORT_
    // Hand-encoding UMWAIT r32 as `F2 0F AE /6`, with the control in ECX and the deadline in EDX:EAX:
    std::uint32_t const deadline_lo = static_cast<std::uint32_t>(deadline);
    std::uint32_t const deadline_hi = static_cast<std::uint32_t>(deadline >> 32);
    __asm__ __volatile__(".byte 0xf2, 0x0f, 0xae, 0xf1"
                         :
                         : "a"(deadline_lo), "d"(deadline_hi), "c"(sleep_state)
                         : "cc", "memory");
#else
    (void)_umwait(sleep_state, deadline);
#endif
}

/**
 *  @brief On x86 `WAITPKG`, a monitored wait built on `UMONITOR` + `UMWAIT`.
 *
 *  `UMONITOR` arms an address-range monitor on the watched line; `UMWAIT` drops the logical processor
 *  into an optimized sleep until @b any agent's store to that line, an interrupt, or the TSC deadline.
 *  The store @b pushes the waiter awake as an event, so a fork-join handoff is observed rather than
 *  polled, and on an SMT core the sibling reclaims the freed pipeline slots for the duration.
 *
 *  There are several older ways to wait on x86, but they may require different privileges:
 *  - `MONITOR` and `MWAIT` in SSE - used for power management, require RING 0 privilege.
 *  - `MWAITX` in `MONITORX` ISA on AMD - used for power management, requires RING 0 privilege.
 *  - `TPAUSE` in `WAITPKG` - a blind timed pause that watches @b no address, so a store never wakes
 *    it early. It is what this used to be; `UMWAIT` is strictly better when there is a word to watch.
 *
 *  @note `UMWAIT`'s control selects the sleep depth: bit 0 = 1 picks @b C0.1,
 *      shallow and fast-waking; bit 0 = 0 picks @b C0.2, deeper and slower. A fork-join barrier resolves in
 *      tens of nanoseconds, so we ask for C0.1. Neither lowers voltage: like ARM's `WFE`, this
 *      clock-gates and saves dynamic power only, and never releases the core to the scheduler.
 *
 *  @warning The `UMONITOR` and `UMWAIT` opcodes are hand-encoded and, unlike the AArch64 path, have
 *      not been exercised on `WAITPKG` silicon in this tree. Gated at runtime by `capability_x86_tpause_k`.
 */
struct x86_tpause_t {
    static constexpr capabilities_t capability_k = capability_x86_tpause_k;
    /** @brief Waits until a deadline ~1 micro-second ahead, for a loop that also guards another line.
     *      Accepts a `std::atomic` object or a bare `std::atomic_ref`-owned word alike. */
    template <typename watched_type_, typename value_type_, typename thread_index_type_>
    inline void operator()(watched_type_ const &watched, value_type_ const observed, thread_index_type_,
                           wait_capped_t = {}) const noexcept {
        if (!x86_arm_monitor(watched, observed)) return;
        // A deadline one microsecond of TSC cycles ahead of now, in the shallow fast-waking state:
        // a fork-join barrier resolves in tens of nanoseconds, so wake latency dominates the choice.
        x86_umwait_until(x86_now_tsc() + x86_tsc_cycles_per_micro(), x86_umwait_shallow_c01_k);
    }

    /** @brief Waits for the store with no effective cap, for a single-word loop. */
    template <typename watched_type_, typename value_type_, typename thread_index_type_>
    inline void operator()(watched_type_ const &watched, value_type_ const observed, thread_index_type_,
                           wait_uncapped_t) const noexcept {
        if (!x86_arm_monitor(watched, observed)) return;
        // A TSC deadline centuries away: the monitor-clearing store or an interrupt ends the wait first.
        x86_umwait_until(~std::uint64_t {0}, x86_umwait_shallow_c01_k);
    }
};

/**
 *  @brief Sibling of `x86_tpause_t` for saturated hosts - every logical core busy, SMT siblings
 *      competing for pipeline slots.
 *
 *  The policy differs only in what each wait-bound tag selects. Capped two-word waits - lock and
 *  capacity gates whose wakes are rare - sleep in the deeper C0.2 state with a quarter-microsecond
 *  deadline, trading ~100 ns of extra wake latency for pipeline resources the sibling hyper-thread
 *  reclaims meanwhile. Uncapped single-word waits - serialized publication convoys where every
 *  nanosecond of wake latency lands on the critical chain - keep the shallow C0.1 state.
 *
 *  Motivated by a 128-thread convoy workload where the shallow-everywhere policy of `x86_tpause_t`
 *  measured 13-16% behind plain `std::this_thread::yield` at full occupancy, while leading at the
 *  7/8-occupancy operating point - the saturated sibling is the tool for the former regime.
 */
struct x86_tpause_saturated_t {
    static constexpr capabilities_t capability_k = capability_x86_tpause_k;

    /** @brief Rare-wake wait: a quarter-microsecond deadline in the deeper C0.2 state. */
    template <typename watched_type_, typename value_type_, typename thread_index_type_>
    inline void operator()(watched_type_ const &watched, value_type_ const observed, thread_index_type_,
                           wait_capped_t = {}) const noexcept {
        if (!x86_arm_monitor(watched, observed)) return;
        x86_umwait_until(x86_now_tsc() + (x86_tsc_cycles_per_micro() >> 2), x86_umwait_deeper_c02_k);
    }

    /** @brief Critical-chain wait: uncapped, but shallow - the waking store must land instantly. */
    template <typename watched_type_, typename value_type_, typename thread_index_type_>
    inline void operator()(watched_type_ const &watched, value_type_ const observed, thread_index_type_,
                           wait_uncapped_t) const noexcept {
        if (!x86_arm_monitor(watched, observed)) return;
        x86_umwait_until(~std::uint64_t {0}, x86_umwait_shallow_c01_k);
    }
};

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // FU_DETECT_ARCH_X86_64_

#if FU_DETECT_ARCH_ARM64_

/** @brief On Arm, hints the core to release its pipeline slot to a sibling hardware thread. */
struct arm64_yield_t {
    static constexpr capabilities_t capability_k = capability_arm64_yield_k;
    /** @brief Any waited word - a `std::atomic` object or a bare address - the hint watches nothing. */
    template <typename watched_type_, typename value_type_, typename thread_index_type_,
              typename bound_type_ = wait_capped_t>
    inline void operator()(watched_type_ const &, value_type_, thread_index_type_, bound_type_ = {}) const noexcept {
#if FU_DETECT_INLINE_ASM_SUPPORT_
        __asm__ __volatile__("yield");
#else
        __yield();
#endif
    }
};

// `WFET` and the exclusive-monitor `LDAXR`/`CLREX` it rides on have no MSVC intrinsic, so the timed
// waiter is inline-assembly only; MSVC-ARM64 stays on the `arm64_yield_t` hint above.
#if FU_DETECT_INLINE_ASM_SUPPORT_

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a")
#endif

/**
 *  @brief On AArch64, a monitored wait built on the `WFET` "Wait For Event, Timed" instruction.
 *
 *  `LDAXR` arms this core's exclusive monitor on the watched word; `WFET` then places the core into
 *  light sleep until @b any core's store to that line clears the monitor as an event, or the timeout
 *  expires. The store @b pushes the waiter awake through the cache-coherence transaction, rather than
 *  the waiter polling for it - on an Apple M5 this wakes in ~42ns, faster than a bare `YIELD` spin,
 *  while the core is clock-gated. The timeout is the belt: a fork-join loop can watch two independent
 *  lines while the monitor arms only one, and the cap bounds how late a change to the @b other line
 *  is noticed.
 *
 *  @note `WFET` clock-gates the core to save @b dynamic power. It does @b not lower voltage, so it
 *  saves no @b static leakage power, and it does @b not release the core to the OS scheduler: the
 *  thread stays runnable and no sibling can be placed on it. A real retention or power-down C-state
 *  needs firmware - `PSCI CPU_SUSPEND` via `SMC` at EL3. This is a power hint for a busy core.
 *
 *  @note `WFET` is hand-encoded with `.inst` because `target("arch=armv8-a+wfxt")` breaks Apple
 *  Clang and neither compiler defines a `WFxT` feature macro. It is reached only when
 *  `capability_arm64_wfet_k` reports `FEAT_WFxT` at runtime.
 */
struct arm64_wfet_t {
    static constexpr capabilities_t capability_k = capability_arm64_wfet_k;
    /** @brief Waits with a ~1 micro-second cap, for a loop that also guards another line. Accepts a
     *      `std::atomic` object or a bare `std::atomic_ref`-owned word alike. */
    template <typename watched_type_, typename value_type_, typename thread_index_type_>
    inline void operator()(watched_type_ const &watched, value_type_ const observed, thread_index_type_,
                           wait_capped_t = {}) const noexcept {
        if (!arm_monitor_(watched_address(watched), observed)) return;
        wfet_one_micro_();
    }

    /** @brief Waits with no cap, for a single-word loop where the armed line is the only wake source. */
    template <typename watched_type_, typename value_type_, typename thread_index_type_>
    inline void operator()(watched_type_ const &watched, value_type_ const observed, thread_index_type_,
                           wait_uncapped_t) const noexcept {
        if (!arm_monitor_(watched_address(watched), observed)) return;
        // `WFE` returns on the monitor-clearing store; Apple's event stream is a periodic safety net.
        __asm__ __volatile__("wfe" ::: "memory");
    }

  private:
    /** @brief Enters a timed wait with a deadline ~1 micro-second ahead of the generic timer. */
    static inline void wfet_one_micro_() noexcept {
        std::uint64_t cntfrq_el0, cntvct_el0;
        // Read the timer frequency (ticks per second)
        __asm__ __volatile__("mrs %0, CNTFRQ_EL0" : "=r"(cntfrq_el0));
        // Convert one micro-second to timer ticks
        std::uint64_t const ticks_per_us = cntfrq_el0 / 1'000'000;
        // Fetch current counter value and build the deadline
        __asm__ __volatile__("mrs %0, CNTVCT_EL0" : "=r"(cntvct_el0));
        std::uint64_t const deadline = cntvct_el0 + ticks_per_us;
        // We want to enter a timed wait as `WFET <Xt>`, but current compilers reject the mnemonic:
        //
        //      __asm__ __volatile__("wfet %x0\n\t" : : "r"(deadline) : "memory", "cc");
        //
        // So instead, we encode the instruction manually as `D50310XX`, where XX encodes the lower
        // bits of Xt - the deadline register number.
        __asm__ __volatile__(    //
            "mov x0, %0\n"       // move the deadline to x0
            ".inst 0xD5031000\n" // wfet x0
            :
            : "r"(deadline)
            : "x0", "memory", "cc");
    }

    /**
     *  @brief Arms this core's exclusive monitor on the word at @p watched_address and reports
     *      whether to enter the wait - it only ever needed the address and the bit pattern, so
     *      both `std::atomic` objects and in-place `std::atomic_ref`-owned words route here.
     *  @retval true if the monitor is armed and the word still holds @p observed - proceed to wait.
     *  @retval false if the word already moved - the monitor is dropped and the caller must re-check.
     */
    template <typename value_type_>
    static inline bool arm_monitor_(void const *watched_address, value_type_ const observed) noexcept {
        static_assert(sizeof(value_type_) <= 8, "The exclusive monitor watches at most a 64-bit word");

        // Compare bit patterns, so an enum or any trivially-copyable word works unchanged.
        std::uint64_t observed_bits = 0;
        copy_bytes(&observed, reinterpret_cast<value_type_ *>(&observed_bits));

        // Arm the monitor with a single-copy-atomic exclusive load of the matching width. A store from
        // any core clears the monitor, and that is what releases the wait below without a timeout. The
        // sub-word and word widths share one 32-bit destination, so only the instruction suffix differs.
        std::uint64_t current_bits;
        if constexpr (sizeof(value_type_) <= 4) {
            std::uint32_t narrow;
            if constexpr (sizeof(value_type_) == 1)
                __asm__ __volatile__("ldaxrb %w0, [%1]" : "=r"(narrow) : "r"(watched_address) : "memory");
            else if constexpr (sizeof(value_type_) == 2)
                __asm__ __volatile__("ldaxrh %w0, [%1]" : "=r"(narrow) : "r"(watched_address) : "memory");
            else
                __asm__ __volatile__("ldaxr %w0, [%1]" : "=r"(narrow) : "r"(watched_address) : "memory");
            current_bits = narrow;
        }
        else
            __asm__ __volatile__("ldaxr %0, [%1]" : "=r"(current_bits) : "r"(watched_address) : "memory");

        // The word moved between the caller's check and our load: drop the monitor and re-check.
        if (current_bits != observed_bits) {
            __asm__ __volatile__("clrex" ::: "memory");
            return false;
        }
        return true;
    }
};

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // FU_DETECT_INLINE_ASM_SUPPORT_

#endif // FU_DETECT_ARCH_ARM64_

#if FU_DETECT_ARCH_RISC5_

/** @brief On RISC-V, the `Zihintpause` spin-wait hint. */
struct risc5_pause_t {
    static constexpr capabilities_t capability_k = capability_risc5_pause_k;
    /** @brief Any waited word - a `std::atomic` object or a bare address - the hint watches nothing. */
    template <typename watched_type_, typename value_type_, typename thread_index_type_,
              typename bound_type_ = wait_capped_t>
    inline void operator()(watched_type_ const &, value_type_, thread_index_type_, bound_type_ = {}) const noexcept {
        // Zihintpause `PAUSE` is `FENCE W, 0`; the mnemonic needs `-march=...+zihintpause` to assemble,
        // so the fixed encoding is emitted directly - it decodes as a no-op fence on cores without it.
        __asm__ __volatile__(".4byte 0x0100000f");
    }
};

/**
 *  @brief On RISC-V `Zawrs`, a monitored wait built on `LR` + `WRS.STO`.
 *
 *  `LR.W` / `LR.D` arms a reservation on the watched word - the same reservation an `LR`/`SC` pair
 *  would use - and reads its current value; `WRS.STO` "Wait-for-Reservation-Set, Short TimeOut" then
 *  parks the hart until @b any other hart's store breaks the reservation as an event, an interrupt
 *  arrives, or an implementation-bounded timeout elapses. The store @b pushes the waiter awake,
 *  exactly like AArch64's `LDAXR` + `WFET`, and the bounded timeout is the same belt for a loop that
 *  guards two words while the reservation covers one.
 *
 *  RISC-V has `LR` only at word and double-word width, so a 1- or 2-byte watch falls back to the
 *  `Zihintpause` spin - which the pool never needs, as its loops watch the 32-bit mood and the
 *  register-width epoch.
 *
 *  @note Like ARM's `WFE`, this clock-gates and saves dynamic power only; it never releases the hart
 *      to a scheduler. The sibling `WRS.NTO` "No TimeOut" waits unbounded, for a single-word loop.
 *
 *  @note `WRS.STO` is @b manually encoded as `.4byte 0x01d00073` so the surrounding build needs no
 *      `-march=...+zawrs` assembler support; `LR` is plain `A`-extension, present on any core that
 *      would carry `Zawrs`.
 *
 *  @warning Hand-encoded and not exercised on `Zawrs` silicon in this tree; it needs a runtime
 *      `riscv_hwprobe(RISCV_HWPROBE_KEY_IMA_EXT_0, ..._ZAWRS)` probe (not yet wired) before it
 *      may be selected.
 */
struct risc5_wrs_t {
    static constexpr capabilities_t capability_k = capability_risc5_wrs_k;
    /** @brief Waits with the implementation-bounded short timeout, for a loop that also guards another
     *      line. Accepts a `std::atomic` object or a bare `std::atomic_ref`-owned word alike. */
    template <typename watched_type_, typename value_type_, typename thread_index_type_>
    inline void operator()(watched_type_ const &watched, value_type_ const observed, thread_index_type_,
                           wait_capped_t = {}) const noexcept {
        if (!arm_reservation_(watched_address(watched), observed)) return;
        // WRS.STO: wait for the reservation set, short bounded timeout.
        __asm__ __volatile__(".4byte 0x01d00073" ::: "memory");
    }

    /** @brief Waits unbounded, for a single-word loop where the reservation is the only wake source. */
    template <typename watched_type_, typename value_type_, typename thread_index_type_>
    inline void operator()(watched_type_ const &watched, value_type_ const observed, thread_index_type_,
                           wait_uncapped_t) const noexcept {
        if (!arm_reservation_(watched_address(watched), observed)) return;
        // WRS.NTO: wait for the reservation set, no timeout.
        __asm__ __volatile__(".4byte 0x00d00073" ::: "memory");
    }

  private:
    /**
     *  @brief Arms a reservation on the word at @p watched_address and reports whether to enter the
     *      wait - it only ever needed the address and the bit pattern, so both `std::atomic` objects
     *      and in-place `std::atomic_ref`-owned words route here.
     *  @retval true if the reservation is set and the word still holds @p observed - proceed to `WRS`.
     *  @retval false if the word moved, or the width has no `LR` (a `pause` spin was emitted instead).
     */
    template <typename value_type_>
    static inline bool arm_reservation_(void const *watched_address, value_type_ const observed) noexcept {

        // Compare bit patterns, so an enum or any trivially-copyable word works unchanged.
        std::uint64_t observed_bits = 0;
        copy_bytes(&observed, reinterpret_cast<value_type_ *>(&observed_bits));

        if constexpr (sizeof(value_type_) == 8) {
            // Arm the reservation and read the double-word.
            std::uint64_t current_bits;
            __asm__ __volatile__("lr.d %0, (%1)" : "=r"(current_bits) : "r"(watched_address) : "memory");
            // The word moved between the caller's check and our load: return and let the caller re-check.
            return current_bits == observed_bits;
        }
        else if constexpr (sizeof(value_type_) == 4) {
            // LR.W sign-extends into the register, so mask to 32 bits before comparing.
            std::uint64_t current_bits;
            __asm__ __volatile__("lr.w %0, (%1)" : "=r"(current_bits) : "r"(watched_address) : "memory");
            return (current_bits & 0xffffffffull) == (observed_bits & 0xffffffffull);
        }
        else {
            // No sub-word LR exists, so fall back to the pause hint and skip the `WRS`. Same fixed
            // Zihintpause encoding as `risc5_pause_t` - the mnemonic needs `+zihintpause` to assemble.
            __asm__ __volatile__(".4byte 0x0100000f");
            return false;
        }
    }
};

#endif // FU_DETECT_ARCH_RISC5_

/**
 *  @brief The fastest waiter this build may use with @b no runtime feature probe.
 *
 *  Upgrades to the monitored waiter - `x86_tpause_t`, `risc5_wrs_t` - exactly when the compiler
 *  @b guarantees the feature through a predefined macro, so the choice needs no runtime check: if
 *  `__WAITPKG__` or `__riscv_zawrs` is defined, the build targets a CPU that has the instruction, and
 *  running it can never be illegal. Otherwise it stays on the plain architectural hint every CPU runs.
 *
 *  AArch64 has no such upgrade: neither GCC nor Clang defines a `WFxT` feature macro - not even under
 *  `-march=armv8.7-a+wfxt` - so there is nothing to key on, and it stays `arm64_yield_t`. `FEAT_WFxT`
 *  is a runtime fact there, detected via `sysctl` on Apple or `HWCAP2_WFXT` on Linux, so a caller
 *  reaches `arm64_wfet_t` through the C ABI's runtime capability dispatch rather than at compile time.
 */
#if FU_DETECT_ARCH_X86_64_
#if defined(__WAITPKG__)
using preferred_yield_t = x86_tpause_t;
#else
using preferred_yield_t = x86_pause_t;
#endif
#elif FU_DETECT_ARCH_ARM64_
using preferred_yield_t = arm64_yield_t;
#elif FU_DETECT_INLINE_ASM_SUPPORT_ && FU_DETECT_ARCH_RISC5_
#if defined(__riscv_zawrs)
using preferred_yield_t = risc5_wrs_t;
#else
using preferred_yield_t = risc5_pause_t;
#endif
#else
using preferred_yield_t = standard_yield_t;
#endif

#if FU_DETECT_ARCH_X86_64_ && (FU_DETECT_INLINE_ASM_SUPPORT_ || FU_DETECT_HINT_INTRINSICS_)
/**
 *  @brief x86 cache hints: `CLDEMOTE` toward the LLC, `PREFETCHW` for write-intent promotion.
 *  @note Both live in hint or reserved-NOP space, so neither can fault on any x86-64 part; whether
 *      `CLDEMOTE` actually bites is reported by `capability_x86_cldemote_k` - detected, never
 *      dispatched on. Hand-assembled so stock toolchains need no `-mcldemote` / `-mprfchw`;
 *      MSVC encodes the same bytes through `_mm_cldemote` / `_m_prefetchw`, no `/arch` needed.
 */
struct x86_cache_hints_t {
    static constexpr capabilities_t capability_k = capability_x86_cldemote_k;
    inline void operator()(void const *address, demote_line_t) const noexcept {
#if FU_DETECT_INLINE_ASM_SUPPORT_
        __asm__ __volatile__(".byte 0x0f, 0x1c, 0x00" ::"a"(address) : "memory"); // ? `cldemote (%rax)`
#else
        _mm_cldemote(address); // ? The same `0F 1C /0` hint; `<immintrin.h>`, VS 2019 16.2+
#endif
    }
    inline void operator()(void const *address, promote_line_t) const noexcept {
#if FU_DETECT_INLINE_ASM_SUPPORT_
        __asm__ __volatile__(".byte 0x0f, 0x0d, 0x08" ::"a"(address) : "memory"); // ? `prefetchw (%rax)`
#else
        _m_prefetchw(address); // ? `<intrin.h>`; PREFETCHW is a Windows 8.1 x64 install requirement
#endif
    }
};
#endif // FU_DETECT_ARCH_X86_64_

#if FU_DETECT_ARCH_ARM64_ && FU_DETECT_INLINE_ASM_SUPPORT_
/**
 *  @brief AArch64 cache hints: `DC CVAC` cleans to the coherency point, `PRFM PSTL1KEEP` promotes.
 *  @note There is no demote on Arm - the clean is the nearest thing: the next claimer's snoop finds
 *      a clean line instead of forcing a dirty intervention, at the price of a memory write. The
 *      clean is EL0-legal only where the kernel sets `SCTLR_EL1.UCI`; Linux does, and the
 *      `FU_WITH_DEMOTE_CACHE_LINES` gate requires `FU_ON_LINUX` on this architecture. The
 *      persistence-targeted `DC CVAP`/`CVADP` are deliberately absent: UNDEFINED without
 *      `FEAT_DPB`/`FEAT_DPB2`, and they buy a NUMA hand-off nothing.
 */
struct arm64_cache_hints_t {
    static constexpr capabilities_t capability_k = capability_arm64_dc_cvac_k;
    inline void operator()(void const *address, demote_line_t) const noexcept {
        __asm__ __volatile__("dc cvac, %0" ::"r"(address) : "memory");
    }
    inline void operator()(void const *address, promote_line_t) const noexcept {
        __asm__ __volatile__("prfm pstl1keep, [%0]" ::"r"(address) : "memory");
    }
};
#endif // FU_DETECT_ARCH_ARM64_

#if FU_DETECT_ARCH_ARM64_ && (FU_DETECT_INLINE_ASM_SUPPORT_ || FU_DETECT_HINT_INTRINSICS_)
/**
 *  @brief AArch64 promotion only, for kernels that keep `SCTLR_EL1.UCI` clear - Windows and the
 *      BSDs do, so an EL0 `DC CVAC` traps there and the demote stays a no-op.
 *  @note Mirrors `risc5_cache_hints_t`'s shape: the promote is a `PRFM` hint that cannot fault
 *      anywhere. MSVC reaches it through `__prefetch2(address, 0x10)`, whose prfop immediate
 *      `0b10000` spells PST-L1-KEEP - the same encoding the asm arm emits.
 */
struct arm64_prefetch_cache_hints_t {
    static constexpr capabilities_t capability_k = capabilities_unknown_k;
    inline void operator()(void const *, demote_line_t) const noexcept {}
    inline void operator()(void const *address, promote_line_t) const noexcept {
#if FU_DETECT_INLINE_ASM_SUPPORT_
        __asm__ __volatile__("prfm pstl1keep, [%0]" ::"r"(address) : "memory");
#else
        __prefetch2(address, 0x10); // ? `prfm pstl1keep, [x0]`; `<intrin.h>`, VS 2019 16.1+
#endif
    }
};
#endif // FU_DETECT_ARCH_ARM64_

#if FU_DETECT_ARCH_RISC5_ && FU_DETECT_INLINE_ASM_SUPPORT_
/**
 *  @brief RISC-V promotion only: `prefetch.w` is an `ORI x0, ...` hint that cannot fault, with or
 *      without Zicbop silicon.
 *  @note The `cbo.clean` demote is deliberately a no-op here: it raises illegal-instruction unless
 *      the kernel set `senvcfg.CBCFE`, which only `hwprobe` can attest at runtime - so it belongs
 *      to a runtime-dispatch tier behind `capability_risc5_zicbom_k`, never a compile-time policy.
 */
struct risc5_cache_hints_t {
    static constexpr capabilities_t capability_k = capabilities_unknown_k;
    inline void operator()(void const *, demote_line_t) const noexcept {}
    inline void operator()(void const *address, promote_line_t) const noexcept {
        register void const *address_register __asm__("a0") = address;
        __asm__ __volatile__(".4byte 0x00356013" ::"r"(address_register) : "memory"); // ? `prefetch.w 0(a0)`
    }
};

/**
 *  @brief RISC-V cache hints where `hwprobe` attested Zicbom: `cbo.clean` writes the dirty block
 *      back toward another cache or memory, and `prefetch.w` promotes with write intent.
 *  @note Never selected at compile time - `cbo.clean` raises illegal-instruction unless the kernel
 *      set `senvcfg.CBCFE`, which only the `capability_risc5_zicbom_k` runtime bit can attest -
 *      so this functor is reachable exclusively through the C ABI's runtime cascade.
 */
struct risc5_cbo_cache_hints_t {
    static constexpr capabilities_t capability_k = capability_risc5_zicbom_k;
    inline void operator()(void const *address, demote_line_t) const noexcept {
        register void const *address_register __asm__("a0") = address;
        __asm__ __volatile__(".4byte 0x0015200f" ::"r"(address_register) : "memory"); // ? `cbo.clean (a0)`
    }
    inline void operator()(void const *address, promote_line_t) const noexcept {
        register void const *address_register __asm__("a0") = address;
        __asm__ __volatile__(".4byte 0x00356013" ::"r"(address_register) : "memory"); // ? `prefetch.w 0(a0)`
    }
};
#endif // FU_DETECT_ARCH_RISC5_

/*  One deterministic cache-hints policy per build, mirroring `preferred_yield_t`: the gate is the
 *  Layer-2 tri-state, never runtime silicon - everything a selected functor emits is trap-free
 *  wherever its gate holds, so no dispatch and no reporting bit ever guards an emission.  */
#if FU_WITH_DEMOTE_CACHE_LINES && FU_DETECT_ARCH_X86_64_
using preferred_cache_hints_t = x86_cache_hints_t;
#elif FU_WITH_DEMOTE_CACHE_LINES && FU_DETECT_ARCH_ARM64_
using preferred_cache_hints_t = arm64_cache_hints_t;
#elif FU_WITH_PROMOTE_CACHE_LINES && FU_DETECT_ARCH_ARM64_
using preferred_cache_hints_t = arm64_prefetch_cache_hints_t; // ? Windows/BSD: the clean traps, the hint stays
#elif FU_WITH_PROMOTE_CACHE_LINES && FU_DETECT_ARCH_RISC5_
using preferred_cache_hints_t = risc5_cache_hints_t;
#else
using preferred_cache_hints_t = standard_cache_hints_t;
#endif

/**
 *  @brief Represents the CPU capabilities for hardware-friendly yielding.
 *  @sa `ram_capabilities` to get the full set of library capabilities.
 */
inline capabilities_t cpu_capabilities() noexcept {
    capabilities_t caps = capabilities_unknown_k;

#if FU_DETECT_ARCH_X86_64_

    // Check for basic PAUSE instruction support (always available on x86-64)
    caps |= capability_x86_pause_k;

    // CPUID leaf 7, sub-leaf 0, ECX: WAITPKG (backing UMWAIT/TPAUSE) is bit 5; CLDEMOTE is bit 25.
    // The CLDEMOTE bit reports whether the hint bites - Sapphire-Rapids-class parts - it never
    // gates emission, which the compile-time `preferred_cache_hints_t` decides.
    cpuid_registers_t const leaf7 = cpuid(7u, 0);
    if (leaf7.ecx & (1u << 5)) caps |= capability_x86_tpause_k;
    if (leaf7.ecx & (1u << 25)) caps |= capability_x86_cldemote_k;

#elif FU_DETECT_ARCH_ARM64_

    // Basic YIELD is always available on AArch64
    caps |= capability_arm64_yield_k;

    // Use sysctl to check for WFET support on Apple platforms
#if FU_ON_APPLE
    int wfet_support = 0;
    size_t size = sizeof(wfet_support);
    if (sysctlbyname("hw.optional.arm.FEAT_WFxT", &wfet_support, &size, NULL, 0) == 0 && wfet_support)
        caps |= capability_arm64_wfet_k;
#elif FU_DETECT_INLINE_ASM_SUPPORT_ // We use inline assembly - unavailable in MSVC
    // On non-Apple ARM systems, try to read the system register
    // Note: This may fail on some systems where userspace access is restricted
    std::uint64_t id_aa64isar2_el0 = 0;
    // `ID_AA64ISAR2_EL0` is `S3_0_C0_C6_2`; the named form needs `-march=armv8.6-a+` to assemble, so the
    // generic `S<op0>_<op1>_<Cn>_<Cm>_<op2>` encoding is used instead - every assembler accepts it.
    __asm__ __volatile__("mrs %0, S3_0_C0_C6_2" : "=r"(id_aa64isar2_el0) : : "memory");
    // WFET is bits [3:0], value 2 indicates WFET support
    std::uint64_t const wfet_field = id_aa64isar2_el0 & 0xF;
    if (wfet_field >= 2) caps |= capability_arm64_wfet_k;
#endif

    // `DC CVAC` is a base-ISA clean; what varies is whether EL0 may issue it. Linux sets
    // `SCTLR_EL1.UCI`, so the capability is a kernel attestation, not a silicon probe.
#if FU_ON_LINUX
    caps |= capability_arm64_dc_cvac_k;
#endif

#elif FU_DETECT_ARCH_RISC5_

    // Basic PAUSE is available on RISC-V with the Zihintpause extension
    caps |= capability_risc5_pause_k;

    // Zawrs (`WRS.STO` / `WRS.NTO`) is learned one of two ways:
#if defined(__riscv_zawrs)
    // The compiler was told the target has it (`-march=...+zawrs`), so it is guaranteed present here.
    caps |= capability_risc5_wrs_k;
#elif defined(FU_DETECT_RISCV_HWPROBE_) && defined(SYS_riscv_hwprobe) && defined(RISCV_HWPROBE_EXT_ZAWRS)
    // Otherwise ask the kernel. With no CPU set, the value is the AND across all online harts.
    riscv_hwprobe probe {RISCV_HWPROBE_KEY_IMA_EXT_0, 0};
    long const probe_result = ::syscall(SYS_riscv_hwprobe, &probe, static_cast<std::size_t>(1),
                                        static_cast<std::size_t>(0), static_cast<void *>(nullptr), 0u);
    if (probe_result == 0 && (probe.value & RISCV_HWPROBE_EXT_ZAWRS) != 0) caps |= capability_risc5_wrs_k;
#endif

    // Zicbom user-mode cache-block management: `hwprobe` is the only sound attestation, since the
    // kernel only advertises the extension where it also set `senvcfg.CBCFE` - a compile-time
    // `+zicbom` proves nothing about the kernel, so unlike Zawrs there is no compile-time shortcut.
    // The `#ifdef` guards older uapi headers that predate the key.
#if defined(FU_DETECT_RISCV_HWPROBE_) && defined(SYS_riscv_hwprobe) && defined(RISCV_HWPROBE_EXT_ZICBOM)
    riscv_hwprobe cbo_probe {RISCV_HWPROBE_KEY_IMA_EXT_0, 0};
    long const cbo_result = ::syscall(SYS_riscv_hwprobe, &cbo_probe, static_cast<std::size_t>(1),
                                      static_cast<std::size_t>(0), static_cast<void *>(nullptr), 0u);
    if (cbo_result == 0 && (cbo_probe.value & RISCV_HWPROBE_EXT_ZICBOM) != 0) caps |= capability_risc5_zicbom_k;
#endif

#endif

    return caps;
}

#if FU_WITH_PLACE_MEMORY_ON_DOMAIN && FU_ON_LINUX
/**
 *  @brief Binds [@p ptr, @p ptr + @p size_bytes) to the single memory domain @p memory_domain_id.
 *  @param[in] mode An `MPOL_*` policy, already OR-ed with whatever mode flags the caller wants.
 *  @note Lives here, not beside its allocator callers, because this is the last header both
 *      `topology.hpp` and `allocators.hpp` see - so the `maxnode` quirk below is spelled once.
 */
FU_MAYBE_UNUSED_ static inline bool linux_bind_range_to_domain(void *ptr, std::size_t size_bytes,
                                                               memory_domain_id_t memory_domain_id, int mode) noexcept {
    if (memory_domain_id < 0 || static_cast<std::size_t>(memory_domain_id) >= max_memory_domains_k) return false;
    std::size_t const bit = static_cast<std::size_t>(memory_domain_id);
    std::size_t const bits_per_word = sizeof(unsigned long) * 8;
    unsigned long node_mask[nodemask_words_k] {};
    node_mask[bit / bits_per_word] = 1ul << (bit % bits_per_word);
    // ! `+ 1`: the manual says the mask holds "up to `maxnode`" bits, but the kernel decrements it
    // ! before sizing, so the exact width masks the top bit back off and the bind silently fails.
    return ::syscall(SYS_mbind, ptr, size_bytes, mode, node_mask, max_memory_domains_k + 1, 0) == 0;
}

/**
 *  @brief Probes whether this process may actually place memory - a kernel that offers `mbind` still
 *      lets seccomp or a cgroup `cpuset.mems` refuse it, and only the call itself can say.
 */
inline bool linux_can_place_memory_on_domain() noexcept {
    std::size_t const page_bytes = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    void *probe = ::mmap(nullptr, page_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (probe == MAP_FAILED) return false;
    // ? Node 0, always present where a topology exists. Bare `MPOL_BIND` - this asks only whether the
    // ? kernel will place at all, not about a mode flag no caller requested.
    bool const bound = linux_bind_range_to_domain(probe, page_bytes, 0, mpol_bind_k);
    ::munmap(probe, page_bytes);
    return bound;
}
#endif

/**
 *  @brief The memory-placement facilities this machine offers - NUMA and huge/large pages.
 *  @sa `cpu_capabilities` for the busy-wait side; together they form `runtime_capabilities`.
 */
inline capabilities_t ram_capabilities() noexcept {
    capabilities_t caps = capabilities_unknown_k;

#if FU_WITH_PLACE_MEMORY_ON_DOMAIN && FU_ON_WINDOWS
    // Windows always exposes the NUMA placement API (`VirtualAllocExNuma`); a single-node box simply
    // reports one node. Large-page availability hinges on a privilege the caller may not hold, so it
    // is probed by its minimum page size rather than a directory.
    caps |= capability_place_memory_on_domain_k;
    if (::GetLargePageMinimum() != 0) caps |= capability_place_huge_pages_on_domain_k;

#elif FU_WITH_PLACE_MEMORY_ON_DOMAIN && FU_ON_LINUX
    // NUMA placement is claimed only when a real one-page `mbind` succeeds - which subsumes every
    // weaker question, including the one `numa_available` used to be asked here.
    if (linux_can_place_memory_on_domain()) caps |= capability_place_memory_on_domain_k;

    // Huge pages are placed *on a domain*, so this capability cannot outlast memory placement itself -
    // the prerequisite the compile-time layer spells with an `#error`. A host that mounts the hugepages
    // sysfs but refuses `mbind` - a seccomp sandbox, a restricted container, qemu-user - must not report
    // the impossible pair, or a caller that trusts the runtime set walks into a placement that cannot work.
    if (caps & capability_place_memory_on_domain_k) {
        DIR *hugepages_dir = ::opendir("/sys/kernel/mm/hugepages");
        if (hugepages_dir) {
            caps |= capability_place_huge_pages_on_domain_k;
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
                    caps |= capability_huge_transparent_pages_k;
            ::fclose(thp_enabled);
        }
    }

#elif FU_WITH_PLACE_MEMORY_ON_DOMAIN && FU_ON_FREEBSD
    // The domainset syscalls always ship with the kernel; `vm.ndomains` answers whether it reports
    // any memory domains to place on. Superpages ride along: `MAP_ALIGNED_SUPER` is an alignment
    // hint with a base-page fallback, so claiming it can never promise more than the kernel honours.
    {
        int domains = 0;
        std::size_t domains_size = sizeof(domains);
        if (::sysctlbyname("vm.ndomains", &domains, &domains_size, nullptr, 0) == 0 && domains >= 1)
            caps |= capability_place_memory_on_domain_k | capability_place_huge_pages_on_domain_k;
    }

#endif // FU_WITH_PLACE_MEMORY_ON_DOMAIN

    return caps;
}

/**
 *  @brief Which features this machine turned out to offer, probing the CPU and the memory system.
 *  @sa `comptime_capabilities` for what this build is able to ask for in the first place.
 */
inline capabilities_t runtime_capabilities() noexcept {
    return static_cast<capabilities_t>(cpu_capabilities() | ram_capabilities());
}

/**
 *  @brief Which kernel facilities this translation unit was compiled to use, one bit per `FU_WITH_*`.
 *  @sa `runtime_capabilities` for what the machine underneath turned out to offer.
 *
 *  Consult it before reaching for a domain-aware API. Without `capability_colocate_pools_on_domain_k`
 *  there is no `colocated_pool` to spawn and no `linux_numa_allocator` to construct, and a caller has
 *  no other way to tell that apart from a machine that merely has one compute domain.
 *
 *  These are the @b same facility bits `runtime_capabilities` reports, asked the other way: this says
 *  the code was compiled, that says the machine offers it. A facility is usable only where both agree.
 */
constexpr capabilities_t comptime_capabilities() noexcept {
    // Both arms of each `?:` are `capabilities_t`, so there is no enumerator-versus-`0` mismatch for
    // `-Wextra` to object to, and `operator|` folds them into the result.
    return                                                                        //
        (FU_WITH_OS_THREADS ? capability_os_threads_k : capabilities_unknown_k) | //
        (FU_WITH_TOPOLOGY ? capability_topology_k : capabilities_unknown_k) |     //
        (FU_WITH_PLACE_THREADS_BY_AFFINITY ? capability_place_threads_by_affinity_k : capabilities_unknown_k) |
        (FU_WITH_PLACE_THREADS_BY_CORE_CLASS ? capability_place_threads_by_core_class_k : capabilities_unknown_k) |
        (FU_WITH_RESCHEDULE_THREADS_BY_CLASS ? capability_reschedule_threads_by_class_k : capabilities_unknown_k) |
        (FU_WITH_PLACE_MEMORY_ON_DOMAIN ? capability_place_memory_on_domain_k : capabilities_unknown_k) | //
        (FU_WITH_PLACE_HUGE_PAGES_ON_DOMAIN ? capability_place_huge_pages_on_domain_k : capabilities_unknown_k) |
        (FU_WITH_COLOCATE_POOLS_ON_DOMAIN ? capability_colocate_pools_on_domain_k : capabilities_unknown_k);
}

} // namespace forkunion
} // namespace ashvardanian
