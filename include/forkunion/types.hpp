/**
 *  @file types.hpp
 *  @brief Vocabulary and utilities: prongs, padded buffers, index splitting, claim cursors.
 *  @note Included by `<forkunion.hpp>`; not meant to be included on its own.
 */
#pragma once
#if defined(_MSC_VER)
#pragma warning(disable : 4505) // unreferenced function with internal linkage has been removed
#pragma warning(disable : 4324) // structure was padded due to alignment specifier
#pragma warning(disable : 4996) // `strncpy` etc. flagged "unsafe"; a preceding CRT include may have
                                // already marked them deprecated, so the define below cannot undo it
#pragma warning(disable : 4191) // `GetProcAddress` -> typed function pointer is the documented idiom
#endif

/*  Must precede the first CRT header below: `strncpy` and friends are only "unsafe" to MSVC, and the
 *  suppression is inert once `<cstring>` has already been parsed. `NOMINMAX` is hoisted for the same
 *  reason - it has to be set before the eventual `<windows.h>`. */
#if defined(_WIN32)
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <array>   // `std::array`
#include <atomic>  // `std::atomic`
#include <memory>  // `std::allocator`
#include <new>     // `std::hardware_destructive_interference_size`
#include <thread>  // `std::thread`
#include <utility> // `std::exchange`, `std::addressof`
#include <cassert> // `assert`
#include <cstddef> // `std::max_align_t`
#include <cstdio>  // `std::snprintf`
#include <cstdlib> // `std::strtoull`
#include <cstring> // `std::strlen`

#define FORKUNION_VERSION_MAJOR 3
#define FORKUNION_VERSION_MINOR 0
#define FORKUNION_VERSION_PATCH 0

#if defined(__cpp_exceptions) || defined(__EXCEPTIONS)
#define FU_DETECT_EXCEPTIONS_ 1
#else
#define FU_DETECT_EXCEPTIONS_ 0
#endif

#if !defined(FU_ALLOW_UNSAFE)
#define FU_ALLOW_UNSAFE FU_DETECT_EXCEPTIONS_
#endif

/*  Layer 1 is identity: where are we? Derived once, from compiler predefines, and used only to derive
 *  the capabilities below. Nothing else in the library may ask `__linux__` again. Identity is the kernel
 *  ABI - pthreads, `sched_setaffinity`, `gettid`, `/proc`, `/sys` - which Android shares in full, and
 *  which is all the capabilities below now ask for. What Bionic lacks is GLibC, and that is the separate
 *  `FU_ON_GLIBC` axis below; it gates no capability, because none is glibc's to grant.  */
#if defined(__linux__)
#define FU_ON_LINUX 1
#else
#define FU_ON_LINUX 0
#endif

/*  A Bionic sub-identity of Linux: Android shares the kernel ABI but not every GLibC extension, so the
 *  handful of capabilities that differ - affinity, whose `pthread_setaffinity_np` Bionic lacks before
 *  NDK 36 - key on this rather than re-asking `__ANDROID__` at the use site.  */
#if defined(__ANDROID__)
#define FU_ON_ANDROID 1
#else
#define FU_ON_ANDROID 0
#endif

#if defined(__APPLE__)
#define FU_ON_APPLE 1
#else
#define FU_ON_APPLE 0
#endif

#if defined(_WIN32)
#define FU_ON_WINDOWS 1
#else
#define FU_ON_WINDOWS 0
#endif

#if defined(__FreeBSD__)
#define FU_ON_FREEBSD 1
#else
#define FU_ON_FREEBSD 0
#endif

#define FU_ON_POSIX (FU_ON_LINUX || FU_ON_APPLE || FU_ON_FREEBSD)

#if FU_ON_LINUX && __has_include(<features.h>)
#include <features.h> // `__GLIBC__`
#endif

/*  Is-glibc, for facilities glibc provides that Bionic and musl do not - `backtrace`, say. Stays 0 on
 *  Apple and FreeBSD, whose libc is not glibc. Gates no capability below: sysfs and the syscall table
 *  are the kernel's, and every libc on Linux shares them.  */
#if defined(__GLIBC__)
#define FU_ON_GLIBC 1
#else
#define FU_ON_GLIBC 0
#endif

/*  The kernel UAPI headers ship separately from libc - `linux-headers` on Alpine and other musl
 *  distributions - so `<linux/mman.h>` must be probed, not assumed from the platform.  */
#if FU_ON_LINUX && __has_include(<linux/mman.h>)
#define FU_DETECT_LINUX_MMAN_ 1
#else
#define FU_DETECT_LINUX_MMAN_ 0
#endif

/*  Layer 2 is capabilities. Each answers exactly one question, and is named for the @b kernel @b
 *  facility rather than for the library that happens to provide it - so Windows' `VirtualAllocExNuma`
 *  satisfies `FU_WITH_PLACE_MEMORY_ON_DOMAIN` without inventing a second macro.
 *
 *  Every one auto-derives from Layer 1, and from the capabilities it leans on - so switching one off
 *  cascades to everything downstream, and `-DFU_WITH_TOPOLOGY=0` alone yields a coherent build rather
 *  than an `#error` about the four capabilities it silently orphaned. The `#error`s below then only
 *  ever fire on a contradiction the caller wrote out by hand.
 *
 *  A build system may @b override, never re-derive: that keeps the default in exactly one place,
 *  instead of duplicated across CMake, `build.rs`, and `build.zig`, where three copies would drift.  */

/** @brief Can we create operating-system threads directly, rather than through `std::thread`? */
#if !defined(FU_WITH_OS_THREADS)
#define FU_WITH_OS_THREADS (FU_ON_POSIX || FU_ON_WINDOWS)
#endif

/** @brief Can we enumerate this machine's cores, compute domains, and memory domains? */
#if !defined(FU_WITH_TOPOLOGY)
/*  Windows needs no separate library for this: `GetLogicalProcessorInformationEx` ships with the
 *  kernel since Vista and reports NUMA nodes, cores, processor groups, and caches in one call. */
/*  FreeBSD needs no separate library either: the in-kernel `cpuset`/NUMA framework enumerates memory
 *  domains through `sysctl vm.ndomains` and `cpuset_getaffinity(CPU_WHICH_DOMAIN)`. */
/*  Nor Linux: the harvest reads `/sys/devices/system/node`, which the kernel mounts wherever there are
 *  domains to report - so no libc and no `libnuma-dev` on the build host gates it. */
#define FU_WITH_TOPOLOGY (FU_ON_APPLE || FU_ON_WINDOWS || FU_ON_FREEBSD || FU_ON_LINUX)
#endif

/**
 *  @brief Can we bind a thread to a set of cores, and have the kernel honour it?
 *  @note Deliberately independent of `FU_WITH_PLACE_MEMORY_ON_DOMAIN`. `pthread_setaffinity_np` needs no
 *        `libnuma`, and a Linux box without it could pin perfectly well - it simply never did,
 *        because a single NUMA macro guarded both.
 *  @note False on Apple Silicon, where `thread_policy_set(THREAD_AFFINITY_POLICY)` answers
 *        `KERN_NOT_SUPPORTED`. Its only placement lever is `FU_WITH_PLACE_THREADS_BY_CORE_CLASS`.
 */
#if !defined(FU_WITH_PLACE_THREADS_BY_AFFINITY)
#define FU_WITH_PLACE_THREADS_BY_AFFINITY (FU_ON_LINUX || FU_ON_FREEBSD || FU_ON_WINDOWS)
#endif

/** @brief Can we hint which class of core a thread should run on, at creation time? */
#if !defined(FU_WITH_PLACE_THREADS_BY_CORE_CLASS)
#define FU_WITH_PLACE_THREADS_BY_CORE_CLASS FU_ON_APPLE
#endif

/** @brief Can we change @b another thread's scheduling class, to sleep or wake it cheaply?
 *  @note Linux spells it `sched_setscheduler(SCHED_IDLE)`; FreeBSD rejects `SCHED_IDLE` but reaches the
 *        same idle class through `rtprio_thread(RTP_SET, {RTP_PRIO_IDLE})`. */
#if !defined(FU_WITH_RESCHEDULE_THREADS_BY_CLASS)
#define FU_WITH_RESCHEDULE_THREADS_BY_CLASS (FU_ON_LINUX || FU_ON_FREEBSD)
#endif

/** @brief Can we place pages on a chosen memory domain? */
#if !defined(FU_WITH_PLACE_MEMORY_ON_DOMAIN)
/*  Linux places with `mbind`; Windows with `VirtualAllocExNuma`; FreeBSD sets the calling thread's
 *  `domainset` to a PREFER policy and first-touches. Same capability, named for the facility. */
#define FU_WITH_PLACE_MEMORY_ON_DOMAIN ((FU_ON_LINUX || FU_ON_WINDOWS || FU_ON_FREEBSD) && FU_WITH_TOPOLOGY)
#endif

/** @brief Can we request pages larger than the base page? */
#if !defined(FU_WITH_PLACE_HUGE_PAGES_ON_DOMAIN)
/*  Linux calls them huge pages (`MAP_HUGETLB`); Windows calls them large pages (`MEM_LARGE_PAGES`),
 *  gated behind the `SeLockMemoryPrivilege` the caller must already hold; FreeBSD hints the alignment
 *  with `MAP_ALIGNED_SUPER` and lets its transparent superpages promote.
 *
 *  On Linux the `MAP_HUGE_2MB`-family constants live in `<linux/mman.h>`, and the kernel UAPI headers
 *  ship separately from libc (`linux-headers` on Alpine and other musl distributions) - the platform
 *  alone does not guarantee the header, and assuming it would fail at `#include`, not at a check.  */
#define FU_WITH_PLACE_HUGE_PAGES_ON_DOMAIN \
    (((FU_ON_LINUX && FU_DETECT_LINUX_MMAN_) || FU_ON_WINDOWS || FU_ON_FREEBSD) && FU_WITH_PLACE_MEMORY_ON_DOMAIN)
#endif

/*  Layer 3 is aggregates. Never hand-written, always implied, so they cannot drift.  */

/**
 *  @brief Whether the domain-aware `colocated_pool` and `distributed_pool` are compiled at all.
 *
 *  They need threads, and a topology to spawn onto. They do @b not need to be able to pin, or to
 *  place memory: a machine may report a rich compute topology and refuse both. Every Apple Silicon
 *  Mac is one, and conflating these erased both pools from every platform but Linux - including the
 *  ones whose topology we already harvest.
 */
#define FU_WITH_COLOCATE_POOLS_ON_DOMAIN (FU_WITH_OS_THREADS && FU_WITH_TOPOLOGY)

/*  A bad override should fail at the `#include`, not at link time - whether it is a contradiction the
 *  caller wrote by hand, or a retired spelling that would otherwise be silently ignored, handing a
 *  build system or a downstream a default build unlike the one it asked for.  */
#if FU_WITH_PLACE_THREADS_BY_AFFINITY && FU_ON_APPLE
#error \
    "FU_WITH_PLACE_THREADS_BY_AFFINITY: Apple answers KERN_NOT_SUPPORTED to thread_policy_set; pinning cannot be forced on"
#endif
#if FU_WITH_PLACE_MEMORY_ON_DOMAIN && !(FU_ON_LINUX || FU_ON_WINDOWS || FU_ON_FREEBSD)
#error "FU_WITH_PLACE_MEMORY_ON_DOMAIN needs a kernel that can place pages on a node"
#endif
#if FU_WITH_PLACE_MEMORY_ON_DOMAIN && !FU_WITH_TOPOLOGY
#error "FU_WITH_PLACE_MEMORY_ON_DOMAIN places pages on domains we would not have discovered"
#endif
#if FU_WITH_PLACE_HUGE_PAGES_ON_DOMAIN && !FU_WITH_PLACE_MEMORY_ON_DOMAIN
#error "FU_WITH_PLACE_HUGE_PAGES_ON_DOMAIN places huge pages on a domain; it needs FU_WITH_PLACE_MEMORY_ON_DOMAIN"
#endif
#if defined(FU_ENABLE_NUMA) || defined(FU_WITH_NUMA_MEMORY) || defined(FU_WITH_HUGE_PAGES) || \
    defined(FU_WITH_THREAD_PINNING) || defined(FU_WITH_TOPOLOGY_METRICS)
#error \
    "Retired capability macro. Use FU_WITH_PLACE_MEMORY_ON_DOMAIN, FU_WITH_PLACE_HUGE_PAGES_ON_DOMAIN, or FU_WITH_PLACE_THREADS_BY_AFFINITY"
#endif

#if FU_ALLOW_UNSAFE
#include <exception> // `std::exception_ptr`
#endif

/*  No `<numa.h>`, no `<numaif.h>`: sysfs needs no header, and `mbind` arrives by syscall number.  */
#if FU_WITH_PLACE_MEMORY_ON_DOMAIN && FU_ON_LINUX
#include <sys/syscall.h> // `SYS_mbind`, the policy syscall no libc wraps
#include <unistd.h>      // `syscall`
#include <sys/mman.h>    // `mmap`, `MAP_PRIVATE`, `MAP_ANONYMOUS`
#if __has_include(<linux/mempolicy.h>)
#include <linux/mempolicy.h> // `MPOL_BIND`, `MPOL_F_STATIC_NODES` - checked against, never needed
#endif
#endif

#if FU_WITH_PLACE_HUGE_PAGES_ON_DOMAIN && FU_ON_LINUX
#include <linux/mman.h> // `MAP_HUGE_2MB`, `MAP_HUGE_1GB`
#endif

/*  Both the huge-page inventory and the topology harvest walk sysfs directories - a Linux-only
 *  concern. Windows has no `<dirent.h>` under MSVC, and its large pages are probed by size, not path. */
#if (FU_WITH_PLACE_HUGE_PAGES_ON_DOMAIN || FU_WITH_TOPOLOGY) && FU_ON_LINUX
#include <dirent.h> // `opendir`, `readdir`, `closedir`
#endif

#if FU_WITH_OS_THREADS && FU_ON_POSIX
#include <pthread.h> // `pthread_create`, `pthread_setname_np`
#include <ctime>     // `nanosleep`, `clock_nanosleep`
#endif

#if FU_WITH_PLACE_THREADS_BY_AFFINITY && FU_ON_POSIX
#include <cerrno>  // `errno`, `EINVAL` - the kernel's way of saying "your mask is too narrow"
#include <sched.h> // `cpu_set_t`, `sched_getaffinity`, `pthread_setaffinity_np`
#endif

#if FU_WITH_PLACE_THREADS_BY_CORE_CLASS
#include <sys/qos.h> // `qos_class_t`, `pthread_attr_set_qos_class_np`
#endif

#if FU_ON_POSIX
#include <unistd.h> // `gettid`, `sysconf`
#endif

#if FU_ON_APPLE
#include <sys/sysctl.h> // `sysctl`
#endif

/*  FreeBSD's placement facilities are all base-system, so - like `<windows.h>` on Windows and
 *  `<sys/sysctl.h>` on Apple - one identity-gated block pulls them once, rather than a per-capability
 *  block re-`#include`-ing `<sys/cpuset.h>` for pinning, topology, and memory placement in turn. */
#if FU_ON_FREEBSD
#include <sys/param.h>     // ! Must precede `<sys/cpuset.h>`
#include <sys/cpuset.h>    // `cpuset_t`, `cpuset_getaffinity`, `cpuset_setdomain`, `CPU_WHICH_DOMAIN`
#include <pthread_np.h>    // `pthread_setaffinity_np`, `pthread_getthreadid_np` for the rtprio lwpid
#include <sys/domainset.h> // `domainset_t`, `DOMAINSET_SET`, `DOMAINSET_POLICY_PREFER` for memory placement
#include <sys/rtprio.h>    // `rtprio_thread`, `RTP_SET`, `RTP_PRIO_IDLE`, `RTP_PRIO_NORMAL` for reschedule
#include <sys/sysctl.h>    // `sysctlbyname` for `vm.ndomains`
#include <sys/mman.h>      // `mmap`, `MAP_PRIVATE`, `MAP_ANONYMOUS`, `MAP_ALIGNED_SUPER`
#endif

#if FU_ON_WINDOWS
// `NOMINMAX` and `_CRT_SECURE_NO_WARNINGS` are already defined at the top of this header, before the
// CRT includes, where they can still take effect.
#include <windows.h> // `GlobalMemoryStatusEx`, `GetLogicalProcessorInformationEx`, `VirtualAllocExNuma`
#include <io.h>      // `_isatty`, `_fileno`
#if defined(_MSC_VER)
#pragma comment(lib, "advapi32.lib") // `OpenProcessToken`, `LookupPrivilegeValueW` for large pages
#endif
#endif

/**
 *  On C++17 and later we can detect misuse of lambdas that are not properly annotated.
 *  On C++20 and later we can use concepts for cleaner compile-time checks.
 */
#if __cplusplus >= 202002L
#define FU_DETECT_CPP_20_ 1
#else
#define FU_DETECT_CPP_20_ 0
#endif
#if __cplusplus >= 201703L
#define FU_DETECT_CPP_17_ 1
#else
#define FU_DETECT_CPP_17_ 0
#endif

/*  Detect target CPU architecture.
 *  We'll only use it when compiling Inline Assembly code on GCC or Clang.
 */
#if defined(__arm64__) || defined(__aarch64__) || defined(_M_ARM64)
#define FU_DETECT_ARCH_ARM64_ 1
#else
#define FU_DETECT_ARCH_ARM64_ 0
#endif
#if defined(__x86_64__) || defined(__amd64__) || defined(_M_X64) || defined(_M_AMD64)
#define FU_DETECT_ARCH_X86_64_ 1
#else
#define FU_DETECT_ARCH_X86_64_ 0
#endif
#if defined(__riscv)
#define FU_DETECT_ARCH_RISC5_ 1
#else
#define FU_DETECT_ARCH_RISC5_ 0
#endif
#if defined(__s390x__)
#define FU_DETECT_ARCH_S390X_ 1
#else
#define FU_DETECT_ARCH_S390X_ 0
#endif
#if defined(__powerpc64__) || defined(__ppc64__)
#define FU_DETECT_ARCH_PPC64_ 1
#else
#define FU_DETECT_ARCH_PPC64_ 0
#endif

#if defined(__i386__) || defined(_M_IX86)
#define FU_DETECT_ARCH_X86_32_ 1
#else
#define FU_DETECT_ARCH_X86_32_ 0
#endif

#if FU_DETECT_CPP_17_
#include <type_traits> // `std::is_nothrow_invocable_r`
#endif

#if FU_DETECT_CPP_20_
#include <concepts> // `std::same_as`, `std::invocable`
#include <bit>      // `std::popcount`
#endif

#if FU_DETECT_CPP_17_
#define FU_MAYBE_UNUSED_ [[maybe_unused]]
#else
#if defined(__GNUC__) || defined(__clang__)
#define FU_MAYBE_UNUSED_ __attribute__((unused))
#elif defined(_MSC_VER)
#define FU_MAYBE_UNUSED_ __pragma(warning(suppress : 4100 4189))
#else
#define FU_MAYBE_UNUSED_
#endif
#endif

#define fu_unused_(x) ((void)(x))

#if defined(__GNUC__) || defined(__clang__)
#define fu_unlikely_(x) __builtin_expect(!!(x), 0)
#else
#define fu_unlikely_(x) (x)
#endif

/*  Whether GNU-style inline assembly (`__asm__`) is available. GCC and Clang have it; MSVC does not on
 *  x86-64 or AArch64, and reaches the same instructions through intrinsics instead. Gates only the paths
 *  that genuinely need inline asm - the hand-encoded `WFET`/`WRS` opcodes and the MSR/register reads. */
#if defined(__GNUC__) || defined(__clang__)
#define FU_DETECT_INLINE_ASM_SUPPORT_ 1
#else
#define FU_DETECT_INLINE_ASM_SUPPORT_ 0
#endif

/*  Whether MSVC's cache-hint intrinsics are available: `_mm_cldemote` and `_m_prefetchw` on x64,
 *  `__prefetch2` on AArch64 - all encoded unconditionally, no `/arch` flag needed. 1922 == VS 2019
 *  16.2, the release that introduced `_mm_cldemote`. clang-cl takes the inline-asm path above. */
#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER >= 1922
#define FU_DETECT_HINT_INTRINSICS_ 1
#else
#define FU_DETECT_HINT_INTRINSICS_ 0
#endif

/** @brief Can we deterministically push a freshly-written cache line away from this core?
 *  @note x86 `CLDEMOTE` moves it toward the LLC and retains it; AArch64 has no demote, only the
 *        `DC CVAC` clean, legal at EL0 only where the kernel sets `SCTLR_EL1.UCI` - Linux does,
 *        and Windows traps it, so MSVC-ARM64 never reaches this gate. RISC-V `cbo.clean` traps
 *        unless the kernel set `senvcfg.CBCFE`, which no compile-time macro can prove, so it is
 *        reached only through the runtime capability, never this gate. */
#if !defined(FU_WITH_DEMOTE_CACHE_LINES)
#define FU_WITH_DEMOTE_CACHE_LINES                                    \
    ((FU_DETECT_INLINE_ASM_SUPPORT_ || FU_DETECT_HINT_INTRINSICS_) && \
     (FU_DETECT_ARCH_X86_64_ || (FU_DETECT_ARCH_ARM64_ && FU_ON_LINUX)))
#endif

/** @brief Can we pull a cache line toward this core with write intent, ahead of an atomic claim?
 *  @note Every ISA here places its write-prefetch in hint space - x86 `PREFETCHW`, AArch64
 *        `PRFM PSTL1KEEP`, RISC-V `prefetch.w` - so emission can never fault, on any part. */
#if !defined(FU_WITH_PROMOTE_CACHE_LINES)
#define FU_WITH_PROMOTE_CACHE_LINES                                   \
    ((FU_DETECT_INLINE_ASM_SUPPORT_ || FU_DETECT_HINT_INTRINSICS_) && \
     (FU_DETECT_ARCH_X86_64_ || FU_DETECT_ARCH_ARM64_ || FU_DETECT_ARCH_RISC5_))
#endif

#if FU_WITH_DEMOTE_CACHE_LINES && !(FU_DETECT_INLINE_ASM_SUPPORT_ || FU_DETECT_HINT_INTRINSICS_)
#error "FU_WITH_DEMOTE_CACHE_LINES emits hint opcodes; it needs GNU inline assembly or MSVC intrinsics"
#endif
#if FU_WITH_DEMOTE_CACHE_LINES && !(FU_DETECT_ARCH_X86_64_ || FU_DETECT_ARCH_ARM64_)
#error "FU_WITH_DEMOTE_CACHE_LINES names no demote-capable ISA on this target"
#endif

namespace ashvardanian {
namespace forkunion {

/** @brief The OS's NUMA node number - on Linux, an `N` for which `/sys/devices/system/node/nodeN` exists. */
using memory_domain_id_t = int;
/** @brief Opaque logical-processor id; on Windows it packs a group and a bit. */
using core_id_t = int;
/** @brief Physical CPU socket, or -1 where the OS will not say. */
using socket_id_t = int;
/** @brief A core's performance tier, ranked from fastest to lowest-power, like "performance" or "efficiency". */
using core_quality_t = int;

#if FU_WITH_PLACE_MEMORY_ON_DOMAIN && FU_ON_LINUX
/*  The two `mbind` inputs `<numaif.h>` supplied. Syscall ABI, so naming them needs no header - and
 *  where `<linux/mempolicy.h>` is installed, the `static_assert`s hold us to the kernel's spelling.  */
static constexpr int mpol_bind_k = 2;               // ? `MPOL_BIND` - allocate strictly from the mask
static constexpr int mpol_static_nodes_k = 1 << 15; // ? `MPOL_F_STATIC_NODES` - literal ids, not cpuset-relative

#if defined(MPOL_BIND)
static_assert(mpol_bind_k == MPOL_BIND, "MPOL_BIND is the kernel's; ours must match it");
static_assert(mpol_static_nodes_k == MPOL_F_STATIC_NODES, "MPOL_F_STATIC_NODES is the kernel's; ours must match it");
#endif

/**
 *  @brief One past the highest memory-domain id we will bind - the node mask's width.
 *  @note The kernel's own ceiling: `MAX_NUMNODES` is `1 << CONFIG_NODES_SHIFT`, which peaks at 10.
 *        An id at or past this is one the kernel cannot represent, so declining it declines nothing.
 *        Bounds @b domains, never cores - those are a `core_mask`, which grows.
 */
static constexpr std::size_t max_memory_domains_k = 1024;
static constexpr std::size_t nodemask_words_k = max_memory_domains_k / (sizeof(unsigned long) * 8);
#endif // FU_WITH_PLACE_MEMORY_ON_DOMAIN && FU_ON_LINUX

/**
 *  @brief A position in `machine_topology`'s array of @b compute domains, in [0, compute_domains_count).
 *  @sa `memory_domain_index_t`, which indexes a different array and must not be confused with this.
 *
 *  These are plain enums with a fixed underlying type, not `enum class`, deliberately. The implicit
 *  widening to `std::size_t` survives, so comparisons and array subscripts read exactly as before;
 *  what does @b not survive is passing one where the other is expected, because no conversion exists
 *  between two enumeration types. Minting one costs a `static_cast`, which is the point: it marks the
 *  spot where an untyped integer becomes a claim about @b which axis it indexes.
 *
 *  This is not hypothetical. `RoundRobinVec` handed a compute-domain index to an allocator expecting a
 *  memory-domain index, and `nbody.cpp` did the same to an array of per-node replicas. Both compiled,
 *  both ran on every machine where the two counts happened to match, and both broke on the first chip
 *  with three compute domains over one memory domain.
 *
 *  @note The implicit widening also means a raw `array[compute_domain]` still compiles. The enums stop
 *        the argument-passing mistake, not the subscript one. Rust's newtypes stop both.
 */
enum compute_domain_index_t : std::size_t {};

/** @brief A position in `machine_topology`'s array of @b memory domains, in [0, memory_domains_count). */
enum memory_domain_index_t : std::size_t {};

/**
 *  @brief Defines the in- and exclusivity of the calling thread in for the executing task.
 *  @sa `caller_inclusive_k` and `caller_exclusive_k`
 *
 *  This enum affects how the join is performed. If the caller is inclusive, 1/Nth of the call
 *  will be executed by the calling thread (as opposed to workers) and the join will happen
 *  inside of the calling scope.
 */
enum caller_exclusivity_t : unsigned int {
    caller_inclusive_k = 0,
    caller_exclusive_k = 1,
};

/**
 *  @brief Defines the mood of the thread-pool, whether it is busy or about to die.
 *  @sa `mood_t::grind_k`, `mood_t::chill_k`, `mood_t::die_k`
 */
enum class mood_t : unsigned int {
    /** That's our default ;) */
    grind_k = 0,
    /** Sleepy and tired, but just a wake-up call away. */
    chill_k,
    /** The thread is about to die, we must exit the loop peacefully. */
    die_k,
};

/**
 *  @brief Describes all the special library features, both those compiled in and those found here.
 *  @sa `comptime_capabilities` and `runtime_capabilities`
 *
 *  One bit per facility, and two accessors ask two questions of the same bit. `comptime_capabilities()`
 *  reports whether the code for a facility was @b built: `capability_place_huge_pages_on_domain_k` there
 *  means we compiled the path that asks the kernel for them. `runtime_capabilities()` reports whether the
 *  facility is @b present on this machine: the same bit means the kernel is offering them now.
 *
 *  Neither implies the other. A binary that built `capability_place_memory_on_domain_k` runs well on a
 *  single-node box, where the runtime accessor never sets that bit; and a machine with four NUMA nodes
 *  reports none of them to a build that left the topology out. A facility is usable here only when
 *  both accessors agree, so `comptime_capabilities() & runtime_capabilities()` is the honest set.
 */
enum capabilities_t : unsigned int {
    capabilities_unknown_k = 0,

    /** The `PAUSE` spin hint, on every x86 since the Pentium 4. */
    capability_x86_pause_k = 1 << 0,
    /** `TPAUSE` sleeps the core until a deadline, rather than spinning. Needs the `WAITPKG` feature. */
    capability_x86_tpause_k = 1 << 1,
    /** The `YIELD` hint, on every AArch64. Releases the pipeline to a sibling hardware thread. */
    capability_arm64_yield_k = 1 << 2,
    /** `WFET` sleeps the core until a deadline or an event. Needs `FEAT_WFxT`. */
    capability_arm64_wfet_k = 1 << 3,
    /** The `PAUSE` spin hint, from the `Zihintpause` extension. */
    capability_risc5_pause_k = 1 << 4,
    /** `WRS.STO` sleeps the hart until a reservation breaks or a timeout. Needs the `Zawrs` extension. */
    capability_risc5_wrs_k = 1 << 5,

    /** Own the raw OS thread handle instead of a `std::thread` - the substrate the thread levers stand on. Built:
       `FU_WITH_OS_THREADS`. */
    capability_os_threads_k = 1 << 6,
    /** Enumerate this machine's cores, compute domains, and memory domains. The root the placements need. Built:
       `FU_WITH_TOPOLOGY`. */
    capability_topology_k = 1 << 7,
    /**
     *  Bind a thread to a set of cores, choosing where it runs. Built: `FU_WITH_PLACE_THREADS_BY_AFFINITY`.
     *  @sa `capability_os_threads_k` - the owned handle this needs.
     */
    capability_place_threads_by_affinity_k = 1 << 8,
    /**
     *  Steer a thread onto a class of core at creation, choosing where it runs. Built:
     * `FU_WITH_PLACE_THREADS_BY_CORE_CLASS`.
     *  @sa `capability_os_threads_k` - the owned handle this needs.
     */
    capability_place_threads_by_core_class_k = 1 << 9,
    /**
     *  Reclass a thread's scheduler to sleep or wake it, choosing when it runs. Built:
     * `FU_WITH_RESCHEDULE_THREADS_BY_CLASS`.
     *  @sa `capability_os_threads_k` - the owned handle this needs.
     */
    capability_reschedule_threads_by_class_k = 1 << 10,

    /**
     *  Place a buffer's pages on a chosen memory domain. Built: `FU_WITH_PLACE_MEMORY_ON_DOMAIN`.
     *  @sa `capability_topology_k` - the enumerated domains this places onto.
     */
    capability_place_memory_on_domain_k = 1 << 11,
    /**
     *  Place larger-than-base pages on a chosen memory domain. A narrower case of memory placement. Built:
     * `FU_WITH_PLACE_HUGE_PAGES_ON_DOMAIN`.
     *  @sa `capability_place_memory_on_domain_k` - the placement this specializes.
     */
    capability_place_huge_pages_on_domain_k = 1 << 12,
    /**
     *  The kernel promotes base pages to huge pages on its own. A passive, runtime-only observation.
     *  @sa `capability_place_huge_pages_on_domain_k` - the explicit request this is the automatic counterpart to.
     */
    capability_huge_transparent_pages_k = 1 << 13,

    /**
     *  Compile the domain-aware `colocated_pool` and `distributed_pool`. Built: `FU_WITH_COLOCATE_POOLS_ON_DOMAIN`.
     *  @sa `capability_os_threads_k` and `capability_topology_k` - both are needed; memory placement is not.
     */
    capability_colocate_pools_on_domain_k = 1 << 14,

    /**
     *  `CLDEMOTE` moves a just-written line from this core's private caches toward the shared LLC and
     *  retains it there. Runtime-detected on Sapphire-Rapids-class parts; the emitting functor is
     *  chosen at compile time by `FU_WITH_DEMOTE_CACHE_LINES`, so this bit reports, it never dispatches.
     */
    capability_x86_cldemote_k = 1 << 15,
    /**
     *  `DC CVAC` cleans a dirty line to the coherency point - the nearest thing AArch64 has to a
     *  demote: the next claimer's snoop finds a clean line instead of forcing a dirty intervention.
     *  Set where EL0 execution is known-legal, i.e. Linux, which sets `SCTLR_EL1.UCI`.
     */
    capability_arm64_dc_cvac_k = 1 << 16,
    /**
     *  The kernel enabled user-mode Zicbom cache-block management (`senvcfg.CBCFE`), attested through
     *  `hwprobe` - the only sound signal, since a compile-time `+zicbom` proves nothing about the
     *  kernel. No compile-time policy emits `cbo.clean` yet; the bit is the hook for runtime dispatch.
     */
    capability_risc5_zicbom_k = 1 << 17,

    /** Composite mask of every busy-wait waiter bit above, to enumerate the ones a machine offers. */
    capability_any_yield_k = capability_x86_pause_k | capability_x86_tpause_k | capability_arm64_yield_k |
                             capability_arm64_wfet_k | capability_risc5_pause_k | capability_risc5_wrs_k,

};

/**
 *  @brief Which of the three pool shapes an instance is, independent of the waiter it uses.
 *  @sa `flat_k`, `colocated_k`, and `distributed_k`
 *
 *  This is the shape discriminator, not a capability: `capability_colocate_pools_on_domain_k` says the
 *  colocated and distributed shapes were @b compiled, while this says which shape a given pool @b is. A
 *  `flat_k` pool ignores the machine's domains; a `colocated_k` pool pins to one compute domain; a
 *  `distributed_k` pool spans every domain and replicates per memory domain.
 */
enum class pool_kind_t : unsigned int {
    /** No pool constructed - the shape is undecided until a spawn builds one. */
    unknown_k = 0,
    /** No domain awareness - one worker set over all cores, as if the machine were uniform. */
    flat_k,
    /** Pinned to a single compute domain, so its workers share a cache and a memory domain. */
    colocated_k,
    /** Spans every compute domain, replicating hot state per memory domain. */
    distributed_k,
};

constexpr capabilities_t operator|(capabilities_t a, capabilities_t b) {
    return static_cast<capabilities_t>(static_cast<unsigned int>(a) | static_cast<unsigned int>(b));
}

inline capabilities_t &operator|=(capabilities_t &a, capabilities_t b) noexcept { return a = a | b; }

/**
 *  @brief The lower-case name of a single capability bit, or `nullptr` for a composite or unknown one.
 *  @note The one source of truth for capability names - the string builders and the C ABI both defer here.
 *  @sa `capability_named`, its inverse.
 */
constexpr char const *capability_name(capabilities_t const capability) noexcept {
    switch (capability) {
    case capability_x86_pause_k: return "x86_pause";
    case capability_x86_tpause_k: return "x86_tpause";
    case capability_arm64_yield_k: return "arm64_yield";
    case capability_arm64_wfet_k: return "arm64_wfet";
    case capability_risc5_pause_k: return "risc5_pause";
    case capability_risc5_wrs_k: return "risc5_wrs";
    case capability_x86_cldemote_k: return "x86_cldemote";
    case capability_arm64_dc_cvac_k: return "arm64_dc_cvac";
    case capability_risc5_zicbom_k: return "risc5_zicbom";
    case capability_os_threads_k: return "os_threads";
    case capability_topology_k: return "topology";
    case capability_place_threads_by_affinity_k: return "place_threads_by_affinity";
    case capability_place_threads_by_core_class_k: return "place_threads_by_core_class";
    case capability_reschedule_threads_by_class_k: return "reschedule_threads_by_class";
    case capability_place_memory_on_domain_k: return "place_memory_on_domain";
    case capability_place_huge_pages_on_domain_k: return "place_huge_pages_on_domain";
    case capability_huge_transparent_pages_k: return "huge_transparent_pages";
    case capability_colocate_pools_on_domain_k: return "colocate_pools_on_domain";
    default: return nullptr;
    }
}

/**
 *  @brief The single capability bit named @p name, or `capabilities_unknown_k` if none matches.
 *  @note The inverse of `capability_name`; it defers to that one table, so the two cannot drift.
 */
inline capabilities_t capability_named(char const *name) noexcept {
    if (name == nullptr) return capabilities_unknown_k;
    for (unsigned int bit_index = 0; bit_index < sizeof(capabilities_t) * 8; ++bit_index) {
        capabilities_t const candidate = static_cast<capabilities_t>(1u << bit_index);
        char const *const candidate_name = capability_name(candidate);
        if (candidate_name != nullptr && std::strcmp(candidate_name, name) == 0) return candidate;
    }
    return capabilities_unknown_k;
}

/**
 *  @brief Defines variable alignment to avoid false sharing.
 *  @see https://en.cppreference.com/w/cpp/thread/hardware_destructive_interference_size
 *  @see https://docs.rs/crossbeam-utils/latest/crossbeam_utils/struct.CachePadded.html
 *
 *  The C++ STL way to do it is to use `std::hardware_destructive_interference_size` if available:
 *
 *  @code{.cpp}
 *  #if defined(__cpp_lib_hardware_interference_size)
 *  static constexpr std::size_t default_alignment_k = std::hardware_destructive_interference_size;
 *  #else
 *  static constexpr std::size_t default_alignment_k = alignof(std::max_align_t);
 *  #endif
 *  @endcode
 *
 *  That however results into all kinds of ABI warnings with GCC, and suboptimal alignment choice,
 *  unless you hard-code `--param hardware_destructive_interference_size=64` or disable the warning
 *  with `-Wno-interference-size`.
 */
static constexpr std::size_t default_alignment_k = 128;

/**
 *  @brief Defines saturated addition for a given unsigned integer type.
 *  @see https://en.cppreference.com/w/cpp/numeric/add_sat
 */
template <typename scalar_type_>
inline scalar_type_ add_sat(scalar_type_ a, scalar_type_ b) noexcept {
    static_assert(std::is_unsigned<scalar_type_>::value, "Scalar type must be an unsigned integer");
#if defined(__cpp_lib_saturation_arithmetic)
    return std::add_sat(a, b); // In C++26
#else
    return (std::numeric_limits<scalar_type_>::max() - a < b) ? std::numeric_limits<scalar_type_>::max() : a + b;
#endif
}

/**
 *  @brief Byte-wise `memcpy` of @p from into @p to, written as an explicit loop.
 *
 *  Not `std::memcpy`: the wait monitors that call this (`arm64_wfet_t`, `risc5_wrs_t`) are pinned to a
 *  narrower `target(...)` than `-march=native`, where the fortified `always_inline` `memcpy` cannot inline
 *  ("target specific option mismatch"). Reading and writing through `unsigned char` keeps it well-defined.
 */
template <typename value_type_>
inline void copy_bytes(value_type_ const *from, value_type_ *to) noexcept {
    unsigned char const *from_bytes = reinterpret_cast<unsigned char const *>(from);
    unsigned char *to_bytes = reinterpret_cast<unsigned char *>(to);
    for (std::size_t byte_index = 0; byte_index < sizeof(value_type_); ++byte_index)
        to_bytes[byte_index] = from_bytes[byte_index];
}

/** @brief Checks if the @p x is a power of two. */
constexpr bool is_power_of_two(std::size_t x) noexcept { return x && ((x & (x - 1)) == 0); }

/**
 *  @brief Counts the set bits in @p value.
 *  @see https://en.cppreference.com/w/cpp/numeric/popcount
 */
template <typename scalar_type_>
constexpr int popcount(scalar_type_ value) noexcept {
    static_assert(std::is_unsigned<scalar_type_>::value, "Scalar type must be an unsigned integer");
#if FU_DETECT_CPP_20_
    return std::popcount(value); // In C++20
#else
    // Kernighan's trick: each `value &= value - 1` clears the lowest set bit, so the loop runs once
    // per set bit rather than once per bit width.
    int count = 0;
    for (; value; value &= static_cast<scalar_type_>(value - 1)) ++count;
    return count;
#endif
}

/**
 *  @brief Ceiling of @p value divided by @p divisor - how many buckets of that size a value needs.
 *  @note @p divisor must be non-zero; overflow of @p value near the type maximum is not guarded.
 */
constexpr std::size_t div_ceil(std::size_t value, std::size_t divisor) noexcept {
    return (value + divisor - 1) / divisor;
}

/**
 *  @brief Smallest multiple of @p multiple that is not less than @p value.
 *  @note @p multiple must be non-zero; overflow of @p value near the type maximum is not guarded.
 */
constexpr std::size_t round_up_to_multiple(std::size_t value, std::size_t multiple) noexcept {
    return div_ceil(value, multiple) * multiple;
}

/**
 *  @brief The SplitMix64 avalanche - a well-mixed pure function of the @p counter.
 *
 *  A counter-based generator instead of a stateful one: every draw is independent, so parallel
 *  consumers need no shared state, and a sequence is reproducible from indices alone. The same
 *  constants drive the benchmark generators in `scripts/`, ported bit-identically to Rust and Zig.
 */
constexpr std::uint64_t split_mix(std::uint64_t const counter) noexcept {
    std::uint64_t x = (counter + 1) * 0x9E37'79B9'7F4A'7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58'476D'1CE4'E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D0'49BB'1331'11EBull;
    return x ^ (x >> 31);
}

template <typename value_type_, typename comparator_type_ = std::less<value_type_>>
void bubble_sort(value_type_ *array, std::size_t size, comparator_type_ comp = {}) noexcept {
    if (size < 2) return; // ? Already sorted; also guards the `size - 1` unsigned underflow
    assert(array != nullptr && "Array must not be null");
    for (std::size_t i = 0; i < size - 1; ++i)
        for (std::size_t j = 0; j < size - i - 1; ++j)
            if (comp(array[j + 1], array[j])) std::swap(array[j], array[j + 1]);
}

/**
 *  @brief Dense-ranks `count` items by an ascending integer key, writing each item's 0-based rank.
 *  @return The number of distinct keys, at least 1 when `count > 0`.
 *
 *  `key(index)` must read a @b stable source and `assign(index, rank)` write a @b different field, so
 *  ranking in place never corrupts a not-yet-ranked item whose key repeats. Used to turn raw CPU
 *  capacities into compute levels and raw memory tiers into memory levels.
 */
template <typename key_type_, typename assign_type_>
std::size_t dense_rank(std::size_t count, key_type_ const &key, assign_type_ const &assign) noexcept {
    for (std::size_t i = 0; i < count; ++i) {
        std::size_t rank = 0;
        for (std::size_t j = 0; j < count; ++j)
            if (key(j) < key(i)) {
                bool counted = false;
                for (std::size_t k = 0; k < j; ++k)
                    if (key(k) == key(j)) counted = true;
                if (!counted) rank += 1;
            }
        assign(i, rank);
    }
    std::size_t distinct = 0;
    for (std::size_t i = 0; i < count; ++i) {
        bool seen = false;
        for (std::size_t j = 0; j < i; ++j)
            if (key(j) == key(i)) seen = true;
        if (!seen) distinct += 1;
    }
    return distinct ? distinct : 1;
}

/**
 *  @brief A "prong" - is a tip of a "fork" - pinning "task" to a "thread".
 */
template <typename index_type_ = std::size_t>
struct prong {
    using index_t = index_type_;
    using task_index_t = index_t;   // ? A.k.a. "task index" in [0, prongs_count)
    using thread_index_t = index_t; // ? A.k.a. "core index" or "thread ID" in [0, threads_count)

    /** @brief The task index, in [0, prongs_count). */
    task_index_t task {0};
    /** @brief The thread (core) index running the task, in [0, threads_count). */
    thread_index_t thread {0};

    constexpr prong() noexcept = default;
    constexpr prong(prong &&) noexcept = default;
    constexpr prong(prong const &) noexcept = default;
    constexpr prong &operator=(prong const &) noexcept = default;
    constexpr prong &operator=(prong &&) noexcept = default;

    explicit prong(task_index_t task_index, thread_index_t thread_index) noexcept
        : task(task_index), thread(thread_index) {}

    inline operator task_index_t() const noexcept { return task; }
};

using prong_t = prong<>; // ? Default prong type with `std::size_t` indices

/**
 *  @brief A "prong" - is a tip of a "fork" - pinning "task" to a "thread" and "memory" location.
 */
template <typename index_type_ = std::size_t>
struct local_prong {
    using index_t = index_type_;
    using task_index_t = index_t;           // ? A.k.a. "task index" in [0, prongs_count)
    using thread_index_t = index_t;         // ? A.k.a. "core index" or "thread ID" in [0, threads_count)
    using compute_domain_index_t = index_t; // ? A.k.a. NUMA-specific QoS-specific "compute_domain ID"

    /** @brief The task index, in [0, prongs_count). */
    task_index_t task {0};
    /** @brief The thread (core) index running the task, in [0, threads_count). */
    thread_index_t thread {0};
    /** @brief The compute domain the thread is pinned to, in [0, compute_domains_count). */
    compute_domain_index_t compute_domain {0};

    constexpr local_prong() noexcept = default;
    constexpr local_prong(local_prong &&) noexcept = default;
    constexpr local_prong(local_prong const &) noexcept = default;
    constexpr local_prong &operator=(local_prong const &) noexcept = default;
    constexpr local_prong &operator=(local_prong &&) noexcept = default;

    explicit local_prong(task_index_t task_index, thread_index_t thread_index,
                         compute_domain_index_t compute_domain_index) noexcept
        : task(task_index), thread(thread_index), compute_domain(compute_domain_index) {}

    local_prong(prong<index_t> const &prong) noexcept : task(prong.task), thread(prong.thread), compute_domain(0) {}

    inline operator task_index_t() const noexcept { return task; }
    inline operator prong<index_t>() const noexcept { return prong<index_t> {task, thread}; }
};

using local_prong_t = local_prong<>; // ? Default prong type with `std::size_t` indices

/**
 *  @brief Describes a thread ID pinned to a specific compute domain.
 */
template <typename index_type_ = std::size_t>
struct local_thread {
    using index_t = index_type_;
    using thread_index_t = index_t;         // ? A.k.a. "core index" or "thread ID" in [0, threads_count)
    using compute_domain_index_t = index_t; // ? A.k.a. NUMA-specific QoS-specific "compute_domain ID"

    /** @brief The thread (core) index, in [0, threads_count). */
    thread_index_t thread {0};
    /** @brief The compute domain the thread is pinned to, in [0, compute_domains_count). */
    compute_domain_index_t compute_domain {0};

    constexpr local_thread() noexcept = default;
    constexpr local_thread(local_thread &&) noexcept = default;
    constexpr local_thread(local_thread const &) noexcept = default;
    constexpr local_thread &operator=(local_thread const &) noexcept = default;
    constexpr local_thread &operator=(local_thread &&) noexcept = default;

    local_thread(thread_index_t thread_index, compute_domain_index_t compute_domain_index = 0) noexcept
        : thread(thread_index), compute_domain(compute_domain_index) {}

    inline operator thread_index_t() const noexcept { return thread; }
};

using local_thread_t = local_thread<>; // ? Default thread-locator type with `std::size_t` indices

/**
 *  @brief Back-ports the C++ 23 `std::allocation_result`. Unlike STL, also contains the page size.
 *  @see https://en.cppreference.com/w/cpp/memory/allocator/allocate_at_least
 */
template <typename pointer_type_ = char, typename size_type_ = std::size_t>
struct allocation_result {
    using pointer_type = pointer_type_;
    using size_type = size_type_;

    /** @brief Pointer to the allocated memory, or nullptr if allocation failed. */
    pointer_type ptr {nullptr};
    /** @brief Number of elements allocated, or 0 if allocation failed. */
    size_type count {0};
    /** @brief Total volume of memory allocated, in bytes. */
    size_type bytes {0};
    /** @brief Number of memory pages allocated. */
    size_type pages {0};

    constexpr allocation_result() noexcept = default;
    constexpr allocation_result(pointer_type ptr_address, size_type count_index, size_type bytes_index,
                                size_type pages_index) noexcept
        : ptr(ptr_address), count(count_index), bytes(bytes_index), pages(pages_index) {}

    explicit constexpr operator bool() const noexcept { return ptr != nullptr && count > 0; }

    size_type bytes_per_page() const noexcept { return bytes / pages; }

    /**
     *  The standard says, that `std::allocation_result` must have 2 template arguments:
     *  pointer type and size type. Clang until version 19 disagrees and results in a
     *  compilation error, so we use some ugly SFINAE to detect which form is available.
     *
     *  `_LIBCPP_VERSION` is encoded  as (MAJOR * 10000 + MINOR * 100 + PATCH).
     *  @see https://github.com/llvm/llvm-project/blob/main/libcxx/include/__config
     */
#if defined(__cpp_lib_allocate_at_least)
#if defined(_LIBCPP_VERSION) && _LIBCPP_VERSION < 190000
    operator std::allocation_result<pointer_type>() const noexcept {
        return std::allocation_result<pointer_type> {ptr, static_cast<std::size_t>(count)};
    }
#else
    operator std::allocation_result<pointer_type, size_type>() const noexcept {
        return std::allocation_result<pointer_type, size_type>(ptr, count);
    }
#endif
#endif
};

/**
 *  @brief Result of a @b symmetric allocation - one mapping whose equal-stride slices sit on each domain.
 *  @tparam value_type_ The element type; the base pointer is `value_type_ *`.
 *
 *  Slice @b `d` begins at the byte address `ptr + d * stride_bytes` - see `slice` - and holds `count`
 *  elements. Every slice is a uniform distance `stride_bytes` apart - the CPU analog of a GPU symmetric
 *  heap. The stride is in bytes, not elements, because it is page-aligned and a page rarely divides
 *  `sizeof(value_type)`.
 */
template <typename value_type_, typename size_type_ = std::size_t>
struct symmetric_allocation_result {
    using value_type = value_type_;
    using pointer_type = value_type_ *;
    using size_type = size_type_;

    /** @brief Base of the mapping, or nullptr on failure. */
    pointer_type ptr {nullptr};
    /** @brief Usable elements per domain slice - what was requested. */
    size_type count {0};
    /** @brief Page-aligned @b byte distance between consecutive slice bases; `>= count * sizeof(value_type)`. */
    size_type stride_bytes {0};
    /** @brief Number of domain slices. */
    size_type domains {0};
    /** @brief Total mapped volume in bytes - `domains * stride_bytes`. */
    size_type bytes {0};
    /** @brief Number of memory pages mapped across all slices. */
    size_type pages {0};

    explicit constexpr operator bool() const noexcept { return ptr != nullptr && count > 0; }
    size_type bytes_per_page() const noexcept { return pages ? bytes / pages : 0; }

    /** @brief Base of the slice on memory domain @p domain_index, at `ptr + domain_index * stride_bytes`. */
    pointer_type slice(size_type domain_index) const noexcept {
        return reinterpret_cast<pointer_type>(reinterpret_cast<char *>(ptr) + domain_index * stride_bytes);
    }
};

/**
 *  @brief Detects allocators exposing our @b sized `allocate_at_least`, reporting `bytes` and `pages`.
 *
 *  Deliberately keys on the `bytes` member rather than on the function name. C++ 23 gave
 *  `std::allocator` an `allocate_at_least` of its own, but its `std::allocation_result` carries only
 *  `ptr` and `count`, so probing the name alone would match it and then fail to compile on `bytes`.
 */
template <typename allocator_type_, typename = void>
struct has_sized_allocate_at_least : std::false_type {};

template <typename allocator_type_>
struct has_sized_allocate_at_least<
    allocator_type_, std::void_t<decltype(std::declval<allocator_type_ &>().allocate_at_least(std::size_t {}).bytes)>>
    : std::true_type {};

/**
 *  @brief A fixed-capacity array with inline storage, so it never allocates.
 *  @sa `dynamic_array` when the count is only known at runtime.
 *
 *  Sized for the small, bounded lists a machine hands us - the huge page sizes of a NUMA node, the
 *  processor groups of a Windows box. Refuses to grow past `capacity_k` rather than truncating in
 *  silence, because a list quietly cut short is a topology quietly misreported.
 */
template <typename value_type_, std::size_t capacity_>
class limited_array {
    static_assert(std::is_nothrow_default_constructible_v<value_type_>,
                  "limited_array requires noexcept-default-constructible values");

    using value_t = value_type_;
    /** @brief Inline storage for up to `capacity_` values. */
    std::array<value_t, capacity_> values_ {};
    /** @brief Number of values currently stored, in [0, capacity_]. */
    std::size_t size_ {0};

  public:
    static constexpr std::size_t capacity_k = capacity_;

    constexpr limited_array() noexcept = default;

    /** @retval false when already at capacity; the value is not stored. */
    bool try_push_back(value_t const &value) noexcept {
        if (size_ == capacity_k) return false;
        values_[size_++] = value;
        return true;
    }

    void clear() noexcept { size_ = 0; }
    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }
    bool full() const noexcept { return size_ == capacity_k; }

    value_t &operator[](std::size_t i) noexcept { return values_[i]; }
    value_t const &operator[](std::size_t i) const noexcept { return values_[i]; }
    value_t *begin() noexcept { return values_.data(); }
    value_t *end() noexcept { return values_.data() + size_; }
    value_t const *begin() const noexcept { return values_.data(); }
    value_t const *end() const noexcept { return values_.data() + size_; }
    value_t *data() noexcept { return values_.data(); }
    value_t const *data() const noexcept { return values_.data(); }
};

/**
 *  @brief An owning, allocator-aware array whose size is fixed once, at `try_resize`.
 *  @sa `limited_array` for bounded counts, `dynamic_padded_array` when each element wants its own line.
 */
template <typename value_type_, typename allocator_type_ = std::allocator<value_type_>>
class dynamic_array {
    static_assert(std::is_nothrow_default_constructible_v<value_type_>,
                  "dynamic_array requires noexcept-default-constructible values");
    static_assert(std::is_nothrow_destructible_v<value_type_>, "dynamic_array requires noexcept-destructible values");

    using value_t = value_type_;
    using allocator_t = typename std::allocator_traits<allocator_type_>::template rebind_alloc<value_t>;

    /** @brief Allocator used to acquire and release the heap block. */
    allocator_t allocator_ {};
    /** @brief Pointer to the heap block, or nullptr when empty. */
    value_t *data_ {nullptr};
    /** @brief Number of live elements. */
    std::size_t size_ {0};
    /** @brief Allocated element slots; `>= size_`, doubled by `try_push_back` when full. */
    std::size_t capacity_ {0};

    void destroy_all() noexcept {
        if constexpr (!std::is_trivially_destructible_v<value_t>)
            for (std::size_t i = 0; i < size_; ++i) data_[i].~value_t();
    }

  public:
    using value_type = value_t;

    constexpr dynamic_array() noexcept = default;
    explicit dynamic_array(allocator_type_ const &allocator) noexcept : allocator_(allocator) {}

    dynamic_array(dynamic_array &&other) noexcept
        : allocator_(std::move(other.allocator_)), data_(std::exchange(other.data_, nullptr)),
          size_(std::exchange(other.size_, 0)), capacity_(std::exchange(other.capacity_, 0)) {}

    dynamic_array &operator=(dynamic_array &&other) noexcept {
        if (this != &other) {
            reset();
            allocator_ = std::move(other.allocator_);
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
            capacity_ = std::exchange(other.capacity_, 0);
        }
        return *this;
    }

    dynamic_array(dynamic_array const &) = delete;
    dynamic_array &operator=(dynamic_array const &) = delete;
    ~dynamic_array() noexcept { reset(); }

    void reset() noexcept {
        if (data_) {
            destroy_all();
            allocator_.deallocate(data_, capacity_);
            data_ = nullptr;
        }
        size_ = 0;
        capacity_ = 0;
    }

    /** @brief Reallocates to exactly @p new_size value-initialized elements, discarding any prior contents.
     *  @retval false on allocation failure, leaving the array empty rather than half-built. */
    bool try_resize(std::size_t const new_size) noexcept {
        reset();
        if (new_size == 0) return true;
        value_t *fresh = allocator_.allocate(new_size);
        if (!fresh) return false;
        // Value-initialization of a trivial type is a zero-fill; say so, rather than trusting the
        // optimizer to turn a placement-new loop back into one.
        if constexpr (std::is_trivially_default_constructible_v<value_t>)
            std::memset(fresh, 0, new_size * sizeof(value_t));
        else
            for (std::size_t i = 0; i < new_size; ++i) ::new (static_cast<void *>(fresh + i)) value_t();
        data_ = fresh;
        size_ = new_size;
        capacity_ = new_size;
        return true;
    }

    /** @brief Like `try_resize`, but skips the zero-fill so the caller controls the first touch.
     *  @note Trivial value types only - nothing is constructed, so every element must be written
     *        before it is read. @sa `sharded_array::try_resize_uninitialized`, the same contract. */
    bool try_resize_uninitialized(std::size_t const new_size) noexcept {
        static_assert(std::is_trivially_default_constructible_v<value_t> && std::is_trivially_destructible_v<value_t>,
                      "Uninitialized storage is only safe for trivial value types");
        reset();
        if (new_size == 0) return true;
        value_t *fresh = allocator_.allocate(new_size);
        if (!fresh) return false;
        data_ = fresh;
        size_ = new_size;
        capacity_ = new_size;
        return true;
    }

    /** @brief Grows capacity to at least @p new_capacity, preserving the live elements. */
    bool try_reserve(std::size_t const new_capacity) noexcept {
        static_assert(std::is_trivially_copyable_v<value_t> || std::is_nothrow_move_constructible_v<value_t>,
                      "try_reserve moves elements; the value type must be trivially copyable or nothrow-movable");
        if (new_capacity <= capacity_) return true;
        value_t *fresh = allocator_.allocate(new_capacity);
        if (!fresh) return false; // ! Allocation failed; the array is untouched
        if (size_ != 0) {
            if constexpr (std::is_trivially_copyable_v<value_t>) { std::memcpy(fresh, data_, size_ * sizeof(value_t)); }
            else
                for (std::size_t i = 0; i < size_; ++i) {
                    ::new (static_cast<void *>(fresh + i)) value_t(std::move(data_[i]));
                    data_[i].~value_t();
                }
        }
        if (data_) allocator_.deallocate(data_, capacity_);
        data_ = fresh;
        capacity_ = new_capacity;
        return true;
    }

    /** @brief Appends @p value, doubling capacity when full. @retval false on allocation failure. */
    bool try_push_back(value_t const &value) noexcept {
        static_assert(std::is_nothrow_copy_constructible_v<value_t>,
                      "try_push_back copies the value; the value type must be nothrow-copy-constructible");
        if (size_ == capacity_ && !try_reserve(capacity_ ? capacity_ * 2 : 4)) return false;
        ::new (static_cast<void *>(data_ + size_)) value_t(value);
        ++size_;
        return true;
    }

    std::size_t capacity() const noexcept { return capacity_; }

    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }
    value_t *data() noexcept { return data_; }
    value_t const *data() const noexcept { return data_; }
    value_t &operator[](std::size_t i) noexcept { return data_[i]; }
    value_t const &operator[](std::size_t i) const noexcept { return data_[i]; }
    value_t *begin() noexcept { return data_; }
    value_t *end() noexcept { return data_ + size_; }
    value_t const *begin() const noexcept { return data_; }
    value_t const *end() const noexcept { return data_ + size_; }
    explicit operator bool() const noexcept { return data_ != nullptr; }
};

/**
 *  @brief A `dynamic_array` whose elements sit at a caller-chosen stride, each on its own cache line.
 *  @sa `dynamic_array` for the packed counterpart at natural alignment.
 *
 *  Move-only and owning like `std::unique_ptr<T[]>`, but it spaces objects by `stride()` bytes rather
 *  than `sizeof(T)` and honours over-alignment, so an `alignas(128)` pool cell never false-shares and
 *  never lands in under-aligned storage. It also accepts an `allocate_at_least` allocator, the way the
 *  NUMA backends hand back more bytes than asked.
 */
template <typename object_type_, typename allocator_type_>
class dynamic_padded_array {

    using object_t = object_type_;
    static_assert(std::is_nothrow_default_constructible_v<object_t>,
                  "dynamic_padded_array requires noexcept-default-constructible object type");

    using allocator_t = allocator_type_;
    using allocator_traits_t = std::allocator_traits<allocator_t>;
    using raw_allocator_t = typename allocator_traits_t::template rebind_alloc<char>;

    /** @brief Aligned base the objects live at. */
    char *raw_ {nullptr};
    /** @brief What the allocator actually handed us, and what we must give back. */
    char *raw_owned_ {nullptr};
    /** @brief Number of objects currently held. */
    std::size_t objects_count_ {0};
    /** @brief Stride between consecutive objects, in bytes; at least `sizeof(object_t)`. */
    std::size_t bytes_per_object_ {sizeof(object_t)};
    /** @brief Total bytes owned, and what we must free. */
    std::size_t bytes_total_ {0};
    /** @brief Raw byte allocator that backs the buffer. */
    raw_allocator_t allocator_ {};

    object_t *ptr(std::size_t i) noexcept { return reinterpret_cast<object_t *>(raw_ + i * bytes_per_object_); }
    object_t const *ptr(std::size_t i) const noexcept {
        return reinterpret_cast<object_t const *>(raw_ + i * bytes_per_object_);
    }

    void destroy_all() noexcept {
        if constexpr (!std::is_trivially_destructible_v<object_t>)
            for (std::size_t i = 0; i < objects_count_; ++i) ptr(i)->~object_t();
    }

    void deallocate() noexcept {
        if (raw_owned_) {
            allocator_.deallocate(raw_owned_, bytes_total_);
            raw_owned_ = nullptr;
            raw_ = nullptr;
        }
        objects_count_ = bytes_total_ = 0;
    }

  public:
    dynamic_padded_array() noexcept = default;

    explicit dynamic_padded_array(allocator_t const &alloc, std::size_t bytes_per_object = sizeof(object_t)) noexcept
        : bytes_per_object_(bytes_per_object), allocator_(alloc) {}

    dynamic_padded_array(dynamic_padded_array &&o) noexcept
        : raw_(std::exchange(o.raw_, nullptr)), raw_owned_(std::exchange(o.raw_owned_, nullptr)),
          objects_count_(std::exchange(o.objects_count_, 0)), bytes_per_object_(o.bytes_per_object_),
          bytes_total_(std::exchange(o.bytes_total_, 0)), allocator_(std::move(o.allocator_)) {}

    dynamic_padded_array &operator=(dynamic_padded_array &&o) noexcept {
        if (this != &o) {
            destroy_all();
            deallocate();
            raw_ = std::exchange(o.raw_, nullptr);
            raw_owned_ = std::exchange(o.raw_owned_, nullptr);
            objects_count_ = std::exchange(o.objects_count_, 0);
            bytes_per_object_ = o.bytes_per_object_;
            bytes_total_ = std::exchange(o.bytes_total_, 0);
            allocator_ = std::move(o.allocator_);
        }
        return *this;
    }

    dynamic_padded_array(dynamic_padded_array const &) = delete;
    dynamic_padded_array &operator=(dynamic_padded_array const &) = delete;

    ~dynamic_padded_array() noexcept {
        destroy_all();
        deallocate();
    }

    bool try_resize(std::size_t new_objects_count) noexcept {
        destroy_all();
        deallocate();

        if (new_objects_count == 0) return true;

        // An `alignas(128)` object placement-newed into 16-byte-aligned storage is undefined, and it
        // is exactly what happens when the object is a sub-pool and the allocator is `std::allocator`.
        // `linux_numa_allocator` hands back page-aligned memory and hides the bug; Apple's does not.
        constexpr std::size_t object_alignment_k = alignof(object_t);
        constexpr bool over_aligned_k = object_alignment_k > alignof(std::max_align_t);
        std::size_t const slack = over_aligned_k ? object_alignment_k - 1 : 0;
        std::size_t const total = new_objects_count * bytes_per_object_ + slack;

        // NUMA-aware allocators can hand back more than we asked for, and tell us how much. Plain
        // `std::allocator` cannot, so take exactly `total` and remember that as the size to free.
        char *raw = nullptr;
        std::size_t bytes = 0;
        if constexpr (has_sized_allocate_at_least<raw_allocator_t>::value) {
            auto new_result = allocator_.allocate_at_least(total);
            if (!new_result) return false;
            raw = new_result.ptr;
            bytes = new_result.bytes;
        }
        else {
            raw = allocator_.allocate(total);
            if (!raw) return false;
            bytes = total;
        }

        raw_owned_ = raw;
        if constexpr (over_aligned_k) {
            auto const address = reinterpret_cast<std::uintptr_t>(raw);
            auto const aligned =
                (address + object_alignment_k - 1) & ~static_cast<std::uintptr_t>(object_alignment_k - 1);
            raw = reinterpret_cast<char *>(aligned);
        }
        raw_ = raw;
        objects_count_ = new_objects_count;
        bytes_total_ = bytes;

        for (std::size_t i = 0; i < objects_count_; ++i) ::new (static_cast<void *>(ptr(i))) object_t();

        return true;
    }

    object_t &operator[](std::size_t i) noexcept { return *ptr(i); }
    object_t const &operator[](std::size_t i) const noexcept { return *ptr(i); }
    object_t *data() noexcept { return ptr(0); }
    object_t const *data() const noexcept { return ptr(0); }
    std::size_t size() const noexcept { return objects_count_; }
    std::size_t stride() const noexcept { return bytes_per_object_; }
    void set_stride(std::size_t b) noexcept { bytes_per_object_ = b ? b : sizeof(object_t); }
    explicit operator bool() const noexcept { return raw_ != nullptr && objects_count_ > 0; }
};

/**
 *  @brief Placeholder type for Parallel Algorithms.
 */
struct dummy_lambda_t {};

/**
 *  @brief Compile-time tag: a monitored wait bounded by a timeout, so it re-checks on its own.
 *  @sa `wait_uncapped_t`
 *
 *  For a loop guarding more than one word: the monitor arms a single line, so a store to another word
 *  goes unseen and the cap bounds how late it is noticed. Selects the timed instruction - `WFET`,
 *  `WRS.STO`. A spin waiter ignores the tag.
 */
struct wait_capped_t {};

/**
 *  @brief Compile-time tag: a monitored wait with no timeout, woken only by the store it watches.
 *  @sa `wait_capped_t`
 *
 *  For a loop guarding a single word: the monitor covers every wake source, so a timeout would only
 *  wake the core to learn nothing and waste power. Selects the untimed instruction - `WFE`, `WRS.NTO`.
 */
struct wait_uncapped_t {};

/** @brief The canonical `wait_capped_t` value to pass as a wait tag. */
inline constexpr wait_capped_t wait_capped_k {};
/** @brief The canonical `wait_uncapped_t` value to pass as a wait tag. */
inline constexpr wait_uncapped_t wait_uncapped_k {};

/**
 *  @brief The portable busy-wait: hands the core back to the scheduler. Works everywhere, cheap nowhere.
 *
 *  A spin waiter ignores the watched word - the caller's loop already re-checks its own condition, so
 *  this only needs to emit one backoff hint per turn. The monitored waiters (`arm64_wfet_t`,
 *  `x86_tpause_t`, `risc5_wrs_t`) use the word to sleep the core until that line changes.
 */
struct standard_yield_t {
    static constexpr capabilities_t capability_k = capabilities_unknown_k;
    template <typename value_type_, typename thread_index_type_, typename bound_type_ = wait_capped_t>
    inline void operator()(std::atomic<value_type_> const &, value_type_, thread_index_type_,
                           bound_type_ = {}) const noexcept {
        std::this_thread::yield();
    }
};

/**
 *  @brief Whether @p yield_type_ is a valid waiter: callable as `yield(watched, observed, thread_index)`.
 *
 *  Every waiter takes the atomic being watched, the value last seen, and the calling thread's index.
 *  A spin functor ignores the first two; a monitored functor arms a hardware address monitor on the
 *  watched line and sleeps the core until it changes. The thread index is carried for a future
 *  adaptive backoff, and dropped by every waiter today.
 */
template <typename yield_type_, typename value_type_, typename thread_index_type_>
struct is_wait_functor {
    static constexpr bool value =
        std::is_nothrow_invocable_v<yield_type_ &, std::atomic<value_type_> const &, value_type_, thread_index_type_>;
};

/** @brief Tag for pushing a just-written line away, toward the LLC or the coherency point. */
struct demote_line_t {};
/** @brief Tag for pulling a line toward this core with write intent, ahead of an atomic claim. */
struct promote_line_t {};
/** @brief Canonical `demote_line_t` value, mirroring the `wait_capped_k` tag convention. */
inline constexpr demote_line_t demote_line_k {};
/** @brief Canonical `promote_line_t` value, mirroring the `wait_uncapped_k` tag convention. */
inline constexpr promote_line_t promote_line_k {};

/**
 *  @brief The do-nothing cache-hints policy - the default, and the fallback for every ISA gap.
 *  @sa `preferred_cache_hints_t` in `capabilities.hpp`, which picks the per-ISA emitters where
 *      `FU_WITH_DEMOTE_CACHE_LINES` / `FU_WITH_PROMOTE_CACHE_LINES` hold.
 */
struct standard_cache_hints_t {
    static constexpr capabilities_t capability_k = capabilities_unknown_k;
    inline void operator()(void const *, demote_line_t) const noexcept {}
    inline void operator()(void const *, promote_line_t) const noexcept {}
};

/**
 *  @brief Whether @p hints_type_ is a valid cache-hints policy: callable with both line tags.
 *
 *  Every policy takes the address of the line being handed away or claimed, plus a tag choosing the
 *  direction. A policy for an ISA with no matching instruction implements the overload as a no-op,
 *  so callers never branch - the emptiness compiles away.
 */
template <typename hints_type_>
struct is_cache_hints_functor {
    static constexpr bool value = std::is_nothrow_invocable_v<hints_type_ &, void const *, demote_line_t> &&
                                  std::is_nothrow_invocable_v<hints_type_ &, void const *, promote_line_t>;
};

/**
 *  @brief A trivial minimalistic lock-free "mutex" implementation over a single `std::atomic<bool>`.
 *  @tparam micro_yield_type_ The type of the waiter to be used for busy-waiting.
 *  @tparam alignment_ The alignment of the mutex. Defaults to `default_alignment_k`.
 *
 *  The C++ standard would recommend using `std::hardware_destructive_interference_size`
 *  alignment, as well as `std::atomic_flag::notify_one` and `std::this_thread::yield` APIs,
 *  but our solution is better despite being more primitive.
 *
 *  A `std::atomic<bool>` is used rather than `std::atomic_flag` so the flag has an @b address a
 *  monitored waiter can arm: the lock loop is test-and-test-and-set, spinning on a plain load, so a
 *  monitored `micro_yield` sleeps the core until `unlock`'s store to that line wakes it. On every
 *  platform we target `std::atomic<bool>` is lock-free, so the `atomic_flag` guarantee buys nothing.
 *
 *  @see Compatible with STL unique locks: https://en.cppreference.com/w/cpp/thread/unique_lock.html
 */
template <typename micro_yield_type_ = standard_yield_t, std::size_t alignment_ = default_alignment_k>
class spin_mutex {
    using micro_yield_t = micro_yield_type_;
    static constexpr std::size_t alignment_k = alignment_;
    alignas(alignment_k) std::atomic<bool> flag_ {false};

  public:
    void lock() noexcept {
        micro_yield_t micro_yield;
        while (true) {
            // Claim the lock with the only store in the loop, so a monitored waiter is woken once per
            // `unlock` rather than by every contender's attempt.
            if (!flag_.exchange(true, std::memory_order_acquire)) return;
            // Contended: spin on a non-writing load until the line looks free, sleeping the core meanwhile.
            while (flag_.load(std::memory_order_relaxed)) micro_yield(flag_, true, static_cast<std::size_t>(0));
        }
    }
    bool try_lock() noexcept { return !flag_.exchange(true, std::memory_order_acquire); }
    void unlock() noexcept { flag_.store(false, std::memory_order_release); }
};

using spin_mutex_t = spin_mutex<>;

/** @brief A half-open slice `[first, first + count)` of a task index space. */
template <typename index_type_ = std::size_t>
struct indexed_range {
    using index_t = index_type_;

    /** @brief The first task index in the slice. */
    index_t first {0};
    /** @brief How many tasks the slice covers; zero means an empty slice. */
    index_t count {0};
};

using indexed_range_t = indexed_range<>;

/**
 *  @brief Splits a range of tasks into fair-sized chunks for each thread.
 *  @see https://lemire.me/blog/2025/05/22/dividing-an-array-into-fair-sized-chunks/
 *
 *  The first `(tasks % threads)` chunks have size `ceil(tasks / threads)`.
 *  The remaining `tasks - (tasks % threads)` chunks have size `floor(tasks / threads)`
 *  Has the convenient added property that the difference between the largest and smallest
 *  chunk size is at most 1, which can be used in some ordering algorithms.
 */
template <typename index_type_ = std::size_t>
struct indexed_split {
    using index_t = index_type_;
    using indexed_range_t = indexed_range<index_t>;

  private:
    /** @brief Floor of tasks divided by threads; the smaller chunk size. */
    index_t quotient_ {0};
    /** @brief Tasks left over; the first `remainder_` chunks get one extra task. */
    index_t remainder_ {0};

  public:
    inline indexed_split() noexcept = default;

    /**
     *  @brief Constructs an indexed split for a given number of tasks and threads.
     *  @param[in] tasks_count The total number of tasks to split; can be any unsigned integer.
     *  @param[in] threads_count The number of threads to split the tasks into; can't be zero.
     */
    inline indexed_split(index_t const tasks_count, index_t const threads_count) noexcept
        : quotient_(tasks_count / threads_count), remainder_(tasks_count % threads_count) {
        assert(threads_count > 0 && "Threads count must be greater than zero, or expect division by zero");
    }

    inline indexed_range_t operator[](index_t const i) const noexcept {
        index_t const begin = static_cast<index_t>(quotient_ * i + (i < remainder_ ? i : remainder_));
        index_t const count = static_cast<index_t>(quotient_ + (i < remainder_ ? 1 : 0));
        return {begin, count};
    }

    inline index_t smallest_size() const noexcept { return quotient_; }
    inline index_t largest_size() const noexcept { return quotient_ + (remainder_ > 0); }

    /**
     *  @brief The chunk owning task @p task - the inverse of `operator[]`, in closed form.
     *  @note The first `remainder_` chunks are one task larger, so the boundary between the two
     *        regimes sits at `remainder_ * (quotient_ + 1)`; a `quotient_` of zero puts every valid
     *        task in the first regime, so the division by `quotient_` below is never reached.
     */
    inline index_t index_of(index_t const task) const noexcept {
        index_t const larger_chunks_end = static_cast<index_t>(remainder_ * (quotient_ + 1));
        if (task < larger_chunks_end) return static_cast<index_t>(task / (quotient_ + 1));
        return static_cast<index_t>(remainder_ + (task - larger_chunks_end) / quotient_);
    }
};

using indexed_split_t = indexed_split<>;

/**
 *  @brief Pre-C++20 sentinel type for iterators.
 *  @see   https://en.cppreference.com/w/cpp/iterator/default_sentinel.html
 */
struct default_sentinel_t {};

/**
 *  @brief Iterator range over integers using a stride that is co-prime with length.
 *
 *  - O(1) dereference: two integer ops and a branchless wrap-around.
 *  - Every value appears exactly once before `end()`.
 *
 *  @code{.cpp}
 *  coprime_permutation_range<> perm(start, length, seed);
 *  for (auto v : perm) steal_from(v);
 *  @endcode
 */
template <typename index_type_ = std::size_t>
struct coprime_permutation_range {
    using index_t = index_type_;

  private:
    /** @brief First value of the domain. */
    index_t start_ {0};
    /** @brief Size of the domain being permuted. */
    index_t length_ {1};
    /** @brief Co-prime step between consecutive values. */
    index_t stride_ {1};
    /** @brief Where this seed's walk begins, in [0, length_). */
    index_t first_offset_ {0};

  public:
    struct iterator {
        using iterator_category = std::forward_iterator_tag;
        using value_type = index_t;
        using difference_type = std::ptrdiff_t;
        using pointer = void;
        using reference = value_type;

        inline value_type operator*() const noexcept { return static_cast<index_t>(start_ + offset_); }

        inline iterator &operator++() noexcept {
            assert(elements_left_ != 0 && "Attempting to increment an iterator beyond bounds");

            // Avoid modulo division by using wrap-around logic. Both `offset_` and `stride_` are below
            // `length_`, but their @b sum need not fit `index_t` - on `std::uint8_t` with a length of
            // 253, `200 + 100` truncates to 44 rather than wrapping to 47, and the walk stops being a
            // permutation. Subtracting first keeps every intermediate value inside the domain.
            index_t const room_left = static_cast<index_t>(length_ - offset_); // ? Always positive
            offset_ = stride_ < room_left ? static_cast<index_t>(offset_ + stride_)
                                          : static_cast<index_t>(stride_ - room_left);
            --elements_left_;
            return *this;
        }

        inline iterator operator++(int) noexcept {
            iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        inline bool operator==(default_sentinel_t) const noexcept { return elements_left_ == 0; }
        inline bool operator!=(default_sentinel_t s) const noexcept { return !(*this == s); }

      private:
        friend struct coprime_permutation_range;

        inline iterator(index_t const start, index_t const length, index_t const stride, index_t const first_offset,
                        index_t const elements_left) noexcept
            : start_(start), length_(length), stride_(stride), offset_(first_offset), elements_left_(elements_left) {}

        /** @brief First value of the domain. */
        index_t start_ {0};
        /** @brief Size of the domain being permuted. */
        index_t length_ {1};
        /** @brief Co-prime step between consecutive values. */
        index_t stride_ {1};
        /** @brief Current offset into the domain, in [0, length_). */
        index_t offset_ {0};
        /** @brief Countdown of values remaining until `end()`. */
        index_t elements_left_ {0};
    };

    coprime_permutation_range() noexcept = default;

    /**
     *  @param[in] start First element of the permutation.
     *  @param[in] length Size of the domain to permute; must be > 0.
     *  @param[in] seed Thread-specific value used to derive a unique stride.
     */
    coprime_permutation_range(index_t const start, index_t const length, index_t const seed) noexcept
        : start_(start), length_(length), stride_(pick_stride(seed, length_)),
          first_offset_(static_cast<index_t>(seed % length)) {
        assert(length_ > 0 && "Length must be greater than zero, or expect division by zero");
    }

    /**
     *  @note The seed shifts where the walk @b starts, not only how it steps. Deriving the stride
     *        alone would leave every seed emitting the same first value, so a pool of drained threads
     *        would descend on that one victim together before their strides pulled them apart.
     */
    iterator begin() const noexcept { return iterator(start_, length_, stride_, first_offset_, length_); }
    default_sentinel_t end() const noexcept { return {}; }
    index_t size() const noexcept { return length_; }

  private:
    static constexpr index_t gcd(index_t a, index_t b) noexcept {
        while (b) {
            index_t const t = a % b;
            a = b;
            b = t;
        }
        return a;
    }

    static index_t pick_stride(index_t seed, index_t const length) noexcept {
        // Pick an odd stride derived from @p seed that is co-prime with @p length.
        if (length <= 1) return 0;                              // degenerate case
        seed = static_cast<index_t>((seed * 2u + 1u) % length); // force odd
        while (gcd(seed, length) != 1) {                        // insure co-prime
            seed += 2u;
            if (seed >= length) seed -= length;
        }
        return seed;
    }
};

using coprime_permutation_range_t = coprime_permutation_range<>;

/** @brief Wraps the metadata needed for `for_slices` APIs for `broadcast_join` compatibility. */
template <typename fork_type_, typename index_type_>
class invoke_for_slices {
    fork_type_ fork_;
    indexed_split<index_type_> split_;

  public:
    invoke_for_slices(index_type_ n, index_type_ threads, fork_type_ &&fork) noexcept
        : fork_(std::forward<fork_type_>(fork)), split_(n, threads) {}

    void operator()(index_type_ const thread) const noexcept {
        indexed_range<index_type_> const range = split_[thread];
        if (range.count == 0) return; // ? No work for this thread
        fork_(prong<index_type_> {range.first, thread}, range.count);
    }
};

/** @brief Wraps the metadata needed for `for_n` APIs for `broadcast_join` compatibility. */
template <typename fork_type_, typename index_type_>
class invoke_for_n {
    fork_type_ fork_;
    indexed_split<index_type_> split_;

  public:
    invoke_for_n(index_type_ n, index_type_ threads, fork_type_ &&fork) noexcept
        : fork_(std::forward<fork_type_>(fork)), split_(n, threads) {}

    void operator()(index_type_ const thread) const noexcept {
        indexed_range<index_type_> const range = split_[thread];
        for (index_type_ i = 0; i < range.count; ++i)
            fork_(prong<index_type_> {static_cast<index_type_>(range.first + i), thread});
    }
};

/**
 *  @brief One thread's private cursor into its own slice of a `for_n_dynamic` dispatch.
 *  @sa `invoke_for_n_dynamic` hands each thread a slice; idle threads drain their neighbours'.
 *
 *  A single shared counter serializes an entire dispatch: only one core may own its cache line at
 *  a time, so no dispatch retires tasks faster than that line circulates. Handing every thread its
 *  own cursor turns the common claim into an @b uncontended read-modify-write on a line nobody else
 *  touches, which is roughly fifty times cheaper. The line is only shared once a thread runs dry and
 *  starts helping a neighbour, which is exactly when the extra cost is worth paying.
 *
 *  Pad these to a full cache line - two cursors sharing a line would reintroduce the very traffic
 *  the split exists to avoid. @sa `dynamic_padded_array`, which spaces them by the pool's alignment.
 */
template <typename index_type_ = std::size_t>
struct dynamic_claim {
    /** @brief Next task in this slice; only ever grows, and may overshoot `end` by `threads`. */
    std::atomic<index_type_> next {0};
    /** @brief One past this slice's last task; written once before the dispatch, then read-only. */
    index_type_ end {0};
};

using dynamic_claim_t = dynamic_claim<>;

/**
 *  @brief Drains whatever is left of one thread's slice into @p fork - the shared core of both the
 *         flat and the distributed `for_n_dynamic` invokers, so its invariants live in one place.
 *
 *  A read-only probe first: a drained slice is skipped in Shared state - no dirtying add, no line
 *  migration - which is the entire cost of visiting an empty neighbour once a small dispatch runs
 *  dry. A live slice is promoted with write intent while the probe's branch resolves, claimed one
 *  task at a time - the makespan guarantee of greedy list scheduling - and demoted once our
 *  overshooting add has dirtied it, so the next visitor snoops the LLC instead of this core.
 */
template <typename index_type_, typename prong_type_, typename fork_type_, typename cache_hints_type_>
inline void drain_claim(dynamic_claim<index_type_> &claim, prong_type_ &prong, fork_type_ &fork,
                        cache_hints_type_ cache_hints) noexcept {
    if (claim.next.load(std::memory_order_relaxed) >= claim.end) return;
    cache_hints(&claim, promote_line_k); // ? Overlap the exclusive-ownership fetch with the branch
    while (true) {
        index_type_ const task = claim.next.fetch_add(1, std::memory_order_relaxed);
        if (task >= claim.end) break; // ? Overshoots by one, and only once per thread
        prong.task = task;
        fork(prong);
    }
    cache_hints(&claim, demote_line_k); // ? Our overshooting add left the line dirty; hand it away
}

/**
 *  @brief Drains whatever is left of the @p slice thread's claim in @p pool, whether or not we own it.
 *  @sa The `dynamic_claim` overload above, where the probe and the overshoot invariants live.
 */
template <typename pool_type_, typename index_type_, typename prong_type_, typename fork_type_>
inline void drain_claim(pool_type_ &pool, index_type_ const slice, prong_type_ &prong, fork_type_ &fork) noexcept {
    drain_claim(pool.unsafe_dynamic_claim_ref(slice), prong, fork, typename pool_type_::cache_hints_t {});
}

/**
 *  @brief Wraps the metadata needed for `for_n_dynamic` APIs for `broadcast_join` compatibility.
 *
 *  @section Scheduling Logic
 *
 *  Tasks are split into one contiguous slice per thread. A thread first drains its own slice, then
 *  walks the others in a `coprime_permutation_range` order and drains theirs, one task per claim.
 *  Claiming one task at a time is what preserves the makespan guarantee of greedy list scheduling:
 *  a thread can never be handed a batch of tasks that turn out to be expensive, because it is never
 *  handed a batch. Claiming from a @b private cursor is what makes that guarantee affordable.
 *
 *  Probing the neighbours in a coprime order rather than linearly keeps two drained threads from
 *  descending on the same victim, which would serialize them on one line for no reason.
 *  @sa `invoke_distributed_for_n_dynamic`, which applies the same trick one level up, across
 *  compute domains, so a thread exhausts local work before touching a remote node's memory.
 *
 *  @section Overflow Considerations
 *
 *  If we run a default for-loop at 1 Billion times per second on a 64-bit machine, then every 585 years
 *  of computational time we will wrap around the `std::size_t` capacity for the `prong.task` index.
 *  In case we `n + thread >= std::size_t(-1)`, a simple condition won't be enough.
 *  Alternatively, we can make sure, that each thread can do at least one increment of a cursor
 *  without worrying about the overflow. The way to achieve that is to preprocess the trailing `threads`
 *  of elements externally, before entering this loop!
 *
 *  That trailing reservation also bounds the cursors. Every thread visits a given slice exactly once -
 *  each thread's coprime walk is a permutation of all the slices, beginning with its own - and a visit
 *  overshoots by at most one increment, since `drain_claim` leaves the moment it reads `>= end` and its
 *  read-only probe skips already-drained slices without any increment at all. A cursor therefore never
 *  passes `end + threads`.
 *
 *  Two regimes bound that. When `n > threads` the last slice ends at `n - threads`, so no cursor passes
 *  `n`. When `n <= threads` every slice is empty and `end == 0`, so no cursor passes `threads` - which
 *  may exceed `n`, but is still an index the type must represent to have spawned the pool at all.
 *  Either way `max(cursor) == max(n, threads)`, and no index type can wrap.
 */
template <typename pool_type_, typename fork_type_, typename index_type_>
class invoke_for_n_dynamic {
    /** @brief The pool, owning one padded `dynamic_claim` per thread; we never allocate. */
    pool_type_ &pool_;
    /** @brief The per-task callback to invoke. */
    fork_type_ fork_;
    /** @brief Total number of tasks to dispatch. */
    index_type_ n_;
    /** @brief Number of worker threads sharing the dispatch. */
    index_type_ threads_;

    /** @brief Number of tasks handed out dynamically; the trailing `threads_` are static prongs. */
    index_type_ dynamic_count() const noexcept { return n_ > threads_ ? static_cast<index_type_>(n_ - threads_) : 0; }

  public:
    invoke_for_n_dynamic(pool_type_ &pool, index_type_ n, index_type_ threads, fork_type_ &&fork) noexcept
        : pool_(pool), fork_(std::forward<fork_type_>(fork)), n_(n), threads_(threads) {
        reset_slices_();
    }

    void operator()(index_type_ const thread) noexcept {

        // Run (up to) one static prong on the current thread
        index_type_ const n_dynamic = dynamic_count();
        index_type_ const one_static_prong_index = static_cast<index_type_>(n_dynamic + thread);
        prong<index_type_> prong(one_static_prong_index, thread);
        if (one_static_prong_index < n_) fork_(prong);

        // Help everyone, in a coprime order so drained threads don't collide on one victim. The walk
        // starts at our own slice - `first_offset_ = seed % length` with `seed = thread < threads_` -
        // so the uncontended line is drained first and no self-guard is needed.
        coprime_permutation_range<index_type_> victims(0, threads_, thread);
        for (auto victim = victims.begin(); victim != default_sentinel_t {}; ++victim)
            drain_claim(pool_, *victim, prong, fork_);
    }

  private:
    /** @brief Publishes one contiguous slice per thread. Runs on the caller, before the broadcast. */
    void reset_slices_() noexcept {
        typename pool_type_::cache_hints_t cache_hints;
        index_type_ const n_dynamic = dynamic_count();
        indexed_split<index_type_> const split(n_dynamic, threads_);
        for (index_type_ thread = 0; thread < threads_; ++thread) {
            indexed_range<index_type_> const range = split[thread];
            dynamic_claim<index_type_> &claim = pool_.unsafe_dynamic_claim_ref(thread);
            claim.end = static_cast<index_type_>(range.first + range.count);
            claim.next.store(range.first, std::memory_order_release);
            cache_hints(&claim, demote_line_k); // ? Publish away, so each owner's first claim skips this core
        }
    }
};

/**
 *  @brief A synchronization point that waits for all threads to finish the last fork.
 *  @note You don't have to explicitly call any of the APIs, it's like `std::jthread` ;)
 *
 *  The lifecycle is keyed on the pool's exclusivity:
 *  - On `caller_exclusive_k` pools the fork is dispatched at @b construction: the workers
 *    start immediately, the caller may overlap its own work, poll `is_complete`, and the
 *    `join` call (or the destructor) waits for completion.
 *  - On `caller_inclusive_k` pools the dispatch is deferred to @b join (or the destructor),
 *    where the calling thread contributes its own slice - a deferred blocking call.
 *
 *  You don't have to explicitly handle the return value and wait on it.
 *  According to the C++ standard, the destructor of the `broadcast_join` will
 *  be called in the end of the `for_threads`-calling expression.
 *
 *  The object is immovable: on caller-exclusive pools the pool holds a pointer to the
 *  `fork_` member for the lifetime of the broadcast, so the object must never relocate.
 *  Guaranteed copy elision (C++17) still allows returning it by value from `for_threads`.
 */
template <typename pool_type_, typename fork_type_>
struct broadcast_join {

    using pool_t = pool_type_;
    using fork_t = fork_type_;
    using generation_t = typename pool_t::generation_t;

  private:
    /** @brief The pool this broadcast dispatches onto and joins. */
    pool_t &pool_ref_;
    /** @brief The wrapped fork; held to extend the lifetime of the lambda object. */
    fork_t fork_;
    /** @brief Generation token of this broadcast; real tokens are odd, zero means "not yet dispatched". */
    generation_t generation_ {0};

  public:
    broadcast_join(pool_t &pool_ref, fork_t &&f) noexcept : pool_ref_(pool_ref), fork_(std::forward<fork_t>(f)) {
        if (pool_ref_.caller_exclusivity() == caller_exclusive_k) generation_ = pool_ref_.unsafe_for_threads(fork_);
    }

    /** @brief The wrapped fork; on caller-exclusive pools only read it after `join`. */
    fork_t &fork_ref() noexcept { return fork_; }

    /** @brief The generation token of this broadcast; always odd once dispatched, zero before. */
    generation_t generation() const noexcept { return generation_; }

    /** @brief Non-blocking check; can only turn `true` before `join` on caller-exclusive pools. */
    bool is_complete() const noexcept { return generation_ != 0 && pool_ref_.is_complete(generation_); }

    void join() noexcept {
        if (generation_ == 0) generation_ = pool_ref_.unsafe_for_threads(fork_);
        pool_ref_.unsafe_join(generation_); // ? Idempotent for already-joined generations
    }

    ~broadcast_join() noexcept { join(); }
    broadcast_join(broadcast_join &&) = delete;
    broadcast_join(broadcast_join const &) = delete;
    broadcast_join &operator=(broadcast_join &&) = delete;
    broadcast_join &operator=(broadcast_join const &) = delete;
};

template <typename fork_type_, typename index_type_ = std::size_t>
constexpr bool can_be_for_thread_callback() noexcept {
    using fork_t = fork_type_;
    using index_t = index_type_;
#if FU_DETECT_CPP_17_ && defined(__cpp_lib_is_invocable)
    return std::is_nothrow_invocable_r_v<void, fork_t, local_thread<index_t>> ||
           std::is_nothrow_invocable_r_v<void, fork_t, index_t>;
#else
    return true;
#endif
}

template <typename fork_type_, typename index_type_ = std::size_t>
constexpr bool can_be_for_task_callback() noexcept {
    using fork_t = fork_type_;
    using index_t = index_type_;
#if FU_DETECT_CPP_17_ && defined(__cpp_lib_is_invocable)
    return std::is_nothrow_invocable_r_v<void, fork_t, local_prong<index_t>> ||
           std::is_nothrow_invocable_r_v<void, fork_t, prong<index_t>> ||
           std::is_nothrow_invocable_r_v<void, fork_t, index_t>;
#else
    return true;
#endif
}

template <typename fork_type_, typename index_type_ = std::size_t>
constexpr bool can_be_for_slice_callback() noexcept {
    using fork_t = fork_type_;
    using index_t = index_type_;
#if FU_DETECT_CPP_17_ && defined(__cpp_lib_is_invocable)
    return std::is_nothrow_invocable_r_v<void, fork_t, local_prong<index_t>, index_t> ||
           std::is_nothrow_invocable_r_v<void, fork_t, prong<index_t>, index_t> ||
           std::is_nothrow_invocable_r_v<void, fork_t, index_t, index_t>;
#else
    return true;
#endif
}

#if FU_DETECT_CPP_20_ && defined(__cpp_concepts)
#define FU_DETECT_CONCEPTS_ 1
#define FU_REQUIRES_(condition) requires(condition)
#else
#define FU_DETECT_CONCEPTS_ 0
#define FU_REQUIRES_(condition)
#endif // FU_DETECT_CPP_20_

/**
 *  @brief A zero-setup thread-pool that runs every task on the calling thread.
 *
 *  A drop-in @b serial executor - it satisfies `is_pool` and `is_unsafe_pool` and offers the same
 *  scheduling surface as `flat_pool` - `for_threads`, `for_n`, `for_n_dynamic`, `for_slices` - but with
 *  one thread on one compute domain, no `try_spawn`, and no allocation. `unsafe_for_threads` runs the
 *  fork synchronously as thread 0; everything else composes through `broadcast_join` exactly as the real
 *  pools do. Useful as a serial baseline and as the default executor for the domain-aware containers.
 */
struct dummy_pool_t {
    using index_t = std::size_t;
    using thread_index_t = index_t;
    using compute_domain_index_t = index_t;
    using epoch_index_t = index_t;
    using generation_t = epoch_index_t;
    using prong_t = prong<index_t>;
    using indexed_split_t = indexed_split<index_t>;

    thread_index_t threads_count() const noexcept { return 1; }
    caller_exclusivity_t caller_exclusivity() const noexcept { return caller_inclusive_k; }
    index_t compute_domains_count() const noexcept { return 1; }
    thread_index_t threads_count(FU_MAYBE_UNUSED_ index_t compute_domain) const noexcept { return 1; }
    index_t thread_compute_domain(FU_MAYBE_UNUSED_ thread_index_t thread) const noexcept { return 0; }
    thread_index_t thread_local_index(FU_MAYBE_UNUSED_ thread_index_t thread,
                                      FU_MAYBE_UNUSED_ index_t compute_domain) const noexcept {
        return 0;
    }

    template <typename fork_type_>
    FU_REQUIRES_((can_be_for_thread_callback<fork_type_, index_t>()))
    generation_t unsafe_for_threads(fork_type_ &fork) noexcept {
        fork(thread_index_t {0});
        return 1; // ! Tokens are always odd; the work already ran
    }
    void unsafe_join(FU_MAYBE_UNUSED_ generation_t generation) noexcept {}
    void unsafe_join() noexcept {}
    bool is_complete(FU_MAYBE_UNUSED_ generation_t generation) const noexcept { return true; }

    template <typename fork_type_ = dummy_lambda_t>
    FU_REQUIRES_((can_be_for_thread_callback<fork_type_, index_t>()))
    broadcast_join<dummy_pool_t, fork_type_> for_threads(fork_type_ &&fork) noexcept {
        return {*this, std::forward<fork_type_>(fork)};
    }
    template <typename fork_type_ = dummy_lambda_t>
    FU_REQUIRES_((can_be_for_slice_callback<fork_type_, index_t>()))
    broadcast_join<dummy_pool_t, invoke_for_slices<fork_type_, index_t>> for_slices(index_t n,
                                                                                    fork_type_ &&fork) noexcept {
        return {*this, {n, threads_count(), std::forward<fork_type_>(fork)}};
    }
    template <typename fork_type_ = dummy_lambda_t>
    FU_REQUIRES_((can_be_for_task_callback<fork_type_, index_t>()))
    broadcast_join<dummy_pool_t, invoke_for_n<fork_type_, index_t>> for_n(index_t n, fork_type_ &&fork) noexcept {
        return {*this, {n, threads_count(), std::forward<fork_type_>(fork)}};
    }
    template <typename fork_type_ = dummy_lambda_t>
    FU_REQUIRES_((can_be_for_task_callback<fork_type_, index_t>()))
    broadcast_join<dummy_pool_t, invoke_for_n<fork_type_, index_t>> for_n_dynamic(index_t n,
                                                                                  fork_type_ &&fork) noexcept {
        return {*this, {n, threads_count(), std::forward<fork_type_>(fork)}};
    }
};

} // namespace forkunion
} // namespace ashvardanian
