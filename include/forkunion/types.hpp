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

#include <memory>  // `std::allocator`
#include <thread>  // `std::thread`
#include <atomic>  // `std::atomic`
#include <cstddef> // `std::max_align_t`
#include <cassert> // `assert`
#include <cstring> // `std::strlen`
#include <cstdio>  // `std::snprintf`
#include <cstdlib> // `std::strtoull`
#include <utility> // `std::exchange`, `std::addressof`
#include <new>     // `std::hardware_destructive_interference_size`
#include <array>   // `std::array`

#define FORKUNION_VERSION_MAJOR 2
#define FORKUNION_VERSION_MINOR 3
#define FORKUNION_VERSION_PATCH 1

#if !defined(FU_ALLOW_UNSAFE)
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS)
#define FU_ALLOW_UNSAFE 1
#else
#define FU_ALLOW_UNSAFE 0
#endif
#endif

/*  Layer 1 is identity: where are we? Derived once, from compiler predefines, and used only to derive
 *  the capabilities below. Nothing else in the library may ask `__linux__` again.  */
/*  Android is Linux, but Bionic has neither `libnuma` nor GLibC; it must not take the Linux path. */
#if defined(__linux__) && !defined(__ANDROID__)
#define FU_ON_LINUX 1
#else
#define FU_ON_LINUX 0
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

/*  An implementation detail, not a capability: several Linux capabilities are provided by one
 *  library, and its absence must lower all of them together. `gettid` needs GLibC 2.30+.
 *
 *  The header must be present, not merely the GLibC that usually ships beside it: `libnuma-dev` is a
 *  separate package on every distribution, and a build that assumed it from the GLibC version alone
 *  would enable `FU_WITH_TOPOLOGY` and then fail at `#include <numa.h>`.
 *  @see https://man7.org/linux/man-pages/man2/gettid.2.html  */
#if FU_ON_LINUX
#if __has_include(<features.h>)
#include <features.h> // `__GLIBC__`, `__GLIBC_PREREQ`
#endif
#if defined(__GLIBC__) && defined(__GLIBC_PREREQ) && __GLIBC_PREREQ(2, 30) && __has_include(<numa.h>)
#define FU_HAS_LIBNUMA_ 1
#else
#define FU_HAS_LIBNUMA_ 0
#endif
#else
#define FU_HAS_LIBNUMA_ 0
#endif

/*  Layer 2 is capabilities. Each answers exactly one question, and is named for the @b kernel @b
 *  facility rather than for the library that happens to provide it - so Windows' `VirtualAllocExNuma`
 *  satisfies `FU_WITH_NUMA_MEMORY` without inventing a second macro.
 *
 *  Every one auto-derives from Layer 1, and from the capabilities it leans on - so switching one off
 *  cascades to everything downstream, and `-DFU_WITH_TOPOLOGY=0` alone yields a coherent build rather
 *  than an `#error` about the four capabilities it silently orphaned. The `#error`s below then only
 *  ever fire on a contradiction the caller wrote out by hand.
 *
 *  A build system may @b override, never re-derive: that keeps the default in exactly one place,
 *  instead of duplicated across CMake, `build.rs`, and `build.zig`, where three copies would drift.  */

/** @brief Can we create operating-system threads directly, rather than through `std::thread`? */
#if !defined(FU_WITH_THREADS)
#define FU_WITH_THREADS (FU_ON_POSIX || FU_ON_WINDOWS)
#endif

/** @brief Can we enumerate this machine's cores, compute domains, and memory domains? */
#if !defined(FU_WITH_TOPOLOGY)
/*  Windows needs no separate library for this: `GetLogicalProcessorInformationEx` ships with the
 *  kernel since Vista and reports NUMA nodes, cores, processor groups, and caches in one call. */
#define FU_WITH_TOPOLOGY (FU_ON_APPLE || FU_ON_WINDOWS || (FU_ON_LINUX && FU_HAS_LIBNUMA_))
#endif

/** @brief Can we see which cores share a cache, so a compute domain can be cut at a cluster? */
#if !defined(FU_WITH_TOPOLOGY_CACHES)
#define FU_WITH_TOPOLOGY_CACHES FU_WITH_TOPOLOGY
#endif

/** @brief Can we read inter-domain distance, bandwidth, and latency? (ACPI SLIT and HMAT) */
#if !defined(FU_WITH_TOPOLOGY_METRICS)
#define FU_WITH_TOPOLOGY_METRICS (FU_ON_LINUX && FU_WITH_TOPOLOGY)
#endif

/**
 *  @brief Can we bind a thread to a set of cores, and have the kernel honour it?
 *  @note Deliberately independent of `FU_WITH_NUMA_MEMORY`. `pthread_setaffinity_np` needs no
 *        `libnuma`, and a Linux box without it could pin perfectly well - it simply never did,
 *        because a single NUMA macro guarded both.
 *  @note False on Apple Silicon, where `thread_policy_set(THREAD_AFFINITY_POLICY)` answers
 *        `KERN_NOT_SUPPORTED`. Its only placement lever is `FU_WITH_THREAD_QOS`.
 */
#if !defined(FU_WITH_THREAD_PINNING)
#define FU_WITH_THREAD_PINNING (FU_ON_LINUX || FU_ON_FREEBSD || FU_ON_WINDOWS)
#endif

/** @brief Can we hint which class of core a thread should run on, at creation time? */
#if !defined(FU_WITH_THREAD_QOS)
#define FU_WITH_THREAD_QOS FU_ON_APPLE
#endif

/** @brief Can we change @b another thread's scheduling class, to sleep or wake it cheaply? */
#if !defined(FU_WITH_THREAD_SCHED_CLASS)
#define FU_WITH_THREAD_SCHED_CLASS (FU_ON_LINUX || FU_ON_FREEBSD)
#endif

/** @brief Can we place pages on a chosen memory domain? */
#if !defined(FU_WITH_NUMA_MEMORY)
/*  Linux places with `mbind`; Windows with `VirtualAllocExNuma`. Same capability, named for the
 *  facility, not the library - so both kernels answer it without a second macro. */
#define FU_WITH_NUMA_MEMORY ((FU_ON_LINUX || FU_ON_WINDOWS) && FU_WITH_TOPOLOGY)
#endif

/** @brief Can we request pages larger than the base page? */
#if !defined(FU_WITH_HUGE_PAGES)
/*  Linux calls them huge pages (`MAP_HUGETLB`); Windows calls them large pages (`MEM_LARGE_PAGES`),
 *  gated behind the `SeLockMemoryPrivilege` the caller must already hold. */
#define FU_WITH_HUGE_PAGES ((FU_ON_LINUX || FU_ON_WINDOWS) && FU_WITH_NUMA_MEMORY)
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
#define FU_WITH_COLOCATED_POOLS (FU_WITH_THREADS && FU_WITH_TOPOLOGY)

/*  A bad override should fail at the `#include`, not at link time.  */
#if FU_WITH_THREAD_PINNING && FU_ON_APPLE
#error "FU_WITH_THREAD_PINNING: Apple answers KERN_NOT_SUPPORTED to thread_policy_set; pinning cannot be forced on"
#endif
#if FU_WITH_NUMA_MEMORY && !(FU_ON_LINUX || FU_ON_WINDOWS || FU_ON_FREEBSD)
#error "FU_WITH_NUMA_MEMORY needs a kernel that can place pages on a node"
#endif
#if FU_WITH_TOPOLOGY_METRICS && !FU_WITH_TOPOLOGY
#error "FU_WITH_TOPOLOGY_METRICS describes distances between domains we would not have discovered"
#endif
#if FU_WITH_NUMA_MEMORY && !FU_WITH_TOPOLOGY
#error "FU_WITH_NUMA_MEMORY places pages on domains we would not have discovered"
#endif
#if FU_WITH_HUGE_PAGES && FU_ON_LINUX && !FU_WITH_NUMA_MEMORY
#error "On Linux the hugetlb path maps with `mmap` and places with `mbind`; it needs FU_WITH_NUMA_MEMORY"
#endif

#if FU_ALLOW_UNSAFE
#include <exception> // `std::exception_ptr`
#endif

#if FU_WITH_TOPOLOGY && FU_ON_LINUX
#include <numa.h> // `numa_available`, `numa_node_to_cpus`, `numa_distance`
#endif

#if FU_WITH_NUMA_MEMORY && FU_ON_LINUX
#include <numa.h>     // `numa_alloc_onnode`, `numa_free`
#include <numaif.h>   // `mbind` manual assignment of `mmap` pages
#include <sys/mman.h> // `mmap`, `MAP_PRIVATE`, `MAP_ANONYMOUS`
#endif

#if FU_WITH_HUGE_PAGES && FU_ON_LINUX
#include <linux/mman.h> // `MAP_HUGE_2MB`, `MAP_HUGE_1GB`
#endif

/*  Both the huge-page inventory and the memory-tier probe walk sysfs directories - a Linux-only
 *  concern. Windows has no `<dirent.h>` under MSVC, and its large pages are probed by size, not path. */
#if (FU_WITH_HUGE_PAGES || FU_WITH_TOPOLOGY) && FU_ON_LINUX
#include <dirent.h> // `opendir`, `readdir`, `closedir`
#endif

#if FU_WITH_THREADS && FU_ON_POSIX
#include <pthread.h> // `pthread_create`, `pthread_setname_np`
#include <ctime>     // `nanosleep`, `clock_nanosleep`
#endif

#if FU_WITH_THREAD_PINNING && FU_ON_POSIX
#include <sched.h> // `cpu_set_t`, `CPU_ALLOC`, `pthread_setaffinity_np`
#endif

#if FU_WITH_THREAD_QOS
#include <sys/qos.h> // `qos_class_t`, `pthread_attr_set_qos_class_np`
#endif

#if defined(__unix__) || defined(__unix) || defined(unix) || defined(__APPLE__)
#include <unistd.h> // `gettid`, `sysconf`
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h> // `sysctl`
#endif

#if defined(_WIN32)
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

#if defined(__GNUC__) || defined(__clang__)
#define FU_WITH_ASM_YIELDS_ 1
#else
#define FU_WITH_ASM_YIELDS_ 0
#endif

/*  Detect target CPU architecture.
 *  We'll only use it when compiling Inline Assembly code on GCC or Clang.
 */
#if defined(__arm64__) || defined(__arm64__) || defined(_M_ARM64)
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

namespace ashvardanian {
namespace forkunion {

using numa_node_id_t = int;   // ? A.k.a. NUMA node ID, in [0, numa_max_node())
using numa_core_id_t = int;   // ? A.k.a. CPU core ID, in [0, threads_count)
using numa_socket_id_t = int; // ? A.k.a. physical CPU socket ID
using qos_level_t = int;      // ? Quality of Service, like: "performance", "efficiency", "low-power"

/**
 *  @brief A position in `numa_topology`'s array of @b compute domains, in [0, compute_domains_count).
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

/** @brief A position in `numa_topology`'s array of @b memory domains, in [0, memory_domains_count). */
enum memory_domain_index_t : std::size_t {};

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
 *  @brief Smallest multiple of @p multiple that is not less than @p value.
 *  @note @p multiple must be non-zero; overflow of @p value near the type maximum is not guarded.
 */
constexpr std::size_t round_up_to_multiple(std::size_t value, std::size_t multiple) noexcept {
    return (value + multiple - 1) / multiple * multiple;
}

/**
 *  @brief The kernel's own identifier for the calling thread, or 0 where there is none.
 *  @sa `numa_pthread_t::id`, which caches it so other threads can read it.
 *
 *  Linux calls it a `pid_t` and hands it out through `gettid`. Darwin has no `gettid` at all, and
 *  spells the same idea `pthread_threadid_np`, returning 64 bits. Both are the number a scheduler
 *  or a profiler will show you; neither is a `pthread_t`.
 */
FU_MAYBE_UNUSED_ static inline std::uint64_t current_thread_id() noexcept {
#if FU_ON_LINUX && FU_WITH_THREADS
    return static_cast<std::uint64_t>(::gettid());
#elif FU_ON_APPLE
    std::uint64_t thread_id = 0;
    ::pthread_threadid_np(nullptr, &thread_id);
    return thread_id;
#elif FU_ON_WINDOWS && FU_WITH_THREADS
    // A `DWORD` that a debugger or Task Manager will show you; distinct from the `HANDLE`.
    return static_cast<std::uint64_t>(::GetCurrentThreadId());
#else
    return 0;
#endif
}

/**
 *  @brief Names the @b calling thread, which is the only thread every platform lets us name.
 *
 *  Linux's `pthread_setname_np` takes a thread and a name, so a spawner can name its workers. Apple's
 *  takes only a name and always renames the caller. Rather than branch on that at every call, the
 *  worker names itself once it is running - the one shape both kernels agree on.
 */
FU_MAYBE_UNUSED_ static inline void set_current_thread_name(FU_MAYBE_UNUSED_ char const *thread_name) noexcept {
#if FU_ON_LINUX && FU_WITH_THREADS
    (void)::pthread_setname_np(::pthread_self(), thread_name);
#elif FU_ON_APPLE
    (void)::pthread_setname_np(thread_name);
#elif FU_ON_WINDOWS && FU_WITH_THREADS
    // `SetThreadDescription` wants UTF-16 and only exists on Windows 10 1607+. Resolve it at runtime
    // so a binary keeps loading on older Windows, where the name is simply not applied - the same
    // "best effort, never fatal" contract the POSIX paths keep.
    using set_thread_description_t = HRESULT(WINAPI *)(HANDLE, PCWSTR);
    HMODULE const kernel32 = ::GetModuleHandleW(L"kernel32.dll");
    if (!kernel32) return;
    // The `FARPROC`-to-typed-pointer cast is the documented `GetProcAddress` idiom; MSVC's C4191 for it
    // is suppressed with the other Windows pragmas up top, and it is clean under `-Wextra`/clang-tidy.
    auto const set_thread_description =
        reinterpret_cast<set_thread_description_t>(::GetProcAddress(kernel32, "SetThreadDescription"));
    if (!set_thread_description) return;

    // POSIX thread names cap at 16 bytes; the same buffer never needs more than 16 wide chars.
    wchar_t wide_name[16] = {};
    int const written =
        ::MultiByteToWideChar(CP_UTF8, 0, thread_name, -1, wide_name, static_cast<int>(std::size(wide_name)));
    if (written <= 0) return; // ? Nothing usable to hand over
    wide_name[std::size(wide_name) - 1] = L'\0';
    (void)set_thread_description(::GetCurrentThread(), wide_name);
#endif
}

/**
 *  @brief Upper bound on core IDs this machine may ever report, for sizing masks and names.
 *
 *  Not the same as `hardware_concurrency()` on Linux, where cores can be hot-plugged and the kernel
 *  reserves IDs for cores that are offline right now. Elsewhere the distinction does not exist.
 */
FU_MAYBE_UNUSED_ static inline std::size_t possible_cores() noexcept {
#if FU_ON_POSIX
    // ! Not `_SC_NPROCESSORS_ONLN`: a core that is offline right now still owns an ID, and a mask
    // ! sized to the online count would refuse to name it.
    long const configured = ::sysconf(_SC_NPROCESSORS_CONF);
    if (configured > 0) return static_cast<std::size_t>(configured);
#elif FU_ON_WINDOWS
    DWORD const configured = ::GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (configured > 0) return static_cast<std::size_t>(configured);
#endif
    return static_cast<std::size_t>(std::thread::hardware_concurrency());
}

/**
 *  @brief The cores the calling thread may run on - this process's contract with the kernel.
 *
 *  A machine is not the same thing as the slice of it we were handed. `taskset`, a cgroup `cpuset`,
 *  and a batch scheduler all narrow this mask, and `hardware_concurrency` sees none of them. Sizing
 *  a pool from the machine and pinning to cores outside the mask either escapes the restriction, or
 *  - where the kernel enforces it - crowds every spinning worker onto the few cores that remain.
 *
 *  Owns its mask, so the same type serves as the snapshot a pool restores on teardown. @sa `restore`.
 */
struct affinity_mask {
#if FU_WITH_THREAD_PINNING && FU_ON_POSIX

  private:
    cpu_set_t *cores_ {nullptr};
    std::size_t bytes_ {0};

  public:
    affinity_mask() noexcept = default;
    affinity_mask(affinity_mask const &) = delete;
    affinity_mask &operator=(affinity_mask const &) = delete;
    affinity_mask(affinity_mask &&other) noexcept : cores_(other.cores_), bytes_(other.bytes_) {
        other.cores_ = nullptr;
        other.bytes_ = 0;
    }
    affinity_mask &operator=(affinity_mask &&other) noexcept {
        if (this != &other) {
            reset();
            cores_ = std::exchange(other.cores_, nullptr);
            bytes_ = std::exchange(other.bytes_, 0);
        }
        return *this;
    }
    ~affinity_mask() noexcept { reset(); }

    void reset() noexcept {
        if (cores_) CPU_FREE(cores_);
        cores_ = nullptr;
        bytes_ = 0;
    }

    bool valid() const noexcept { return cores_ != nullptr; }

    /** @brief Reads the calling thread's current affinity. @retval false if it could not be read. */
    bool try_capture() noexcept {
        reset();
        std::size_t const max_cores = possible_cores();
        cores_ = CPU_ALLOC(max_cores);
        if (!cores_) return false;
        bytes_ = CPU_ALLOC_SIZE(max_cores);
        CPU_ZERO_S(bytes_, cores_);
        if (::sched_getaffinity(0, bytes_, cores_) == 0) return true;
        reset();
        return false;
    }

    /** @brief Reinstates this mask on the calling thread. @retval false if the kernel refused. */
    bool restore() const noexcept {
        if (!cores_) return false;
        return ::pthread_setaffinity_np(::pthread_self(), bytes_, cores_) == 0;
    }

    bool contains(numa_core_id_t const core) const noexcept {
        return cores_ && core >= 0 && CPU_ISSET_S(static_cast<std::size_t>(core), bytes_, cores_) != 0;
    }
    std::size_t count() const noexcept { return cores_ ? static_cast<std::size_t>(CPU_COUNT_S(bytes_, cores_)) : 0; }

#else // ? Nothing is ever narrowed here, so every core is allowed and there is nothing to restore

  public:
    void reset() noexcept {}
    bool valid() const noexcept { return false; }
    bool try_capture() noexcept { return false; }
    bool restore() const noexcept { return false; }
    bool contains(FU_MAYBE_UNUSED_ numa_core_id_t const core) const noexcept { return true; }
    std::size_t count() const noexcept { return 0; }
#endif
};

/**
 *  @brief Number of cores the calling thread may run on, or `possible_cores()` where unknowable.
 *  @note Prefer this to `std::thread::hardware_concurrency` when sizing a pool: the latter counts
 *        the machine's cores, not the ones this process was given.
 */
FU_MAYBE_UNUSED_ static inline std::size_t count_allowed_cores() noexcept {
    affinity_mask allowed;
    if (allowed.try_capture()) {
        std::size_t const allowed_count = allowed.count();
        if (allowed_count > 0) return allowed_count;
    }
    return possible_cores();
}

#if FU_ON_WINDOWS
/*  Windows addresses a logical processor by (processor group, bit within the group's 64-bit
 *  `KAFFINITY` mask), not by a flat global id. A `numa_core_id_t` therefore packs both, so the free
 *  function `pin_thread_to_cores` can rebuild a `GROUP_AFFINITY` from an id alone - no side table
 *  threaded through its signature. The low 6 bits hold the in-group index (a mask is 64 bits, so the
 *  index is 0..63); the remaining bits hold the group number. Everywhere else a `numa_core_id_t` is
 *  still just an opaque, comparable id - only the pinning path decodes it. */
static constexpr int win_core_group_shift_k = 6;
static constexpr numa_core_id_t win_core_index_mask_k = (numa_core_id_t {1} << win_core_group_shift_k) - 1;
/** @brief Logical processors per Windows processor group - the `KAFFINITY` bit-width, a hard ABI cap
 *         of 64 @b per @b group, never a cap on total cores (a machine with more uses several groups). */
static constexpr unsigned win_processors_per_group_k = 1u << win_core_group_shift_k;

FU_MAYBE_UNUSED_ static inline numa_core_id_t win_encode_core_id(WORD group, unsigned bit) noexcept {
    return (static_cast<numa_core_id_t>(group) << win_core_group_shift_k) |
           (static_cast<numa_core_id_t>(bit) & win_core_index_mask_k);
}
FU_MAYBE_UNUSED_ static inline WORD win_core_group(numa_core_id_t id) noexcept {
    return static_cast<WORD>(id >> win_core_group_shift_k);
}
FU_MAYBE_UNUSED_ static inline unsigned win_core_index(numa_core_id_t id) noexcept {
    return static_cast<unsigned>(id & win_core_index_mask_k);
}
#endif // FU_ON_WINDOWS

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
 *  Two questions share one bit-space, and the names say which is which. An unmarked bit is a fact
 *  about @b this @b machine: `capability_huge_pages_k` means the kernel is offering them. A bit
 *  marked `comptime_` is a fact about @b this @b build: `capability_comptime_huge_pages_k` means we
 *  compiled the code that would ask for them.
 *
 *  Neither implies the other. A binary carrying `capability_comptime_numa_memory_k` runs perfectly
 *  well on a single-node box, where `capability_numa_aware_k` never appears; and a machine with four
 *  NUMA nodes reports none of them to a build that left the topology out.
 */
enum capabilities_t : unsigned int {
    capabilities_unknown_k = 0,

    /** The `PAUSE` spin hint, on every x86 since the Pentium 4. */
    capability_x86_pause_k = 1 << 1,
    /** `TPAUSE` sleeps the core until a deadline, rather than spinning. Needs the `WAITPKG` feature. */
    capability_x86_tpause_k = 1 << 2,
    /** The `YIELD` hint, on every AArch64. Releases the pipeline to a sibling hardware thread. */
    capability_arm64_yield_k = 1 << 3,
    /** `WFET` sleeps the core until a deadline or an event. Needs `FEAT_WFxT`. */
    capability_arm64_wfet_k = 1 << 4,
    /** The `PAUSE` spin hint, from the `Zihintpause` extension. */
    capability_risc5_pause_k = 1 << 5,

    /** Pinned to a single compute_domain (a same-QoS core cluster). */
    capability_compute_domain_k = 1 << 6,

    /** NUMA-aware memory allocations. */
    capability_numa_aware_k = 1 << 10,
    /** Reducing TLB pressure with huge pages. */
    capability_huge_pages_k = 1 << 11,
    /** ... doing the same "transparently". */
    capability_huge_pages_transparent_k = 1 << 12,

    /** Can spawn OS threads directly, rather than through `std::thread`. `FU_WITH_THREADS`. */
    capability_comptime_threads_k = 1 << 16,
    /** Can enumerate this machine's cores, compute domains, and memory domains. `FU_WITH_TOPOLOGY`. */
    capability_comptime_topology_k = 1 << 17,
    /** Can see which cores share a cache, so a domain is cut at a cluster. `FU_WITH_TOPOLOGY_CACHES`. */
    capability_comptime_topology_caches_k = 1 << 18,
    /** Can read inter-domain distance, bandwidth, and latency. `FU_WITH_TOPOLOGY_METRICS`. */
    capability_comptime_topology_metrics_k = 1 << 19,
    /** Can bind a thread to a set of cores, and have the kernel honour it. `FU_WITH_THREAD_PINNING`. */
    capability_comptime_thread_pinning_k = 1 << 20,
    /** Can hint which class of core a thread runs on, at creation. `FU_WITH_THREAD_QOS`. */
    capability_comptime_thread_qos_k = 1 << 21,
    /** Can change another thread's scheduling class, to sleep or wake it. `FU_WITH_THREAD_SCHED_CLASS`. */
    capability_comptime_thread_sched_class_k = 1 << 22,
    /** Can place pages on a chosen memory domain. `FU_WITH_NUMA_MEMORY`. */
    capability_comptime_numa_memory_k = 1 << 23,
    /** Can request pages larger than the base page. `FU_WITH_HUGE_PAGES`. */
    capability_comptime_huge_pages_k = 1 << 24,
    /** The `colocated_pool` and `distributed_pool` are compiled in. `FU_WITH_COLOCATED_POOLS`. */
    capability_comptime_colocated_pools_k = 1 << 25,
};

inline capabilities_t operator|(capabilities_t a, capabilities_t b) {
    return static_cast<capabilities_t>(static_cast<unsigned int>(a) | static_cast<unsigned int>(b));
}

/** @brief The portable busy-wait hint: hands the core back to the scheduler. Works everywhere, cheap nowhere. */
struct standard_yield_t {
    inline void operator()() const noexcept { std::this_thread::yield(); }
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
    pool_t &pool_ref_;
    /** We need this to extend the lifetime of the lambda object. */
    fork_t fork_;
    /** Real tokens are odd; zero means "not yet dispatched". */
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

/**
 *  @brief A "prong" - is a tip of a "fork" - pinning "task" to a "thread".
 */
template <typename index_type_ = std::size_t>
struct prong {
    using index_t = index_type_;
    using task_index_t = index_t;   // ? A.k.a. "task index" in [0, prongs_count)
    using thread_index_t = index_t; // ? A.k.a. "core index" or "thread ID" in [0, threads_count)

    task_index_t task {0};
    thread_index_t thread {0};

    constexpr prong() noexcept = default;
    constexpr prong(prong &&) noexcept = default;
    constexpr prong(prong const &) noexcept = default;
    constexpr prong &operator=(prong &&) noexcept = default;
    constexpr prong &operator=(prong const &) noexcept = default;

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

    task_index_t task {0};
    thread_index_t thread {0};
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

    thread_index_t thread {0};
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

using local_thread_t = local_thread<>; // ? Default prong type with `std::size_t` indices

/**
 *  @brief Back-ports the C++ 23 `std::allocation_result`. Unlike STL, also contains the page size.
 *  @see https://en.cppreference.com/w/cpp/memory/allocator/allocate_at_least
 */
template <typename pointer_type_ = char, typename size_type_ = std::size_t>
struct allocation_result {
    using pointer_type = pointer_type_;
    using size_type = size_type_;

    /** Pointer to the allocated memory, or nullptr if allocation failed. */
    pointer_type ptr {nullptr};
    /** Number of elements allocated, or 0 if allocation failed. */
    size_type count {0};
    /** Reports the total volume of memory allocated, in bytes. */
    size_type bytes {0};
    /** Reports the number of memory pages allocated. */
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
    std::array<value_t, capacity_> values_ {};
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
 *  @sa `limited_array` for bounded counts, `unique_padded_buffer` when each element wants its own line.
 *
 *  Deliberately not a `std::vector`: there is no capacity, no growth policy, and no exception. A
 *  `try_resize` either hands back a fully-constructed array or leaves an empty one, which is what
 *  lets a harvest fail without a `goto` unwinding three raw pointers by hand.
 *
 *  Elements are value-initialized and never reallocated, so a pointer taken into the array stays
 *  valid until the next `try_resize` - the topology relies on that to slice its core-id list.
 */
template <typename value_type_, typename allocator_type_ = std::allocator<value_type_>>
class dynamic_array {
    static_assert(std::is_nothrow_default_constructible_v<value_type_>,
                  "dynamic_array requires noexcept-default-constructible values");
    static_assert(std::is_nothrow_destructible_v<value_type_>, "dynamic_array requires noexcept-destructible values");

    using value_t = value_type_;
    using allocator_t = typename std::allocator_traits<allocator_type_>::template rebind_alloc<value_t>;

    allocator_t allocator_ {};
    value_t *data_ {nullptr};
    std::size_t size_ {0};

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
          size_(std::exchange(other.size_, 0)) {}

    dynamic_array &operator=(dynamic_array &&other) noexcept {
        if (this != &other) {
            reset();
            allocator_ = std::move(other.allocator_);
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    dynamic_array(dynamic_array const &) = delete;
    dynamic_array &operator=(dynamic_array const &) = delete;
    ~dynamic_array() noexcept { reset(); }

    void reset() noexcept {
        if (data_) {
            destroy_all();
            allocator_.deallocate(data_, size_);
            data_ = nullptr;
        }
        size_ = 0;
    }

    /** @retval false on allocation failure, leaving the array empty rather than half-built. */
    bool try_resize(std::size_t const new_size) noexcept {
        reset();
        if (new_size == 0) return true;
        value_t *fresh = allocator_.allocate(new_size);
        if (!fresh) return false;
        for (std::size_t i = 0; i < new_size; ++i) ::new (static_cast<void *>(fresh + i)) value_t();
        data_ = fresh;
        size_ = new_size;
        return true;
    }

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
 *  @brief Analogous to `std::unique_ptr<T[]>`, but designed for large padded allocations.
 *  @see https://en.cppreference.com/w/cpp/memory/unique_ptr.html
 */
template <typename object_type_, typename allocator_type_>
class unique_padded_buffer {

    using object_t = object_type_;
    static_assert(std::is_nothrow_default_constructible_v<object_t>,
                  "unique_padded_buffer requires noexcept-default-constructible object type");

    using allocator_t = allocator_type_;
    using allocator_traits_t = std::allocator_traits<allocator_t>;
    using raw_allocator_t = typename allocator_traits_t::template rebind_alloc<char>;

    char *raw_ {nullptr};       // ? Aligned base the objects live at
    char *raw_owned_ {nullptr}; // ? What the allocator actually handed us, and what we must give back
    std::size_t objects_count_ {0};
    std::size_t bytes_per_object_ {sizeof(object_t)};
    std::size_t bytes_total_ {0};
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
    unique_padded_buffer() noexcept = default;

    explicit unique_padded_buffer(allocator_t const &alloc, std::size_t bytes_per_object = sizeof(object_t)) noexcept
        : bytes_per_object_(bytes_per_object), allocator_(alloc) {}

    unique_padded_buffer(unique_padded_buffer &&o) noexcept
        : raw_(std::exchange(o.raw_, nullptr)), raw_owned_(std::exchange(o.raw_owned_, nullptr)),
          objects_count_(std::exchange(o.objects_count_, 0)), bytes_per_object_(o.bytes_per_object_),
          bytes_total_(std::exchange(o.bytes_total_, 0)), allocator_(std::move(o.allocator_)) {}

    unique_padded_buffer &operator=(unique_padded_buffer &&o) noexcept {
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

    unique_padded_buffer(unique_padded_buffer const &) = delete;
    unique_padded_buffer &operator=(unique_padded_buffer const &) = delete;

    ~unique_padded_buffer() noexcept {
        destroy_all();
        deallocate();
    }

    bool try_resize(std::size_t new_objects_count) noexcept {
        destroy_all();
        deallocate();

        if (new_objects_count == 0) return true;

        // An `alignas(128)` object placement-newed into 16-byte-aligned storage is undefined, and it
        // is exactly what happens when the object is a pool cell and the allocator is `std::allocator`.
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

    object_t &only() noexcept {
        assert(objects_count_ == 1 && "Buffer must contain exactly one object to use `only()`");
        return *ptr(0);
    }
    object_t const &only() const noexcept {
        assert(objects_count_ == 1 && "Buffer must contain exactly one object to use `only()`");
        return *ptr(0);
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
 *  @brief Which call shapes a busy-wait functor accepts, so `call_yield_` can pick one.
 *
 *  A yield may want the calling thread's index - to back off proportionally to it, or to log - and
 *  may equally ignore it. Rather than force every functor to take an argument it will not read, we
 *  detect both shapes and dispatch, rejecting anything that supports neither.
 */
template <typename yield_type_, typename thread_index_type_>
struct yield_traits {
    static constexpr bool supports_no_arg = std::is_nothrow_invocable_r_v<void, yield_type_>;
    static constexpr bool supports_thread_index = std::is_nothrow_invocable_r_v<void, yield_type_, thread_index_type_>;
    static constexpr bool valid = supports_no_arg || supports_thread_index;
};

template <typename yield_type_, typename thread_index_type_>
inline void call_yield_(yield_type_ &yield, thread_index_type_ thread_index) noexcept {
    if constexpr (yield_traits<yield_type_, thread_index_type_>::supports_thread_index) { yield(thread_index); }
    else { yield(); }
}

/**
 *  @brief A trivial minimalistic lock-free "mutex" implementation using `std::atomic_flag`.
 *  @tparam micro_yield_type_ The type of the yield function to be used for busy-waiting.
 *  @tparam alignment_ The alignment of the mutex. Defaults to `default_alignment_k`.
 *
 *  The C++ standard would recommend using `std::hardware_destructive_interference_size`
 *  alignment, as well as `std::atomic_flag::notify_one` and `std::this_thread::yield` APIs,
 *  but our solution is better despite being more primitive.
 *
 *  @see Compatible with STL unique locks: https://en.cppreference.com/w/cpp/thread/unique_lock.html
 */
#if FU_DETECT_CPP_20_

template <typename micro_yield_type_ = standard_yield_t, std::size_t alignment_ = default_alignment_k>
class spin_mutex {
    using micro_yield_t = micro_yield_type_;
    static constexpr std::size_t alignment_k = alignment_;
    alignas(alignment_k) std::atomic_flag flag_ = ATOMIC_FLAG_INIT;

  public:
    void lock() noexcept {
        micro_yield_t micro_yield;
        while (flag_.test_and_set(std::memory_order_acquire)) call_yield_(micro_yield);
    }
    bool try_lock() noexcept { return !flag_.test_and_set(std::memory_order_acquire); }
    void unlock() noexcept { flag_.clear(std::memory_order_release); }
};

#else // FU_DETECT_CPP_20_

template <typename micro_yield_type_ = standard_yield_t, std::size_t alignment_ = default_alignment_k>
class spin_mutex {
    using micro_yield_t = micro_yield_type_;
    static constexpr std::size_t alignment_k = alignment_;

    /**
     *  Theoretically, the choice of `std::atomic<bool>` is suboptimal in the presence of `std::atomic_flag`.
     *  The latter is guaranteed to be lock-free, while the former is not. But until C++20, the flag doesn't
     *  have a non-modifying load operation - the `std::atomic_flag::test` was added in C++20.
     *  @see https://en.cppreference.com/w/cpp/atomic/atomic_flag.html
     */
    std::atomic<bool> flag_ {false};

  public:
    void lock() noexcept {
        micro_yield_t micro_yield;
        while (flag_.exchange(true, std::memory_order_acquire)) call_yield_(micro_yield);
    }
    bool try_lock() noexcept { return !flag_.exchange(true, std::memory_order_acquire); }
    void unlock() noexcept { flag_.store(false, std::memory_order_release); }
};

#endif // FU_DETECT_CPP_20_

using spin_mutex_t = spin_mutex<>;

/** @brief A half-open slice `[first, first + count)` of a task index space. */
template <typename index_type_ = std::size_t>
struct indexed_range {
    using index_t = index_type_;

    /** The first task index in the slice. */
    index_t first {0};
    /** How many tasks the slice covers; zero means an empty slice. */
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

  private:
    index_t quotient_ {0};
    index_t remainder_ {0};
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

        index_t start_ {0};         // first value of the domain
        index_t length_ {1};        // |domain|
        index_t stride_ {1};        // co-prime step
        index_t offset_ {0};        // current offset 0 ... length_-1
        index_t elements_left_ {0}; // countdown until `end()`
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

    index_t start_ {0};
    index_t length_ {1};
    index_t stride_ {1};
    /** Where this seed's walk begins, in [0, length_). */
    index_t first_offset_ {0};
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
 *  the split exists to avoid. @sa `unique_padded_buffer`, which spaces them by the pool's alignment.
 */
template <typename index_type_ = std::size_t>
struct dynamic_claim {
    /** Next task in this slice; only ever grows, and may overshoot `end` by `threads`. */
    std::atomic<index_type_> next {0};
    /** One past this slice's last task. Written once before the dispatch, then read-only. */
    index_type_ end {0};
};

using dynamic_claim_t = dynamic_claim<>;

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
 *  That trailing reservation also bounds the cursors. Every thread touches a given slice exactly once
 *  - the owner drains it, each other thread helps drain it once, and the `!= thread` guard below keeps
 *  the owner from doing both - and each visit overshoots by at most one increment, since `drain_` leaves
 *  the moment it reads `>= end`. A cursor therefore settles at exactly `end + threads`.
 *
 *  Two regimes bound that. When `n > threads` the last slice ends at `n - threads`, so no cursor passes
 *  `n`. When `n <= threads` every slice is empty and `end == 0`, so no cursor passes `threads` - which
 *  may exceed `n`, but is still an index the type must represent to have spawned the pool at all.
 *  Either way `max(cursor) == max(n, threads)`, and no index type can wrap.
 */
template <typename pool_type_, typename fork_type_, typename index_type_>
class invoke_for_n_dynamic {
    pool_type_ &pool_; // ? Owns one padded `dynamic_claim` per thread; we never allocate
    fork_type_ fork_;
    index_type_ n_;
    index_type_ threads_;

    /** @brief Number of tasks handed out dynamically; the trailing `threads_` are static prongs. */
    index_type_ dynamic_count() const noexcept { return n_ > threads_ ? static_cast<index_type_>(n_ - threads_) : 0; }

    /** @brief Runs whatever is left of @p slice, whether or not we own it. */
    void drain_(index_type_ const slice, prong<index_type_> &prong) noexcept {
        dynamic_claim<index_type_> &claim = pool_.unsafe_dynamic_claim_ref(slice);
        while (true) {
            index_type_ const task = claim.next.fetch_add(1, std::memory_order_relaxed);
            if (task >= claim.end) break; // ? Overshoots by one, and only once per thread
            prong.task = task;
            fork_(prong);
        }
    }

  public:
    invoke_for_n_dynamic(pool_type_ &pool, index_type_ n, index_type_ threads, fork_type_ &&fork) noexcept
        : pool_(pool), fork_(std::forward<fork_type_>(fork)), n_(n), threads_(threads) {
        reset_slices_();
    }

    invoke_for_n_dynamic(invoke_for_n_dynamic &&other) noexcept // ? Need to manually define the `move` due to atomics
        : pool_(other.pool_), fork_(std::move(other.fork_)), n_(other.n_), threads_(other.threads_) {
        other.n_ = 0;
        reset_slices_();
    }

    void operator()(index_type_ const thread) noexcept {

        // A single-thread pool has no neighbours and keeps no cursors - just run the loop.
        if (threads_ == 1) {
            prong<index_type_> prong(0, thread);
            for (index_type_ task = 0; task < n_; ++task) {
                prong.task = task;
                fork_(prong);
            }
            return;
        }

        index_type_ const n_dynamic = dynamic_count();
        assert((n_dynamic + threads_) >= n_dynamic && "Overflow detected");

        // Run (up to) one static prong on the current thread
        index_type_ const one_static_prong_index = static_cast<index_type_>(n_dynamic + thread);
        prong<index_type_> prong(one_static_prong_index, thread);
        if (one_static_prong_index < n_) fork_(prong);

        // Drain our own slice first - nobody else is touching this cache line yet
        drain_(thread, prong);

        // Then help the others, in a coprime order so drained threads don't collide on one victim
        coprime_permutation_range<index_type_> victims(0, threads_, thread);
        for (auto victim = victims.begin(); victim != default_sentinel_t {}; ++victim)
            if (*victim != thread) drain_(*victim, prong);
    }

  private:
    /** @brief Publishes one contiguous slice per thread. Runs on the caller, before the broadcast. */
    void reset_slices_() noexcept {
        if (threads_ <= 1) return; // ? No cursors exist on a single-thread pool
        index_type_ const n_dynamic = dynamic_count();
        indexed_split<index_type_> const split(n_dynamic, threads_);
        for (index_type_ thread = 0; thread < threads_; ++thread) {
            indexed_range<index_type_> const range = split[thread];
            dynamic_claim<index_type_> &claim = pool_.unsafe_dynamic_claim_ref(thread);
            claim.end = static_cast<index_type_>(range.first + range.count);
            claim.next.store(range.first, std::memory_order_release);
        }
    }
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

} // namespace forkunion
} // namespace ashvardanian
