/**
 *  @brief  Low-latency OpenMP-style NUMA-aware cross-platform fine-grained parallelism library.
 *  @file   fork_union.hpp
 *  @author Ash Vardanian
 *  @date   May 2, 2025
 *
 *  Fork Union provides a minimalistic cross-platform thread-pool implementation and Parallel Algorithms,
 *  avoiding dynamic memory allocations, exceptions, system calls, and heavy Compare-And-Swap instructions.
 *  The library leverages the "weak memory model" to allow Arm and IBM Power CPUs to aggressively optimize
 *  execution at runtime. It also aggressively tests against overflows on smaller index types, and is safe
 *  to use even with the maximal `std::size_t` values.
 *
 *  @code{.cpp}
 *  #include <cstdio> // `std::printf`
 *  #include <cstdlib> // `EXIT_FAILURE`, `EXIT_SUCCESS`
 *  #include <fork_union.hpp> // `fu::basic_pool_t`
 *
 *  using fu = ashvardanian::fork_union;
 *  int main(int argc, char *argv[]) {
 *
 *      fu::basic_pool_t pool;
 *      if (!pool.try_spawn(std::thread::hardware_concurrency()))
 *          return EXIT_FAILURE;
 *
 *      pool.for_n(argc, [=](auto prong) noexcept {
 *          auto [task_index, thread_index, colocation_index] = prong;
 *          std::printf(
 *              "Printing argument # %zu (of %zu) from thread # %zu at colocation # %zu: %s\n",
 *              task_index, argc, thread_index, colocation_index, argv[task_index]);
 *      });
 *      return EXIT_SUCCESS;
 *  }
 *  @endcode
 *
 *  The next layer of logic is for basic index-addressable tasks. It includes basic parallel loops:
 *
 *  - `for_n` - for iterating over a range of similar duration tasks, addressable by an index.
 *  - `for_n_dynamic` - for unevenly distributed tasks, where each task may take a different time.
 *  - `for_slices` - for iterating over a range of similar duration tasks, addressable by a slice.
 *
 *  ------------------------------------------------------------------------------------------------
 *
 *  On Linux, when NUMA and PThreads are available, the library can also leverage @b NUMA-aware
 *  memory allocations and pin threads to specific physical cores to increase memory locality.
 *  It should reduce memory access latency by around 35% on average, compared to remote accesses.
 *  @sa `numa_topology_t`, `linux_colocated_pool_t`, `linux_distributed_pool_t`.
 *
 *  On heterogeneous chips, cores with a different @b "Quality-of-Service" (QoS) may be combined.
 *  A typical example is laptop/desktop chips, having 1 NUMA node, but 3 tiers of CPU cores:
 *  performance, efficiency, and power-saving cores. Each group will have vastly different speed,
 *  so considering them equal in tasks scheduling is a bad idea... and separating them automatically
 *  isn't feasible either. It's up to the user to isolate those groups into individual pools.
 *  @sa `qos_level_t`
 *
 *  On x86, Arm, and RISC-V (internally referred to as RISC5) architectures, depending on the CPU
 *  features available, the library also exposes cheaper @b "busy-waiting" mechanisms, such as
 *  `tpause`, `wfet`, & `yield` instructions.
 *  @sa `arm64_yield_t`, `arm64_wfet_t`, `x86_yield_t`, `x86_tpause_t`, `risc5_yield_t`.
 *
 *  The library uses modern C++ features and requires @b C++17 or newer.
 *  Using C++20 will enable additional compile-time checks (concepts) where available.
 */
#pragma once
#if defined(_MSC_VER)
#pragma warning(disable : 4505) // unreferenced function with internal linkage has been removed
#pragma warning(disable : 4324) // structure was padded due to alignment specifier
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

#define FORK_UNION_VERSION_MAJOR 2
#define FORK_UNION_VERSION_MINOR 3
#define FORK_UNION_VERSION_PATCH 0

#if !defined(FU_ALLOW_UNSAFE)
#define FU_ALLOW_UNSAFE 0
#endif

/**
 *  We auto-enable NUMA in Linux builds with GLibC 2.30+ due to `gettid` support.
 *  @see https://man7.org/linux/man-pages/man2/gettid.2.html
 */
#if !defined(FU_ENABLE_NUMA)
#if defined(__linux__)
#if __has_include(<features.h>)
#include <features.h> // `__GLIBC__`, `__GLIBC_PREREQ`
#endif
#if defined(__GLIBC__) && defined(__GLIBC_PREREQ) && __GLIBC_PREREQ(2, 30)
#define FU_ENABLE_NUMA 1
#else
#define FU_ENABLE_NUMA 0
#endif
#else
#define FU_ENABLE_NUMA 0
#endif
#endif

#if FU_ALLOW_UNSAFE
#include <exception> // `std::exception_ptr`
#endif

#if FU_ENABLE_NUMA
#include <numa.h>       // `numa_available`, `numa_node_of_cpu`, `numa_alloc_onnode`
#include <numaif.h>     // `mbind` manual assignment of `mmap` pages
#include <pthread.h>    // `pthread_getaffinity_np`
#include <sys/mman.h>   // `mmap`, `MAP_PRIVATE`, `MAP_ANONYMOUS`
#include <linux/mman.h> // `MAP_HUGE_2MB`, `MAP_HUGE_1GB`
#include <dirent.h>     // `opendir`, `readdir`, `closedir`
#endif

#if defined(__unix__) || defined(__unix) || defined(unix) || defined(__APPLE__)
#include <unistd.h> // `gettid`, `sysconf`
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h> // `sysctl`
#endif

#if defined(_WIN32)
#define NOMINMAX                // Disable `max` macros conflicting with STL symbols
#define _CRT_SECURE_NO_WARNINGS // Disable "This function or variable may be unsafe" warnings
#include <windows.h>            // `GlobalMemoryStatusEx`
#include <io.h>                 // `_isatty`, `_fileno`
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
namespace fork_union {

#pragma region - Helpers and Constants

using numa_node_id_t = int;   // ? A.k.a. NUMA node ID, in [0, numa_max_node())
using numa_core_id_t = int;   // ? A.k.a. CPU core ID, in [0, threads_count)
using numa_socket_id_t = int; // ? A.k.a. physical CPU socket ID
using qos_level_t = int;      // ? Quality of Service, like: "performance", "efficiency", "low-power"

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
    grind_k = 0, // ? That's our default ;)
    chill_k,     // ? Sleepy and tired, but just a wake-up call away
    die_k,       // ? The thread is about to die, we must exit the loop peacefully
};

/**
 *  @brief Describes all the special library features.
 */
enum capabilities_t : unsigned int {
    capabilities_unknown_k = 0,

    // CPU-specific capabilities:
    capability_x86_pause_k = 1 << 1,   // ? x86
    capability_x86_tpause_k = 1 << 2,  // ? x86-64 with `WAITPKG` support
    capability_arm64_yield_k = 1 << 3, // ? Arm
    capability_arm64_wfet_k = 1 << 4,  // ? AArch64 with `WFET` support
    capability_risc5_pause_k = 1 << 5, // ? RISC-V

    // RAM-specific capabilities:
    capability_numa_aware_k = 1 << 10,             // ? NUMA-aware memory allocations
    capability_huge_pages_k = 1 << 11,             // ? Reducing TLB pressure with huge pages
    capability_huge_pages_transparent_k = 1 << 12, // ? ... doing the same "transparently"
};

inline capabilities_t operator|(capabilities_t a, capabilities_t b) {
  return static_cast<capabilities_t>(static_cast<unsigned int>(a) | static_cast<unsigned int>(b));
}

struct standard_yield_t {
    inline void operator()() const noexcept { std::this_thread::yield(); }
};

/**
 *  @brief A synchronization point that waits for all threads to finish the last fork.
 *  @note You don't have to explicitly call any of the APIs, it's like `std::jthread` ;)
 *
 *  You don't have to explicitly handle the return value and wait on it.
 *  According to the  C++ standard, the destructor of the `broadcast_join_t` will
 *  be called in the end of the `for_threads`-calling expression.
 */
template <typename pool_type_, typename fork_type_>
struct broadcast_join {

    using pool_t = pool_type_;
    using fork_t = fork_type_;

  private:
    pool_t &pool_ref_;
    fork_t fork_;                // ? We need this to extend the lifetime of the lambda object
    bool did_broadcast_ {false}; // ? Both

  public:
    broadcast_join(pool_t &pool_ref, fork_t &&f) noexcept : pool_ref_(pool_ref), fork_(std::forward<fork_t>(f)) {}
    fork_t &fork_ref() noexcept { return fork_; }

    void broadcast() noexcept {
        if (did_broadcast_) return; // ? No need to broadcast again
        pool_ref_.unsafe_for_threads(fork_);
        did_broadcast_ = true;
    }
    void join() noexcept {
        if (!did_broadcast_) {
            pool_ref_.unsafe_for_threads(fork_);
            did_broadcast_ = true;
        }
        pool_ref_.unsafe_join();
    }

    ~broadcast_join() noexcept { join(); }
    broadcast_join(broadcast_join &&) noexcept = default;
    broadcast_join(broadcast_join const &) = delete;
    broadcast_join &operator=(broadcast_join &&) noexcept = default;
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
struct colocated_prong {
    using index_t = index_type_;
    using task_index_t = index_t;       // ? A.k.a. "task index" in [0, prongs_count)
    using thread_index_t = index_t;     // ? A.k.a. "core index" or "thread ID" in [0, threads_count)
    using colocation_index_t = index_t; // ? A.k.a. NUMA-specific QoS-specific "colocation ID"

    task_index_t task {0};
    thread_index_t thread {0};
    colocation_index_t colocation {0};

    constexpr colocated_prong() noexcept = default;
    constexpr colocated_prong(colocated_prong &&) noexcept = default;
    constexpr colocated_prong(colocated_prong const &) noexcept = default;
    constexpr colocated_prong &operator=(colocated_prong const &) noexcept = default;
    constexpr colocated_prong &operator=(colocated_prong &&) noexcept = default;

    explicit colocated_prong(task_index_t task_index, thread_index_t thread_index,
                             colocation_index_t colocation_index) noexcept
        : task(task_index), thread(thread_index), colocation(colocation_index) {}

    colocated_prong(prong<index_t> const &prong) noexcept : task(prong.task), thread(prong.thread), colocation(0) {}

    inline operator task_index_t() const noexcept { return task; }
    inline operator prong<index_t>() const noexcept { return prong<index_t> {task, thread}; }
};

using colocated_prong_t = colocated_prong<>; // ? Default prong type with `std::size_t` indices

/**
 *  @brief Describes a thread ID pinned to a specific NUMA node or QoS level.
 */
template <typename index_type_ = std::size_t>
struct colocated_thread {
    using index_t = index_type_;
    using thread_index_t = index_t;     // ? A.k.a. "core index" or "thread ID" in [0, threads_count)
    using colocation_index_t = index_t; // ? A.k.a. NUMA-specific QoS-specific "colocation ID"

    thread_index_t thread {0};
    colocation_index_t colocation {0};

    constexpr colocated_thread() noexcept = default;
    constexpr colocated_thread(colocated_thread &&) noexcept = default;
    constexpr colocated_thread(colocated_thread const &) noexcept = default;
    constexpr colocated_thread &operator=(colocated_thread const &) noexcept = default;
    constexpr colocated_thread &operator=(colocated_thread &&) noexcept = default;

    colocated_thread(thread_index_t thread_index, colocation_index_t colocation_index = 0) noexcept
        : thread(thread_index), colocation(colocation_index) {}

    inline operator thread_index_t() const noexcept { return thread; }
};

using colocated_thread_t = colocated_thread<>; // ? Default prong type with `std::size_t` indices

/**
 *  @brief Back-ports the C++ 23 `std::allocation_result`. Unlike STL, also contains the page size.
 *  @see https://en.cppreference.com/w/cpp/memory/allocator/allocate_at_least
 */
template <typename pointer_type_ = char, typename size_type_ = std::size_t>
struct allocation_result {
    using pointer_type = pointer_type_;
    using size_type = size_type_;

    pointer_type ptr {nullptr}; // ? Pointer to the allocated memory, or nullptr if allocation failed
    size_type count {0};        // ? Number of elements allocated, or 0 if allocation failed
    size_type bytes {0};        // ? Reports the total volume of memory allocated, in bytes
    size_type pages {0};        // ? Reports the number of memory pages allocated

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

    char *raw_ {nullptr};
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
        if (raw_) {
            allocator_.deallocate(raw_, bytes_total_);
            raw_ = nullptr;
        }
        objects_count_ = bytes_total_ = 0;
    }

  public:
    unique_padded_buffer() noexcept = default;

    explicit unique_padded_buffer(allocator_t const &alloc, std::size_t bytes_per_object = sizeof(object_t)) noexcept
        : bytes_per_object_(bytes_per_object), allocator_(alloc) {}

    unique_padded_buffer(unique_padded_buffer &&o) noexcept
        : raw_(std::exchange(o.raw_, nullptr)), objects_count_(std::exchange(o.objects_count_, 0)),
          bytes_per_object_(o.bytes_per_object_), bytes_total_(std::exchange(o.bytes_total_, 0)),
          allocator_(std::move(o.allocator_)) {}

    unique_padded_buffer &operator=(unique_padded_buffer &&o) noexcept {
        if (this != &o) {
            destroy_all();
            deallocate();
            raw_ = std::exchange(o.raw_, nullptr);
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

        std::size_t const total = new_objects_count * bytes_per_object_;
        auto new_result = allocator_.allocate_at_least(total);
        if (!new_result) return false;

        raw_ = new_result.ptr;
        objects_count_ = new_objects_count;
        bytes_total_ = new_result.bytes;

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
        while (flag_.test_and_set(std::memory_order_acquire)) micro_yield();
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
        while (flag_.exchange(true, std::memory_order_acquire)) micro_yield();
    }
    bool try_lock() noexcept { return !flag_.exchange(true, std::memory_order_acquire); }
    void unlock() noexcept { flag_.store(false, std::memory_order_release); }
};

#endif // FU_DETECT_CPP_20_

using spin_mutex_t = spin_mutex<>;

template <typename index_type_ = std::size_t>
struct indexed_range {
    using index_t = index_type_;

    index_t first {0};
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
            offset_ = static_cast<index_t>(offset_ + stride_);

            // Avoid modulo division by using wrap-around logic.
            if (offset_ >= length_) offset_ = static_cast<index_t>(offset_ - length_);
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

        inline iterator(index_t const start, index_t const length, index_t const stride,
                        index_t const elements_left) noexcept
            : start_(start), length_(length), stride_(stride), offset_(0), elements_left_(elements_left) {}

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
        : start_(start), length_(length), stride_(pick_stride(seed, length_)) {
        assert(length_ > 0 && "Length must be greater than zero, or expect division by zero");
    }

    iterator begin() const noexcept { return iterator(start_, length_, stride_, length_); }
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
 *  @brief Wraps the metadata needed for `for_n_dynamic` APIs for `broadcast_join` compatibility.
 *
 *  @section Scheduling Logic & Overflow Considerations
 *
 *  If we run a default for-loop at 1 Billion times per second on a 64-bit machine, then every 585 years
 *  of computational time we will wrap around the `std::size_t` capacity for the `prong.task` index.
 *  In case we `n + thread >= std::size_t(-1)`, a simple condition won't be enough.
 *  Alternatively, we can make sure, that each thread can do at least one increment of `progress_`
 *  without worrying about the overflow. The way to achieve that is to preprocess the trailing `threads`
 *  of elements externally, before entering this loop!
 *
 *  A simpler, potentially more logical implementation would keep the `progress_` as an internal atomic.
 *  That, however, places the variable on the stack of the calling thread, which may be different from the
 *  target NUMA node.
 */
template <typename fork_type_, typename index_type_>
class invoke_for_n_dynamic {
    fork_type_ fork_;
    std::atomic<index_type_> &progress_;
    index_type_ n_;
    index_type_ threads_;

  public:
    invoke_for_n_dynamic(index_type_ n, index_type_ threads, std::atomic<index_type_> &progress,
                         fork_type_ &&fork) noexcept
        : fork_(std::forward<fork_type_>(fork)), progress_(progress), n_(n), threads_(threads) {
        progress_.store(0, std::memory_order_release);
    }

    invoke_for_n_dynamic(invoke_for_n_dynamic &&other) noexcept // ? Need to manually define the `move` due to atomics
        : fork_(std::move(other.fork_)), progress_(other.progress_), n_(other.n_), threads_(other.threads_) {
        other.n_ = 0;
        assert(other.progress_.load(std::memory_order_acquire) == 0 && "Moving an in-progress fork is not allowed");
        progress_.store(0, std::memory_order_release);
    }

    void operator()(index_type_ const thread) noexcept {

        index_type_ const n_dynamic = n_ > threads_ ? n_ - threads_ : 0;
        assert((n_dynamic + threads_) >= n_dynamic && "Overflow detected");

        // Run (up to) one static prong on the current thread
        index_type_ const one_static_prong_index = static_cast<index_type_>(n_dynamic + thread);
        prong<index_type_> prong(one_static_prong_index, thread);
        if (one_static_prong_index < n_) fork_(prong);

        // The rest can be synchronized with a trivial atomic counter
        while (true) {
            prong.task = progress_.fetch_add(1, std::memory_order_relaxed);
            bool const beyond_last_prong = prong.task >= n_dynamic;
            if (beyond_last_prong) break;
            fork_(prong);
        }
    }
};

template <typename fork_type_, typename index_type_ = std::size_t>
constexpr bool can_be_for_thread_callback() noexcept {
    using fork_t = fork_type_;
    using index_t = index_type_;
#if FU_DETECT_CPP_17_ && defined(__cpp_lib_is_invocable)
    return std::is_nothrow_invocable_r_v<void, fork_t, colocated_thread<index_t>> ||
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
    return std::is_nothrow_invocable_r_v<void, fork_t, colocated_prong<index_t>> ||
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
    return std::is_nothrow_invocable_r_v<void, fork_t, colocated_prong<index_t>, index_t> ||
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

#pragma endregion - Helpers and Constants

#pragma region - Basic Pool

/**
 *  @brief Minimalistic STL-based non-resizable thread-pool for simultaneous blocking tasks.
 *
 *  This thread-pool @b can't:
 *  - dynamically @b resize: all threads must be stopped and re-initialized to grow/shrink.
 *  - @b re-enter: it can't be used recursively and will deadlock if you try to do so.
 *  - @b copy/move: the threads depend on the address of the parent structure.
 *  - handle @b exceptions: you must `try-catch` them yourself and return `void`.
 *  - @b stop early: assuming the user can do it better, knowing the task granularity.
 *  - @b overflow: as all APIs are aggressively tested with smaller index types.
 *
 *  This allows this thread-pool to be extremely lightweight and fast, @b without heap allocations
 *  and no expensive abstractions. It only uses `std::thread` and `std::atomic`, but avoids
 *  `std::function`, `std::future`, `std::promise`, `std::condition_variable`, that bring
 *  unnecessary overhead.
 *  @see https://ashvardanian.com/posts/beyond-openmp-in-cpp-rust/#four-horsemen-of-performance
 *
 *  Repeated operations are performed with a @b "weak" memory model, to leverage in-hardware
 *  support for atomic fence-less operations on Arm and IBM Power architectures. Most atomic
 *  counters use the "acquire-release" model, and some going further to "relaxed" model.
 *  @see https://en.cppreference.com/w/cpp/atomic/memory_order#Release-Acquire_ordering
 *  @see https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p2055r0.pdf
 *
 *  ------------------------------------------------------------------------------------------------
 *
 *  A minimal example, similar to `#pragma omp parallel` in OpenMP:
 *
 *  @code{.cpp}
 *  #include <cstdio> // `std::printf`
 *  #include <cstdlib> // `EXIT_FAILURE`, `EXIT_SUCCESS`
 *  #include <fork_union.hpp> // `basic_pool_t`
 *
 *  using fu = ashvardanian::fork_union;
 *  int main() {
 *      fu::basic_pool_t pool; // ? Alias to `fu::basic_pool<>` template
 *      if (!pool.try_spawn(std::thread::hardware_concurrency())) return EXIT_FAILURE;
 *      pool.for_threads([](std::size_t i) noexcept { std::printf("Hi from thread %zu\n", i); });
 *      return EXIT_SUCCESS;
 *  }
 *  @endcode
 *
 *  Unlike OpenMP, however, separate thread-pools can be created isolating work and resources.
 *  This is handy when when some logic has to be split between "performance" & "efficiency" cores,
 *  between different NUMA nodes, between GUI and background tasks, etc. It may look like this:
 *
 *  @code{.cpp}
 *  #include <cstdio> // `std::printf`
 *  #include <cstdlib> // `EXIT_FAILURE`, `EXIT_SUCCESS`
 *  #include <fork_union.hpp> // `basic_pool_t`
 *
 *  using fu = ashvardanian::fork_union;
 *  int main() {
 *      fu::basic_pool_t first_pool, second_pool;
 *      if (!first_pool.try_spawn(2) || !second_pool.try_spawn(2, fu::caller_exclusive_k)) return EXIT_FAILURE;
 *      auto join = second_pool.for_threads([](std::size_t i) noexcept { poll_ssd(i); });
 *      first_pool.for_threads([](std::size_t i) noexcept { poll_nic(i); });
 *      join.wait(); // ! Wait for the second pool to finish
 *      return EXIT_SUCCESS;
 *  }
 *  @endcode
 *
 *  ------------------------------------------------------------------------------------------------
 *
 *  @tparam allocator_type_ The type of the allocator to be used for the thread pool.
 *  @tparam micro_yield_type_ The type of the yield function to be used for busy-waiting.
 *  @tparam index_type_ Use `std::size_t`, but or a smaller type for debugging.
 *  @tparam alignment_ The alignment of the thread pool. Defaults to `default_alignment_k`.
 */
template <                                                  //
    typename allocator_type_ = std::allocator<std::thread>, //
    typename micro_yield_type_ = standard_yield_t,          //
    typename index_type_ = std::size_t,                     //
    std::size_t alignment_ = default_alignment_k            //
    >
class basic_pool {

  public:
    using allocator_t = allocator_type_;
    using micro_yield_t = micro_yield_type_;
    static_assert(std::is_nothrow_invocable_r<void, micro_yield_t>::value,
                  "Yield must be callable w/out arguments & return void");
    static constexpr std::size_t alignment_k = alignment_;
    static_assert(is_power_of_two(alignment_k), "Alignment must be a power of 2");

    using index_t = index_type_;
    static_assert(std::is_unsigned<index_t>::value, "Index type must be an unsigned integer");
    using epoch_index_t = index_t;      // ? A.k.a. number of previous API calls in [0, UINT_MAX)
    using thread_index_t = index_t;     // ? A.k.a. "core index" or "thread ID" in [0, threads_count)
    using colocation_index_t = index_t; // ? A.k.a. "NUMA node ID" in [0, numa_nodes_count)
    using indexed_split_t = indexed_split<index_t>;
    using prong_t = prong<index_t>;
    using colocated_thread_t = colocated_thread<index_t>;

    using punned_fork_context_t = void *;                                 // ? Pointer to the on-stack lambda
    using trampoline_t = void (*)(punned_fork_context_t, thread_index_t); // ? Wraps lambda's `operator()`

  private:
    // Thread-pool-specific variables:
    allocator_t allocator_ {};
    std::thread *workers_ {nullptr};
    thread_index_t threads_count_ {0};
    caller_exclusivity_t exclusivity_ {caller_inclusive_k}; // ? Whether the caller thread is included in the count
    std::size_t sleep_length_micros_ {0}; // ? How long to sleep in microseconds when waiting for tasks
    alignas(alignment_k) std::atomic<mood_t> mood_ {mood_t::grind_k};

    // Task-specific variables:
    punned_fork_context_t fork_state_ {nullptr}; // ? Pointer to the users lambda
    trampoline_t fork_trampoline_ {nullptr};     // ? Calls the lambda
    alignas(alignment_k) std::atomic<thread_index_t> threads_to_sync_ {0};
    alignas(alignment_k) std::atomic<epoch_index_t> epoch_ {0};

    alignas(alignment_k) std::atomic<index_t> dynamic_progress_ {0}; // ? Only used in `for_n_dynamic`

  public:
    basic_pool(basic_pool &&) = delete;
    basic_pool(basic_pool const &) = delete;
    basic_pool &operator=(basic_pool &&) = delete;
    basic_pool &operator=(basic_pool const &) = delete;

    basic_pool(allocator_t const &alloc = {}) noexcept : allocator_(alloc) {}
    ~basic_pool() noexcept { terminate(); }

    /**
     *  @brief Estimates the amount of memory managed by this pool handle and internal structures.
     *  @note This API is @b not synchronized.
     */
    std::size_t memory_usage() const noexcept { return sizeof(basic_pool) + threads_count() * sizeof(std::thread); }

    /** @brief Checks if the thread-pool's core synchronization points are lock-free. */
    bool is_lock_free() const noexcept { return mood_.is_lock_free() && threads_to_sync_.is_lock_free(); }

    /**
     *  @brief Returns the NUMA node ID this thread-pool is pinned to.
     *  @retval -1 as this thread-pool is not NUMA-aware.
     */
    constexpr numa_node_id_t numa_node_id() const noexcept { return -1; }

    /**
     *  @brief Returns the first thread index in the thread-pool.
     *  @retval 0 as this pool isn't intended for colocated/distributed topologies.
     */
    constexpr thread_index_t first_thread() const noexcept { return 0; }

    /** @brief Exposes access to the internal atomic progress counter. Use with caution. */
    std::atomic<index_t> &unsafe_dynamic_progress_ref() noexcept { return dynamic_progress_; }

#pragma region Core API

    /**
     *  @brief Returns the number of threads in the thread-pool, including the main thread.
     *  @retval 0 if the thread-pool is not initialized, 1 if only the main thread is used.
     *  @note This API is @b not synchronized.
     */
    thread_index_t threads_count() const noexcept { return threads_count_; }

    /**
     *  @brief Reports if the current calling thread will be used for broadcasts.
     *  @note This API is @b not synchronized.
     */
    caller_exclusivity_t caller_exclusivity() const noexcept { return exclusivity_; }

    /**
     *  @brief Creates a thread-pool with the given number of threads.
     *  @param[in] threads The number of threads to be used.
     *  @param[in] exclusivity Should we count the calling thread as one of the threads?
     *  @retval false if the number of threads is zero or the "workers" allocation failed.
     *  @retval true if the thread-pool was created successfully, started, and is ready to use.
     *  @note This is the de-facto @b constructor - you only call it again after `terminate`.
     */
    bool try_spawn(                   //
        thread_index_t const threads, //
        caller_exclusivity_t const exclusivity = caller_inclusive_k) noexcept {

        if (threads == 0) return false;        // ! Can't have zero threads working on something
        if (threads_count_ != 0) return false; // ! Already initialized

        bool const use_caller_thread = exclusivity == caller_inclusive_k;
        if (threads == 1 && use_caller_thread) {
            threads_count_ = 1;
            return true; // ! The current thread will always be used
        }

        // Allocate the thread pool
        thread_index_t const worker_threads = threads - use_caller_thread;
        std::thread *const workers = allocator_.allocate(worker_threads);
        if (!workers) return false; // ! Allocation failed

        // Before we start the threads, make sure we set some of the shared
        // state variables that will be used in the `_worker_loop` function.
        workers_ = workers;
        threads_count_ = threads;
        exclusivity_ = exclusivity;
        mood_.store(mood_t::grind_k, std::memory_order_release);
        auto reset_on_failure = [&]() noexcept {
            allocator_.deallocate(workers, threads);
            workers_ = nullptr;
            threads_count_ = 0;
        };

        // Initializing the thread pool can fail for all kinds of reasons,
        // that the `std::thread` documentation describes as "implementation-defined".
        // https://en.cppreference.com/w/cpp/thread/thread/thread
        for (thread_index_t i = 0; i < worker_threads; ++i) {
            try {
                thread_index_t const i_with_caller = i + use_caller_thread;
                new (&workers[i]) std::thread([this, i_with_caller] { _worker_loop(i_with_caller); });
            }
            catch (...) {
                mood_.store(mood_t::die_k, std::memory_order_release);
                for (thread_index_t j = 0; j < i; ++j) {
                    workers[j].join(); // ? Wait for the thread to exit
                    workers[j].~thread();
                }
                reset_on_failure();
                return false;
            }
        }

        return true;
    }

    /**
     *  @brief Executes a @p fork function in parallel on all threads.
     *  @param[in] fork The callback object, receiving the thread index as an argument.
     *  @return `broadcast_join` synchronization point that waits in the destructor.
     *  @note Even in the `caller_exclusive_k` mode, can be called from just one thread!
     *  @sa For advanced resource management, consider `unsafe_for_threads` and `unsafe_join`.
     */
    template <typename fork_type_>
    FU_REQUIRES_((can_be_for_thread_callback<fork_type_, index_t>()))
    broadcast_join<basic_pool, fork_type_> for_threads(fork_type_ &&fork) noexcept {
        return {*this, std::forward<fork_type_>(fork)};
    }

    /**
     *  @brief Executes a @p fork function in parallel on all threads, not waiting for the result.
     *  @param[in] fork The callback @b reference, receiving the thread index as an argument.
     *  @sa Use in conjunction with `unsafe_join`.
     */
    template <typename fork_type_>
    FU_REQUIRES_((can_be_for_thread_callback<fork_type_, index_t>()))
    void unsafe_for_threads(fork_type_ &fork) noexcept {

        thread_index_t const threads = threads_count();
        assert(threads != 0 && "Thread pool not initialized");
        caller_exclusivity_t const exclusivity = caller_exclusivity();
        bool const use_caller_thread = exclusivity == caller_inclusive_k;

        // Optional check: even in exclusive mode, only one thread can call this function.
        assert((use_caller_thread || threads_to_sync_.load(std::memory_order_acquire) == 0) &&
               "The broadcast function can't be called concurrently or recursively");

        // Configure "fork" details
        fork_state_ = std::addressof(fork);
        fork_trampoline_ = &_call_as_lambda<fork_type_>;
        threads_to_sync_.store(threads - use_caller_thread, std::memory_order_relaxed);

        // We are most likely already "grinding", but in the unlikely case we are not,
        // let's wake up from the "chilling" state with relaxed semantics. Assuming the sleeping
        // logic for the workers also checks the epoch counter, no synchronization is needed and
        // no immediate wake-up is required.
        mood_t may_be_chilling = mood_t::chill_k;
        mood_.compare_exchange_weak(          //
            may_be_chilling, mood_t::grind_k, //
            std::memory_order_relaxed, std::memory_order_relaxed);
        epoch_.fetch_add(1, std::memory_order_release); // ? Wake up sleepers
    }

    /** @brief Blocks the calling thread until the currently broadcasted task finishes. */
    void unsafe_join() noexcept {
        caller_exclusivity_t const exclusivity = caller_exclusivity();
        bool const use_caller_thread = exclusivity == caller_inclusive_k;

        // Execute on the current "main" thread
        if (use_caller_thread) fork_trampoline_(fork_state_, static_cast<thread_index_t>(0));

        // Actually wait for everyone to finish
        micro_yield_t micro_yield;
        while (threads_to_sync_.load(std::memory_order_acquire)) micro_yield();
    }

#pragma endregion Core API

#pragma region Control Flow

    /**
     *  @brief Stops all threads and deallocates the thread-pool after the last call finishes.
     *  @note Can be called from @b any thread at any time.
     *  @note Must `try_spawn` again to re-use the pool.
     *
     *  When and how @b NOT to use this function:
     *  - as a synchronization point between concurrent tasks.
     *
     *  When and how to use this function:
     *  - as a de-facto @b destructor, to stop all threads and deallocate the pool.
     *  - when you want to @b restart with a different number of threads.
     */
    void terminate() noexcept {
        if (threads_count_ == 0) return; // ? Uninitialized

        caller_exclusivity_t const exclusivity = caller_exclusivity();
        bool const use_caller_thread = exclusivity == caller_inclusive_k;
        if (threads_count_ == 1 && use_caller_thread) {
            threads_count_ = 0;
            return; // ? No worker threads to join
        }
        assert(threads_to_sync_.load(std::memory_order_seq_cst) == 0); // ! No tasks must be running

        // Notify all worker threads...
        mood_.store(mood_t::die_k, std::memory_order_release);

        // ... and wait for them to finish
        thread_index_t const worker_threads = threads_count_ - use_caller_thread;
        for (thread_index_t i = 0; i != worker_threads; ++i) {
            workers_[i].join();    // ? Wait for the thread to finish
            workers_[i].~thread(); // ? Call destructor
        }

        // Deallocate the thread pool
        allocator_.deallocate(workers_, worker_threads);

        // Prepare for future spawns
        threads_count_ = 0;
        workers_ = nullptr;
        _reset_fork();
        mood_.store(mood_t::grind_k, std::memory_order_relaxed);
        epoch_.store(0, std::memory_order_relaxed);
    }

    /**
     *  @brief Transitions "workers" to a sleeping state, waiting for a wake-up call.
     *  @param[in] wake_up_periodicity_micros How often to check for new work in microseconds.
     *  @note Can only be called @b between the tasks for a single thread. No synchronization is performed.
     *
     *  This function may be used in some batch-processing operations when we clearly understand
     *  that the next task won't be arriving for a while and power can be saved without major
     *  latency penalties.
     *
     *  It may also be used in a high-level Python or JavaScript library offloading some parallel
     *  operations to an underlying C++ engine, where latency is irrelevant.
     */
    void sleep(std::size_t wake_up_periodicity_micros) noexcept {
        assert(wake_up_periodicity_micros > 0 && "Sleep length must be positive");
        sleep_length_micros_ = wake_up_periodicity_micros;
        mood_.store(mood_t::chill_k, std::memory_order_release);
    }

    /** @brief Helper function to create a spin mutex with same yield characteristics. */
    static spin_mutex<micro_yield_t, alignment_k> make_mutex() noexcept { return {}; }

#pragma endregion Control Flow

#pragma region Indexed Task Scheduling

    /**
     *  @brief Distributes @p `n` similar duration calls between threads in slices, as opposed to individual indices.
     *  @param[in] n The total length of the range to split between threads.
     *  @param[in] fork The callback object, receiving the first @b `prong_t` and the slice length.
     */
    template <typename fork_type_ = dummy_lambda_t>
    FU_REQUIRES_((can_be_for_slice_callback<fork_type_, index_t>()))
    broadcast_join<basic_pool, invoke_for_slices<fork_type_, index_t>> //
        for_slices(index_t const n, fork_type_ &&fork) noexcept {

        return {*this, {n, threads_count(), std::forward<fork_type_>(fork)}};
    }

    /**
     *  @brief Distributes @p `n` similar duration calls between threads.
     *  @param[in] n The number of times to call the @p fork.
     *  @param[in] fork The callback object, receiving @b `prong_t` or a call index as an argument.
     *
     *  Is designed for a "balanced" workload, where all threads have roughly the same amount of work.
     *  @sa `for_n_dynamic` for a more dynamic workload.
     *  The @p fork is called @p `n` times, and each thread receives a slice of consecutive tasks.
     *  @sa `for_slices` if you prefer to receive workload slices over individual indices.
     */
    template <typename fork_type_ = dummy_lambda_t>
    FU_REQUIRES_((can_be_for_task_callback<fork_type_, index_t>()))
    broadcast_join<basic_pool, invoke_for_n<fork_type_, index_t>> //
        for_n(index_t const n, fork_type_ &&fork) noexcept {

        return {*this, {n, threads_count(), std::forward<fork_type_>(fork)}};
    }

    /**
     *  @brief Executes uneven tasks on all threads, greedying for work.
     *  @param[in] n The number of times to call the @p fork.
     *  @param[in] fork The callback object, receiving the `prong_t` or the task index as an argument.
     *  @sa `for_n` for a more "balanced" evenly-splittable workload.
     */
    template <typename fork_type_ = dummy_lambda_t>
    FU_REQUIRES_((can_be_for_task_callback<fork_type_, index_t>()))
    broadcast_join<basic_pool, invoke_for_n_dynamic<fork_type_, index_t>> //
        for_n_dynamic(index_t const n, fork_type_ &&fork) noexcept {

        return {*this, {n, threads_count(), dynamic_progress_, std::forward<fork_type_>(fork)}};
    }

#pragma endregion Indexed Task Scheduling

#pragma region Colocations Compatibility

    /**
     *  @brief Number of individual sub-pool with the same NUMA-locality and QoS.
     *  @retval 1 constant for compatibility.
     */
    constexpr index_t colocations_count() const noexcept { return 1; }

    /**
     *  @brief Returns the number of threads in one NUMA-specific local @b colocation.
     *  @return Same value as `threads_count()`, as we only support one colocation.
     */
    thread_index_t threads_count(FU_MAYBE_UNUSED_ index_t colocation_index) const noexcept {
        assert(colocation_index == 0 && "Only one colocation is supported");
        return threads_count();
    }

    /**
     *  @brief Converts a @p `global_thread_index` to a local thread index within a @b colocation.
     *  @return Same value as `global_thread_index`, as we only support one colocation.
     */
    constexpr thread_index_t thread_local_index(thread_index_t global_thread_index,
                                                FU_MAYBE_UNUSED_ index_t colocation_index) const noexcept {
        assert(colocation_index == 0 && "Only one colocation is supported");
        return global_thread_index;
    }

#pragma endregion Colocations Compatibility

  private:
    void _reset_fork() noexcept {
        fork_state_ = nullptr;
        fork_trampoline_ = nullptr;
    }

    /**
     *  @brief A trampoline function that is used to call the user-defined lambda.
     *  @param[in] punned_lambda_pointer The pointer to the user-defined lambda.
     *  @param[in] prong The index of the thread & task index packed together.
     */
    template <typename fork_type_>
    static void _call_as_lambda(punned_fork_context_t punned_lambda_pointer, thread_index_t thread_index) noexcept {
        fork_type_ &lambda_object = *static_cast<fork_type_ *>(punned_lambda_pointer);
        lambda_object(colocated_thread_t {thread_index, 0});
    }

    /**
     *  @brief The worker thread loop that is called by each of `this->workers_`.
     *  @param[in] thread_index The index of the thread that is executing this function.
     */
    void _worker_loop(thread_index_t const thread_index) noexcept {
        caller_exclusivity_t const exclusivity = caller_exclusivity();
        bool const use_caller_thread = exclusivity == caller_inclusive_k;
        if (use_caller_thread) assert(thread_index != 0 && "The zero index is for the main thread, not worker!");

        epoch_index_t last_epoch = 0;
        while (true) {
            // Wait for either: a new ticket or a stop flag
            epoch_index_t new_epoch;       // Will definitely be initialized in the loop
            mood_t mood = mood_t::grind_k; // May not be initialized in the loop
            micro_yield_t micro_yield;
            while ((new_epoch = epoch_.load(std::memory_order_acquire)) == last_epoch &&
                   (mood = mood_.load(std::memory_order_acquire)) == mood_t::grind_k)
                micro_yield();

            if (fu_unlikely_(mood == mood_t::die_k)) break;
            if (fu_unlikely_(mood == mood_t::chill_k) && (new_epoch == last_epoch)) {
                std::this_thread::sleep_for(std::chrono::microseconds(sleep_length_micros_));
                continue;
            }

            fork_trampoline_(fork_state_, thread_index);
            last_epoch = new_epoch;

            // ! The decrement must come after the task is executed
            FU_MAYBE_UNUSED_ thread_index_t const before_decrement =
                threads_to_sync_.fetch_sub(1, std::memory_order_release);
            assert(before_decrement > 0 && "We can't be here if there are no worker threads");
        }
    }
};

using basic_pool_t = basic_pool<>;

#pragma region Concepts
#if FU_DETECT_CONCEPTS_

struct broadcasted_noop_t {
    template <typename index_type_>
    void operator()(index_type_) const noexcept
        requires(std::unsigned_integral<index_type_> && std::convertible_to<index_type_, std::size_t>)
    {}
};

template <typename pool_type_>
concept is_pool = //
    std::unsigned_integral<decltype(std::declval<pool_type_ const &>().threads_count())> &&
    std::convertible_to<decltype(std::declval<pool_type_ const &>().threads_count()), std::size_t> &&
    requires(pool_type_ &p) {
        { p.for_threads(broadcasted_noop_t {}) }; // Passing the callback by value
    } &&                                          //
    requires(pool_type_ &p, broadcasted_noop_t const &noop) {
        { p.for_threads(noop) }; // Passing the callback by const reference
    } &&                         //
    requires(pool_type_ &p, broadcasted_noop_t &noop) {
        { p.for_threads(noop) }; // Passing the callback by non-const reference
    };

template <typename pool_type_>
concept is_unsafe_pool =   //
    is_pool<pool_type_> && //
    requires(pool_type_ &p, broadcasted_noop_t &noop) {
        { p.unsafe_for_threads(noop) } -> std::same_as<void>;
    } && //
    requires(pool_type_ &p) {
        { p.unsafe_join() } -> std::same_as<void>;
    };

#endif // FU_DETECT_CONCEPTS_
#pragma endregion Concepts

#pragma endregion - Basic Pool

#pragma region - Hardware Friendly Yield

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

#if FU_ENABLE_NUMA
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

#endif // FU_ENABLE_NUMA

    return caps;
}

#pragma endregion - Hardware Friendly Yield

#pragma region - NUMA Pools

enum numa_pin_granularity_t {
    numa_pin_to_core_k = 0,
    numa_pin_to_node_k,
};

struct ram_page_setting_t {
    std::size_t bytes_per_page {0};  // ? Huge page size in bytes, e.g. 4 KB, 2 MB, or 1 GB
    std::size_t available_pages {0}; // ? Number of pages available for this size, 0 if not available
    std::size_t free_pages {0};      // ? Number of pages available and unused, 0 if not available
};

/**
 *  @brief Fetches the socket ID for a given CPU core.
 *  @param[in] core_id The CPU core ID to query.
 *  @retval Socket ID (>= 0) if successful.
 *  @retval -1 if failed.
 */
FU_MAYBE_UNUSED_ static inline numa_socket_id_t get_socket_id_for_core(
    FU_MAYBE_UNUSED_ numa_core_id_t core_id) noexcept {

    int socket_id = -1;

#if defined(__linux__)
    char socket_path[256];
    int path_result = std::snprintf(      //
        socket_path, sizeof(socket_path), //
        "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", core_id);
    if (path_result < 0 || static_cast<std::size_t>(path_result) >= sizeof(socket_path)) return -1; // ? Path too long

    FILE *socket_file = ::fopen(socket_path, "r");
    if (!socket_file) return -1; // ? Can't read socket info

    if (::fscanf(socket_file, "%d", &socket_id) != 1) socket_id = -1; // ? Failed to parse
    ::fclose(socket_file);
#endif

    return socket_id;
}

/**
 *  @brief Fetches the RAM page size in bytes.
 *  @retval The size of a memory page in bytes, typically 4096 on most systems.
 *  @note On Linux, this is the system page size, which may differ from Huge Pages sizes.
 */
FU_MAYBE_UNUSED_ static inline std::size_t get_ram_page_size() noexcept {
#if FU_ENABLE_NUMA
    return static_cast<std::size_t>(::numa_pagesize());
#elif defined(__unix__) || defined(__unix) || defined(unix) || defined(__APPLE__)
    return static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
#else
    return 4096;
#endif
}

/**
 *  @brief Fetches the total RAM amount available on the system in bytes.
 *  @retval Total system RAM in bytes, or 0 if detection fails.
 *  @note This function provides cross-platform detection of total physical memory.
 */
FU_MAYBE_UNUSED_ static inline std::size_t get_ram_total_volume() noexcept {
#if defined(__linux__)
    // On Linux, read from /proc/meminfo
    FILE *meminfo_file = ::fopen("/proc/meminfo", "r");
    if (!meminfo_file) return 0;

    char line[256];
    while (::fgets(line, sizeof(line), meminfo_file)) {
        if (::strncmp(line, "MemTotal:", 9) == 0) {
            std::size_t memory_kb = 0;
            if (::sscanf(line, "MemTotal: %zu kB", &memory_kb) == 1) {
                ::fclose(meminfo_file);
                return memory_kb * 1024; // Convert kB to bytes
            }
        }
    }
    ::fclose(meminfo_file);
    return 0;
#elif defined(__APPLE__)
    // On macOS, use sysctl
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    std::uint64_t memory_bytes = 0;
    std::size_t size = sizeof(memory_bytes);
    if (::sysctl(mib, 2, &memory_bytes, &size, nullptr, 0) == 0) return static_cast<std::size_t>(memory_bytes);
    return 0;
#elif defined(_WIN32)
    // On Windows, use GlobalMemoryStatusEx
    MEMORYSTATUSEX mem_status;
    mem_status.dwLength = sizeof(mem_status);
    if (::GlobalMemoryStatusEx(&mem_status)) return static_cast<std::size_t>(mem_status.ullTotalPhys);
    return 0;
#elif defined(__unix__) || defined(__unix) || defined(unix)
    // On other Unix systems, try sysconf
    long pages = ::sysconf(_SC_PHYS_PAGES);
    long page_size = ::sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) return static_cast<std::size_t>(pages) * static_cast<std::size_t>(page_size);
    return 0;
#else
    // Fallback: return 0 if platform is not supported
    return 0;
#endif
}

/**
 *  @brief Describes the configured & supported (by OS & CPU) memory pages sizes.
 *
 *  This class avoids HugeTLBfs in favor of a direct access to the @b `/sys` filesystem.
 *  Aside from fetching the stats, it also allows us to change settings if admin privileges
 *  are granted to running process.
 *
 *  @section Huge Pages & Transparent Huge Pages
 *
 *  Virtual Address Space (VAS) is divided into pages, typically 4 KB in size.
 *  Converting a virtual address to a physical address requires a page table lookup.
 *  Think of it as a hash table... and as everyone knows, hash table lookups and updates
 *  aren't free, so most chips have a "Translation Lookaside Buffer" @b (TLB) cache
 *  as part of the "Memory Management Unit" @b (MMU) to speed up the process.
 *
 *  To keep it fast, in Big Data applications, one would like to use larger pages,
 *  to reduce the number of distinct entries in the TLB cache. Going from 4 KB to
 *  2 MB or 1 GB "Huge Pages" @b (HPs), reduces the table size by 512 or 262K times,
 *  respectively.
 *
 *  To benefit from those, some applications rely on "Transparent Huge Pages" @b (THP),
 *  which are automatically allocated by the kernel. Such implicit behaviour isn't
 *  great for performance-oriented applications, so the `linux_numa_allocator` provides
 *  a @b `fetch_max_huge_size` API.
 *
 *  @see https://docs.kernel.org/admin-guide/mm/hugetlbpage.html
 */
template <std::size_t max_page_sizes_ = 4>
class ram_page_settings {
    static constexpr std::size_t max_page_sizes_k = max_page_sizes_;
    std::array<ram_page_setting_t, max_page_sizes_k> sizes_ {0}; // ? Huge page sizes in bytes
    std::size_t count_sizes_ {0};                                // ? Number of supported huge page sizes
    std::size_t total_memory_bytes_ {0};                         // ? Total memory available on this NUMA node
  public:
    /**
     *  @brief Finds the largest Huge Pages size available for the given NUMA node.
     */
    ram_page_setting_t largest_free() const noexcept {
        if (!count_sizes_) return {};
        ram_page_setting_t largest = sizes_[0];
        for (std::size_t i = 1; i < count_sizes_; ++i)
            if (sizes_[i].free_pages > largest.free_pages) largest = sizes_[i];
        return largest;
    }

    /**
     *  @brief Fetches all available huge page sizes for the given NUMA node.
     *  @note Kernel support doesn't mean that pages of that size have a valid mount point.
     */
    bool try_harvest(FU_MAYBE_UNUSED_ numa_node_id_t node_id) noexcept {
        assert(node_id >= 0 && "NUMA node ID must be non-negative");

#if FU_ENABLE_NUMA // We need Linux for `opendir`

        std::size_t count_sizes = 0; // ? Number of sizes found

        // Build path to NUMA node's hugepages directory
        char hugepages_path[256];
        int path_result = std::snprintf(            //
            hugepages_path, sizeof(hugepages_path), //
            "/sys/devices/system/node/node%d/hugepages", node_id);
        if (path_result < 0 || static_cast<std::size_t>(path_result) >= sizeof(hugepages_path))
            return false; // ? Path too long

        DIR *hugepages_dir = ::opendir(hugepages_path);
        if (!hugepages_dir) return false; // ? Can't open NUMA node hugepages directory

        struct dirent *entry;
        while ((entry = ::readdir(hugepages_dir)) != nullptr && count_sizes < max_page_sizes_k) {
            // Look for directories named "hugepages-*kB"
            if (entry->d_type != DT_DIR) continue;
            if (::strncmp(entry->d_name, "hugepages-", 10) != 0) continue;

            // Extract size from directory name (e.g., "hugepages-2048kB" -> 2048)
            char const *size_start = entry->d_name + 10; // ? Skip "hugepages-"
            char *size_end = nullptr;
            std::size_t bytes_per_page_kb = static_cast<std::size_t>(::strtoull(size_start, &size_end, 10));

            // Verify the suffix is "kB"
            if (!size_end || std::strcmp(size_end, "kB") != 0) continue;
            if (bytes_per_page_kb == 0) continue; // ? Invalid size

            std::size_t const bytes_per_page = bytes_per_page_kb * 1024;

            // Read NUMA-node-specific huge page statistics
            char nr_hugepages_path[512];
            char free_hugepages_path[512];

            path_result = std::snprintf(                      //
                nr_hugepages_path, sizeof(nr_hugepages_path), //
                "%s/%s/nr_hugepages", hugepages_path, entry->d_name);
            if (path_result < 0 || static_cast<std::size_t>(path_result) >= sizeof(nr_hugepages_path))
                continue; // ? Path too long

            path_result = std::snprintf(                          //
                free_hugepages_path, sizeof(free_hugepages_path), //
                "%s/%s/free_hugepages", hugepages_path, entry->d_name);
            if (path_result < 0 || static_cast<std::size_t>(path_result) >= sizeof(free_hugepages_path))
                continue; // ? Path too long

            // Read allocated huge pages count
            FILE *nr_file = ::fopen(nr_hugepages_path, "r");
            if (!nr_file) continue; // ? Can't read allocation count

            std::size_t allocated_pages = 0;
            std::size_t free_pages = 0;
            if (::fscanf(nr_file, "%zu", &allocated_pages) != 1) {
                ::fclose(nr_file);
                continue; // ? Failed to parse allocated count
            }
            ::fclose(nr_file);

            // Read free huge pages count
            FILE *free_file = ::fopen(free_hugepages_path, "r");
            if (free_file) {
                if (::fscanf(free_file, "%zu", &free_pages) != 1) {
                    free_pages = 0; // ? Assume none are free if parsing fails
                }
                ::fclose(free_file);
            }

            // Add to our list with NUMA node information
            sizes_[count_sizes].bytes_per_page = bytes_per_page;
            sizes_[count_sizes].available_pages = allocated_pages;
            sizes_[count_sizes].free_pages = free_pages;
            ++count_sizes;
        }
        ::closedir(hugepages_dir);

        // Read total memory for this NUMA node from meminfo
        char meminfo_path[256];
        path_result =
            std::snprintf(meminfo_path, sizeof(meminfo_path), "/sys/devices/system/node/node%d/meminfo", node_id);
        if (path_result > 0 && static_cast<std::size_t>(path_result) < sizeof(meminfo_path)) {
            FILE *meminfo_file = ::fopen(meminfo_path, "r");
            if (meminfo_file) {
                char line[256];
                while (::fgets(line, sizeof(line), meminfo_file)) {
                    if (::strncmp(line, "Node ", 5) == 0 && ::strstr(line, " MemTotal:")) {
                        // Parse line like "Node 0 MemTotal:    32768000 kB"
                        std::size_t memory_kb = 0;
                        if (::sscanf(line, "Node %*d MemTotal: %zu kB", &memory_kb) == 1) {
                            total_memory_bytes_ = memory_kb * 1024; // Convert kB to bytes
                            break;
                        }
                    }
                }
                ::fclose(meminfo_file);
            }
        }

        count_sizes_ = count_sizes;
        return true;
#else
        fu_unused_(node_id);
        return false;
#endif
    }

    std::size_t size() const noexcept { return count_sizes_; }
    std::size_t total_memory_bytes() const noexcept { return total_memory_bytes_; }
    ram_page_setting_t const *begin() const noexcept { return sizes_.data(); }
    ram_page_setting_t const *end() const noexcept { return sizes_.data() + count_sizes_; }
    ram_page_setting_t const &operator[](std::size_t const index) const noexcept {
        assert(index < count_sizes_ && "Index is out of bounds");
        return sizes_[index];
    }

    /**
     *  @brief Attempts to reserve huge pages of a specific size on the current NUMA node.
     *  @param[in] page_size_bytes The size of huge pages to reserve (must match an available size)
     *  @param[in] num_pages Number of pages to reserve
     *  @return true if reservation was successful, false otherwise
     *  @note Requires root privileges or appropriate capabilities
     */
    bool try_change(numa_node_id_t node_id, std::size_t page_size_bytes, std::size_t num_pages) noexcept {
        assert(node_id >= 0 && "NUMA node ID must be non-negative");

        // Find the matching page size entry
        std::size_t page_index = count_sizes_;
        for (std::size_t i = 0; i < count_sizes_; ++i) {
            if (sizes_[i].bytes_per_page == page_size_bytes) {
                page_index = i;
                break;
            }
        }
        if (page_index >= count_sizes_) return false; // ? Page size not found

        // Calculate the page size in kB for the directory name
        std::size_t const page_size_kb = page_size_bytes / 1024;

        // Build path to the nr_hugepages file
        char nr_hugepages_path[512];
        int const path_result = std::snprintf(                                        //
            nr_hugepages_path, sizeof(nr_hugepages_path),                             //
            "/sys/devices/system/node/node%d/hugepages/hugepages-%zukB/nr_hugepages", //
            node_id, page_size_kb);

        if (path_result < 0 || static_cast<std::size_t>(path_result) >= sizeof(nr_hugepages_path))
            return false; // ? Path too long

        // Write the new reservation count
        FILE *nr_file = ::fopen(nr_hugepages_path, "w");
        if (!nr_file) return false; // ? Can't open for writing (likely permissions issue)

        bool const update_success = (::fprintf(nr_file, "%zu", num_pages) > 0);
        ::fclose(nr_file);
        if (!update_success) return false; // ? Failed to write the number of pages

        // Refresh our internal state if write was successful
        return try_harvest(node_id);
    }
};

using ram_page_settings_t = ram_page_settings<>;

/**
 *  @brief Describes a NUMA node, containing its ID, memory size, and core IDs.
 *  @sa Views different slices of the `numa_topology` structure.
 */
template <std::size_t max_page_sizes_ = 4>
struct numa_node {
    static constexpr std::size_t max_page_sizes_k = max_page_sizes_;

    numa_node_id_t node_id {-1};                       // ? Unique NUMA node ID, in [0, numa_max_node())
    numa_socket_id_t socket_id {-1};                   // ? Physical CPU socket ID
    std::size_t memory_size {0};                       // ? RAM volume in bytes
    numa_core_id_t const *first_core_id {nullptr};     // ? Pointer to the first core ID in the `core_ids` array
    std::size_t core_count {0};                        // ? Number of items in `core_ids` array
    ram_page_settings<max_page_sizes_k> page_sizes {}; // ? Huge page sizes available on this NUMA node
};

using numa_node_t = numa_node<>;

template <typename value_type_, typename comparator_type_ = std::less<value_type_>>
void bubble_sort(value_type_ *array, std::size_t size, comparator_type_ comp = {}) noexcept {
    assert(array != nullptr && "Array must not be null");
    for (std::size_t i = 0; i < size - 1; ++i)
        for (std::size_t j = 0; j < size - i - 1; ++j)
            if (comp(array[j + 1], array[j])) std::swap(array[j], array[j + 1]);
}

/**
 *  @brief NUMA topology descriptor: describing memory pools and core counts next to them.
 *
 *  Uses dynamic memory to store the NUMA nodes and their cores. Assuming we may soon have
 *  Intel "Sierra Forest"-like CPUs with 288 cores with up to 8 sockets per node, this structure
 *  can easily grow to 10 KB.
 */
template <std::size_t max_page_sizes_ = 4, typename allocator_type_ = std::allocator<char>>
struct numa_topology {

    using allocator_t = allocator_type_;
    using cores_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<int>;
    using nodes_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<numa_node_t>;
    static constexpr std::size_t max_page_sizes_k = max_page_sizes_;

  private:
    allocator_t allocator_ {};
    numa_node_t *nodes_ {nullptr};
    numa_core_id_t *node_core_ids_ {nullptr}; // ? Unsigned integers in [0, threads_count), grouped by NUMA node
    std::size_t nodes_count_ {0};             // ? Number of NUMA nodes
    std::size_t cores_count_ {0};             // ? Total number of cores in all nodes

  public:
    constexpr numa_topology() noexcept = default;
    numa_topology(numa_topology &&o) noexcept
        : allocator_(std::move(o.allocator_)), nodes_(o.nodes_), node_core_ids_(o.node_core_ids_),
          nodes_count_(o.nodes_count_), cores_count_(o.cores_count_) {
        o.nodes_ = nullptr;
        o.node_core_ids_ = nullptr;
        o.nodes_count_ = 0;
        o.cores_count_ = 0;
    }

    numa_topology &operator=(numa_topology &&other) noexcept {
        if (this != &other) {
            reset(); // ? Reset the current state
            allocator_ = std::move(other.allocator_);
            nodes_ = std::exchange(other.nodes_, nullptr);
            node_core_ids_ = std::exchange(other.node_core_ids_, nullptr);
            nodes_count_ = std::exchange(other.nodes_count_, 0);
            cores_count_ = std::exchange(other.cores_count_, 0);
        }
        return *this;
    }

    numa_topology(numa_topology const &) = delete;
    numa_topology &operator=(numa_topology const &) = delete;

    ~numa_topology() noexcept { reset(); }

    void reset() noexcept {
        cores_allocator_t cores_alloc {allocator_};
        nodes_allocator_t nodes_alloc {allocator_};

        if (node_core_ids_) cores_alloc.deallocate(node_core_ids_, cores_count_);
        if (nodes_) nodes_alloc.deallocate(nodes_, nodes_count_);

        nodes_ = nullptr;
        node_core_ids_ = nullptr;
        nodes_count_ = cores_count_ = 0;
    }

    std::size_t nodes_count() const noexcept { return nodes_count_; }
    std::size_t threads_count() const noexcept { return cores_count_; }
    numa_node_t const &node(std::size_t const node_index) const noexcept {
        assert(node_index < nodes_count_ && "Node ID is out of bounds");
        return nodes_[node_index];
    }

    /**
     *  @brief Harvests CPU-memory topology.
     *  @retval false if the kernel lacks NUMA support or the harvest failed.
     *  @retval true if the harvest was successful and the topology is ready to use.
     */
    bool try_harvest() noexcept {
#if FU_ENABLE_NUMA
        struct bitmask *numa_mask = nullptr;
        numa_node_t *nodes_ptr = nullptr;
        numa_core_id_t *core_ids_ptr = nullptr;
        numa_node_id_t max_numa_node_id = -1;

        // Allocators must be visible to the cleanup path
        nodes_allocator_t nodes_alloc {allocator_};
        cores_allocator_t cores_alloc {allocator_};

        // These counters are reused in the failure handler
        std::size_t fetched_nodes = 0, fetched_cores = 0;

        if (::numa_available() < 0) goto failed_harvest; // ! Linux kernel lacks NUMA support
        ::numa_node_to_cpu_update();                     // ? Reset the outdated stale state

        numa_mask = ::numa_allocate_cpumask();
        if (!numa_mask) goto failed_harvest; // ! Allocation failed

        // First pass - measure
        max_numa_node_id = ::numa_max_node();
        for (numa_node_id_t node_id = 0; node_id <= max_numa_node_id; ++node_id) {
            long long dummy;
            if (::numa_node_size64(node_id, &dummy) < 0) continue; // ! Offline node
            ::numa_bitmask_clearall(numa_mask);
            if (::numa_node_to_cpus(node_id, numa_mask) < 0) continue; // ! Invalid CPU map
            std::size_t const node_cores = static_cast<std::size_t>(::numa_bitmask_weight(numa_mask));
            assert(node_cores > 0 && "Node must have at least one core");
            fetched_nodes += 1;
            fetched_cores += node_cores;
        }
        if (fetched_nodes == 0) goto failed_harvest; // ! Zero nodes is not a valid state

        // Second pass - allocate
        nodes_ptr = nodes_alloc.allocate(fetched_nodes);
        core_ids_ptr = cores_alloc.allocate(fetched_cores);
        if (!nodes_ptr || !core_ids_ptr) goto failed_harvest; // ! Allocation failed

        // Populate
        for (numa_node_id_t node_id = 0, core_index = 0, node_index = 0; node_id <= max_numa_node_id; ++node_id) {
            long long memory_size;
            if (::numa_node_size64(node_id, &memory_size) < 0) continue;
            ::numa_bitmask_clearall(numa_mask);
            if (::numa_node_to_cpus(node_id, numa_mask) < 0) continue;

            numa_node_t &node = nodes_ptr[node_index];
            node.node_id = node_id;
            node.memory_size = static_cast<std::size_t>(memory_size);
            node.first_core_id = core_ids_ptr + core_index;
            node.core_count = static_cast<std::size_t>(::numa_bitmask_weight(numa_mask));
            assert(node.core_count > 0 && "Node is known to have at least one core");
            node.socket_id = get_socket_id_for_core(node.first_core_id[0]);

            // Most likely, this will fill `core_ids_ptr` with `std::iota`-like values
            for (std::size_t bit_offset = 0; bit_offset < numa_mask->size; ++bit_offset)
                if (::numa_bitmask_isbitset(numa_mask, static_cast<unsigned int>(bit_offset)))
                    core_ids_ptr[core_index++] = static_cast<numa_core_id_t>(bit_offset);

            // Fetch Huge Page sizes for this NUMA node
            node.page_sizes.try_harvest(node_id); // ! We are not raising the failure - Huge Pages are optional
            node_index++;
        }

        // Commit
        nodes_ = nodes_ptr;
        node_core_ids_ = core_ids_ptr;
        nodes_count_ = fetched_nodes;
        cores_count_ = fetched_cores;
        ::numa_free_cpumask(numa_mask); // ? Clean up

        // Let's sort all the nodes by their socket ID, then by number of cores, then by first core ID
        bubble_sort(nodes_, nodes_count_, [](numa_node_t const &a, numa_node_t const &b) noexcept {
            if (a.socket_id != b.socket_id) return a.socket_id < b.socket_id;
            if (a.core_count != b.core_count) return a.core_count > b.core_count; // ? Sort by descending core count
            return a.first_core_id[0] < b.first_core_id[0];                       // ? Sort by first core ID
        });

        return true;

    failed_harvest:
        if (nodes_ptr) nodes_alloc.deallocate(nodes_ptr, fetched_nodes);
        if (core_ids_ptr) cores_alloc.deallocate(core_ids_ptr, fetched_cores);
        if (numa_mask) ::numa_free_cpumask(numa_mask);
#endif // FU_ENABLE_NUMA
        return false;
    }

    /**
     *  @brief Copy-assigns the topology from @p other.
     *
     *  Instead of a copy-constructor we expose an explicit operation that can
     *  FAIL - returning `false` if *any* intermediate allocation fails.
     *
     *  @param other Source topology.
     *  @retval true  Success, the current instance now owns a deep copy.
     *  @retval false Allocation failed, the current instance is unchanged.
     */
    bool try_assign(numa_topology const &other) noexcept {
        if (this == &other) return true; // ? Self-assignment is a no-op

        // Prepare scratch
        nodes_allocator_t nodes_alloc {allocator_};
        cores_allocator_t cores_alloc {allocator_};

        numa_node_t *scratch_nodes = nullptr;
        numa_core_id_t *scratch_core_ids = nullptr;
        if (other.nodes_count_) {
            scratch_nodes = nodes_alloc.allocate(other.nodes_count_);
            if (!scratch_nodes) return false; // ! OOM
        }
        if (other.cores_count_) {
            scratch_core_ids = cores_alloc.allocate(other.cores_count_);
            if (!scratch_core_ids) {
                if (scratch_nodes) nodes_alloc.deallocate(scratch_nodes, other.nodes_count_);
                return false; // ! OOM
            }
        }

        // Deep copy
        if (other.cores_count_ > 0)
            std::memcpy(scratch_core_ids, other.node_core_ids_, other.cores_count_ * sizeof(numa_core_id_t));
        for (std::size_t i = 0; i < other.nodes_count_; ++i) {
            scratch_nodes[i] = other.nodes_[i];
            // Re-base `first_core_id` so it points into our own core-id block
            std::ptrdiff_t const offset = other.nodes_[i].first_core_id - other.node_core_ids_;
            scratch_nodes[i].first_core_id = scratch_core_ids + offset;
        }

        reset(); // ? Free old buffers
        nodes_ = scratch_nodes;
        node_core_ids_ = scratch_core_ids;
        nodes_count_ = other.nodes_count_;
        cores_count_ = other.cores_count_;
        return true;
    }
};

using numa_topology_t = numa_topology<>;

static constexpr std::size_t page_size_4k = 4ull * 1024ull;                       // 4 KB
static constexpr std::size_t page_size_2m_k = 2ull * 1024ull * 1024ull;           // 2 MB
static constexpr std::size_t page_size_1g_k = 1ull * 1024ull * 1024ull * 1024ull; // 1 GB

/**
 *  @brief Tries binding the given address range to a specific NUMA @p `node_id`.
 *  @retval true if binding succeeded, false otherwise.
 */
FU_MAYBE_UNUSED_ static inline bool linux_numa_bind(void *ptr, std::size_t size_bytes,
                                                    numa_node_id_t node_id) noexcept {
#if FU_ENABLE_NUMA
    // Pin the memory - that may require an extra allocation for `node_mask` on some systems
    ::nodemask_t node_mask;
    ::bitmask node_mask_as_bitset;
    node_mask_as_bitset.size = sizeof(node_mask) * 8;
    node_mask_as_bitset.maskp = &node_mask.n[0];
    ::numa_bitmask_setbit(&node_mask_as_bitset, static_cast<unsigned int>(node_id));
    int mbind_flags;
#if defined(MPOL_F_STATIC_NODES)
    mbind_flags = MPOL_F_STATIC_NODES;
#else
    mbind_flags = 1 << 15;
#endif // MPOL_F_STATIC_NODES

    long binding_status = ::mbind(ptr, size_bytes, MPOL_BIND, &node_mask.n[0], sizeof(node_mask) * 8 - 1,
                                  static_cast<unsigned int>(mbind_flags));
    if (binding_status < 0) return false; // ! Binding failed
    return true;                          // ? Binding succeeded
#else
    fu_unused_(ptr);
    fu_unused_(size_bytes);
    fu_unused_(node_id);
    return false;
#endif // FU_ENABLE_NUMA
}

/**
 *  @brief Tries allocating uninitialized memory and binding it to a specific NUMA @p `node_id`.
 *  @retval nullptr if allocation failed or the page size is unsupported.
 *  @retval pointer to the allocated memory on success.
 */
FU_MAYBE_UNUSED_ static inline void *linux_numa_allocate(std::size_t size_bytes, std::size_t page_size_bytes,
                                                         numa_node_id_t node_id) noexcept {
    assert(node_id >= 0 && "NUMA node ID must be non-negative");

#if FU_ENABLE_NUMA

    // Fast path: regular pages – let `libnuma` handle any rounding internally.
    if (page_size_bytes == static_cast<std::size_t>(::numa_pagesize())) return ::numa_alloc_onnode(size_bytes, node_id);

    // Huge/explicit page sizes must be exact multiples
    assert(size_bytes % page_size_bytes == 0 && "Size must be a multiple of page size");

    // Make sure the page size makes sense for Linux
    int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS;
    if (page_size_bytes == page_size_4k) { mmap_flags |= MAP_HUGETLB; }
    else if (page_size_bytes == page_size_2m_k) { mmap_flags |= MAP_HUGETLB | static_cast<int>(MAP_HUGE_2MB); }
    else if (page_size_bytes == page_size_1g_k) { mmap_flags |= MAP_HUGETLB | static_cast<int>(MAP_HUGE_1GB); }
    else { return nullptr; } // ! Unsupported page size

    // Under the hood, `numa_alloc_onnode` uses `mmap` and `mbind` to allocate memory
    void *result_ptr = ::mmap(nullptr, size_bytes, PROT_READ | PROT_WRITE, mmap_flags, -1, 0);
    if (result_ptr == MAP_FAILED) return nullptr; // ! Allocation failed

    if (!linux_numa_bind(result_ptr, size_bytes, node_id)) {
        ::munmap(result_ptr, size_bytes); // ? Unbind failed, clean up
        return nullptr;                   // ! Binding failed
    }
    return result_ptr;
#else
    fu_unused_(size_bytes);
    fu_unused_(page_size_bytes);
    fu_unused_(node_id);
    return nullptr;
#endif // FU_ENABLE_NUMA
}

FU_MAYBE_UNUSED_ static inline void linux_numa_free(void *ptr, std::size_t size_bytes) noexcept {
    assert(ptr != nullptr && "Pointer must not be null");
    assert(size_bytes > 0 && "Size must be greater than zero");
#if FU_ENABLE_NUMA
    numa_free(ptr, size_bytes);
#else
    fu_unused_(ptr);
    fu_unused_(size_bytes);
#endif
}

/**
 *  @brief STL-compatible allocator pinned to a NUMA node, prioritizing Huge Pages.
 *
 *  A light-weight, but high-latency BLOB allocator, tied to a specific NUMA node ID.
 *  Every allocation is a system call to `mmap` and subsequent `mbind`, aligned to at
 *  least 4 KB page size.
 *
 *  @section C++ 23 Functionality
 *
 *  Whenever possible, the newer `allocate_at_least` API should be used to reduce the
 *  number of reallocations.
 */
template <typename value_type_ = char>
struct linux_numa_allocator {
    using value_type = value_type_;
    using size_type = std::size_t;
    using propagate_on_container_move_assignment = std::true_type;

  private:
    numa_node_id_t node_id_ {-1};     // ? Unique NUMA node ID, in [0, numa_max_node())
    size_type default_page_size_ {0}; // ? RAM page size in bytes, typically 4 KB

  public:
    numa_node_id_t node_id() const noexcept { return node_id_; }
    size_type default_page_size() const noexcept { return default_page_size_; }

    constexpr linux_numa_allocator() noexcept = default;
    explicit constexpr linux_numa_allocator(numa_node_id_t id, size_type paging = get_ram_page_size()) noexcept
        : node_id_(id), default_page_size_(paging) {}

    template <typename other_type_>
    explicit constexpr linux_numa_allocator(linux_numa_allocator<other_type_> const &o) noexcept
        : node_id_(o.node_id()), default_page_size_(o.default_page_size()) {}

    /**
     *  @brief Allocates memory for at least `size` elements of `value_type`.
     *  @param[in] size The number of elements to allocate.
     *  @param[in] page_size_bytes The size of the memory page to allocate, must be a multiple of `sizeof(value_type)`.
     *  @return allocation_result with a pointer to the allocated memory and the number of elements allocated.
     *  @retval empty object if the allocation failed or the size is not a multiple of `sizeof(value_type)`.
     */
    allocation_result<value_type *, size_type> allocate_at_least(size_type size, size_type page_size_bytes) noexcept {
        size_type const size_bytes = size * sizeof(value_type);
        size_type const aligned_size_bytes = (size_bytes + page_size_bytes - 1) / page_size_bytes * page_size_bytes;

        // Check if the new size is actually perfectly divisible by the `sizeof(value_type)`
        if (aligned_size_bytes % sizeof(value_type)) return {}; // ! Not a size multiple
        auto result_ptr = allocate(aligned_size_bytes / sizeof(value_type), page_size_bytes);
        if (!result_ptr) return {}; // ! Allocation failed
        size_type const pages_count = (page_size_bytes == 0) ? 0 : (aligned_size_bytes / page_size_bytes);
        return {result_ptr, size, aligned_size_bytes, pages_count};
    }

    /**
     *  @brief Allocates a memory for `size` elements of `value_type`.
     *  @param[in] size The number of elements to allocate.
     *  @param[in] page_size_bytes The size of the memory page to allocate, must be a multiple of `sizeof(value_type)`.
     *  @return allocation_result with a pointer to the allocated memory and the number of elements allocated.
     *  @retval empty object if the allocation failed or the size is not a multiple of `sizeof(value_type)`.
     */
    value_type *allocate(size_type size, size_type page_size_bytes) noexcept {
        size_type const size_bytes = size * sizeof(value_type);
        void *result_ptr = linux_numa_allocate(size_bytes, page_size_bytes, node_id_);
        if (!result_ptr) return {}; // ! Allocation failed
        return static_cast<value_type *>(result_ptr);
    }

    /**
     *  @brief Allocates memory for at least `size` elements of `value_type`.
     *  @param[in] size The number of elements to allocate.
     *  @return allocation_result with a pointer to the allocated memory and the number of elements allocated.
     *  @retval empty object if the allocation failed or the size is not a multiple of `sizeof(value_type)`.
     */
    allocation_result<value_type *, size_type> allocate_at_least(size_type size) noexcept {
        // Go through all of the typical Linux page sizes,
        // finding the largest one that makes sense and doesn't fail.
        size_type const size_bytes = size * sizeof(value_type);

        // Try 1 GB Huge Pages, for buffers larger than 2 GB
        if (size_bytes >= (2u * page_size_1g_k))
            if (auto result = allocate_at_least(size, page_size_1g_k); result) return result;

        // Try 2 MB Huge Pages, for buffers larger than 4 MB
        if (size_bytes >= (2u * page_size_2m_k))
            if (auto result = allocate_at_least(size, page_size_2m_k); result) return result;

        return allocate_at_least(size, default_page_size_);
    }

    /**
     *  @brief Allocates memory for `size` elements of `value_type`.
     *  @param[in] size The number of elements to allocate.
     *  @return allocation_result with a pointer to the allocated memory and the number of elements allocated.
     *  @retval empty object if the allocation failed or the size is not a multiple of `sizeof(value_type)`.
     */
    value_type *allocate(size_type size) noexcept {
        // Go through all of the typical Linux page sizes,
        // finding the largest one that makes sense and doesn't fail.
        size_type const size_bytes = size * sizeof(value_type);

        // Try 1 GB Huge Pages, for buffers larger than 2 GB
        if (size_bytes >= (2u * page_size_1g_k))
            if (auto result = allocate(size, page_size_1g_k); result) return result;

        // Try 2 MB Huge Pages, for buffers larger than 4 MB
        if (size_bytes >= (2u * page_size_2m_k))
            if (auto result = allocate(size, page_size_2m_k); result) return result;

        return allocate(size, default_page_size_);
    }

    void deallocate(value_type *p, size_type n) noexcept { linux_numa_free(p, n * sizeof(value_type)); }

    template <typename other_type_>
    bool operator==(linux_numa_allocator<other_type_> const &o) const noexcept {
        return node_id_ == o.node_id_ && default_page_size_ == o.default_page_size_;
    }

    template <typename other_type_>
    bool operator!=(linux_numa_allocator<other_type_> const &o) const noexcept {
        return node_id_ != o.node_id_ || default_page_size_ != o.default_page_size_;
    }
};

using linux_numa_allocator_t = linux_numa_allocator<>;

#if FU_ENABLE_NUMA

/**
 *  @brief Used inside `linux_colocated_pool` to describe a pinned thread.
 *
 *  On Linux, we can advise the scheduler on the importance of certain execution threads.
 *  For that we need to know the thread IDs - `pid_t`, which is not the same as `pthread_t`,
 *  and not a process ID, but a thread ID... counter-intuitive, I know.
 *  @see https://man7.org/linux/man-pages/man2/gettid.2.html
 *
 *  That `pid_t` can only be retrieved from inside the thread via `gettid` system call,
 *  so we need some shared memory to make those IDs visible to other threads. Moreover,
 *  we need to safeguard the reads/writes with atomics to avoid race conditions.
 *  @see https://stackoverflow.com/a/558815
 */
struct alignas(default_alignment_k) numa_pthread_t {
    std::atomic<pthread_t> handle {};
    std::atomic<pid_t> id {};
    numa_core_id_t core_id {-1};
    qos_level_t qos_level {-1}; // TODO: Populate from VFS, if available
};

#pragma region - Linux Colocated Pool

/**
 *  @brief A Linux-only thread-pool pinned to one NUMA node and same QoS level physical cores.
 *
 *  Differs from the `basic_pool` template in the following ways:
 *  - constructor API: receives a name for the threads.
 *  - implementation & API of `try_spawn`: uses POSIX APIs to allocate, name, & pin threads.
 *  - worker loop: using Linux-specific napping mechanism to reduce power consumption.
 *  - implementation `sleep`: informing the scheduler to move the thread to IDLE state.
 *  - availability of `terminate`: which can be called mid-air to shred the pool.
 *
 *  When not to use this thread-pool?
 *  - don't use outside of Linux or in UMA (Uniform Memory Access) systems.
 *  - don't use if you just need to pin everything to a single NUMA node,
 *    for that: `numactl --cpunodebind=2 --membind=2 your_program`
 *
 *  How to best leverage this thread-pool?
 *  - use in conjunction with @b `linux_numa_allocator` to pin memory to the same NUMA node.
 *  - make sure the Linux kernel is built with @b `CONFIG_SCHED_IDLE` support.
 *  - avoid recreating the @b `numa_topology`, as it's expensive to harvest.
 */
template <typename micro_yield_type_ = standard_yield_t, std::size_t alignment_ = default_alignment_k>
struct linux_colocated_pool {

  public:
    using allocator_t = linux_numa_allocator_t;
    using micro_yield_t = micro_yield_type_;
    static_assert(std::is_nothrow_invocable_r<void, micro_yield_t>::value,
                  "Yield must be callable w/out arguments & return void");
    static constexpr std::size_t alignment_k = alignment_;
    static_assert(alignment_k > 0 && (alignment_k & (alignment_k - 1)) == 0, "Alignment must be a power of 2");

    using index_t = std::size_t;
    static_assert(std::is_unsigned<index_t>::value, "Index type must be an unsigned integer");
    using epoch_index_t = index_t;  // ? A.k.a. number of previous API calls in [0, UINT_MAX)
    using thread_index_t = index_t; // ? A.k.a. "core index" or "thread ID" in [0, threads_count)
    using colocated_thread_t = colocated_thread<thread_index_t>;
    using prong_t = colocated_prong<index_t>;

    using punned_fork_context_t = void *;                                     // ? Pointer to the on-stack lambda
    using trampoline_t = void (*)(punned_fork_context_t, colocated_thread_t); // ? Wraps lambda's `operator()`

  private:
    using allocator_traits_t = std::allocator_traits<allocator_t>;
    using numa_pthread_allocator_t = typename allocator_traits_t::template rebind_alloc<numa_pthread_t>;

    // Thread-pool-specific variables:
    allocator_t allocator_ {};

    /**
     *  Differs from STL `workers_` in base in type and size, as it may contain the `pthread_self`
     *  at the first position. If the @b `numa_pin_to_core_k` granularity is used, the `numa_pthread_t::core_id`
     *  will be set to the individual core IDs.
     */
    unique_padded_buffer<numa_pthread_t, numa_pthread_allocator_t> pthreads_ {};

    thread_index_t first_thread_ {0};                       // ? The index of the first thread to start from
    caller_exclusivity_t exclusivity_ {caller_inclusive_k}; // ? Whether the caller thread is included in the count
    std::size_t sleep_length_micros_ {0}; // ? How long to sleep in microseconds when waiting for tasks

    using char16_name_t = char[16];    // ? Fixed-size thread name buffer, for POSIX thread naming
    char16_name_t name_ {};            // ? Thread name buffer, for POSIX thread naming
    numa_node_id_t numa_node_id_ {-1}; // ? Unique NUMA node ID, in [0, numa_max_node())
    index_t colocation_index_ {0};     // ? Unique {NUMA node + QoS level} colocation ID, defined externally
    numa_pin_granularity_t pin_granularity_ {numa_pin_to_core_k};

    alignas(alignment_k) std::atomic<mood_t> mood_ {mood_t::grind_k};

    // Task-specific variables:
    punned_fork_context_t fork_state_ {nullptr}; // ? Pointer to the users lambda
    trampoline_t fork_trampoline_ {nullptr};     // ? Calls the lambda
    alignas(alignment_k) std::atomic<thread_index_t> threads_to_sync_ {0};
    alignas(alignment_k) std::atomic<epoch_index_t> epoch_ {0};

    alignas(alignment_k) std::atomic<index_t> dynamic_progress_ {0}; // ? Only used in `for_n_dynamic`

  public:
    linux_colocated_pool(linux_colocated_pool &&) = delete;
    linux_colocated_pool(linux_colocated_pool const &) = delete;
    linux_colocated_pool &operator=(linux_colocated_pool &&) = delete;
    linux_colocated_pool &operator=(linux_colocated_pool const &) = delete;

    explicit linux_colocated_pool(char const *name = "fork_union") noexcept {
        // Accept NULL or empty names by falling back to a sensible default
        char const *effective_name = (name && name[0] != '\0') ? name : "fork_union";
        std::strncpy(name_, effective_name, sizeof(name_) - 1);
        name_[sizeof(name_) - 1] = '\0';
    }

    ~linux_colocated_pool() noexcept { terminate(); }

    /**
     *  @brief Estimates the amount of memory managed by this pool handle and internal structures.
     *  @note This API is @b not synchronized.
     */
    std::size_t memory_usage() const noexcept {
        return sizeof(linux_colocated_pool) + threads_count() * sizeof(numa_pthread_t);
    }

    /** @brief Checks if the thread-pool's core synchronization points are lock-free. */
    bool is_lock_free() const noexcept { return mood_.is_lock_free() && threads_to_sync_.is_lock_free(); }

    /**
     *  @brief Returns the NUMA node ID this thread-pool is pinned to.
     *  @retval -1 if the thread-pool is not initialized or the NUMA node ID is unknown.
     *  @note This API is @b not synchronized.
     */
    numa_node_id_t numa_node_id() const noexcept { return numa_node_id_; }

    /**
     *  @brief Returns the colocation index of this thread-pool.
     *  @retval 0 if the thread-pool is not initialized or the colocation index is unknown.
     *  @note This API is @b not synchronized.
     */
    index_t colocation_index() const noexcept { return colocation_index_; }

    /**
     *  @brief Returns the first thread index in the thread-pool.
     *  @retval 0 in most cases, when the last argument to `try_spawn` is not specified.
     *  @note This API is @b not synchronized.
     */
    thread_index_t first_thread() const noexcept { return first_thread_; }

    /** @brief Exposes access to the internal atomic progress counter. Use with caution. */
    std::atomic<index_t> &unsafe_dynamic_progress_ref() noexcept { return dynamic_progress_; }

#pragma region Core API

    /**
     *  @brief Returns the number of threads in the thread-pool, including the main thread.
     *  @retval 0 if the thread-pool is not initialized, 1 if only the main thread is used.
     *  @note This API is @b not synchronized.
     */
    thread_index_t threads_count() const noexcept { return pthreads_.size(); }

    /**
     *  @brief Reports if the current calling thread will be used for broadcasts.
     *  @note This API is @b not synchronized.
     */
    caller_exclusivity_t caller_exclusivity() const noexcept { return exclusivity_; }

    /**
     *  @brief Creates a thread-pool addressing all cores on the given NUMA @p node.
     *  @param[in] node Describes the NUMA node to use, with its ID, memory size, and core IDs.
     *  @param[in] exclusivity Should we count the calling thread as one of the threads?
     *  @retval false if the number of threads is zero or if spawning has failed.
     *  @retval true if the thread-pool was created successfully, started, and is ready to use.
     *  @note This is the de-facto @b constructor - you only call it again after `terminate`.
     *  @sa Other overloads of `try_spawn` that allow to specify the number of threads.
     */
    bool try_spawn(numa_node_t const &node, caller_exclusivity_t const exclusivity = caller_inclusive_k) noexcept {
        return try_spawn(node, node.core_count, exclusivity);
    }

    /**
     *  @brief Creates a thread-pool with the given number of @p threads on the given NUMA @p node.
     *  @param[in] node Describes the NUMA node to use, with its ID, memory size, and core IDs.
     *  @param[in] threads The number of threads to be used.
     *  @param[in] exclusivity Should we count the calling thread as one of the threads?
     *  @param[in] pin_granularity How to pin the threads to the NUMA node?
     *  @param[in] first_thread The index of the first thread to start from, defaults to 0.
     *  @param[in] colocation_index A unique index for the {NUMA node + QoS level} colocation.
     *  @retval false if the number of threads is zero or if spawning has failed.
     *  @retval true if the thread-pool was created successfully, started, and is ready to use.
     *  @note This is the de-facto @b constructor - you only call it again after `terminate`.
     *
     *  @section Over- and Under-subscribing Cores and Pinning
     *
     *  We may accept @p threads different from the @p node.core_count, which allows us to:
     *  - over-subscribe the cores, i.e. use more threads than cores available on the NUMA node.
     *  - under-subscribe the cores, i.e. use fewer threads than cores available on the NUMA node.
     *
     *  If you only have one thread-pool active at any part of your application, that's meaningless.
     *  You'd be better off using exactly the number of cores available on the NUMA node and pinning
     *  them to individual cores with @b `numa_pin_to_core_k` granularity.
     */
    bool try_spawn(numa_node_t const &node, thread_index_t const threads,
                   caller_exclusivity_t const exclusivity = caller_inclusive_k,
                   numa_pin_granularity_t const pin_granularity = numa_pin_to_core_k,
                   thread_index_t const first_thread = 0, index_t const colocation_index = 0) noexcept {

        if (threads == 0) return false;          // ! Can't have zero threads working on something
        if (pthreads_.size() != 0) return false; // ! Already initialized

        // Allocate the thread pool of `numa_pthread_t` objects
        allocator_ = linux_numa_allocator_t {node.node_id};
        numa_pthread_allocator_t pthread_allocator {allocator_};
        unique_padded_buffer<numa_pthread_t, numa_pthread_allocator_t> pthreads {pthread_allocator};
        if (!pthreads.try_resize(threads)) return false; // ! Allocation failed

        // Allocate the `cpu_set_t` structure, assuming we may be on a machine
        // with a ridiculously large number of cores.
        int const max_possible_cores = ::numa_num_possible_cpus();
        cpu_set_t *cpu_set_ptr = CPU_ALLOC(max_possible_cores);

        // Before we start the threads, make sure we set some of the shared
        // state variables that will be used in the `_posix_worker_loop` function.
        pthreads_ = std::move(pthreads);
        first_thread_ = first_thread;
        colocation_index_ = colocation_index;
        exclusivity_ = exclusivity;
        numa_node_id_ = node.node_id;
        pin_granularity_ = pin_granularity;
        auto reset_on_failure = [&]() noexcept {
            pthreads_ = {};
            numa_node_id_ = -1;
            pin_granularity_ = numa_pin_to_core_k;
        };

        // Include the main thread into the list of handles
        bool const use_caller_thread = exclusivity == caller_inclusive_k;
        if (use_caller_thread) {
            pthreads_[0].handle.store(::pthread_self(), std::memory_order_release);
            pthreads_[0].id.store(::gettid(), std::memory_order_release);
        }

        // The startup sequence for the POSIX threads differs from the `basic_pool`,
        // where at start up there is a race condition to read the `pthreads_`.
        // So we mark the threads as "chilling" until the
        mood_.store(mood_t::chill_k, std::memory_order_release);

        // Initializing the thread pool can fail for all kinds of reasons, like:
        // - `EAGAIN` if we reach the `RLIMIT_NPROC` soft resource limit.
        // - `EINVAL` if an invalid attribute was specified.
        // - `EPERM` if we don't have the right permissions.
        for (thread_index_t i = use_caller_thread; i < threads; ++i) {

            pthread_t new_pthread_handle;
            int creation_result = ::pthread_create(&new_pthread_handle, nullptr, &_posix_worker_loop, this);
            pthreads_[i].handle.store(new_pthread_handle, std::memory_order_relaxed);
            pthreads_[i].id.store(-1, std::memory_order_relaxed);
            pthreads_[i].core_id = -1; // ? Not pinned yet

            if (creation_result != 0) {
                mood_.store(mood_t::die_k, std::memory_order_release);
                for (thread_index_t j = use_caller_thread; j < i; ++j) {
                    pthread_t cancel_pthread_handle = pthreads_[j].handle.load(std::memory_order_relaxed);
                    FU_MAYBE_UNUSED_ int cancel_result = ::pthread_cancel(cancel_pthread_handle);
                    assert(cancel_result == 0 && "Failed to cancel a thread");
                }
                reset_on_failure();
                CPU_FREE(cpu_set_ptr);
                return false; // ! Thread creation failed
            }
        }

        // Name all of the threads
        char16_name_t name;
        for (thread_index_t i = 0; i < pthreads_.size(); ++i) {
            fill_thread_name(                                    //
                name, name_,                                     //
                static_cast<std::size_t>(node.first_core_id[i]), //
                static_cast<std::size_t>(max_possible_cores));
            pthread_t naming_pthread_handle = pthreads_[i].handle.load(std::memory_order_relaxed);
            FU_MAYBE_UNUSED_ int naming_result = ::pthread_setname_np(naming_pthread_handle, name);
            assert(naming_result == 0 && "Failed to name a thread");
        }

        // Pin all of the threads
        std::size_t const cpu_set_size = CPU_ALLOC_SIZE(max_possible_cores);
        if (pin_granularity == numa_pin_to_core_k) {
            // Configure a mask for each thread, pinning it to a specific core
            for (thread_index_t i = 0; i < pthreads_.size(); ++i) {
                // Assign to a core in a round-robin fashion
                numa_core_id_t cpu = node.first_core_id[i % node.core_count];
                assert(cpu >= 0 && "Invalid CPU core ID");
                CPU_ZERO_S(cpu_set_size, cpu_set_ptr);
                CPU_SET_S(cpu, cpu_set_size, cpu_set_ptr);

                // Assign the mask to the thread
                pthread_t pin_pthread_handle = pthreads_[i].handle.load(std::memory_order_relaxed);
                FU_MAYBE_UNUSED_ int pin_result =
                    ::pthread_setaffinity_np(pin_pthread_handle, cpu_set_size, cpu_set_ptr);
                assert(pin_result == 0 && "Failed to pin a thread to a NUMA node");
                pthreads_[i].core_id = cpu;
            }
        }
        else {
            // Configure one mask that will be shared by all threads
            CPU_ZERO_S(cpu_set_size, cpu_set_ptr);
            for (std::size_t i = 0; i < node.core_count; ++i) {
                numa_core_id_t cpu = node.first_core_id[i];
                assert(cpu >= 0 && "Invalid CPU core ID");
                CPU_SET_S(cpu, cpu_set_size, cpu_set_ptr);
            }
            assert(static_cast<std::size_t>(CPU_COUNT_S(cpu_set_size, cpu_set_ptr)) == node.core_count &&
                   "The CPU set must match the number of cores in the NUMA node");

            // Assign the same mask to all threads
            for (thread_index_t i = 0; i < pthreads_.size(); ++i) {
                pthread_t pin_pthread_handle = pthreads_[i].handle.load(std::memory_order_relaxed);
                FU_MAYBE_UNUSED_ int pin_result =
                    ::pthread_setaffinity_np(pin_pthread_handle, cpu_set_size, cpu_set_ptr);
                assert(pin_result == 0 && "Failed to pin a thread to a NUMA node");
            }
        }

        // If all went well, we can store the thread-pool and start using it
        CPU_FREE(cpu_set_ptr); // ? Clean up the CPU set
        mood_.store(mood_t::grind_k, std::memory_order_release);
        return true;
    }

    /**
     *  @brief Executes a @p fork function in parallel on all threads.
     *  @param[in] fork The callback object, receiving the thread index as an argument.
     *  @return A `broadcast_join` synchronization point that waits in the destructor.
     *  @note Even in the `caller_exclusive_k` mode, can be called from just one thread!
     *  @sa For advanced resource management, consider `unsafe_for_threads` and `unsafe_join`.
     */
    template <typename fork_type_>
    FU_REQUIRES_((can_be_for_task_callback<fork_type_, index_t>()))
    broadcast_join<linux_colocated_pool, fork_type_> for_threads(fork_type_ &&fork) noexcept {
        return {*this, std::forward<fork_type_>(fork)};
    }

    /**
     *  @brief Executes a @p fork function in parallel on all threads, not waiting for the result.
     *  @param[in] fork The callback @b reference, receiving the thread index as an argument.
     *  @sa Use in conjunction with `unsafe_join`.
     */
    template <typename fork_type_>
    FU_REQUIRES_((can_be_for_thread_callback<fork_type_, index_t>()))
    void unsafe_for_threads(fork_type_ &fork) noexcept {

        thread_index_t const threads = threads_count();
        assert(threads != 0 && "Thread pool not initialized");
        caller_exclusivity_t const exclusivity = caller_exclusivity();
        bool const use_caller_thread = exclusivity == caller_inclusive_k;

        // Optional check: even in exclusive mode, only one thread can call this function.
        assert((use_caller_thread || threads_to_sync_.load(std::memory_order_acquire) == 0) &&
               "The broadcast function can't be called concurrently or recursively");

        // Configure "fork" details
        fork_state_ = std::addressof(fork);
        fork_trampoline_ = &_call_as_lambda<fork_type_>;
        threads_to_sync_.store(threads - use_caller_thread, std::memory_order_relaxed);

        // We are most likely already "grinding", but in the unlikely case we are not,
        // let's wake up from the "chilling" state with relaxed semantics. Assuming the sleeping
        // logic for the workers also checks the epoch counter, no synchronization is needed and
        // no immediate wake-up is required.
        mood_t may_be_chilling = mood_t::chill_k;
        bool const was_chilling = mood_.compare_exchange_weak( //
            may_be_chilling, mood_t::grind_k,                  //
            std::memory_order_relaxed, std::memory_order_relaxed);
        epoch_.fetch_add(1, std::memory_order_release); // ? Wake up sleepers

        // If the workers were indeed "chilling", we can inform the scheduler to wake them up.
        if (was_chilling) {
            for (std::size_t i = use_caller_thread; i < pthreads_.size(); ++i) {
                pid_t const pthread_id = pthreads_[i].id.load(std::memory_order_acquire);
                if (pthread_id < 0) continue; // ? Not set yet
                sched_param param {};
                ::sched_setscheduler(pthread_id, SCHED_FIFO | SCHED_RR, &param);
            }
        }
    }

    /** @brief Blocks the calling thread until the currently broadcasted task finishes. */
    void unsafe_join() noexcept {
        caller_exclusivity_t const exclusivity = caller_exclusivity();
        bool const use_caller_thread = exclusivity == caller_inclusive_k;

        // Execute on the current "main" thread
        if (use_caller_thread)
            fork_trampoline_(fork_state_, colocated_thread_t {static_cast<thread_index_t>(0), colocation_index_});

        // Actually wait for everyone to finish
        micro_yield_t micro_yield;
        while (threads_to_sync_.load(std::memory_order_acquire)) micro_yield();
    }

#pragma endregion Core API

#pragma region Control Flow

    /**
     *  @brief Stops all threads and deallocates the thread-pool after the last call finishes.
     *  @note Can be called from @b any thread at any time.
     *  @note Must `try_spawn` again to re-use the pool.
     *
     *  When and how @b NOT to use this function:
     *  - as a synchronization point between concurrent tasks.
     *
     *  When and how to use this function:
     *  - as a de-facto @b destructor, to stop all threads and deallocate the pool.
     *  - when you want to @b restart with a different number of threads.
     */
    void terminate() noexcept {
        assert(threads_to_sync_.load(std::memory_order_seq_cst) == 0); // ! No tasks must be running
        if (pthreads_.size() == 0) return;                             // ? Uninitialized

        numa_pthread_allocator_t pthread_allocator {allocator_};

        // Stop all threads and wait for them to finish
        mood_.store(mood_t::die_k, std::memory_order_release);

        caller_exclusivity_t const exclusivity = caller_exclusivity();
        bool const use_caller_thread = exclusivity == caller_inclusive_k;
        thread_index_t const threads = pthreads_.size();
        for (thread_index_t i = use_caller_thread; i != threads; ++i) {
            void *returned_value = nullptr;
            pthread_t const join_pthread_handle = pthreads_[i].handle.load(std::memory_order_relaxed);
            FU_MAYBE_UNUSED_ int const join_result = ::pthread_join(join_pthread_handle, &returned_value);
            assert(join_result == 0 && "Thread join failed");
        }

        // Deallocate the handles and IDs
        pthreads_ = {};

        // Unpin the caller thread if it was part of this pool and was pinned to the NUMA node.
        if (use_caller_thread) _reset_affinity();
        _reset_fork();

        mood_.store(mood_t::grind_k, std::memory_order_relaxed);
        epoch_.store(0, std::memory_order_relaxed);
    }

    /**
     *  @brief Transitions "workers" to a sleeping state, waiting for a wake-up call.
     *  @param[in] wake_up_periodicity_micros How often to check for new work in microseconds.
     *  @note Can only be called @b between the tasks for a single thread. No synchronization is performed.
     *
     *  This function may be used in some batch-processing operations when we clearly understand
     *  that the next task won't be arriving for a while and power can be saved without major
     *  latency penalties.
     *
     *  It may also be used in a high-level Python or JavaScript library offloading some parallel
     *  operations to an underlying C++ engine, where latency is irrelevant.
     */
    void sleep(std::size_t wake_up_periodicity_micros) noexcept {
        assert(wake_up_periodicity_micros > 0 && "Sleep length must be positive");
        sleep_length_micros_ = wake_up_periodicity_micros;
        mood_.store(mood_t::chill_k, std::memory_order_release);

        // On Linux we can update the thread's scheduling class to IDLE,
        // which will reduce the power consumption:
        caller_exclusivity_t const exclusivity = caller_exclusivity();
        bool const use_caller_thread = exclusivity == caller_inclusive_k;
        for (std::size_t i = use_caller_thread; i < pthreads_.size(); ++i) {
            pid_t const pthread_id = pthreads_[i].id.load(std::memory_order_acquire);
            if (pthread_id < 0) continue; // ? Not set yet
            sched_param param {};
            ::sched_setscheduler(pthread_id, SCHED_IDLE, &param);
        }
    }

    /** @brief Helper function to create a spin mutex with same yield characteristics. */
    static spin_mutex<micro_yield_t, alignment_k> make_mutex() noexcept { return {}; }

#pragma endregion Control Flow

#pragma region Indexed Task Scheduling

    /**
     *  @brief Distributes @p `n` similar duration calls between threads in slices, as opposed to individual indices.
     *  @param[in] n The total length of the range to split between threads.
     *  @param[in] fork The callback object, receiving the first @b `prong_t` and the slice length.
     */
    template <typename fork_type_ = dummy_lambda_t>
    FU_REQUIRES_((can_be_for_slice_callback<fork_type_, index_t>()))
    broadcast_join<linux_colocated_pool, invoke_for_slices<fork_type_, index_t>> //
        for_slices(index_t const n, fork_type_ &&fork) noexcept {

        return {*this, {n, threads_count(), std::forward<fork_type_>(fork)}};
    }

    /**
     *  @brief Same as `for_slices`, but doesn't wait for the result or guarantee fork lifetime.
     *  @param[in] n The total length of the range to split between threads.
     *  @param[in] fork The callback @b reference, receiving the first @b `prong_t` and the slice length.
     */
    template <typename fork_type_ = dummy_lambda_t>
    FU_REQUIRES_((can_be_for_slice_callback<fork_type_, index_t>()))
    void unsafe_for_slices(index_t const n, fork_type_ &fork) noexcept {

        invoke_for_slices<fork_type_ const &, index_t> invoker {n, threads_count(), fork};
        unsafe_for_threads(invoker);
    }

    /**
     *  @brief Distributes @p `n` similar duration calls between threads.
     *  @param[in] n The number of times to call the @p fork.
     *  @param[in] fork The callback object, receiving @b `prong_t` or a call index as an argument.
     *
     *  Is designed for a "balanced" workload, where all threads have roughly the same amount of work.
     *  @sa `for_n_dynamic` for a more dynamic workload.
     *  The @p fork is called @p `n` times, and each thread receives a slice of consecutive tasks.
     *  @sa `for_slices` if you prefer to receive workload slices over individual indices.
     */
    template <typename fork_type_ = dummy_lambda_t>
    FU_REQUIRES_((can_be_for_task_callback<fork_type_, index_t>()))
    broadcast_join<linux_colocated_pool, invoke_for_n<fork_type_, index_t>> //
        for_n(index_t const n, fork_type_ &&fork) noexcept {

        return {*this, {n, threads_count(), std::forward<fork_type_>(fork)}};
    }

    /**
     *  @brief Executes uneven tasks on all threads, greedying for work.
     *  @param[in] n The number of times to call the @p fork.
     *  @param[in] fork The callback object, receiving the `prong_t` or the task index as an argument.
     *  @sa `for_n` for a more "balanced" evenly-splittable workload.
     */
    template <typename fork_type_ = dummy_lambda_t>
    FU_REQUIRES_((can_be_for_task_callback<fork_type_, index_t>()))
    broadcast_join<linux_colocated_pool, invoke_for_n_dynamic<fork_type_, index_t>> //
        for_n_dynamic(index_t const n, fork_type_ &&fork) noexcept {

        return {*this, {n, threads_count(), dynamic_progress_, std::forward<fork_type_>(fork)}};
    }

#pragma endregion Indexed Task Scheduling

#pragma region Colocations Compatibility

    /**
     *  @brief Number of individual sub-pool with the same NUMA-locality and QoS.
     *  @retval 1 constant for compatibility.
     */
    constexpr index_t colocations_count() const noexcept { return 1; }

    /**
     *  @brief Returns the number of threads in one NUMA-specific local @b colocation.
     *  @retval Same value as `threads_count()`, as we only support one colocation.
     */
    thread_index_t threads_count(FU_MAYBE_UNUSED_ index_t colocation_index) const noexcept { return threads_count(); }

    /**
     *  @brief Converts a @p `global_thread_index` to a local thread index within a @b colocation.
     *  @retval Same value as @p `global_thread_index`, as we only support one colocation.
     */
    constexpr thread_index_t thread_local_index(thread_index_t global_thread_index,
                                                FU_MAYBE_UNUSED_ index_t colocation_index = 0) const noexcept {
        return global_thread_index;
    }

#pragma endregion Colocations Compatibility

  private:
    void _reset_fork() noexcept {
        fork_state_ = nullptr;
        fork_trampoline_ = nullptr;
    }

    void _reset_affinity() noexcept {
        int const max_possible_cores = ::numa_num_possible_cpus();
        if (max_possible_cores <= 0) return; // ? No cores available, nothing to reset
        cpu_set_t *cpu_set_ptr = CPU_ALLOC(static_cast<unsigned long>(max_possible_cores));
        if (!cpu_set_ptr) return;
        std::size_t const cpu_set_size = CPU_ALLOC_SIZE(static_cast<unsigned long>(max_possible_cores));
        CPU_ZERO_S(cpu_set_size, cpu_set_ptr);
        for (int cpu = 0; cpu < max_possible_cores; ++cpu) CPU_SET_S(cpu, cpu_set_size, cpu_set_ptr);
        FU_MAYBE_UNUSED_ int pin_result = ::pthread_setaffinity_np(::pthread_self(), cpu_set_size, cpu_set_ptr);
        assert(pin_result == 0 && "Failed to reset the caller thread's affinity");
        CPU_FREE(cpu_set_ptr);
        FU_MAYBE_UNUSED_ int spread_result = ::numa_run_on_node(-1); // !? Shouldn't it be `numa_all_nodes`
        assert(spread_result == 0 && "Failed to reset the caller thread's NUMA node affinity");
    }

    /**
     *  @brief A trampoline function that is used to call the user-defined lambda.
     *  @param[in] punned_lambda_pointer The pointer to the user-defined lambda.
     *  @param[in] prong The index of the thread & task index packed together.
     */
    template <typename fork_type_>
    static void _call_as_lambda(punned_fork_context_t punned_lambda_pointer,
                                colocated_thread_t colocated_thread) noexcept {
        fork_type_ &lambda_object = *static_cast<fork_type_ *>(punned_lambda_pointer);
        lambda_object(colocated_thread);
    }

    static void *_posix_worker_loop(void *arg) noexcept {
        linux_colocated_pool *pool = static_cast<linux_colocated_pool *>(arg);

        // Following section untile the main `while` loop may introduce race conditions,
        // so spin-loop for a bit until the pool is ready.
        mood_t mood;
        micro_yield_t micro_yield;
        while ((mood = pool->mood_.load(std::memory_order_acquire)) == mood_t::chill_k) micro_yield();

        // If we are ready to start grinding, export this threads metadata to make it externally
        // observable and controllable.
        thread_index_t local_thread_index = 0;
        if (mood == mood_t::grind_k) {
            // We locate the thread index by enumerating the `pthreads_` array
            auto &numa_pthreads = pool->pthreads_;
            thread_index_t const numa_pthreads_count = pool->pthreads_.size();
            pthread_t const thread_handle = ::pthread_self();
            for (local_thread_index = 0; local_thread_index < numa_pthreads_count; ++local_thread_index)
                if (::pthread_equal(numa_pthreads[local_thread_index].handle.load(std::memory_order_relaxed),
                                    thread_handle))
                    break;
            assert(local_thread_index < numa_pthreads_count && "Thread index must be in [0, threads_count)");

            // Assign the pthread ID to the shared memory
            pid_t const pthread_id = ::gettid();
            numa_pthreads[local_thread_index].id.store(pthread_id, std::memory_order_release);

            // Ensure this function isn't used by the main caller
            caller_exclusivity_t const exclusivity = pool->caller_exclusivity();
            bool const use_caller_thread = exclusivity == caller_inclusive_k;
            if (use_caller_thread)
                assert(local_thread_index != 0 && "The zero index is for the main thread, not worker!");
        }
        thread_index_t const global_thread_index = pool->first_thread_ + local_thread_index;

        // Run the infinite loop, using Linux-specific napping mechanism
        epoch_index_t last_epoch = 0;
        epoch_index_t new_epoch;
        while (true) {
            // Wait for either: a new ticket or a stop flag
            while ((new_epoch = pool->epoch_.load(std::memory_order_acquire)) == last_epoch &&
                   (mood = pool->mood_.load(std::memory_order_acquire)) == mood_t::grind_k)
                micro_yield();

            if (fu_unlikely_(mood == mood_t::die_k)) break;
            if (fu_unlikely_(mood == mood_t::chill_k) && (new_epoch == last_epoch)) {
                struct timespec ts {0, static_cast<long>(pool->sleep_length_micros_ * 1000)};
                ::clock_nanosleep(CLOCK_MONOTONIC, 0, &ts, nullptr);
                continue;
            }

            pool->fork_trampoline_(pool->fork_state_,
                                   colocated_thread_t {global_thread_index, pool->colocation_index_});
            last_epoch = new_epoch;

            // ! The decrement must come after the task is executed
            FU_MAYBE_UNUSED_ thread_index_t const before_decrement =
                pool->threads_to_sync_.fetch_sub(1, std::memory_order_release);
            assert(before_decrement > 0 && "We can't be here if there are no worker threads");
        }

        return nullptr;
    }

    static void fill_thread_name(                          //
        char16_name_t &output_name, char const *base_name, //
        std::size_t const index, std::size_t const max_possible_cores) noexcept {

        constexpr int max_visible_chars = sizeof(char16_name_t) - 1; // room left after the terminator
        int const digits = max_possible_cores < 10      ? 1
                           : max_possible_cores < 100   ? 2
                           : max_possible_cores < 1000  ? 3
                           : max_possible_cores < 10000 ? 4
                                                        : 0; // fall-through – let `snprintf` clip

        if (digits == 0) {
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
#endif
            //  "%s:%zu" - worst-case  (base up to 11 chars) + ":" + up-to-2-digit index
            std::snprintf(&output_name[0], sizeof(char16_name_t), "%s:%zu", base_name, index + 1);
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
        }
        else {
            int const base_len = max_visible_chars - digits - 1; // -1 for ':'
            // "%.*s" - truncates base_name to base_len
            // "%0*zu" - prints zero-padded index using exactly `digits` characters
            std::snprintf(&output_name[0], sizeof(char16_name_t), "%.*s:%0*zu", base_len, base_name, digits, index + 1);
        }
    }
};

#pragma endregion - Linux Colocated Pool

#pragma region - Linux Pool

/**
 *  @brief Wraps the metadata needed for `for_slices` APIs for `broadcast_join` compatibility.
 *  @note Similar to `invoke_for_slices`, but dynamically determines the threads' colocation.
 */
template <typename pool_type_, typename fork_type_, typename index_type_>
class invoke_distributed_for_slices {

    pool_type_ &pool_;
    indexed_split<index_type_> split_;
    fork_type_ fork_;

  public:
    invoke_distributed_for_slices(pool_type_ &pool, index_type_ n, index_type_ threads, fork_type_ &&fork) noexcept
        : pool_(pool), split_(n, threads), fork_(std::forward<fork_type_>(fork)) {}

    void operator()(index_type_ const thread) const noexcept {
        indexed_range<index_type_> const range = split_[thread];
        if (range.count == 0) return; // ? No work for this thread
        index_type_ const colocation = pool_.thread_colocation(thread);
        fork_(colocated_prong<index_type_> {range.first, thread, colocation}, range.count);
    }
};

/**
 *  @brief Wraps the metadata needed for `for_n` APIs for `broadcast_join` compatibility.
 *  @note Similar to `invoke_for_n`, but dynamically determines the threads' colocation.
 */
template <typename pool_type_, typename fork_type_, typename index_type_>
class invoke_distributed_for_n {
    pool_type_ &pool_;
    indexed_split<index_type_> split_;
    fork_type_ fork_;

  public:
    invoke_distributed_for_n(pool_type_ &pool, index_type_ n, index_type_ threads, fork_type_ &&fork) noexcept
        : pool_(pool), split_(n, threads), fork_(std::forward<fork_type_>(fork)) {}

    void operator()(index_type_ const thread) const noexcept {
        indexed_range<index_type_> const range = split_[thread];
        index_type_ const colocation = pool_.thread_colocation(thread);
        for (index_type_ i = 0; i < range.count; ++i)
            fork_(colocated_prong<index_type_> {static_cast<index_type_>(range.first + i), thread, colocation});
    }
};

/**
 *  @brief Wraps the metadata needed for `for_n_dynamic` APIs for `broadcast_join` compatibility.
 *  @note Similar to `invoke_for_n_dynamic`, but dynamically determines the threads' colocation.
 *
 *  @section Scheduling Logic
 *
 *  Assuming the latency of accessing an atomic variable on a remote NUMA node is high, this "invoker"
 *  performs work-stealing in a different way. Let's say we receive N tasks and we have T threads
 *  across C colocations. Each colocation takes (N/C) tasks and splits them between (T/C) threads.
 *  Once threads in one pool saturate their local (N/C) tasks, they start looping through other
 *  colocations and stealing tasks from them, until all tasks are completed.
 *
 *  The hardest decision there is to how to chose the next "non-native" colocation to steal from.
 *  Linear probing will produce unbalanced contention. A tree-like probing will produce a more balanced
 *  outcome.
 */
template <typename pool_type_, typename fork_type_, typename index_type_>
class invoke_distributed_for_n_dynamic {

    pool_type_ &pool_;
    index_type_ n_;
    fork_type_ fork_;

  public:
    invoke_distributed_for_n_dynamic(pool_type_ &pool, index_type_ n, fork_type_ &&fork) noexcept
        : pool_(pool), n_(n), fork_(std::forward<fork_type_>(fork)) {

        // Reset the local progress to zero in each colocation
        index_type_ const colocations_count = pool_.colocations_count();
        for (index_type_ i = 0; i < colocations_count; ++i)
            pool_.unsafe_dynamic_progress_ref(i).store(0, std::memory_order_release);
    }

    void operator()(index_type_ const thread) noexcept {
        index_type_ const colocations_count = pool_.colocations_count();
        assert(colocations_count > 0 && "There must be at least one colocation");

        // In each colocations part, take one static prong per thread, if present.
        indexed_split<index_type_> split_between_colocations(n_, colocations_count);
        index_type_ const native_colocation = pool_.thread_colocation(thread);
        {
            index_type_ const threads_local = pool_.threads_count(native_colocation);
            indexed_range<index_type_> const range_local = split_between_colocations[native_colocation];
            index_type_ const n_local = range_local.count;
            index_type_ const n_local_dynamic = n_local > threads_local ? n_local - threads_local : 0;

            // Run (up to) one static prong on the current thread
            index_type_ const thread_local_index = pool_.thread_local_index(thread, native_colocation);
            index_type_ const one_static_prong_index = static_cast<index_type_>(n_local_dynamic + thread_local_index);
            colocated_prong<index_type_> prong( //
                static_cast<index_type_>(range_local.first + one_static_prong_index), thread, native_colocation);
            if (one_static_prong_index < n_local) fork_(prong);
        }

        coprime_permutation_range<index_type_> probing_strategy(0, colocations_count, thread);
        auto probe_iterator = probing_strategy.begin();

        // Next we will probe every colocation:
        index_type_ colocations_remaining = colocations_count;
        index_type_ current_colocation = native_colocation;
        while (colocations_remaining) {
            index_type_ const threads_local = pool_.threads_count(current_colocation);
            std::atomic<index_type_> &local_progress = pool_.unsafe_dynamic_progress_ref(current_colocation);
            indexed_range<index_type_> const range_local = split_between_colocations[current_colocation];
            index_type_ const n_local = range_local.count;
            index_type_ const n_local_dynamic = n_local > threads_local ? n_local - threads_local : 0;

            // Same loop as in `invoke_for_n_dynamic::operator()`
            while (true) {
                index_type_ prong_local_offset = local_progress.fetch_add(1, std::memory_order_relaxed);
                bool const beyond_last_prong = prong_local_offset >= n_local_dynamic;
                if (beyond_last_prong) break;
                colocated_prong<index_type_> prong(range_local.first + prong_local_offset, thread, current_colocation);
                fork_(prong);
            }

            // Now pick some other colocation to probe.
            colocations_remaining--;
            if (colocations_remaining) {
                do { ++probe_iterator; } while (*probe_iterator == native_colocation); // At most 2 iterations
                current_colocation = *probe_iterator;
            }
        }
    }
};

/**
 *  @brief A Linux-only pool over all distributed "thread colocations", NUMA nodes, and QoS levels.
 *
 *  Differs from the `basic_pool` template in the following ways:
 *  - constructor API: receives the NUMA nodes topology, & a name for threads.
 *  - implementation of `try_spawn`: redirects to individual `linux_colocated_pool` instances.
 *
 *  Many of the parallel ops benefit from having some minimal amount of @b "scratch-space" that
 *  can be used as an output buffer for partial results, before they can be aggregated from the
 *  calling thread. Reductions are a great example, and allocating a new buffer for each thread
 *  on each call is quite wasteful, so we always keep some around.
 *
 *  This thread-pool doesn't (yet) provide "reductions" or other reach operations, but uses a
 *  small pool of NUMA-local memory to dampen the cost of `for_n_dynamic` scheduling.
 */
template <typename micro_yield_type_ = standard_yield_t, std::size_t alignment_ = default_alignment_k>
struct linux_distributed_pool {

    using linux_colocated_pool_t = linux_colocated_pool<micro_yield_type_, alignment_>;
    using numa_topology_t = numa_topology<>;

    using allocator_t = linux_numa_allocator_t;
    using micro_yield_t = typename linux_colocated_pool_t::micro_yield_t;
    using index_t = typename linux_colocated_pool_t::index_t;
    using epoch_index_t = typename linux_colocated_pool_t::epoch_index_t;
    using thread_index_t = typename linux_colocated_pool_t::thread_index_t;
    static constexpr std::size_t alignment_k = linux_colocated_pool_t::alignment_k;
    using prong_t = colocated_prong<index_t>;

  private:
    numa_topology_t topology_ {};
    char name_[16] {}; // ? Thread name buffer, for POSIX thread naming
    thread_index_t threads_count_ {0};
    caller_exclusivity_t exclusivity_ {caller_inclusive_k}; // ? Whether the caller thread is included in the count

    struct colocation_t {
        alignas(alignment_k) linux_colocated_pool_t pool {};
    };

    using unique_colocation_buffer_t = unique_padded_buffer<colocation_t, linux_numa_allocator_t>;
    using colocations_t = unique_padded_buffer<unique_colocation_buffer_t, linux_numa_allocator_t>;
    /**
     *  @brief A heap allocated array of individual thread pools.
     *
     *  Similar to a @b `std::vector<std::unique_ptr<linux_colocated_pool_t>>`, but with each colocation placed
     *  on its own NUMA node, and with a custom allocator. All the entries are sorted/grouped by the colocation
     *  index in ascending order, and the first one always contains the current thread.
     */
    colocations_t colocations_ {};

  public:
    linux_distributed_pool(linux_distributed_pool &&) = delete;
    linux_distributed_pool(linux_distributed_pool const &) = delete;
    linux_distributed_pool &operator=(linux_distributed_pool &&) = delete;
    linux_distributed_pool &operator=(linux_distributed_pool const &) = delete;

    linux_distributed_pool(numa_topology_t topo = {}) noexcept
        : linux_distributed_pool("fork_union", std::move(topo)) {}

    explicit linux_distributed_pool(char const *name, numa_topology_t topo = {}) noexcept : topology_(std::move(topo)) {
        // Accept null or empty names by falling back to a sensible default
        char const *effective_name = (name && name[0] != '\0') ? name : "fork_union";
        std::strncpy(name_, effective_name, sizeof(name_) - 1);
        name_[sizeof(name_) - 1] = '\0';
    }

    ~linux_distributed_pool() noexcept { terminate(); }

    /**
     *  @brief Checks if the thread-pool's core synchronization points are lock-free.
     *  @note Only valid after the `try_spawn` call.
     */
    bool is_lock_free() const noexcept {
        return colocations_ && colocations_[0] && colocations_[0].only().pool.is_lock_free();
    }

    /**
     *  @brief Returns the NUMA topology used by this thread-pool.
     *  @note This API is @b not synchronized.
     */
    numa_topology_t const &topology() const noexcept { return topology_; }

    /**
     *  @brief Estimates the amount of memory managed by this pool handle and internal structures.
     *  @note This API is @b not synchronized.
     */
    std::size_t memory_usage() const noexcept {
        std::size_t total_bytes = sizeof(linux_distributed_pool);
        for (std::size_t i = 0; i < colocations_.size(); ++i) total_bytes += colocations_[i].only().pool.memory_usage();
        return total_bytes;
    }

#pragma region Core API

    /**
     *  @brief Returns the number of threads in the thread-pool, including the main thread.
     *  @retval 0 if the thread-pool is not initialized, 1 if only the main thread is used.
     *  @note This API is @b not synchronized.
     */
    thread_index_t threads_count() const noexcept { return threads_count_; }

    /**
     *  @brief Reports if the current calling thread will be used for broadcasts.
     *  @note This API is @b not synchronized.
     */
    caller_exclusivity_t caller_exclusivity() const noexcept { return exclusivity_; }

    /**
     *  @brief Creates a thread-pool addressing all cores across all NUMA nodes.
     *  @param[in] threads The number of threads to be used.
     *  @param[in] exclusivity Should we count the calling thread as one of the threads?
     *  @param[in] pin_granularity How to pin the threads to the NUMA node?
     *  @retval false if the number of threads is zero or if spawning has failed.
     *  @retval true if the thread-pool was created successfully, started, and is ready to use.
     *  @note This is the de-facto @b constructor - you only call it again after `terminate`.
     */
    bool try_spawn(                   //
        thread_index_t const threads, //
        caller_exclusivity_t const exclusivity = caller_inclusive_k,
        numa_pin_granularity_t const pin_granularity = numa_pin_to_core_k) noexcept {
        return try_spawn(topology_, threads, exclusivity, pin_granularity);
    }

    /**
     *  @brief Creates a thread-pool addressing all cores across all NUMA nodes.
     *  @param[in] topology The NUMA topology to use for the thread-pool.
     *  @param[in] exclusivity Should we count the calling thread as one of the threads?
     *  @param[in] pin_granularity How to pin the threads to the NUMA node?
     *  @retval false if the number of threads is zero or if spawning has failed.
     *  @retval true if the thread-pool was created successfully, started, and is ready to use.
     *  @note This is the de-facto @b constructor - you only call it again after `terminate`.
     */
    bool try_spawn( //
        numa_topology_t const &topology, caller_exclusivity_t const exclusivity = caller_inclusive_k,
        numa_pin_granularity_t const pin_granularity = numa_pin_to_core_k) noexcept {
        return try_spawn(topology, topology.threads_count(), exclusivity, pin_granularity);
    }

    /**
     *  @brief Creates a thread-pool addressing all cores across all NUMA nodes.
     *  @param[in] topology The NUMA topology to use for the thread-pool.
     *  @param[in] threads The number of threads to be used.
     *  @param[in] exclusivity Should we count the calling thread as one of the threads?
     *  @param[in] pin_granularity How to pin the threads to the NUMA node?
     *  @retval false if the number of threads is zero or if spawning has failed.
     *  @retval true if the thread-pool was created successfully, started, and is ready to use.
     *  @note This is the de-facto @b constructor - you only call it again after `terminate`.
     */
    bool try_spawn( //
        numa_topology_t const &topology,
        thread_index_t const threads, //
        caller_exclusivity_t const exclusivity = caller_inclusive_k,
        numa_pin_granularity_t const pin_granularity = numa_pin_to_core_k) noexcept {

        if (threads == 0) return false;        // ! Can't have zero threads working on something
        if (threads_count_ != 0) return false; // ! Already initialized

        numa_topology_t new_topology;
        if (!new_topology.try_assign(topology)) return false; // ! Copy-construction failed

        // We are going to place the control structures on the first NUMA node,
        // and pin the caller thread to it as well.
        numa_node_t const &first_node = new_topology.node(0);
        numa_node_id_t const first_node_id = first_node.node_id; // ? Typically zero
        linux_numa_allocator_t allocator {first_node_id};
        index_t const colocations_count = std::min(new_topology.nodes_count(), threads);

        colocations_t colocations(allocator);
        if (!colocations.try_resize(colocations_count)) return false; // ! Allocation failed

        // Now allocate each "local pool" on its own NUMA node
        for (index_t colocation_index = 0; colocation_index < colocations_count; ++colocation_index) {
            numa_node_t const &node = new_topology.node(colocation_index);
            numa_node_id_t const node_id = node.node_id;
            linux_numa_allocator_t node_allocator {node_id};
            unique_colocation_buffer_t colocation_padded_buffer(node_allocator);
            colocation_padded_buffer.try_resize(1);
            colocations[colocation_index] = std::move(colocation_padded_buffer);
        }

        auto reset_on_failure = [&]() noexcept {
            for (index_t colocation_index = 0; colocation_index < colocations_count; ++colocation_index) {
                if (colocations[colocation_index].size() == 0) continue; // ? No pool allocated
                colocations[colocation_index].only().pool.terminate();   // ? Stop the pool if it was started
            }
        };

        // If any one of the allocations failed, we need to clean up
        for (index_t colocation_index = 0; colocation_index < colocations_count; ++colocation_index) {
            if (colocations[colocation_index].size() == 1) continue;
            reset_on_failure();
            return false; // ! Allocation failed
        }

        // Every NUMA pool is allocated separately
        // - the first one may be "inclusive".
        // - others are always "exclusive" to the caller thread.
        indexed_split<thread_index_t> threads_per_node(threads, colocations_count);
        if (!colocations[0].only().pool.try_spawn(first_node, threads_per_node[0].count, exclusivity, pin_granularity,
                                                  0, 0)) {
            reset_on_failure();
            return false; // ! Spawning failed
        }

        for (index_t colocation_index = 1; colocation_index < colocations_count; ++colocation_index) {
            numa_node_t const &node = new_topology.node(colocation_index);
            colocation_t &colocation = colocations[colocation_index].only();
            if (!colocation.pool.try_spawn(node, threads_per_node[colocation_index].count, caller_exclusive_k,
                                           pin_granularity, threads_per_node[colocation_index].first,
                                           colocation_index)) {
                reset_on_failure();
                return false; // ! Spawning failed
            }
        }

        topology_ = std::move(new_topology);
        colocations_ = std::move(colocations);
        threads_count_ = threads;
        exclusivity_ = exclusivity;
        return true;
    }

    /**
     *  @brief Executes a @p fork function in parallel on all threads.
     *  @param[in] fork The callback object, receiving the thread index as an argument.
     *  @return A `broadcast_join` synchronization point that waits in the destructor.
     *  @note Even in the `caller_exclusive_k` mode, can be called from just one thread!
     *  @sa For advanced resource management, consider `unsafe_for_threads` and `unsafe_join`.
     */
    template <typename fork_type_>
    FU_REQUIRES_((can_be_for_thread_callback<fork_type_, index_t>()))
    broadcast_join<linux_distributed_pool, fork_type_> for_threads(fork_type_ &&fork) noexcept {
        return {*this, std::forward<fork_type_>(fork)};
    }

    /**
     *  @brief Executes a @p fork function in parallel on all threads, not waiting for the result.
     *  @param[in] fork The callback @b reference, receiving the thread index as an argument.
     *  @sa Use in conjunction with `unsafe_join`.
     */
    template <typename fork_type_>
    FU_REQUIRES_((can_be_for_thread_callback<fork_type_, index_t>()))
    void unsafe_for_threads(fork_type_ &fork) noexcept {
        assert(colocations_ && "Thread pools must be initialized before broadcasting");

        // Submit to every thread pool
        for (std::size_t i = 1; i < colocations_.size(); ++i) colocations_[i].only().pool.unsafe_for_threads(fork);
        colocations_[0].only().pool.unsafe_for_threads(fork);
    }

    /** @brief Blocks the calling thread until the currently broadcasted task finishes. */
    void unsafe_join() noexcept {
        assert(colocations_ && "Thread pools must be initialized before broadcasting");

        // Wait for everyone to finish
        for (std::size_t i = 1; i < colocations_.size(); ++i) colocations_[i].only().pool.unsafe_join();
        colocations_[0].only().pool.unsafe_join();
    }

#pragma endregion Core API

#pragma region Control Flow

    /**
     *  @brief Stops all threads and deallocates the thread-pool after the last call finishes.
     *  @note Can be called from @b any thread at any time.
     *  @note Must `try_spawn` again to re-use the pool.
     *
     *  When and how @b NOT to use this function:
     *  - as a synchronization point between concurrent tasks.
     *
     *  When and how to use this function:
     *  - as a de-facto @b destructor, to stop all threads and deallocate the pool.
     *  - when you want to @b restart with a different number of threads.
     */
    void terminate() noexcept {
        if (!colocations_) return; // ? Uninitialized
        for (std::size_t i = 0; i < colocations_.size(); ++i) colocations_[i].only().pool.terminate();

        colocations_ = {};
        threads_count_ = 0;
        exclusivity_ = caller_inclusive_k;
    }

    /**
     *  @brief Transitions "workers" to a sleeping state, waiting for a wake-up call.
     *  @param[in] wake_up_periodicity_micros How often to check for new work in microseconds.
     *  @note Can only be called @b between the tasks for a single thread. No synchronization is performed.
     *
     *  This function may be used in some batch-processing operations when we clearly understand
     *  that the next task won't be arriving for a while and power can be saved without major
     *  latency penalties.
     *
     *  It may also be used in a high-level Python or JavaScript library offloading some parallel
     *  operations to an underlying C++ engine, where latency is irrelevant.
     */
    void sleep(std::size_t wake_up_periodicity_micros) noexcept {
        assert(wake_up_periodicity_micros > 0 && "Sleep length must be positive");
        for (std::size_t i = 0; i < colocations_.size(); ++i)
            colocations_[i].only().pool.sleep(wake_up_periodicity_micros);
    }

    /** @brief Helper function to create a spin mutex with same yield characteristics. */
    static spin_mutex<micro_yield_t, alignment_k> make_mutex() noexcept { return {}; }

#pragma endregion Control Flow

#pragma region Indexed Task Scheduling

    /**
     *  @brief Distributes @p `n` similar duration calls between threads in slices, as opposed to individual indices.
     *  @param[in] n The total length of the range to split between threads.
     *  @param[in] fork The callback, receiving the first @b `prong_t` and the slice length.
     */
    template <typename fork_type_ = dummy_lambda_t>
    FU_REQUIRES_((can_be_for_slice_callback<fork_type_, index_t>()))
    broadcast_join<linux_distributed_pool,
                   invoke_distributed_for_slices<linux_distributed_pool, fork_type_, index_t>> //
        for_slices(index_t const n, fork_type_ &&fork) noexcept {

        return {*this, {*this, n, threads_count(), std::forward<fork_type_>(fork)}};
    }

    /**
     *  @brief Distributes @p `n` similar duration calls between threads.
     *  @param[in] n The number of times to call the @p fork.
     *  @param[in] fork The callback object, receiving @b `prong_t` or a call index as an argument.
     *
     *  Is designed for a "balanced" workload, where all threads have roughly the same amount of work.
     *  @sa `for_n_dynamic` for a more dynamic workload.
     *  The @p fork is called @p `n` times, and each thread receives a slice of consecutive tasks.
     *  @sa `for_slices` if you prefer to receive workload slices over individual indices.
     */
    template <typename fork_type_ = dummy_lambda_t>
    FU_REQUIRES_((can_be_for_task_callback<fork_type_, index_t>()))
    broadcast_join<linux_distributed_pool, invoke_distributed_for_n<linux_distributed_pool, fork_type_, index_t>> //
        for_n(index_t const n, fork_type_ &&fork) noexcept {

        return {*this, {*this, n, threads_count(), std::forward<fork_type_>(fork)}};
    }

    /**
     *  @brief Executes uneven tasks on all threads, greedying for work.
     *  @param[in] n The number of times to call the @p fork.
     *  @param[in] fork The callback object, receiving the `prong_t` or the task index as an argument.
     *  @sa `for_n` for a more "balanced" evenly-splittable workload.
     */
    template <typename fork_type_ = dummy_lambda_t>
    FU_REQUIRES_((can_be_for_task_callback<fork_type_, index_t>()))
    broadcast_join<linux_distributed_pool,
                   invoke_distributed_for_n_dynamic<linux_distributed_pool, fork_type_, index_t>> //
        for_n_dynamic(index_t const n, fork_type_ &&fork) noexcept {

        return {*this, {*this, n, std::forward<fork_type_>(fork)}};
    }

#pragma endregion Indexed Task Scheduling

#pragma region Colocations Compatibility

    /**
     *  @brief Number of individual sub-pool with the same NUMA-locality and QoS.
     */
    index_t colocations_count() const noexcept { return colocations_.size(); }

    /**
     *  @brief Returns the number of threads in one NUMA-specific local @b colocation.
     *  @retval 0 if the thread-pool is not initialized, 1 if only the main thread is used.
     *  @note This API is @b not synchronized and doesn't check for out-of-bounds access.
     */
    thread_index_t threads_count(index_t colocation) const noexcept {
        assert(colocations_ && "Local pools must be initialized");
        assert(colocation < colocations_.size() && "Local pool index out of bounds");
        return colocations_[colocation].only().pool.threads_count();
    }

    /**
     *  @brief Converts a @p `global_thread_index` to a local thread index within a @b colocation.
     *  @retval 0 if the thread-pool is not initialized, 1 if only the main thread is used.
     *  @note This API is @b not synchronized and doesn't check for out-of-bounds access.
     */
    thread_index_t thread_local_index(thread_index_t global_thread_index, index_t colocation) const noexcept {
        assert(colocations_ && "Local pools must be initialized");
        assert(colocation < colocations_.size() && "Local pool index out of bounds");
        return global_thread_index - colocations_[colocation].only().pool.first_thread();
    }

    index_t thread_colocation(thread_index_t global_thread_index) const noexcept {
        index_t colocation_index = 0;
        for (; colocation_index < colocations_.size(); ++colocation_index) {
            colocation_t const &colocation = colocations_[colocation_index].only();
            if (global_thread_index < colocation.pool.first_thread()) continue;
            if (global_thread_index < colocation.pool.first_thread() + colocation.pool.threads_count())
                return colocation_index;
        }
        return colocation_index; // ? Not found
    }

    std::atomic<index_t> &unsafe_dynamic_progress_ref(index_t colocation) noexcept {
        return colocations_[colocation].only().pool.unsafe_dynamic_progress_ref();
    }

#pragma endregion Colocations Compatibility
};

using linux_colocated_pool_t = linux_colocated_pool<>;
using linux_distributed_pool_t = linux_distributed_pool<>;

#if FU_DETECT_CONCEPTS_
static_assert(is_unsafe_pool<basic_pool_t> && is_unsafe_pool<linux_colocated_pool_t>,
              "These thread pools must be flexible and support unsafe operations");
static_assert(is_pool<basic_pool_t> && is_pool<linux_colocated_pool_t> && is_pool<linux_distributed_pool_t>,
              "These thread pools must be fully compatible with the high-level APIs");
#endif // FU_DETECT_CONCEPTS_

#endif // FU_ENABLE_NUMA
#pragma endregion - NUMA Pools

#pragma region - Logging

/**
 *  @brief Detects if the output stream supports ANSI color codes.
 */
struct logging_colors_t {
    bool use_colors_ = false;

    explicit logging_colors_t(bool use_colors) noexcept : use_colors_(use_colors) {}

    explicit logging_colors_t() noexcept {
#if defined(_WIN32)
        if (!::_isatty(_fileno(stdout))) return;
#endif
#if defined(__unix__) || defined(__APPLE__)
        if (!::isatty(STDOUT_FILENO)) return;
#endif
#if defined(_WIN32)
        // On Windows, assume color support is available
        use_colors_ = true;
#else
        char const *term = std::getenv("TERM");
        if (!term) return;
        use_colors_ = std::strstr(term, "color") != nullptr || std::strstr(term, "xterm") != nullptr ||
                      std::strstr(term, "screen") != nullptr || std::strcmp(term, "linux") == 0;
#endif
    }

    /* ANSI style codes */
    char const *reset() const noexcept { return use_colors_ ? "\033[0m" : ""; }
    char const *bold() const noexcept { return use_colors_ ? "\033[1m" : ""; }
    char const *dim() const noexcept { return use_colors_ ? "\033[2m" : ""; }

    /* ANSI color codes */
    char const *red() const noexcept { return use_colors_ ? "\033[31m" : ""; }
    char const *green() const noexcept { return use_colors_ ? "\033[32m" : ""; }
    char const *yellow() const noexcept { return use_colors_ ? "\033[33m" : ""; }
    char const *blue() const noexcept { return use_colors_ ? "\033[34m" : ""; }
    char const *magenta() const noexcept { return use_colors_ ? "\033[35m" : ""; }
    char const *cyan() const noexcept { return use_colors_ ? "\033[36m" : ""; }
    char const *white() const noexcept { return use_colors_ ? "\033[37m" : ""; }
    char const *gray() const noexcept { return use_colors_ ? "\033[90m" : ""; }

    /* Compound styles */
    char const *bold_red() const noexcept { return use_colors_ ? "\033[1;31m" : ""; }
    char const *bold_green() const noexcept { return use_colors_ ? "\033[1;32m" : ""; }
    char const *bold_yellow() const noexcept { return use_colors_ ? "\033[1;33m" : ""; }
    char const *bold_blue() const noexcept { return use_colors_ ? "\033[1;34m" : ""; }
    char const *bold_magenta() const noexcept { return use_colors_ ? "\033[1;35m" : ""; }
    char const *bold_cyan() const noexcept { return use_colors_ ? "\033[1;36m" : ""; }
    char const *bold_white() const noexcept { return use_colors_ ? "\033[1;37m" : ""; }
    char const *bold_gray() const noexcept { return use_colors_ ? "\033[1;90m" : ""; }
};

/**
 *  @brief Formats memory volume in @p `bytes` with appropriate units and precision, like @b "1.5 GiB".
 */
struct log_memory_volume_t {

    void operator()(std::size_t bytes, char *buffer, std::size_t buffer_size, logging_colors_t colors) const noexcept {

        char const *value_color = colors.bold_white();
        char const *unit_color = colors.dim();
        char const *reset_color = colors.reset();

        if (bytes >= (1ull << 40)) {
            double tb = static_cast<double>(bytes) / (1ull << 40);
            std::snprintf(buffer, buffer_size, "%s%.1f%s %sTiB%s", value_color, tb, unit_color, unit_color,
                          reset_color);
        }
        else if (bytes >= (1ull << 30)) {
            double gb = static_cast<double>(bytes) / (1ull << 30);
            std::snprintf(buffer, buffer_size, "%s%.1f%s %sGiB%s", value_color, gb, unit_color, unit_color,
                          reset_color);
        }
        else if (bytes >= (1ull << 20)) {
            double mb = static_cast<double>(bytes) / (1ull << 20);
            std::snprintf(buffer, buffer_size, "%s%.1f%s %sMiB%s", value_color, mb, unit_color, unit_color,
                          reset_color);
        }
        else if (bytes >= (1ull << 10)) {
            double kb = static_cast<double>(bytes) / (1ull << 10);
            std::snprintf(buffer, buffer_size, "%s%.1f%s %sKiB%s", value_color, kb, unit_color, unit_color,
                          reset_color);
        }
        else {
            std::snprintf(buffer, buffer_size, "%s%zu%s %sB%s", value_color, bytes, unit_color, unit_color,
                          reset_color);
        }
    }
};

/**
 *  @brief Formats a set of CPU core IDs in a compact and readable way, like @b "0-3,5,7,8,10-12".
 */
struct log_core_range_t {

    void operator()(                                       //
        numa_core_id_t const *core_ids, std::size_t count, //
        char *buffer, std::size_t buffer_size, logging_colors_t colors) const noexcept {

        if (count == 0) {
            std::snprintf(buffer, buffer_size, "%snone%s", colors.dim(), colors.reset());
            return;
        }

        char const *value_color = colors.bold_white();
        char const *reset_color = colors.reset();

        if (count == 1) {
            std::snprintf(buffer, buffer_size, "%s%d%s", value_color, core_ids[0], reset_color);
            return;
        }

        // Check if it's a contiguous range
        bool is_contiguous = true;
        for (std::size_t i = 1; i < count && is_contiguous; ++i)
            if (core_ids[i] != core_ids[i - 1] + 1) is_contiguous = false;

        if (is_contiguous) {
            std::snprintf(                            //
                buffer, buffer_size, "%s%d%s-%s%d%s", //
                value_color, core_ids[0], reset_color, value_color, core_ids[count - 1], reset_color);
        }
        else {
            // Show first few and last few with ellipsis if many cores
            if (count <= 8) {
                int written = std::snprintf(buffer, buffer_size, "%s%d%s", value_color, core_ids[0], reset_color);
                for (std::size_t i = 1; i < count && written < static_cast<int>(buffer_size) - 1; ++i)
                    written += std::snprintf(                                                         //
                        buffer + written, buffer_size - static_cast<std::size_t>(written), ",%s%d%s", //
                        value_color, core_ids[i], reset_color);
            }
            else {
                std::snprintf(                                                        //
                    buffer, buffer_size, "%s%d%s,%s%d%s,%s%d%s…%s%d%s,%s%d%s,%s%d%s", //
                    value_color, core_ids[0], reset_color, value_color, core_ids[1], reset_color, value_color,
                    core_ids[2], reset_color, value_color, core_ids[count - 3], reset_color, value_color,
                    core_ids[count - 2], reset_color, value_color, core_ids[count - 1], reset_color);
            }
        }
    }
};

/**
 *  @brief NUMA topology logger with compact tree design and color support.
 */
struct log_numa_topology_t {

    /**
     *  @brief Logs NUMA topology in compact tree format with colors.
     *  @param topology The NUMA topology to log
     *  @param colors Color scheme for output formatting
     *  @param output Output file stream (defaults to stdout)
     */
    template <std::size_t max_page_sizes_, typename allocator_type_>
    void operator()(numa_topology<max_page_sizes_, allocator_type_> const &topology, logging_colors_t colors,
                    std::FILE *output = stdout) const noexcept {

        // Line buffer for assembly
        char line_buffer[1024];
        logging_colors_t colorless {false};

        // Helper lambda to flush line buffer
        auto flush_line = [&]() { std::fprintf(output, "%s", line_buffer); };

        // Main header
        std::snprintf(line_buffer, sizeof(line_buffer), "%sNUMA Layout%s\n", colors.bold_cyan(), colors.reset());
        flush_line();

        if (topology.nodes_count() == 0) {
            std::snprintf(line_buffer, sizeof(line_buffer), "%sNo NUMA nodes detected%s\n", colors.dim(),
                          colors.reset());
            flush_line();
            return;
        }

        // Get the last socket ID for comparison
        int last_socket_id = topology.node(topology.nodes_count() - 1).socket_id;
        int current_socket_id = -1;

        for (std::size_t i = 0; i < topology.nodes_count(); ++i) {
            auto const node = topology.node(i);

            // Print socket header when we encounter a new socket
            if (node.socket_id != current_socket_id) {
                current_socket_id = node.socket_id;
                bool is_last_socket = current_socket_id == last_socket_id;

                std::snprintf(                                                     //
                    line_buffer, sizeof(line_buffer), "%s%s─ %sSocket%s %s%d%s\n", //
                    colors.dim(), is_last_socket ? "└" : "├",                      //
                    colors.blue(), /* "Socket" */ colors.reset(),                  //
                    colors.bold_blue(), current_socket_id, colors.reset());
                flush_line();
            }

            // Check if this is the last node in current socket
            bool is_last_node_in_socket =
                (i + 1 >= topology.nodes_count() || topology.node(i + 1).socket_id != current_socket_id);

            // Format core range and memory
            char cores_str[256], memory_str[64];
            log_core_range_t {}(node.first_core_id, node.core_count, cores_str, sizeof(cores_str), colorless);
            log_memory_volume_t {}(node.memory_size, memory_str, sizeof(memory_str), colorless);

            // Tree structure prefixes
            bool is_last_socket = current_socket_id == last_socket_id;
            char const *socket_prefix = is_last_socket ? "   " : "│  ";
            char const *node_connector = is_last_node_in_socket ? "└─ " : "├─ ";

            // Start building node line
            int pos = std::snprintf(                                                    //
                line_buffer, sizeof(line_buffer),                                       //
                "%s%s%s%sNode%s %s%d%s • %sCores:%s %s%s (%zu)%s • %sMemory:%s %s%s%s", //
                colors.dim(), socket_prefix, node_connector,                            //
                colors.cyan(), /* "Node" */ colors.reset(),                             //
                colors.bold_cyan(), node.node_id, colors.reset(),                       //
                colors.green(), /* "Cores:" */ colors.reset(),                          //
                colors.bold_green(), cores_str, node.core_count, colors.reset(),        //
                colors.yellow(), /* "Memory:" */ colors.reset(),                        //
                colors.bold_yellow(), memory_str, colors.reset());

            // Add huge pages if any exist
            auto const &page_settings = node.page_sizes;
            bool first_page = true;

            for (std::size_t j = 0; j < page_settings.size(); ++j) {
                auto const &ps = page_settings[j];
                if (ps.bytes_per_page <= 4096) continue; // Skip regular pages

                if (first_page) {
                    pos += static_cast<std::size_t>(std::snprintf(                      //
                        line_buffer + pos, sizeof(line_buffer) - pos, " • %sPages:%s ", //
                        colors.magenta(), /* "Pages:" */ colors.reset()));
                    first_page = false;
                }
                else
                    pos += static_cast<std::size_t>(std::snprintf(line_buffer + pos, sizeof(line_buffer) - pos, " "));

                char page_size_str[32], page_volume_str[32];
                std::size_t free_bytes = ps.free_pages * ps.bytes_per_page;
                log_memory_volume_t {}(ps.bytes_per_page, page_size_str, sizeof(page_size_str), colorless);
                log_memory_volume_t {}(free_bytes, page_volume_str, sizeof(page_volume_str), colorless);

                pos += static_cast<std::size_t>(std::snprintf(                   //
                    line_buffer + pos, sizeof(line_buffer) - pos, "%s%s (%s)%s", //
                    colors.bold_magenta(), page_size_str, page_volume_str, colors.reset()));
            }

            std::snprintf(line_buffer + pos, sizeof(line_buffer) - pos, "\n");
            flush_line();
        }

        // Final newline
        std::snprintf(line_buffer, sizeof(line_buffer), "\n");
        flush_line();
    }
};

/**
 *  @brief Logs CPU and memory capabilities summary with compact formatting.
 */
struct log_capabilities_t {

    void operator()(capabilities_t caps, logging_colors_t colors, std::FILE *output = stdout) const noexcept {

        // Line buffer for assembly
        char line_buffer[1024];

        // Helper lambda to flush line buffer
        auto flush_line = [&]() { std::fprintf(output, "%s", line_buffer); };

        // Main header
        std::snprintf(line_buffer, sizeof(line_buffer), "%sSystem Capabilities%s\n", colors.bold_cyan(),
                      colors.reset());
        flush_line();

        // CPU Capabilities row
        std::snprintf(line_buffer, sizeof(line_buffer), "%s├─ %sCPU:%s ", colors.dim(), colors.cyan(), colors.reset());
        std::size_t pos = std::strlen(line_buffer);

        bool first_cpu = true;
        if (caps & capability_x86_pause_k) {
            pos +=
                static_cast<std::size_t>(std::snprintf(line_buffer + pos, sizeof(line_buffer) - pos, "%s%sx86 PAUSE%s",
                                                       first_cpu ? "" : " • ", colors.bold_green(), colors.reset()));
            first_cpu = false;
        }
        if (caps & capability_x86_tpause_k) {
            pos +=
                static_cast<std::size_t>(std::snprintf(line_buffer + pos, sizeof(line_buffer) - pos, "%s%sx86 TPAUSE%s",
                                                       first_cpu ? "" : " • ", colors.bold_green(), colors.reset()));
            first_cpu = false;
        }
        if (caps & capability_arm64_yield_k) {
            pos += static_cast<std::size_t>(std::snprintf(line_buffer + pos, sizeof(line_buffer) - pos,
                                                          "%s%sARM64 YIELD%s", first_cpu ? "" : " • ",
                                                          colors.bold_green(), colors.reset()));
            first_cpu = false;
        }
        if (caps & capability_arm64_wfet_k) {
            pos +=
                static_cast<std::size_t>(std::snprintf(line_buffer + pos, sizeof(line_buffer) - pos, "%s%sARM64 WFET%s",
                                                       first_cpu ? "" : " • ", colors.bold_green(), colors.reset()));
            first_cpu = false;
        }
        if (caps & capability_risc5_pause_k) {
            pos += static_cast<std::size_t>(std::snprintf(line_buffer + pos, sizeof(line_buffer) - pos,
                                                          "%s%sRISC-V PAUSE%s", first_cpu ? "" : " • ",
                                                          colors.bold_green(), colors.reset()));
            first_cpu = false;
        }

        if (first_cpu) {
            pos += static_cast<std::size_t>(std::snprintf(line_buffer + pos, sizeof(line_buffer) - pos,
                                                          "%sNone detected%s", colors.dim(), colors.reset()));
        }

        std::snprintf(line_buffer + pos, sizeof(line_buffer) - pos, "\n");
        flush_line();

        // Memory Capabilities row
        std::snprintf(line_buffer, sizeof(line_buffer), "%s└─ %sRAM:%s ", colors.dim(), colors.cyan(), colors.reset());
        pos = std::strlen(line_buffer);

        bool first_mem = true;
        if (caps & capability_numa_aware_k) {
            pos +=
                static_cast<std::size_t>(std::snprintf(line_buffer + pos, sizeof(line_buffer) - pos, "%s%sNUMA%s",
                                                       first_mem ? "" : " • ", colors.bold_yellow(), colors.reset()));
            first_mem = false;
        }
        if (caps & capability_huge_pages_k) {
            pos +=
                static_cast<std::size_t>(std::snprintf(line_buffer + pos, sizeof(line_buffer) - pos, "%s%sHuge Pages%s",
                                                       first_mem ? "" : " • ", colors.bold_yellow(), colors.reset()));
            first_mem = false;
        }
        if (caps & capability_huge_pages_transparent_k) {
            pos += static_cast<std::size_t>(std::snprintf(line_buffer + pos, sizeof(line_buffer) - pos,
                                                          "%s%sTransparent Huge Pages%s", first_mem ? "" : " • ",
                                                          colors.bold_yellow(), colors.reset()));
            first_mem = false;
        }

        if (first_mem) {
            pos += static_cast<std::size_t>(std::snprintf(line_buffer + pos, sizeof(line_buffer) - pos,
                                                          "%sNone detected%s", colors.dim(), colors.reset()));
        }

        std::snprintf(line_buffer + pos, sizeof(line_buffer) - pos, "\n\n");
        flush_line();
    }
};
#pragma endregion - Logging

} // namespace fork_union
} // namespace ashvardanian
