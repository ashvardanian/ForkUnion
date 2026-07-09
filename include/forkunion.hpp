/**
 *  @brief  Low-latency OpenMP-style NUMA-aware cross-platform fine-grained parallelism library.
 *  @file   forkunion.hpp
 *  @author Ash Vardanian
 *  @date   May 2, 2025
 *
 *  ForkUnion provides a minimalistic cross-platform thread-pool implementation and Parallel Algorithms,
 *  avoiding dynamic memory allocations, exceptions, system calls, and heavy Compare-And-Swap instructions.
 *  The library leverages the "weak memory model" to allow Arm and IBM Power CPUs to aggressively optimize
 *  execution at runtime. It also aggressively tests against overflows on smaller index types, and is safe
 *  to use even with the maximal `std::size_t` values.
 *
 *  @code{.cpp}
 *  #include <cstdio> // `std::printf`
 *  #include <cstdlib> // `EXIT_FAILURE`, `EXIT_SUCCESS`
 *  #include <forkunion.hpp> // `fu::basic_pool_t`
 *
 *  using fu = ashvardanian::forkunion;
 *  int main(int argc, char *argv[]) {
 *
 *      fu::basic_pool_t pool;
 *      if (!pool.try_spawn(std::thread::hardware_concurrency()))
 *          return EXIT_FAILURE;
 *
 *      pool.for_n(argc, [=](auto prong) noexcept {
 *          auto [task_index, thread_index, compute_domain_index] = prong;
 *          std::printf(
 *              "Printing argument # %zu (of %zu) from thread # %zu at compute_domain # %zu: %s\n",
 *              task_index, argc, thread_index, compute_domain_index, argv[task_index]);
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
 *  On Linux, when NUMA and PThreads are available, the library can also leverage @b NUMA-aware
 *  memory allocations and pin threads to specific physical cores to increase memory locality.
 *  It should reduce memory access latency by around 35% on average, compared to remote accesses.
 *  @sa `numa_topology_t`, `linux_compute_domain_pool_t`, `linux_distributed_pool_t`.
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
namespace forkunion {

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

    // Pool-topology capabilities:
    capability_compute_domain_k = 1 << 6, // ? Pinned to a single compute_domain (a same-QoS core cluster)

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
    fork_t fork_;                 // ? We need this to extend the lifetime of the lambda object
    generation_t generation_ {0}; // ? Real tokens are odd; zero means "not yet dispatched"

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
    /** @brief Next task in this slice; only ever grows, and may overshoot `end` by `threads`. */
    std::atomic<index_type_> next {0};
    /** @brief One past this slice's last task. Written once before the dispatch, then read-only. */
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
 *  That trailing reservation also bounds the cursors. Every thread overshoots a given slice at most
 *  once - it increments, sees `>= end`, and leaves - so a cursor tops out at `end + threads`. Since
 *  the last slice ends at `n - threads`, no cursor can exceed `n`, whatever the index type.
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
 *  A minimal example, similar to `#pragma omp parallel` in OpenMP:
 *
 *  @code{.cpp}
 *  #include <cstdio> // `std::printf`
 *  #include <cstdlib> // `EXIT_FAILURE`, `EXIT_SUCCESS`
 *  #include <forkunion.hpp> // `basic_pool_t`
 *
 *  using fu = ashvardanian::forkunion;
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
 *  #include <forkunion.hpp> // `basic_pool_t`
 *
 *  using fu = ashvardanian::forkunion;
 *  int main() {
 *      fu::basic_pool_t first_pool, second_pool;
 *      if (!first_pool.try_spawn(2) || !second_pool.try_spawn(2, fu::caller_exclusive_k)) return EXIT_FAILURE;
 *      auto broadcast = second_pool.for_threads([](std::size_t i) noexcept { poll_ssd(i); });
 *      first_pool.for_threads([](std::size_t i) noexcept { poll_nic(i); });
 *      broadcast.join(); // ! Wait for the second pool to finish
 *      return EXIT_SUCCESS;
 *  }
 *  @endcode
 *
 *  @section pool_concurrency_model Concurrency Model
 *
 *  Three roles interact with a pool:
 *  - the @b dispatcher - exactly one external thread operating the pool at a time: it dispatches,
 *    polls, joins, and terminates; this is contractual and not enforced;
 *  - the @b contributors - threads executing one slice each per generation: all workers, plus the
 *    calling thread itself on `caller_inclusive_k` pools, whose slice runs inside `unsafe_join`;
 *  - the @b pollers - any threads calling `is_complete`, which is a read-only probe.
 *
 *  All synchronization is built from three cache-line-aligned atomics and plain loads, stores,
 *  and fetch-add/sub increments - no compare-and-swap chains and no mutexes on the hot path:
 *  - `epoch_` - the generation clock: @b odd while a fork is in flight, @b even when idle;
 *    incremented once by the dispatcher on dispatch and once by the last contributor on
 *    completion, so every generation advances it by exactly two;
 *  - `threads_to_sync_` - the countdown identifying the @b last contributor - the only thread
 *    allowed to make the completion increment;
 *  - `mood_` - the lifecycle switch between spinning, sleeping, and exiting workers.
 *
 *  Four synchronization edges keep the non-atomic fork state safe:
 *  1. @b publish: the dispatcher writes the fork state, resets the countdown, and releases the
 *     dispatch increment; contributors acquire it and see both;
 *  2. @b completion @b chain: every contributor decrements the countdown with `acq_rel`, chaining
 *     each contributor's writes into the last one;
 *  3. @b completion @b edge: the last contributor releases the completion increment, so any
 *     acquire-load observing it sees @b all contributors' results - `is_complete` included;
 *  4. @b join: the dispatcher blocks until the completion increment, so a new dispatch can never
 *     race with the previous completion, and generation tokens are always odd.
 *
 *  On `caller_inclusive_k` pools the calling thread owes a slice that only runs inside
 *  `unsafe_join`, so `is_complete` stays `false` until then: the poll-then-join pattern is
 *  reserved for `caller_exclusive_k` pools.
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
    static constexpr std::size_t alignment_k = alignment_;
    static_assert(is_power_of_two(alignment_k), "Alignment must be a power of 2");

    using index_t = index_type_;
    static_assert(std::is_unsigned<index_t>::value, "Index type must be an unsigned integer");
    using epoch_index_t = index_t;      // ? A.k.a. number of previous API calls in [0, UINT_MAX)
    using generation_t = epoch_index_t; // ? A.k.a. token returned from `unsafe_for_threads`; always odd
    // ! With small index types (like the `fu8_t`/`fu16_t` debug configs) a worker stalled across
    // ! exactly 2^bits epochs would alias its `last_epoch` - astronomically unlikely at `size_t`.
    using thread_index_t = index_t;         // ? A.k.a. "core index" or "thread ID" in [0, threads_count)
    using compute_domain_index_t = index_t; // ? A.k.a. "NUMA node ID" in [0, numa_nodes_count)
    using indexed_split_t = indexed_split<index_t>;
    using prong_t = prong<index_t>;
    using local_thread_t = local_thread<index_t>;
    using claim_t = dynamic_claim<index_t>; // ? One private cursor per thread

    /**
     *  @brief Everything the pool keeps @b per @b thread, on a cache line of its own.
     *
     *  The claim cursor must not share a line with anything, or the dynamic scheduler reintroduces
     *  the very coherence traffic that giving each thread a private cursor exists to remove. Rather
     *  than allocate a second array beside `std::thread`, both live in one padded cell, so the pool
     *  still performs exactly one allocation - in `try_spawn`, never on a dispatch path.
     *
     *  Cells are indexed by @b thread @b index, so on inclusive pools cell 0 belongs to the caller
     *  and holds no `std::thread`. That costs one cell and buys `claim` and `worker` the same index.
     *
     *  @note Separation comes from the buffer's @b stride, not from an `alignas` on this type. A
     *        `std::allocator` only promises `__STDCPP_DEFAULT_NEW_ALIGNMENT__`, so over-aligning the
     *        cell would placement-new it into storage that cannot satisfy the request.
     */
    struct worker_cell_t {
        claim_t claim {};
        std::thread worker {}; // ? Default-constructed, and left so for the caller's own cell
    };
    static_assert(sizeof(worker_cell_t) <= alignment_k, "A worker cell must fit within one stride");

    using worker_cell_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<worker_cell_t>;
    using worker_cells_t = unique_padded_buffer<worker_cell_t, worker_cell_allocator_t>;

    using punned_fork_context_t = void *;                                 // ? Pointer to the on-stack lambda
    using trampoline_t = void (*)(punned_fork_context_t, thread_index_t); // ? Wraps lambda's `operator()`

    using micro_yield_traits_t = yield_traits<micro_yield_t, thread_index_t>;
    static_assert(micro_yield_traits_t::valid, "Yield must be invocable w/out args or with a thread index");

  private:
    // Thread-pool-specific variables:
    allocator_t allocator_ {};
    worker_cells_t workers_ {}; // ? One padded cell per thread: its `std::thread` and its claim cursor
    thread_index_t threads_count_ {0};
    caller_exclusivity_t exclusivity_ {caller_inclusive_k}; // ? Whether the caller thread is included in the count
    std::size_t sleep_length_micros_ {0}; // ? How long to sleep in microseconds when waiting for tasks
    alignas(alignment_k) std::atomic<mood_t> mood_ {mood_t::grind_k};

    // Task-specific variables:
    punned_fork_context_t fork_state_ {nullptr}; // ? Pointer to the users lambda
    trampoline_t fork_trampoline_ {nullptr};     // ? Calls the lambda
    alignas(alignment_k) std::atomic<thread_index_t> threads_to_sync_ {0};
    alignas(alignment_k) std::atomic<epoch_index_t> epoch_ {0};

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
    std::size_t memory_usage() const noexcept { return sizeof(basic_pool) + workers_.size() * workers_.stride(); }

    /** @brief Checks if the thread-pool's core synchronization points are lock-free. */
    bool is_lock_free() const noexcept { return mood_.is_lock_free() && threads_to_sync_.is_lock_free(); }

    /**
     *  @brief Returns the NUMA node ID this thread-pool is pinned to.
     *  @retval -1 as this thread-pool is not NUMA-aware.
     */
    constexpr numa_node_id_t numa_node_id() const noexcept { return -1; }

    /**
     *  @brief Returns the first thread index in the thread-pool.
     *  @retval 0 as this pool isn't intended for compute_domain/distributed topologies.
     */
    constexpr thread_index_t first_thread() const noexcept { return 0; }

    /** @brief Exposes a thread's private claim cursor. Use with caution. */
    claim_t &unsafe_dynamic_claim_ref(thread_index_t const thread) noexcept { return workers_[thread].claim; }

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
            return true; // ! The current thread will always be used, and allocates nothing
        }

        // Allocate the thread pool: one padded cell per thread, holding its worker and its cursor.
        // This is the pool's only allocation, and `for_n_dynamic` performs none of its own. Striding
        // by `alignment_k` is what keeps two threads' cursors off a shared cache line.
        worker_cells_t cells {worker_cell_allocator_t {allocator_}, alignment_k};
        if (!cells.try_resize(threads)) return false; // ! Allocation failed

        // Before we start the threads, make sure we set some of the shared
        // state variables that will be used in the `_worker_loop` function.
        workers_ = std::move(cells);
        threads_count_ = threads;
        exclusivity_ = exclusivity;
        mood_.store(mood_t::grind_k, std::memory_order_release);
        auto reset_on_failure = [&]() noexcept {
            workers_ = {}; // ? Cells are default-constructed, so no `std::thread` is joinable here
            threads_count_ = 0;
        };

        // Initializing the thread pool can fail for all kinds of reasons,
        // that the `std::thread` documentation describes as "implementation-defined".
        // https://en.cppreference.com/w/cpp/thread/thread/thread
        thread_index_t const worker_threads = threads - use_caller_thread;
        auto spawn_worker = [&](thread_index_t i) noexcept -> bool {
            thread_index_t const i_with_caller = i + use_caller_thread;
#if FU_ALLOW_UNSAFE
            try {
                workers_[i_with_caller].worker = std::thread([this, i_with_caller] { _worker_loop(i_with_caller); });
                return true;
            }
            catch (...) {
                return false;
            }
#else
            workers_[i_with_caller].worker = std::thread([this, i_with_caller] { _worker_loop(i_with_caller); });
            return true;
#endif
        };

        for (thread_index_t i = 0; i < worker_threads; ++i) {
            if (spawn_worker(i)) continue;

            // ! Failed to spawn a thread, roll back everything
            mood_.store(mood_t::die_k, std::memory_order_release);
            for (thread_index_t j = 0; j < i; ++j) workers_[j + use_caller_thread].worker.join();
            reset_on_failure();
            return false;
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
     *  @return A `generation_t` token identifying this dispatch.
     *  @sa Use in conjunction with `unsafe_join`.
     */
    template <typename fork_type_>
    FU_REQUIRES_((can_be_for_thread_callback<fork_type_, index_t>()))
    generation_t unsafe_for_threads(fork_type_ &fork) noexcept {

        thread_index_t const threads = threads_count();
        assert(threads != 0 && "Thread pool not initialized");

        // Only one dispatch can be in flight, and it must be fully joined - the caller's
        // slice included - before the next one starts.
        assert(threads_to_sync_.load(std::memory_order_acquire) == 0 &&
               "The broadcast function can't be called concurrently or recursively");
        assert((epoch_.load(std::memory_order_relaxed) & 1u) == 0 && "Previous dispatch not joined");

        // Configure "fork" details
        fork_state_ = std::addressof(fork);
        fork_trampoline_ = &_call_as_lambda<fork_type_>;

        // Every contributor gets counted: all worker threads, plus the calling thread itself
        // on `caller_inclusive_k` pools, where its slice runs inside `unsafe_join`.
        threads_to_sync_.store(threads, std::memory_order_relaxed);

        // We are most likely already "grinding", but in the unlikely case we are not,
        // let's wake up from the "chilling" state with relaxed semantics. Assuming the sleeping
        // logic for the workers also checks the epoch counter, no synchronization is needed and
        // no immediate wake-up is required.
        mood_t may_be_chilling = mood_t::chill_k;
        mood_.compare_exchange_weak(          //
            may_be_chilling, mood_t::grind_k, //
            std::memory_order_relaxed, std::memory_order_relaxed);
        return static_cast<generation_t>(epoch_.fetch_add(1, std::memory_order_release) + 1);
    }

    /**
     *  @brief Returns true if the generation identified by @p generation has completed.
     *  @note A `true` result synchronizes with all contributors: their writes are visible.
     *
     *  On `caller_inclusive_k` pools this can only turn `true` once `unsafe_join`
     *  contributes the calling thread's slice, so the poll-then-join pattern is
     *  reserved for `caller_exclusive_k` pools.
     */
    bool is_complete(generation_t generation) const noexcept {
        return generation != epoch_.load(std::memory_order_acquire);
    }

    /**
     *  @brief Blocks the calling thread until the generation identified by @p generation finishes.
     *  @note On `caller_inclusive_k` pools, first executes the calling thread's slice.
     *  Idempotent: returns immediately for already-joined or stale generations.
     */
    void unsafe_join(generation_t generation) noexcept {
        assert((generation & 1u) == 1 && "Generation tokens are always odd");
        if (epoch_.load(std::memory_order_acquire) != generation) return; // ? Stale or already complete

        // On inclusive pools the calling thread is a contributor: execute its slice
        // and count it down exactly like a worker thread would.
        bool const use_caller_thread = caller_exclusivity() == caller_inclusive_k;
        if (use_caller_thread) {
            fork_trampoline_(fork_state_, static_cast<thread_index_t>(0));
            thread_index_t const before_decrement = threads_to_sync_.fetch_sub(1, std::memory_order_acq_rel);
            assert(before_decrement > 0 && "The contributor count must include the caller");

            // The last contributor to finish increments the epoch, signaling completion
            if (before_decrement == 1) epoch_.fetch_add(1, std::memory_order_release);
        }

        // Wait for the last contributor's completion increment
        micro_yield_t micro_yield;
        while (epoch_.load(std::memory_order_acquire) == generation)
            call_yield_(micro_yield, static_cast<thread_index_t>(0));
    }

    /** @brief Blocks the calling thread until the currently broadcasted task finishes. */
    void unsafe_join() noexcept {
        epoch_index_t const current_epoch = epoch_.load(std::memory_order_acquire);
        if (current_epoch & 1u) unsafe_join(static_cast<generation_t>(current_epoch)); // ? Even means idle
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
            return; // ? No worker threads to join, and nothing was allocated
        }
        assert(threads_to_sync_.load(std::memory_order_seq_cst) == 0); // ! No tasks must be running
        assert((epoch_.load(std::memory_order_seq_cst) & 1u) == 0);    // ! Last dispatch must be joined

        // Notify all worker threads...
        mood_.store(mood_t::die_k, std::memory_order_release);

        // ... and wait for them to finish
        thread_index_t const worker_threads = threads_count_ - use_caller_thread;
        for (thread_index_t i = 0; i != worker_threads; ++i)
            workers_[i + use_caller_thread].worker.join(); // ? Wait for the thread to finish

        // Prepare for future spawns. Joined threads are no longer joinable, so destroying the
        // cells here runs `~thread` on quiescent objects rather than calling `std::terminate`.
        threads_count_ = 0;
        workers_ = {};
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
    broadcast_join<basic_pool, invoke_for_n_dynamic<basic_pool, fork_type_, index_t>> //
        for_n_dynamic(index_t const n, fork_type_ &&fork) noexcept {

        return {*this, {*this, n, threads_count(), std::forward<fork_type_>(fork)}};
    }

#pragma endregion Indexed Task Scheduling

#pragma region ComputeDomains Compatibility

    /**
     *  @brief Number of individual sub-pool with the same NUMA-locality and QoS.
     *  @retval 1 constant for compatibility.
     */
    constexpr index_t compute_domains_count() const noexcept { return 1; }

    /**
     *  @brief Returns the number of threads in one NUMA-specific local @b compute_domain.
     *  @return Same value as `threads_count()`, as we only support one compute_domain.
     */
    thread_index_t threads_count(FU_MAYBE_UNUSED_ index_t compute_domain_index) const noexcept {
        assert(compute_domain_index == 0 && "Only one compute_domain is supported");
        return threads_count();
    }

    /**
     *  @brief Converts a @p `global_thread_index` to a local thread index within a @b compute_domain.
     *  @return Same value as `global_thread_index`, as we only support one compute_domain.
     */
    constexpr thread_index_t thread_local_index(thread_index_t global_thread_index,
                                                FU_MAYBE_UNUSED_ index_t compute_domain_index) const noexcept {
        assert(compute_domain_index == 0 && "Only one compute_domain is supported");
        return global_thread_index;
    }

#pragma endregion ComputeDomains Compatibility

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
        lambda_object(local_thread_t {thread_index, 0});
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
                call_yield_(micro_yield, thread_index);

            if (fu_unlikely_(mood == mood_t::die_k)) break;
            if (fu_unlikely_(mood == mood_t::chill_k) && (new_epoch == last_epoch)) {
                std::this_thread::sleep_for(std::chrono::microseconds(sleep_length_micros_));
                continue;
            }

            // Odd epochs are dispatches, even epochs are completions — skip even
            if (new_epoch & 1) {
                fork_trampoline_(fork_state_, thread_index);

                // ! The decrement must come after the task is executed. The `acq_rel`
                // ! ordering chains every contributor's writes into the last one, so the
                // ! completion increment below publishes all of them at once.
                thread_index_t const before_decrement = threads_to_sync_.fetch_sub(1, std::memory_order_acq_rel);
                assert(before_decrement > 0 && "We can't be here if there are no worker threads");

                // The last contributor to finish increments the epoch again, signaling completion
                if (before_decrement == 1) epoch_.fetch_add(1, std::memory_order_release);
            }
            last_epoch = new_epoch;
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
        { p.unsafe_for_threads(noop) } -> std::same_as<typename pool_type_::generation_t>;
    } && //
    requires(pool_type_ &p, typename pool_type_::generation_t generation) {
        { p.unsafe_join() } -> std::same_as<void>;
        { p.unsafe_join(generation) } -> std::same_as<void>;
        { p.is_complete(generation) } -> std::same_as<bool>;
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
 *  @brief Fetches the scheduler capacity of a CPU core, used to separate QoS classes.
 *  @retval A capacity value where larger means more performant, or 0 if unavailable.
 *
 *  Reads `/sys/devices/system/cpu/cpuN/cpu_capacity`, which the kernel populates from the
 *  Energy Model on ARM @b big.LITTLE/DynamIQ and from ITMT/Thread-Director on Intel @b hybrid
 *  chips (performance cores report ~1024, efficiency cores less). A return of 0 means the core
 *  is homogeneous or the kernel does not expose capacities - the whole node is then one class.
 */
FU_MAYBE_UNUSED_ static inline std::size_t get_capacity_for_core(FU_MAYBE_UNUSED_ numa_core_id_t core_id) noexcept {

    std::size_t capacity = 0;

#if defined(__linux__)
    char capacity_path[256];
    int path_result = std::snprintf(          //
        capacity_path, sizeof(capacity_path), //
        "/sys/devices/system/cpu/cpu%d/cpu_capacity", core_id);
    if (path_result < 0 || static_cast<std::size_t>(path_result) >= sizeof(capacity_path)) return 0; // ? Path too long

    FILE *capacity_file = ::fopen(capacity_path, "r");
    if (!capacity_file) return 0; // ? Homogeneous cores or unsupported kernel

    unsigned long long parsed = 0;
    if (::fscanf(capacity_file, "%llu", &parsed) == 1) capacity = static_cast<std::size_t>(parsed);
    ::fclose(capacity_file);
#endif

    return capacity;
}

/**
 *  @brief Whether @p node_id appears in a Linux range-list file such as "0", "0-3", or "0,2-4".
 *  @sa Used to map a NUMA node onto its kernel memory tier.
 */
FU_MAYBE_UNUSED_ static inline bool nodelist_contains(FU_MAYBE_UNUSED_ char const *path,
                                                      FU_MAYBE_UNUSED_ numa_node_id_t node_id) noexcept {
#if defined(__linux__)
    FILE *file = ::fopen(path, "r");
    if (!file) return false;

    char line[256];
    bool found = false;
    if (::fgets(line, sizeof(line), file)) {
        char const *cursor = line;
        while (*cursor) {
            char *next = nullptr;
            long const low = ::strtol(cursor, &next, 10);
            if (next == cursor) break; // ? No number left to parse
            long high = low;
            cursor = next;
            if (*cursor == '-') high = ::strtol(cursor + 1, &next, 10), cursor = next;
            if (node_id >= low && node_id <= high) {
                found = true;
                break;
            }
            while (*cursor == ',' || *cursor == ' ' || *cursor == '\n') ++cursor;
        }
    }
    ::fclose(file);
    return found;
#else
    return false;
#endif
}

/**
 *  @brief Fetches the kernel memory-tier ordinal for a NUMA node, used to separate memory levels.
 *  @retval A tier number where smaller means faster, or `numeric_limits<size_t>::max()` if unknown.
 *
 *  Scans `/sys/devices/virtual/memory_tiering/memory_tierN/nodelist`, which the kernel populates by
 *  abstract distance - HBM below DRAM, CXL and persistent memory above it. When the sysfs is absent
 *  (older kernels, no tiering) every node reports the sentinel and collapses to a single memory level.
 */
FU_MAYBE_UNUSED_ static inline std::size_t get_memory_tier_for_node(FU_MAYBE_UNUSED_ numa_node_id_t node_id) noexcept {
#if FU_ENABLE_NUMA
    DIR *dir = ::opendir("/sys/devices/virtual/memory_tiering");
    if (!dir) return std::numeric_limits<std::size_t>::max();

    std::size_t tier = std::numeric_limits<std::size_t>::max();
    for (dirent *entry; (entry = ::readdir(dir)) != nullptr;) {
        unsigned parsed = 0;
        if (::sscanf(entry->d_name, "memory_tier%u", &parsed) != 1) continue;
        char path[256];
        int const written = std::snprintf(path, sizeof(path), //
                                          "/sys/devices/virtual/memory_tiering/%s/nodelist", entry->d_name);
        if (written < 0 || static_cast<std::size_t>(written) >= sizeof(path)) continue; // ? Path too long
        if (nodelist_contains(path, node_id)) {
            tier = parsed;
            break;
        }
    }
    ::closedir(dir);
    return tier;
#else
    return std::numeric_limits<std::size_t>::max();
#endif
}

/**
 *  @brief Reads one HMAT performance metric for the initiator-to-target NUMA edge, or 0 if unknown.
 *  @param metric_name A sysfs leaf: "read_bandwidth", "write_bandwidth", "read_latency", "write_latency".
 *  @retval Bandwidth in MB/s or latency in nanoseconds, or 0 when the machine exposes no HMAT table.
 *
 *  The kernel publishes per-node HMAT numbers under `access0` (nearest initiators) and `access1` (all
 *  initiators); we take the first access class that lists @p initiator among its initiators.
 */
FU_MAYBE_UNUSED_ static inline std::size_t read_hmat_metric(FU_MAYBE_UNUSED_ numa_node_id_t initiator,
                                                            FU_MAYBE_UNUSED_ numa_node_id_t target,
                                                            FU_MAYBE_UNUSED_ char const *metric_name) noexcept {
#if defined(__linux__)
    for (int access_class = 0; access_class < 2; ++access_class) {
        char initiator_path[320];
        int written = std::snprintf(initiator_path, sizeof(initiator_path),
                                    "/sys/devices/system/node/node%d/access%d/initiators/node%d", target, access_class,
                                    initiator);
        if (written < 0 || static_cast<std::size_t>(written) >= sizeof(initiator_path)) continue;
        if (::access(initiator_path, F_OK) != 0) continue; // ? Initiator is not in this access class

        char metric_path[320];
        written =
            std::snprintf(metric_path, sizeof(metric_path), "/sys/devices/system/node/node%d/access%d/initiators/%s",
                          target, access_class, metric_name);
        if (written < 0 || static_cast<std::size_t>(written) >= sizeof(metric_path)) continue;

        FILE *file = ::fopen(metric_path, "r");
        if (!file) continue;
        unsigned long long parsed = 0;
        std::size_t value = 0;
        if (::fscanf(file, "%llu", &parsed) == 1) value = static_cast<std::size_t>(parsed);
        ::fclose(file);
        if (value) return value;
    }
    return 0;
#else
    return 0;
#endif
}

#if defined(__APPLE__)
/**
 *  @brief Reads an unsigned integer `sysctl` by name (e.g. "hw.nperflevels"), or 0 if unavailable.
 *  @sa Used to harvest the Apple Silicon performance-level topology.
 */
FU_MAYBE_UNUSED_ static inline std::size_t sysctl_uint(char const *name) noexcept {
    unsigned long long value = 0;
    std::size_t length = sizeof(value);
    if (::sysctlbyname(name, &value, &length, nullptr, 0) != 0) return 0;
    return static_cast<std::size_t>(value);
}
#endif

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

    /** @brief Unique NUMA node ID, in [0, numa_max_node()). */
    numa_node_id_t node_id {-1};
    /** @brief Physical CPU socket ID. */
    numa_socket_id_t socket_id {-1};
    /** @brief RAM volume in bytes. */
    std::size_t memory_size {0};
    /** @brief Memory tier ordinal, sorted fastest-to-slowest (0 = fastest). */
    std::size_t memory_level {0};
    /** @brief Pointer to the first core ID in the `core_ids` array. */
    numa_core_id_t const *first_core_id {nullptr};
    /** @brief Number of items in the `core_ids` array. */
    std::size_t core_count {0};
    /** @brief Huge page sizes available on this NUMA node. */
    ram_page_settings<max_page_sizes_k> page_sizes {};
};

using numa_node_t = numa_node<>;

/**
 *  @brief One bindable cluster of cores sharing a QoS class and locality - the compute axis.
 *  @sa `numa_node` is the memory axis; a `numa_topology` exposes both plus their affinity.
 *
 *  A compute domain is a contiguous run of same-capacity cores within a single NUMA node. It is
 *  what a pool binds to and the index a worker callback receives. Several compute domains may map
 *  to one memory domain (performance and efficiency cores sharing a memory controller), which is
 *  why compute and memory are separate axes rather than a single "colocation" cell.
 */
struct compute_domain {
    /** @brief The NUMA node these cores live on. */
    numa_node_id_t node_id {-1};
    /** @brief Index of the local memory domain (this node). */
    std::size_t memory_domain_index {0};
    /** @brief QoS ordinal, sorted least-to-most performant. */
    std::size_t compute_level {0};
    /**
     *  @brief Relative throughput of @b one core here; `capacity_unknown_k` when unavailable.
     *  @sa `compute_level` is a dense ordinal for grouping - never divide by it.
     *
     *  Unlike `compute_level`, this is a magnitude, so it may be summed and divided. Where the
     *  kernel publishes a per-core rating - such as the Linux scheduler's `cpu_capacity`, scaled so
     *  the fastest core present reads 1024 - it lands here. Platforms that rank cores without
     *  quantifying them leave this unknown, and callers weigh domains by `core_count` instead.
     */
    std::size_t capacity {0};
    /**
     *  @brief Bytes of the deepest cache private to this domain's cores; 0 when unknown.
     *
     *  Sizes a cache-resident chunk, which is a different question from how many chunks a domain
     *  should receive - cores of equal throughput may back onto very differently sized caches, so
     *  neither number can be derived from the other. Shared by every core in the domain.
     */
    std::size_t cache_bytes {0};
    /** @brief Pointer to the first core ID in this domain. */
    numa_core_id_t const *first_core_id {nullptr};
    /** @brief Number of cores in this domain. */
    std::size_t core_count {0};
};

/** @brief Sentinel for `compute_domain::capacity` when the platform exposes no per-core throughput. */
static constexpr std::size_t capacity_unknown_k = 0;

using compute_domain_t = compute_domain;

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
    using domains_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<compute_domain_t>;
    using capacities_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<std::size_t>;
    static constexpr std::size_t max_page_sizes_k = max_page_sizes_;

  private:
    allocator_t allocator_ {};
    numa_node_t *nodes_ {nullptr};                // ? Memory domains (one per NUMA node)
    numa_core_id_t *node_core_ids_ {nullptr};     // ? Core IDs in [0, threads_count), grouped by node then QoS
    compute_domain_t *compute_domains_ {nullptr}; // ? Compute domains (same-QoS core runs within a node)
    std::size_t nodes_count_ {0};                 // ? Number of memory domains / NUMA nodes
    std::size_t cores_count_ {0};                 // ? Total number of cores in all nodes
    std::size_t compute_domains_count_ {0};       // ? Number of compute domains
    std::size_t compute_levels_count_ {1};        // ? Number of distinct QoS classes (>= 1)
    std::size_t memory_levels_count_ {1};         // ? Number of distinct memory tiers (>= 1)

  public:
    constexpr numa_topology() noexcept = default;
    numa_topology(numa_topology &&o) noexcept
        : allocator_(std::move(o.allocator_)), nodes_(o.nodes_), node_core_ids_(o.node_core_ids_),
          compute_domains_(o.compute_domains_), nodes_count_(o.nodes_count_), cores_count_(o.cores_count_),
          compute_domains_count_(o.compute_domains_count_), compute_levels_count_(o.compute_levels_count_),
          memory_levels_count_(o.memory_levels_count_) {
        o.nodes_ = nullptr;
        o.node_core_ids_ = nullptr;
        o.compute_domains_ = nullptr;
        o.nodes_count_ = 0;
        o.cores_count_ = 0;
        o.compute_domains_count_ = 0;
        o.compute_levels_count_ = 1;
        o.memory_levels_count_ = 1;
    }

    numa_topology &operator=(numa_topology &&other) noexcept {
        if (this != &other) {
            reset(); // ? Reset the current state
            allocator_ = std::move(other.allocator_);
            nodes_ = std::exchange(other.nodes_, nullptr);
            node_core_ids_ = std::exchange(other.node_core_ids_, nullptr);
            compute_domains_ = std::exchange(other.compute_domains_, nullptr);
            nodes_count_ = std::exchange(other.nodes_count_, 0);
            cores_count_ = std::exchange(other.cores_count_, 0);
            compute_domains_count_ = std::exchange(other.compute_domains_count_, 0);
            compute_levels_count_ = std::exchange(other.compute_levels_count_, 1);
            memory_levels_count_ = std::exchange(other.memory_levels_count_, 1);
        }
        return *this;
    }

    numa_topology(numa_topology const &) = delete;
    numa_topology &operator=(numa_topology const &) = delete;

    ~numa_topology() noexcept { reset(); }

    void reset() noexcept {
        cores_allocator_t cores_alloc {allocator_};
        nodes_allocator_t nodes_alloc {allocator_};
        domains_allocator_t domains_alloc {allocator_};

        if (node_core_ids_) cores_alloc.deallocate(node_core_ids_, cores_count_);
        if (nodes_) nodes_alloc.deallocate(nodes_, nodes_count_);
        if (compute_domains_) domains_alloc.deallocate(compute_domains_, cores_count_);

        nodes_ = nullptr;
        node_core_ids_ = nullptr;
        compute_domains_ = nullptr;
        nodes_count_ = cores_count_ = compute_domains_count_ = 0;
        compute_levels_count_ = 1;
        memory_levels_count_ = 1;
    }

    /** @brief Number of memory domains (one per NUMA node). @sa `compute_domains_count`. */
    std::size_t nodes_count() const noexcept { return nodes_count_; }
    std::size_t memory_domains_count() const noexcept { return nodes_count_; }
    std::size_t threads_count() const noexcept { return cores_count_; }

    /** @brief The memory domain at @p node_index, in [0, `memory_domains_count()`). */
    numa_node_t const &node(std::size_t const node_index) const noexcept {
        assert(node_index < nodes_count_ && "Node ID is out of bounds");
        return nodes_[node_index];
    }
    numa_node_t const &memory_domain(std::size_t const memory_domain_index) const noexcept {
        return node(memory_domain_index);
    }

    /** @brief Number of compute domains (one per same-QoS core run within a node). */
    std::size_t compute_domains_count() const noexcept { return compute_domains_count_; }
    /** @brief Number of distinct QoS classes across all compute domains (>= 1). */
    std::size_t compute_levels_count() const noexcept { return compute_levels_count_; }
    /** @brief Number of distinct memory tiers across all memory domains (>= 1). */
    std::size_t memory_levels_count() const noexcept { return memory_levels_count_; }

    /** @brief The compute domain at @p compute_domain_index, in [0, `compute_domains_count()`). */
    compute_domain_t const &compute_domain_at(std::size_t const compute_domain_index) const noexcept {
        assert(compute_domain_index < compute_domains_count_ && "Compute domain ID is out of bounds");
        return compute_domains_[compute_domain_index];
    }

    /** @brief The memory domain nearest a compute domain (its NUMA node); 0 if out of range. */
    std::size_t local_memory_of(std::size_t const compute_domain_index) const noexcept {
        if (compute_domain_index >= compute_domains_count_) return 0;
        return compute_domains_[compute_domain_index].memory_domain_index;
    }

    /** @brief Relative access distance from a compute domain to a memory domain (10 = local). */
    std::size_t distance(std::size_t const compute_domain_index, std::size_t const memory_domain_index) const noexcept {
        if (compute_domain_index >= compute_domains_count_ || memory_domain_index >= nodes_count_) return 0;
#if FU_ENABLE_NUMA
        numa_node_id_t const from = compute_domains_[compute_domain_index].node_id;
        numa_node_id_t const to = nodes_[memory_domain_index].node_id;
        int const numa_dist = ::numa_distance(from, to);
        return numa_dist > 0 ? static_cast<std::size_t>(numa_dist) : (from == to ? 10u : 20u);
#else
        return compute_domains_[compute_domain_index].memory_domain_index == memory_domain_index ? 10u : 20u;
#endif
    }

    /** @brief HMAT read bandwidth (MB/s) from a compute domain to a memory domain, or 0 if unknown. */
    std::size_t memory_bandwidth(std::size_t const compute_domain_index,
                                 std::size_t const memory_domain_index) const noexcept {
        if (compute_domain_index >= compute_domains_count_ || memory_domain_index >= nodes_count_) return 0;
        numa_node_id_t const from = compute_domains_[compute_domain_index].node_id;
        numa_node_id_t const to = nodes_[memory_domain_index].node_id;
        return read_hmat_metric(from, to, "read_bandwidth");
    }

    /** @brief HMAT read latency (nanoseconds) from a compute domain to a memory domain, or 0 if unknown. */
    std::size_t memory_latency(std::size_t const compute_domain_index,
                               std::size_t const memory_domain_index) const noexcept {
        if (compute_domain_index >= compute_domains_count_ || memory_domain_index >= nodes_count_) return 0;
        numa_node_id_t const from = compute_domains_[compute_domain_index].node_id;
        numa_node_id_t const to = nodes_[memory_domain_index].node_id;
        return read_hmat_metric(from, to, "read_latency");
    }

    /**
     *  @brief Harvests CPU-memory topology - Linux NUMA nodes, or Apple Silicon performance levels.
     *  @retval false if the platform lacks topology support or the harvest failed.
     *  @retval true if the harvest was successful and the topology is ready to use.
     */
    bool try_harvest() noexcept {
#if FU_ENABLE_NUMA
        struct bitmask *numa_mask = nullptr;
        numa_node_t *nodes_ptr = nullptr;
        numa_core_id_t *core_ids_ptr = nullptr;
        compute_domain_t *domains_ptr = nullptr;
        std::size_t *core_capacities = nullptr; // ? Scheduler capacity keyed by core id, cached for the QoS split
        numa_node_id_t max_numa_node_id = -1;

        // Allocators must be visible to the cleanup path
        nodes_allocator_t nodes_alloc {allocator_};
        cores_allocator_t cores_alloc {allocator_};
        domains_allocator_t domains_alloc {allocator_};
        capacities_allocator_t capacities_alloc {allocator_};

        // These counters are reused in the failure handler
        std::size_t fetched_nodes = 0, fetched_cores = 0, configured_cores = 0;

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
            // A cpuless memory domain (HBM-flat, CXL expander, GPU HBM) reports zero cores, yet is a
            // valid memory domain - count the node and add its (possibly zero) cores.
            std::size_t const node_cores = static_cast<std::size_t>(::numa_bitmask_weight(numa_mask));
            fetched_nodes += 1;
            fetched_cores += node_cores;
        }
        if (fetched_nodes == 0) goto failed_harvest; // ! Zero nodes is not a valid state

        // Second pass - allocate. At most one compute domain per core (fully heterogeneous node).
        nodes_ptr = nodes_alloc.allocate(fetched_nodes);
        core_ids_ptr = cores_alloc.allocate(fetched_cores);
        domains_ptr = domains_alloc.allocate(fetched_cores);
        if (!nodes_ptr || !core_ids_ptr || !domains_ptr) goto failed_harvest; // ! Allocation failed

        // A scratch table of every configured CPU's capacity, filled once below and read by the
        // per-node QoS split (which is O(n^2) in comparisons) instead of re-opening sysfs each time.
        configured_cores = static_cast<std::size_t>(::numa_num_configured_cpus());
        if (configured_cores == 0) goto failed_harvest; // ! No CPUs is not a valid state
        core_capacities = capacities_alloc.allocate(configured_cores);
        if (!core_capacities) goto failed_harvest; // ! Allocation failed

        // Populate
        for (numa_node_id_t node_id = 0, core_index = 0, node_index = 0; node_id <= max_numa_node_id; ++node_id) {
            long long free_memory_size; // ? Only an out-parameter, the total size comes back as the return value
            long long const total_memory_size = ::numa_node_size64(node_id, &free_memory_size);
            if (total_memory_size < 0) continue;
            ::numa_bitmask_clearall(numa_mask);
            if (::numa_node_to_cpus(node_id, numa_mask) < 0) continue;

            numa_node_t &node = nodes_ptr[node_index];
            node.node_id = node_id;
            node.memory_size = static_cast<std::size_t>(total_memory_size);
            node.first_core_id = core_ids_ptr + core_index;
            node.core_count = static_cast<std::size_t>(::numa_bitmask_weight(numa_mask));

            // Most likely, this will fill `core_ids_ptr` with `std::iota`-like values
            for (std::size_t bit_offset = 0; bit_offset < numa_mask->size; ++bit_offset)
                if (::numa_bitmask_isbitset(numa_mask, static_cast<unsigned int>(bit_offset)))
                    core_ids_ptr[core_index++] = static_cast<numa_core_id_t>(bit_offset);

            // ? Cpuless memory domains have no core to query - default the socket and skip the lookup.
            // ! Only valid once `first_core_id` points to initialized entries, hence after the loop above
            node.socket_id = node.core_count > 0 ? get_socket_id_for_core(node.first_core_id[0]) : -1;

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
            if (a.core_count != b.core_count) return a.core_count > b.core_count;  // ? Sort by descending core count
            numa_core_id_t const a_first = a.core_count ? a.first_core_id[0] : -1; // ? Cpuless slices are empty
            numa_core_id_t const b_first = b.core_count ? b.first_core_id[0] : -1;
            return a_first < b_first; // ? Sort by first core ID
        });

        // Cache each harvested core's scheduler capacity once, keyed by core id. The QoS split
        // below sorts and compares capacities repeatedly, so reading sysfs per comparison would be
        // O(cores^2) file opens on a large node - a single pass here amortizes that to O(cores).
        for (std::size_t i = 0; i < configured_cores; ++i) core_capacities[i] = 0;
        for (std::size_t core_index = 0; core_index < cores_count_; ++core_index) {
            numa_core_id_t const core_id = node_core_ids_[core_index];
            if (static_cast<std::size_t>(core_id) < configured_cores)
                core_capacities[static_cast<std::size_t>(core_id)] = get_capacity_for_core(core_id);
        }

        // Split each memory domain's cores into compute domains by QoS class. We sort each node's
        // cores by scheduler capacity, then cut the sorted run at every capacity change. Cores
        // within a `numa_node` slice are mutable here - `first_core_id` still bounds the slice.
        {
            std::size_t domains_written = 0;
            for (std::size_t node_index = 0; node_index < nodes_count_; ++node_index) {
                numa_node_t &node = nodes_[node_index];
                numa_core_id_t *node_cores = const_cast<numa_core_id_t *>(node.first_core_id);

                // Ascending capacity groups efficiency cores before performance cores.
                bubble_sort(node_cores, node.core_count,
                            [core_capacities](numa_core_id_t const &a, numa_core_id_t const &b) noexcept {
                                return core_capacities[static_cast<std::size_t>(a)] <
                                       core_capacities[static_cast<std::size_t>(b)];
                            });

                std::size_t run_begin = 0;
                for (std::size_t core = 1; core <= node.core_count; ++core) {
                    bool const at_end = core == node.core_count;
                    bool const capacity_changed =
                        !at_end && core_capacities[static_cast<std::size_t>(node_cores[core])] !=
                                       core_capacities[static_cast<std::size_t>(node_cores[run_begin])];
                    if (!at_end && !capacity_changed) continue;

                    compute_domain_t &domain = domains_ptr[domains_written++];
                    domain.node_id = node.node_id;
                    domain.memory_domain_index = node_index;
                    domain.compute_level = core_capacities[static_cast<std::size_t>(
                        node_cores[run_begin])]; // ? Raw capacity, ranked below
                    // ! Keep the raw magnitude too - `compute_level` is about to collapse into an ordinal.
                    domain.capacity = core_capacities[static_cast<std::size_t>(node_cores[run_begin])];
                    domain.cache_bytes = 0; // ? Not yet read from `sys/devices/system/cpu/cpu*/cache`
                    domain.first_core_id = node_cores + run_begin;
                    domain.core_count = core - run_begin;
                    run_begin = core;
                }
            }

            // Re-rank the raw capacities into dense QoS ordinals, sorted least-to-most performant.
            compute_domains_ = domains_ptr;
            compute_domains_count_ = domains_written;
            compute_levels_count_ = dense_rank(
                domains_written,
                [core_capacities, domains_ptr](std::size_t index) noexcept {
                    return core_capacities[static_cast<std::size_t>(domains_ptr[index].first_core_id[0])];
                },
                [domains_ptr](std::size_t index, std::size_t rank) noexcept {
                    domains_ptr[index].compute_level = rank;
                });
        }

        // Rank memory domains into dense tier ordinals, sorted fastest-to-slowest (lower = faster). Raw
        // tiers are snapshotted into scratch so ranking in place never corrupts a repeated tier. Absent
        // the memory-tiering sysfs, every node collapses to a single memory level.
        if (std::size_t *raw_tiers = capacities_alloc.allocate(nodes_count_)) {
            for (std::size_t i = 0; i < nodes_count_; ++i) raw_tiers[i] = get_memory_tier_for_node(nodes_[i].node_id);
            memory_levels_count_ = dense_rank(
                nodes_count_, [raw_tiers](std::size_t index) noexcept { return raw_tiers[index]; },
                [this, raw_tiers](std::size_t index, std::size_t rank) noexcept { nodes_[index].memory_level = rank; });
            capacities_alloc.deallocate(raw_tiers, nodes_count_);
        }

        capacities_alloc.deallocate(core_capacities, configured_cores); // ? Scratch, not part of the committed state
        return true;

    failed_harvest:
        if (nodes_ptr) nodes_alloc.deallocate(nodes_ptr, fetched_nodes);
        if (core_ids_ptr) cores_alloc.deallocate(core_ids_ptr, fetched_cores);
        if (domains_ptr) domains_alloc.deallocate(domains_ptr, fetched_cores);
        if (core_capacities) capacities_alloc.deallocate(core_capacities, configured_cores);
        if (numa_mask) ::numa_free_cpumask(numa_mask);
#endif // FU_ENABLE_NUMA
#if defined(__APPLE__)
        return try_harvest_apple();
#else
        return false;
#endif
    }

#if defined(__APPLE__)
    /**
     *  @brief Harvests the Apple Silicon topology from `sysctl` performance levels.
     *  @retval false if the machine reports no logical CPUs or an allocation failed.
     *
     *  Apple Silicon is one UMA memory domain shared by every core, so we build a single memory
     *  domain. The compute axis is cut twice: first by `hw.perflevelN`, then by `cpusperl2` within
     *  each level, because a performance level may span several L2 clusters that share no cache.
     *  A compute domain is cores sharing a QoS class @b and locality, so the cluster is the unit.
     *  Domains from one level all carry that level's `compute_level`; `hw.perflevel0` ranks highest.
     *
     *  Performance levels rank cores without rating them, and the parts they distinguish need not
     *  differ in clock - some pair equal scalar throughput with unequal cache. So `capacity` stays
     *  `capacity_unknown_k` and only `cache_bytes` is populated; weighting work by the level ordinal
     *  would hand a wide-cache cluster more tasks than it can necessarily retire any faster.
     *
     *  @note macOS exposes no hard pinning - `thread_policy_set(THREAD_AFFINITY_POLICY)` returns
     *        `KERN_NOT_SUPPORTED` on arm64, and QoS classes are the only placement lever. These
     *        domains are therefore descriptive: `spawn_on` reports them, the kernel still migrates.
     */
    bool try_harvest_apple() noexcept {
        std::size_t const total_cores = sysctl_uint("hw.logicalcpu");
        if (total_cores == 0) return false;
        std::size_t const memory_size = sysctl_uint("hw.memsize");
        std::size_t const levels = sysctl_uint("hw.nperflevels");

        // Count the non-empty performance levels so each becomes one compute level.
        std::size_t nonempty_levels = 0;
        for (std::size_t level = 0; level < levels; ++level) {
            char name[64];
            std::snprintf(name, sizeof(name), "hw.perflevel%zu.logicalcpu", level);
            if (sysctl_uint(name)) nonempty_levels += 1;
        }
        if (nonempty_levels == 0) nonempty_levels = 1; // ? One level covering every core

        nodes_allocator_t nodes_alloc {allocator_};
        cores_allocator_t cores_alloc {allocator_};
        domains_allocator_t domains_alloc {allocator_};

        // `compute_domains_` is sized to `cores_count_` across the class (at most one domain per core).
        numa_node_t *nodes_ptr = nodes_alloc.allocate(1);
        numa_core_id_t *core_ids_ptr = cores_alloc.allocate(total_cores);
        compute_domain_t *domains_ptr = domains_alloc.allocate(total_cores);
        if (!nodes_ptr || !core_ids_ptr || !domains_ptr) {
            if (nodes_ptr) nodes_alloc.deallocate(nodes_ptr, 1);
            if (core_ids_ptr) cores_alloc.deallocate(core_ids_ptr, total_cores);
            if (domains_ptr) domains_alloc.deallocate(domains_ptr, total_cores);
            return false;
        }
        for (std::size_t i = 0; i < total_cores; ++i) core_ids_ptr[i] = static_cast<numa_core_id_t>(i);

        numa_node_t &node = nodes_ptr[0];
        node.node_id = 0;
        node.socket_id = 0;
        node.memory_size = memory_size;
        node.memory_level = 0;
        node.first_core_id = core_ids_ptr;
        node.core_count = total_cores;

        // One compute domain per L2 cluster. Apple lists cores fastest-level first, so `hw.perflevel0`
        // takes the highest compute level; every cluster carved out of it repeats that same level.
        std::size_t core_offset = 0, domains_written = 0, levels_written = 0;
        for (std::size_t level = 0; level < levels && core_offset < total_cores; ++level) {
            char name[64];
            std::snprintf(name, sizeof(name), "hw.perflevel%zu.logicalcpu", level);
            std::size_t level_cores = sysctl_uint(name);
            if (level_cores == 0) continue;
            if (core_offset + level_cores > total_cores) level_cores = total_cores - core_offset;

            std::snprintf(name, sizeof(name), "hw.perflevel%zu.l2cachesize", level);
            std::size_t const level_cache_bytes = sysctl_uint(name);
            std::snprintf(name, sizeof(name), "hw.perflevel%zu.cpusperl2", level);
            std::size_t cores_per_cluster = sysctl_uint(name);
            // ? A level with no `cpusperl2` is one undivided cluster, not zero-sized ones.
            if (cores_per_cluster == 0 || cores_per_cluster > level_cores) cores_per_cluster = level_cores;

            std::size_t const level_rank = nonempty_levels - 1 - levels_written;
            for (std::size_t cut = 0; cut < level_cores; cut += cores_per_cluster) {
                std::size_t const cluster_cores = (level_cores - cut) < cores_per_cluster //
                                                      ? (level_cores - cut)
                                                      : cores_per_cluster;
                compute_domain_t &domain = domains_ptr[domains_written];
                domain.node_id = 0;
                domain.memory_domain_index = 0;
                domain.compute_level = level_rank; // ? Sibling clusters share their level's rank
                domain.capacity = capacity_unknown_k;
                domain.cache_bytes = level_cache_bytes; // ? L2 is private to the cluster, shared within it
                domain.first_core_id = core_ids_ptr + core_offset + cut;
                domain.core_count = cluster_cores;
                domains_written += 1;
            }
            core_offset += level_cores;
            levels_written += 1;
        }

        // Fallback: no perflevel data - one compute domain over every core.
        if (domains_written == 0) {
            compute_domain_t &domain = domains_ptr[0];
            domain.node_id = 0;
            domain.memory_domain_index = 0;
            domain.compute_level = 0;
            domain.capacity = capacity_unknown_k;
            domain.cache_bytes = 0;
            domain.first_core_id = core_ids_ptr;
            domain.core_count = total_cores;
            domains_written = 1;
            levels_written = 1;
        }

        reset(); // ? Free any prior state before committing
        nodes_ = nodes_ptr;
        node_core_ids_ = core_ids_ptr;
        compute_domains_ = domains_ptr;
        nodes_count_ = 1;
        cores_count_ = total_cores;
        compute_domains_count_ = domains_written;
        compute_levels_count_ = levels_written; // ! Several clusters may share one level - not `domains_written`
        memory_levels_count_ = 1;
        return true;
    }
#endif // defined(__APPLE__)

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
        domains_allocator_t domains_alloc {allocator_};

        numa_node_t *scratch_nodes = nullptr;
        numa_core_id_t *scratch_core_ids = nullptr;
        compute_domain_t *scratch_domains = nullptr;
        if (other.nodes_count_) {
            scratch_nodes = nodes_alloc.allocate(other.nodes_count_);
            if (!scratch_nodes) return false; // ! OOM
        }
        if (other.cores_count_) {
            scratch_core_ids = cores_alloc.allocate(other.cores_count_);
            scratch_domains = domains_alloc.allocate(other.cores_count_);
            if (!scratch_core_ids || !scratch_domains) {
                if (scratch_nodes) nodes_alloc.deallocate(scratch_nodes, other.nodes_count_);
                if (scratch_core_ids) cores_alloc.deallocate(scratch_core_ids, other.cores_count_);
                if (scratch_domains) domains_alloc.deallocate(scratch_domains, other.cores_count_);
                return false; // ! OOM
            }
        }

        // Deep copy, re-basing every `first_core_id` into our own core-id block
        if (other.cores_count_ > 0)
            std::memcpy(scratch_core_ids, other.node_core_ids_, other.cores_count_ * sizeof(numa_core_id_t));
        for (std::size_t i = 0; i < other.nodes_count_; ++i) {
            scratch_nodes[i] = other.nodes_[i];
            std::ptrdiff_t const offset = other.nodes_[i].first_core_id - other.node_core_ids_;
            scratch_nodes[i].first_core_id = scratch_core_ids + offset;
        }
        for (std::size_t i = 0; i < other.compute_domains_count_; ++i) {
            scratch_domains[i] = other.compute_domains_[i];
            std::ptrdiff_t const offset = other.compute_domains_[i].first_core_id - other.node_core_ids_;
            scratch_domains[i].first_core_id = scratch_core_ids + offset;
        }

        reset(); // ? Free old buffers
        nodes_ = scratch_nodes;
        node_core_ids_ = scratch_core_ids;
        compute_domains_ = scratch_domains;
        nodes_count_ = other.nodes_count_;
        cores_count_ = other.cores_count_;
        compute_domains_count_ = other.compute_domains_count_;
        compute_levels_count_ = other.compute_levels_count_;
        memory_levels_count_ = other.memory_levels_count_;
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

        // ! Unlike `allocate_at_least`, we can't round the request up to the page boundary here:
        // ! the matching `deallocate(p, n)` only knows `n`, so it would unmap less than we mapped.
        // ! Huge Pages are therefore only an option for exact multiples of their size.

        // Try 1 GB Huge Pages, for buffers larger than 2 GB
        if (size_bytes >= (2u * page_size_1g_k) && size_bytes % page_size_1g_k == 0)
            if (auto result = allocate(size, page_size_1g_k); result) return result;

        // Try 2 MB Huge Pages, for buffers larger than 4 MB
        if (size_bytes >= (2u * page_size_2m_k) && size_bytes % page_size_2m_k == 0)
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
 *  @brief Used inside `linux_compute_domain_pool` to describe a pinned thread.
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
    /**
     *  @brief This thread's private cursor for `for_n_dynamic`. @sa `dynamic_claim`.
     *  @note Lives here, rather than in a second array, so the pool allocates once and the cursor
     *        inherits both this record's cache-line padding and its NUMA node.
     */
    dynamic_claim<std::size_t> claim {};
};

#pragma region - Linux ComputeDomain Pool

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
 *
 *  The synchronization protocol - epochs, generations, contributor counting, and the memory
 *  ordering rules - is identical to `basic_pool`; @sa @ref pool_concurrency_model.
 */
template <typename micro_yield_type_ = standard_yield_t, std::size_t alignment_ = default_alignment_k>
struct linux_compute_domain_pool {

  public:
    using allocator_t = linux_numa_allocator_t;
    using micro_yield_t = micro_yield_type_;
    static constexpr std::size_t alignment_k = alignment_;
    static_assert(alignment_k > 0 && (alignment_k & (alignment_k - 1)) == 0, "Alignment must be a power of 2");

    using index_t = std::size_t;
    static_assert(std::is_unsigned<index_t>::value, "Index type must be an unsigned integer");
    using epoch_index_t = index_t;      // ? A.k.a. number of previous API calls in [0, UINT_MAX)
    using generation_t = epoch_index_t; // ? A.k.a. token returned from `unsafe_for_threads`
    using thread_index_t = index_t;     // ? A.k.a. "core index" or "thread ID" in [0, threads_count)
    using local_thread_t = local_thread<thread_index_t>;
    using prong_t = local_prong<index_t>;

    using punned_fork_context_t = void *;                                 // ? Pointer to the on-stack lambda
    using trampoline_t = void (*)(punned_fork_context_t, local_thread_t); // ? Wraps lambda's `operator()`

    using micro_yield_traits_t = yield_traits<micro_yield_t, thread_index_t>;
    static_assert(micro_yield_traits_t::valid, "Yield must be invocable w/out args or with a thread index");

  private:
    using allocator_traits_t = std::allocator_traits<allocator_t>;
    using numa_pthread_allocator_t = typename allocator_traits_t::template rebind_alloc<numa_pthread_t>;
    using claim_t = dynamic_claim<index_t>; // ? Lives inside each `numa_pthread_t`, so no extra array

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
    index_t compute_domain_index_ {0}; // ? Unique {NUMA node + QoS level} compute_domain ID, defined externally
    numa_pin_granularity_t pin_granularity_ {numa_pin_to_core_k};

    alignas(alignment_k) std::atomic<mood_t> mood_ {mood_t::grind_k};

    // Task-specific variables:
    punned_fork_context_t fork_state_ {nullptr}; // ? Pointer to the users lambda
    trampoline_t fork_trampoline_ {nullptr};     // ? Calls the lambda
    alignas(alignment_k) std::atomic<thread_index_t> threads_to_sync_ {0};
    alignas(alignment_k) std::atomic<epoch_index_t> epoch_ {0};

    // ! Still the single cursor `invoke_distributed_for_n_dynamic` drains across compute domains;
    // ! this pool's own `for_n_dynamic` uses the per-thread `dynamic_claims_` below instead.
    alignas(alignment_k) std::atomic<index_t> dynamic_progress_ {0};

  public:
    linux_compute_domain_pool(linux_compute_domain_pool &&) = delete;
    linux_compute_domain_pool(linux_compute_domain_pool const &) = delete;
    linux_compute_domain_pool &operator=(linux_compute_domain_pool &&) = delete;
    linux_compute_domain_pool &operator=(linux_compute_domain_pool const &) = delete;

    explicit linux_compute_domain_pool(char const *name = "forkunion") noexcept {
        // Accept NULL or empty names by falling back to a sensible default
        char const *effective_name = (name && name[0] != '\0') ? name : "forkunion";
        std::strncpy(name_, effective_name, sizeof(name_) - 1);
        name_[sizeof(name_) - 1] = '\0';
    }

    ~linux_compute_domain_pool() noexcept { terminate(); }

    /**
     *  @brief Estimates the amount of memory managed by this pool handle and internal structures.
     *  @note This API is @b not synchronized.
     */
    std::size_t memory_usage() const noexcept {
        return sizeof(linux_compute_domain_pool) + threads_count() * sizeof(numa_pthread_t);
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
     *  @brief Returns the compute_domain index of this thread-pool.
     *  @retval 0 if the thread-pool is not initialized or the compute_domain index is unknown.
     *  @note This API is @b not synchronized.
     */
    index_t compute_domain_index() const noexcept { return compute_domain_index_; }

    /**
     *  @brief Returns the first thread index in the thread-pool.
     *  @retval 0 in most cases, when the last argument to `try_spawn` is not specified.
     *  @note This API is @b not synchronized.
     */
    thread_index_t first_thread() const noexcept { return first_thread_; }

    /** @brief Exposes the cross-domain cursor drained by `invoke_distributed_for_n_dynamic`. */
    std::atomic<index_t> &unsafe_dynamic_progress_ref() noexcept { return dynamic_progress_; }

    /** @brief Exposes a thread's private claim cursor, kept inside its `numa_pthread_t`. */
    claim_t &unsafe_dynamic_claim_ref(thread_index_t const thread) noexcept { return pthreads_[thread].claim; }

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
    bool try_spawn(compute_domain_t const &domain,
                   caller_exclusivity_t const exclusivity = caller_inclusive_k) noexcept {
        return try_spawn(domain, domain.core_count, exclusivity);
    }

    /**
     *  @brief Creates a thread-pool with the given number of @p threads on the given NUMA @p node.
     *  @param[in] node Describes the NUMA node to use, with its ID, memory size, and core IDs.
     *  @param[in] threads The number of threads to be used.
     *  @param[in] exclusivity Should we count the calling thread as one of the threads?
     *  @param[in] pin_granularity How to pin the threads to the NUMA node?
     *  @param[in] first_thread The index of the first thread to start from, defaults to 0.
     *  @param[in] compute_domain_index A unique index for the {NUMA node + QoS level} compute_domain.
     *  @retval false if the number of threads is zero or if spawning has failed.
     *  @retval true if the thread-pool was created successfully, started, and is ready to use.
     *  @note This is the de-facto @b constructor - you only call it again after `terminate`.
     *
     *  @section Over- and Under-subscribing Cores and Pinning
     *
     *  We may accept @p threads different from the @p domain.core_count, which allows us to:
     *  - over-subscribe the cores, i.e. use more threads than cores available on the NUMA node.
     *  - under-subscribe the cores, i.e. use fewer threads than cores available on the NUMA node.
     *
     *  If you only have one thread-pool active at any part of your application, that's meaningless.
     *  You'd be better off using exactly the number of cores available on the NUMA node and pinning
     *  them to individual cores with @b `numa_pin_to_core_k` granularity.
     */
    bool try_spawn(compute_domain_t const &domain, thread_index_t const threads,
                   caller_exclusivity_t const exclusivity = caller_inclusive_k,
                   numa_pin_granularity_t const pin_granularity = numa_pin_to_core_k,
                   thread_index_t const first_thread = 0, index_t const compute_domain_index = 0) noexcept {

        if (threads == 0) return false;          // ! Can't have zero threads working on something
        if (pthreads_.size() != 0) return false; // ! Already initialized

        // Allocate the thread pool of `numa_pthread_t` objects
        allocator_ = linux_numa_allocator_t {domain.node_id};
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
        compute_domain_index_ = compute_domain_index;
        exclusivity_ = exclusivity;
        numa_node_id_ = domain.node_id;
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
            fill_thread_name(                                      //
                name, name_,                                       //
                static_cast<std::size_t>(domain.first_core_id[i]), //
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
                numa_core_id_t cpu = domain.first_core_id[i % domain.core_count];
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
            for (std::size_t i = 0; i < domain.core_count; ++i) {
                numa_core_id_t cpu = domain.first_core_id[i];
                assert(cpu >= 0 && "Invalid CPU core ID");
                CPU_SET_S(cpu, cpu_set_size, cpu_set_ptr);
            }
            assert(static_cast<std::size_t>(CPU_COUNT_S(cpu_set_size, cpu_set_ptr)) == domain.core_count &&
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
    broadcast_join<linux_compute_domain_pool, fork_type_> for_threads(fork_type_ &&fork) noexcept {
        return {*this, std::forward<fork_type_>(fork)};
    }

    /**
     *  @brief Executes a @p fork function in parallel on all threads, not waiting for the result.
     *  @param[in] fork The callback @b reference, receiving the thread index as an argument.
     *  @return A `generation_t` token identifying this dispatch.
     *  @sa Use in conjunction with `unsafe_join`.
     */
    template <typename fork_type_>
    FU_REQUIRES_((can_be_for_thread_callback<fork_type_, index_t>()))
    generation_t unsafe_for_threads(fork_type_ &fork) noexcept {

        thread_index_t const threads = threads_count();
        assert(threads != 0 && "Thread pool not initialized");
        caller_exclusivity_t const exclusivity = caller_exclusivity();
        bool const use_caller_thread = exclusivity == caller_inclusive_k;

        // Only one dispatch can be in flight, and it must be fully joined - the caller's
        // slice included - before the next one starts.
        assert(threads_to_sync_.load(std::memory_order_acquire) == 0 &&
               "The broadcast function can't be called concurrently or recursively");
        assert((epoch_.load(std::memory_order_relaxed) & 1u) == 0 && "Previous dispatch not joined");

        // Configure "fork" details
        fork_state_ = std::addressof(fork);
        fork_trampoline_ = &_call_as_lambda<fork_type_>;

        // Every contributor gets counted: all worker threads, plus the calling thread itself
        // on `caller_inclusive_k` pools, where its slice runs inside `unsafe_join`.
        threads_to_sync_.store(threads, std::memory_order_relaxed);

        // We are most likely already "grinding", but in the unlikely case we are not,
        // let's wake up from the "chilling" state with relaxed semantics. Assuming the sleeping
        // logic for the workers also checks the epoch counter, no synchronization is needed and
        // no immediate wake-up is required.
        mood_t may_be_chilling = mood_t::chill_k;
        bool const was_chilling = mood_.compare_exchange_weak( //
            may_be_chilling, mood_t::grind_k,                  //
            std::memory_order_relaxed, std::memory_order_relaxed);
        generation_t const generation = static_cast<generation_t>(epoch_.fetch_add(1, std::memory_order_release) + 1);

        // If the workers were indeed "chilling", we can inform the scheduler to wake them up.
        if (was_chilling) {
            for (std::size_t i = use_caller_thread; i < pthreads_.size(); ++i) {
                pid_t const pthread_id = pthreads_[i].id.load(std::memory_order_acquire);
                if (pthread_id < 0) continue; // ? Not set yet
                sched_param param {};
                ::sched_setscheduler(pthread_id, SCHED_FIFO | SCHED_RR, &param);
            }
        }
        return generation;
    }

    /**
     *  @brief Returns true if the generation identified by @p generation has completed.
     *  @note A `true` result synchronizes with all contributors: their writes are visible.
     *
     *  On `caller_inclusive_k` pools this can only turn `true` once `unsafe_join`
     *  contributes the calling thread's slice, so the poll-then-join pattern is
     *  reserved for `caller_exclusive_k` pools.
     */
    bool is_complete(generation_t generation) const noexcept {
        return generation != epoch_.load(std::memory_order_acquire);
    }

    /**
     *  @brief Blocks the calling thread until the generation identified by @p generation finishes.
     *  @note On `caller_inclusive_k` pools, first executes the calling thread's slice.
     *  Idempotent: returns immediately for already-joined or stale generations.
     */
    void unsafe_join(generation_t generation) noexcept {
        assert((generation & 1u) == 1 && "Generation tokens are always odd");
        if (epoch_.load(std::memory_order_acquire) != generation) return; // ? Stale or already complete

        // On inclusive pools the calling thread is a contributor: execute its slice
        // and count it down exactly like a worker thread would.
        bool const use_caller_thread = caller_exclusivity() == caller_inclusive_k;
        if (use_caller_thread) {
            fork_trampoline_(fork_state_, local_thread_t {static_cast<thread_index_t>(0), compute_domain_index_});
            thread_index_t const before_decrement = threads_to_sync_.fetch_sub(1, std::memory_order_acq_rel);
            assert(before_decrement > 0 && "The contributor count must include the caller");

            // The last contributor to finish increments the epoch, signaling completion
            if (before_decrement == 1) epoch_.fetch_add(1, std::memory_order_release);
        }

        // Wait for the last contributor's completion increment
        micro_yield_t micro_yield;
        while (epoch_.load(std::memory_order_acquire) == generation)
            call_yield_(micro_yield, static_cast<thread_index_t>(0));
    }

    /** @brief Blocks the calling thread until the currently broadcasted task finishes. */
    void unsafe_join() noexcept {
        epoch_index_t const current_epoch = epoch_.load(std::memory_order_acquire);
        if (current_epoch & 1u) unsafe_join(static_cast<generation_t>(current_epoch)); // ? Even means idle
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
        assert((epoch_.load(std::memory_order_seq_cst) & 1u) == 0);    // ! Last dispatch must be joined
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

        // Deallocate the handles, IDs, and the claim cursors they carry
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
    broadcast_join<linux_compute_domain_pool, invoke_for_slices<fork_type_, index_t>> //
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
    broadcast_join<linux_compute_domain_pool, invoke_for_n<fork_type_, index_t>> //
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
    broadcast_join<linux_compute_domain_pool, invoke_for_n_dynamic<linux_compute_domain_pool, fork_type_, index_t>> //
        for_n_dynamic(index_t const n, fork_type_ &&fork) noexcept {

        return {*this, {*this, n, threads_count(), std::forward<fork_type_>(fork)}};
    }

#pragma endregion Indexed Task Scheduling

#pragma region ComputeDomains Compatibility

    /**
     *  @brief Number of individual sub-pool with the same NUMA-locality and QoS.
     *  @retval 1 constant for compatibility.
     */
    constexpr index_t compute_domains_count() const noexcept { return 1; }

    /**
     *  @brief Returns the number of threads in one NUMA-specific local @b compute_domain.
     *  @retval Same value as `threads_count()`, as we only support one compute_domain.
     */
    thread_index_t threads_count(FU_MAYBE_UNUSED_ index_t compute_domain_index) const noexcept {
        return threads_count();
    }

    /**
     *  @brief Converts a @p `global_thread_index` to a local thread index within a @b compute_domain.
     *  @retval Same value as @p `global_thread_index`, as we only support one compute_domain.
     */
    constexpr thread_index_t thread_local_index(thread_index_t global_thread_index,
                                                FU_MAYBE_UNUSED_ index_t compute_domain_index = 0) const noexcept {
        return global_thread_index;
    }

#pragma endregion ComputeDomains Compatibility

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
    static void _call_as_lambda(punned_fork_context_t punned_lambda_pointer, local_thread_t local_thread) noexcept {
        fork_type_ &lambda_object = *static_cast<fork_type_ *>(punned_lambda_pointer);
        lambda_object(local_thread);
    }

    static void *_posix_worker_loop(void *arg) noexcept {
        linux_compute_domain_pool *pool = static_cast<linux_compute_domain_pool *>(arg);

        // Following section untile the main `while` loop may introduce race conditions,
        // so spin-loop for a bit until the pool is ready.
        mood_t mood;
        micro_yield_t micro_yield;
        while ((mood = pool->mood_.load(std::memory_order_acquire)) == mood_t::chill_k)
            // Technically, we are not on the zero thread index, but we don't know our index yet.
            call_yield_(micro_yield, static_cast<thread_index_t>(0));

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
                call_yield_(micro_yield, global_thread_index);

            if (fu_unlikely_(mood == mood_t::die_k)) break;
            if (fu_unlikely_(mood == mood_t::chill_k) && (new_epoch == last_epoch)) {
                struct timespec ts {0, static_cast<long>(pool->sleep_length_micros_ * 1000)};
                ::clock_nanosleep(CLOCK_MONOTONIC, 0, &ts, nullptr);
                continue;
            }

            // Odd epochs are dispatches, even epochs are completions — skip even
            if (new_epoch & 1) {
                pool->fork_trampoline_(pool->fork_state_,
                                       local_thread_t {global_thread_index, pool->compute_domain_index_});

                // ! The decrement must come after the task is executed. The `acq_rel`
                // ! ordering chains every contributor's writes into the last one, so the
                // ! completion increment below publishes all of them at once.
                thread_index_t const before_decrement = pool->threads_to_sync_.fetch_sub(1, std::memory_order_acq_rel);
                assert(before_decrement > 0 && "We can't be here if there are no worker threads");

                // The last contributor to finish increments the epoch again, signaling completion
                if (before_decrement == 1) pool->epoch_.fetch_add(1, std::memory_order_release);
            }
            last_epoch = new_epoch;
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

#pragma endregion - Linux ComputeDomain Pool

#pragma region - Linux Pool

/**
 *  @brief Wraps the metadata needed for `for_slices` APIs for `broadcast_join` compatibility.
 *  @note Similar to `invoke_for_slices`, but dynamically determines the threads' compute_domain.
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
        index_type_ const compute_domain = pool_.thread_compute_domain(thread);
        fork_(local_prong<index_type_> {range.first, thread, compute_domain}, range.count);
    }
};

/**
 *  @brief Wraps the metadata needed for `for_n` APIs for `broadcast_join` compatibility.
 *  @note Similar to `invoke_for_n`, but dynamically determines the threads' compute_domain.
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
        index_type_ const compute_domain = pool_.thread_compute_domain(thread);
        for (index_type_ i = 0; i < range.count; ++i)
            fork_(local_prong<index_type_> {static_cast<index_type_>(range.first + i), thread, compute_domain});
    }
};

/**
 *  @brief Wraps the metadata needed for `for_n_dynamic` APIs for `broadcast_join` compatibility.
 *  @note Similar to `invoke_for_n_dynamic`, but dynamically determines the threads' compute_domain.
 *
 *  @section Scheduling Logic
 *
 *  Assuming the latency of accessing an atomic variable on a remote NUMA node is high, this "invoker"
 *  performs work-stealing in a different way. Let's say we receive N tasks and we have T threads
 *  across C compute_domains. Each compute_domain takes (N/C) tasks and splits them between (T/C) threads.
 *  Once threads in one pool saturate their local (N/C) tasks, they start looping through other
 *  compute_domains and stealing tasks from them, until all tasks are completed.
 *
 *  The hardest decision there is to how to chose the next "non-native" compute_domain to steal from.
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

        // Reset the local progress to zero in each compute_domain
        index_type_ const compute_domains_count = pool_.compute_domains_count();
        for (index_type_ i = 0; i < compute_domains_count; ++i)
            pool_.unsafe_dynamic_progress_ref(i).store(0, std::memory_order_release);
    }

    void operator()(index_type_ const thread) noexcept {
        index_type_ const compute_domains_count = pool_.compute_domains_count();
        assert(compute_domains_count > 0 && "There must be at least one compute_domain");

        // In each compute_domains part, take one static prong per thread, if present.
        indexed_split<index_type_> split_between_compute_domains(n_, compute_domains_count);
        index_type_ const native_compute_domain = pool_.thread_compute_domain(thread);
        {
            index_type_ const threads_local = pool_.threads_count(native_compute_domain);
            indexed_range<index_type_> const range_local = split_between_compute_domains[native_compute_domain];
            index_type_ const n_local = range_local.count;
            index_type_ const n_local_dynamic = n_local > threads_local ? n_local - threads_local : 0;

            // Run (up to) one static prong on the current thread
            index_type_ const thread_local_index = pool_.thread_local_index(thread, native_compute_domain);
            index_type_ const one_static_prong_index = static_cast<index_type_>(n_local_dynamic + thread_local_index);
            local_prong<index_type_> prong( //
                static_cast<index_type_>(range_local.first + one_static_prong_index), thread, native_compute_domain);
            if (one_static_prong_index < n_local) fork_(prong);
        }

        coprime_permutation_range<index_type_> probing_strategy(0, compute_domains_count, thread);
        auto probe_iterator = probing_strategy.begin();

        // Next we will probe every compute_domain:
        index_type_ compute_domains_remaining = compute_domains_count;
        index_type_ current_compute_domain = native_compute_domain;
        while (compute_domains_remaining) {
            index_type_ const threads_local = pool_.threads_count(current_compute_domain);
            std::atomic<index_type_> &local_progress = pool_.unsafe_dynamic_progress_ref(current_compute_domain);
            indexed_range<index_type_> const range_local = split_between_compute_domains[current_compute_domain];
            index_type_ const n_local = range_local.count;
            index_type_ const n_local_dynamic = n_local > threads_local ? n_local - threads_local : 0;

            // Same loop as in `invoke_for_n_dynamic::operator()`
            while (true) {
                index_type_ prong_local_offset = local_progress.fetch_add(1, std::memory_order_relaxed);
                bool const beyond_last_prong = prong_local_offset >= n_local_dynamic;
                if (beyond_last_prong) break;
                local_prong<index_type_> prong(range_local.first + prong_local_offset, thread, current_compute_domain);
                fork_(prong);
            }

            // Now pick some other compute_domain to probe.
            compute_domains_remaining--;
            if (compute_domains_remaining) {
                do { ++probe_iterator; } while (*probe_iterator == native_compute_domain); // At most 2 iterations
                current_compute_domain = *probe_iterator;
            }
        }
    }
};

/**
 *  @brief A Linux-only pool over all distributed "thread compute_domains", NUMA nodes, and QoS levels.
 *
 *  Differs from the `basic_pool` template in the following ways:
 *  - constructor API: receives the NUMA nodes topology, & a name for threads.
 *  - implementation of `try_spawn`: redirects to individual `linux_compute_domain_pool` instances.
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

    using linux_compute_domain_pool_t = linux_compute_domain_pool<micro_yield_type_, alignment_>;
    using numa_topology_t = numa_topology<>;

    using allocator_t = linux_numa_allocator_t;
    using micro_yield_t = typename linux_compute_domain_pool_t::micro_yield_t;
    using index_t = typename linux_compute_domain_pool_t::index_t;
    using epoch_index_t = typename linux_compute_domain_pool_t::epoch_index_t;
    using generation_t = epoch_index_t;
    using thread_index_t = typename linux_compute_domain_pool_t::thread_index_t;
    static constexpr std::size_t alignment_k = linux_compute_domain_pool_t::alignment_k;
    using prong_t = local_prong<index_t>;

  private:
    numa_topology_t topology_ {};
    char name_[16] {}; // ? Thread name buffer, for POSIX thread naming
    thread_index_t threads_count_ {0};
    caller_exclusivity_t exclusivity_ {caller_inclusive_k}; // ? Whether the caller thread is included in the count

    struct compute_domain_cell_t {
        alignas(alignment_k) linux_compute_domain_pool_t pool {};
    };

    using unique_domain_cell_buffer_t = unique_padded_buffer<compute_domain_cell_t, linux_numa_allocator_t>;
    using compute_domain_cells_t = unique_padded_buffer<unique_domain_cell_buffer_t, linux_numa_allocator_t>;
    /**
     *  @brief A heap allocated array of individual thread pools.
     *
     *  Similar to a @b `std::vector<std::unique_ptr<linux_compute_domain_pool_t>>`, but with each compute_domain placed
     *  on its own NUMA node, and with a custom allocator. All the entries are sorted/grouped by the compute_domain
     *  index in ascending order, and the first one always contains the current thread.
     */
    compute_domain_cells_t compute_domain_cells_ {};

  public:
    linux_distributed_pool(linux_distributed_pool &&) = delete;
    linux_distributed_pool(linux_distributed_pool const &) = delete;
    linux_distributed_pool &operator=(linux_distributed_pool &&) = delete;
    linux_distributed_pool &operator=(linux_distributed_pool const &) = delete;

    linux_distributed_pool(numa_topology_t topo = {}) noexcept : linux_distributed_pool("forkunion", std::move(topo)) {}

    explicit linux_distributed_pool(char const *name, numa_topology_t topo = {}) noexcept : topology_(std::move(topo)) {
        // Accept null or empty names by falling back to a sensible default
        char const *effective_name = (name && name[0] != '\0') ? name : "forkunion";
        std::strncpy(name_, effective_name, sizeof(name_) - 1);
        name_[sizeof(name_) - 1] = '\0';
    }

    ~linux_distributed_pool() noexcept { terminate(); }

    /**
     *  @brief Checks if the thread-pool's core synchronization points are lock-free.
     *  @note Only valid after the `try_spawn` call.
     */
    bool is_lock_free() const noexcept {
        return compute_domain_cells_ && compute_domain_cells_[0] && compute_domain_cells_[0].only().pool.is_lock_free();
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
        for (std::size_t i = 0; i < compute_domain_cells_.size(); ++i)
            total_bytes += compute_domain_cells_[i].only().pool.memory_usage();
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

        // Place the control structures on the first compute domain, pinning the caller there too.
        // We spawn one sub-pool per compute domain (a same-QoS core run), not per NUMA node, so
        // performance and efficiency cores on one node become separate, independently pinned pools.
        compute_domain_t const &first_domain = new_topology.compute_domain_at(0);
        linux_numa_allocator_t allocator {first_domain.node_id};
        index_t const compute_domain_cells_count = std::min(new_topology.compute_domains_count(), threads);

        compute_domain_cells_t compute_domain_cells(allocator);
        if (!compute_domain_cells.try_resize(compute_domain_cells_count)) return false; // ! Allocation failed

        // Allocate each sub-pool on its own compute domain's NUMA node
        for (index_t compute_domain_index = 0; compute_domain_index < compute_domain_cells_count;
             ++compute_domain_index) {
            numa_node_id_t const node_id = new_topology.compute_domain_at(compute_domain_index).node_id;
            linux_numa_allocator_t node_allocator {node_id};
            unique_domain_cell_buffer_t domain_cell_buffer(node_allocator);
            domain_cell_buffer.try_resize(1);
            compute_domain_cells[compute_domain_index] = std::move(domain_cell_buffer);
        }

        auto reset_on_failure = [&]() noexcept {
            for (index_t compute_domain_index = 0; compute_domain_index < compute_domain_cells_count;
                 ++compute_domain_index) {
                if (compute_domain_cells[compute_domain_index].size() == 0) continue; // ? No pool allocated
                compute_domain_cells[compute_domain_index].only().pool.terminate(); // ? Stop the pool if it was started
            }
        };

        // If any one of the allocations failed, we need to clean up
        for (index_t compute_domain_index = 0; compute_domain_index < compute_domain_cells_count;
             ++compute_domain_index) {
            if (compute_domain_cells[compute_domain_index].size() == 1) continue;
            reset_on_failure();
            return false; // ! Allocation failed
        }

        // Every compute-domain pool is spawned separately
        // - the first one may be "inclusive".
        // - others are always "exclusive" to the caller thread.
        indexed_split<thread_index_t> threads_per_domain(threads, compute_domain_cells_count);
        if (!compute_domain_cells[0].only().pool.try_spawn(first_domain, threads_per_domain[0].count, exclusivity,
                                                           pin_granularity, 0, 0)) {
            reset_on_failure();
            return false; // ! Spawning failed
        }

        for (index_t compute_domain_index = 1; compute_domain_index < compute_domain_cells_count;
             ++compute_domain_index) {
            compute_domain_t const &domain = new_topology.compute_domain_at(compute_domain_index);
            compute_domain_cell_t &cell = compute_domain_cells[compute_domain_index].only();
            if (!cell.pool.try_spawn(domain, threads_per_domain[compute_domain_index].count, caller_exclusive_k,
                                     pin_granularity, threads_per_domain[compute_domain_index].first,
                                     compute_domain_index)) {
                reset_on_failure();
                return false; // ! Spawning failed
            }
        }

        topology_ = std::move(new_topology);
        compute_domain_cells_ = std::move(compute_domain_cells);
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
     *  @return A `generation_t` token identifying this dispatch.
     *  @sa Use in conjunction with `unsafe_join`.
     */
    template <typename fork_type_>
    FU_REQUIRES_((can_be_for_thread_callback<fork_type_, index_t>()))
    generation_t unsafe_for_threads(fork_type_ &fork) noexcept {
        assert(compute_domain_cells_ && "Thread pools must be initialized before broadcasting");

        // Submit to every thread pool. All sub-pool epochs advance in lockstep as long as
        // every dispatch goes through this wrapper - never dispatch to a sub-pool directly.
        generation_t last_sub_generation {};
        for (std::size_t i = 1; i < compute_domain_cells_.size(); ++i)
            last_sub_generation = compute_domain_cells_[i].only().pool.unsafe_for_threads(fork);
        generation_t const generation = compute_domain_cells_[0].only().pool.unsafe_for_threads(fork);
        assert((compute_domain_cells_.size() == 1 || last_sub_generation == generation) &&
               "ComputeDomain sub-pools must advance in generation lockstep");
        (void)last_sub_generation;
        return generation;
    }

    /**
     *  @brief Returns true if the generation identified by @p generation has completed on all compute_domain_cells.
     *  @note A `true` result synchronizes with all contributors: their writes are visible.
     *
     *  On `caller_inclusive_k` pools this can only turn `true` once `unsafe_join`
     *  contributes the calling thread's slice, so the poll-then-join pattern is
     *  reserved for `caller_exclusive_k` pools.
     */
    bool is_complete(generation_t generation) const noexcept {
        for (std::size_t i = 0; i < compute_domain_cells_.size(); ++i)
            if (!compute_domain_cells_[i].only().pool.is_complete(generation)) return false;
        return true;
    }

    /**
     *  @brief Blocks the calling thread until the generation identified by @p generation finishes.
     *  @note On `caller_inclusive_k` pools, first executes the calling thread's slice.
     *  Idempotent: returns immediately for already-joined or stale generations.
     */
    void unsafe_join(generation_t generation) noexcept {
        assert(compute_domain_cells_ && "Thread pools must be initialized before broadcasting");

        // Join the caller-hosting compute_domain first: on inclusive pools its slice runs here
        // and overlaps the remote compute_domain_cells' completion instead of waiting behind them.
        compute_domain_cells_[0].only().pool.unsafe_join(generation);
        for (std::size_t i = 1; i < compute_domain_cells_.size(); ++i)
            compute_domain_cells_[i].only().pool.unsafe_join(generation);
    }

    /** @brief Blocks the calling thread until the currently broadcasted task finishes. */
    void unsafe_join() noexcept {
        assert(compute_domain_cells_ && "Thread pools must be initialized before broadcasting");

        // Wait for everyone to finish, starting from the caller-hosting compute_domain
        compute_domain_cells_[0].only().pool.unsafe_join();
        for (std::size_t i = 1; i < compute_domain_cells_.size(); ++i)
            compute_domain_cells_[i].only().pool.unsafe_join();
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
        if (!compute_domain_cells_) return; // ? Uninitialized
        for (std::size_t i = 0; i < compute_domain_cells_.size(); ++i) compute_domain_cells_[i].only().pool.terminate();

        compute_domain_cells_ = {};
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
        for (std::size_t i = 0; i < compute_domain_cells_.size(); ++i)
            compute_domain_cells_[i].only().pool.sleep(wake_up_periodicity_micros);
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

#pragma region ComputeDomains Compatibility

    /**
     *  @brief Number of compute domains this pool spans (one pinned sub-pool each).
     */
    index_t compute_domains_count() const noexcept { return compute_domain_cells_.size(); }

    /**
     *  @brief Returns the number of threads in one NUMA-specific local @b compute_domain.
     *  @retval 0 if the thread-pool is not initialized, 1 if only the main thread is used.
     *  @note This API is @b not synchronized and doesn't check for out-of-bounds access.
     */
    thread_index_t threads_count(index_t compute_domain) const noexcept {
        assert(compute_domain_cells_ && "Local pools must be initialized");
        assert(compute_domain < compute_domain_cells_.size() && "Local pool index out of bounds");
        return compute_domain_cells_[compute_domain].only().pool.threads_count();
    }

    /**
     *  @brief Converts a @p `global_thread_index` to a local thread index within a @b compute_domain.
     *  @retval 0 if the thread-pool is not initialized, 1 if only the main thread is used.
     *  @note This API is @b not synchronized and doesn't check for out-of-bounds access.
     */
    thread_index_t thread_local_index(thread_index_t global_thread_index, index_t compute_domain) const noexcept {
        assert(compute_domain_cells_ && "Local pools must be initialized");
        assert(compute_domain < compute_domain_cells_.size() && "Local pool index out of bounds");
        return global_thread_index - compute_domain_cells_[compute_domain].only().pool.first_thread();
    }

    index_t thread_compute_domain(thread_index_t global_thread_index) const noexcept {
        index_t compute_domain_index = 0;
        for (; compute_domain_index < compute_domain_cells_.size(); ++compute_domain_index) {
            compute_domain_cell_t const &compute_domain = compute_domain_cells_[compute_domain_index].only();
            if (global_thread_index < compute_domain.pool.first_thread()) continue;
            if (global_thread_index < compute_domain.pool.first_thread() + compute_domain.pool.threads_count())
                return compute_domain_index;
        }
        return compute_domain_index; // ? Not found
    }

    std::atomic<index_t> &unsafe_dynamic_progress_ref(index_t compute_domain) noexcept {
        return compute_domain_cells_[compute_domain].only().pool.unsafe_dynamic_progress_ref();
    }

#pragma endregion ComputeDomains Compatibility
};

using linux_compute_domain_pool_t = linux_compute_domain_pool<>;
using linux_distributed_pool_t = linux_distributed_pool<>;

#if FU_DETECT_CONCEPTS_
static_assert(is_unsafe_pool<basic_pool_t> && is_unsafe_pool<linux_compute_domain_pool_t>,
              "These thread pools must be flexible and support unsafe operations");
static_assert(is_pool<basic_pool_t> && is_pool<linux_compute_domain_pool_t> && is_pool<linux_distributed_pool_t>,
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

            // Memory tier, shown only when the machine actually exposes more than one
            if (topology.memory_levels_count() > 1)
                pos += static_cast<std::size_t>(std::snprintf(    //
                    line_buffer + pos, sizeof(line_buffer) - pos, //
                    " • %sTier:%s %s%zu%s",                       //
                    colors.blue(), /* "Tier:" */ colors.reset(),  //
                    colors.bold_blue(), node.memory_level, colors.reset()));

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

} // namespace forkunion
} // namespace ashvardanian
