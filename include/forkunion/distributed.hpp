/**
 *  @file distributed.hpp
 *  @brief Pools that know about compute domains: `colocated_pool` and `distributed_pool`.
 *  @note Included by `<forkunion.hpp>`; not meant to be included on its own.
 *
 *  These are the only pools that touch an operating system beyond `std::thread`: they pin threads to
 *  cores, place their own state on a memory domain, and steal work across domains. The concurrency
 *  protocol - epochs, generation tokens, claim cursors, the invokers - is the same one `basic_pool`
 *  runs, so it is not repeated per platform. Where a kernel call is unavoidable, it appears inline,
 *  guarded, rather than behind a trait: there are three call sites, not thirty.
 */
#pragma once
#include "topology.hpp"
#include "standard.hpp"

namespace ashvardanian {
namespace forkunion {

/**
 *  @brief How tightly a spawned worker is bound to the hardware beneath it.
 *
 *  Pinning to a core keeps a thread's caches warm and its `capacity` predictable, at the cost of
 *  letting it idle while a sibling core is busy. Pinning to a node hands the kernel the whole
 *  domain to schedule within, which survives a core going offline and suits oversubscribed hosts.
 */
enum numa_pin_granularity_t {
    /** Bind each worker to exactly one logical core. */
    numa_pin_to_core_k = 0,
    /** Bind each worker to every core of its NUMA node, and let the kernel choose among them. */
    numa_pin_to_node_k,
};

/**
 *  @brief Tries binding the given address range to a specific NUMA @p `node_id`.
 *  @retval true if binding succeeded, false otherwise.
 */
FU_MAYBE_UNUSED_ static inline bool linux_numa_bind(void *ptr, std::size_t size_bytes,
                                                    numa_node_id_t node_id) noexcept {
#if FU_WITH_NUMA_MEMORY
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
#endif // FU_WITH_NUMA_MEMORY
}

/**
 *  @brief Tries allocating uninitialized memory and binding it to a specific NUMA @p `node_id`.
 *  @retval nullptr if allocation failed or the page size is unsupported.
 *  @retval pointer to the allocated memory on success.
 */
FU_MAYBE_UNUSED_ static inline void *linux_numa_allocate(std::size_t size_bytes, std::size_t page_size_bytes,
                                                         numa_node_id_t node_id) noexcept {
    assert(node_id >= 0 && "NUMA node ID must be non-negative");

#if FU_WITH_NUMA_MEMORY

    // Fast path: regular pages – let `libnuma` handle any rounding internally.
    if (page_size_bytes == static_cast<std::size_t>(::numa_pagesize())) return ::numa_alloc_onnode(size_bytes, node_id);

#if FU_WITH_HUGE_PAGES

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
    return nullptr; // ! Every page size but the base one needs `MAP_HUGETLB`
#endif

#else
    fu_unused_(size_bytes);
    fu_unused_(page_size_bytes);
    fu_unused_(node_id);
    return nullptr;
#endif // FU_WITH_NUMA_MEMORY
}

FU_MAYBE_UNUSED_ static inline void linux_numa_free(void *ptr, std::size_t size_bytes) noexcept {
    assert(ptr != nullptr && "Pointer must not be null");
    assert(size_bytes > 0 && "Size must be greater than zero");
#if FU_WITH_NUMA_MEMORY
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
    /** Unique NUMA node ID, in [0, numa_max_node()). */
    numa_node_id_t node_id_ {-1};
    /** RAM page size in bytes, typically 4 KB. */
    size_type default_page_size_ {0};

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

#if FU_WITH_COLOCATED_POOLS

/**
 *  @brief Sleeps the calling thread for @p micros microseconds.
 *  @note Linux's `clock_nanosleep` lets us name the clock; Darwin only has `nanosleep`, whose clock
 *        is monotonic anyway. Neither is interruptible by our wake path - the sleep is short.
 */
FU_MAYBE_UNUSED_ static inline void nap_for_micros(std::size_t const micros) noexcept {
    struct timespec ts {0, static_cast<long>(micros * 1000)};
#if FU_ON_LINUX
    ::clock_nanosleep(CLOCK_MONOTONIC, 0, &ts, nullptr);
#else
    ::nanosleep(&ts, nullptr);
#endif
}

/**
 *  @brief Confines @p thread to @p cores, where the kernel allows it.
 *  @retval false when the platform exposes no thread placement, which is @b not an error.
 *
 *  Linux hands out a `cpu_set_t` and honours it. Apple Silicon answers `KERN_NOT_SUPPORTED` to
 *  `thread_policy_set(THREAD_AFFINITY_POLICY)` - measured, not assumed - and offers only a
 *  Quality-of-Service class, chosen at creation. So a pool there partitions the @b work by domain and
 *  lets the scheduler place the @b threads. The data stays cluster-local; the threads do not.
 */
FU_MAYBE_UNUSED_ static inline bool pin_thread_to_cores(FU_MAYBE_UNUSED_ pthread_t thread,
                                                        FU_MAYBE_UNUSED_ numa_core_id_t const *cores,
                                                        FU_MAYBE_UNUSED_ std::size_t const count) noexcept {
#if FU_WITH_THREAD_PINNING
    std::size_t const max_cores = possible_cores();
    cpu_set_t *cpu_set_ptr = CPU_ALLOC(max_cores);
    if (!cpu_set_ptr) return false;
    std::size_t const cpu_set_size = CPU_ALLOC_SIZE(max_cores);
    CPU_ZERO_S(cpu_set_size, cpu_set_ptr);
    for (std::size_t i = 0; i < count; ++i) {
        assert(cores[i] >= 0 && "Invalid CPU core ID");
        CPU_SET_S(cores[i], cpu_set_size, cpu_set_ptr);
    }
    bool const pinned = ::pthread_setaffinity_np(thread, cpu_set_size, cpu_set_ptr) == 0;
    CPU_FREE(cpu_set_ptr);
    return pinned;
#else
    return false; // ? No placement here; the harvest still reports the domains
#endif
}

/**
 *  @brief Widens the calling thread back to every core the machine has.
 *  @note A no-op where nothing was ever narrowed.
 */
FU_MAYBE_UNUSED_ static inline void unpin_current_thread() noexcept {
#if FU_WITH_THREAD_PINNING
    std::size_t const max_cores = possible_cores();
    cpu_set_t *cpu_set_ptr = CPU_ALLOC(max_cores);
    if (!cpu_set_ptr) return;
    std::size_t const cpu_set_size = CPU_ALLOC_SIZE(max_cores);
    CPU_ZERO_S(cpu_set_size, cpu_set_ptr);
    for (std::size_t cpu = 0; cpu < max_cores; ++cpu) CPU_SET_S(cpu, cpu_set_size, cpu_set_ptr);
    FU_MAYBE_UNUSED_ int const pin_result = ::pthread_setaffinity_np(::pthread_self(), cpu_set_size, cpu_set_ptr);
    assert(pin_result == 0 && "Failed to reset the caller thread's affinity");
    CPU_FREE(cpu_set_ptr);
#endif
#if FU_WITH_NUMA_MEMORY
    FU_MAYBE_UNUSED_ int const spread_result = ::numa_run_on_node(-1);
    assert(spread_result == 0 && "Failed to reset the caller thread's NUMA node affinity");
#endif
}

/**
 *  @brief Used inside `colocated_pool` to describe a pinned thread.
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
    std::atomic<std::uint64_t> id {}; // ? `gettid` on Linux, `pthread_threadid_np` on Apple
    numa_core_id_t core_id {-1};
    char name[16] {};           // ? Written by the spawner, applied by the worker to itself
    qos_level_t qos_level {-1}; // TODO: Populate from VFS, if available
    /**
     *  @brief This thread's private cursor for `for_n_dynamic`. @sa `dynamic_claim`.
     *  @note Lives here, rather than in a second array, so the pool allocates once and the cursor
     *        inherits both this record's cache-line padding and its NUMA node.
     *  @note Fixed to `std::size_t` because `colocated_pool` is not templated on an index
     *        width, unlike `basic_pool`. The narrow-index debug configs, and the cursor's overflow
     *        argument, therefore only ever exercise `basic_pool`.
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
struct colocated_pool {

  public:
#if FU_WITH_NUMA_MEMORY
    using allocator_t = linux_numa_allocator_t; // ? Places the pool's own state on its node
#else
    using allocator_t = std::allocator<char>; // ? One memory domain; there is nothing to place
#endif
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

    /** The index of the first thread to start from. */
    thread_index_t first_thread_ {0};
    /** Whether the caller thread is included in the count. */
    caller_exclusivity_t exclusivity_ {caller_inclusive_k};
    /** How long to sleep in microseconds when waiting for tasks. */
    std::size_t sleep_length_micros_ {0};

    using char16_name_t = char[16]; // ? Fixed-size thread name buffer, for POSIX thread naming
    /** Thread name buffer, for POSIX thread naming. */
    char16_name_t name_ {};
    /** Unique NUMA node ID, in [0, numa_max_node()). */
    numa_node_id_t numa_node_id_ {-1};
    /** Unique {NUMA node + QoS level} compute_domain ID, defined externally. */
    index_t compute_domain_index_ {0};
    numa_pin_granularity_t pin_granularity_ {numa_pin_to_core_k};

    alignas(alignment_k) std::atomic<mood_t> mood_ {mood_t::grind_k};

    // Task-specific variables:
    /** Pointer to the users lambda. */
    punned_fork_context_t fork_state_ {nullptr};
    /** Calls the lambda. */
    trampoline_t fork_trampoline_ {nullptr};
    alignas(alignment_k) std::atomic<thread_index_t> threads_to_sync_ {0};
    alignas(alignment_k) std::atomic<epoch_index_t> epoch_ {0};

    // ! Still the single cursor `invoke_distributed_for_n_dynamic` drains across compute domains;
    // ! this pool's own `for_n_dynamic` uses the per-thread `dynamic_claims_` below instead.
    alignas(alignment_k) std::atomic<index_t> dynamic_progress_ {0};

  public:
    colocated_pool(colocated_pool &&) = delete;
    colocated_pool(colocated_pool const &) = delete;
    colocated_pool &operator=(colocated_pool &&) = delete;
    colocated_pool &operator=(colocated_pool const &) = delete;

    explicit colocated_pool(char const *name = "forkunion") noexcept {
        // Accept NULL or empty names by falling back to a sensible default
        char const *effective_name = (name && name[0] != '\0') ? name : "forkunion";
        std::strncpy(name_, effective_name, sizeof(name_) - 1);
        name_[sizeof(name_) - 1] = '\0';
    }

    ~colocated_pool() noexcept { terminate(); }

    /**
     *  @brief Estimates the amount of memory managed by this pool handle and internal structures.
     *  @note This API is @b not synchronized.
     */
    std::size_t memory_usage() const noexcept {
        return sizeof(colocated_pool) + threads_count() * sizeof(numa_pthread_t);
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
                   thread_index_t const first_thread = 0, index_t const compute_domain_index = 0,
                   FU_MAYBE_UNUSED_ index_t const compute_levels = 1) noexcept {

        if (threads == 0) return false;          // ! Can't have zero threads working on something
        if (pthreads_.size() != 0) return false; // ! Already initialized

        // Allocate the thread pool of `numa_pthread_t` objects
#if FU_WITH_NUMA_MEMORY
        allocator_ = linux_numa_allocator_t {domain.node_id};
#endif
        numa_pthread_allocator_t pthread_allocator {allocator_};
        unique_padded_buffer<numa_pthread_t, numa_pthread_allocator_t> pthreads {pthread_allocator};
        if (!pthreads.try_resize(threads)) return false; // ! Allocation failed

        // Core IDs may outrun the online core count where cores can be hot-plugged.
        std::size_t const max_possible_cores = possible_cores();

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
            pthreads_[0].id.store(current_thread_id(), std::memory_order_release);
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
            pthread_attr_t attributes;
            ::pthread_attr_init(&attributes);
#if FU_WITH_THREAD_QOS
            // Apple offers no pinning; a Quality-of-Service class is the whole placement story, and
            // it must be chosen before the thread exists. On a chip with efficiency cores, `UTILITY`
            // is what confines a thread to them; on an all-performance chip the class is inert.
            ::pthread_attr_set_qos_class_np(&attributes, _qos_for_level(domain.compute_level, compute_levels), 0);
#endif
            int creation_result = ::pthread_create(&new_pthread_handle, &attributes, &_posix_worker_loop, this);
            ::pthread_attr_destroy(&attributes);
            pthreads_[i].handle.store(new_pthread_handle, std::memory_order_relaxed);
            pthreads_[i].id.store(0, std::memory_order_relaxed); // ? 0 means "not published yet"
            pthreads_[i].core_id = -1;                           // ? Not pinned yet

            if (creation_result != 0) {
                mood_.store(mood_t::die_k, std::memory_order_release);
                for (thread_index_t j = use_caller_thread; j < i; ++j) {
                    pthread_t cancel_pthread_handle = pthreads_[j].handle.load(std::memory_order_relaxed);
                    FU_MAYBE_UNUSED_ int cancel_result = ::pthread_cancel(cancel_pthread_handle);
                    assert(cancel_result == 0 && "Failed to cancel a thread");
                }
                reset_on_failure();
                return false; // ! Thread creation failed
            }
        }

        // Compose each thread's name. Apple can only name the calling thread, so the worker applies it
        // to itself; we merely publish it here, before the workers read their cells.
        // ! `i % core_count` because a pool may hold more threads than its domain has cores.
        for (thread_index_t i = 0; i < pthreads_.size(); ++i)
            fill_thread_name(pthreads_[i].name, name_,
                             static_cast<std::size_t>(domain.first_core_id[i % domain.core_count]), max_possible_cores);
        if (use_caller_thread) set_current_thread_name(pthreads_[0].name);

        // Pin all of the threads. Where the kernel refuses, the domains still describe the machine
        // and the pool still partitions work by them - it simply cannot hold a thread in place.
        if (pin_granularity == numa_pin_to_core_k) {
            for (thread_index_t i = 0; i < pthreads_.size(); ++i) {
                numa_core_id_t const cpu = domain.first_core_id[i % domain.core_count];
                pthread_t const pin_pthread_handle = pthreads_[i].handle.load(std::memory_order_relaxed);
                if (pin_thread_to_cores(pin_pthread_handle, &cpu, 1)) pthreads_[i].core_id = cpu;
            }
        }
        else {
            for (thread_index_t i = 0; i < pthreads_.size(); ++i) {
                pthread_t const pin_pthread_handle = pthreads_[i].handle.load(std::memory_order_relaxed);
                (void)pin_thread_to_cores(pin_pthread_handle, domain.first_core_id, domain.core_count);
            }
        }

        // If all went well, we can store the thread-pool and start using it
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
    broadcast_join<colocated_pool, fork_type_> for_threads(fork_type_ &&fork) noexcept {
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
                std::uint64_t const pthread_id = pthreads_[i].id.load(std::memory_order_acquire);
                if (pthread_id == 0) continue; // ! Unsigned now: `< 0` could never fire
#if FU_WITH_THREAD_SCHED_CLASS
                // Nudge the sleeping worker back onto a runnable class. Darwin has no equivalent
                // for another thread; its QoS class is fixed at creation.
                sched_param param {};
                ::sched_setscheduler(static_cast<pid_t>(pthread_id), SCHED_FIFO | SCHED_RR, &param);
#else
                fu_unused_(pthread_id);
#endif
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
            std::uint64_t const pthread_id = pthreads_[i].id.load(std::memory_order_acquire);
            if (pthread_id == 0) continue; // ! Unsigned now: `< 0` could never fire
#if FU_WITH_THREAD_SCHED_CLASS
            sched_param param {};
            ::sched_setscheduler(static_cast<pid_t>(pthread_id), SCHED_IDLE, &param);
#else
            fu_unused_(pthread_id); // ? No idle scheduling class on Darwin
#endif
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
    broadcast_join<colocated_pool, invoke_for_slices<fork_type_, index_t>> //
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
    broadcast_join<colocated_pool, invoke_for_n<fork_type_, index_t>> //
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
    broadcast_join<colocated_pool, invoke_for_n_dynamic<colocated_pool, fork_type_, index_t>> //
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

    void _reset_affinity() noexcept { unpin_current_thread(); }

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
        colocated_pool *pool = static_cast<colocated_pool *>(arg);

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
            std::uint64_t const pthread_id = current_thread_id();
            numa_pthreads[local_thread_index].id.store(pthread_id, std::memory_order_release);

            // Apple can only name the calling thread, so every worker names itself.
            set_current_thread_name(numa_pthreads[local_thread_index].name);

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
                nap_for_micros(pool->sleep_length_micros_);
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

#if FU_WITH_THREAD_QOS
    /**
     *  @brief Maps a compute level onto the only placement control Darwin offers.
     *  @param[in] level The domain's `compute_level`, where higher is more performant.
     *  @param[in] levels Distinct levels the machine reports; 1 means every core is interchangeable.
     *
     *  The top tier asks for `USER_INITIATED`, not `USER_INTERACTIVE`: the latter is reserved for work
     *  a person is waiting on, and a compute pool is not that. Everything below it takes `DEFAULT`.
     *
     *  @note It is tempting to give the bottom tier `QOS_CLASS_UTILITY`, which is what confines a
     *        thread to efficiency cores. That is wrong here. `compute_level` is a dense rank over
     *        `hw.perflevelN`, and the bottom rank is only an @b efficiency tier on the chips that have
     *        one. An M5 Pro reports two levels named "Super" and "Performance", both big cores - and
     *        `UTILITY` would deprioritize two thirds of the machine. Telling the two cases apart needs
     *        `hw.perflevelN.name`, which the harvest does not yet keep.
     */
    static qos_class_t _qos_for_level(std::size_t const level, std::size_t const levels) noexcept {
        if (levels <= 1) return QOS_CLASS_USER_INITIATED;
        if (level + 1 == levels) return QOS_CLASS_USER_INITIATED; // ? The fastest tier this chip has
        return QOS_CLASS_DEFAULT;
    }
#endif

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
 *  - implementation of `try_spawn`: redirects to individual `colocated_pool` instances.
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
struct distributed_pool {

    using colocated_pool_t = colocated_pool<micro_yield_type_, alignment_>;
    using numa_topology_t = numa_topology<>;

#if FU_WITH_NUMA_MEMORY
    using allocator_t = linux_numa_allocator_t; // ? Places the pool's own state on its node
#else
    using allocator_t = std::allocator<char>; // ? One memory domain; there is nothing to place
#endif

    /**
     *  @brief An allocator that places bytes on @p node_id, where the kernel can honour that.
     *  @note On a machine without NUMA memory the node is meaningless and the argument is dropped.
     */
    static allocator_t allocator_for_node(FU_MAYBE_UNUSED_ numa_node_id_t const node_id) noexcept {
#if FU_WITH_NUMA_MEMORY
        return allocator_t {node_id};
#else
        return allocator_t {};
#endif
    }

    using micro_yield_t = typename colocated_pool_t::micro_yield_t;
    using index_t = typename colocated_pool_t::index_t;
    using epoch_index_t = typename colocated_pool_t::epoch_index_t;
    using generation_t = epoch_index_t;
    using thread_index_t = typename colocated_pool_t::thread_index_t;
    static constexpr std::size_t alignment_k = colocated_pool_t::alignment_k;
    using prong_t = local_prong<index_t>;

  private:
    numa_topology_t topology_ {};
    char name_[16] {}; // ? Thread name buffer, for POSIX thread naming
    thread_index_t threads_count_ {0};
    /** Whether the caller thread is included in the count. */
    caller_exclusivity_t exclusivity_ {caller_inclusive_k};

    /** @brief One `colocated_pool_t`, padded so neighbouring domains never share a cache line. */
    struct compute_domain_cell_t {
        alignas(alignment_k) colocated_pool_t pool {};
    };

    using unique_domain_cell_buffer_t = unique_padded_buffer<compute_domain_cell_t, allocator_t>;
    using compute_domain_cells_t = unique_padded_buffer<unique_domain_cell_buffer_t, allocator_t>;
    /**
     *  @brief A heap allocated array of individual thread pools.
     *
     *  Similar to a @b `std::vector<std::unique_ptr<colocated_pool_t>>`, but with each compute_domain placed
     *  on its own NUMA node, and with a custom allocator. All the entries are sorted/grouped by the compute_domain
     *  index in ascending order, and the first one always contains the current thread.
     */
    compute_domain_cells_t compute_domain_cells_ {};

  public:
    distributed_pool(distributed_pool &&) = delete;
    distributed_pool(distributed_pool const &) = delete;
    distributed_pool &operator=(distributed_pool &&) = delete;
    distributed_pool &operator=(distributed_pool const &) = delete;

    distributed_pool(numa_topology_t topo = {}) noexcept : distributed_pool("forkunion", std::move(topo)) {}

    explicit distributed_pool(char const *name, numa_topology_t topo = {}) noexcept : topology_(std::move(topo)) {
        // Accept null or empty names by falling back to a sensible default
        char const *effective_name = (name && name[0] != '\0') ? name : "forkunion";
        std::strncpy(name_, effective_name, sizeof(name_) - 1);
        name_[sizeof(name_) - 1] = '\0';
    }

    ~distributed_pool() noexcept { terminate(); }

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
        std::size_t total_bytes = sizeof(distributed_pool);
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
        compute_domain_t const &first_domain = new_topology.compute_domain_at(compute_domain_index_t {});
        allocator_t allocator = allocator_for_node(first_domain.node_id);
        index_t const compute_domain_cells_count = std::min(new_topology.compute_domains_count(), threads);

        compute_domain_cells_t compute_domain_cells(allocator);
        if (!compute_domain_cells.try_resize(compute_domain_cells_count)) return false; // ! Allocation failed

        // Allocate each sub-pool on its own compute domain's NUMA node
        for (index_t compute_domain_index = 0; compute_domain_index < compute_domain_cells_count;
             ++compute_domain_index) {
            numa_node_id_t const node_id =
                new_topology.compute_domain_at(static_cast<compute_domain_index_t>(compute_domain_index)).node_id;
            allocator_t node_allocator = allocator_for_node(node_id);
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
        index_t const compute_levels = static_cast<index_t>(new_topology.compute_levels_count());
        if (!compute_domain_cells[0].only().pool.try_spawn(first_domain, threads_per_domain[0].count, exclusivity,
                                                           pin_granularity, 0, 0, compute_levels)) {
            reset_on_failure();
            return false; // ! Spawning failed
        }

        for (index_t compute_domain_index = 1; compute_domain_index < compute_domain_cells_count;
             ++compute_domain_index) {
            compute_domain_t const &domain =
                new_topology.compute_domain_at(static_cast<compute_domain_index_t>(compute_domain_index));
            compute_domain_cell_t &cell = compute_domain_cells[compute_domain_index].only();
            if (!cell.pool.try_spawn(domain, threads_per_domain[compute_domain_index].count, caller_exclusive_k,
                                     pin_granularity, threads_per_domain[compute_domain_index].first,
                                     compute_domain_index, compute_levels)) {
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
    broadcast_join<distributed_pool, fork_type_> for_threads(fork_type_ &&fork) noexcept {
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
    broadcast_join<distributed_pool, invoke_distributed_for_slices<distributed_pool, fork_type_, index_t>> //
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
    broadcast_join<distributed_pool, invoke_distributed_for_n<distributed_pool, fork_type_, index_t>> //
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
    broadcast_join<distributed_pool, invoke_distributed_for_n_dynamic<distributed_pool, fork_type_, index_t>> //
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

using colocated_pool_t = colocated_pool<>;
using distributed_pool_t = distributed_pool<>;

#if FU_DETECT_CONCEPTS_
static_assert(is_unsafe_pool<basic_pool_t> && is_unsafe_pool<colocated_pool_t>,
              "These thread pools must be flexible and support unsafe operations");
static_assert(is_pool<basic_pool_t> && is_pool<colocated_pool_t> && is_pool<distributed_pool_t>,
              "These thread pools must be fully compatible with the high-level APIs");
#endif // FU_DETECT_CONCEPTS_

#endif // FU_WITH_COLOCATED_POOLS

#pragma endregion - Linux Pool

} // namespace forkunion
} // namespace ashvardanian
