/**
 *  @file distributed.hpp
 *  @brief Pools that know about compute domains: `colocated_pool` and `distributed_pool`.
 *  @note Included by `<forkunion.hpp>`; not meant to be included on its own.
 *
 *  These are the only pools that touch an operating system beyond `std::thread`: they pin threads to
 *  cores, place their own state on a memory domain, and steal work across domains. The concurrency
 *  protocol - epochs, generation tokens, claim cursors, the invokers - is the same one `flat_pool`
 *  runs, so it is not repeated per platform. Where a kernel call is unavoidable, it appears inline,
 *  guarded, rather than behind a trait: there are three call sites, not thirty.
 */
#pragma once
#include "topology.hpp"
#include "flat.hpp"
#include "allocators.hpp" // `domain_allocator_t`, `replicated_array`, `sharded_array`

namespace ashvardanian {
namespace forkunion {

#if FU_WITH_OS_THREADS

/**
 *  @brief Sleeps the calling thread for @p micros microseconds.
 *  @note Linux's `clock_nanosleep` lets us name the clock; Darwin only has `nanosleep`, whose clock
 *      is monotonic anyway. Neither is interruptible by our wake path - the sleep is short.
 */
FU_MAYBE_UNUSED_ static inline void sleep_for_micros(FU_MAYBE_UNUSED_ std::size_t const micros) noexcept {
#if FU_ON_WINDOWS
    // Reached only after the pool is told to `sleep` to save power, where the docs promise latency is
    // irrelevant - so a millisecond-granular `Sleep` (rounded up) is enough, and spares us the per-nap
    // timer object a sub-millisecond wait would cost.
    ::Sleep(static_cast<DWORD>(div_ceil(micros, 1000)));
#elif FU_ON_LINUX
    struct timespec ts {0, static_cast<long>(micros * 1000)};
    ::clock_nanosleep(CLOCK_MONOTONIC, 0, &ts, nullptr); // ? A named clock; Darwin has only `nanosleep`
#else
    struct timespec ts {0, static_cast<long>(micros * 1000)};
    ::nanosleep(&ts, nullptr);
#endif
}

/**
 *  @brief How tightly a spawned worker is bound to the hardware beneath it.
 *
 *  Pinning to a core keeps a thread's caches warm and its `capacity` predictable, at the cost of
 *  letting it idle while a sibling core is busy. Pinning to a node hands the kernel the whole
 *  domain to schedule within, which survives a core going offline and suits oversubscribed hosts.
 */
enum pin_granularity_t {
    /** Bind each worker to exactly one logical core. */
    pin_to_core_k = 0,
    /** Bind each worker to every core of its NUMA node, and let the kernel choose among them. */
    pin_to_memory_domain_k,
};

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
struct alignas(default_alignment_k) pinned_thread_t {
    /** @brief The OS thread handle: `pthread_t` on POSIX, `HANDLE` on Windows. */
    std::atomic<native_thread_t> handle {};
    /** @brief The OS thread id: `gettid` on Linux, `pthread_threadid_np` on Apple, `GetCurrentThreadId` on Windows. */
    std::atomic<std::uint64_t> id {};
    /** @brief The core this worker is pinned to, or -1 when unpinned. */
    core_id_t core_id {-1};
    /** @brief Thread name, written by the spawner and applied by the worker to itself. */
    char name[16] {};
#if FU_WITH_PLACE_THREADS_BY_CORE_CLASS
    /** @brief Apple's absolute class for this worker's domain, or -1 when unnamed.
     *  @sa `compute_domain_t::apple_core_quality`, copied here at spawn. */
    core_quality_t apple_core_quality {-1};
#endif
    /**
     *  @brief This thread's private cursor for `for_n_dynamic`. @sa `dynamic_claim`.
     *  @note Lives here, rather than in a second array, so the pool allocates once and the cursor
     *      inherits both this record's cache-line padding and its NUMA node.
     *  @note Fixed to `std::size_t` because `colocated_pool` is not templated on an index
     *      width, unlike `flat_pool`. The narrow-index debug configs, and the cursor's overflow
     *      argument, therefore only ever exercise `flat_pool`.
     */
    dynamic_claim<std::size_t> claim {};
};

#pragma region Colocated Pool

/**
 *  @brief A Linux-only thread-pool pinned to one NUMA node and same QoS level physical cores.
 *
 *  Differs from the `flat_pool` template in the following ways:
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
 *  - avoid recreating the @b `machine_topology`, as it's expensive to harvest.
 *
 *  The synchronization protocol - epochs, generations, contributor counting, and the memory
 *  ordering rules - is identical to `flat_pool`; @sa @ref pool_concurrency_model.
 */
template <typename micro_yield_type_ = standard_yield_t, typename cache_hints_type_ = standard_cache_hints_t,
          std::size_t alignment_ = default_alignment_k>
struct colocated_pool {

  public:
#if FU_WITH_PLACE_MEMORY_ON_DOMAIN
    using allocator_t = domain_allocator_t; // ? Places the pool's own state on its node
#else
    using allocator_t = std::allocator<char>; // ? One memory domain; there is nothing to place
#endif
    using micro_yield_t = micro_yield_type_;
    using cache_hints_t = cache_hints_type_;
    static constexpr pool_kind_t kind_k = pool_kind_t::colocated_k;
    static constexpr std::size_t alignment_k = alignment_;
    static_assert(is_power_of_two(alignment_k), "Alignment must be a power of 2");

    using index_t = std::size_t;        // ? Not templated like `flat_pool`; narrow-index debug configs live there
    using epoch_index_t = index_t;      // ? A.k.a. number of previous API calls in [0, UINT_MAX)
    using generation_t = epoch_index_t; // ? A.k.a. token returned from `unsafe_for_threads`
    using thread_index_t = index_t;     // ? A.k.a. "core index" or "thread ID" in [0, threads_count)
    using local_thread_t = local_thread<thread_index_t>;
    using prong_t = local_prong<index_t>;

    using punned_fork_context_t = void *;                                 // ? Pointer to the on-stack lambda
    using trampoline_t = void (*)(punned_fork_context_t, local_thread_t); // ? Wraps lambda's `operator()`

    static_assert(is_wait_functor<micro_yield_t, epoch_index_t, thread_index_t>::value,
                  "Yield must be callable as `yield(watched_atomic, observed_value, thread_index)`");
    static_assert(is_cache_hints_functor<cache_hints_t>::value,
                  "Cache hints must be callable as `hints(address, demote_line_k)` and `(address, promote_line_k)`");

  private:
    using allocator_traits_t = std::allocator_traits<allocator_t>;
    using pinned_threads_allocator_t = typename allocator_traits_t::template rebind_alloc<pinned_thread_t>;
    using claim_t = dynamic_claim<index_t>; // ? Lives inside each `pinned_thread_t`, so no extra array

    // Thread-pool-specific variables:
    /** @brief Allocator placing the pool's own state on its memory domain. */
    allocator_t allocator_ {};

    /**
     *  @brief One padded `pinned_thread_t` per thread: its handle, kernel id, and claim cursor.
     *
     *  Differs from STL `workers_` in base in type and size, as it may contain the `pthread_self`
     *  at the first position. If the @b `pin_to_core_k` granularity is used, the `pinned_thread_t::core_id`
     *  will be set to the individual core IDs.
     */
    dynamic_padded_array<pinned_thread_t, pinned_threads_allocator_t> pthreads_ {};

    /** @brief The global index of this pool's first thread, offsetting its local indices. */
    thread_index_t first_thread_ {0};
    /** @brief How long to nap in microseconds while `chill_k`, waiting for work. */
    std::size_t sleep_length_micros_ {0};

    using char16_name_t = char[16]; // ? Fixed-size thread name buffer, for POSIX thread naming
    /** @brief Thread name buffer applied to each worker for POSIX/OS naming. */
    char16_name_t name_ {};
    /** @brief Whether the caller thread is counted as one of the contributors. */
    caller_exclusivity_t exclusivity_ {caller_inclusive_k};
    /** @brief The OS's id for the memory domain this pool allocates and runs on. */
    memory_domain_id_t memory_domain_id_ {-1};
    /** @brief Our dense index for this pool's compute domain, assigned by the caller. */
    index_t compute_domain_index_ {0};
    /** @brief Whether workers pin to one core each or to the whole memory domain. */
    pin_granularity_t pin_granularity_ {pin_to_core_k};

    /** @brief The caller's affinity as it was before this pool narrowed it. @sa `_reset_affinity`. */
    core_mask_t caller_affinity_;
    /** @brief Workers the kernel refused to place, so a caller can tell a pinned pool from a crowded one. */
    thread_index_t unpinned_threads_ {0};

    /** @brief Lifecycle switch between spinning (`grind_k`), sleeping (`chill_k`), and exiting (`die_k`). */
    alignas(alignment_k) std::atomic<mood_t> mood_ {mood_t::grind_k};

    // Task-specific variables:
    /** @brief Type-erased pointer to the caller's on-stack fork lambda. */
    punned_fork_context_t fork_state_ {nullptr};
    /** @brief Invokes the punned fork lambda for a given `local_thread_t`. */
    trampoline_t fork_trampoline_ {nullptr};
    /** @brief Countdown of contributors still running; the one reaching zero signals completion. */
    alignas(alignment_k) std::atomic<thread_index_t> threads_to_sync_ {0};
    /** @brief Generation clock: odd while a fork is in flight, even when idle. */
    alignas(alignment_k) std::atomic<epoch_index_t> epoch_ {0};

  public:
    colocated_pool(colocated_pool &&) = delete;
    colocated_pool(colocated_pool const &) = delete;
    colocated_pool &operator=(colocated_pool &&) = delete;
    colocated_pool &operator=(colocated_pool const &) = delete;

    explicit colocated_pool(char const *name = "forkunion") noexcept { rename(name); }

    /** @brief Replaces the pool's name; only threads spawned after the call pick it up. */
    void rename(char const *name) noexcept {
        // Accept NULL or empty names by falling back to a sensible default
        char const *effective_name = (name && name[0] != '\0') ? name : "forkunion";
        std::size_t const source_length = std::strlen(effective_name);
        std::size_t const name_length = source_length < sizeof(name_) ? source_length : sizeof(name_) - 1;
        std::memcpy(name_, effective_name, name_length);
        name_[name_length] = '\0';
    }

    ~colocated_pool() noexcept { terminate(); }

    /**
     *  @brief Estimates the amount of memory managed by this pool handle and internal structures.
     *  @note This API is @b not synchronized.
     */
    std::size_t memory_usage() const noexcept {
        return sizeof(colocated_pool) + threads_count() * sizeof(pinned_thread_t);
    }

    /** @brief Checks if the thread-pool's core synchronization points are lock-free. */
    bool is_lock_free() const noexcept { return mood_.is_lock_free() && threads_to_sync_.is_lock_free(); }

    /**
     *  @brief Returns the memory domain this thread-pool is pinned to.
     *  @retval -1 if the thread-pool is not initialized or the memory domain is unknown.
     *  @note This API is @b not synchronized.
     */
    memory_domain_id_t memory_domain_id() const noexcept { return memory_domain_id_; }

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

    /** @brief Exposes a thread's private claim cursor, kept inside its `pinned_thread_t`. */
    claim_t &unsafe_dynamic_claim_ref(thread_index_t const thread) noexcept { return pthreads_[thread].claim; }

#pragma region Core API

    /**
     *  @brief Returns the number of threads in the thread-pool, including the main thread.
     *  @retval 0 if the thread-pool is not initialized, 1 if only the main thread is used.
     *  @note This API is @b not synchronized.
     */
    thread_index_t threads_count() const noexcept { return pthreads_.size(); }

    /** @brief Workers the kernel refused to place; zero on a fully pinned pool. */
    thread_index_t unpinned_threads_count() const noexcept { return unpinned_threads_; }

    /**
     *  @brief Whether every worker sits on the core this pool asked for.
     *  @note False on platforms with no thread placement, and false when a `cpuset` crowded the pool
     *      onto fewer cores than it has threads - which is where spinning workers fall apart.
     */
    bool all_threads_pinned() const noexcept { return threads_count() != 0 && unpinned_threads_ == 0; }

    /**
     *  @brief Reports if the current calling thread will be used for broadcasts.
     *  @note This API is @b not synchronized.
     */
    caller_exclusivity_t caller_exclusivity() const noexcept { return exclusivity_; }

    /**
     *  @brief Creates a thread-pool addressing every core of the given compute @p domain.
     *  @param[in] domain The compute domain to spawn on: its memory domain, cores, and QoS level.
     *  @param[in] exclusivity Should we count the calling thread as one of the threads?
     *  @retval false if the number of threads is zero or if spawning has failed.
     *  @retval true if the thread-pool was created successfully, started, and is ready to use.
     *  @note This is the de-facto @b constructor - you only call it again after `terminate`.
     *  @sa Other overloads of `try_spawn` that allow to specify the number of threads.
     */
    bool try_spawn(compute_domain_t const &domain,
                   caller_exclusivity_t const exclusivity = caller_inclusive_k) noexcept {
        return try_spawn(domain, domain.logical_cores_count, exclusivity);
    }

    /**
     *  @brief Creates a thread-pool with the given number of @p threads on the given NUMA @p node.
     *  @param[in] domain The compute domain to spawn on: its memory domain, cores, and QoS level.
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
     *  We may accept @p threads different from the @p domain.logical_cores_count, which allows us to:
     *  - over-subscribe the cores, i.e. use more threads than cores available on the NUMA node.
     *  - under-subscribe the cores, i.e. use fewer threads than cores available on the NUMA node.
     *
     *  If you only have one thread-pool active at any part of your application, that's meaningless.
     *  You'd be better off using exactly the number of cores available on the NUMA node and pinning
     *  them to individual cores with @b `pin_to_core_k` granularity.
     */
    bool try_spawn(compute_domain_t const &domain, thread_index_t const threads,
                   caller_exclusivity_t const exclusivity = caller_inclusive_k,
                   pin_granularity_t const pin_granularity = pin_to_core_k, thread_index_t const first_thread = 0,
                   index_t const compute_domain_index = 0, FU_MAYBE_UNUSED_ index_t const compute_levels = 1) noexcept {

        if (threads == 0) return false;          // ! Can't have zero threads working on something
        if (pthreads_.size() != 0) return false; // ! Already initialized

        // Allocate the thread pool of `pinned_thread_t` objects
#if FU_WITH_PLACE_MEMORY_ON_DOMAIN
        allocator_ = domain_allocator_t {domain.memory_domain_id};
#endif
        pinned_threads_allocator_t pthread_allocator {allocator_};
        dynamic_padded_array<pinned_thread_t, pinned_threads_allocator_t> pthreads {pthread_allocator};
        if (!pthreads.try_resize(threads)) return false; // ! Allocation failed

        // Core IDs may outrun the online core count where cores can be hot-plugged.
        std::size_t const max_possible_cores = possible_cores();

        // Before we start the threads, make sure we set some of the shared
        // state variables that will be used in the `_posix_worker_loop` function.
        pthreads_ = std::move(pthreads);
        first_thread_ = first_thread;
        compute_domain_index_ = compute_domain_index;
        exclusivity_ = exclusivity;
        memory_domain_id_ = domain.memory_domain_id;
        pin_granularity_ = pin_granularity;
        auto reset_on_failure = [&]() noexcept {
            pthreads_ = {};
            memory_domain_id_ = -1;
            pin_granularity_ = pin_to_core_k;
        };

        // Snapshot the caller's affinity before we narrow it, so teardown can put back exactly what
        // it had rather than the whole machine. Captured even when the caller is excluded: the
        // failure path below may still have touched it.
        try_capture_thread_cores(caller_affinity_);

        // Include the main thread into the list of handles
        bool const use_caller_thread = exclusivity == caller_inclusive_k;
        if (use_caller_thread) {
#if FU_ON_WINDOWS
            // A pseudo-handle that always means "this thread"; valid because slot 0 is only ever
            // pinned from within this very call, on this very thread, and is never joined.
            pthreads_[0].handle.store(::GetCurrentThread(), std::memory_order_release);
#else
            pthreads_[0].handle.store(::pthread_self(), std::memory_order_release);
#endif
            pthreads_[0].id.store(current_thread_id(), std::memory_order_release);
#if FU_WITH_PLACE_THREADS_BY_CORE_CLASS
            pthreads_[0].apple_core_quality = domain.apple_core_quality;
#endif
        }

        // The startup sequence for the POSIX threads differs from the `flat_pool`,
        // where at start up there is a race condition to read the `pthreads_`.
        // So we mark the threads as "chilling" until the
        mood_.store(mood_t::chill_k, std::memory_order_release);

        // Initializing the thread pool can fail for all kinds of reasons, like:
        // - `EAGAIN` if we reach the `RLIMIT_NPROC` soft resource limit.
        // - `EINVAL` if an invalid attribute was specified.
        // - `EPERM` if we don't have the right permissions.
        for (thread_index_t i = use_caller_thread; i < threads; ++i) {

            // Spawn one worker. POSIX hands back a `pthread_t` and learns the kernel thread id only
            // from inside (via `gettid`); Windows hands back both a `HANDLE` and the thread id at once,
            // so the parent can publish the id here and let the worker match on it.
            bool created = false;
#if FU_ON_WINDOWS
            DWORD new_thread_id = 0;
            HANDLE const new_handle = ::CreateThread(nullptr, 0, &_win_worker_loop, this, 0, &new_thread_id);
            created = new_handle != nullptr;
            pthreads_[i].handle.store(new_handle, std::memory_order_relaxed);
            pthreads_[i].id.store(static_cast<std::uint64_t>(new_thread_id), std::memory_order_relaxed);
#else
            pthread_t new_pthread_handle;
            pthread_attr_t attributes;
            ::pthread_attr_init(&attributes);
#if FU_WITH_PLACE_THREADS_BY_CORE_CLASS
            // Apple offers no pinning; a Quality-of-Service class is the whole placement story, and
            // it must be chosen before the thread exists. On a chip with efficiency cores, `UTILITY`
            // is what confines a thread to them; on an all-performance chip the class is inert.
            ::pthread_attr_set_qos_class_np(&attributes, _qos_for_domain(domain, compute_levels), 0);
#endif
            created = ::pthread_create(&new_pthread_handle, &attributes, &_posix_worker_loop, this) == 0;
            ::pthread_attr_destroy(&attributes);
            pthreads_[i].handle.store(new_pthread_handle, std::memory_order_relaxed);
            pthreads_[i].id.store(0, std::memory_order_relaxed); // ? 0 means "not published yet"
#endif
            pthreads_[i].core_id = -1; // ? Not pinned yet
#if FU_WITH_PLACE_THREADS_BY_CORE_CLASS
            pthreads_[i].apple_core_quality = domain.apple_core_quality;
#endif

            if (!created) {
                mood_.store(mood_t::die_k, std::memory_order_release);
                for (thread_index_t j = use_caller_thread; j < i; ++j) {
                    native_thread_t const started = pthreads_[j].handle.load(std::memory_order_relaxed);
#if FU_ON_WINDOWS
                    // Workers already see `die_k` and are exiting; reap and close each to avoid a leak.
                    ::WaitForSingleObject(started, INFINITE);
                    ::CloseHandle(started);
#else
                    // Spin-loop workers see `die_k` and exit; join reaps each, mirroring the
                    // Windows path. Cancellation is inert without a cancel point, and unwanted.
                    FU_MAYBE_UNUSED_ int join_result = ::pthread_join(started, nullptr);
                    assert(join_result == 0 && "Failed to join a thread");
#endif
                }
                reset_on_failure();
                return false; // ! Thread creation failed
            }
        }

        // Compose each thread's name. Apple can only name the calling thread, so the worker applies it
        // to itself; we merely publish it here, before the workers read their cells.
        // ! `i % logical_cores_count` because a pool may hold more threads than its domain has cores.
        for (thread_index_t i = 0; i < pthreads_.size(); ++i)
            fill_thread_name(pthreads_[i].name, name_,
                             static_cast<std::size_t>(domain.first_core_id[i % domain.logical_cores_count]),
                             max_possible_cores);
        if (use_caller_thread) set_current_thread_name(pthreads_[0].name);

        // Pin all of the threads. Where the kernel refuses, the domains still describe the machine
        // and the pool still partitions work by them - it simply cannot hold a thread in place.
        // The refusals are counted rather than dropped: a pool the kernel crowded onto a handful of
        // cores spins itself to a standstill, and `all_threads_pinned` is how a caller finds out.
        unpinned_threads_ = 0;
        if (pin_granularity == pin_to_core_k) {
            for (thread_index_t i = 0; i < pthreads_.size(); ++i) {
                core_id_t const cpu = domain.first_core_id[i % domain.logical_cores_count];
                native_thread_t const pin_handle = pthreads_[i].handle.load(std::memory_order_relaxed);
                if (try_pin_thread_to_cores(pin_handle, &cpu, 1)) pthreads_[i].core_id = cpu;
                else
                    ++unpinned_threads_;
            }
        }
        else {
            for (thread_index_t i = 0; i < pthreads_.size(); ++i) {
                native_thread_t const pin_handle = pthreads_[i].handle.load(std::memory_order_relaxed);
                if (!try_pin_thread_to_cores(pin_handle, domain.first_core_id, domain.logical_cores_count))
                    ++unpinned_threads_;
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
    FU_REQUIRES_((can_be_for_thread_callback<fork_type_, index_t>()))
    broadcast_join<colocated_pool, fork_type_> for_threads(fork_type_ &&fork) noexcept {
        return {*this, std::forward<fork_type_>(fork)};
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

        pinned_threads_allocator_t pthread_allocator {allocator_};

        // Stop all threads and wait for them to finish
        mood_.store(mood_t::die_k, std::memory_order_release);

        caller_exclusivity_t const exclusivity = caller_exclusivity();
        bool const use_caller_thread = exclusivity == caller_inclusive_k;
        thread_index_t const threads = pthreads_.size();
        for (thread_index_t i = use_caller_thread; i != threads; ++i) {
            native_thread_t const join_handle = pthreads_[i].handle.load(std::memory_order_relaxed);
#if FU_ON_WINDOWS
            ::WaitForSingleObject(join_handle, INFINITE);
            ::CloseHandle(join_handle); // ? Release the reference `CreateThread` handed us
#else
            void *returned_value = nullptr;
            FU_MAYBE_UNUSED_ int const join_result = ::pthread_join(join_handle, &returned_value);
            assert(join_result == 0 && "Thread join failed");
#endif
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
#if FU_WITH_RESCHEDULE_THREADS_BY_CLASS
        bool const use_caller_thread = caller_exclusivity() == caller_inclusive_k;
        for (std::size_t i = use_caller_thread; i < pthreads_.size(); ++i) {
            std::uint64_t const pthread_id = pthreads_[i].id.load(std::memory_order_acquire);
            if (pthread_id == 0) continue; // ! Unsigned now: `< 0` could never fire
#if FU_ON_FREEBSD
            // FreeBSD rejects `SCHED_IDLE`; its idle class is reached through `rtprio` instead.
            // ! Elaborated `struct` tag: `<sys/rtprio.h>` also declares an `rtprio()` function that
            // ! would otherwise hide the type name here.
            // ? Lowest priority within the idle class
            struct ::rtprio rtp {RTP_PRIO_IDLE, RTP_PRIO_MAX};
            ::rtprio_thread(RTP_SET, static_cast<lwpid_t>(pthread_id), &rtp);
#else
            sched_param param {};
            ::sched_setscheduler(static_cast<pid_t>(pthread_id), SCHED_IDLE, &param);
#endif
        }
#endif // ? No idle scheduling class on Darwin or Windows
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

#pragma region Advanced

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
        bool const was_chilling = mood_.compare_exchange_weak( //
            may_be_chilling, mood_t::grind_k,                  //
            std::memory_order_relaxed, std::memory_order_relaxed);
        generation_t const generation = static_cast<generation_t>(epoch_.fetch_add(1, std::memory_order_release) + 1);

        // If the workers were indeed "chilling", we can inform the scheduler to wake them up.
#if FU_WITH_RESCHEDULE_THREADS_BY_CLASS
        if (was_chilling) {
            bool const use_caller_thread = caller_exclusivity() == caller_inclusive_k;
            for (std::size_t i = use_caller_thread; i < pthreads_.size(); ++i) {
                std::uint64_t const pthread_id = pthreads_[i].id.load(std::memory_order_acquire);
                if (pthread_id == 0) continue; // ! Unsigned now: `< 0` could never fire
                // Nudge the sleeping worker back onto a runnable class. Darwin has no equivalent
                // for another thread; its QoS class is fixed at creation.
#if FU_ON_FREEBSD
                // Restore the timesharing class - "make runnable", not "boost to realtime".
                struct ::rtprio rtp {RTP_PRIO_NORMAL, 0};
                ::rtprio_thread(RTP_SET, static_cast<lwpid_t>(pthread_id), &rtp);
#else
                sched_param param {};
                ::sched_setscheduler(static_cast<pid_t>(pthread_id), SCHED_FIFO | SCHED_RR, &param);
#endif
            }
        }
#else
        fu_unused_(was_chilling); // ? No runnable-class nudge on Darwin or Windows
#endif
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

        // Wait for the last contributor's completion increment. Only `epoch_` moves here, so a
        // monitored waiter arms exactly the right line and the completing store wakes it as an event.
        micro_yield_t micro_yield;
        while (epoch_.load(std::memory_order_acquire) == generation)
            micro_yield(epoch_, generation, static_cast<thread_index_t>(0), wait_uncapped_k);
    }

    /** @brief Blocks the calling thread until the currently broadcasted task finishes. */
    void unsafe_join() noexcept {
        epoch_index_t const current_epoch = epoch_.load(std::memory_order_acquire);
        if (current_epoch & 1u) unsafe_join(static_cast<generation_t>(current_epoch)); // ? Even means idle
    }

#pragma endregion Advanced

#pragma region ComputeDomains Compatibility

    /**
     *  @brief Number of individual sub-pool with the same NUMA-locality and QoS.
     *  @retval 1 constant for compatibility.
     */
    constexpr index_t compute_domains_count() const noexcept { return 1; }

    /**
     *  @brief Returns the number of threads in one NUMA-specific local @b compute_domain.
     *  @retval Same value as `threads_count()`, as we only support one compute_domain.
     *  @note Shape parity with `distributed_pool`: generic callers - the C ABI's `visit` and the
     *      distributed invokers - call `pool.threads_count(domain)` on every pool kind.
     */
    thread_index_t threads_count(FU_MAYBE_UNUSED_ index_t compute_domain_index) const noexcept {
        assert(compute_domain_index == 0 && "Only one compute_domain is supported");
        return threads_count();
    }

    /**
     *  @brief Converts a @p `global_thread_index` to a local thread index within a @b compute_domain.
     *  @retval Same value as @p `global_thread_index`, as we only support one compute_domain.
     */
    constexpr thread_index_t thread_local_index(thread_index_t global_thread_index,
                                                FU_MAYBE_UNUSED_ index_t compute_domain_index = 0) const noexcept {
        assert(compute_domain_index == 0 && "Only one compute_domain is supported");
        return global_thread_index;
    }

#pragma endregion ComputeDomains Compatibility

  private:
    /** @brief Clears the fork state and trampoline between dispatches. */
    void _reset_fork() noexcept {
        fork_state_ = nullptr;
        fork_trampoline_ = nullptr;
    }

    /** @brief Restores the caller's CPU affinity to its pre-spawn snapshot. @sa `try_restore_thread_cores`. */
    void _reset_affinity() noexcept {
        try_restore_thread_cores(caller_affinity_);
        caller_affinity_.reset();
    }

    /**
     *  @brief A trampoline function that is used to call the user-defined lambda.
     *  @param[in] punned_lambda_pointer The pointer to the user-defined lambda.
     *  @param[in] local_thread The thread index paired with this pool's compute domain.
     */
    template <typename fork_type_>
    static void _call_as_lambda(punned_fork_context_t punned_lambda_pointer, local_thread_t local_thread) noexcept {
        fork_type_ &lambda_object = *static_cast<fork_type_ *>(punned_lambda_pointer);
        lambda_object(local_thread);
    }

    /**
     *  @brief The worker's run loop, shared by every platform's thread entry point.
     *  @note POSIX and Windows entry points differ only in signature, so both forward to this body.
     */
    static void _worker_loop_body(colocated_pool *pool) noexcept {

        // Following section untile the main `while` loop may introduce race conditions,
        // so spin-loop for a bit until the pool is ready.
        mood_t mood;
        micro_yield_t micro_yield;
        // Only `mood_` moves here, so a monitored waiter arms it and wakes on the store out of `chill_k`.
        while ((mood = pool->mood_.load(std::memory_order_acquire)) == mood_t::chill_k)
            // Technically, we are not on the zero thread index, but we don't know our index yet.
            micro_yield(pool->mood_, mood_t::chill_k, static_cast<thread_index_t>(0), wait_uncapped_k);

        // If we are ready to start grinding, export this threads metadata to make it externally
        // observable and controllable.
        thread_index_t local_thread_index = 0;
        if (mood == mood_t::grind_k) {
            // Find our own slot in the `pthreads_` array. POSIX matches on the `pthread_t` the parent
            // stored; Windows matches on the thread id the parent published at creation - a `HANDLE`
            // is not reliable identity, since one thread may own several.
            auto &numa_pthreads = pool->pthreads_;
            thread_index_t const numa_pthreads_count = pool->pthreads_.size();
#if FU_ON_WINDOWS
            std::uint64_t const self_id = current_thread_id();
            for (local_thread_index = 0; local_thread_index < numa_pthreads_count; ++local_thread_index)
                if (numa_pthreads[local_thread_index].id.load(std::memory_order_acquire) == self_id) break;
#else
            pthread_t const thread_handle = ::pthread_self();
            for (local_thread_index = 0; local_thread_index < numa_pthreads_count; ++local_thread_index)
                if (::pthread_equal(numa_pthreads[local_thread_index].handle.load(std::memory_order_relaxed),
                                    thread_handle))
                    break;
#endif
            assert(local_thread_index < numa_pthreads_count && "Thread index must be in [0, threads_count)");

            // Publish the kernel thread id to shared memory. On Windows it already holds this value;
            // re-storing it is harmless and keeps the release-publish uniform across platforms.
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
            // Two independent lines guard this loop - arm the hot one (`epoch_`, bumped by a dispatch)
            // and let the waiter's timeout cap bound how late a rare `mood_` change is noticed.
            while ((new_epoch = pool->epoch_.load(std::memory_order_acquire)) == last_epoch &&
                   (mood = pool->mood_.load(std::memory_order_acquire)) == mood_t::grind_k)
                micro_yield(pool->epoch_, last_epoch, global_thread_index);

            if (fu_unlikely_(mood == mood_t::die_k)) break;
            if (fu_unlikely_(mood == mood_t::chill_k) && (new_epoch == last_epoch)) {
                sleep_for_micros(pool->sleep_length_micros_);
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
    }

    // Platform thread entry points: same body, different ABI. Only the one this build spawns exists.
#if FU_ON_WINDOWS
    /** @brief Windows thread entry point; forwards to `_worker_loop_body`. */
    static DWORD WINAPI _win_worker_loop(LPVOID arg) noexcept {
        _worker_loop_body(static_cast<colocated_pool *>(arg));
        return 0;
    }
#else
    /** @brief POSIX thread entry point; forwards to `_worker_loop_body`. */
    static void *_posix_worker_loop(void *arg) noexcept {
        _worker_loop_body(static_cast<colocated_pool *>(arg));
        return nullptr;
    }
#endif

#if FU_WITH_PLACE_THREADS_BY_CORE_CLASS
    /**
     *  @brief Maps a compute domain onto the only placement control Darwin offers: a QoS class.
     *  @param[in] levels Distinct levels the machine reports; 1 means every core is interchangeable.
     *
     *  The fastest tier present runs the hot path at `USER_INITIATED` - not `USER_INTERACTIVE`, which
     *  is reserved for work a person is waiting on. Below it, only a tier the OS names "Efficiency"
     *  takes `UTILITY`, the class that confines threads to E-cores. The rank alone cannot decide that:
     *  the bottom rank is E-cores on an A18 but big cores on an M5 Pro, where `UTILITY` would banish
     *  two thirds of the machine. Checking rank first keeps an all-efficiency chip honest - its E-cores
     *  are the fastest thing present. A hidden or unknown name parses to `apple_performance_k`, so the
     *  fallback is rank-only and `UTILITY` is never guessed.
     */
    static qos_class_t _qos_for_domain(compute_domain_t const &domain, std::size_t const levels) noexcept {
        if (domain.compute_level + 1 >= levels) return QOS_CLASS_USER_INITIATED; // ? The fastest tier present
        if (domain.apple_core_quality == apple_efficiency_k)
            return QOS_CLASS_UTILITY; // ? Genuine E-cores - confine here
        return QOS_CLASS_DEFAULT;     // ? A slower big tier stays big
    }
#endif

    /** @brief Composes a worker's thread name into @p output_name, truncating the base and zero-padding the index. */
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

#pragma endregion Colocated Pool

#pragma region Distributed Pool

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
 *  The same protocol as `invoke_for_n_dynamic`, applied at two levels. Let's say we receive N tasks
 *  for T threads across C compute_domains. Each compute_domain takes (N/C) tasks, reserves one
 *  trailing static prong per local thread, and splits the rest into one contiguous slice per local
 *  thread - so every thread drains its own slice with an @b uncontended `fetch_add` on a cursor line
 *  nobody else touches. A drained thread first helps its same-domain neighbours, walking them in a
 *  coprime order, and only when the whole domain runs dry does it cross the interconnect - walking
 *  the other domains in a coprime order, and each domain's threads in a coprime order again.
 *
 *  Tasks are still claimed one at a time, so the makespan guarantee of greedy list scheduling
 *  survives; a cursor line is only ever shared once a thread actually runs dry, which is exactly
 *  when the extra line transfer is worth paying.
 *
 *  @section Overflow Considerations, One Level Up
 *
 *  The flat invoker's proof balances a slice's trailing reservation against its visitor count -
 *  both equal `threads` there, so no cursor passes `max(n, threads)`. Here the reservation is per
 *  domain (`threads_local` static prongs) while helping is pool-wide, so a slice may be visited by
 *  all T threads and its cursor may settle past `n` - by less than `T` increments. No out-of-range
 *  task is ever dispatched (`task >= end` guards every claim), and the read-only probe in
 *  `drain_claim` drives the drained-slice regime to zero overshoot, so wrapping would need `n`
 *  within a core-count of the index type's maximum.
 */
template <typename pool_type_, typename fork_type_, typename index_type_>
class invoke_distributed_for_n_dynamic {

    pool_type_ &pool_;
    fork_type_ fork_;
    index_type_ n_;

    /** @brief Where one domain's tasks live and how they split across its threads. Computed the
     *      same way by `reset_slices_` (to publish cursors) and `operator()` (to place the
     *      static prong), so the two can never drift. */
    struct domain_layout_t {
        /** @brief This domain's task span inside `[0, n_)`. */
        indexed_range<index_type_> range;
        /** @brief Workers pinned here; 0 for a domain the spawn skipped. */
        index_type_ threads;
        /** @brief Global index of this domain's first worker. */
        index_type_ first_thread;
        /** @brief Leading cursor-drained tasks; the trailing `threads` are static prongs. */
        index_type_ dynamic;
    };

    domain_layout_t layout_of_(indexed_range<index_type_> const range, index_type_ const domain) const noexcept {
        index_type_ const threads = pool_.threads_count(domain);
        index_type_ const dynamic = range.count > threads ? static_cast<index_type_>(range.count - threads) : 0;
        return {range, threads, pool_.first_thread(domain), dynamic};
    }

    /** @brief Helps every thread of @p compute_domain, in a coprime order seeded by the caller. */
    void drain_domain_(index_type_ const compute_domain, index_type_ const thread,
                       local_prong<index_type_> &prong) noexcept {
        index_type_ const threads_local = pool_.threads_count(compute_domain);
        if (!threads_local) return; // ? A domain the spawn left empty owns no slices
        index_type_ const first_thread = pool_.first_thread(compute_domain);
        coprime_permutation_range<index_type_> victims(first_thread, threads_local, thread);
        for (auto victim = victims.begin(); victim != default_sentinel_t {}; ++victim)
            if (*victim != thread) drain_claim(pool_, *victim, prong, fork_);
    }

  public:
    invoke_distributed_for_n_dynamic(pool_type_ &pool, index_type_ n, fork_type_ &&fork) noexcept
        : pool_(pool), fork_(std::forward<fork_type_>(fork)), n_(n) {
        reset_slices_();
    }

    void operator()(index_type_ const thread) noexcept {
        index_type_ const compute_domains_count = pool_.compute_domains_count();
        assert(compute_domains_count > 0 && "There must be at least one compute_domain");

        // The prong's compute_domain is INVARIANT across every steal: it names the domain THIS
        // thread is pinned to, never the victim's - so a stolen task still reads the thief's
        // node-local replica; replicas are identical, only their distances differ. The drain
        // helpers mutate `.task` only.
        index_type_ const native_compute_domain = pool_.thread_compute_domain(thread);
        local_prong<index_type_> prong(0, thread, native_compute_domain);

        // Run (up to) one static prong from the native domain's trailing reservation.
        indexed_split<index_type_> const split_between_compute_domains(n_, compute_domains_count);
        domain_layout_t const home = layout_of_(split_between_compute_domains[native_compute_domain], //
                                                native_compute_domain);
        index_type_ const local = pool_.thread_local_index(thread, native_compute_domain);
        index_type_ const static_index = static_cast<index_type_>(home.dynamic + local);
        if (static_index < home.range.count) { // ? Fewer tasks than threads leaves the domain's tail idle
            prong.task = static_cast<index_type_>(home.range.first + static_index);
            fork_(prong);
        }

        // Home domain first: our own slice (still uncontended), then the same-domain neighbours -
        // the `!= thread` guard inside drain_domain_ keeps us from re-draining the slice we just
        // finished.
        drain_claim(pool_, thread, prong, fork_);
        drain_domain_(native_compute_domain, thread, prong);

        // Only once the whole home domain is dry do we cross the interconnect, coprime over the
        // domains (the `!= native` guard skips the one we already drained) and coprime over each
        // domain's threads. This guard cannot dissolve the way the flat invoker's did: both the
        // walk's stride and its start derive from `seed % length`, so seeding every thread to start
        // at `native` would also hand every thread the same stride - the very stampede onto one
        // remote domain the coprime order exists to prevent.
        coprime_permutation_range<index_type_> other_domains(0, compute_domains_count, thread);
        for (auto domain = other_domains.begin(); domain != default_sentinel_t {}; ++domain)
            if (*domain != native_compute_domain) drain_domain_(*domain, thread, prong);
    }

  private:
    /** @brief Publishes one contiguous slice per thread in every domain. Runs before the broadcast. */
    void reset_slices_() noexcept {
        typename pool_type_::cache_hints_t cache_hints;
        index_type_ const compute_domains_count = pool_.compute_domains_count();
        indexed_split<index_type_> const split_between_compute_domains(n_, compute_domains_count);
        for (index_type_ domain = 0; domain < compute_domains_count; ++domain) {
            domain_layout_t const layout = layout_of_(split_between_compute_domains[domain], domain);
            if (!layout.threads) continue; // ? A domain the spawn left empty owns no slices

            indexed_split<index_type_> const split_local(layout.dynamic, layout.threads);
            for (index_type_ local = 0; local < layout.threads; ++local) {
                indexed_range<index_type_> const slice = split_local[local];
                auto &claim = pool_.unsafe_dynamic_claim_ref(static_cast<index_type_>(layout.first_thread + local));
                claim.end = static_cast<index_type_>(layout.range.first + slice.first + slice.count);
                claim.next.store(static_cast<index_type_>(layout.range.first + slice.first), std::memory_order_release);
                cache_hints(&claim, demote_line_k); // ? Publish away, so each owner's first claim skips this core
            }
        }
    }
};

/**
 *  @brief A Linux-only pool over all distributed "thread compute_domains", NUMA nodes, and QoS levels.
 *
 *  Differs from the `flat_pool` template in the following ways:
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
template <typename micro_yield_type_ = standard_yield_t, typename cache_hints_type_ = standard_cache_hints_t,
          std::size_t alignment_ = default_alignment_k>
struct distributed_pool {

    using colocated_pool_t = colocated_pool<micro_yield_type_, cache_hints_type_, alignment_>;
    using machine_topology_t = machine_topology<>;
    static constexpr pool_kind_t kind_k = pool_kind_t::distributed_k;

    /** @brief Same allocator as the sub-pools: domain-placing where the kernel can honour it. */
    using allocator_t = typename colocated_pool_t::allocator_t;

    using micro_yield_t = typename colocated_pool_t::micro_yield_t;
    using cache_hints_t = typename colocated_pool_t::cache_hints_t;
    using index_t = typename colocated_pool_t::index_t;
    using epoch_index_t = typename colocated_pool_t::epoch_index_t;
    using generation_t = epoch_index_t;
    using thread_index_t = typename colocated_pool_t::thread_index_t;
    static constexpr std::size_t alignment_k = colocated_pool_t::alignment_k;
    using prong_t = local_prong<index_t>;

  private:
    using colocations_t = dynamic_padded_array<colocated_pool_t, allocator_t>;

    /** @brief Thread name buffer, forwarded to each sub-pool for OS thread naming. */
    char name_[16] {};
    /** @brief Total threads across all compute domains, including the caller on inclusive pools. */
    thread_index_t threads_count_ {0};
    /** @brief Whether the caller thread is counted as one of the contributors. */
    caller_exclusivity_t exclusivity_ {caller_inclusive_k};
    /**
     *  @brief One pinned sub-pool per compute domain, in one flat contiguous array.
     *
     *  The array lives on the first domain's node, and that is enough: a worker spinning on its
     *  pool's `mood_` / `epoch_` holds a Shared copy in its own cache, and the `threads_to_sync_`
     *  line ping-pongs between its sharers, not their directory home - per-domain placement of
     *  these signal words measured as pure noise on a 2-socket machine. The hot per-task state -
     *  every worker's `dynamic_claim` cursor inside `pthreads_` - is node-local regardless, since
     *  each `colocated_pool` allocates it with its own domain allocator in `try_spawn`. Entries
     *  are sorted by compute-domain index, and the first one always contains the current thread.
     */
    colocations_t colocations_ {};

  public:
    distributed_pool(distributed_pool &&) = delete;
    distributed_pool(distributed_pool const &) = delete;
    distributed_pool &operator=(distributed_pool &&) = delete;
    distributed_pool &operator=(distributed_pool const &) = delete;

    distributed_pool() noexcept : distributed_pool("forkunion") {}

    explicit distributed_pool(char const *name) noexcept {
        // Accept null or empty names by falling back to a sensible default
        char const *effective_name = (name && name[0] != '\0') ? name : "forkunion";
        std::size_t const source_length = std::strlen(effective_name);
        std::size_t const name_length = source_length < sizeof(name_) ? source_length : sizeof(name_) - 1;
        std::memcpy(name_, effective_name, name_length);
        name_[name_length] = '\0';
    }

    ~distributed_pool() noexcept { terminate(); }

    /**
     *  @brief Estimates the amount of memory managed by this pool handle and internal structures.
     *  @note This API is @b not synchronized.
     */
    std::size_t memory_usage() const noexcept {
        std::size_t total_bytes = sizeof(distributed_pool);
        for (index_t i = 0; i < colocations_.size(); ++i) total_bytes += colocations_[i].memory_usage();
        return total_bytes;
    }

    /**
     *  @brief Checks if the thread-pool's core synchronization points are lock-free.
     *  @note Only valid after the `try_spawn` call.
     */
    bool is_lock_free() const noexcept { return colocations_ && colocations_[0].is_lock_free(); }

    /** @brief Global index of the first worker in @p compute_domain; workers are numbered contiguously. */
    thread_index_t first_thread(index_t compute_domain) const noexcept {
        assert(compute_domain < colocations_.size() && "Compute domain index out of bounds");
        return colocations_[compute_domain].first_thread();
    }

    /** @brief Exposes one worker's private claim cursor, drained by `invoke_distributed_for_n_dynamic`. */
    dynamic_claim<index_t> &unsafe_dynamic_claim_ref(thread_index_t const thread) noexcept {
        index_t const compute_domain = thread_compute_domain(thread);
        return colocations_[compute_domain].unsafe_dynamic_claim_ref(thread_local_index(thread, compute_domain));
    }

#pragma region Core API

    /**
     *  @brief Returns the number of threads in the thread-pool, including the main thread.
     *  @retval 0 if the thread-pool is not initialized, 1 if only the main thread is used.
     *  @note This API is @b not synchronized.
     */
    thread_index_t threads_count() const noexcept { return threads_count_; }

    /** @brief Workers the kernel refused to place, summed over every compute domain. */
    thread_index_t unpinned_threads_count() const noexcept {
        thread_index_t unpinned = 0;
        for (index_t i = 0; i < colocations_.size(); ++i) unpinned += colocations_[i].unpinned_threads_count();
        return unpinned;
    }

    /**
     *  @brief Whether every worker sits on the core this pool asked for.
     *  @note False on platforms with no thread placement, and false when a `cpuset` crowded the pool
     *      onto fewer cores than it has threads - which is where spinning workers fall apart.
     */
    bool all_threads_pinned() const noexcept { return threads_count_ != 0 && unpinned_threads_count() == 0; }

    /**
     *  @brief Reports if the current calling thread will be used for broadcasts.
     *  @note This API is @b not synchronized.
     */
    caller_exclusivity_t caller_exclusivity() const noexcept { return exclusivity_; }

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
        machine_topology_t const &topology, caller_exclusivity_t const exclusivity = caller_inclusive_k,
        pin_granularity_t const pin_granularity = pin_to_core_k) noexcept {
        return try_spawn(topology, topology.logical_cores_count(), exclusivity, pin_granularity);
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
        machine_topology_t const &topology,
        thread_index_t const threads, //
        caller_exclusivity_t const exclusivity = caller_inclusive_k,
        pin_granularity_t const pin_granularity = pin_to_core_k) noexcept {

        if (threads == 0) return false;        // ! Can't have zero threads working on something
        if (threads_count_ != 0) return false; // ! Already initialized

        // The topology is borrowed for the duration of this call only - every sub-pool below
        // captures what it needs by value, so the pool never retains a reference to it.
        // Place the array itself on the first compute domain, pinning the caller there too.
        // We spawn one sub-pool per compute domain (a same-QoS core run), not per NUMA node, so
        // performance and efficiency cores on one node become separate, independently pinned pools.
        compute_domain_t const &first_domain = topology.compute_domain_at(compute_domain_index_t {});
        allocator_t allocator = allocator_for_node(first_domain.memory_domain_id);
        index_t const colocations_count = std::min(topology.compute_domains_count(), threads);

        colocations_t colocations(allocator);
        if (!colocations.try_resize(colocations_count)) return false; // ! Allocation failed
        for (index_t compute_domain_index = 0; compute_domain_index < colocations_count; ++compute_domain_index)
            colocations[compute_domain_index].rename(name_);

        auto reset_on_failure = [&]() noexcept {
            for (index_t compute_domain_index = 0; compute_domain_index < colocations_count; ++compute_domain_index)
                colocations[compute_domain_index].terminate(); // ? A no-op on the pools not yet spawned
        };

        // Every compute-domain pool is spawned separately
        // - the first one may be "inclusive".
        // - others are always "exclusive" to the caller thread.
        indexed_split<thread_index_t> threads_per_domain(threads, colocations_count);
        index_t const compute_levels = static_cast<index_t>(topology.compute_levels_count());
        if (!colocations[0].try_spawn(first_domain, threads_per_domain[0].count, exclusivity, //
                                      pin_granularity, 0, 0, compute_levels)) {
            reset_on_failure();
            return false; // ! Spawning failed
        }

        for (index_t compute_domain_index = 1; compute_domain_index < colocations_count; ++compute_domain_index) {
            compute_domain_t const &domain =
                topology.compute_domain_at(static_cast<compute_domain_index_t>(compute_domain_index));
            if (!colocations[compute_domain_index].try_spawn(
                    domain, threads_per_domain[compute_domain_index].count, caller_exclusive_k, pin_granularity,
                    threads_per_domain[compute_domain_index].first, compute_domain_index, compute_levels)) {
                reset_on_failure();
                return false; // ! Spawning failed
            }
        }

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
    broadcast_join<distributed_pool, fork_type_> for_threads(fork_type_ &&fork) noexcept {
        return {*this, std::forward<fork_type_>(fork)};
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
        for (index_t i = 0; i < colocations_.size(); ++i) colocations_[i].terminate();

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
        for (index_t i = 0; i < colocations_.size(); ++i) colocations_[i].sleep(wake_up_periodicity_micros);
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

#pragma region Advanced

    /**
     *  @brief Executes a @p fork function in parallel on all threads, not waiting for the result.
     *  @param[in] fork The callback @b reference, receiving the thread index as an argument.
     *  @return A `generation_t` token identifying this dispatch.
     *  @sa Use in conjunction with `unsafe_join`.
     */
    template <typename fork_type_>
    FU_REQUIRES_((can_be_for_thread_callback<fork_type_, index_t>()))
    generation_t unsafe_for_threads(fork_type_ &fork) noexcept {
        assert(colocations_ && "Thread pools must be initialized before broadcasting");

        // Submit to every thread pool. All sub-pool epochs advance in lockstep as long as
        // every dispatch goes through this wrapper - never dispatch to a sub-pool directly.
        generation_t last_sub_generation {};
        for (std::size_t i = 1; i < colocations_.size(); ++i)
            last_sub_generation = colocations_[i].unsafe_for_threads(fork);
        generation_t const generation = colocations_[0].unsafe_for_threads(fork);
        assert((colocations_.size() == 1 || last_sub_generation == generation) &&
               "ComputeDomain sub-pools must advance in generation lockstep");
        (void)last_sub_generation;
        return generation;
    }

    /**
     *  @brief Returns true if the generation identified by @p generation has completed on all sub-pools.
     *  @note A `true` result synchronizes with all contributors: their writes are visible.
     *
     *  On `caller_inclusive_k` pools this can only turn `true` once `unsafe_join`
     *  contributes the calling thread's slice, so the poll-then-join pattern is
     *  reserved for `caller_exclusive_k` pools.
     */
    bool is_complete(generation_t generation) const noexcept {
        for (index_t i = 0; i < colocations_.size(); ++i)
            if (!colocations_[i].is_complete(generation)) return false;
        return true;
    }

    /**
     *  @brief Blocks the calling thread until the generation identified by @p generation finishes.
     *  @note On `caller_inclusive_k` pools, first executes the calling thread's slice.
     *  Idempotent: returns immediately for already-joined or stale generations.
     */
    void unsafe_join(generation_t generation) noexcept {
        assert(colocations_ && "Thread pools must be initialized before broadcasting");

        // Join the caller-hosting compute_domain first: on inclusive pools its slice runs here
        // and overlaps the remote sub-pools' completion instead of waiting behind them.
        colocations_[0].unsafe_join(generation);
        for (std::size_t i = 1; i < colocations_.size(); ++i) colocations_[i].unsafe_join(generation);
    }

    /** @brief Blocks the calling thread until the currently broadcasted task finishes. */
    void unsafe_join() noexcept {
        assert(colocations_ && "Thread pools must be initialized before broadcasting");

        // Wait for everyone to finish, starting from the caller-hosting compute_domain
        colocations_[0].unsafe_join();
        for (std::size_t i = 1; i < colocations_.size(); ++i) colocations_[i].unsafe_join();
    }

#pragma endregion Advanced

#pragma region ComputeDomains Compatibility

    /**
     *  @brief Number of compute domains this pool spans (one pinned sub-pool each).
     */
    index_t compute_domains_count() const noexcept { return colocations_.size(); }

    /**
     *  @brief Returns the number of threads in one NUMA-specific local @b compute_domain.
     *  @retval 0 if the thread-pool is not initialized, 1 if only the main thread is used.
     *  @note This API is @b not synchronized and doesn't check for out-of-bounds access.
     */
    thread_index_t threads_count(index_t compute_domain) const noexcept {
        assert(colocations_ && "Local pools must be initialized");
        assert(compute_domain < colocations_.size() && "Local pool index out of bounds");
        return colocations_[compute_domain].threads_count();
    }

    /**
     *  @brief Converts a @p `global_thread_index` to a local thread index within a @b compute_domain.
     *  @retval 0 if the thread-pool is not initialized, 1 if only the main thread is used.
     *  @note This API is @b not synchronized and doesn't check for out-of-bounds access.
     */
    thread_index_t thread_local_index(thread_index_t global_thread_index, index_t compute_domain) const noexcept {
        assert(colocations_ && "Local pools must be initialized");
        assert(compute_domain < colocations_.size() && "Local pool index out of bounds");
        return global_thread_index - colocations_[compute_domain].first_thread();
    }

    /** @brief Returns the compute domain index owning @p global_thread_index, or the count if none. */
    index_t thread_compute_domain(thread_index_t global_thread_index) const noexcept {
        index_t compute_domain_index = 0;
        for (; compute_domain_index < colocations_.size(); ++compute_domain_index) {
            colocated_pool_t const &colocation = colocations_[compute_domain_index];
            if (global_thread_index < colocation.first_thread()) continue;
            if (global_thread_index < colocation.first_thread() + colocation.threads_count())
                return compute_domain_index;
        }
        return compute_domain_index; // ? Not found
    }

#pragma endregion ComputeDomains Compatibility

  private:
    /**
     *  @brief An allocator that places bytes on @p memory_domain_id, where the kernel can honour that.
     *  @note On a machine without NUMA memory the node is meaningless and the argument is dropped.
     */
    static allocator_t allocator_for_node(FU_MAYBE_UNUSED_ memory_domain_id_t const memory_domain_id) noexcept {
#if FU_WITH_PLACE_MEMORY_ON_DOMAIN
        return allocator_t {memory_domain_id};
#else
        return allocator_t {};
#endif
    }
};

#pragma region Measured Memory Distances

/**
 *  @brief One recorded observation of a memory-fabric edge: what @p initiator's cores see when
 *      reaching memory resident on @p target.
 *
 *  The chase and the stream are separate experiments, so each record carries the metric its
 *  experiment produced, 0 for the other; `measured_fabric` folds the log into per-metric envelopes.
 */
struct measured_edge_t {
    /** @brief The compute domain whose cores issued the loads - the pool's pinning unit, never a
     *      memory domain: sibling domains on one controller still measure apart. */
    compute_domain_index_t initiator {};
    /** @brief The memory domain whose DRAM answered the loads. */
    memory_domain_index_t target {};
    /** @brief Dependent-load latency of a lone core, in nanoseconds; 0 if unobserved. */
    std::size_t nanoseconds {0};
    /** @brief Saturated read bandwidth of ALL the initiator's cores at once, in MB/s; 0 if unobserved. */
    std::size_t megabytes_per_second {0};
};

/** @brief One hop of the latency walk, padded to a cache line so consecutive slots never share one. */
struct alignas(64) chase_slot_t {
    /** @brief Index of the slot the walk visits next; no initializer, as `try_resize_uninitialized`
     *      demands trivial construction and `thread_chase_list_` writes every slot anyway. */
    std::uint32_t next_slot_index;
};

/**
 *  @brief Threads a single Sattolo cycle through @p slots - a linked list where every load's address
 *      depends on the previous load, so prefetchers see noise and the walk pays true latency.
 *  @note Writing the links is also the FIRST touch of every page, which is what places the list on
 *      the toucher's memory domain on every OS - no `mbind`, no `libnuma`, no ACPI.
 */
inline void thread_chase_list_(chase_slot_t *slots, std::size_t const count) noexcept {
    for (std::size_t i = 0; i != count; ++i) slots[i].next_slot_index = static_cast<std::uint32_t>(i);
    for (std::size_t i = count - 1; i != 0; --i) { // ? Sattolo: swap below self, never with self
        std::size_t const j = split_mix(i) % i;
        std::uint32_t const swapped = slots[i].next_slot_index;
        slots[i].next_slot_index = slots[j].next_slot_index;
        slots[j].next_slot_index = swapped;
    }
}

/**
 *  @brief Walks dependent loads through the list in @p segments timed stretches, returning the
 *      @b fastest stretch's nanoseconds per hop - the least-contended glimpse of the fabric.
 *  @note Each stretch continues where the last stopped, and the single Sattolo cycle revisits no
 *      slot until it closes, so every stretch walks lines no stretch has cached before.
 */
inline std::size_t chase_ns_per_hop_(chase_slot_t const *slots, std::size_t const hops,
                                     std::size_t const segments) noexcept {
    std::uint32_t position = 0;
    std::size_t best_nanoseconds = ~std::size_t(0);
    for (std::size_t segment = 0; segment != segments; ++segment) {
        auto const started = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i != hops; ++i) position = slots[position].next_slot_index;
        auto const elapsed = std::chrono::steady_clock::now() - started;
        std::size_t const nanoseconds =
            static_cast<std::size_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
        best_nanoseconds = (std::min)(best_nanoseconds, nanoseconds);
    }
    return best_nanoseconds / hops + (position == ~0u); // ? The data-dependent tail defeats elision
}

/** @brief Runs @p work on the one pool worker with the given global @p thread index; the other
 *      workers pass through the broadcast untouched. */
template <typename pool_type_, typename work_type_>
static void run_on_worker_(pool_type_ &pool, std::size_t const thread, work_type_ &&work) noexcept {
    // ? A `local_thread` argument converts to its global index, so this fits every pool's callback shape
    pool.for_threads([&](std::size_t const local_thread_index) noexcept {
        if (local_thread_index == thread) work();
    });
}

/** @brief Bytes a chase list needs, within what the @p target domain itself can hold.
 *
 *  Deliberately @b not scaled by cache size: the Sattolo cycle revisits no slot within a lap, so
 *  no cache can shortcut the walk - while every extra page costs TLB reach and risks billing each
 *  hop for a memory-bound page walk. 128 MiB fits three revisit-free million-hop stretches. */
inline std::size_t chase_list_bytes_(memory_domain_t const &target) noexcept {
    std::size_t bytes = std::size_t(128) << 20;
    if (target.volume_ram) bytes = (std::min)(bytes, target.volume_ram / 8);
    return bytes;
}

/** @brief Fills @p words with `split_mix` draws - the FIRST touch that places every page on the
 *      filling worker's memory domain, and data no reduction can constant-fold away. */
inline void fill_stream_words_(std::uint64_t *words, std::size_t const count) noexcept {
    for (std::size_t i = 0; i != count; ++i) words[i] = split_mix(i);
}

/** @brief Sums @p count words - a plain reduction the compiler is free to vectorize, since
 *      saturating the controller is exactly what a bandwidth probe wants. */
inline std::uint64_t stream_words_(std::uint64_t const *words, std::size_t const count) noexcept {
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i != count; ++i) sum += words[i];
    return sum;
}

/** @brief Bytes a stream needs to dwarf every cache the harvest can name - its repeats re-read
 *      the same buffer, and a cache-resident buffer would report the cache's bandwidth - and
 *      to run long past the fork-join overhead: at least 8 MiB per worker reading it. */
inline std::size_t stream_bytes_(machine_topology_t const &topology, memory_domain_t const &target,
                                 std::size_t const widest_domain_threads) noexcept {
    std::uint64_t largest_cache = 0;
    for (std::size_t domain = 0; domain != topology.compute_domains_count(); ++domain)
        largest_cache =
            (std::max)(largest_cache,
                       static_cast<std::uint64_t>(
                           topology.compute_domain_at(static_cast<compute_domain_index_t>(domain)).cache_bytes));
    // ? 64-bit math: a giant L3 times 8 overflows a 32-bit `size_t` before the RAM clamp can bite
    std::uint64_t bytes = (std::max)((std::max)(largest_cache * 8, std::uint64_t(128) << 20),
                                     static_cast<std::uint64_t>(widest_domain_threads) * (std::uint64_t(8) << 20));
    if (target.volume_ram) bytes = (std::min)(bytes, static_cast<std::uint64_t>(target.volume_ram) / 8);
    return static_cast<std::size_t>((std::min)(bytes, static_cast<std::uint64_t>(~std::size_t(0)) / 2));
}

/** @brief Marks a fabric position the pool has no worker on - cpuless, or beyond a partial spawn. */
inline constexpr std::size_t unreachable_position_k = ~static_cast<std::size_t>(0);

/** @brief One pool worker per memory domain, for first-touching lists onto it - placement is a
 *      property of the touching thread's position, so any local domain's worker serves. */
template <typename pool_type_>
static bool touchers_per_position_(machine_topology_t const &topology, pool_type_ &pool,
                                   dynamic_array<std::size_t> &touchers) noexcept {
    // ? The C ABI lets callers pair a pool with a topology it never spawned from - decline, don't trust
    if (pool.compute_domains_count() == 0 || pool.compute_domains_count() > topology.compute_domains_count())
        return false;
    if (!touchers.try_resize(topology.memory_domains_count())) return false;
    for (std::size_t position = 0; position != touchers.size(); ++position) touchers[position] = unreachable_position_k;
    for (std::size_t domain = 0; domain != pool.compute_domains_count(); ++domain) {
        memory_domain_index_t const position =
            topology.compute_domain_at(static_cast<compute_domain_index_t>(domain)).memory_domain_index;
        if (position >= touchers.size()) return false; // ? A foreign topology's indices prove the mismatch
        if (touchers[position] == unreachable_position_k) touchers[position] = pool.first_thread(domain);
    }
    return true;
}

/**
 *  @brief Measures the saturated read bandwidth from every initiator compute domain to one @p target
 *      memory domain, appending the observations to @p edges.
 *  @retval false when the stream cannot be allocated or an edge cannot be recorded.
 *
 *  One @p toucher-filled buffer serves every initiator: streams need no cold start, they evict
 *  themselves. Each initiator's workers read disjoint stripes inside one broadcast, wall-clocked
 *  fork to join - what a fork-join workload actually gets - best of three laps, the first doubling
 *  as warm-up. Worker checksums land in @p checksums and fold into a data-dependent tail.
 */
template <typename pool_type_, typename edges_array_type_>
static bool try_measure_bandwidth_edges_(machine_topology_t const &topology, pool_type_ &pool, std::size_t const target,
                                         std::size_t const toucher, dynamic_array<std::uint64_t> &checksums,
                                         edges_array_type_ &edges) noexcept {

    memory_domain_t const &target_domain = topology.memory_domain_at(static_cast<memory_domain_index_t>(target));
    std::size_t widest_domain_threads = 1;
    for (std::size_t domain = 0; domain != pool.compute_domains_count(); ++domain)
        widest_domain_threads = (std::max)(widest_domain_threads, static_cast<std::size_t>(pool.threads_count(domain)));

    std::size_t const words =
        (std::max)(stream_bytes_(topology, target_domain, widest_domain_threads) / sizeof(std::uint64_t),
                   widest_domain_threads);
    dynamic_array<std::uint64_t> stream;
    if (!stream.try_resize_uninitialized(words)) return false;
    run_on_worker_(pool, toucher, [&]() noexcept { fill_stream_words_(stream.data(), words); });

    for (std::size_t initiator = 0; initiator != pool.compute_domains_count(); ++initiator) {
        std::size_t const first_thread = pool.first_thread(initiator);
        std::size_t const domain_threads = pool.threads_count(initiator);
        indexed_split<std::size_t> const stripes(words, domain_threads);

        std::size_t best_megabytes_per_second = 0;
        for (std::size_t repeat = 0; repeat != 3; ++repeat) {
            auto const started = std::chrono::steady_clock::now();
            pool.for_threads([&](std::size_t const thread) noexcept {
                if (thread - first_thread >= domain_threads) return; // ? Another domain's worker sits out
                auto const stripe = stripes[thread - first_thread];
                checksums[thread] = stream_words_(stream.data() + stripe.first, stripe.count);
            });
            auto const elapsed = std::chrono::steady_clock::now() - started;
            std::uint64_t const nanoseconds =
                (std::max)(static_cast<std::uint64_t>(
                               std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()),
                           std::uint64_t(1));
            std::uint64_t folded = 0;
            for (std::size_t thread = 0; thread != domain_threads; ++thread) folded ^= checksums[first_thread + thread];
            std::size_t const megabytes_per_second =
                static_cast<std::size_t>(static_cast<std::uint64_t>(words) * sizeof(std::uint64_t) * 1000u /
                                         nanoseconds) +
                (folded == 0x5Fu); // ? The data-dependent tail defeats elision
            best_megabytes_per_second = (std::max)(best_megabytes_per_second, megabytes_per_second);
        }

        // ? A bandwidth-only observation: the streaming experiment says nothing about latency
        measured_edge_t const edge {static_cast<compute_domain_index_t>(initiator),
                                    static_cast<memory_domain_index_t>(target), 0, best_megabytes_per_second};
        if (!edges.try_push_back(edge)) return false;
    }
    return true;
}

/**
 *  @brief Derives dense memory-tier ordinals from an edge log, writing one rank per memory domain
 *      into @p levels and returning the number of distinct tiers (>= 1).
 *  @param scratch Caller-provided workspace of `4 * memory_domains_count` entries.
 *
 *  A tier is a property of the MEDIUM, independent of any initiator: each target is keyed by the
 *  best bandwidth any initiator sustained to it, ties split by the best latency - a 3 TB/s HBM
 *  pool outranks DDR even at equal latency. Targets cluster greedily along the sorted keys: a new
 *  tier opens where bandwidth trails its tier's anchor by over 1.25x, or, within one bandwidth
 *  band, where latency trails by over 1.5x - each band above its probe's jitter under load, below
 *  every real gap. Unobserved targets share one tier past the slowest observed.
 */
FU_MAYBE_UNUSED_ static std::size_t derive_memory_levels_(measured_edge_t const *edges, std::size_t const edges_count,
                                                          std::size_t *levels, std::size_t const memory_domains_count,
                                                          std::size_t *scratch) noexcept {
    for (std::size_t i = 0; i != memory_domains_count; ++i) levels[i] = 0;
    if (memory_domains_count == 0) return 1;

    // ? Four scratch regions: the two per-target envelope keys, the sort order, the tier buckets
    std::size_t *const bandwidths = scratch;
    std::size_t *const latencies = scratch + memory_domains_count;
    std::size_t *const order = scratch + memory_domains_count * 2;
    std::size_t *const buckets = scratch + memory_domains_count * 3;

    for (std::size_t i = 0; i != memory_domains_count; ++i) bandwidths[i] = 0, latencies[i] = 0, order[i] = i;
    for (std::size_t i = 0; i != edges_count; ++i) {
        measured_edge_t const &edge = edges[i];
        if (edge.target >= memory_domains_count) continue;
        if (edge.megabytes_per_second > bandwidths[edge.target]) bandwidths[edge.target] = edge.megabytes_per_second;
        if (edge.nanoseconds && (latencies[edge.target] == 0 || edge.nanoseconds < latencies[edge.target]))
            latencies[edge.target] = edge.nanoseconds;
    }

    bool any_observed = false;
    for (std::size_t i = 0; i != memory_domains_count && !any_observed; ++i)
        any_observed = bandwidths[i] != 0 || latencies[i] != 0;
    if (!any_observed) return 1; // ? Nothing probed yet - a single tier covers every domain

    bubble_sort(order, memory_domains_count,
                [bandwidths, latencies](std::size_t const &a, std::size_t const &b) noexcept {
                    if (bandwidths[a] != bandwidths[b]) return bandwidths[a] > bandwidths[b];
                    std::size_t const latency_a = latencies[a] ? latencies[a] : ~std::size_t(0);
                    std::size_t const latency_b = latencies[b] ? latencies[b] : ~std::size_t(0);
                    return latency_a < latency_b; // ? Unobserved metrics sort as worst
                });

    std::size_t anchor_bandwidth = bandwidths[order[0]];
    std::size_t anchor_latency = latencies[order[0]];
    std::size_t bucket = 0;
    for (std::size_t i = 0; i != memory_domains_count; ++i) {
        std::size_t const bandwidth = bandwidths[order[i]];
        std::size_t const latency = latencies[order[i]];
        if (bandwidth == 0 && latency == 0) { // ? The unobserved tail shares one tier past the slowest
            for (; i != memory_domains_count; ++i) buckets[order[i]] = bucket + 1;
            break;
        }
        bool const bandwidth_fell = anchor_bandwidth * 4 > bandwidth * 5;
        bool const latency_grew = anchor_latency != 0 && latency * 2 > anchor_latency * 3;
        if (bandwidth_fell || latency_grew) bucket += 1, anchor_bandwidth = bandwidth, anchor_latency = latency;
        if (anchor_latency == 0) anchor_latency = latency; // ? A latency-less opener adopts its tier's first
        buckets[order[i]] = bucket;
    }

    return dense_rank(
        memory_domains_count, [buckets](std::size_t index) noexcept { return buckets[index]; },
        [levels](std::size_t index, std::size_t rank) noexcept { levels[index] = rank; });
}

/**
 *  @brief The measured memory fabric - what this process @b observed, as opposed to the structure
 *      the OS @b declared in `machine_topology`. Two query families: EDGE queries `(initiator,
 *      target)` describe one interconnect link; MEDIUM queries `(target)` describe the memory
 *      pool itself, independent of any initiator.
 *
 *  Completes the `try_harvest` pipeline: a `machine_topology` is harvested first and stays
 *  immutable, a `distributed_pool` spawns on it, and the fabric then harvests through that pool's
 *  pinned workers, snapshotting what it needs so the topology may be freed after. `try_harvest`
 *  is the only mutator and replaces the whole snapshot; before it, every query answers 0 and
 *  `memory_levels_count` answers 1.
 */
template <typename allocator_type_ = std::allocator<char>>
class measured_fabric {

  public:
    using allocator_t = allocator_type_;
    using edges_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<measured_edge_t>;
    using indices_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<std::size_t>;

  private:
    /** @brief Allocator the derivation scratch rebinds from; the arrays rebind their own. */
    allocator_t allocator_ {};
    /** @brief The observation log: sparse, only walked edges exist, each metric enveloped on query. */
    dynamic_array<measured_edge_t, edges_allocator_t> edges_;
    /** @brief Snapshot of `local_memory_of` per compute domain, so `memory_distance` needs no topology. */
    dynamic_array<std::size_t, indices_allocator_t> local_memory_;
    /** @brief Derived memory-tier ordinal per memory domain, 0 = fastest. */
    dynamic_array<std::size_t, indices_allocator_t> memory_levels_;
    /** @brief Number of distinct derived tiers (>= 1). */
    std::size_t memory_levels_count_ {1};
    /** @brief Snapshot of the topology's compute domain count; 0 marks an unharvested fabric. */
    std::size_t compute_domains_count_ {0};
    /** @brief Snapshot of the topology's memory domain count; 0 marks an unharvested fabric. */
    std::size_t memory_domains_count_ {0};

  public:
    constexpr measured_fabric() noexcept = default;

    measured_fabric(measured_fabric &&o) noexcept
        : allocator_(std::move(o.allocator_)), edges_(std::move(o.edges_)), local_memory_(std::move(o.local_memory_)),
          memory_levels_(std::move(o.memory_levels_)), memory_levels_count_(std::exchange(o.memory_levels_count_, 1)),
          compute_domains_count_(std::exchange(o.compute_domains_count_, 0)),
          memory_domains_count_(std::exchange(o.memory_domains_count_, 0)) {}

    measured_fabric &operator=(measured_fabric &&other) noexcept {
        if (this != &other) {
            allocator_ = std::move(other.allocator_);
            edges_ = std::move(other.edges_);
            local_memory_ = std::move(other.local_memory_);
            memory_levels_ = std::move(other.memory_levels_);
            memory_levels_count_ = std::exchange(other.memory_levels_count_, 1);
            compute_domains_count_ = std::exchange(other.compute_domains_count_, 0);
            memory_domains_count_ = std::exchange(other.memory_domains_count_, 0);
        }
        return *this;
    }

    measured_fabric(measured_fabric const &) = delete;
    measured_fabric &operator=(measured_fabric const &) = delete;

    ~measured_fabric() noexcept { reset(); }

    void reset() noexcept {
        edges_.reset();
        local_memory_.reset();
        memory_levels_.reset();
        memory_levels_count_ = 1;
        compute_domains_count_ = memory_domains_count_ = 0;
    }

    /** @brief Snapshot of the topology's compute domain count at harvest time; 0 before any harvest. */
    std::size_t compute_domains_count() const noexcept { return compute_domains_count_; }
    /** @brief Snapshot of the topology's memory domain count at harvest time; 0 before any harvest. */
    std::size_t memory_domains_count() const noexcept { return memory_domains_count_; }

    /** @brief Best observed dependent-load latency (nanoseconds) on an edge; 0 if unwalked or out of range.
     *  @note The minimum across recordings, since interference only ever adds nanoseconds. */
    std::size_t memory_latency(compute_domain_index_t const initiator,
                               memory_domain_index_t const target) const noexcept {
        if (initiator >= compute_domains_count_ || target >= memory_domains_count_) return 0;
        std::size_t best = 0;
        for (std::size_t i = 0; i != edges_.size(); ++i) {
            measured_edge_t const &edge = edges_[i];
            if (edge.initiator != initiator || edge.target != target || edge.nanoseconds == 0) continue;
            if (best == 0 || edge.nanoseconds < best) best = edge.nanoseconds;
        }
        return best;
    }

    /** @brief Best observed saturated read bandwidth (MB/s) on an edge; 0 if unwalked or out of range.
     *  @note The maximum across recordings, since interference only ever subtracts megabytes. */
    std::size_t memory_bandwidth(compute_domain_index_t const initiator,
                                 memory_domain_index_t const target) const noexcept {
        if (initiator >= compute_domains_count_ || target >= memory_domains_count_) return 0;
        std::size_t best = 0;
        for (std::size_t i = 0; i != edges_.size(); ++i) {
            measured_edge_t const &edge = edges_[i];
            if (edge.initiator == initiator && edge.target == target && edge.megabytes_per_second > best)
                best = edge.megabytes_per_second;
        }
        return best;
    }

    /**
     *  @brief Relative access distance on an edge, 10 = local per the SLIT convention; 0 if out of range.
     *
     *  The measured latency ratio to the initiator's local domain, times 10, rounded half-up and
     *  clamped to at least 10, so the local domain always carries the row's minimum. Unwalked
     *  edges fall back to 10-local / 20-remote; unclamped nanoseconds live in `memory_latency`.
     */
    std::size_t memory_distance(compute_domain_index_t const initiator,
                                memory_domain_index_t const target) const noexcept {
        if (initiator >= compute_domains_count_ || target >= memory_domains_count_) return 0;
        std::size_t const local = local_memory_[initiator];
        std::size_t const to_target = memory_latency(initiator, target);
        std::size_t const to_local = memory_latency(initiator, static_cast<memory_domain_index_t>(local));
        if (to_target == 0 || to_local == 0) return local == target ? 10u : 20u;
        return (std::max)((10 * to_target + to_local / 2) / to_local, std::size_t(10));
    }

    /** @brief The pool's derived speed class, 0 = fastest; keyed by the best bandwidth any
     *      initiator sustains to it, ties split by the best latency. 0 if out of range.
     *  @note Tier boundaries are measurement-derived and can shift between harvests. */
    std::size_t memory_level_in(memory_domain_index_t const memory_domain_index) const noexcept {
        if (memory_domain_index >= memory_domains_count_) return 0;
        return memory_levels_[memory_domain_index];
    }

    /** @brief Number of distinct derived memory tiers (>= 1). */
    std::size_t memory_levels_count() const noexcept { return memory_levels_count_; }

    /** @brief Number of recorded observations; each carries one experiment's metrics for one edge. */
    std::size_t edges_count() const noexcept { return edges_.size(); }
    /** @brief The observation at @p index, in [0, `edges_count()`). */
    measured_edge_t const &edge_at(std::size_t const index) const noexcept {
        assert(index < edges_.size() && "Edge index is out of bounds");
        return edges_[index];
    }

    /**
     *  @brief Harvests every reachable edge and the tiers derived from them through @p pool's
     *      pinned workers, replacing any previous snapshot. The @p topology is only read.
     *  @retval false when the pool spans no memory domains or a probe buffer cannot be allocated;
     *      the fabric is then left empty, never half-written.
     *  @note Not thread-safe: dispatches on the pool and rebuilds this fabric, so call it between
     *      task batches and do not query concurrently. Expect seconds of runtime on large fabrics.
     *
     *  Targets are the memory domains some worker can first-touch; @b cpuless domains, like CXL
     *  expanders, stay unwalked, since portable first-touch cannot place pages there.
     */
    template <typename micro_yield_type_, typename cache_hints_type_, std::size_t alignment_>
    bool try_harvest(machine_topology_t const &topology,
                     distributed_pool<micro_yield_type_, cache_hints_type_, alignment_> &pool) noexcept {
        reset();
        if (try_harvest_(topology, pool)) return true;
        reset(); // ? Bulk construction: a failed harvest leaves no partial matrix behind
        return false;
    }

  private:
    template <typename micro_yield_type_, typename cache_hints_type_, std::size_t alignment_>
    bool try_harvest_(machine_topology_t const &topology,
                      distributed_pool<micro_yield_type_, cache_hints_type_, alignment_> &pool) noexcept {

        // Snapshot the coordinate system, so the topology can be freed once this call returns.
        std::size_t const compute_domains = topology.compute_domains_count();
        std::size_t const memory_domains = topology.memory_domains_count();
        if (!local_memory_.try_resize(compute_domains)) return false;
        for (std::size_t domain = 0; domain != compute_domains; ++domain)
            local_memory_[domain] = topology.local_memory_of(static_cast<compute_domain_index_t>(domain));
        if (!memory_levels_.try_resize(memory_domains)) return false;
        compute_domains_count_ = compute_domains;
        memory_domains_count_ = memory_domains;

        dynamic_array<std::size_t> touchers;
        if (!touchers_per_position_(topology, pool, touchers)) return false;

        // A scratch the toucher streams through after each fill, flushing the freshly written list
        // out of its own caches; generous, but never past what a tight cgroup can spare.
        std::size_t evictor_bytes = std::size_t(256) << 20;
        if (volume_ram()) evictor_bytes = (std::min)(evictor_bytes, volume_ram() / 8);
        dynamic_array<chase_slot_t> evictor;
        if (!evictor.try_resize((std::max)(evictor_bytes / sizeof(chase_slot_t), std::size_t(1)))) return false;

        for (std::size_t target = 0; target != touchers.size(); ++target) {
            if (touchers[target] == unreachable_position_k) continue; // ? Nothing nearby can first-touch it

            // Sattolo's single cycle never revisits a slot before the walk closes, so three timed
            // stretches fit revisit-free as long as together they stay within one lap of the cycle.
            std::size_t const segments = 3;
            memory_domain_t const &target_domain =
                topology.memory_domain_at(static_cast<memory_domain_index_t>(target));
            std::size_t const slots = (std::max)(chase_list_bytes_(target_domain) / sizeof(chase_slot_t), segments);
            std::size_t const hops =
                (std::min)(slots / segments, std::size_t(1) << 20); // ? At least 1, as slots >= segments

            // Every compute domain chases separately: sibling domains on one memory controller
            // still differ - efficiency cores trail performance cores to the very same DRAM.
            for (std::size_t initiator = 0; initiator != pool.compute_domains_count(); ++initiator) {

                // Every edge gets a FRESH list, first-touched by the target's worker - that write
                // is the whole portable placement story - then evicted, since a chase must start
                // cold or the previous walker's cached copies re-route it through the directory.
                dynamic_array<chase_slot_t> list;
                if (!list.try_resize_uninitialized(slots)) return false;
                std::size_t drained = 0; // ! Escapes the broadcast and feeds the tail below, or the drain elides
                run_on_worker_(pool, touchers[target], [&]() noexcept {
                    thread_chase_list_(list.data(), slots);
                    std::size_t sum = 0;
                    for (std::size_t i = 0; i != evictor.size(); ++i) sum += evictor[i].next_slot_index;
                    drained = sum;
                });

                std::size_t nanoseconds = 0;
                run_on_worker_(pool, pool.first_thread(initiator),
                               [&]() noexcept { nanoseconds = chase_ns_per_hop_(list.data(), hops, segments); });
                nanoseconds += drained == ~std::size_t(0); // ? The data-dependent tail defeats elision
                // ? A latency-only observation: the lone-core chase says nothing about bandwidth
                measured_edge_t const edge {static_cast<compute_domain_index_t>(initiator),
                                            static_cast<memory_domain_index_t>(target), nanoseconds, 0};
                if (!edges_.try_push_back(edge)) return false;
            }
        }

        evictor.reset(); // ? Streams evict themselves, so the scratch would only crowd them out

        dynamic_array<std::uint64_t> checksums;
        if (!checksums.try_resize(pool.threads_count())) return false;
        for (std::size_t target = 0; target != touchers.size(); ++target) {
            if (touchers[target] == unreachable_position_k) continue; // ? Nothing nearby can first-touch it
            if (!try_measure_bandwidth_edges_(topology, pool, target, touchers[target], checksums, edges_))
                return false;
        }

        // The tiers need both metrics - bandwidth ranks the medium, latency splits ties.
        dynamic_array<std::size_t, indices_allocator_t> scratch {indices_allocator_t {allocator_}};
        if (!scratch.try_resize(memory_domains * 4)) return false;
        memory_levels_count_ =
            derive_memory_levels_(edges_.data(), edges_.size(), memory_levels_.data(), memory_domains, scratch.data());
        return true;
    }
};

using measured_fabric_t = measured_fabric<>;

#pragma endregion Measured Memory Distances

using colocated_pool_t = colocated_pool<>;
using distributed_pool_t = distributed_pool<>;

#if FU_DETECT_CONCEPTS_
static_assert(is_unsafe_pool<flat_pool_t> && is_unsafe_pool<colocated_pool_t>,
              "These thread pools must be flexible and support unsafe operations");
static_assert(is_pool<flat_pool_t> && is_pool<colocated_pool_t> && is_pool<distributed_pool_t>,
              "These thread pools must be fully compatible with the high-level APIs");
#endif // FU_DETECT_CONCEPTS_

#endif // FU_WITH_OS_THREADS

#pragma endregion Distributed Pool

} // namespace forkunion
} // namespace ashvardanian
