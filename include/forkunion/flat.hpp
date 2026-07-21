/**
 *  @file flat.hpp
 *  @brief The portable `flat_pool`, built on `std::thread`.
 *  @note Included by `<forkunion.hpp>`; not meant to be included on its own.
 */
#pragma once
#include "types.hpp"

namespace ashvardanian {
namespace forkunion {

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
 *  #include <forkunion.hpp> // `flat_pool_t`
 *
 *  using fu = ashvardanian::forkunion;
 *  int main() {
 *      fu::flat_pool_t pool; // ? Alias to `fu::flat_pool<>` template
 *      if (!pool.try_spawn(allowed_cores_count())) return EXIT_FAILURE;
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
 *  #include <forkunion.hpp> // `flat_pool_t`
 *
 *  using fu = ashvardanian::forkunion;
 *  int main() {
 *      fu::flat_pool_t first_pool, second_pool;
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
 *  @tparam cache_hints_type_ The cache-line demote/promote policy for the dynamic claim cursors.
 *  @tparam index_type_ Use `std::size_t`, but or a smaller type for debugging.
 *  @tparam alignment_ The alignment of the thread pool. Defaults to `default_alignment_k`.
 */
template <                                                  //
    typename allocator_type_ = std::allocator<std::thread>, //
    typename micro_yield_type_ = standard_yield_t,          //
    typename cache_hints_type_ = standard_cache_hints_t,    //
    typename index_type_ = std::size_t,                     //
    std::size_t alignment_ = default_alignment_k            //
    >
class flat_pool {

  public:
    using allocator_t = allocator_type_;
    using micro_yield_t = micro_yield_type_;
    using cache_hints_t = cache_hints_type_;
    static constexpr pool_kind_t kind_k = pool_kind_t::flat_k;
    static constexpr std::size_t alignment_k = alignment_;
    static_assert(is_power_of_two(alignment_k), "Alignment must be a power of 2");

    using index_t = index_type_;
    static_assert(std::is_unsigned<index_t>::value, "Index type must be an unsigned integer");
    using epoch_index_t = index_t;      // ? A.k.a. number of previous API calls in [0, UINT_MAX)
    using generation_t = epoch_index_t; // ? A.k.a. token returned from `unsafe_for_threads`; always odd
    // ! With small index types (like the `fu8_t`/`fu16_t` debug configs) a worker stalled across
    // ! exactly 2^bits epochs would alias its `last_epoch` - astronomically unlikely at `size_t`.
    using thread_index_t = index_t;         // ? A.k.a. "core index" or "thread ID" in [0, threads_count)
    using compute_domain_index_t = index_t; // ? Dense index in [0, compute_domains_count)
    using indexed_split_t = indexed_split<index_t>;
    using local_thread_t = local_thread<index_t>;
    using prong_t = prong<index_t>;
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
     *      `std::allocator` only promises `__STDCPP_DEFAULT_NEW_ALIGNMENT__`, so over-aligning the
     *      cell would placement-new it into storage that cannot satisfy the request.
     */
    struct worker_cell_t {
        /** @brief This thread's private cursor for `for_n_dynamic`. @sa `dynamic_claim`. */
        claim_t claim {};
        /** @brief The worker thread; default-constructed, and left so for the caller's own cell. */
        std::thread worker {};
    };
    static_assert(sizeof(worker_cell_t) <= alignment_k, "A worker cell must fit within one stride");

    using worker_cell_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<worker_cell_t>;
    using worker_cells_t = dynamic_padded_array<worker_cell_t, worker_cell_allocator_t>;

    using punned_fork_context_t = void *;                                 // ? Pointer to the on-stack lambda
    using trampoline_t = void (*)(punned_fork_context_t, thread_index_t); // ? Wraps lambda's `operator()`

    static_assert(is_wait_functor<micro_yield_t, epoch_index_t, thread_index_t>::value,
                  "Yield must be callable as `yield(watched_atomic, observed_value, thread_index)`");
    static_assert(is_cache_hints_functor<cache_hints_t>::value,
                  "Cache hints must be callable as `hints(address, demote_line_k)` and `(address, promote_line_k)`");

  private:
    // Thread-pool-specific variables:
    /** @brief Allocator backing the pool's single worker-cells allocation. */
    allocator_t allocator_ {};
    /** @brief One padded cell per thread: its `std::thread` and its claim cursor. */
    worker_cells_t workers_ {};
    /** @brief Total threads in the pool, including the caller on inclusive pools. */
    thread_index_t threads_count_ {0};
    /** @brief Whether the caller thread is counted as one of the contributors. */
    caller_exclusivity_t exclusivity_ {caller_inclusive_k};
    /** @brief How long to nap in microseconds while `chill_k`, waiting for work. */
    std::size_t sleep_length_micros_ {0};
    /** @brief Lifecycle switch between spinning (`grind_k`), sleeping (`chill_k`), and exiting (`die_k`). */
    alignas(alignment_k) std::atomic<mood_t> mood_ {mood_t::grind_k};

    // Task-specific variables:
    /** @brief Type-erased pointer to the caller's on-stack fork lambda. */
    punned_fork_context_t fork_state_ {nullptr};
    /** @brief Invokes the punned fork lambda for a given thread index. */
    trampoline_t fork_trampoline_ {nullptr};
    /** @brief Countdown of contributors still running; the one reaching zero signals completion. */
    alignas(alignment_k) std::atomic<thread_index_t> threads_to_sync_ {0};
    /** @brief Generation clock: odd while a fork is in flight, even when idle. */
    alignas(alignment_k) std::atomic<epoch_index_t> epoch_ {0};

  public:
    flat_pool(flat_pool &&) = delete;
    flat_pool(flat_pool const &) = delete;
    flat_pool &operator=(flat_pool &&) = delete;
    flat_pool &operator=(flat_pool const &) = delete;

    flat_pool(allocator_t const &alloc = {}) noexcept : allocator_(alloc) {}
    ~flat_pool() noexcept { terminate(); }

    /**
     *  @brief Estimates the amount of memory managed by this pool handle and internal structures.
     *  @note This API is @b not synchronized.
     */
    std::size_t memory_usage() const noexcept { return sizeof(flat_pool) + workers_.size() * workers_.stride(); }

    /** @brief Checks if the thread-pool's core synchronization points are lock-free. */
    bool is_lock_free() const noexcept { return mood_.is_lock_free() && threads_to_sync_.is_lock_free(); }

    /**
     *  @brief Returns the memory domain this thread-pool is pinned to.
     *  @retval -1 as this pool is not memory-domain-aware.
     */
    constexpr memory_domain_id_t memory_domain_id() const noexcept { return -1; }

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
    broadcast_join<flat_pool, fork_type_> for_threads(fork_type_ &&fork) noexcept {
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
    broadcast_join<flat_pool, invoke_for_slices<fork_type_, index_t>> //
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
    broadcast_join<flat_pool, invoke_for_n<fork_type_, index_t>> //
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
    broadcast_join<flat_pool, invoke_for_n_dynamic<flat_pool, fork_type_, index_t>> //
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
     *  @return Same value as `threads_count()`, as we only support one compute_domain.
     *  @note Shape parity with `distributed_pool`: generic callers - the C ABI's `visit` and the
     *      distributed invokers - call `pool.threads_count(domain)` on every pool kind.
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
    /** @brief Clears the fork state and trampoline between dispatches. */
    void _reset_fork() noexcept {
        fork_state_ = nullptr;
        fork_trampoline_ = nullptr;
    }

    /**
     *  @brief A trampoline function that is used to call the user-defined lambda.
     *  @param[in] punned_lambda_pointer The pointer to the user-defined lambda.
     *  @param[in] thread_index The thread whose slice of the broadcast this call runs.
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
            // This loop guards two independent lines - the epoch and the mood. A monitor can arm only
            // one, so arm the hot one (a dispatch bumps `epoch_`) and let the waiter's timeout cap
            // bound how late a `mood_` change - a `sleep` or a `terminate`, both rare - is noticed.
            while ((new_epoch = epoch_.load(std::memory_order_acquire)) == last_epoch &&
                   (mood = mood_.load(std::memory_order_acquire)) == mood_t::grind_k)
                micro_yield(epoch_, last_epoch, thread_index);

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

using flat_pool_t = flat_pool<>;

#pragma region Concepts
#if FU_DETECT_CONCEPTS_

/** @brief Does nothing on every thread. The default fork for a `broadcast_join` that only needs the join. */
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

} // namespace forkunion
} // namespace ashvardanian
