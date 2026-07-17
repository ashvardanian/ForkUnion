/**
 *  @brief  A libgomp-ABI shim, so OpenMP-compiled binaries run on a ForkUnion pool without being recompiled.
 *  @file   forkunion_gomp.cpp
 *  @author Ash Vardanian
 *
 *  scikit-learn, PyTorch, XGBoost, and LightGBM all reach their thread pool through libgomp's ABI, and none of them
 *  offers a way to substitute one - PyTorch deleted its TBB backend in 2.4, and Numba validates its layer list against
 *  a hardcoded three-element set. The absence of a plug-in seam is what makes this ABI the only way in, and it is a
 *  narrow one: the union of what those four import is about 35 functions.
 *
 *  Interposed with `LD_PRELOAD`, or by `dlopen`-ing this into the global scope before libgomp is pulled in as a later
 *  `DT_NEEDED` - see `python/openmp.py`. glibc sanctions the former outright: `elf/dl-lookup.c` accepts an unversioned
 *  definition for a versioned reference, commenting "this can happen during symbol interposition". So this library is
 *  deliberately @b unversioned; half-versioned symbols match nothing at all and are worse than none.
 *
 *  `GOMP_task` runs the task @b immediately, on the encountering thread, which the spec allows and libgomp itself does
 *  for `if(0)` - so no task queue is needed and none is added. Leaving the symbol undefined would be far worse than
 *  implementing it: faiss would bind `GOMP_parallel` here and `GOMP_task` to the real libgomp, splitting one library's
 *  calls across two runtimes that share no team state.
 */
#include <atomic>  // `std::atomic`
#include <cstdint> // `std::uint32_t`
#include <cstdio>  // `std::fprintf` for the interposition report
#include <cstdlib> // `getenv`, `atoi`

#include <forkunion.h>   // `fu_pool_t`, the pool we dispatch onto
#include <forkunion.hpp> // `spin_mutex`, `preferred_yield_t` - the waiter the barrier parks on

namespace fu = ashvardanian::forkunion;

#if defined(__GNUC__)
#pragma GCC visibility push(default) // ! Every GOMP entry point must be interposable
#endif

#pragma region Team State

/**
 *  @brief The team currently inside a `GOMP_parallel` region.
 *
 *  One team at a time: the pool runs one generation at a time, and nesting is refused, so a single instance suffices.
 *  Everything here is touched by all workers, so the hot lines are separated - a shared `barrier_epoch` and a shared
 *  `loop_cursor` on one cache line would make every barrier pay for the loop's traffic.
 */
struct team_state_t {
    /** @brief Workers that have arrived at the current barrier. */
    alignas(fu::default_alignment_k) std::atomic<unsigned> barrier_arrived {0};
    /** @brief Bumped once per barrier release; waiters watch this line rather than the counter. */
    alignas(fu::default_alignment_k) std::atomic<unsigned> barrier_epoch {0};
    /** @brief The next un-taken iteration, handed out by `fetch_add` - never a compare-and-swap. */
    alignas(fu::default_alignment_k) std::atomic<long long> loop_cursor {0};
    /** @brief Claimed by exactly one worker per `GOMP_single_start`. */
    alignas(fu::default_alignment_k) std::atomic<unsigned> single_epoch {0};
    alignas(fu::default_alignment_k) std::atomic<unsigned> single_claimed {0};

    /*  Loop bounds, written once before the region and read-only inside it. */
    long long loop_end {0};
    long long loop_step {1};
    long long loop_chunk {1};
    bool loop_up {true};

    /** @brief Threads in this team - what `omp_get_num_threads` must report inside the region. */
    unsigned size {1};
};

static team_state_t team_;

/*  Per-thread view of the team. `level_` is what makes nesting safe: libgomp's `max_active_levels_var` defaults to 1,
 *  so an inner region runs on the calling thread with a team of one, and mirroring that exactly is not a compromise -
 *  it is what stock libgomp already does.
 */
static thread_local unsigned tls_thread_num_ = 0;
static thread_local unsigned tls_team_size_ = 1;
static thread_local unsigned tls_level_ = 0;

/** @brief The pool every region runs on, and the lock that keeps two regions from sharing it. */
static fu_topology_t topology_ = nullptr;
static fu_pool_t pool_ = nullptr;
static fu::spin_mutex_t pool_lock_;

/** @brief `OMP_NUM_THREADS` / `omp_set_num_threads`, i.e. what a *subsequent* region would use. */
static std::atomic<unsigned> nthreads_var_ {0};

/*  Interposition is invisible when it works and equally invisible when it silently does not - a program that never
 *  binds these symbols behaves exactly like one that does, only slower. Counting the regions is the only way to tell
 *  the two apart, so `FORKUNION_GOMP_VERBOSE=1` reports them on the way out.
 */
static std::atomic<unsigned long long> regions_parallel_ {0};
static std::atomic<unsigned long long> regions_serial_ {0};

__attribute__((destructor)) static void report_interposition_() noexcept {
    char const *verbose = std::getenv("FORKUNION_GOMP_VERBOSE");
    if (!verbose || verbose[0] != '1') return;
    std::fprintf(stderr, "[forkunion-gomp] intercepted %llu parallel + %llu serial regions on %u threads\n",
                 regions_parallel_.load(), regions_serial_.load(), nthreads_var_.load());
}

/** @brief Spawns the pool once. Returns 0 if the machine or the pool refuses, so callers fall back to serial. */
static unsigned ensure_pool_() noexcept {
    static unsigned const threads_ = []() -> unsigned {
        topology_ = fu_topology_new();
        if (!topology_) return 0;
        size_t cores = fu_logical_cores_count(topology_);
        if (char const *env = std::getenv("OMP_NUM_THREADS")) {
            int const requested = std::atoi(env);
            if (requested > 0) cores = static_cast<size_t>(requested);
        }
        if (!cores) return 0;
        pool_ = fu_pool_new("gomp", fu_capabilities_all_k);
        // ! Zero threads is not "all" - it fails the spawn - so the count is always explicit.
        if (!pool_ || !fu_pool_spawn(topology_, pool_, cores, fu_caller_inclusive_k)) return 0;
        nthreads_var_.store(static_cast<unsigned>(cores), std::memory_order_relaxed);
        return static_cast<unsigned>(cores);
    }();
    return threads_;
}

#pragma endregion Team State

#pragma region Barrier

extern "C" void GOMP_barrier(void) {
    unsigned const size = tls_team_size_;
    if (size <= 1) return; // ? A team of one is always at the barrier

    // Sense-reversing barrier: `fetch_add` to arrive, and the last arrival releases everyone with a single store to a
    // line the others are parked on. ggml spins here ~998 times per token with no backoff.
    //
    // ! `preferred_yield_t`, never `standard_yield_t` - the latter is `std::this_thread::yield()`, a `sched_yield`
    // ! syscall on every turn of a loop that may turn thousands of times. This resolves to `PAUSE` by default and to
    // ! the `UMWAIT` monitor when built with `-mwaitpkg`, which parks the core until the release store lands.
    unsigned const epoch = team_.barrier_epoch.load(std::memory_order_relaxed);
    if (team_.barrier_arrived.fetch_add(1, std::memory_order_acq_rel) == size - 1) {
        team_.barrier_arrived.store(0, std::memory_order_relaxed);
        team_.barrier_epoch.fetch_add(1, std::memory_order_release);
        return;
    }
    fu::preferred_yield_t micro_yield;
    while (team_.barrier_epoch.load(std::memory_order_acquire) == epoch)
        micro_yield(team_.barrier_epoch, epoch, static_cast<std::size_t>(tls_thread_num_));
}

#pragma endregion Barrier

#pragma region Parallel Regions

struct gomp_context_t {
    void (*function)(void *);
    void *data;
    unsigned size;
};

static void trampoline_(fu_lambda_context_t punned, size_t thread, size_t compute_domain) noexcept {
    (void)compute_domain;
    gomp_context_t const *context = static_cast<gomp_context_t const *>(punned);
    if (thread >= context->size) return; // ? The pool is larger than the team the caller asked for
    tls_thread_num_ = static_cast<unsigned>(thread);
    tls_team_size_ = context->size;
    ++tls_level_;
    context->function(context->data);
    --tls_level_;
    tls_thread_num_ = 0;
    tls_team_size_ = 1;
}

/** @brief Resolves the team size the way `gomp_resolve_num_threads` does, refusals included. */
static unsigned resolve_threads_(unsigned requested) noexcept {
    // ! A hard contract, not a hint. GCC folds `if (cond)` into the count as `cond ? val : 1u`, so a false `if`
    // ! clause arrives here as 1 - and code taking a serial path is written assuming no concurrency, often without
    // ! the locks that would make it safe. Running it on many threads corrupts silently.
    if (requested == 1) return 1;
    if (tls_level_ >= 1) return 1; // ? `max_active_levels_var` defaults to 1: libgomp does not nest either
    unsigned const available = ensure_pool_();
    if (!available) return 1;
    if (requested == 0) requested = nthreads_var_.load(std::memory_order_relaxed); // ? "Unspecified" means the ICV
    if (!requested) requested = available;
    return requested < available ? requested : available;
}

extern "C" void GOMP_parallel(void (*function)(void *), void *data, unsigned num_threads, unsigned flags) {
    // `flags` carries only `proc_bind`, which libgomp masks as `flags & 7`. It never changes what is computed, and the
    // pool's own pinning supersedes it, so it is safe to ignore.
    (void)flags;
    unsigned const size = resolve_threads_(num_threads);
    (size == 1 ? regions_serial_ : regions_parallel_).fetch_add(1, std::memory_order_relaxed);
    if (size == 1) { // ? Serial, on the calling thread - exactly where libgomp runs it
        unsigned const saved_num = tls_thread_num_, saved_size = tls_team_size_;
        tls_thread_num_ = 0, tls_team_size_ = 1, ++tls_level_;
        function(data);
        --tls_level_, tls_thread_num_ = saved_num, tls_team_size_ = saved_size;
        return;
    }

    // Two Python threads can both drop the GIL and land here at once. The pool cannot serve both, and nesting is
    // banned, so the loser runs the region itself rather than waiting.
    if (!pool_lock_.try_lock()) {
        tls_thread_num_ = 0, tls_team_size_ = 1, ++tls_level_;
        function(data);
        --tls_level_;
        return;
    }

    team_.size = size;
    team_.barrier_arrived.store(0, std::memory_order_relaxed);
    team_.barrier_epoch.store(0, std::memory_order_relaxed);
    team_.single_epoch.store(0, std::memory_order_relaxed);
    team_.single_claimed.store(0, std::memory_order_relaxed);

    gomp_context_t context {function, data, size};
    fu_pool_for_threads(pool_, &trampoline_, &context);
    pool_lock_.unlock();
}

extern "C" void GOMP_parallel_start(void (*function)(void *), void *data, unsigned num_threads) {
    // The pre-4.0 spelling splits the region across two calls, which cannot map onto a blocking `for_threads`.
    // Running it serially is always correct, only slower - and modern GCC emits `GOMP_parallel` instead.
    (void)num_threads;
    ++tls_level_;
    function(data);
    --tls_level_;
}

extern "C" void GOMP_parallel_end(void) {}

#pragma endregion Parallel Regions

#pragma region Work Sharing

/** @brief Hands this thread the next chunk, or reports the loop is drained. */
static bool loop_next_(long *start, long *end) noexcept {
    long long const chunk = team_.loop_chunk;
    long long const taken = team_.loop_cursor.fetch_add(chunk * team_.loop_step, std::memory_order_relaxed);
    if (team_.loop_up ? taken >= team_.loop_end : taken <= team_.loop_end) return false;
    long long stop = taken + chunk * team_.loop_step;
    if (team_.loop_up ? stop > team_.loop_end : stop < team_.loop_end) stop = team_.loop_end;
    *start = static_cast<long>(taken);
    *end = static_cast<long>(stop);
    return true;
}

static bool loop_start_(long start, long end, long step, long chunk, long *first, long *last) noexcept {
    if (tls_thread_num_ == 0) {
        team_.loop_end = end;
        team_.loop_step = step ? step : 1;
        team_.loop_chunk = chunk > 0 ? chunk : 1;
        team_.loop_up = step > 0;
        team_.loop_cursor.store(start, std::memory_order_relaxed);
    }
    GOMP_barrier(); // ! Every worker must see the bounds before any of them takes a chunk
    return loop_next_(first, last);
}

extern "C" bool GOMP_loop_dynamic_start(long start, long end, long step, long chunk, long *first, long *last) {
    return loop_start_(start, end, step, chunk, first, last);
}
extern "C" bool GOMP_loop_dynamic_next(long *first, long *last) { return loop_next_(first, last); }

extern "C" bool GOMP_loop_nonmonotonic_dynamic_start(long start, long end, long step, long chunk, long *first,
                                                     long *last) {
    return loop_start_(start, end, step, chunk, first, last);
}
extern "C" bool GOMP_loop_nonmonotonic_dynamic_next(long *first, long *last) { return loop_next_(first, last); }

/*  Guided differs from dynamic only in how the chunk shrinks, and a schedule is a performance choice, not a semantic
 *  one - any partition of the range is a legal answer. Serving it from the same cursor is correct, if not clever.
 */
extern "C" bool GOMP_loop_guided_start(long start, long end, long step, long chunk, long *first, long *last) {
    return loop_start_(start, end, step, chunk, first, last);
}
extern "C" bool GOMP_loop_guided_next(long *first, long *last) { return loop_next_(first, last); }

extern "C" bool GOMP_loop_nonmonotonic_guided_start(long start, long end, long step, long chunk, long *first,
                                                    long *last) {
    return loop_start_(start, end, step, chunk, first, last);
}
extern "C" bool GOMP_loop_nonmonotonic_guided_next(long *first, long *last) { return loop_next_(first, last); }

extern "C" bool GOMP_loop_static_start(long start, long end, long step, long chunk, long *first, long *last) {
    return loop_start_(start, end, step, chunk ? chunk : 1, first, last);
}
extern "C" bool GOMP_loop_static_next(long *first, long *last) { return loop_next_(first, last); }

/** @brief `runtime` schedule: five arguments, not six - it reads the chunk from `OMP_SCHEDULE` rather than the call. */
extern "C" bool GOMP_loop_runtime_start(long start, long end, long step, long *first, long *last) {
    return loop_start_(start, end, step, 1, first, last);
}
extern "C" bool GOMP_loop_runtime_next(long *first, long *last) { return loop_next_(first, last); }

extern "C" void GOMP_loop_end(void) { GOMP_barrier(); }
extern "C" void GOMP_loop_end_nowait(void) {}

extern "C" bool GOMP_single_start(void) {
    if (tls_team_size_ <= 1) return true;
    // The first arrival claims it; `exchange` decides, so no compare-and-swap is needed.
    return team_.single_claimed.exchange(1, std::memory_order_acq_rel) == 0;
}

/*  The `ull` family is the same protocol over `unsigned long long`, with one extra leading argument: `up`, the
 *  iteration direction. Missing that argument shifts every other one.
 */
static bool loop_ull_next_(unsigned long long *first, unsigned long long *last) noexcept {
    long start = 0, end = 0;
    if (!loop_next_(&start, &end)) return false;
    *first = static_cast<unsigned long long>(start), *last = static_cast<unsigned long long>(end);
    return true;
}

static bool loop_ull_start_(bool up, unsigned long long start, unsigned long long end, unsigned long long step,
                            unsigned long long chunk, unsigned long long *first, unsigned long long *last) noexcept {
    long f = 0, l = 0;
    bool const taken =
        loop_start_(static_cast<long>(start), static_cast<long>(end),
                    up ? static_cast<long>(step) : -static_cast<long>(step), static_cast<long>(chunk), &f, &l);
    *first = static_cast<unsigned long long>(f), *last = static_cast<unsigned long long>(l);
    return taken;
}

extern "C" bool GOMP_loop_ull_dynamic_start(bool up, unsigned long long start, unsigned long long end,
                                            unsigned long long step, unsigned long long chunk,
                                            unsigned long long *first, unsigned long long *last) {
    return loop_ull_start_(up, start, end, step, chunk, first, last);
}
extern "C" bool GOMP_loop_ull_dynamic_next(unsigned long long *first, unsigned long long *last) {
    return loop_ull_next_(first, last);
}
extern "C" bool GOMP_loop_ull_nonmonotonic_dynamic_start(bool up, unsigned long long start, unsigned long long end,
                                                         unsigned long long step, unsigned long long chunk,
                                                         unsigned long long *first, unsigned long long *last) {
    return loop_ull_start_(up, start, end, step, chunk, first, last);
}
extern "C" bool GOMP_loop_ull_nonmonotonic_dynamic_next(unsigned long long *first, unsigned long long *last) {
    return loop_ull_next_(first, last);
}
extern "C" bool GOMP_loop_ull_guided_start(bool up, unsigned long long start, unsigned long long end,
                                           unsigned long long step, unsigned long long chunk, unsigned long long *first,
                                           unsigned long long *last) {
    return loop_ull_start_(up, start, end, step, chunk, first, last);
}
extern "C" bool GOMP_loop_ull_guided_next(unsigned long long *first, unsigned long long *last) {
    return loop_ull_next_(first, last);
}
extern "C" bool GOMP_loop_ull_nonmonotonic_guided_start(bool up, unsigned long long start, unsigned long long end,
                                                        unsigned long long step, unsigned long long chunk,
                                                        unsigned long long *first, unsigned long long *last) {
    return loop_ull_start_(up, start, end, step, chunk, first, last);
}
extern "C" bool GOMP_loop_ull_nonmonotonic_guided_next(unsigned long long *first, unsigned long long *last) {
    return loop_ull_next_(first, last);
}

/*  An `ordered` loop must run its ordered regions in iteration order, and a lock enforces mutual exclusion without
 *  enforcing order - it would quietly compute the wrong answer. Give every iteration to the team's first thread
 *  instead: order is then trivially preserved, `GOMP_ordered_start` has nobody to wait for, and the cost is the
 *  parallelism of a construct that is rare in this ecosystem.
 */
extern "C" bool GOMP_loop_ordered_dynamic_start(long start, long end, long step, long chunk, long *first, long *last) {
    if (tls_thread_num_ != 0) {
        GOMP_barrier(); // ! Still pair with thread zero's barrier inside `loop_start_`
        return false;
    }
    bool const taken = loop_start_(start, end, step, end - start ? end - start : 1, first, last);
    return taken;
}
extern "C" bool GOMP_loop_ordered_dynamic_next(long *first, long *last) {
    return tls_thread_num_ == 0 ? loop_next_(first, last) : false;
}
extern "C" bool GOMP_loop_ordered_static_start(long start, long end, long step, long chunk, long *first, long *last) {
    return GOMP_loop_ordered_dynamic_start(start, end, step, chunk, first, last);
}
extern "C" bool GOMP_loop_ordered_static_next(long *first, long *last) {
    return GOMP_loop_ordered_dynamic_next(first, last);
}
extern "C" bool GOMP_loop_ordered_guided_start(long start, long end, long step, long chunk, long *first, long *last) {
    return GOMP_loop_ordered_dynamic_start(start, end, step, chunk, first, last);
}
extern "C" bool GOMP_loop_ordered_guided_next(long *first, long *last) {
    return GOMP_loop_ordered_dynamic_next(first, last);
}
extern "C" bool GOMP_loop_ordered_runtime_start(long start, long end, long step, long *first, long *last) {
    return GOMP_loop_ordered_dynamic_start(start, end, step, 1, first, last);
}
extern "C" bool GOMP_loop_ordered_runtime_next(long *first, long *last) {
    return GOMP_loop_ordered_dynamic_next(first, last);
}

/*  Only the loop's single worker ever enters an ordered region, so ordering is already guaranteed by construction.
 */
extern "C" void GOMP_ordered_start(void) {}
extern "C" void GOMP_ordered_end(void) {}

/*  `sections` hands each thread a section number in `[1, count]`, and zero once they are gone - a cursor, like a loop.
 */
extern "C" unsigned GOMP_sections_start(unsigned count) {
    if (tls_thread_num_ == 0) {
        team_.loop_end = count + 1;
        team_.loop_step = 1;
        team_.loop_chunk = 1;
        team_.loop_up = true;
        team_.loop_cursor.store(1, std::memory_order_relaxed);
    }
    GOMP_barrier();
    long first = 0, last = 0;
    return loop_next_(&first, &last) ? static_cast<unsigned>(first) : 0u;
}

extern "C" unsigned GOMP_sections_next(void) {
    long first = 0, last = 0;
    return loop_next_(&first, &last) ? static_cast<unsigned>(first) : 0u;
}

extern "C" void GOMP_sections_end(void) { GOMP_barrier(); }
extern "C" void GOMP_sections_end_nowait(void) {}

#pragma endregion Work Sharing

#pragma region Tasks

/**
 *  @brief Runs the task now, on this thread, rather than queueing it.
 *
 *  The spec lets an implementation execute a task immediately - libgomp does exactly this for `if(0)` - and an
 *  undeferred task trivially satisfies any `depend` clause, because every task created before it has already finished.
 *  So this costs parallelism between sibling tasks and costs nothing in correctness, and needs no queue.
 */
extern "C" void GOMP_task(void (*function)(void *), void *data, void (*copy)(void *, void *), long argument_size,
                          long argument_align, bool if_clause, unsigned flags, void **depend, int priority,
                          void *detach) {
    (void)if_clause, (void)flags, (void)depend, (void)priority, (void)detach;
    if (!copy) return function(data);

    // With a copy constructor the callee owns its arguments, so they need a correctly aligned buffer of their own.
    // The stack suffices: the task runs and returns before this frame does.
    if (argument_align < 1) argument_align = 1;
    void *raw = __builtin_alloca(static_cast<size_t>(argument_size + argument_align));
    auto address = reinterpret_cast<std::uintptr_t>(raw);
    address = (address + static_cast<std::uintptr_t>(argument_align) - 1) &
              ~(static_cast<std::uintptr_t>(argument_align) - 1);
    void *arguments = reinterpret_cast<void *>(address);
    copy(arguments, data);
    function(arguments);
}

/*  Nothing is ever outstanding, so every wait is already satisfied.
 */
extern "C" void GOMP_taskwait(void) {}
extern "C" void GOMP_taskyield(void) {}
extern "C" void GOMP_taskgroup_start(void) {}
extern "C" void GOMP_taskgroup_end(void) {}

#pragma endregion Tasks

#pragma region Mutual Exclusion

static fu::spin_mutex_t critical_;
static fu::spin_mutex_t atomic_;

extern "C" void GOMP_critical_start(void) { critical_.lock(); }
extern "C" void GOMP_critical_end(void) { critical_.unlock(); }
extern "C" void GOMP_atomic_start(void) { atomic_.lock(); }
extern "C" void GOMP_atomic_end(void) { atomic_.unlock(); }

/*  Named criticals get one lock rather than one per name: over-serializing two different names is slow, never wrong,
 *  and the alternative is a name-keyed table allocated on a path that must not allocate.
 */
extern "C" void GOMP_critical_name_start(void **pointer) { (void)pointer, critical_.lock(); }
extern "C" void GOMP_critical_name_end(void **pointer) { (void)pointer, critical_.unlock(); }

#pragma endregion Mutual Exclusion

#pragma region Locks

/**
 *  @brief `omp_lock_t` is @b four bytes, 4-aligned - measured, not assumed.
 *
 *  Callers embed it inline in their own structs, so anything larger overwrites whatever sits next to it and corrupts
 *  memory quietly. A `pthread_mutex_t` is 40 bytes and `fu::spin_mutex_t` is cache-line aligned; neither fits. That
 *  leaves one atomic word, spinning on the same monitor the barrier uses.
 */
struct alignas(4) tiny_lock_t {
    std::atomic<std::uint32_t> flag;
};
static_assert(sizeof(tiny_lock_t) == 4, "omp_lock_t is 4 bytes on the ABI we shadow");

extern "C" void omp_init_lock(void *lock) { new (lock) tiny_lock_t {{0u}}; }
extern "C" void omp_destroy_lock(void *lock) { (void)lock; }

extern "C" void omp_set_lock(void *lock) {
    tiny_lock_t *self = static_cast<tiny_lock_t *>(lock);
    fu::preferred_yield_t micro_yield;
    while (self->flag.exchange(1, std::memory_order_acquire))
        while (self->flag.load(std::memory_order_relaxed)) micro_yield(self->flag, 1u, static_cast<std::size_t>(0));
}

extern "C" void omp_unset_lock(void *lock) {
    static_cast<tiny_lock_t *>(lock)->flag.store(0, std::memory_order_release);
}

extern "C" int omp_test_lock(void *lock) {
    return static_cast<tiny_lock_t *>(lock)->flag.exchange(1, std::memory_order_acquire) == 0;
}

#pragma endregion Locks

#pragma region Runtime Queries

/*  The distinction that breaks programs when blurred: `omp_get_num_threads` is the size of the team running *now* -
 *  1 outside any region - while `omp_get_max_threads` is what the *next* region would use and is legal anywhere.
 *
 *  The idiom `for (i = omp_get_thread_num(); i < n; i += omp_get_num_threads())` is everywhere in this ecosystem, and
 *  scikit-learn sizes per-thread buffers by `omp_get_max_threads()` before a region, then indexes them by
 *  `omp_get_thread_num()`. Report a number that disagrees with the team we actually built and the result is skipped
 *  elements or an out-of-bounds write, not an error.
 */
extern "C" int omp_get_num_threads(void) { return static_cast<int>(tls_team_size_); }
extern "C" int omp_get_thread_num(void) { return static_cast<int>(tls_thread_num_); }
extern "C" int omp_in_parallel(void) { return tls_level_ > 0 && tls_team_size_ > 1; }
extern "C" int omp_get_level(void) { return static_cast<int>(tls_level_); }
extern "C" int omp_get_active_level(void) { return tls_team_size_ > 1 ? 1 : 0; }

extern "C" int omp_get_max_threads(void) {
    unsigned const requested = nthreads_var_.load(std::memory_order_relaxed);
    if (requested) return static_cast<int>(requested);
    unsigned const available = ensure_pool_();
    return static_cast<int>(available ? available : 1u);
}

extern "C" void omp_set_num_threads(int count) {
    if (count > 0) nthreads_var_.store(static_cast<unsigned>(count), std::memory_order_relaxed);
}

extern "C" int omp_get_num_procs(void) {
    ensure_pool_();
    return topology_ ? static_cast<int>(fu_logical_cores_count(topology_)) : 1;
}

extern "C" int omp_get_thread_limit(void) { return omp_get_max_threads(); }

/*  Nesting is refused, so these must answer honestly rather than politely: a caller told "yes, nesting is on" would
 *  build an algorithm around parallelism it never gets.
 */
extern "C" void omp_set_nested(int nested) { (void)nested; }
extern "C" int omp_get_nested(void) { return 0; }

/*  The trailing-underscore spellings are the Fortran bindings' symbol names, which libgomp exports alongside the C
 *  ones. faiss imports them, and a shim that shadows only half of a pair leaves it calling into two runtimes at once.
 */
extern "C" int omp_get_num_threads_(void) { return omp_get_num_threads(); }
extern "C" int omp_get_thread_num_(void) { return omp_get_thread_num(); }
extern "C" int omp_get_max_threads_(void) { return omp_get_max_threads(); }
extern "C" void omp_set_num_threads_(int const *count) { omp_set_num_threads(count ? *count : 0); }
extern "C" void omp_set_dynamic(int dynamic) { (void)dynamic; }
extern "C" int omp_get_dynamic(void) { return 0; }
extern "C" void omp_set_max_active_levels(int levels) { (void)levels; }
extern "C" int omp_get_max_active_levels(void) { return 1; }

#pragma endregion Runtime Queries

#if defined(__GNUC__)
#pragma GCC visibility pop
#endif
