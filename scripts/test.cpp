/**
 *  @brief Unit and stress tests for the pools, the harvested topology, and the parallel algorithms.
 *  @author Ash Vardanian
 *  @file scripts/test.cpp
 *  @date May 2, 2025
 *
 *  Every test is a `static void test_*` reached through the `unit_tests` and `stress_tests` tables in
 *  `main`, so a test absent from its table does not run - the tables are the only index that cannot
 *  drift from what executes. Asserts stay live whatever the build type, which is what the `#undef
 *  NDEBUG` below buys: a failure that only reproduces in Release is the one worth catching.
 */
#include <cstdio>  // `std::printf`, `std::fprintf`
#include <cstdlib> // `EXIT_FAILURE`, `EXIT_SUCCESS`
#include <cstring> // `std::strrchr`

#include <algorithm>   // `std::sort`
#include <bit>         // `std::bit_floor`
#include <type_traits> // `std::is_integral`, `std::is_enum`
#include <vector>      // `std::vector`

#include <forkunion.hpp>

namespace fu = ashvardanian::forkunion;

#undef NDEBUG // ? Keep any library asserts live in the test binary

/*  `backtrace` is a glibc/Apple facility; Bionic, FreeBSD, and musl ship `<execinfo.h>` without it. */
#if FU_ON_POSIX
#include <csignal>  // `std::signal`, `std::raise`
#include <unistd.h> // `::write`, `STDERR_FILENO`
#if (FU_ON_GLIBC || FU_ON_APPLE) && __has_include(<execinfo.h>)
#include <execinfo.h> // `::backtrace`, `::backtrace_symbols_fd`
#define FU_TEST_WITH_BACKTRACE_ 1
#endif
#endif

/*  Correctness only: the throughput/stress suite hammers a race window a slow emulator neither
 *  reproduces nor runs in tolerable time. `FORKUNION_TEST_SKIP_STRESS` in CMake defines this to 1 for
 *  the cross builds; native builds leave it 0 and run the full suite.  */
#ifndef FU_TEST_SKIP_STRESS_
#define FU_TEST_SKIP_STRESS_ 0
#endif

/** Formats an integral, pointer, enum, or bool into @p buffer; anything else prints `?`. */
template <typename value_type_>
static void format_value_(char *buffer, std::size_t capacity, value_type_ const &value) noexcept {
    if constexpr (std::is_same<value_type_, bool>::value)
        std::snprintf(buffer, capacity, "%s", value ? "true" : "false");
    else if constexpr (std::is_enum<value_type_>::value)
        format_value_(buffer, capacity, static_cast<typename std::underlying_type<value_type_>::type>(value));
    else if constexpr (std::is_pointer<value_type_>::value)
        std::snprintf(buffer, capacity, "%p", static_cast<void const *>(value));
    else if constexpr (std::is_integral<value_type_>::value && std::is_signed<value_type_>::value)
        std::snprintf(buffer, capacity, "%lld", static_cast<long long>(value));
    else if constexpr (std::is_integral<value_type_>::value)
        std::snprintf(buffer, capacity, "%llu", static_cast<unsigned long long>(value));
    else std::snprintf(buffer, capacity, "?");
}

/** Prints a located `FAIL` and exits; the preceding `Running ...` line already names the test. */
[[noreturn]] static void report_failure_(char const *file, int line, char const *expr, char const *detail) noexcept {
    fu::logging_colors_t colors;
    char const *slash = std::strrchr(file, '/');
    char const *base = slash ? slash + 1 : file;
    std::printf("%sFAIL%s\n", colors.bold_red(), colors.reset());
    std::fflush(stdout); // ? Order the `FAIL` line before the stderr detail
    std::fprintf(stderr, "  %s:%d: %s%s\n", base, line, expr, detail);
    std::exit(EXIT_FAILURE);
}

static void expect_(bool condition, char const *expr, char const *file, int line) noexcept {
    if (!condition) report_failure_(file, line, expr, "");
}

/** Compares two values, printing both sides on mismatch; `==` and `!=` share this path. */
template <typename a_type_, typename b_type_>
static void expect_cmp_(bool ok, a_type_ const &a, b_type_ const &b, char const *expr, char const *file,
                        int line) noexcept {
    if (ok) return;
    char lhs[48], rhs[48], detail[112];
    format_value_(lhs, sizeof(lhs), a);
    format_value_(rhs, sizeof(rhs), b);
    std::snprintf(detail, sizeof(detail), "  (%s vs %s)", lhs, rhs);
    report_failure_(file, line, expr, detail);
}

/** Notes a skipped test inline; the runner still prints `PASS` after the test returns. */
static void skip_(char const *reason) noexcept {
    fu::logging_colors_t colors;
    std::printf("%s(skip: %s)%s ", colors.dim(), reason, colors.reset());
}

/** Evaluates each side once, so a read-modify-write may sit inside the comparison. */
template <typename a_type_, typename b_type_>
static void expect_eq_(a_type_ const &a, b_type_ const &b, char const *expr, char const *file, int line) noexcept {
    expect_cmp_(a == b, a, b, expr, file, line);
}
template <typename a_type_, typename b_type_>
static void expect_ne_(a_type_ const &a, b_type_ const &b, char const *expr, char const *file, int line) noexcept {
    expect_cmp_(a != b, a, b, expr, file, line);
}

#define expect(cond) expect_((cond), #cond, __FILE__, __LINE__)
#define expect_eq(a, b) expect_eq_((a), (b), #a " == " #b, __FILE__, __LINE__)
#define expect_ne(a, b) expect_ne_((a), (b), #a " != " #b, __FILE__, __LINE__)
#define fail(reason) report_failure_(__FILE__, __LINE__, (reason), "")
#define skip(reason)   \
    do {               \
        skip_(reason); \
        return;        \
    } while (0)

#if FU_ON_POSIX
/** Adds a backtrace to a fatal signal; the flushed `Running ...` line names the test. */
extern "C" void on_fatal_signal_(int signal_number) noexcept {
    ssize_t const written = ::write(STDERR_FILENO, "\nCRASH - backtrace:\n", 20);
    (void)written; // ? Best-effort in a handler; still re-raise below
#if FU_TEST_WITH_BACKTRACE_
    void *frames[64];
    ::backtrace_symbols_fd(frames, ::backtrace(frames, 64), STDERR_FILENO);
#endif
    std::signal(signal_number, SIG_DFL);
    std::raise(signal_number);
}
static void install_crash_handlers_() noexcept {
    for (int signal_number : {SIGSEGV, SIGABRT, SIGFPE, SIGILL, SIGBUS}) std::signal(signal_number, on_fatal_signal_);
}
#else
static void install_crash_handlers_() noexcept {}
#endif

using fu32_t =
    fu::flat_pool<std::allocator<std::thread>, fu::standard_yield_t, fu::preferred_cache_hints_t, std::uint32_t>;
using fu16_t =
    fu::flat_pool<std::allocator<std::thread>, fu::standard_yield_t, fu::preferred_cache_hints_t, std::uint16_t>;
using fu8_t =
    fu::flat_pool<std::allocator<std::thread>, fu::standard_yield_t, fu::preferred_cache_hints_t, std::uint8_t>;

/*
 *  Explicitly instantiate the thread-pools to cover all of their logic, but avoid the
 *  "duplicate explicit instantiation" error on platforms where `std::size_t` is `uint64_t`.
 *
 *  template class fu::flat_pool<std::allocator<std::thread>, fu::standard_yield_t, std::size_t>
 */
template class fu::flat_pool<std::allocator<std::thread>, fu::standard_yield_t, fu::preferred_cache_hints_t,
                             std::uint32_t>;
template class fu::flat_pool<std::allocator<std::thread>, fu::standard_yield_t, fu::preferred_cache_hints_t,
                             std::uint16_t>;
template class fu::flat_pool<std::allocator<std::thread>, fu::standard_yield_t, fu::preferred_cache_hints_t,
                             std::uint8_t>;

#if FU_WITH_COLOCATE_POOLS_ON_DOMAIN
template struct fu::colocated_pool<>;
template struct fu::distributed_pool<>;
#endif

/**
 *  @brief Exhausts every tasks-and-threads pair in the index space, checking the split is a partition.
 *
 *  Every task index must fall in bounds and be visited exactly once across the per-thread subranges;
 *  a gap or an overlap here would mean lost or double-dispatched work in every static scheduler.
 */
template <typename index_type_ = std::uint8_t>
void test_indexed_split() noexcept {
    std::size_t max_tasks = std::numeric_limits<index_type_>::max();
    std::size_t max_threads = std::numeric_limits<index_type_>::max();
    std::vector<bool> visits(max_tasks);

    for (std::size_t threads = 1; threads < max_threads; ++threads) {
        for (std::size_t tasks = 0; tasks < max_tasks; ++tasks) {

            // Reset visits for each test case
            std::fill_n(visits.begin(), max_tasks, false);

            fu::indexed_split<index_type_> split {static_cast<index_type_>(tasks), static_cast<index_type_>(threads)};
            for (std::size_t thread = 0; thread < threads; ++thread) {
                auto subrange = split[static_cast<index_type_>(thread)];
                for (std::size_t task = subrange.first; task < subrange.first + subrange.count; ++task) {
                    expect(task < tasks);  // In bounds
                    expect(!visits[task]); // Not already visited
                    visits[task] = true;   // Mark as visited
                }
            }
        }
    }
}

/**
 *  @brief Exhausts every start-end-seed triple in the index space, checking each walk permutes.
 *
 *  Counting alone can't tell a permutation from a walk that revisits some values and skips others,
 *  so each value's first visit is recorded - a revisit makes a stealing thread drain a victim twice.
 */
template <typename index_type_ = std::uint8_t>
void test_coprime_permutation() noexcept {
    constexpr std::size_t max_tasks = std::numeric_limits<index_type_>::max();
    std::vector<bool> visits(max_tasks);

    for (std::size_t start = 0; start < max_tasks; ++start) {
        for (std::size_t end = start + 1; end < max_tasks; ++end) {
            for (std::size_t seed = 0; seed < max_tasks; ++seed) {

                // Create a coprime permutation and make sure it only covers the range [start, end)
                index_type_ const range_size = static_cast<index_type_>(end - start);
                fu::coprime_permutation_range<index_type_> permutation(static_cast<index_type_>(start), range_size,
                                                                       static_cast<index_type_>(seed));

                // Reset visits for each test case
                std::fill_n(visits.begin(), max_tasks, false);

                // Counting alone can't tell a permutation from a walk that revisits some values and
                // skips others - and a walk that revisits makes a work-stealing thread drain the same
                // victim twice, overrunning its cursor. Check that each value appears exactly once.
                std::size_t count_matches = 0;
                for (auto value : permutation) {
                    expect(value >= start); // In range
                    expect(value < end);

                    std::size_t const offset = static_cast<std::size_t>(value) - start;
                    expect(!visits[offset]); // ! Revisited a value, so this is not a permutation

                    visits[offset] = true;
                    count_matches++;
                }
                expect_eq(count_matches, range_size); // Every value in the range was covered
            }
        }
    }
}

/**
 *  @brief Checks that a harvested topology is internally consistent, on whatever host runs it.
 *
 *  Deliberately @b not gated on `FU_WITH_PLACE_MEMORY_ON_DOMAIN`: some platforms harvest a topology without
 *  compiling the NUMA pools, so gating this on the pools leaves their harvest wholly untested.
 *  A host with no harvest at all reports `false` and is skipped rather than failed - the absence
 *  of a topology is not a broken topology.
 */
static void test_topology_invariants() noexcept {
    fu::machine_topology_t topology;
    if (fu::failed(topology.harvest())) skip("no topology"); // ? No harvest on this host; nothing to check

    std::size_t const compute_domains = topology.compute_domains_count();
    std::size_t const memory_domains = topology.memory_domains_count();
    std::size_t const compute_levels = topology.compute_levels_count();
    expect(compute_domains != 0);
    expect(memory_domains != 0);
    expect(compute_levels != 0);

    // Levels are dense ranks over domains, so they can never outnumber the domains they rank.
    expect(compute_levels <= compute_domains);

    // Levels are a *dense* rank, so each of [0, levels) must be claimed by at least one domain.
    // Merely staying in range is too weak - it would accept a count inflated past the distinct levels.
    std::vector<bool> compute_level_seen(compute_levels, false);

    std::size_t cores_across_domains = 0;
    for (std::size_t i = 0; i < compute_domains; ++i) {
        fu::compute_domain_t const &domain = topology.compute_domain_at(static_cast<fu::compute_domain_index_t>(i));
        expect(domain.logical_cores_count != 0);
        expect(domain.first_core_id != nullptr);
        expect(domain.compute_level < compute_levels);
        expect(domain.memory_domain_index < memory_domains);
        compute_level_seen[domain.compute_level] = true;
        cores_across_domains += domain.logical_cores_count;
    }
    // Every core must belong to exactly one compute domain.
    expect_eq(cores_across_domains, topology.logical_cores_count());

    for (std::size_t level = 0; level < compute_levels; ++level)
        expect(compute_level_seen[level]); // ? An unclaimed rank means the count is inflated
}

constexpr std::size_t default_parallel_tasks_k = 10000; // 10K

/*  The maker vocabulary the whole three-tier test table hangs on: every templated test constructs
 *  its pool through a maker and spawns it on `maker.scope()`, so the same body runs against the
 *  flat, colocated, and distributed pools without knowing which one it drives.  */

/** Makes `flat_pool_t`s scoped to a thread count - the allowed cores, times oversubscription. */
struct make_pool_t {
    fu::flat_pool_t construct() const noexcept { return fu::flat_pool_t(); }
    std::size_t scope(std::size_t oversubscription = 1) const noexcept {
        return fu::allowed_cores_count() * oversubscription;
    }
};

#if FU_WITH_COLOCATE_POOLS_ON_DOMAIN
static fu::machine_topology_t machine_topology;

/** Makes `colocated_pool_t`s scoped to the machine's first compute domain. */
struct make_colocated_pool_t {
    fu::colocated_pool_t construct() const noexcept { return fu::colocated_pool_t("forkunion"); }
    fu::compute_domain_t scope(std::size_t = 0) const noexcept {
        return machine_topology.compute_domain_at(fu::compute_domain_index_t {});
    }
};

/** Makes `distributed_pool_t`s scoped to the whole harvested topology. */
struct make_distributed_pool_t {
    fu::distributed_pool_t construct() const noexcept { return fu::distributed_pool_t("forkunion"); }
    fu::machine_topology_t const &scope(std::size_t = 0) const noexcept { return machine_topology; }
};
#endif

#if FU_WITH_COLOCATE_POOLS_ON_DOMAIN
/**
 *  @brief The tier derivation must rank a synthetic edge log by bandwidth first, latency second.
 *
 *  A tier is a property of the medium: HBM splits from DDR on bandwidth despite a worse latency,
 *  two DDR sockets within both jitter bands share a tier, a same-bandwidth medium past the 1.5x
 *  latency band splits, CXL trails on either key, an unobserved domain lands one tier past the
 *  slowest, and a worse duplicate recording must not move its domain - the envelope keeps the
 *  best per metric.
 */
static void test_fabric_level_derivation() noexcept {
    fu::measured_edge_t const edges[] = {
        {fu::compute_domain_index_t {0}, fu::memory_domain_index_t {0}, 110, 3'000'000}, // ? HBM: 3 TB/s
        {fu::compute_domain_index_t {0}, fu::memory_domain_index_t {1}, 100, 300'000},   // ? DDR, faster latency
        {fu::compute_domain_index_t {0}, fu::memory_domain_index_t {2}, 104, 290'000},   // ? DDR, second socket
        {fu::compute_domain_index_t {0}, fu::memory_domain_index_t {3}, 250, 100'000},   // ? CXL expander
        {fu::compute_domain_index_t {0}, fu::memory_domain_index_t {0}, 400, 500'000},   // ? Worse repeat, ignored
        {fu::compute_domain_index_t {0}, fu::memory_domain_index_t {5}, 170, 285'000},   // ? DDR-wide, 1.7x latency
        // ? Domain 4 stays unobserved - a cpuless target no worker could first-touch
    };
    std::size_t levels[6];
    std::size_t scratch[24];
    std::size_t const count = fu::derive_memory_levels_(edges, sizeof(edges) / sizeof(edges[0]), levels, 6, scratch);

    expect_eq(count, std::size_t(5));     // HBM, DDR, the laggard, CXL, and the unobserved tier
    expect_eq(levels[0], std::size_t(0)); // Bandwidth ranks HBM first despite the worse latency
    expect_eq(levels[1], std::size_t(1)); // DDR opens the next tier
    expect_eq(levels[2], std::size_t(1)); // The second socket sits within both jitter bands
    expect_eq(levels[5], std::size_t(2)); // Same bandwidth band, but latency past 1.5x splits
    expect_eq(levels[3], std::size_t(3)); // CXL trails on both keys
    expect_eq(levels[4], std::size_t(4)); // Unobserved shares one tier past the slowest
}

/**
 *  A harvested fabric must cover every reachable edge with sane bounds and leave
 *  unreachable ones unwalked. No local-beats-remote assertion on purpose: emulated-NUMA
 *  guests legitimately measure every edge the same.
 */
static void test_measured_fabric() noexcept {
    fu::machine_topology_t const &topology = machine_topology;
    alignas(fu::default_alignment_k) fu::distributed_pool<fu::preferred_yield_t> pool;
    expect(fu::succeeded(pool.spawn(topology)));

    fu::measured_fabric_t fabric;
    expect(fabric.memory_latency(fu::compute_domain_index_t {}, fu::memory_domain_index_t {}) == 0);
    expect(fabric.memory_levels_count() == 1); // ? Unharvested: zeros everywhere, one tier
    expect(fu::succeeded(fabric.harvest(topology, pool)));
    expect_eq(fabric.compute_domains_count(), topology.compute_domains_count());
    expect_eq(fabric.memory_domains_count(), topology.memory_domains_count());

    for (std::size_t domain = 0; domain != pool.compute_domains_count(); ++domain) {
        fu::compute_domain_index_t const initiator = static_cast<fu::compute_domain_index_t>(domain);
        fu::memory_domain_index_t const local = topology.local_memory_of(initiator);
        expect(fabric.memory_distance(initiator, local) == 10); // The local edge anchors the SLIT scale

        for (std::size_t target = 0; target != topology.memory_domains_count(); ++target) {
            fu::memory_domain_index_t const to = static_cast<fu::memory_domain_index_t>(target);
            std::size_t const measured = fabric.memory_latency(initiator, to);

            // The harvest only walks edges some pool worker can first-touch, so a cpuless
            // domain - a CXL expander, or a node whose cores sit outside the pool - stays at zero
            // and answers with the unmeasured-remote fallback.
            bool reachable = false;
            for (std::size_t other = 0; other != pool.compute_domains_count() && !reachable; ++other)
                reachable =
                    topology.compute_domain_at(static_cast<fu::compute_domain_index_t>(other)).memory_domain_index ==
                    to;
            if (!reachable) {
                expect(measured == 0);
                expect(fabric.memory_distance(initiator, to) == 20);
                continue;
            }

            expect(measured > 0);       // Every reachable edge was walked
            expect(measured < 100'000); // A dependent load is nanoseconds, not a tenth of a millisecond

            std::size_t const streamed = fabric.memory_bandwidth(initiator, to);
            expect(streamed > 0);           // Every reachable edge was streamed
            expect(streamed < 100'000'000); // 100 TB/s exceeds any fabric - and catches an elided sum

            expect(fabric.memory_distance(initiator, to) >= 10); // Local is the row's floor
        }
    }

    // The derived tiers are dense ordinals: some domain is tier 0, none reaches the count.
    bool some_domain_is_fastest = false;
    for (std::size_t target = 0; target != topology.memory_domains_count(); ++target) {
        std::size_t const level = fabric.memory_level_in(static_cast<fu::memory_domain_index_t>(target));
        expect(level < fabric.memory_levels_count());
        some_domain_is_fastest |= level == 0;
    }
    expect(some_domain_is_fastest);

    // Bulk-snapshot semantics: a second harvest replaces the first and stays sane.
    expect(fu::succeeded(fabric.harvest(topology, pool)));
    expect(fabric.memory_latency(fu::compute_domain_index_t {},
                                 topology.local_memory_of(fu::compute_domain_index_t {})) > 0);
}
#endif // FU_WITH_COLOCATE_POOLS_ON_DOMAIN

/** Zero threads is not a pool: the spawn must be rejected cleanly, not crash or hang. */
static void test_spawn_zero() noexcept {
    fu::flat_pool_t pool;
    expect(fu::failed(pool.spawn(0u)));
}

/** The default spawn - one thread per allowed core - must succeed on every pool type. */
template <typename make_pool_type_ = make_pool_t>
static void test_spawn_success() noexcept {
    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    expect(fu::succeeded(pool.spawn(maker.scope())));
}

/** The pool is the single source of truth for exclusivity, even across re-spawns. */
template <typename make_pool_type_ = make_pool_t>
static void test_caller_exclusivity_query() noexcept {
    auto maker = make_pool_type_ {};
    auto pool = maker.construct();

    expect(fu::succeeded(pool.spawn(maker.scope(), fu::caller_inclusive_k)));
    expect_eq(pool.caller_exclusivity(), fu::caller_inclusive_k);

    // Re-spawn with the opposite mode: the query must follow, not a stale cache
    pool.terminate();
    expect(fu::succeeded(pool.spawn(maker.scope(), fu::caller_exclusive_k)));
    expect_eq(pool.caller_exclusivity(), fu::caller_exclusive_k);
}

/** Make sure that `for_threads` is called from each thread. */
template <typename make_pool_type_ = make_pool_t>
static void test_for_threads() noexcept {

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    expect(fu::succeeded(pool.spawn(maker.scope())));

    std::vector<std::atomic<bool>> visited(pool.threads_count());
    pool.for_threads([&](std::size_t const thread_index) noexcept { //
        visited[thread_index].store(true, std::memory_order_relaxed);
    });

    for (std::size_t i = 0; i < pool.threads_count(); ++i) expect(visited[i]);
}

/** Make sure that `unsafe_for_threads` is called from each thread. */
template <typename make_pool_type_ = make_pool_t>
static void test_unsafe_for_threads() noexcept {

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    expect(fu::succeeded(pool.spawn(maker.scope())));

    std::vector<std::atomic<bool>> visited(pool.threads_count());
    auto on_each_thread = [&](std::size_t const thread_index) noexcept {
        visited[thread_index].store(true, std::memory_order_relaxed);
    };
    pool.unsafe_for_threads(on_each_thread);
    pool.unsafe_join();

    for (std::size_t i = 0; i < pool.threads_count(); ++i) expect(visited[i]);
}

/** Tests generation token polling with two caller-exclusive pools. */
template <typename make_pool_type_ = make_pool_t>
static void test_generation_polling() noexcept {

    auto maker = make_pool_type_ {};
    auto pool_a = maker.construct();
    auto pool_b = maker.construct();

    // Polling before join is the caller-exclusive pattern: on inclusive pools the
    // caller owes a slice that only runs inside `unsafe_join`.
    expect(fu::succeeded(pool_a.spawn(maker.scope(), fu::caller_exclusive_k)));
    expect(fu::succeeded(pool_b.spawn(maker.scope(), fu::caller_exclusive_k)));

    std::vector<std::atomic<bool>> visited_a(pool_a.threads_count());
    std::vector<std::atomic<bool>> visited_b(pool_b.threads_count());

    auto work_a = [&](std::size_t const thread_index) noexcept {
        visited_a[thread_index].store(true, std::memory_order_relaxed);
    };
    auto work_b = [&](std::size_t const thread_index) noexcept {
        visited_b[thread_index].store(true, std::memory_order_relaxed);
    };

    // Drive one pool through the raw token API and the other through the RAII guard
    auto generation_a = pool_a.unsafe_for_threads(work_a);
    auto broadcast_b = pool_b.for_threads(work_b); // ? Dispatches at construction: exclusive pool
    expect((generation_a & 1u) != 0);              // ? Tokens are always odd
    expect((broadcast_b.generation() & 1u) != 0);

    // Poll both pools until complete
    bool a_done = false, b_done = false;
    while (!a_done || !b_done) {
        if (!a_done) a_done = pool_a.is_complete(generation_a);
        if (!b_done) b_done = broadcast_b.is_complete();
    }

    pool_a.unsafe_join(generation_a);
    broadcast_b.join();

    for (std::size_t i = 0; i < pool_a.threads_count(); ++i) expect(visited_a[i]);
    for (std::size_t i = 0; i < pool_b.threads_count(); ++i) expect(visited_b[i]);
}

/** Verifies guard timing: dispatch at construction on exclusive pools, at join on inclusive. */
template <typename make_pool_type_ = make_pool_t>
static void test_guard_lifecycle() noexcept {

    auto maker = make_pool_type_ {};

    // On exclusive pools the work starts at construction, before `join`:
    {
        auto pool = maker.construct();
        expect(fu::succeeded(pool.spawn(maker.scope(), fu::caller_exclusive_k)));

        std::atomic<std::size_t> visited_count {0};
        auto count_visits = [&](std::size_t) noexcept { visited_count.fetch_add(1, std::memory_order_relaxed); };
        auto broadcast = pool.for_threads(count_visits);
        expect(broadcast.generation() != 0);        // ? Must be dispatched at construction
        expect((broadcast.generation() & 1u) != 0); // ? Tokens are always odd
        while (!broadcast.is_complete()) {}         // ? Wait without joining
        expect_eq(visited_count.load(std::memory_order_relaxed), pool.threads_count());
        broadcast.join();
    }

    // On inclusive pools no work may start before `join`:
    {
        auto pool = maker.construct();
        expect(fu::succeeded(pool.spawn(maker.scope(), fu::caller_inclusive_k)));

        std::atomic<std::size_t> visited_count {0};
        auto count_visits = [&](std::size_t) noexcept { visited_count.fetch_add(1, std::memory_order_relaxed); };
        auto broadcast = pool.for_threads(count_visits);
        expect(broadcast.generation() == 0); // ? Must be deferred to join
        expect(!broadcast.is_complete());
        expect(visited_count.load(std::memory_order_relaxed) == 0);
        broadcast.join();
        expect(broadcast.is_complete());
        expect_eq(visited_count.load(std::memory_order_relaxed), pool.threads_count());
    }
}

/** Dropping the guard without an explicit `join` must still join in the destructor. */
template <typename make_pool_type_ = make_pool_t>
static void test_guard_destructor_joins() noexcept {

    auto maker = make_pool_type_ {};
    for (fu::caller_exclusivity_t const exclusivity : {fu::caller_exclusive_k, fu::caller_inclusive_k}) {
        auto pool = maker.construct();
        expect(fu::succeeded(pool.spawn(maker.scope(), exclusivity)));

        std::atomic<std::size_t> visited_count {0};
        auto count_visits = [&](std::size_t) noexcept { visited_count.fetch_add(1, std::memory_order_relaxed); };
        { auto broadcast = pool.for_threads(count_visits); } // ! The destructor is the only join here
        expect_eq(visited_count.load(std::memory_order_relaxed), pool.threads_count());
    }
}

/** Covers the caller-as-contributor protocol on inclusive pools. */
template <typename make_pool_type_ = make_pool_t>
static void test_generation_inclusive() noexcept {

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    expect(fu::succeeded(pool.spawn(maker.scope(), fu::caller_inclusive_k)));

    std::vector<std::atomic<bool>> visited(pool.threads_count());
    auto mark_visited = [&](std::size_t const thread_index) noexcept {
        visited[thread_index].store(true, std::memory_order_relaxed);
    };

    auto generation = pool.unsafe_for_threads(mark_visited);
    expect((generation & 1u) != 0);        // ? Tokens are always odd
    expect(!pool.is_complete(generation)); // ? Impossible before the caller's slice
    pool.unsafe_join(generation);          // ? Runs the caller's slice, then waits
    expect(pool.is_complete(generation));
    for (std::size_t i = 0; i < pool.threads_count(); ++i) expect(visited[i]);
    pool.unsafe_join(generation); // ? Idempotent: double-join must be a no-op
}

/** Degenerate single-thread inclusive pool: the caller is the only contributor. */
static void test_generation_single_thread() noexcept {
    fu::flat_pool_t pool;
    expect(fu::succeeded(pool.spawn(1))); // ? Default is caller-inclusive: zero workers

    std::atomic<bool> visited {false};
    auto mark_visited = [&](std::size_t) noexcept { visited.store(true, std::memory_order_relaxed); };

    auto generation = pool.unsafe_for_threads(mark_visited);
    expect((generation & 1u) != 0);
    expect(!pool.is_complete(generation)); // ? Nothing can complete before the caller's slice
    pool.unsafe_join(generation);          // ? The caller both runs and signals completion
    expect(pool.is_complete(generation));
    expect(visited.load(std::memory_order_relaxed));
}

/** The exclusive mirror: one lone worker, while the caller only polls and never contributes. */
static void test_generation_single_thread_exclusive() noexcept {
    fu::flat_pool_t pool;
    expect(fu::succeeded(pool.spawn(1, fu::caller_exclusive_k)));

    std::atomic<bool> visited {false};
    auto mark_visited = [&](std::size_t) noexcept { visited.store(true, std::memory_order_relaxed); };

    auto generation = pool.unsafe_for_threads(mark_visited);
    expect((generation & 1u) != 0);
    while (!pool.is_complete(generation)) {} // ? The lone worker finishes alone
    pool.unsafe_join(generation);
    expect(visited.load(std::memory_order_relaxed));
}

/** Hammers the dispatch/join race window with tight iterations on exclusive pools. */
template <typename make_pool_type_ = make_pool_t>
static void test_generation_stress() noexcept {

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    auto polled_pool = maker.construct();
    expect(fu::succeeded(pool.spawn(maker.scope(), fu::caller_exclusive_k)));
    expect(fu::succeeded(polled_pool.spawn(maker.scope(), fu::caller_exclusive_k)));

    std::atomic<std::size_t> counter {0};
    auto count_up = [&](std::size_t) noexcept { counter.fetch_add(1, std::memory_order_relaxed); };

    // A second in-flight pool is polled between iterations to stress `is_complete`
    auto polled_generation = polled_pool.unsafe_for_threads(count_up);

    constexpr std::size_t iterations_k = 10000;
    for (std::size_t iteration = 0; iteration < iterations_k; ++iteration) {
        auto generation = pool.unsafe_for_threads(count_up);
        expect((generation & 1u) != 0); // ? The old dispatch/completion race made these even
        (void)polled_pool.is_complete(polled_generation);
        pool.unsafe_join(generation);
        expect(pool.is_complete(generation));
    }
    polled_pool.unsafe_join(polled_generation);

    std::size_t const expected = pool.threads_count() * iterations_k + polled_pool.threads_count();
    expect_eq(counter.load(std::memory_order_relaxed), expected);
}

/** Overlaps an inclusive and an exclusive pool from one caller, through the raw token API. */
template <typename make_pool_type_ = make_pool_t>
static void test_exclusivity() noexcept {

    auto maker = make_pool_type_ {};
    auto first_pool = maker.construct();
    auto second_pool = maker.construct();
    expect(fu::succeeded(first_pool.spawn(maker.scope(), fu::caller_inclusive_k)));
    expect(fu::succeeded(second_pool.spawn(maker.scope(), fu::caller_exclusive_k)));

    std::size_t const first_size = first_pool.threads_count();
    std::size_t const second_size = second_pool.threads_count();
    std::size_t const total_size = first_size + second_size;
    std::vector<std::atomic<bool>> visited(total_size);

    // Externally defined lambdas with a clearly long lifetime:
    auto do_second = [&](std::size_t const thread_index) noexcept {
        visited[first_size + thread_index].store(true, std::memory_order_relaxed);
    };
    auto do_first = [&](std::size_t const thread_index) noexcept {
        visited[thread_index].store(true, std::memory_order_relaxed);
    };

    // Repeat the same logic a few times and check for correctness:
    for (std::size_t iteration = 0; iteration < 3; ++iteration) {
        auto second_generation = second_pool.unsafe_for_threads(do_second);
        auto first_generation = first_pool.unsafe_for_threads(do_first);
        first_pool.unsafe_join(first_generation); // ? Contributes the caller's slice: inclusive pool
        second_pool.unsafe_join(second_generation);

        for (std::size_t i = 0; i < total_size; ++i) expect(visited[i]);
    }
}

/** The same overlap through the guard API: inline lambdas re-packaged into returned objects. */
template <typename make_pool_type_ = make_pool_t>
static void test_exclusivity_inline_guards() noexcept {

    auto maker = make_pool_type_ {};
    auto first_pool = maker.construct();
    auto second_pool = maker.construct();
    expect(fu::succeeded(first_pool.spawn(maker.scope(), fu::caller_inclusive_k)));
    expect(fu::succeeded(second_pool.spawn(maker.scope(), fu::caller_exclusive_k)));

    std::size_t const first_size = first_pool.threads_count();
    std::size_t const second_size = second_pool.threads_count();
    std::size_t const total_size = first_size + second_size;
    std::vector<std::atomic<bool>> visited(total_size);

    auto join_second = second_pool.for_threads([&](std::size_t const thread_index) noexcept {
        visited[first_size + thread_index].store(true, std::memory_order_relaxed);
    });
    first_pool.for_threads(
        [&](std::size_t const thread_index) noexcept { visited[thread_index].store(true, std::memory_order_relaxed); });
    join_second.join();

    for (std::size_t i = 0; i < total_size; ++i) expect(visited[i]);
}

/** Make sure that `for_n` is called from each thread. */
template <typename make_pool_type_ = make_pool_t>
static void test_uncomfortable_input_size() noexcept {

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    expect(fu::succeeded(pool.spawn(maker.scope())));

    std::size_t const max_input_size = pool.threads_count() * 3; // Arbitrary size, larger than the number of threads
    for (std::size_t input_size = 0; input_size <= max_input_size; ++input_size) {
        std::atomic<bool> out_of_bounds(false);
        std::atomic<std::size_t> executed(0);
        auto probe = [&](std::size_t const task, fu::thread_in_domain_t) noexcept {
            if (task >= input_size) out_of_bounds.store(true, std::memory_order_relaxed);
            executed.fetch_add(1, std::memory_order_relaxed);
        };

        // Both schedulers must cover the awkward sizes exactly - a dropped task is as wrong as a stray one.
        pool.for_n(input_size, probe);
        expect_eq(executed.load(std::memory_order_relaxed), input_size);

        executed.store(0, std::memory_order_relaxed);
        pool.for_n_dynamic(input_size, probe);
        expect_eq(executed.load(std::memory_order_relaxed), input_size);
        expect(!out_of_bounds.load(std::memory_order_relaxed));
    }
}

/**
 *  @brief One `for_slices` dispatch of @p n tasks must partition [0, n) into non-empty slices.
 *
 *  A slice callback receives a `tasks_range_t`, so a broken split shows up as an index covered
 *  twice, never, or past the range. Every thread is dispatched exactly once, an idle one with an
 *  empty range, and no dispatch may produce more ranges than there are threads.
 */
template <typename pool_type_>
static void expect_for_slices_cover_(pool_type_ &pool, std::size_t const n) noexcept {
    std::vector<std::atomic<unsigned>> executions(n);
    std::atomic<std::size_t> slices_count {0};
    std::atomic<bool> out_of_bounds {false};

    pool.for_slices(n, [&](fu::tasks_range_t range, fu::thread_in_domain_t) noexcept {
        if (range.first + range.count > n) {
            out_of_bounds.store(true, std::memory_order_relaxed);
            return;
        }
        slices_count.fetch_add(1, std::memory_order_relaxed);
        for (std::size_t const task : range) executions[task].fetch_add(1, std::memory_order_relaxed);
    });

    // Every thread is dispatched exactly once, whether or not it drew any tasks.
    expect_eq(slices_count.load(std::memory_order_relaxed), pool.threads_count());
    expect(!out_of_bounds.load(std::memory_order_relaxed));
    expect(slices_count.load(std::memory_order_relaxed) <= pool.threads_count());
    for (std::size_t i = 0; i < n; ++i) expect_eq(executions[i].load(std::memory_order_relaxed), 1u);
}

/** Sweeps `for_slices` through the awkward sizes around the thread count and one large run. */
template <typename make_pool_type_ = make_pool_t>
static void test_for_slices() noexcept {
    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    expect(fu::succeeded(pool.spawn(maker.scope())));

    std::size_t const threads = pool.threads_count();
    std::size_t const sizes[] = {0, 1, threads - 1, threads, threads + 1, 3 * threads, default_parallel_tasks_k};
    for (std::size_t const n : sizes) expect_for_slices_cover_(pool, n);
}

/** A joined token must stay complete even while a newer fork is in flight. */
template <typename make_pool_type_ = make_pool_t>
static void test_stale_generation_completes() noexcept {
    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    expect(fu::succeeded(pool.spawn(maker.scope(), fu::caller_exclusive_k)));

    std::atomic<std::size_t> counter {0};
    auto count_up = [&](std::size_t) noexcept { counter.fetch_add(1, std::memory_order_relaxed); };

    auto stale = pool.unsafe_for_threads(count_up);
    pool.unsafe_join(stale);
    expect(pool.is_complete(stale));

    auto fresh = pool.unsafe_for_threads(count_up);
    expect_ne(stale, fresh);
    expect(pool.is_complete(stale)); // ? A joined token must not flip back while a new fork runs
    pool.unsafe_join(fresh);
    expect(pool.is_complete(fresh));
    expect_eq(counter.load(std::memory_order_relaxed), 2 * pool.threads_count());
}

/**
 *  @brief Hammers the spawn/terminate lifecycle; every re-spawned crew must dispatch exactly-once.
 *
 *  Worker ids are published by the workers themselves and reset on `terminate`, so lifecycle churn
 *  is where a stale id or an unjoined thread would surface - as a lost task or a crash on re-spawn.
 */
template <typename make_pool_type_ = make_pool_t>
static void test_spawn_terminate_churn() noexcept {
    auto maker = make_pool_type_ {};
    auto pool = maker.construct();

    constexpr std::size_t rounds_k = 16;
    std::atomic<std::size_t> executed {0};
    for (std::size_t round = 0; round != rounds_k; ++round) {
        expect(fu::succeeded(pool.spawn(maker.scope())));
        executed.store(0, std::memory_order_relaxed);
        std::size_t const n = pool.threads_count() + round % 3; // ? Vary the remainder across rounds
        pool.for_n(
            n, [&](std::size_t, fu::thread_in_domain_t) noexcept { executed.fetch_add(1, std::memory_order_relaxed); });
        expect_eq(executed.load(std::memory_order_relaxed), n);
        pool.terminate();
    }
}

/** Two caller threads drive two independent pools at once; no cross-pool state may bleed. */
static void test_concurrent_caller_threads() noexcept {
    constexpr std::size_t tasks_k = 4096;
    std::atomic<bool> first_ok {true}, second_ok {true};

    auto drive = [&](std::atomic<bool> &ok) noexcept {
        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(fu::allowed_cores_count()))) {
            ok.store(false, std::memory_order_relaxed);
            return;
        }
        std::atomic<std::size_t> counter {0};
        for (std::size_t round = 0; round != 8; ++round) {
            counter.store(0, std::memory_order_relaxed);
            pool.for_n(tasks_k, [&](std::size_t, fu::thread_in_domain_t) noexcept {
                counter.fetch_add(1, std::memory_order_relaxed);
            });
            if (counter.load(std::memory_order_relaxed) != tasks_k) {
                ok.store(false, std::memory_order_relaxed);
                return;
            }
        }
    };

    std::thread first([&] { drive(first_ok); });
    std::thread second([&] { drive(second_ok); });
    first.join();
    second.join();
    expect(first_ok.load(std::memory_order_relaxed));
    expect(second_ok.load(std::memory_order_relaxed));
}

/** Convenience structure to ensure we output match locations to independent cache lines. */
struct alignas(fu::default_alignment_k) aligned_visit_t {
    std::size_t task = 0;
    bool operator<(aligned_visit_t const &other) const noexcept { return task < other.task; }
    bool operator==(aligned_visit_t const &other) const noexcept { return task == other.task; }
    bool operator!=(std::size_t other_index) const noexcept { return task != other_index; }
    bool operator==(std::size_t other_index) const noexcept { return task == other_index; }
};

/** Sorts the visit records in place and checks they cover [0, size) exactly once. */
bool contains_iota(std::vector<aligned_visit_t> &visited) noexcept {
    std::sort(visited.begin(), visited.end());
    std::size_t visited_progress = 0;
    for (; visited_progress < visited.size(); ++visited_progress)
        if (visited[visited_progress] != visited_progress) break;
    if (visited_progress != visited.size()) {
        return false; // ! Put on a separate line for a breakpoint
    }
    return true;
}

/** Make sure that `for_n` is called the right number of times with the right task indices. */
template <typename make_pool_type_ = make_pool_t>
static void test_for_n() noexcept {

    std::atomic<std::size_t> counter(0);
    std::vector<aligned_visit_t> visited(default_parallel_tasks_k);

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    expect(fu::succeeded(pool.spawn(maker.scope())));

    using pool_t = decltype(pool);

    pool.for_n(default_parallel_tasks_k, [&](std::size_t const task, fu::thread_in_domain_t at) noexcept {
        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = task;
    });

    // Make sure that all task indices are unique and form the full range of [0, `default_parallel_tasks_k`).
    expect_eq(counter.load(), default_parallel_tasks_k);
    expect(contains_iota(visited));

    // Make sure repeated calls to `for_n` work
    counter = 0;
    pool.for_n(default_parallel_tasks_k, [&](std::size_t const task, fu::thread_in_domain_t at) noexcept {
        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = task;
    });

    // Make sure that all task indices are unique and form the full range of [0, `default_parallel_tasks_k`).
    expect_eq(counter.load(), default_parallel_tasks_k);
    expect(contains_iota(visited));

    // Make sure `for_n` is being executed on different threads.
    std::vector<aligned_visit_t> visited_threads(pool.threads_count());
    constexpr std::size_t invalid_task = std::numeric_limits<std::size_t>::max();
    for (auto &visit : visited_threads) visit.task = invalid_task;
    pool.for_n(
        default_parallel_tasks_k, // ? Could have been an arbitrary number `>= pool.threads_count()`
        [&](std::size_t const, fu::thread_in_domain_t at) noexcept { visited_threads[at.thread].task = at.thread; });

    expect(contains_iota(visited_threads));
}

/** Make sure that `for_n_dynamic` is called the right number of times with the right task indices. */
template <typename make_pool_type_ = make_pool_t>
static void test_for_n_dynamic() noexcept {

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    expect(fu::succeeded(pool.spawn(maker.scope())));

    std::vector<aligned_visit_t> visited(default_parallel_tasks_k);
    std::atomic<std::size_t> counter(0);
    pool.for_n_dynamic(default_parallel_tasks_k, [&](std::size_t const task, fu::thread_in_domain_t) noexcept {
        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = task;
    });

    // Make sure that all task indices are unique and form the full range of [0, `default_parallel_tasks_k`).
    expect_eq(counter.load(), default_parallel_tasks_k);
    expect(contains_iota(visited));

    // Make sure repeated calls to `for_n` work
    counter = 0;
    pool.for_n_dynamic(default_parallel_tasks_k, [&](std::size_t const task, fu::thread_in_domain_t) noexcept {
        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = task;
    });

    expect_eq(counter.load(), default_parallel_tasks_k);
    expect(contains_iota(visited));
}

/**
 *  @brief Stalls one thread and checks its neighbours drain the slice it can't reach.
 *
 *  `for_n_dynamic` reserves a contiguous slice per thread and lets idle threads help drain the rest.
 *  Correctness must not depend on any thread keeping up: with one thread crawling, every task still
 *  runs, and runs @b once. The stalled thread must also end up executing fewer tasks than its own
 *  slice held - otherwise nobody stole, and the test is silently proving nothing.
 */
template <typename make_pool_type_ = make_pool_t>
static void test_for_n_dynamic_stealing() noexcept {

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    expect(fu::succeeded(pool.spawn(maker.scope())));

    std::size_t const threads = pool.threads_count();
    if (threads < 2) skip("needs 2+ threads"); // ? Nobody to steal from, and nobody to steal

    constexpr std::size_t tasks_k = 8192;

    std::atomic<std::size_t> total {0};
    std::atomic<std::size_t> thread_0_runs {0};
    std::vector<std::atomic<unsigned>> executions(tasks_k);
    for (auto &e : executions) e.store(0, std::memory_order_relaxed);

    pool.for_n_dynamic(tasks_k, [&](std::size_t const task, fu::thread_in_domain_t at) noexcept {
        // Thread 0 crawls: a spin, not a sleep, so the pool's own yields don't mask the stall.
        if (at.thread == 0) {
            volatile std::size_t sink = 0;
            for (std::size_t i = 0; i < 200000; ++i) sink = sink + i;
            thread_0_runs.fetch_add(1, std::memory_order_relaxed);
        }
        executions[task].fetch_add(1, std::memory_order_relaxed);
        total.fetch_add(1, std::memory_order_relaxed);
    });

    expect_eq(total.load(), tasks_k);                                              // ! Some task ran twice, or never
    for (std::size_t i = 0; i < tasks_k; ++i) expect_eq(executions[i].load(), 1u); // ! Every task exactly once

    // Thread 0's own slice - had nobody helped, it would have run all of it, plus a static prong.
    std::size_t const dynamic_tasks = tasks_k > threads ? tasks_k - threads : 0;
    fu::indexed_split<std::size_t> const split(dynamic_tasks, threads);
    std::size_t const own_slice = split[0].count;
    expect(thread_0_runs.load() < own_slice); // ! Nobody stole, so this test proves nothing
}

/**
 *  @brief One dynamic dispatch of @p n tasks must run exactly-once; @p crawl stalls a whole domain.
 *
 *  With domain 0 crawling, the other domains drain their own slices and must reach across the
 *  interconnect for more, so the steal path is exercised rather than silently idle. The owner of a
 *  task index mirrors the invoker's own split: domain `d` owns `split[d]`.
 *
 *  @note The crawl is @b per @b domain, not per thread. Stalling a single thread only forces a steal
 *      where a domain has few of them: with 64 threads to a domain, one crawler is 1/64th of its
 *      capacity, its neighbours absorb the slice, and no steal ever needs to cross - so the
 *      assertion below passed on small CI runners and was a coin-flip on real hardware.
 */
template <typename pool_type_>
static void expect_dynamic_regime_covers_(pool_type_ &pool, std::size_t const n, bool const crawl) noexcept {
    std::size_t const domains = pool.compute_domains_count();

    std::atomic<std::size_t> total {0};
    std::atomic<std::size_t> cross_domain_runs {0};
    std::vector<std::atomic<unsigned>> executions(n);
    for (auto &e : executions) e.store(0, std::memory_order_relaxed);

    pool.for_n_dynamic(n, [&](std::size_t const task, fu::thread_in_domain_t at) noexcept {
        // ? A spin, not a sleep, so the pool's yields don't mask it. Keyed on the executing thread's
        // ? domain, so a stealer from elsewhere runs the stolen task at full speed.
        if (crawl && at.compute_domain == 0) {
            volatile std::size_t sink = 0;
            for (std::size_t i = 0; i < 200000; ++i) sink = sink + i;
        }
        std::size_t const owner_domain = fu::indexed_split<std::size_t>(n, domains).index_of(task);
        if (owner_domain != static_cast<std::size_t>(at.compute_domain))
            cross_domain_runs.fetch_add(1, std::memory_order_relaxed);
        executions[task].fetch_add(1, std::memory_order_relaxed);
        total.fetch_add(1, std::memory_order_relaxed);
    });

    expect_eq(total.load(), n);                                              // ! Some task ran twice, or never
    for (std::size_t i = 0; i < n; ++i) expect_eq(executions[i].load(), 1u); // ! Every task exactly once
    if (crawl) expect(cross_domain_runs.load() > 0); // ! No steal crossed the interconnect; proves nothing
}

/**
 *  @brief Exhausts the distributed `for_n_dynamic` across task-count regimes, on a real multi-domain box.
 *
 *  Every boundary the two-level slicing exposes gets a regime - fewer tasks than threads, exactly as
 *  many, one more, and a large skewed run with a crawling thread. The index type here is
 *  `std::size_t`; the small-type wrap-around regimes live in the flat pool's `fu8`/`fu16` stress suite.
 */
template <typename make_pool_type_ = make_pool_t>
static void test_distributed_for_n_dynamic_exhaustive() noexcept {

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    expect(fu::succeeded(pool.spawn(maker.scope())));

    std::size_t const threads = pool.threads_count();
    std::size_t const domains = pool.compute_domains_count();
    if (domains < 2) skip("needs 2+ compute domains"); // ? The cross-domain walk is dead code otherwise
    if (threads < 4) skip("needs 4+ threads");

    std::size_t const regimes[] = {1, threads / 2, threads, threads + 1, 2 * threads + 3, 8192};
    for (std::size_t const n : regimes) {
        bool const crawl = n == 8192; // ? Only the large run needs forced imbalance
        expect_dynamic_regime_covers_(pool, n, crawl);
    }
}

/** Stress-tests the implementation by oversubscribing the number of threads. */
template <typename make_pool_type_ = make_pool_t>
static void test_oversubscribed_threads() noexcept {
    constexpr std::size_t oversubscription = 3;

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    expect(fu::succeeded(pool.spawn(maker.scope(oversubscription))));

    std::vector<aligned_visit_t> visited(default_parallel_tasks_k);
    std::atomic<std::size_t> counter(0);
    thread_local volatile std::size_t some_local_work = 0;
    pool.for_n_dynamic(default_parallel_tasks_k, [&](std::size_t const task, fu::thread_in_domain_t) noexcept {
        // Perform some weird amount of work, that is not very different between consecutive tasks.
        for (std::size_t i = 0; i != task % oversubscription; ++i) some_local_work = some_local_work + i * i;

        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = task;
    });

    // Make sure that all task indices are unique and form the full range of [0, `default_parallel_tasks_k`).
    expect_eq(counter.load(), default_parallel_tasks_k);
    expect(contains_iota(visited));
}

/**
 *  @brief Naps the workers between batches; every wake must still dispatch exactly-once.
 *
 *  `sleep` flips the pool to `chill_k` and, where the platform allows, demotes workers to the
 *  idle scheduling class; the next dispatch must wake and restore them. Losing a worker to a
 *  missed wake-up shows up here as a hung join, and a double-dispatched task as a broken iota.
 */
template <typename make_pool_type_ = make_pool_t>
static void test_sleep_wake() noexcept {
    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    expect(fu::succeeded(pool.spawn(maker.scope())));

    std::vector<aligned_visit_t> visited(default_parallel_tasks_k);
    std::atomic<std::size_t> counter {0};
    for (std::size_t batch = 0; batch != 4; ++batch) {
        pool.sleep(100); // ? Nap in 100 us intervals until the next dispatch
        counter.store(0, std::memory_order_relaxed);
        pool.for_n(default_parallel_tasks_k, [&](std::size_t const task, fu::thread_in_domain_t) noexcept {
            std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
            visited[count_populated].task = task;
        });
        expect_eq(counter.load(std::memory_order_relaxed), default_parallel_tasks_k);
        expect(contains_iota(visited));
    }
}

/** Make sure that that we can combine static & dynamic loads over the same pool with & w/out resetting. */
template <bool should_restart_, typename make_pool_type_ = make_pool_t>
static void test_mixed_restart() noexcept {

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    expect(fu::succeeded(pool.spawn(maker.scope())));

    std::vector<aligned_visit_t> visited(default_parallel_tasks_k);
    std::atomic<std::size_t> counter(0);

    pool.for_n(default_parallel_tasks_k, [&](std::size_t const task, fu::thread_in_domain_t) noexcept {
        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = task;
    });
    expect_eq(counter.load(), default_parallel_tasks_k);
    expect(contains_iota(visited));

    // Make sure that the pool can be reset and reused
    if (should_restart_) {
        pool.terminate();
        expect(fu::succeeded(pool.spawn(maker.scope())));
    }

    // Make sure repeated calls to `for_n` work
    counter = 0;
    pool.for_n_dynamic(default_parallel_tasks_k, [&](std::size_t const task, fu::thread_in_domain_t) noexcept {
        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = task;
    });

    expect_eq(counter.load(), default_parallel_tasks_k);
    expect(contains_iota(visited));
}

/**
 *  @brief Alternates static & dynamic dispatch over a small-index pool for many generations.
 *
 *  The epoch is the same width as the pool's index, so a `fu8` pool wraps it every 128
 *  dispatch/join cycles and a `fu16` every 32768. Enough @p cycles drives the counter through
 *  several wraps, proving the token parity and completion comparisons survive the overflow.
 */
template <typename pool_type_>
static void stress_test_composite(std::size_t const threads_count, std::size_t const parallel_tasks_count,
                                  std::size_t const cycles) noexcept {

    using pool_t = pool_type_;
    using index_t = typename pool_t::index_t;

    pool_t pool;
    expect(fu::succeeded(pool.spawn(static_cast<index_t>(threads_count))));

    std::atomic<std::size_t> counter(0);
    std::vector<aligned_visit_t> visited(parallel_tasks_count);

    // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
    auto log_visit = [&](index_t const task, typename pool_t::thread_in_domain_t) noexcept {
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = task;
    };
    auto dispatch_and_verify = [&](auto dispatch) noexcept {
        counter = 0;
        dispatch();
        expect_eq(counter.load(), parallel_tasks_count);
        expect(contains_iota(visited));
    };

    index_t const n = static_cast<index_t>(parallel_tasks_count);
    for (std::size_t cycle = 0; cycle != cycles; ++cycle) {
        dispatch_and_verify([&] { pool.for_n(n, log_visit); });
        dispatch_and_verify([&] { pool.for_n_dynamic(n, log_visit); });
    }
}

/** `replicated_array` is one uninitialized per-domain buffer; the caller fills and reads its slices. */
static void test_replicated_array() noexcept {
    fu::machine_topology_t topology;
    if (fu::failed(topology.harvest())) skip("no topology"); // ? No topology here; nothing to check

    std::size_t const n = 4096;
    fu::replicated_array<std::uint32_t> replicas;
    expect(fu::succeeded(replicas.resize_uninitialized(topology, n)));
    expect_eq(replicas.size(), n);
    expect_eq(replicas.memory_domains_count(), topology.memory_domains_count());

    // Fill every replica with a domain-dependent pattern, so replicas that aliased would be caught.
    for (std::size_t domain = 0; domain < replicas.memory_domains_count(); ++domain) {
        fu::span<std::uint32_t> const replica =
            replicas.on_memory_domain(static_cast<fu::memory_domain_index_t>(domain));
        expect_eq(replica.size(), n);
        for (std::size_t i = 0; i < n; ++i) replica[i] = static_cast<std::uint32_t>(domain * n + i);
    }
    for (std::size_t domain = 0; domain < replicas.memory_domains_count(); ++domain)
        for (std::size_t i = 0; i < n; ++i)
            expect_eq(replicas.at(static_cast<fu::memory_domain_index_t>(domain), i), domain * n + i);

    // A single-element replica is the smallest shape the stride math must survive.
    fu::replicated_array<std::uint32_t> tiny;
    expect(fu::succeeded(tiny.resize_uninitialized(topology, 1)));
    expect_eq(tiny.size(), 1u);
    for (std::size_t domain = 0; domain < tiny.memory_domains_count(); ++domain)
        tiny.at(static_cast<fu::memory_domain_index_t>(domain), 0) = static_cast<std::uint32_t>(domain);
    for (std::size_t domain = 0; domain < tiny.memory_domains_count(); ++domain)
        expect_eq(tiny.at(static_cast<fu::memory_domain_index_t>(domain), 0), domain);
}

/** `sharded_array` stores each element once; `location_of`/`logical_index_of` round-trip the segment map. */
static void test_sharded_array() noexcept {
    fu::machine_topology_t topology;
    if (fu::failed(topology.harvest())) skip("no topology");

    std::size_t const n = 4096;
    fu::sharded_array<std::uint32_t> shards;
    expect(fu::succeeded(shards.resize_uninitialized(topology, n)));

    // Each element lives exactly once: the shard lengths sum to the logical length.
    std::size_t footprint = 0;
    for (std::size_t domain = 0; domain < shards.memory_domains_count(); ++domain)
        footprint += shards.length_on_memory_domain(static_cast<fu::memory_domain_index_t>(domain));
    expect_eq(shards.size(), n);
    expect_eq(footprint, n);

    // Write each element to its own logical index, then read it back through `location_of`.
    for (std::size_t domain = 0; domain < shards.memory_domains_count(); ++domain) {
        auto const memory_domain = static_cast<fu::memory_domain_index_t>(domain);
        for (std::size_t local = 0; local < shards.length_on_memory_domain(memory_domain); ++local)
            shards.at(memory_domain, local) = static_cast<std::uint32_t>(shards.logical_index_of(memory_domain, local));
    }
    for (std::size_t i = 0; i < n; ++i) {
        fu::sharded_array<std::uint32_t>::location_t const home = shards.location_of(i);
        expect_eq(shards.at(home.memory_domain, home.local_index), i);
    }

    // A logical length shorter than the domain count leaves trailing shards empty, never negative.
    fu::sharded_array<std::uint32_t> tiny;
    expect(fu::succeeded(tiny.resize_uninitialized(topology, 1)));
    expect_eq(tiny.segment(), 1u);
    std::size_t tiny_footprint = 0;
    for (std::size_t domain = 0; domain < tiny.memory_domains_count(); ++domain)
        tiny_footprint += tiny.length_on_memory_domain(static_cast<fu::memory_domain_index_t>(domain));
    expect_eq(tiny_footprint, 1u);
    fu::sharded_array<std::uint32_t>::location_t const home = tiny.location_of(0);
    tiny.at(home.memory_domain, home.local_index) = 42u;
    expect_eq(tiny.at(home.memory_domain, home.local_index), 42u);
}

#if FU_WITH_COLOCATE_POOLS_ON_DOMAIN
/** One distributed pool of exactly @p threads workers must dispatch every task exactly once. */
static void expect_spawn_shape_dispatches_(std::size_t const threads) noexcept {
    constexpr std::size_t tasks_k = 4096;
    std::vector<aligned_visit_t> visited(tasks_k);
    std::atomic<std::size_t> counter {0};

    fu::distributed_pool_t pool("forkunion");
    expect(fu::succeeded(pool.spawn(machine_topology, threads)));
    expect_eq(static_cast<std::size_t>(pool.threads_count()), threads); // ! A shape was silently resized

    pool.for_n_dynamic(tasks_k, [&](std::size_t const task, fu::thread_in_domain_t) noexcept {
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = task;
    });
    expect_eq(counter.load(std::memory_order_relaxed), tasks_k);
    expect(contains_iota(visited));
    pool.terminate();
}

/**
 *  @brief Spawns the distributed pool at awkward worker counts and checks exactly-once dispatch.
 *
 *  A pool sized to the machine never exercises the remainder paths of the per-domain split; a
 *  worker count that is prime to the domain count, short of the cores, or past them does.
 */
static void test_distributed_spawn_shapes() noexcept {
    std::size_t const cores = machine_topology.logical_cores_count();
    std::size_t const shapes[] = {1, 2, cores > 1 ? cores - 1 : 1, cores + 3};
    for (std::size_t const threads : shapes) expect_spawn_shape_dispatches_(threads);
}
#endif // FU_WITH_COLOCATE_POOLS_ON_DOMAIN

/** Enhanced NUMA topology logging function using the logger class. */
void log_numa_topology() noexcept {
    fu::logging_colors_t colors;
#if FU_WITH_COLOCATE_POOLS_ON_DOMAIN
    // Harvest topology
    if (fu::failed(machine_topology.harvest())) {
        std::fprintf(stderr, "%sX Failed to harvest NUMA topology%s\n", colors.bold_red(), colors.reset());
        std::exit(EXIT_FAILURE);
    }

    fu::capabilities_t cpu_caps = fu::cpu_capabilities();
    fu::capabilities_t ram_caps = fu::ram_capabilities();

    // Log topology and capabilities
    fu::log_numa_topology_t {}(machine_topology, colors);
    fu::log_capabilities_t {}(static_cast<fu::capabilities_t>(cpu_caps | ram_caps), colors);

#else
    std::printf("%sNUMA support not compiled in%s\n", colors.dim(), colors.reset());
#endif // FU_WITH_COLOCATE_POOLS_ON_DOMAIN
}

#if FU_WITH_COLOCATE_POOLS_ON_DOMAIN && FU_ON_LINUX && FU_WITH_PLACE_THREADS_BY_AFFINITY
/**
 *  @brief A pool must size itself from the cores we were given, and hand the caller back its own mask.
 *
 *  The process is narrowed here the way `taskset` or a cgroup `cpuset` would narrow it. A topology
 *  harvested from the machine rather than from the mask would report every core, the pool would
 *  oversubscribe them, and a "restore" that widens to the machine would leave the caller running on
 *  cores this process was never granted.
 */
static void test_caller_affinity_preserved() noexcept {
    fu::core_mask_t original;
    if (fu::failed(fu::capture_thread_cores(original))) skip("no affinity control");
    if (original.count() < 2) skip("too few cores"); // ? Too narrow to narrow further

    // Narrow the caller the way `taskset` would: keep the two lowest allowed cores.
    fu::core_mask_t narrowed;
    expect(fu::succeeded(narrowed.resize()));
    std::size_t kept = 0;
    for (std::size_t cpu = 0; cpu < original.capacity() && kept < 2; ++cpu)
        if (original.contains(static_cast<fu::core_id_t>(cpu))) narrowed.add(static_cast<fu::core_id_t>(cpu)), ++kept;

    bool succeeded = fu::succeeded(fu::restore_thread_cores(narrowed)); // ? Applies `narrowed`
    if (succeeded) {
        fu::machine_topology_t topology;
        succeeded = fu::succeeded(topology.harvest()) && topology.logical_cores_count() == 2;

        if (succeeded) {
            fu::distributed_pool_t pool;
            succeeded = fu::succeeded(pool.spawn(topology, 2));
            if (succeeded) succeeded = pool.all_threads_pinned();
            pool.terminate();
        }

        // The caller must be back on the two cores it narrowed itself to, not on the machine's cores.
        fu::core_mask_t afterwards;
        succeeded = succeeded && fu::succeeded(fu::capture_thread_cores(afterwards)) && afterwards.count() == 2;
    }

    (void)fu::restore_thread_cores(original); // ? Leave the process as we found it
    expect(succeeded);
}
#endif // FU_WITH_COLOCATE_POOLS_ON_DOMAIN && FU_ON_LINUX && FU_WITH_PLACE_THREADS_BY_AFFINITY

#if defined(__cpp_lib_atomic_ref) && defined(__cpp_lib_bit_cast)

/** The standard operations a reference spells by hand: every return value is the word before. */
template <template <typename> class atomic_ref_>
static void check_atomic_ref_words() noexcept {
    std::uint32_t word = 5;
    atomic_ref_<std::uint32_t> reference(word);
    expect_eq(reference.exchange(9u, std::memory_order_acq_rel), 5u);
    expect_eq(reference.fetch_add(3u, std::memory_order_acquire), 9u);
    expect_eq(reference.fetch_sub(2u, std::memory_order_relaxed), 12u);
    expect_eq(reference.load(std::memory_order_acquire), 10u);
    std::uint32_t expected = 10;
    expect(reference.compare_exchange_strong(expected, 11u, std::memory_order_acq_rel, std::memory_order_acquire));
    expected = 3;
    expect(!reference.compare_exchange_weak(expected, 12u, std::memory_order_acquire, std::memory_order_relaxed));
    expect_eq(expected, 11u); // ? The failed compare reports what it saw
    reference.store(6u, std::memory_order_release);
    expect_eq(word, 6u);

    std::uint64_t bits = 0xF0F0;
    atomic_ref_<std::uint64_t> bits_reference(bits);
    expect_eq(bits_reference.fetch_and(0xFF00ull, std::memory_order_acq_rel), 0xF0F0ull);
    expect_eq(bits_reference.fetch_or(0x1ull, std::memory_order_acq_rel), 0xF000ull);
    expect_eq(bits, 0xF001ull);

    bool flag = false;
    atomic_ref_<bool> flag_reference(flag);
    expect_eq(flag_reference.exchange(true, std::memory_order_acquire), false);
    flag_reference.store(false, std::memory_order_release);
    expect_eq(flag, false);
}

/** The operations past the standard - no-return forms, `fetch_max`/`fetch_min`, the conditional
 *  adds - each refusing exactly at its bound, unsigned and signed alike. */
template <template <typename> class atomic_ref_>
static void check_atomic_ref_extensions() noexcept {
    std::uint32_t word = 11;
    atomic_ref_<std::uint32_t> reference(word);
    expect_eq(reference.fetch_max(7u, std::memory_order_acq_rel), 11u);  // ? Loses: writes nothing
    expect_eq(reference.fetch_max(12u, std::memory_order_acq_rel), 11u); // ? Wins
    expect_eq(reference.fetch_min(3u, std::memory_order_relaxed), 12u);
    reference.add(4u, std::memory_order_release);
    reference.sub(2u, std::memory_order_relaxed);
    expect_eq(word, 5u);
    expect_eq(reference.fetch_add_if_at_most(3u, 8u, std::memory_order_acq_rel), 5u);  // 5 + 3 <= 8: adds
    expect_eq(reference.fetch_add_if_at_most(1u, 8u, std::memory_order_acq_rel), 8u);  // 8 + 1 > 8: refuses
    expect_eq(reference.fetch_sub_if_at_least(8u, 0u, std::memory_order_acquire), 8u); // 8 - 8 >= 0: subtracts
    expect_eq(reference.fetch_sub_if_at_least(1u, 0u, std::memory_order_acquire), 0u); // 0 - 1 < 0: refuses
    expect_eq(word, 0u);

    std::uint64_t bits = 0xF0F0;
    atomic_ref_<std::uint64_t> bits_reference(bits);
    bits_reference.clear(0xF0u, std::memory_order_release);
    bits_reference.set(0x0Fu, std::memory_order_relaxed);
    expect_eq(bits, 0xF00Full);

    std::int64_t signed_word = -5;
    atomic_ref_<std::int64_t> signed_reference(signed_word);
    expect_eq(signed_reference.fetch_max(std::int64_t(-9), std::memory_order_acquire), std::int64_t(-5));
    expect_eq(signed_reference.fetch_sub_if_at_least(std::int64_t(3), std::int64_t(-8), std::memory_order_acq_rel),
              std::int64_t(-5));
    expect_eq(signed_reference.fetch_sub_if_at_least(std::int64_t(1), std::int64_t(-8), std::memory_order_acq_rel),
              std::int64_t(-8)); // ? -8 - 1 < -8: refuses
    expect_eq(signed_word, std::int64_t(-8));
}

/** The shapes the indexes lean on, under contention - a dispenser by the no-return add, a bounded
 *  claim by the conditional add, a high-water mark by `fetch_max`, a lock by `exchange` - each
 *  with an exact expected total. */
template <template <typename> class atomic_ref_>
static void check_atomic_ref_under_contention() noexcept {
    constexpr std::size_t threads_k = 8, rounds_k = 20'000, claim_limit_k = threads_k * rounds_k / 3;
    alignas(128) std::uint32_t dispensed = 0;
    alignas(128) std::uint32_t claimed = 0;
    alignas(128) std::uint64_t high_water = 0;
    alignas(128) bool lock = false;
    alignas(128) std::uint64_t guarded = 0;

    fu::flat_pool_t pool;
    expect(fu::succeeded(pool.spawn(threads_k)));
    pool.for_threads([&](std::size_t const thread) noexcept {
        for (std::size_t round = 0; round != rounds_k; ++round) {
            atomic_ref_<std::uint32_t>(dispensed).add(1u, std::memory_order_relaxed);
            (void)atomic_ref_<std::uint32_t>(claimed).fetch_add_if_at_most(1u, claim_limit_k,
                                                                           std::memory_order_acq_rel);
            (void)atomic_ref_<std::uint64_t>(high_water)
                .fetch_max(thread * rounds_k + round, std::memory_order_relaxed);
            while (atomic_ref_<bool>(lock).exchange(true, std::memory_order_acquire)) {}
            ++guarded;
            atomic_ref_<bool>(lock).store(false, std::memory_order_release);
        }
    });

    expect_eq(dispensed, threads_k * rounds_k);
    expect_eq(claimed, claim_limit_k); // ? The bounded claim stops exactly at the limit
    expect_eq(high_water, threads_k * rounds_k - 1);
    expect_eq(guarded, threads_k * rounds_k); // ? The lock serialized every increment
}

/** Runs one reference through the contract where the machine admits it - the bits it declares are
 *  the same ones a CPU class needs before the runtime dispatch may pick it; a missing one skips by name. */
template <template <typename> class atomic_ref_>
static void check_atomic_ref() noexcept {
    fu::capabilities_t const needed = atomic_ref_<std::uint32_t>::capabilities_k;
    unsigned const missing = needed & ~fu::runtime_capabilities();
    if (missing) {
        char reason[64];
        std::snprintf(reason, sizeof(reason), "no %s",
                      fu::capability_name(static_cast<fu::capabilities_t>(std::bit_floor(missing))));
        skip_(reason);
        return;
    }
    check_atomic_ref_words<atomic_ref_>();
    check_atomic_ref_extensions<atomic_ref_>();
    check_atomic_ref_under_contention<atomic_ref_>();
}

/** Every reference the build spells - the standard one everywhere, the instruction-set ones where
 *  inline assembly exists - each run only where the machine admits it. */
static void test_atomic_refs() noexcept {
    check_atomic_ref<fu::standard_atomic_ref>();
#if FU_DETECT_ARCH_X86_64_ && FU_DETECT_INLINE_ASM_SUPPORT_
    check_atomic_ref<fu::x86_cmpccxadd_atomic_ref>();
    check_atomic_ref<fu::x86_raoint_atomic_ref>();
#elif FU_DETECT_ARCH_ARM64_ && FU_DETECT_INLINE_ASM_SUPPORT_
    check_atomic_ref<fu::arm64_lse_atomic_ref>();
    check_atomic_ref<fu::arm64_rcpc_atomic_ref>();
#elif FU_DETECT_ARCH_RISC5_ && FU_DETECT_INLINE_ASM_SUPPORT_ && __riscv_xlen == 64
    check_atomic_ref<fu::risc5_atomic_ref>();
    check_atomic_ref<fu::risc5_zacas_atomic_ref>();
#endif
}

#else

/** The references need the library's `std::atomic_ref` and `std::bit_cast`; say so rather than vanish. */
static void test_atomic_refs() noexcept { skip("no `std::atomic_ref`"); }

#endif // __cpp_lib_atomic_ref && __cpp_lib_bit_cast

int main(void) {
    install_crash_handlers_();

    std::printf("Welcome to the ForkUnion library test suite!\n");
    log_numa_topology();

    std::printf("Starting unit tests...\n");
    using test_func_t = void() /* noexcept */;
    struct {
        char const *name;
        test_func_t *function;
    } const unit_tests[] = {
        // Helpers
        {"`indexed_split` helpers", test_indexed_split},            //
        {"`coprime_permutation` ranges", test_coprime_permutation}, //
        {"`atomic_ref` contracts per ISA", test_atomic_refs},       //
        // Hardware topology, on every host that reports one
        {"`machine_topology` invariants", test_topology_invariants}, //
        {"`replicated_array` per-domain buffer", test_replicated_array},
        {"`sharded_array` segment round-trip", test_sharded_array},
        // Actual thread-pools
        {"`spawn` zero threads", test_spawn_zero},                          //
        {"`spawn` normal", test_spawn_success},                             //
        {"`caller_exclusivity` query", test_caller_exclusivity_query},      //
        {"`for_threads` dispatch", test_for_threads},                       //
        {"`unsafe_for_threads` dispatch", test_unsafe_for_threads},         //
        {"`generation` polling", test_generation_polling},                  //
        {"`broadcast_join` lifecycle", test_guard_lifecycle},               //
        {"`broadcast_join` destructor joins", test_guard_destructor_joins}, //
        {"`generation` inclusive contract", test_generation_inclusive},     //
        {"`generation` single-thread pool", test_generation_single_thread}, //
        {"`generation` single-worker exclusive", test_generation_single_thread_exclusive},
        {"`generation` stress", test_generation_stress},                         //
        {"`caller_exclusive_k` calls", test_exclusivity},                        //
        {"`caller_exclusive_k` inline guards", test_exclusivity_inline_guards},  //
        {"`for_n` for uncomfortable input size", test_uncomfortable_input_size}, //
        {"`for_n` static scheduling", test_for_n},                               //
        {"`for_slices` slice scheduling", test_for_slices},                      //
        {"`for_n_dynamic` dynamic scheduling", test_for_n_dynamic},              //
        {"`for_n_dynamic` stalled thread stolen from", test_for_n_dynamic_stealing},
        {"`for_n_dynamic` oversubscribed threads", test_oversubscribed_threads},      //
        {"`generation` stale token stays complete", test_stale_generation_completes}, //
        {"`sleep` and wake exactly-once", test_sleep_wake},                           //
        {"two caller threads, two pools", test_concurrent_caller_threads},            //
        {"`terminate` avoided", test_mixed_restart<false>},                           //
        {"`terminate` and re-spawn", test_mixed_restart<true>},                       //
        {"`terminate` and re-spawn churn", test_spawn_terminate_churn},               //
#if FU_WITH_COLOCATE_POOLS_ON_DOMAIN
        // Uniform Memory Access (UMA) tests for threads pinned to the same NUMA node
        {"UMA `spawn` normal", test_spawn_success<make_colocated_pool_t>},
        {"UMA `caller_exclusivity` query", test_caller_exclusivity_query<make_colocated_pool_t>},
        {"UMA `for_threads` dispatch", test_for_threads<make_colocated_pool_t>},
        {"UMA `unsafe_for_threads` dispatch", test_unsafe_for_threads<make_colocated_pool_t>},
        {"UMA `generation` polling", test_generation_polling<make_colocated_pool_t>},
        {"UMA `broadcast_join` lifecycle", test_guard_lifecycle<make_colocated_pool_t>},
        {"UMA `broadcast_join` destructor joins", test_guard_destructor_joins<make_colocated_pool_t>},
        {"UMA `generation` inclusive contract", test_generation_inclusive<make_colocated_pool_t>},
        {"UMA `generation` stress", test_generation_stress<make_colocated_pool_t>},
        {"UMA `caller_exclusive_k` calls", test_exclusivity<make_colocated_pool_t>},
        {"UMA `caller_exclusive_k` inline guards", test_exclusivity_inline_guards<make_colocated_pool_t>},
        {"UMA `for_n` for uncomfortable input size", test_uncomfortable_input_size<make_colocated_pool_t>},
        {"UMA `for_n` static scheduling", test_for_n<make_colocated_pool_t>},
        {"UMA `for_slices` slice scheduling", test_for_slices<make_colocated_pool_t>},
        {"UMA `for_n_dynamic` dynamic scheduling", test_for_n_dynamic<make_colocated_pool_t>},
        {"UMA `for_n_dynamic` stalled thread stolen from", test_for_n_dynamic_stealing<make_colocated_pool_t>},
        {"UMA `for_n_dynamic` oversubscribed threads", test_oversubscribed_threads<make_colocated_pool_t>},
        {"UMA `generation` stale token stays complete", test_stale_generation_completes<make_colocated_pool_t>},
        {"UMA `sleep` and wake exactly-once", test_sleep_wake<make_colocated_pool_t>},
        {"UMA `terminate` avoided", test_mixed_restart<false, make_colocated_pool_t>},
        {"UMA `terminate` and re-spawn", test_mixed_restart<true, make_colocated_pool_t>},
        {"UMA `terminate` and re-spawn churn", test_spawn_terminate_churn<make_colocated_pool_t>},
        // Non-Uniform Memory Access (NUMA) tests for threads addressing all NUMA nodes
        {"NUMA `spawn` normal", test_spawn_success<make_distributed_pool_t>},
        {"NUMA `caller_exclusivity` query", test_caller_exclusivity_query<make_distributed_pool_t>},
        {"NUMA `for_threads` dispatch", test_for_threads<make_distributed_pool_t>},
        {"NUMA `unsafe_for_threads` dispatch", test_unsafe_for_threads<make_distributed_pool_t>},
        {"NUMA `generation` polling", test_generation_polling<make_distributed_pool_t>},
        {"NUMA `broadcast_join` lifecycle", test_guard_lifecycle<make_distributed_pool_t>},
        {"NUMA `broadcast_join` destructor joins", test_guard_destructor_joins<make_distributed_pool_t>},
        {"NUMA `generation` inclusive contract", test_generation_inclusive<make_distributed_pool_t>},
        {"NUMA `generation` stress", test_generation_stress<make_distributed_pool_t>},
        {"NUMA `caller_exclusive_k` calls", test_exclusivity<make_distributed_pool_t>},
        {"NUMA `caller_exclusive_k` inline guards", test_exclusivity_inline_guards<make_distributed_pool_t>},
        {"NUMA `for_n` for uncomfortable input size", test_uncomfortable_input_size<make_distributed_pool_t>},
        {"NUMA `for_n` static scheduling", test_for_n<make_distributed_pool_t>},
        {"NUMA `for_slices` slice scheduling", test_for_slices<make_distributed_pool_t>},
        {"NUMA `for_n_dynamic` dynamic scheduling", test_for_n_dynamic<make_distributed_pool_t>},
        {"NUMA `for_n_dynamic` stalled thread stolen from", test_for_n_dynamic_stealing<make_distributed_pool_t>},
        {"NUMA `for_n_dynamic` exhaustive multi-domain",
         test_distributed_for_n_dynamic_exhaustive<make_distributed_pool_t>},
        {"NUMA `for_n_dynamic` oversubscribed threads", test_oversubscribed_threads<make_distributed_pool_t>},
        {"NUMA `generation` stale token stays complete", test_stale_generation_completes<make_distributed_pool_t>},
        {"NUMA `sleep` and wake exactly-once", test_sleep_wake<make_distributed_pool_t>},
        {"NUMA awkward spawn shapes", test_distributed_spawn_shapes},
        {"NUMA fabric tier derivation", test_fabric_level_derivation},
        {"NUMA measured fabric harvest", test_measured_fabric},
        {"NUMA `terminate` avoided", test_mixed_restart<false, make_distributed_pool_t>},
        {"NUMA `terminate` and re-spawn", test_mixed_restart<true, make_distributed_pool_t>},
        {"NUMA `terminate` and re-spawn churn", test_spawn_terminate_churn<make_distributed_pool_t>},
#if FU_ON_LINUX && FU_WITH_PLACE_THREADS_BY_AFFINITY
        {"NUMA caller affinity preserved", test_caller_affinity_preserved},
#endif
#endif // FU_WITH_COLOCATE_POOLS_ON_DOMAIN
    };

    std::size_t const total_unit_tests = sizeof(unit_tests) / sizeof(unit_tests[0]);
    for (std::size_t i = 0; i < total_unit_tests; ++i) {
        std::printf("Running %s... ", unit_tests[i].name);
        std::fflush(stdout); // ? A failing check or crash follows the flushed name
        unit_tests[i].function();
        std::printf("PASS\n");
    }
    std::printf("All %zu unit tests passed\n", total_unit_tests);

#if FU_TEST_SKIP_STRESS_
    // The stress suite hammers the dispatch/join race window for millions of epochs. A qemu-user
    // emulator neither reproduces the guest memory model this probes nor runs it in tolerable time,
    // so the cross builds define it away and lean on the native Arm64 job, where the weak memory
    // model is actually exercised.
    std::printf("Skipping stress tests: built with FU_TEST_SKIP_STRESS_\n");
#else
    // Start stress-testing the implementation
    std::printf("Starting stress tests...\n");
    std::size_t const max_cores = fu::allowed_cores_count();

    // On 32-bit architectures, limit thread counts to avoid resource exhaustion
    // Each thread needs ~8MB stack, and 255 threads would consume 2GB+ address space
    constexpr bool is_32bit = sizeof(void *) == 4;
    constexpr std::size_t max_stress_threads = is_32bit ? 23 : 255;

    using stress_test_func_t = void(std::size_t, std::size_t, std::size_t) /* noexcept */;
    struct {
        char const *pool_name;
        stress_test_func_t *function;
        std::size_t count_threads;
        std::size_t count_tasks;
        std::size_t count_cycles;
    } const stress_tests[] = {
        {"fu8", &stress_test_composite<fu8_t>, 3, 3, 1},
        {"fu8", &stress_test_composite<fu8_t>, 3, 2, 1},
        {"fu8", &stress_test_composite<fu8_t>, 3, 4, 1},
        {"fu8", &stress_test_composite<fu8_t>, 3, 5, 1},
        {"fu8", &stress_test_composite<fu8_t>, 7, max_stress_threads, 1},
        {"fu8", &stress_test_composite<fu8_t>, max_stress_threads, 7, 1},
        {"fu8", &stress_test_composite<fu8_t>, max_stress_threads - 2, max_stress_threads - 1, 1},
        {"fu8", &stress_test_composite<fu8_t>, max_stress_threads - 2, max_stress_threads, 1},
        {"fu8", &stress_test_composite<fu8_t>, max_stress_threads, max_stress_threads, 1},
        {"fu8", &stress_test_composite<fu8_t>, 3, 5, 160},     // ? 640 epochs: wraps the `uint8` clock twice
        {"fu16", &stress_test_composite<fu16_t>, 3, 5, 33000}, // ? 132K epochs: wraps the `uint16` clock twice
        {"fu16", &stress_test_composite<fu16_t>, max_cores, UINT16_MAX, 1},
        {"fu16", &stress_test_composite<fu16_t>, max_stress_threads, UINT16_MAX, 1},
        {"fu32", &stress_test_composite<fu32_t>, max_cores, 100000, 1}, // ? Instantiated, so exercised
    };

    std::size_t const total_stress_tests = sizeof(stress_tests) / sizeof(stress_tests[0]);
    for (std::size_t i = 0; i < total_stress_tests; ++i) {
        std::printf(                                                       //
            "Running `%s` with %zu threads & %zu inputs x %zu cycles... ", //
            stress_tests[i].pool_name, stress_tests[i].count_threads, stress_tests[i].count_tasks,
            stress_tests[i].count_cycles);
        std::fflush(stdout); // ? A failing check or crash follows the flushed name
        stress_tests[i].function(stress_tests[i].count_threads, stress_tests[i].count_tasks,
                                 stress_tests[i].count_cycles);
        std::printf("PASS\n");
    }
    std::printf("All %zu stress tests passed\n", total_stress_tests);
#endif

    return EXIT_SUCCESS;
}
