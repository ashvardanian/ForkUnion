#include <cstdio>    // `std::printf`, `std::fprintf`
#include <cstdlib>   // `EXIT_FAILURE`, `EXIT_SUCCESS`
#include <vector>    // `std::vector`
#include <algorithm> // `std::sort`
#include <numeric>   // `std::iota`
#include <random>    // `std::random_device`, `std::mt19937`, `std::shuffle`

#include <fork_union.hpp>

/* Namespaces, constants, and explicit type instantiations. */
namespace fu = ashvardanian::fork_union;

using fu32_t = fu::basic_pool<std::allocator<std::thread>, fu::standard_yield_t, std::uint32_t>;
using fu16_t = fu::basic_pool<std::allocator<std::thread>, fu::standard_yield_t, std::uint16_t>;
using fu8_t = fu::basic_pool<std::allocator<std::thread>, fu::standard_yield_t, std::uint8_t>;

/*
 *  Explicitly instantiate the thread-pools to cover all of their logic, but avoid the
 *  "duplicate explicit instantiation" error on platforms where `std::size_t` is `uint64_t`.
 *
 *  template class fu::basic_pool<std::allocator<std::thread>, fu::standard_yield_t, std::size_t>
 */
template class fu::basic_pool<std::allocator<std::thread>, fu::standard_yield_t, std::uint32_t>;
template class fu::basic_pool<std::allocator<std::thread>, fu::standard_yield_t, std::uint16_t>;
template class fu::basic_pool<std::allocator<std::thread>, fu::standard_yield_t, std::uint8_t>;

#if FU_ENABLE_NUMA
template struct fu::linux_colocated_pool<>;
template struct fu::linux_distributed_pool<>;
#endif

template <typename index_type_ = std::uint8_t>
bool test_indexed_split() noexcept {
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
                    if (task >= tasks) return false; // Out of bounds
                    if (visits[task]) return false;  // Already visited
                    visits[task] = true;             // Mark as visited
                }
            }
        }
    }

    return true;
}

template <typename index_type_ = std::uint8_t>
bool test_coprime_permutation() noexcept {
    constexpr std::size_t max_tasks = std::numeric_limits<index_type_>::max();

    for (std::size_t start = 0; start < max_tasks; ++start) {
        for (std::size_t end = start + 1; end < max_tasks; ++end) {
            for (std::size_t seed = 0; seed < max_tasks; ++seed) {

                // Create a coprime permutation and make sure it only covers the range [start, end)
                index_type_ const range_size = static_cast<index_type_>(end - start);
                fu::coprime_permutation_range<index_type_> permutation(static_cast<index_type_>(start), range_size,
                                                                       static_cast<index_type_>(seed));

                std::size_t count_matches = 0;
                for (auto value : permutation) {
                    if (value < start || value >= end) {
                        return false; // Out of range
                    }
                    count_matches++;
                }
                if (count_matches != range_size) {
                    return false; // Not all values in the range were covered
                }
            }
        }
    }
    return true;
}

constexpr std::size_t default_parallel_tasks_k = 10000; // 10K

struct make_pool_t {
    fu::basic_pool_t construct() const noexcept { return fu::basic_pool_t(); }
    std::size_t scope(std::size_t oversubscription = 1) const noexcept {
        return std::thread::hardware_concurrency() * oversubscription;
    }
};

#if FU_ENABLE_NUMA
static fu::numa_topology_t numa_topology;
struct make_linux_colocated_pool_t {
    fu::linux_colocated_pool_t construct() const noexcept { return fu::linux_colocated_pool_t("fork_union"); }
    fu::numa_node_t scope(std::size_t = 0) const noexcept { return numa_topology.node(0); }
};
struct make_linux_distributed_pool_t {
    fu::linux_distributed_pool_t construct() const noexcept { return fu::linux_distributed_pool_t("fork_union"); }
    fu::numa_topology_t const &scope(std::size_t = 0) const noexcept { return numa_topology; }
};
#endif

static bool test_try_spawn_zero() noexcept {
    fu::basic_pool_t pool;
    return !pool.try_spawn(0u);
}

template <typename make_pool_type_ = make_pool_t>
static bool test_try_spawn_success() noexcept {
    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!pool.try_spawn(maker.scope())) return false;
    return true;
}

/** @brief Make sure that `for_threads` is called from each thread. */
template <typename make_pool_type_ = make_pool_t>
static bool test_for_threads() noexcept {

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!pool.try_spawn(maker.scope())) return false;

    std::vector<std::atomic<bool>> visited(pool.threads_count());
    pool.for_threads([&](std::size_t const thread_index) noexcept { //
        visited[thread_index].store(true, std::memory_order_relaxed);
    });

    for (std::size_t i = 0; i < pool.threads_count(); ++i)
        if (!visited[i]) return false;
    return true;
}

/** @brief Make sure that `unsafe_for_threads` is called from each thread. */
template <typename make_pool_type_ = make_pool_t>
static bool test_unsafe_for_threads() noexcept {

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!pool.try_spawn(maker.scope())) return false;

    std::vector<std::atomic<bool>> visited(pool.threads_count());
    auto on_each_thread = [&](std::size_t const thread_index) noexcept {
        visited[thread_index].store(true, std::memory_order_relaxed);
    };
    pool.unsafe_for_threads(on_each_thread);
    pool.unsafe_join();

    for (std::size_t i = 0; i < pool.threads_count(); ++i)
        if (!visited[i]) return false;
    return true;
}

/** @brief Shows how to control multiple thread-pools from the same main thread. */
template <typename make_pool_type_ = make_pool_t>
static bool test_exclusivity() noexcept {

    auto maker = make_pool_type_ {};

    // First try with externally defined lambdas with a clearly long lifetime:
    {
        auto first_pool = maker.construct();
        auto second_pool = maker.construct();
        if (!first_pool.try_spawn(maker.scope(), fu::caller_inclusive_k)) return false;
        if (!second_pool.try_spawn(maker.scope(), fu::caller_exclusive_k)) return false;

        std::size_t const first_size = first_pool.threads_count();
        std::size_t const second_size = second_pool.threads_count();
        std::size_t const total_size = first_size + second_size;
        std::vector<std::atomic<bool>> visited(total_size);

        auto do_second = [&](std::size_t const thread_index) noexcept {
            visited[first_size + thread_index].store(true, std::memory_order_relaxed);
        };
        auto do_first = [&](std::size_t const thread_index) noexcept {
            visited[thread_index].store(true, std::memory_order_relaxed);
        };

        // Repeat the same logic a few times and check for correctness:
        for (std::size_t iteration = 0; iteration < 3; ++iteration) {
            auto join_second = second_pool.for_threads(do_second);
            first_pool.for_threads(do_first);
            join_second.join();

            // Validate:
            for (std::size_t i = 0; i < total_size; ++i)
                if (!visited[i]) return false;
        }
    }

    // Now do the same with inline lambdas, where they should be re-packaged into returned objects:
    {
        auto first_pool = maker.construct();
        auto second_pool = maker.construct();
        if (!first_pool.try_spawn(maker.scope(), fu::caller_inclusive_k)) return false;
        if (!second_pool.try_spawn(maker.scope(), fu::caller_exclusive_k)) return false;

        std::size_t const first_size = first_pool.threads_count();
        std::size_t const second_size = second_pool.threads_count();
        std::size_t const total_size = first_size + second_size;
        std::vector<std::atomic<bool>> visited(total_size);

        auto join_second = second_pool.for_threads([&](std::size_t const thread_index) noexcept {
            visited[first_size + thread_index].store(true, std::memory_order_relaxed);
        });
        first_pool.for_threads([&](std::size_t const thread_index) noexcept {
            visited[thread_index].store(true, std::memory_order_relaxed);
        });
        join_second.join();

        // Validate:
        for (std::size_t i = 0; i < total_size; ++i)
            if (!visited[i]) return false;
    }
    return true;
}

/** @brief Make sure that `for_n` is called from each thread. */
template <typename make_pool_type_ = make_pool_t>
static bool test_uncomfortable_input_size() noexcept {

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!pool.try_spawn(maker.scope())) return false;

    std::size_t const max_input_size = pool.threads_count() * 3; // Arbitrary size, larger than the number of threads
    for (std::size_t input_size = 0; input_size <= max_input_size; ++input_size) {
        std::atomic<bool> out_of_bounds(false);
        pool.for_n(input_size, [&](std::size_t const task) noexcept {
            if (task >= input_size) out_of_bounds.store(true, std::memory_order_relaxed);
        });
        if (out_of_bounds.load(std::memory_order_relaxed)) return false;
    }

    return true;
}

/** @brief Convenience structure to ensure we output match locations to independent cache lines. */
struct alignas(fu::default_alignment_k) aligned_visit_t {
    std::size_t task = 0;
    bool operator<(aligned_visit_t const &other) const noexcept { return task < other.task; }
    bool operator==(aligned_visit_t const &other) const noexcept { return task == other.task; }
    bool operator!=(std::size_t other_index) const noexcept { return task != other_index; }
    bool operator==(std::size_t other_index) const noexcept { return task == other_index; }
};

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

/** @brief Make sure that `for_n` is called the right number of times with the right prong IDs. */
template <typename make_pool_type_ = make_pool_t>
static bool test_for_n() noexcept {

    std::atomic<std::size_t> counter(0);
    std::vector<aligned_visit_t> visited(default_parallel_tasks_k);

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!pool.try_spawn(maker.scope())) return false;

    using pool_t = decltype(pool);
    using prong_t = typename pool_t::prong_t;

    pool.for_n(default_parallel_tasks_k, [&](prong_t prong) noexcept {
        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = prong.task;
    });

    // Make sure that all prong IDs are unique and form the full range of [0, `default_parallel_tasks_k`).
    if (counter.load() != default_parallel_tasks_k) return false;
    if (!contains_iota(visited)) return false;

    // Make sure repeated calls to `for_n` work
    counter = 0;
    pool.for_n(default_parallel_tasks_k, [&](prong_t prong) noexcept {
        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = prong.task;
    });

    // Make sure that all prong IDs are unique and form the full range of [0, `default_parallel_tasks_k`).
    if (counter.load() != default_parallel_tasks_k) return false;
    if (!contains_iota(visited)) return false;

    // Make sure `for_n` is being executed on different threads.
    std::vector<aligned_visit_t> visited_threads(pool.threads_count());
    constexpr std::size_t invalid_task = std::numeric_limits<std::size_t>::max();
    for (auto &visit : visited_threads) visit.task = invalid_task;
    pool.for_n(default_parallel_tasks_k, // ? Could have been an arbitrary number `>= pool.threads_count()`
               [&](prong_t prong) noexcept { visited_threads[prong.thread].task = prong.thread; });

    return contains_iota(visited_threads);
}

/** @brief Make sure that `for_n_dynamic` is called the right number of times with the right prong IDs. */
template <typename make_pool_type_ = make_pool_t>
static bool test_for_n_dynamic() noexcept {

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!pool.try_spawn(maker.scope())) return false;

    std::vector<aligned_visit_t> visited(default_parallel_tasks_k);
    std::atomic<std::size_t> counter(0);
    pool.for_n_dynamic(default_parallel_tasks_k, [&](std::size_t const task) noexcept {
        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = task;
    });

    // Make sure that all prong IDs are unique and form the full range of [0, `default_parallel_tasks_k`).
    if (counter.load() != default_parallel_tasks_k) return false;
    if (!contains_iota(visited)) return false;

    // Make sure repeated calls to `for_n` work
    counter = 0;
    pool.for_n_dynamic(default_parallel_tasks_k, [&](std::size_t const task) noexcept {
        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = task;
    });

    return counter.load() == default_parallel_tasks_k && contains_iota(visited);
}

/** @brief Stress-tests the implementation by oversubscribing the number of threads. */
template <typename make_pool_type_ = make_pool_t>
static bool test_oversubscribed_threads() noexcept {
    constexpr std::size_t oversubscription = 3;

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!pool.try_spawn(maker.scope(oversubscription))) return false;

    std::vector<aligned_visit_t> visited(default_parallel_tasks_k);
    std::atomic<std::size_t> counter(0);
    thread_local volatile std::size_t some_local_work = 0;
    pool.for_n_dynamic(default_parallel_tasks_k, [&](std::size_t const task) noexcept {
        // Perform some weird amount of work, that is not very different between consecutive tasks.
        for (std::size_t i = 0; i != task % oversubscription; ++i) some_local_work = some_local_work + i * i;

        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = task;
    });

    // Make sure that all prong IDs are unique and form the full range of [0, `default_parallel_tasks_k`).
    return counter.load() == default_parallel_tasks_k && contains_iota(visited);
}

/** @brief Make sure that that we can combine static & dynamic loads over the same pool with & w/out resetting. */
template <bool should_restart_, typename make_pool_type_ = make_pool_t>
static bool test_mixed_restart() noexcept {

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!pool.try_spawn(maker.scope())) return false;

    std::vector<aligned_visit_t> visited(default_parallel_tasks_k);
    std::atomic<std::size_t> counter(0);

    pool.for_n(default_parallel_tasks_k, [&](std::size_t const task) noexcept {
        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = task;
    });
    if (counter.load() != default_parallel_tasks_k) return false;
    if (!contains_iota(visited)) return false;

    // Make sure that the pool can be reset and reused
    if (should_restart_) {
        pool.terminate();
        if (!pool.try_spawn(maker.scope())) return false;
    }

    // Make sure repeated calls to `for_n` work
    counter = 0;
    pool.for_n_dynamic(default_parallel_tasks_k, [&](std::size_t const task) noexcept {
        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = task;
    });

    return counter.load() == default_parallel_tasks_k && contains_iota(visited);
}

/** @brief Hard complex example, involving launching multiple tasks, including static and dynamic ones,
 *         stopping them half-way, resetting & reinitializing, and raising exceptions.
 */
template <typename pool_type_>
static bool stress_test_composite(std::size_t const threads_count, std::size_t const parallel_tasks_count) noexcept {

    using pool_t = pool_type_;
    using index_t = typename pool_t::index_t;
    using prong_t = fu::prong<index_t>;

    pool_t pool;
    if (!pool.try_spawn(static_cast<index_t>(threads_count))) return false;

    // Make sure that no overflow happens in the static scheduling
    std::atomic<std::size_t> counter(0);
    std::vector<aligned_visit_t> visited(parallel_tasks_count);
    pool.for_n(static_cast<index_t>(parallel_tasks_count), [&](prong_t prong) noexcept {
        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = prong.task;
    });
    if (counter.load() != parallel_tasks_count) return false;
    if (!contains_iota(visited)) return false;

    // Make sure that no overflow happens in the dynamic scheduling
    counter = 0;
    pool.for_n_dynamic(static_cast<index_t>(parallel_tasks_count), [&](prong_t prong) noexcept {
        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = prong.task;
    });
    if (counter.load() != parallel_tasks_count) return false;
    if (!contains_iota(visited)) return false;

    // Make sure the operations can be interrupted from inside the prong
    return true;
}

template <typename make_pool_type_ = make_pool_t>
static bool test_sort_edge_cases() noexcept {
    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!pool.try_spawn(maker.scope())) return false;

    // Test various problematic size/thread combinations
    std::vector<std::size_t> thread_counts = {1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32};
    std::vector<std::size_t> sizes = {
        0,   1,   2,   3,   7,    8,    15,   16,   31,   32,   63,   64,   127,  128,
        255, 256, 511, 512, 1023, 1024, 1025, 2047, 2048, 4095, 4096, 8191, 8192,
    };

    for (auto size : sizes) {
        for (auto threads : thread_counts) {
            if (threads > pool.threads_count()) continue;

            // Create test data: reverse sorted to stress the algorithm
            std::vector<int> data(size);
            std::iota(data.rbegin(), data.rend(), 0);

            std::vector<int> expected = data;
            std::sort(expected.begin(), expected.end());

            // Test with a pool limited to specific thread count
            auto test_pool = maker.construct();
            if (!test_pool.try_spawn(static_cast<typename decltype(test_pool)::index_t>(threads))) return false;

            fu::sort(test_pool, data.begin(), data.end());

            if (data != expected) return false;
        }
    }

    return true;
}

template <typename make_pool_type_ = make_pool_t>
static bool test_sort_duplicates() noexcept {
    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!pool.try_spawn(maker.scope())) return false;

    // Test with many duplicates - this stresses the three-way quicksort
    std::vector<std::size_t> sizes = {100, 1000, 10000};
    std::vector<unsigned> patterns = {1, 5, 10, 50}; // Number of unique values

    for (auto size : sizes) {
        for (auto pattern : patterns) {
            std::vector<unsigned> data(size);
            for (std::size_t i = 0; i < size; ++i) { data[i] = static_cast<unsigned>(i % pattern); }

            // Shuffle to create random order
            std::random_device random_device;
            std::mt19937 random_generator(random_device());
            std::shuffle(data.begin(), data.end(), random_generator);

            std::vector<unsigned> expected = data;
            std::sort(expected.begin(), expected.end());

            fu::sort(pool, data.begin(), data.end());

            if (data != expected) return false;
        }
    }

    return true;
}

template <typename make_pool_type_ = make_pool_t>
static bool test_sort_pathological() noexcept {
    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!pool.try_spawn(maker.scope())) return false;

    // Test cases that are problematic for quicksort variants
    std::vector<std::vector<int>> test_cases {
        // Already sorted
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
        // Reverse sorted
        {10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
        // All same elements
        {5, 5, 5, 5, 5, 5, 5, 5, 5, 5},
        // Alternating pattern
        {1, 10, 2, 9, 3, 8, 4, 7, 5, 6},
        // Single outlier
        {1, 1, 1, 1, 1000, 1, 1, 1, 1, 1},
        // Two groups
        {1, 1, 1, 1, 1, 100, 100, 100, 100, 100},
    };

    for (auto &test_case : test_cases) {
        // Test with different sizes by replicating the pattern
        for (std::size_t multiplier : {1, 10, 100}) {

            std::vector<int> data;
            for (std::size_t i = 0; i < multiplier; ++i) data.insert(data.end(), test_case.begin(), test_case.end());

            std::vector<int> expected = data;
            std::sort(expected.begin(), expected.end());

            fu::sort(pool, data.begin(), data.end());

            if (data != expected) return false;
        }
    }

    return true;
}

template <typename make_pool_type_ = make_pool_t>
static bool test_sort_chunk_boundaries() noexcept {
    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!pool.try_spawn(maker.scope())) return false;

    std::size_t threads = pool.threads_count();

    // Test sizes that create edge cases in chunk distribution
    std::vector<std::size_t> critical_sizes = {
        threads - 1,       // Fewer elements than threads
        threads,           // Equal to thread count
        threads + 1,       // One more than thread count
        threads * 2 - 1,   // Almost 2 per thread
        threads * 2,       // Exactly 2 per thread
        threads * 2 + 1,   // Just over 2 per thread
        threads * 100 - 1, // Large with uneven distribution
        threads * 100,     // Large with even distribution
        threads * 100 + 1  // Large with slight imbalance
    };

    for (auto size : critical_sizes) {
        std::vector<int> data(size);
        std::iota(data.rbegin(), data.rend(), 0);

        // Add some randomness to avoid best-case scenarios
        std::random_device random_device;
        std::mt19937 random_generator(random_device());
        std::shuffle(data.begin(), data.end(), random_generator);

        std::vector<int> expected = data;
        std::sort(expected.begin(), expected.end());

        fu::sort(pool, data.begin(), data.end());

        if (data != expected) return false;
    }

    return true;
}

/**
 *  @brief Enhanced NUMA topology logging function using the logger class.
 */
void log_numa_topology() noexcept {
    fu::logging_colors_t colors;
#if FU_ENABLE_NUMA
    // Harvest topology
    if (!numa_topology.try_harvest()) {
        std::fprintf(stderr, "%sX Failed to harvest NUMA topology%s\n", colors.bold_red(), colors.reset());
        std::exit(EXIT_FAILURE);
    }

    fu::capabilities_t cpu_caps = fu::cpu_capabilities();
    fu::capabilities_t ram_caps = fu::ram_capabilities();

    // Log topology and capabilities
    fu::log_numa_topology_t {}(numa_topology, colors);
    fu::log_capabilities_t {}(static_cast<fu::capabilities_t>(cpu_caps | ram_caps), colors);

#else
    std::printf("%sNUMA support not compiled in%s\n", colors.dim(), colors.reset());
#endif // FU_ENABLE_NUMA
}

int main(void) {

    std::printf("Welcome to the Fork Union library test suite!\n");
    log_numa_topology();

    std::printf("Starting unit tests...\n");
    using test_func_t = bool() /* noexcept */;
    struct {
        char const *name;
        test_func_t *function;
    } const unit_tests[] = {
        // Helpers
        {"`indexed_split` helpers", test_indexed_split},            //
        {"`coprime_permutation` ranges", test_coprime_permutation}, //
        // Actual thread-pools
        {"`try_spawn` zero threads", test_try_spawn_zero},                       //
        {"`try_spawn` normal", test_try_spawn_success},                          //
        {"`for_threads` dispatch", test_for_threads},                            //
        {"`unsafe_for_threads` dispatch", test_unsafe_for_threads},              //
        {"`caller_exclusive_k` calls", test_exclusivity},                        //
        {"`for_n` for uncomfortable input size", test_uncomfortable_input_size}, //
        {"`for_n` static scheduling", test_for_n},                               //
        {"`for_n_dynamic` dynamic scheduling", test_for_n_dynamic},              //
        {"`for_n_dynamic` oversubscribed threads", test_oversubscribed_threads}, //
        {"`terminate` avoided", test_mixed_restart<false>},                      //
        {"`terminate` and re-spawn", test_mixed_restart<true>},                  //
        // Parallel sort tests
        {"`sort` edge cases", test_sort_edge_cases},             //
        {"`sort` duplicates", test_sort_duplicates},             //
        {"`sort` pathological cases", test_sort_pathological},   //
        {"`sort` chunk boundaries", test_sort_chunk_boundaries}, //
#if FU_ENABLE_NUMA
        // Uniform Memory Access (UMA) tests for threads pinned to the same NUMA node
        {"UMA `try_spawn` normal", test_try_spawn_success<make_linux_colocated_pool_t>},
        {"UMA `for_threads` dispatch", test_for_threads<make_linux_colocated_pool_t>},
        {"UMA `unsafe_for_threads` dispatch", test_unsafe_for_threads<make_linux_colocated_pool_t>},
        {"UMA `caller_exclusive_k` calls", test_exclusivity<make_linux_colocated_pool_t>},
        {"UMA `for_n` for uncomfortable input size", test_uncomfortable_input_size<make_linux_colocated_pool_t>},
        {"UMA `for_n` static scheduling", test_for_n<make_linux_colocated_pool_t>},
        {"UMA `for_n_dynamic` dynamic scheduling", test_for_n_dynamic<make_linux_colocated_pool_t>},
        {"UMA `for_n_dynamic` oversubscribed threads", test_oversubscribed_threads<make_linux_colocated_pool_t>},
        {"UMA `terminate` avoided", test_mixed_restart<false, make_linux_colocated_pool_t>},
        {"UMA `terminate` and re-spawn", test_mixed_restart<true, make_linux_colocated_pool_t>},
        // Non-Uniform Memory Access (NUMA) tests for threads addressing all NUMA nodes
        {"NUMA `try_spawn` normal", test_try_spawn_success<make_linux_distributed_pool_t>},
        {"NUMA `for_threads` dispatch", test_for_threads<make_linux_distributed_pool_t>},
        {"NUMA `unsafe_for_threads` dispatch", test_unsafe_for_threads<make_linux_distributed_pool_t>},
        {"NUMA `caller_exclusive_k` calls", test_exclusivity<make_linux_distributed_pool_t>},
        {"NUMA `for_n` for uncomfortable input size", test_uncomfortable_input_size<make_linux_distributed_pool_t>},
        {"NUMA `for_n` static scheduling", test_for_n<make_linux_distributed_pool_t>},
        {"NUMA `for_n_dynamic` dynamic scheduling", test_for_n_dynamic<make_linux_distributed_pool_t>},
        {"NUMA `for_n_dynamic` oversubscribed threads", test_oversubscribed_threads<make_linux_distributed_pool_t>},
        {"NUMA `terminate` avoided", test_mixed_restart<false, make_linux_distributed_pool_t>},
        {"NUMA `terminate` and re-spawn", test_mixed_restart<true, make_linux_distributed_pool_t>},
#endif // FU_ENABLE_NUMA
    };

    std::size_t const total_unit_tests = sizeof(unit_tests) / sizeof(unit_tests[0]);
    std::size_t failed_unit_tests = 0;
    for (std::size_t i = 0; i < total_unit_tests; ++i) {
        std::printf("Running %s... ", unit_tests[i].name);
        bool const ok = unit_tests[i].function();
        if (ok) { std::printf("PASS\n"); }
        else { std::printf("FAIL\n"); }
        failed_unit_tests += !ok;
    }

    if (failed_unit_tests > 0) {
        std::fprintf(stderr, "%zu/%zu unit tests failed\n", failed_unit_tests, total_unit_tests);
        return EXIT_FAILURE;
    }
    std::printf("All %zu unit tests passed\n", total_unit_tests);

    // Start stress-testing the implementation
    std::printf("Starting stress tests...\n");
    std::size_t const max_cores = std::thread::hardware_concurrency();

    // On 32-bit architectures, limit thread counts to avoid resource exhaustion
    // Each thread needs ~8MB stack, and 255 threads would consume 2GB+ address space
    constexpr bool is_32bit = sizeof(void *) == 4;
    constexpr std::size_t max_stress_threads = is_32bit ? 23 : 255;

    using stress_test_func_t = bool(std::size_t, std::size_t) /* noexcept */;
    struct {
        char const *pool_name;
        stress_test_func_t *function;
        std::size_t count_threads;
        std::size_t count_tasks;
    } const stress_tests[] = {
        {"fu8", &stress_test_composite<fu8_t>, 3, 3},
        {"fu8", &stress_test_composite<fu8_t>, 3, 2},
        {"fu8", &stress_test_composite<fu8_t>, 3, 4},
        {"fu8", &stress_test_composite<fu8_t>, 3, 5},
        {"fu8", &stress_test_composite<fu8_t>, 7, max_stress_threads},
        {"fu8", &stress_test_composite<fu8_t>, max_stress_threads, 7},
        {"fu8", &stress_test_composite<fu8_t>, max_stress_threads - 2, max_stress_threads - 1},
        {"fu8", &stress_test_composite<fu8_t>, max_stress_threads - 2, max_stress_threads},
        {"fu8", &stress_test_composite<fu8_t>, max_stress_threads, max_stress_threads},
        {"fu16", &stress_test_composite<fu16_t>, max_cores, UINT16_MAX},
        {"fu16", &stress_test_composite<fu16_t>, max_stress_threads, UINT16_MAX},
    };

    std::size_t const total_stress_tests = sizeof(stress_tests) / sizeof(stress_tests[0]);
    std::size_t failed_stress_tests = 0;
    for (std::size_t i = 0; i < total_stress_tests; ++i) {
        std::printf(                                          //
            "Running `%s` with %zu threads & %zu inputs... ", //
            stress_tests[i].pool_name, stress_tests[i].count_threads, stress_tests[i].count_tasks);
        bool const ok = stress_tests[i].function(stress_tests[i].count_threads, stress_tests[i].count_tasks);
        if (ok) { std::printf("PASS\n"); }
        else { std::printf("FAIL\n"); }
        failed_stress_tests += !ok;
    }

    if (failed_stress_tests > 0) {
        std::fprintf(stderr, "%zu/%zu stress tests failed\n", failed_stress_tests, total_stress_tests);
        return EXIT_FAILURE;
    }
    std::printf("All %zu stress tests passed\n", total_stress_tests);

    return EXIT_SUCCESS;
}
