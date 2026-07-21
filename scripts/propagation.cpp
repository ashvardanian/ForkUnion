/**
 *  @brief Demo app: Connected Components by label propagation, with ForkUnion, OpenMP, and Taskflow.
 *  @author Ash Vardanian
 *  @file propagation.cpp
 *
 *  The N-body simulation gives every task an identical cost, so it can only measure dispatch
 *  latency. Label propagation is the opposite end of fork-join usage: one parallel sweep per round,
 *  repeated until no label changes - so a single pass pays the dispatch-and-join tax once @b per
 *  @b round, and the graph's topology decides how many rounds there are.
 *
 *  @section The Necklace
 *
 *  A single R-MAT graph converges in a dozen rounds - too few to expose the barrier tax. So the
 *  generator strings @b C independent R-MAT communities on a ring, joined by one bridge edge per
 *  neighbouring pair. The global minimum label must walk the ring, one bridge per round plus a few
 *  rounds to cross each community, so convergence takes O(C) rounds while each round stays a
 *  bandwidth-bound sweep - the same honest steady-state work as any graph benchmark, with the
 *  fork-join frequency as the controlled axis.
 *
 *  Bridge endpoints are drawn from each community's first 64 vertices: R-MAT's quadrant bias piles
 *  the hubs at low indices, so a low endpoint is essentially guaranteed well-connected, and the ring
 *  cannot be severed by an isolated endpoint.
 *
 *  @section Determinism
 *
 *  The labels are double-buffered: every round reads the immutable previous array and each vertex
 *  writes only its own slot in the next - no atomics, no races, and every round is a pure function
 *  of the last. Rounds-to-convergence, every intermediate label, and the final fixed point are
 *  therefore identical across schedules, backends, thread counts, and languages - so the MTEPS
 *  denominator `rounds * edges` is the same number in every cell of a comparison table.
 *
 *  To control the script, several environment variables are used:
 *
 *  - `PROPAGATION_SCALE` - each community has `2^scale` vertices - default 14.
 *  - `PROPAGATION_COMMUNITIES` - communities strung on the ring - default 64.
 *  - `PROPAGATION_EDGE_FACTOR` - edges generated per vertex, before deduplication - default 16.
 *  - `PROPAGATION_BACKEND` - backend to use - default `forkunion_static_shared`.
 *  - `PROPAGATION_THREADS` - number of threads to use - default all hardware threads.
 *  - `PROPAGATION_SECONDS` - wall-clock budget per run, reporting the sustained rate - default 10.
 *  - `PROPAGATION_ITERATIONS` - run an exact pass count instead, when set.
 *  - `PROPAGATION_CHECK` - also converge serially, and fail unless labels and rounds agree exactly.
 *
 *  The ForkUnion backends are the four cells of `forkunion_{static,dynamic}_{shared,replicated}`;
 *  the baselines are `{openmp,taskflow}_{static,dynamic}`.
 *
 *  @section Benchmarking Protocol
 *
 *  Every runtime schedules the same one-vertex dynamic tasks: `schedule(dynamic, 1)` in OpenMP,
 *  `tf::DynamicPartitioner(1)` in Taskflow, and `for_n_dynamic` here. Cells run bare - core-granular
 *  pinning like `OMP_PROC_BIND=spread OMP_PLACES=cores` collapses a bandwidth-bound sweep ~16x for
 *  OpenMP and for any pool inheriting the caller's mask. The residual spread on SMT machines is
 *  preemption - one delayed hyperthread stalls every barrier of a pass - which the fixed window
 *  amortizes. The `_replicated` backends are a deliberate non-win on this workload: the hot traffic
 *  is the shared label array every round must see fresh, so replicating the read-only CSR pays
 *  nothing here, unlike N-body's replicated bodies. To compile and run:
 *
 *  @code{.sh}
 *  cmake -B build_release -D CMAKE_BUILD_TYPE=Release
 *  cmake --build build_release --config Release
 *  PROPAGATION_BACKEND=forkunion_static_shared build_release/forkunion_propagation
 *  @endcode
 */
#include <cstdint> // `std::uint32_t`
#include <cstdio>  // `std::printf`
#include <cstdlib> // `std::getenv`, `EXIT_SUCCESS`
#include <cstring> // `std::memcpy`, `std::memcmp`

#include <algorithm>   // `std::sort`, `std::unique`, `std::min`
#include <chrono>      // `std::chrono::steady_clock`
#include <optional>    // `std::optional` - the executor, spawned only when chosen
#include <string_view> // `std::string_view`

#if defined(_OPENMP)
#if __has_include(<omp.h>)
#include <omp.h>
#else
#undef _OPENMP
#endif
#endif

#if defined(_OPENMP) && defined(__GNUC__) && !defined(__clang__)
#include <parallel/algorithm> // `__gnu_parallel::sort`, OpenMP-backed
#endif

#include <taskflow/taskflow.hpp>           // `tf::Executor`, `tf::Taskflow`
#include <taskflow/algorithm/for_each.hpp> // `for_each_index`, partitioners

#include <forkunion.hpp>

namespace fu = ashvardanian::forkunion;

using vertex_t = std::uint32_t;
using edge_offset_t = std::uint64_t;
using label_t = std::uint32_t; // ? A component name: the smallest vertex index reachable so far

#pragma region Graph

/** @brief A read-only CSR as two spans - the interface every kernel takes. */
struct csr_view_t {
    fu::span<edge_offset_t const> row_offsets;
    fu::span<vertex_t const> column_indices;

    vertex_t vertices() const noexcept { return static_cast<vertex_t>(row_offsets.size() - 1); }
    edge_offset_t edges() const noexcept { return column_indices.size(); }
};

/** @brief The two CSR arrays built once on the host, in growable `dynamic_array`s. */
struct csr_host_t {
    fu::dynamic_array<edge_offset_t> row_offsets;
    fu::dynamic_array<vertex_t> column_indices;

    csr_view_t view() const noexcept {
        return {{row_offsets.data(), row_offsets.size()}, {column_indices.data(), column_indices.size()}};
    }
};

struct edge_t {
    vertex_t row, column;
    bool operator<(edge_t const &o) const noexcept { return row != o.row ? row < o.row : column < o.column; }
    bool operator==(edge_t const &o) const noexcept { return row == o.row && column == o.column; }
};

/** @brief Sorts past every valid edge; marks dropped self-loops, trimmed together with `unique`'s tail. */
static constexpr edge_t sentinel_edge_k {~vertex_t(0), ~vertex_t(0)};

/** @brief One quadrant choice in `[0, 100)` - same draw and counter scheme as every sibling benchmark. */
static inline unsigned random_percent(std::uint64_t const counter) noexcept {
    return static_cast<unsigned>(fu::split_mix(counter) % 100);
}

/** @brief One bridge endpoint in `[0, bound)`, from the same avalanche. */
static inline vertex_t random_index(std::uint64_t const counter, vertex_t const bound) noexcept {
    return static_cast<vertex_t>(fu::split_mix(counter) % bound);
}

/**
 *  @brief Generates the necklace: @p communities independent R-MAT graphs of `2^scale` vertices,
 *      joined in a ring by one bridge per neighbouring pair, and scatters it all into a CSR.
 *  @retval false on any allocation failure, leaving @p graph half-built but valid to destroy.
 *
 *  Community `c` owns global edge indices `[c * raw_local, (c+1) * raw_local)` and the vertex range
 *  `[c << scale, (c+1) << scale)`; the quadrant walk uses the same `e * 64 + bit` counters as the
 *  single-graph generators, so community 0 with `communities == 1` reproduces those graphs exactly.
 *  Bridge draws live in their own counter range above all edge draws, so nothing collides.
 */
static bool generate_necklace(std::size_t const scale, std::size_t const communities, std::size_t const edge_factor,
                              csr_host_t &graph) noexcept {
    std::size_t const community_vertices = std::size_t(1) << scale;
    vertex_t const vertices = static_cast<vertex_t>(communities * community_vertices);
    std::size_t const raw_local = community_vertices * edge_factor;
    std::size_t const raw_edges = communities * raw_local;
    std::size_t const bridges = communities > 1 ? communities : 0;

    // Build the COO edge list, dedupe it, and scatter it into CSR - scoped so `edges` frees before
    // the CSR build and never coexists with the work arrays at peak.
    {
        fu::dynamic_array<edge_t> edges;
        if (!edges.try_resize(raw_edges * 2 + bridges * 2)) return false; // ? Slots `2e, 2e+1` belong to edge `e`

        // Generation is the most expensive setup step - `scale` draws per edge, millions of edges - and
        // the counter-based draws make it embarrassingly parallel with no generator objects at all.
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (std::size_t e = 0; e < raw_edges; ++e) {
            vertex_t row = 0, column = 0;
            for (int bit = static_cast<int>(scale) - 1; bit >= 0; --bit) {
                unsigned const r = random_percent(e * 64 + static_cast<std::size_t>(bit)); // ? `a=57 b=19 c=19 d=5`
                vertex_t const step = static_cast<vertex_t>(1u) << bit;
                if (r < 57) continue; // ? Stay in the dense quadrant
                else if (r < 76)
                    column |= step;
                else if (r < 95)
                    row |= step;
                else
                    row |= step, column |= step;
            }
            vertex_t const base = static_cast<vertex_t>((e / raw_local) << scale);           // ? This community's range
            bool const self_loop = row == column;                                            // ? Dropped via sentinels
            edges[e * 2] = self_loop ? sentinel_edge_k : edge_t {base + row, base + column}; // ? Symmetrize
            edges[e * 2 + 1] = self_loop ? sentinel_edge_k : edge_t {base + column, base + row};
        }

        // Bridges: endpoint `j` in each community's hub core - its first 64 vertices, where R-MAT's
        // quadrant bias guarantees connectivity - so the ring can never be severed.
        vertex_t const hub_core = static_cast<vertex_t>(std::min<std::size_t>(community_vertices, 64));
        std::uint64_t const bridge_base = static_cast<std::uint64_t>(raw_edges) * 64;
        for (std::size_t j = 0; j < bridges; ++j) {
            vertex_t const u = static_cast<vertex_t>((j << scale) + random_index(bridge_base + 2 * j, hub_core));
            vertex_t const v = static_cast<vertex_t>((((j + 1) % communities) << scale) +
                                                     random_index(bridge_base + 2 * j + 1, hub_core));
            edges[raw_edges * 2 + j * 2] = {u, v};
            edges[raw_edges * 2 + j * 2 + 1] = {v, u};
        }

        // GCC's libstdc++ parallel mode sorts with the OpenMP threads already linked here; other
        // compilers keep the serial sort. Both produce the same array - the keys form one multiset.
#if defined(_OPENMP) && defined(__GNUC__) && !defined(__clang__)
        __gnu_parallel::sort(edges.begin(), edges.end());
#else
        std::sort(edges.begin(), edges.end());
#endif
        std::size_t edge_count = static_cast<std::size_t>(std::unique(edges.begin(), edges.end()) - edges.begin());
        // At most one sentinel survives `unique`, at the very end - trim it with the duplicates.
        while (edge_count && edges[edge_count - 1] == sentinel_edge_k) --edge_count;

        // CSR: count the degrees into `row_offsets`, then prefix-sum them into row starts.
        if (!graph.row_offsets.try_resize(vertices + 1)) return false; // ? Zero-filled
        for (std::size_t i = 0; i < edge_count; ++i) graph.row_offsets[edges[i].row + 1]++;
        for (vertex_t v = 0; v < vertices; ++v) graph.row_offsets[v + 1] += graph.row_offsets[v];

        if (!graph.column_indices.try_resize(edge_count)) return false;
        fu::dynamic_array<edge_offset_t> cursor;
        if (!cursor.try_resize(vertices)) return false;
        for (vertex_t v = 0; v < vertices; ++v) cursor[v] = graph.row_offsets[v];
        for (std::size_t i = 0; i < edge_count; ++i) graph.column_indices[cursor[edges[i].row]++] = edges[i].column;
    }
    return true;
}

#pragma endregion Graph

#pragma region Kernel

/** @brief The smallest label visible from @p v: its own, or the smallest among its neighbours'. */
static inline label_t min_label_of(csr_view_t const &graph, label_t const *old_labels, vertex_t const v) noexcept {
    label_t best = old_labels[v];
    edge_offset_t const end = graph.row_offsets[v + 1];
    for (edge_offset_t position = graph.row_offsets[v]; position != end; ++position) {
        label_t const candidate = old_labels[graph.column_indices[position]];
        if (candidate < best) best = candidate;
    }
    return best;
}

/** @brief Converges serially from `labels[v] = v`, returning the rounds taken - the reference for `PROPAGATION_CHECK`.
 */
static std::size_t converge_serially(csr_view_t const &graph, label_t *labels_a, label_t *labels_b) noexcept {
    vertex_t const vertices = graph.vertices();
    for (vertex_t v = 0; v < vertices; ++v) labels_a[v] = v;
    std::size_t rounds = 0;
    for (bool changed = true; changed; ++rounds, std::swap(labels_a, labels_b)) {
        changed = false;
        for (vertex_t v = 0; v < vertices; ++v) {
            label_t const next = min_label_of(graph, labels_a, v);
            labels_b[v] = next;
            changed |= next != labels_a[v];
        }
    }
    // After an odd number of rounds the fixed point sits in the other buffer; both hold it anyway,
    // since the terminal round changed nothing - so no copy is needed.
    return rounds;
}

#pragma endregion Kernel

#pragma region Backends

/** @brief Per-thread change tally, spaced so two threads never share a cache line. */
struct alignas(fu::default_alignment_k) counter_t {
    std::uint64_t value {0};
};

using distributed_pool_t = fu::distributed_pool<fu::preferred_yield_t, fu::preferred_cache_hints_t>;

/**
 *  @brief One read-only replica of the CSR per memory domain, so no adjacency is ever remote.
 *  @note Only the immutable CSR replicates; the two label buffers stay shared by nature - every
 *      round's writes are remote for somebody, whichever node holds them.
 */
struct replicated_csr_t {
    fu::replicated_array<edge_offset_t> row_offsets;
    fu::replicated_array<vertex_t> column_indices;

    bool try_build(csr_host_t const &host, fu::machine_topology_t const &topology) noexcept {
        return replicate(row_offsets, host.row_offsets, topology) &&
               replicate(column_indices, host.column_indices, topology);
    }

    csr_view_t on_memory_domain(fu::memory_domain_index_t const memory_domain) const noexcept {
        return {row_offsets.on_memory_domain(memory_domain), column_indices.on_memory_domain(memory_domain)};
    }

  private:
    template <typename value_type_>
    static bool replicate(fu::replicated_array<value_type_> &destination, fu::dynamic_array<value_type_> const &host,
                          fu::machine_topology_t const &topology) noexcept {
        if (!destination.try_resize_uninitialized(topology, host.size())) return false;
        for (std::size_t domain = 0; domain < destination.memory_domains_count(); ++domain) {
            fu::span<value_type_> const slice =
                destination.on_memory_domain(static_cast<fu::memory_domain_index_t>(domain));
            std::memcpy(slice.data(), host.data(), host.size() * sizeof(value_type_));
        }
        return true;
    }
};

/**
 *  @brief Rewrites @p array into fresh pages, first-touched by @p pool's pinned threads.
 *
 *  Generation first-touches pages on whichever cores the OS handed the unpinned worker threads, so
 *  every process rolls a different page placement and throughput swings ~2x run to run. Copying
 *  into virgin pages from the static split of @b pinned threads makes placement a pure function of
 *  the topology - identical for every backend, process, and language.
 */
template <typename value_type_>
static bool retouch_deterministically(distributed_pool_t &pool, fu::dynamic_array<value_type_> &array) noexcept {
    fu::dynamic_array<value_type_> placed;
    if (!placed.try_resize_uninitialized(array.size())) return false; // ? Pages stay unfaulted until the copy
    value_type_ const *source = array.data();
    value_type_ *destination = placed.data();
    pool.for_slices(array.size(), [=](distributed_pool_t::prong_t prong, std::size_t count) noexcept {
        std::memcpy(destination + prong.task, source + prong.task, count * sizeof(value_type_));
    });
    array = std::move(placed);
    return true;
}

/** @brief Everything a backend reads or writes for one convergence pass; the harness owns the lifetimes. */
struct run_context_t {
    csr_view_t graph;                       // ? The shared host view - what every non-replicated backend reads
    replicated_csr_t const &replicas;       // ? Per-node replicas, populated only for the `_replicated` cells
    fu::machine_topology_t const &topology; // ? The compute-to-memory bridge for the replicated read
    fu::span<counter_t> counters;           // ? Per-thread change tallies, zeroed each round
    fu::span<label_t> labels_a;             // ? Ping-pong label buffers; the fixed point ends in both
    fu::span<label_t> labels_b;
    std::size_t threads;
    std::size_t rounds = 0;             // ? Rounds to convergence, written back by every backend
    distributed_pool_t *pool = nullptr; // ? Spawned by `main` only for the ForkUnion backends
    tf::Executor *taskflow = nullptr;   // ? Spawned by `main` only for the `taskflow_*` backends
};

/** @brief Pre-split across threads vs work-stolen. */
enum class schedule_k : unsigned int { static_k, dynamic_k };
/** @brief One shared CSR vs one read-only CSR replica per memory domain. */
enum class placement_k : unsigned int { shared_k, replicated_k };

/** @brief Runs @p body over `[0, n)`, statically pre-split or work-stolen per the compile-time schedule. */
template <schedule_k schedule_, typename body_type_>
static void for_n_scheduled(distributed_pool_t &pool, std::size_t const n, body_type_ body) noexcept {
    if constexpr (schedule_ == schedule_k::static_k) pool.for_n(n, body);
    else
        pool.for_n_dynamic(n, body);
}

/** @brief Zeroes the per-thread tallies and sums them - the tiny serial bookends of every round. */
static void zero_counters(fu::span<counter_t> counters) noexcept {
    for (std::size_t t = 0; t < counters.size(); ++t) counters[t].value = 0;
}
static std::uint64_t sum_counters(fu::span<counter_t> counters) noexcept {
    std::uint64_t total = 0;
    for (std::size_t t = 0; t < counters.size(); ++t) total += counters[t].value;
    return total;
}

/**
 *  @brief One convergence pass, specialized over the two axes at compile time.
 *
 *  Four ForkUnion backends are the four instantiations of this one body - `if constexpr` picks the
 *  schedule and where each thread reads its CSR from. Every round is one fork-join dispatch, so the
 *  pool's dispatch-and-join cost is paid `rounds` times per pass - the axis this benchmark controls.
 */
template <schedule_k schedule_, placement_k placement_>
static void run(run_context_t &c) noexcept {
    using local_prong_t = typename distributed_pool_t::prong_t;
    auto graph_at = [&](std::size_t compute_domain) noexcept -> csr_view_t {
        if constexpr (placement_ == placement_k::replicated_k)
            return c.replicas.on_memory_domain(
                c.topology.local_memory_of(static_cast<fu::compute_domain_index_t>(compute_domain)));
        else
            return c.graph;
    };

    vertex_t const vertices = c.graph.vertices();
    label_t *old_labels = c.labels_a.data(), *new_labels = c.labels_b.data();
    for (vertex_t v = 0; v < vertices; ++v) old_labels[v] = v;

    std::size_t rounds = 0;
    for (std::uint64_t changes = 1; changes != 0; ++rounds, std::swap(old_labels, new_labels)) {
        zero_counters(c.counters);
        for_n_scheduled<schedule_>(*c.pool, vertices, [&](local_prong_t prong) noexcept {
            vertex_t const v = static_cast<vertex_t>(prong.task);
            label_t const next = min_label_of(graph_at(prong.compute_domain), old_labels, v);
            new_labels[v] = next;
            c.counters[prong.thread].value += next != old_labels[v];
        });
        changes = sum_counters(c.counters);
    }
    c.rounds = rounds;
}

#if defined(_OPENMP)
/** @brief The OpenMP baselines - one `parallel for` with a change reduction per round. */
template <bool dynamic_>
static void run_openmp(run_context_t &c) noexcept {
    csr_view_t const graph = c.graph;
    vertex_t const vertices = graph.vertices();
    label_t *old_labels = c.labels_a.data(), *new_labels = c.labels_b.data();
    for (vertex_t v = 0; v < vertices; ++v) old_labels[v] = v;

    std::size_t rounds = 0;
    for (std::uint64_t changes = 1; changes != 0; ++rounds, std::swap(old_labels, new_labels)) {
        std::uint64_t round_changes = 0;
        if constexpr (dynamic_) {
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : round_changes)
            for (vertex_t v = 0; v < vertices; ++v) {
                label_t const next = min_label_of(graph, old_labels, v);
                new_labels[v] = next;
                round_changes += next != old_labels[v];
            }
        }
        else {
#pragma omp parallel for schedule(static) reduction(+ : round_changes)
            for (vertex_t v = 0; v < vertices; ++v) {
                label_t const next = min_label_of(graph, old_labels, v);
                new_labels[v] = next;
                round_changes += next != old_labels[v];
            }
        }
        changes = round_changes;
    }
    c.rounds = rounds;
}
static void run_openmp_static(run_context_t &c) noexcept { run_openmp<false>(c); }
static void run_openmp_dynamic(run_context_t &c) noexcept { run_openmp<true>(c); }
#endif

/**
 *  @brief The Taskflow baselines - a fresh `tf::Taskflow` per round on the long-lived executor.
 *  @note The per-round flow construction is charged to Taskflow by design: this benchmark measures
 *      exactly the cost of standing up one fork-join round, and a persisted flow would hide it.
 */
template <typename partitioner_>
static void run_taskflow(run_context_t &c, partitioner_ partitioner) noexcept {
    csr_view_t const graph = c.graph;
    fu::span<counter_t> const counters = c.counters;
    tf::Executor &executor = *c.taskflow;
    vertex_t const vertices = graph.vertices();
    label_t *old_labels = c.labels_a.data(), *new_labels = c.labels_b.data();
    for (vertex_t v = 0; v < vertices; ++v) old_labels[v] = v;

    std::size_t rounds = 0;
    for (std::uint64_t changes = 1; changes != 0; ++rounds, std::swap(old_labels, new_labels)) {
        zero_counters(counters);
        tf::Taskflow flow;
        flow.for_each_index(
            vertex_t(0), vertices, vertex_t(1),
            [&, old_labels, new_labels](vertex_t v) noexcept {
                label_t const next = min_label_of(graph, old_labels, v);
                new_labels[v] = next;
                counters[static_cast<std::size_t>(executor.this_worker_id())].value += next != old_labels[v];
            },
            partitioner);
        executor.run(flow).wait();
        changes = sum_counters(counters);
    }
    c.rounds = rounds;
}
static void run_taskflow_static(run_context_t &c) noexcept { run_taskflow(c, tf::StaticPartitioner()); }
static void run_taskflow_dynamic(run_context_t &c) noexcept { run_taskflow(c, tf::DynamicPartitioner(1)); }

/** @brief Which execution engine a backend runs on, so `main` builds exactly the resource it needs. */
enum class engine_t : unsigned int {
    forkunion_k,            // ? Spawns the shared ForkUnion pool
    forkunion_replicated_k, // ? Also builds the per-node CSR replicas
    openmp_k,               // ? Runs under `omp parallel for`, needing no pool object
    taskflow_k,             // ? Runs on a reused `tf::Executor`, needing no pool object
};

/** @brief The dispatch table - a name, its convergence pass, and the engine it runs on. */
struct backend_t {
    std::string_view name;
    void (*run)(run_context_t &) noexcept;
    engine_t engine;
};

using sc = schedule_k;
using pl = placement_k;
static constexpr backend_t backends_k[] = {
    {"forkunion_static_shared", &run<sc::static_k, pl::shared_k>, engine_t::forkunion_k},
    {"forkunion_dynamic_shared", &run<sc::dynamic_k, pl::shared_k>, engine_t::forkunion_k},
    {"forkunion_static_replicated", &run<sc::static_k, pl::replicated_k>, engine_t::forkunion_replicated_k},
    {"forkunion_dynamic_replicated", &run<sc::dynamic_k, pl::replicated_k>, engine_t::forkunion_replicated_k},
#if defined(_OPENMP)
    {"openmp_static", run_openmp_static, engine_t::openmp_k},
    {"openmp_dynamic", run_openmp_dynamic, engine_t::openmp_k},
#endif
    {"taskflow_static", run_taskflow_static, engine_t::taskflow_k},
    {"taskflow_dynamic", run_taskflow_dynamic, engine_t::taskflow_k},
};

#pragma endregion Backends

/** @brief Reads an environment variable, or @p fallback when unset - `getenv_s` on MSVC. */
static char const *env_string(char const *name, char const *fallback) noexcept {
#if defined(_MSC_VER)
    static char buffer[256];
    std::size_t required = 0;
    return (getenv_s(&required, buffer, sizeof(buffer), name) == 0 && required > 0) ? buffer : fallback;
#else
    char const *value = std::getenv(name);
    return value ? value : fallback;
#endif
}

/** @brief Parses a fractional environment variable, or @p fallback when unset. */
static double env_double(char const *name, double fallback) noexcept {
    char const *value = env_string(name, nullptr);
    return value ? std::atof(value) : fallback;
}

/** @brief Parses an unsigned environment variable, or @p fallback when unset. */
static std::size_t env_usize(char const *name, std::size_t fallback) noexcept {
    char const *value = env_string(name, nullptr);
    return value ? static_cast<std::size_t>(std::strtoull(value, nullptr, 10)) : fallback;
}

int main() {
    std::size_t const scale = env_usize("PROPAGATION_SCALE", 14);
    std::size_t const communities = env_usize("PROPAGATION_COMMUNITIES", 64);
    std::size_t const edge_factor = env_usize("PROPAGATION_EDGE_FACTOR", 16);
    std::string_view const backend = env_string("PROPAGATION_BACKEND", "forkunion_static_shared");
    std::size_t threads = env_usize("PROPAGATION_THREADS", 0);
    double const budget_seconds = env_double("PROPAGATION_SECONDS", 10);   // ? The primary knob: a fixed window
    std::size_t const iterations = env_usize("PROPAGATION_ITERATIONS", 0); // ? Overrides with an exact count when set
    bool const check = env_string("PROPAGATION_CHECK", nullptr) != nullptr;
    if (threads == 0) threads = fu::allowed_cores_count();
    if ((communities << scale) > (std::size_t(1) << 32)) {
        std::fprintf(stderr, "PROPAGATION_COMMUNITIES << PROPAGATION_SCALE must fit 32-bit vertex indices\n");
        return EXIT_FAILURE;
    }

    csr_host_t host;
    if (!generate_necklace(scale, communities, edge_factor, host)) {
        std::fprintf(stderr, "Failed to allocate the graph\n");
        return EXIT_FAILURE;
    }
    csr_view_t const graph = host.view();
    vertex_t const vertices = graph.vertices();
    std::printf("vertices %u, directed edges %zu, communities %zu\n", vertices, static_cast<std::size_t>(graph.edges()),
                communities);

    backend_t const *selected = nullptr;
    for (backend_t const &entry : backends_k)
        if (entry.name == backend) selected = &entry;
    if (!selected) {
        std::fprintf(stderr, "Unsupported backend: %.*s\n", static_cast<int>(backend.size()), backend.data());
        std::fprintf(stderr, "Available backends:");
        for (backend_t const &entry : backends_k)
            std::fprintf(stderr, " %.*s", static_cast<int>(entry.name.size()), entry.name.data());
        std::fprintf(stderr, "\n");
        return EXIT_FAILURE;
    }

    // One pinned pool spawns for EVERY backend - first to give the graph and label pages their
    // deterministic first touch, then to serve the ForkUnion backends; the others drop it below.
    bool const needs_pool =
        selected->engine == engine_t::forkunion_k || selected->engine == engine_t::forkunion_replicated_k;
    fu::machine_topology_t topology;
    replicated_csr_t replicas;
    std::optional<distributed_pool_t> pool;
    std::optional<tf::Executor> taskflow; // ? Spawned for the Taskflow backends
    if (!topology.try_harvest()) {
        std::fprintf(stderr, "Failed to harvest the memory topology\n");
        return EXIT_FAILURE;
    }
    pool.emplace();
    if (!pool->try_spawn(topology, threads)) {
        std::fprintf(stderr, "Failed to spawn the thread pool\n");
        return EXIT_FAILURE;
    }

    fu::dynamic_array<counter_t> counters;
    fu::dynamic_array<label_t> labels_a, labels_b;
    if (!counters.try_resize(threads) || !labels_a.try_resize(vertices) || !labels_b.try_resize(vertices)) {
        std::fprintf(stderr, "Failed to allocate the labels\n");
        return EXIT_FAILURE;
    }

    bool const retouched = retouch_deterministically(*pool, host.row_offsets) &&
                           retouch_deterministically(*pool, host.column_indices) &&
                           retouch_deterministically(*pool, labels_a) && retouch_deterministically(*pool, labels_b);
    if (!retouched) {
        std::fprintf(stderr, "Failed to place the graph deterministically\n");
        return EXIT_FAILURE;
    }
    csr_view_t const placed_graph = host.view(); // ! Retouch reallocates; earlier views are stale

    if (needs_pool && selected->engine == engine_t::forkunion_replicated_k && !replicas.try_build(host, topology)) {
        std::fprintf(stderr, "Failed to replicate the graph across memory domains\n");
        return EXIT_FAILURE;
    }
    if (!needs_pool) pool.reset(); // ? Frees the cores before OpenMP or Taskflow spawn their own workers
    if (selected->engine == engine_t::taskflow_k) taskflow.emplace(threads);
#if defined(_OPENMP)
    omp_set_num_threads(static_cast<int>(threads));
#endif

    run_context_t context {placed_graph,
                           replicas,
                           topology,
                           {counters.data(), counters.size()},
                           {labels_a.data(), labels_a.size()},
                           {labels_b.data(), labels_b.size()},
                           threads};
    if (pool) context.pool = &*pool;
    if (taskflow) context.taskflow = &*taskflow;

    // One untimed warmup pass: page-faults and cache warming would otherwise bias the first timed
    // pass, and by a different amount for each backend.
    selected->run(context);

    // A fixed time budget beats a fixed pass count: every backend runs the same wall-clock window -
    // long enough to amortize scheduling noise - and reports the rate it sustained, with no
    // per-backend pass-count guessing. `PROPAGATION_ITERATIONS` forces an exact count instead.
    auto const started = std::chrono::steady_clock::now();
    std::size_t passes = 0;
    if (iterations > 0)
        for (; passes < iterations; ++passes) selected->run(context);
    else
        do {
            selected->run(context), ++passes;
        } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() < budget_seconds);
    double const seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() //
                           / static_cast<double>(passes);

    // The fixed point sits in both buffers - the terminal round changed nothing - so read either.
    std::uint64_t components = 0, checksum = 0;
    for (vertex_t v = 0; v < vertices; ++v) {
        components += labels_a[v] == v;
        checksum += labels_a[v];
    }
    // MTEPS - millions of directed edges scanned per second; `rounds * edges` is the exact scan
    // count, identical in every cell by the double-buffered determinism.
    double const mteps = static_cast<double>(context.rounds) * static_cast<double>(graph.edges()) / seconds / 1e6;
    std::printf("%.*s: %zu components, %zu rounds, checksum %zu, %.2f s/pass, %.1f MTEPS\n",
                static_cast<int>(backend.size()), backend.data(), static_cast<std::size_t>(components), context.rounds,
                static_cast<std::size_t>(checksum), seconds, mteps);

    if (check) {
        fu::dynamic_array<label_t> serial_a, serial_b;
        if (!serial_a.try_resize(vertices) || !serial_b.try_resize(vertices)) {
            std::fprintf(stderr, "Failed to allocate the reference labels\n");
            return EXIT_FAILURE;
        }
        std::size_t const serial_rounds = converge_serially(placed_graph, serial_a.data(), serial_b.data());
        bool const same_labels = std::memcmp(serial_a.data(), labels_a.data(), vertices * sizeof(label_t)) == 0;
        if (serial_rounds != context.rounds || !same_labels) {
            std::fprintf(stderr, "MISMATCH: serial converged in %zu rounds\n", serial_rounds);
            return EXIT_FAILURE;
        }
        std::printf("check: matches the serial labels and rounds\n");
    }
    return EXIT_SUCCESS;
}
