/**
 *  @brief Demo app: triangle counting on a power-law graph, with ForkUnion and OpenMP.
 *  @author Ash Vardanian
 *  @file triangles.cpp
 *
 *  The N-body simulation gives every task an identical cost, so it can only measure dispatch
 *  latency. Triangle counting gives them wildly different costs: the work at a vertex grows with
 *  its degree @b and with the degrees of its neighbours, and an R-MAT graph draws those degrees
 *  from a power law. A handful of hub vertices carry most of the arithmetic.
 *
 *  Crucially, the hubs are @b adjacent. R-MAT keeps recursing into the same quadrant, so the
 *  high-degree vertices cluster at low indices, and a static split hands one thread nearly all of
 *  them. Scattering the heavy tasks - by relabelling the vertices - would let the law of large
 *  numbers balance the slices again, and this benchmark would stop measuring what it exists for.
 *
 *  @section Two Ways to Parallelize
 *
 *  The obvious decomposition is one task per @b vertex, which is what `vertex_centric_static` and
 *  `vertex_centric_dynamic` do. It is also the reason `vertex_centric_static` is hopeless: the slice holding the
 *  hubs decides the makespan, and no amount of arithmetic elsewhere hides it.
 *
 *  A GPU would never write it that way. It flattens the adjacency into a @b tape of equally-sized
 *  work items - here, one item per `u < v` edge - and hands each thread a contiguous run of the
 *  tape. Recovering which vertex an item belongs to costs one binary search into a prefix sum,
 *  paid @b once per slice rather than once per item, exactly as `merge-path` and the
 *  `load-balancing search` of segmented GPU primitives do. Balance then comes from the layout
 *  instead of from a scheduler, so `tape_slices` needs no atomics and no stealing at all.
 *
 *  Balancing the item @b count is not enough, though: an item on a hub scans a longer adjacency, so
 *  `tape_slices` still sinks the slice that owns the hubs. The fix is to weigh each item by the work
 *  it will do - `degree(u) + degree(v)` - prefix-sum those weights, and cut the @b cost axis into
 *  equal parts. A binary search maps each cut back onto the item axis. That is what `edge_centric`
 *  does, and it is why a static dispatch can beat a stealing one here.
 *
 *  @section Memory
 *
 *  The CSR is read-only once built, so every memory domain gets its own replica - a `replicated_array`
 *  per CSR array, striped across the nodes by the symmetric allocator - and no thread ever reaches
 *  across the interconnect for a neighbour list. `edge_centric_replicated` cuts the cost axis between
 *  compute domains, then between the threads of each, and reads only the local replica.
 *
 *  To control the script, several environment variables are used:
 *
 *  - `TRIANGLES_SCALE` - the graph has `2^scale` vertices - default 18.
 *  - `TRIANGLES_EDGE_FACTOR` - edges generated per vertex, before deduplication - default 16.
 *  - `TRIANGLES_BACKEND` - backend to use - default `forkunion_vertex_centric_static`.
 *  - `TRIANGLES_THREADS` - number of threads to use - default all hardware threads.
 *  - `TRIANGLES_ITERATIONS` - repeat the count this many times, reporting the per-pass time - default 1.
 *  - `TRIANGLES_CHECK` - also count serially, and fail unless the totals agree.
 *
 *  The backends include: `forkunion_vertex_centric_static`, `forkunion_vertex_centric_dynamic`,
 * `forkunion_edge_centric`, `forkunion_edge_centric_replicated`, `openmp_static`, `openmp_dynamic`, and
 * `openmp_guided`. To compile and run:
 *
 *  @code{.sh}
 *  cmake -B build_release -D CMAKE_BUILD_TYPE=Release
 *  cmake --build build_release --config Release
 *  time TRIANGLES_SCALE=20 TRIANGLES_BACKEND=forkunion_edge_centric build_release/forkunion_triangles
 *  @endcode
 */
#include <algorithm> // `std::sort`, `std::unique`, `std::lower_bound`, `std::upper_bound`
#include <chrono>    // `std::chrono::steady_clock`
#include <cstdint>   // `std::uint32_t`
#include <cstdio>    // `std::printf`
#include <cstdlib>   // `std::getenv`, `EXIT_SUCCESS`
#include <cstring>   // `std::memcpy`
#include <random>    // `std::mt19937_64`
#include <string_view>

#if defined(_OPENMP)
#if __has_include(<omp.h>)
#include <omp.h>
#else
#undef _OPENMP
#endif
#endif

#include <forkunion.hpp>

namespace fu = ashvardanian::forkunion;

using vertex_t = std::uint32_t;
using edge_offset_t = std::uint64_t;
using tape_offset_t = std::uint64_t;
using degree_t = std::uint32_t; // ? A within-row neighbour count, bounded by the vertex count
using work_t = std::uint64_t;   // ? A tape-item cost `degree(u) + degree(v)`, and the prefix sum over them

#pragma region Graph

/**
 *  @brief A read-only CSR-plus-tape as five spans - the interface every kernel takes.
 *
 *  `row_offsets` and `column_indices` are the usual CSR pair, each adjacency sorted ascending. The rest
 *  turns a tape index back into an edge: `above_offsets[u]` is where the neighbours greater than `u`
 *  begin within `u`'s row - a hot-path cache of `degree(u)` minus `u`'s tape span - `tape_offsets[u]` is
 *  the prefix sum of tape items before `u`, and `work_offsets` is the prefix sum of each item's cost
 *  `degree(u) + degree(v)`. The view is agnostic to where the bytes live - the host build, or a replica.
 */
struct csr_view_t {
    fu::span<edge_offset_t const> row_offsets;
    fu::span<vertex_t const> column_indices;
    fu::span<tape_offset_t const> tape_offsets;
    fu::span<degree_t const> above_offsets;
    fu::span<work_t const> work_offsets;

    vertex_t vertices() const noexcept { return static_cast<vertex_t>(row_offsets.size() - 1); }
    edge_offset_t edges() const noexcept { return column_indices.size(); }
    degree_t degree(vertex_t const v) const noexcept {
        return static_cast<degree_t>(row_offsets[v + 1] - row_offsets[v]);
    }

    /** @brief Number of `u < v` pairs; the length of the work tape. */
    tape_offset_t tape_length() const noexcept { return tape_offsets.back(); }
    /** @brief Total cost of the tape, in units of "one adjacency element compared". */
    work_t total_work() const noexcept { return work_offsets.back(); }

    /** @brief The first tape item whose cumulative cost reaches @p work - the `merge-path` search. */
    tape_offset_t item_at_work(work_t const work) const noexcept {
        auto const it = std::lower_bound(work_offsets.begin(), work_offsets.end(), work);
        return static_cast<tape_offset_t>(it - work_offsets.begin());
    }
    /** @brief The vertex owning tape item @p item, by binary search into the prefix sum. */
    vertex_t owner_of(tape_offset_t const item) const noexcept {
        auto const it = std::upper_bound(tape_offsets.begin(), tape_offsets.end(), item);
        return static_cast<vertex_t>((it - tape_offsets.begin()) - 1);
    }
};

/** @brief The five CSR-plus-tape arrays built once on the host, in growable `dynamic_array`s. */
struct csr_host_t {
    fu::dynamic_array<edge_offset_t> row_offsets;
    fu::dynamic_array<vertex_t> column_indices;
    fu::dynamic_array<tape_offset_t> tape_offsets;
    fu::dynamic_array<degree_t> above_offsets;
    fu::dynamic_array<work_t> work_offsets;

    csr_view_t view() const noexcept {
        return {{row_offsets.data(), row_offsets.size()},
                {column_indices.data(), column_indices.size()},
                {tape_offsets.data(), tape_offsets.size()},
                {above_offsets.data(), above_offsets.size()},
                {work_offsets.data(), work_offsets.size()}};
    }
};

/** @brief A directed edge, ordered so `std::sort` + `std::unique` deduplicate the adjacency. */
struct edge_t {
    vertex_t row, column;
    bool operator<(edge_t const &o) const noexcept { return row != o.row ? row < o.row : column < o.column; }
    bool operator==(edge_t const &o) const noexcept { return row == o.row && column == o.column; }
};

/**
 *  @brief Generates a Kronecker/R-MAT graph, as specified by Graph500, and builds its tape into @p graph.
 *  @retval false on any allocation failure, leaving @p graph half-built but valid to destroy.
 *  @note Recursing into the `a` quadrant with probability 57% is what makes the degrees power-law,
 *        and what keeps the hubs near index zero, where a static split will trip over them.
 */
static bool generate_rmat(std::size_t const scale, std::size_t const edge_factor, csr_host_t &graph) noexcept {
    vertex_t const vertices = static_cast<vertex_t>(1u) << scale;
    std::size_t const raw_edges = static_cast<std::size_t>(vertices) * edge_factor;

    // Build the COO edge list, dedupe it, and scatter it into CSR - scoped so `edges` (up to
    // `2 * raw_edges` items) frees before the tape build and never coexists with the work arrays at peak.
    {
        std::mt19937_64 rng(0x1234'5678'9ABC'DEF0ull);
        fu::dynamic_array<edge_t> edges;
        if (!edges.try_reserve(raw_edges * 2)) return false;

        for (std::size_t e = 0; e < raw_edges; ++e) {
            vertex_t row = 0, column = 0;
            for (int bit = static_cast<int>(scale) - 1; bit >= 0; --bit) {
                unsigned const r = static_cast<unsigned>(rng() % 100); // ? `a=57 b=19 c=19 d=5`, integer and portable
                vertex_t const step = static_cast<vertex_t>(1u) << bit;
                if (r < 57) continue; // ? Stay in the dense quadrant
                else if (r < 76)
                    column |= step;
                else if (r < 95)
                    row |= step;
                else
                    row |= step, column |= step;
            }
            if (row == column) continue;                           // ? Drop self-loops
            if (!edges.try_push_back({row, column})) return false; // ? Symmetrize below
            if (!edges.try_push_back({column, row})) return false;
        }

        std::sort(edges.begin(), edges.end());
        std::size_t const edge_count =
            static_cast<std::size_t>(std::unique(edges.begin(), edges.end()) - edges.begin());

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

    // Tape: one item per `u < v` pair, in vertex order.
    if (!graph.above_offsets.try_resize(vertices)) return false;
    if (!graph.tape_offsets.try_resize(vertices + 1)) return false; // ? Zero-filled
    for (vertex_t u = 0; u < vertices; ++u) {
        vertex_t const *const row = graph.column_indices.data() + graph.row_offsets[u];
        vertex_t const *const row_end = graph.column_indices.data() + graph.row_offsets[u + 1];
        vertex_t const *const above = std::upper_bound(row, row_end, u);
        graph.above_offsets[u] = static_cast<degree_t>(above - row);
        graph.tape_offsets[u + 1] = graph.tape_offsets[u] + static_cast<tape_offset_t>(row_end - above);
    }

    // Weigh each tape item by the two adjacencies it intersects. Balancing the item count leaves the
    // hubs bunched into one slice; balancing this weight is what spreads them.
    tape_offset_t const tape_length = graph.tape_offsets[vertices];
    if (!graph.work_offsets.try_resize(tape_length + 1)) return false; // ? Zero-filled
    for (vertex_t u = 0; u < vertices; ++u) {
        edge_offset_t const u_degree = graph.row_offsets[u + 1] - graph.row_offsets[u];
        edge_offset_t position = graph.row_offsets[u] + graph.above_offsets[u];
        for (tape_offset_t item = graph.tape_offsets[u]; item != graph.tape_offsets[u + 1]; ++item, ++position) {
            vertex_t const v = graph.column_indices[position];
            edge_offset_t const v_degree = graph.row_offsets[v + 1] - graph.row_offsets[v];
            graph.work_offsets[item + 1] = graph.work_offsets[item] + u_degree + v_degree;
        }
    }
    return true;
}

#pragma endregion Graph

#pragma region Kernel

/**
 *  @brief Counts triangles `u < v < w` for the single edge `(u, v)` at tape item @p item.
 *  @param[in] owner The vertex owning @p item, already resolved by the caller.
 *
 *  Both adjacencies are sorted, so the candidates for `w` are the suffix of `N(u)` past `v` and the
 *  suffix of `N(v)` past `v`. Intersecting them counts each triangle exactly once, and needs no atomics.
 */
static inline std::uint64_t count_triangles_on_item(csr_view_t const &graph, tape_offset_t const item,
                                                    vertex_t const owner) noexcept {
    vertex_t const *const columns = graph.column_indices.data();
    edge_offset_t const owner_end = graph.row_offsets[owner + 1];
    edge_offset_t const position = graph.row_offsets[owner] + graph.above_offsets[owner] + //
                                   static_cast<edge_offset_t>(item - graph.tape_offsets[owner]);

    vertex_t const other = columns[position];
    vertex_t const *a = columns + position + 1, *const a_end = columns + owner_end;
    vertex_t const *b = columns + graph.row_offsets[other] + graph.above_offsets[other];
    vertex_t const *const b_end = columns + graph.row_offsets[other + 1];

    std::uint64_t triangles = 0;
    while (a != a_end && b != b_end) {
        if (*a < *b) ++a;
        else if (*b < *a)
            ++b;
        else
            ++triangles, ++a, ++b;
    }
    return triangles;
}

/** @brief Counts every triangle whose smallest vertex is @p u. Cost grows with `degree(u)` squared. */
static inline std::uint64_t count_triangles_at_vertex(csr_view_t const &graph, vertex_t const u) noexcept {
    std::uint64_t triangles = 0;
    for (tape_offset_t item = graph.tape_offsets[u]; item != graph.tape_offsets[u + 1]; ++item)
        triangles += count_triangles_on_item(graph, item, u);
    return triangles;
}

/**
 *  @brief Walks a contiguous run of the tape, resolving the owning vertex only when it changes.
 *  @note One binary search per slice, then a forward walk - the CPU spelling of a merge-path.
 */
static inline std::uint64_t count_triangles_on_slice(csr_view_t const &graph, tape_offset_t const first,
                                                     tape_offset_t const count) noexcept {
    if (count == 0) return 0;
    std::uint64_t triangles = 0;
    vertex_t owner = graph.owner_of(first);
    for (tape_offset_t item = first; item != first + count; ++item) {
        while (item >= graph.tape_offsets[owner + 1]) ++owner; // ? Amortized O(1) per item
        triangles += count_triangles_on_item(graph, item, owner);
    }
    return triangles;
}

#pragma endregion Kernel

#pragma region Backends

/** @brief Per-thread accumulator, spaced so two threads never share a cache line. */
struct alignas(fu::default_alignment_k) counter_t {
    std::uint64_t value {0};
};

using distributed_pool_t = fu::distributed_pool<fu::preferred_yield_t>;

/**
 *  @brief One read-only replica of the CSR per memory domain, so no adjacency is ever remote.
 *
 *  Each CSR array is its own `replicated_array` - a symmetric mapping the allocator stripes across the
 *  nodes with `mbind`, so the one-time fill can be a plain serial copy and still land each slice on its
 *  own node. `on_memory_domain` assembles the five node-local slices back into a `csr_view_t`.
 */
struct replicated_csr_t {
    fu::replicated_array<edge_offset_t> row_offsets;
    fu::replicated_array<vertex_t> column_indices;
    fu::replicated_array<tape_offset_t> tape_offsets;
    fu::replicated_array<degree_t> above_offsets;
    fu::replicated_array<work_t> work_offsets;

    bool try_build(csr_host_t const &host, fu::machine_topology_t const &topology) noexcept {
        return replicate(row_offsets, host.row_offsets, topology) &&
               replicate(column_indices, host.column_indices, topology) &&
               replicate(tape_offsets, host.tape_offsets, topology) &&
               replicate(above_offsets, host.above_offsets, topology) &&
               replicate(work_offsets, host.work_offsets, topology);
    }

    csr_view_t on_memory_domain(fu::memory_domain_index_t const memory_domain) const noexcept {
        return {row_offsets.on_memory_domain(memory_domain), column_indices.on_memory_domain(memory_domain),
                tape_offsets.on_memory_domain(memory_domain), above_offsets.on_memory_domain(memory_domain),
                work_offsets.on_memory_domain(memory_domain)};
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

/** @brief Everything a backend reads or writes for one counting pass; the harness owns the lifetimes. */
struct run_context_t {
    csr_view_t graph;                       // ? The shared host view - what every non-replicated backend reads
    replicated_csr_t const &replicas;       // ? Per-node replicas, populated only for `edge_centric_replicated`
    distributed_pool_t &pool;               // ? One pool spawned for every ForkUnion backend
    fu::machine_topology_t const &topology; // ? The compute-to-memory bridge for the replicated split
    fu::span<counter_t> counters;           // ? Per-thread tallies the harness zeroes and sums
    std::size_t threads;
};

/** @brief One task per vertex, split statically - the naive baseline the hubs sink. */
static void run_vertex_centric_static(run_context_t &c) noexcept {
    c.pool.for_n(c.graph.vertices(), [&](fu::prong_t prong) noexcept {
        c.counters[prong.thread].value += count_triangles_at_vertex(c.graph, static_cast<vertex_t>(prong.task));
    });
}

/** @brief One task per vertex, work-stolen - stealing recovers the balance a static split loses. */
static void run_vertex_centric_dynamic(run_context_t &c) noexcept {
    c.pool.for_n_dynamic(c.graph.vertices(), [&](fu::prong_t prong) noexcept {
        c.counters[prong.thread].value += count_triangles_at_vertex(c.graph, static_cast<vertex_t>(prong.task));
    });
}

/** @brief Cut the cost axis into equal slices, merge-path each cut onto the item axis - atomics-free. */
static void run_edge_centric(run_context_t &c) noexcept {
    fu::indexed_split_t const work_split {c.graph.total_work(), c.threads};
    c.pool.for_threads([&](std::size_t thread) noexcept {
        fu::indexed_range_t const cost = work_split[thread];
        tape_offset_t const first = c.graph.item_at_work(cost.first);
        tape_offset_t const last = c.graph.item_at_work(cost.first + cost.count);
        c.counters[thread].value += count_triangles_on_slice(c.graph, first, last - first);
    });
}

/** @brief `edge_centric` with the cost axis cut per compute domain, each thread reading its node's replica. */
static void run_edge_centric_replicated(run_context_t &c) noexcept {
    c.pool.for_threads([&](fu::local_thread_t thread) noexcept {
        std::size_t const compute_domain = static_cast<std::size_t>(thread.compute_domain);
        fu::memory_domain_index_t const memory_domain =
            c.topology.local_memory_of(static_cast<fu::compute_domain_index_t>(compute_domain));
        csr_view_t const replica = c.replicas.on_memory_domain(memory_domain);

        std::size_t const domains = c.pool.compute_domains_count();
        std::size_t const threads_here = c.pool.threads_count(compute_domain);
        std::size_t const local_index = c.pool.thread_local_index(thread, compute_domain);

        // Two nested balanced splits of the cost axis: first between compute domains, then between the
        // threads of this domain - reusing the same fair-chunk splitter as the count-axis loops.
        fu::indexed_range_t const domain_cost = fu::indexed_split_t {replica.total_work(), domains}[compute_domain];
        fu::indexed_range_t const thread_cost = fu::indexed_split_t {domain_cost.count, threads_here}[local_index];
        tape_offset_t const first = replica.item_at_work(domain_cost.first + thread_cost.first);
        tape_offset_t const last = replica.item_at_work(domain_cost.first + thread_cost.first + thread_cost.count);
        c.counters[thread].value += count_triangles_on_slice(replica, first, last - first);
    });
}

#if defined(_OPENMP)
/** @brief The OpenMP baselines - one reduction, left in `counters[0]` for the harness to read. */
static void run_openmp_static(run_context_t &c) noexcept {
    vertex_t const vertices = c.graph.vertices();
    std::uint64_t sum = 0;
#pragma omp parallel for schedule(static) reduction(+ : sum)
    for (vertex_t v = 0; v < vertices; ++v) sum += count_triangles_at_vertex(c.graph, v);
    c.counters[0].value = sum;
}
static void run_openmp_dynamic(run_context_t &c) noexcept {
    vertex_t const vertices = c.graph.vertices();
    std::uint64_t sum = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : sum)
    for (vertex_t v = 0; v < vertices; ++v) sum += count_triangles_at_vertex(c.graph, v);
    c.counters[0].value = sum;
}
static void run_openmp_guided(run_context_t &c) noexcept {
    vertex_t const vertices = c.graph.vertices();
    std::uint64_t sum = 0;
#pragma omp parallel for schedule(guided) reduction(+ : sum)
    for (vertex_t v = 0; v < vertices; ++v) sum += count_triangles_at_vertex(c.graph, v);
    c.counters[0].value = sum;
}
#endif

/** @brief Which execution engine a backend runs on, so `main` builds exactly the resource it needs. */
enum class engine_t : unsigned int {
    forkunion_k,            // ? Spawns the shared ForkUnion pool
    forkunion_replicated_k, // ? Also builds the per-node CSR replicas
    openmp_k,               // ? Runs under `omp parallel for`, needing no pool object
};

/** @brief The dispatch table - a name, its counting pass, and the engine it runs on. */
struct backend_t {
    std::string_view name;
    void (*run)(run_context_t &) noexcept;
    engine_t engine;
};

static constexpr backend_t backends_k[] = {
    {"forkunion_vertex_centric_static", run_vertex_centric_static, engine_t::forkunion_k},
    {"forkunion_vertex_centric_dynamic", run_vertex_centric_dynamic, engine_t::forkunion_k},
    {"forkunion_edge_centric", run_edge_centric, engine_t::forkunion_k},
    {"forkunion_edge_centric_replicated", run_edge_centric_replicated, engine_t::forkunion_replicated_k},
#if defined(_OPENMP)
    {"openmp_static", run_openmp_static, engine_t::openmp_k},
    {"openmp_dynamic", run_openmp_dynamic, engine_t::openmp_k},
    {"openmp_guided", run_openmp_guided, engine_t::openmp_k},
#endif
};

#pragma endregion Backends

/** @brief Reads an environment variable, or @p fallback when unset - `getenv_s` on MSVC. */
static char const *env_string(char const *name, char const *fallback) noexcept {
#if defined(_MSC_VER)
    static thread_local char buffer[256];
    std::size_t required = 0;
    return (getenv_s(&required, buffer, sizeof(buffer), name) == 0 && required > 0) ? buffer : fallback;
#else
    char const *value = std::getenv(name);
    return value ? value : fallback;
#endif
}
/** @brief Parses an unsigned environment variable, or @p fallback when unset. */
static std::size_t env_usize(char const *name, std::size_t fallback) noexcept {
    char const *value = env_string(name, nullptr);
    return value ? static_cast<std::size_t>(std::strtoull(value, nullptr, 10)) : fallback;
}
/** @brief Whether an environment variable is present at all. */
static bool env_flag(char const *name) noexcept { return env_string(name, nullptr) != nullptr; }

int main() {
    std::size_t const scale = env_usize("TRIANGLES_SCALE", 18);
    std::size_t const edge_factor = env_usize("TRIANGLES_EDGE_FACTOR", 16);
    std::string_view const backend = env_string("TRIANGLES_BACKEND", "forkunion_vertex_centric_static");
    bool const check = env_flag("TRIANGLES_CHECK");
    std::size_t threads = env_usize("TRIANGLES_THREADS", 0);
    std::size_t iterations = env_usize("TRIANGLES_ITERATIONS", 1);
    if (threads == 0) threads = fu::allowed_cores_count();
    if (iterations == 0) iterations = 1;

    csr_host_t host;
    if (!generate_rmat(scale, edge_factor, host)) {
        std::fprintf(stderr, "Failed to allocate the graph\n");
        return EXIT_FAILURE;
    }
    csr_view_t const graph = host.view();
    vertex_t const vertices = graph.vertices();
    tape_offset_t const tape_length = graph.tape_length();

    degree_t max_degree = 0;
    for (vertex_t v = 0; v < vertices; ++v) max_degree = std::max(max_degree, graph.degree(v));
    std::printf("vertices %u, directed edges %zu, tape %zu, max degree %zu (mean %.1f)\n", vertices,
                static_cast<std::size_t>(graph.edges()), static_cast<std::size_t>(tape_length),
                static_cast<std::size_t>(max_degree), static_cast<double>(graph.edges()) / vertices);

    fu::dynamic_array<counter_t> counters;
    if (!counters.try_resize(threads)) {
        std::fprintf(stderr, "Failed to allocate the per-thread counters\n");
        return EXIT_FAILURE;
    }
    auto const zero_counters = [&]() noexcept {
        for (std::size_t t = 0; t < threads; ++t) counters[t].value = 0;
    };
    auto const sum_counters = [&]() noexcept -> std::uint64_t {
        std::uint64_t total = 0;
        for (std::size_t t = 0; t < threads; ++t) total += counters[t].value;
        return total;
    };

    // Runs @p pass `iterations` times, timing it; the pass leaves its per-thread tallies in `counters`,
    // which are zeroed before each run and summed into `triangles` after the last.
    std::uint64_t triangles = 0;
    auto const timed = [&](auto &&pass) {
        auto const started = std::chrono::steady_clock::now();
        for (std::size_t it = 0; it < iterations; ++it) {
            zero_counters();
            pass();
        }
        double const seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() //
                               / static_cast<double>(iterations);
        triangles = sum_counters();
        std::printf("%.*s: %zu triangles in %.3f s\n", static_cast<int>(backend.size()), backend.data(),
                    static_cast<std::size_t>(triangles), seconds);
    };

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

    // One pool serves every ForkUnion backend; the topology harvest and replicas only where needed.
    bool const needs_pool =
        selected->engine == engine_t::forkunion_k || selected->engine == engine_t::forkunion_replicated_k;
    fu::machine_topology_t topology;
    distributed_pool_t pool;
    replicated_csr_t replicas;
    if (needs_pool) {
        if (!topology.try_harvest()) {
            std::fprintf(stderr, "Failed to harvest the memory topology\n");
            return EXIT_FAILURE;
        }
        if (!pool.try_spawn(topology, threads)) {
            std::fprintf(stderr, "Failed to spawn the thread pool\n");
            return EXIT_FAILURE;
        }
        if (selected->engine == engine_t::forkunion_replicated_k && !replicas.try_build(host, topology)) {
            std::fprintf(stderr, "Failed to replicate the graph across memory domains\n");
            return EXIT_FAILURE;
        }
    }
#if defined(_OPENMP)
    omp_set_num_threads(static_cast<int>(threads));
#endif

    run_context_t context {graph, replicas, pool, topology, {counters.data(), counters.size()}, threads};
    timed([&]() noexcept { selected->run(context); });

    if (check) {
        std::uint64_t serial = 0;
        for (vertex_t v = 0; v < vertices; ++v) serial += count_triangles_at_vertex(graph, v);
        if (serial != triangles) {
            std::fprintf(stderr, "MISMATCH: serial counted %zu\n", static_cast<std::size_t>(serial));
            return EXIT_FAILURE;
        }
        std::printf("check: matches the serial count\n");
    }
    return EXIT_SUCCESS;
}
