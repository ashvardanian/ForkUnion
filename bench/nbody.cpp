/**
 *  @file bench/nbody.cpp
 *  @author Ash Vardanian
 *  @date May 23, 2025
 *  @brief Demo app: N-Body simulation with ForkUnion and OpenMP.
 *
 *  The environment variables it reads are listed in `bench/harness.hpp`. The backends include:
 *  `forkunion_{static,dynamic}_{shared,replicated}`, `openmp_{static,dynamic}`, and
 *  `taskflow_{static,dynamic}`.
 *  To compile and run on all cores in Linux:
 *
 *  @code{.sh}
 *  cmake -B build_release -D CMAKE_BUILD_TYPE=Release
 *  cmake --build build_release --config Release
 *  FORKUNION_NBODY_COUNT=128 FORKUNION_THREADS=$(nproc) build_release/forkunion_nbody
 *  @endcode
 *
 *  Each backend runs a fixed wall-clock window - 10 seconds by default - and reports the dispatch
 *  rate it sustained. Contended-atomic paths amplify any background noise, and short dynamic runs
 *  swing ~±30%, so the window sizes the iteration count to the machine instead of guessing it per
 *  backend. Published comparisons run under `numactl --interleave=all` together with
 *  `OMP_PROC_BIND=spread OMP_PLACES=cores`; this C++ build also adds `-ffast-math`, which Rust
 *  cannot express globally.
 *  To benchmark each backend:
 *
 *  @code{.sh}
 *  FORKUNION_NBODY_COUNT=512 FORKUNION_BACKEND=openmp_static build_release/forkunion_nbody
 *  FORKUNION_NBODY_COUNT=512 FORKUNION_BACKEND=openmp_dynamic build_release/forkunion_nbody
 *  FORKUNION_NBODY_COUNT=512 FORKUNION_BACKEND=forkunion_static_shared build_release/forkunion_nbody
 *  FORKUNION_NBODY_COUNT=512 FORKUNION_BACKEND=forkunion_dynamic_shared build_release/forkunion_nbody
 *  @endcode
 *
 *  On macOS, you may need to install OpenMP support via Homebrew:
 *
 *  @code{.sh}
 *  brew install llvm libomp
 *  cmake -B build_release -D CMAKE_BUILD_TYPE=Release \
 *    -D CMAKE_C_COMPILER=$(brew --prefix llvm)/bin/clang \
 *    -D CMAKE_CXX_COMPILER=$(brew --prefix llvm)/bin/clang++ \
 *    -D CMAKE_CXX_FLAGS="-I$(brew --prefix libomp)/include" \
 *    -D CMAKE_EXE_LINKER_FLAGS="-L$(brew --prefix libomp)/lib"
 *  cmake --build build_release --config Release
 *  FORKUNION_NBODY_COUNT=512 FORKUNION_THREADS=$(sysctl -n hw.logicalcpu) \
 *    FORKUNION_BACKEND=forkunion_static_shared build_release/forkunion_nbody
 *  @endcode
 */
#include <cmath>   // `std::floor`
#include <cstring> // `std::memcpy`

#include <optional> // `std::optional` - the executor, spawned only when chosen

/*  Clang generally defines @c _OPENMP when OpenMP, but compiling it is tricky and the header may
 *  not be available. */
#if defined(_OPENMP)
#if __has_include(<omp.h>)
#include <omp.h>
#else
#undef _OPENMP
#endif
#endif

#include <taskflow/taskflow.hpp>           // `tf::Executor`, `tf::Taskflow`
#include <taskflow/algorithm/for_each.hpp> // `for_each_index`, partitioners

#include <forkunion.hpp>

#include "harness.hpp" // `settings_t`, `loop_t`, `fu`

namespace ashvardanian::forkunion::bench {

#pragma region Shared Logic

static constexpr float g_const = 6.674e-11f;
static constexpr float dt_const = 0.01f;
static constexpr float softening_const = 1e-9f;

struct vector3_t {
    float x, y, z;

    inline vector3_t &operator+=(vector3_t const &other) noexcept {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
};

struct body_t {
    vector3_t position;
    vector3_t velocity;
    float mass;
};

inline float fast_rsqrt(float x) noexcept {
    std::uint32_t i;
    std::memcpy(&i, &x, sizeof(i));
    i = 0x5f3759df - (i >> 1);
    float y;
    std::memcpy(&y, &i, sizeof(y));
    float x2 = x * 0.5f;
    y = y * (1.5f - x2 * y * y);
    return y;
}

inline vector3_t gravitational_force(body_t const &bi, body_t const &bj) noexcept {
    float dx = bj.position.x - bi.position.x;
    float dy = bj.position.y - bi.position.y;
    float dz = bj.position.z - bi.position.z;
    float l2_squared = dx * dx + dy * dy + dz * dz + softening_const;
    float l2_reciprocal = fast_rsqrt(l2_squared);
    float l2_cube_reciprocal = l2_reciprocal * l2_reciprocal * l2_reciprocal;
    float mag = g_const * bi.mass * bj.mass * l2_cube_reciprocal;
    return {mag * dx, mag * dy, mag * dz};
}

/**
 *  @brief How many independent accumulator chains the force sweep keeps.
 *
 *  Reassociating a float reduction is exactly what `-ffast-math` permits and strict IEEE forbids -
 *  so the reassociation is written out by hand instead: the same eight lanes in the C++, Rust, and
 *  Zig kernels, reduced in the same fixed order. Every compiler then faces the same strict-IEEE
 *  optimization problem with the same freedom, and the language columns compare schedulers rather
 *  than compiler flag sets.
 */
constexpr std::size_t force_lanes_k = 8;

/** Net gravitational force on @p bi over @p n @p bodies, in eight explicit lanes. */
inline vector3_t net_force(body_t const &bi, body_t const *bodies, std::size_t n) noexcept {
    float fx[force_lanes_k] = {}, fy[force_lanes_k] = {}, fz[force_lanes_k] = {};
    std::size_t const blocked = n - n % force_lanes_k;
    for (std::size_t j = 0; j < blocked; j += force_lanes_k)
        for (std::size_t lane = 0; lane < force_lanes_k; ++lane) {
            vector3_t const f = gravitational_force(bi, bodies[j + lane]);
            fx[lane] += f.x, fy[lane] += f.y, fz[lane] += f.z;
        }
    for (std::size_t j = blocked; j < n; ++j) {
        vector3_t const f = gravitational_force(bi, bodies[j]);
        fx[j - blocked] += f.x, fy[j - blocked] += f.y, fz[j - blocked] += f.z;
    }
    // The one reduction shape every language shares; changing it changes the bits.
    return {((fx[0] + fx[1]) + (fx[2] + fx[3])) + ((fx[4] + fx[5]) + (fx[6] + fx[7])),
            ((fy[0] + fy[1]) + (fy[2] + fy[3])) + ((fy[4] + fy[5]) + (fy[6] + fy[7])),
            ((fz[0] + fz[1]) + (fz[2] + fz[3])) + ((fz[4] + fz[5]) + (fz[6] + fz[7]))};
}

inline void apply_force(body_t &bi, vector3_t const &f) noexcept {
    bi.velocity.x += f.x / bi.mass * dt_const;
    bi.velocity.y += f.y / bi.mass * dt_const;
    bi.velocity.z += f.z / bi.mass * dt_const;
    bi.position.x += bi.velocity.x * dt_const;
    bi.position.y += bi.velocity.y * dt_const;
    bi.position.z += bi.velocity.z * dt_const;
    // ? Wrap into the unit box to keep every distance - and so every force - inside the normal
    // ? `f32` range forever: no overflows into NaN, and no denormals for x86 to stall on.
    bi.position.x -= std::floor(bi.position.x);
    bi.position.y -= std::floor(bi.position.y);
    bi.position.z -= std::floor(bi.position.z);
}

/** Draw @p counter of stream @p key in `[0, 1)`: the top 24 bits over 2^24, exact in @c f32. */
static inline float random_unit(std::uint64_t const key, std::uint64_t const counter) noexcept {
    return static_cast<float>(fu::split_mix(key + counter) >> 40) * (1.0f / 16777216.0f);
}

#pragma endregion Shared Logic

#pragma region Backends

/**
 *  @brief The one pool type behind every ForkUnion backend.
 *
 *  The distributed pool exists wherever we can harvest a topology and spawn POSIX threads onto it -
 *  Linux and Apple both. Only the @b memory placement is NUMA-specific, and @c replicated_array
 *  already falls back to a heap-backed replica where no NUMA API exists, so a machine with one
 *  memory domain sees the per-node replicas collapse to one.
 */
using distributed_pool_t = fu::distributed_pool<fu::preferred_yield_t, fu::preferred_cache_hints_t>;

/**
 *  @brief Copies canonical @p bodies into every per-domain replica.
 *
 *  Each replica is written by the cores local to its node so the pages first-touch there. Every
 *  compute domain sharing a memory domain cooperates on that node's one replica, partitioned across
 *  all its threads so no element is copied twice.
 */
void refresh_replicas(fu::machine_topology_t const &topology, distributed_pool_t &pool,
                      fu::replicated_array<body_t> &replicas, std::span<body_t const> bodies) noexcept {
    std::size_t const n = bodies.size();
    pool.for_threads([&](fu::thread_in_domain_t at) noexcept {
        std::size_t const compute_domain = static_cast<std::size_t>(at.compute_domain);
        fu::memory_domain_index_t const memory_domain =
            topology.local_memory_of(static_cast<fu::compute_domain_index_t>(compute_domain));

        // Locate this thread among every thread on its memory domain, and count them, so the
        // domain's whole team splits [0, n) without overlap even when several share one node.
        std::size_t threads_on_memory_domain = 0, local_index_on_memory_domain = 0;
        for (std::size_t other = 0; other < pool.compute_domains_count(); ++other) {
            if (topology.local_memory_of(static_cast<fu::compute_domain_index_t>(other)) != memory_domain) continue;
            if (other < compute_domain) local_index_on_memory_domain += pool.threads_count(other);
            threads_on_memory_domain += pool.threads_count(other);
        }
        local_index_on_memory_domain += pool.thread_local_index(at.thread, compute_domain);

        fu::tasks_range_t const range = fu::indexed_split_t {n, threads_on_memory_domain}[local_index_on_memory_domain];
        if (range.count == 0) return; // ? A past-the-end slice when the node has more threads than `n` bodies
        std::span<body_t> const replica = replicas.on_memory_domain(memory_domain);
        std::memcpy(&replica[range.first], &bodies[range.first], range.count * sizeof(body_t));
    });
}

/**
 *  @brief State a backend reads or writes for one simulation step; the harness owns its lifetime.
 *
 *  The Taskflow graphs are built once and re-run every step, the way Taskflow is meant to be used -
 *  never rebuilt on the hot path.
 */
struct nbody_context_t {

    /** Canonical positions, updated in place each step. */
    std::span<body_t> bodies;

    /** Scratch for the accumulated force per body. */
    std::span<vector3_t> forces;

    /** Per-node position replicas, filled only for `replicated_*`. */
    fu::replicated_array<body_t> &replicas;

    /** The compute-to-memory bridge for the replicated read. */
    fu::machine_topology_t const &topology;

    /** Spawned by @c main only for the ForkUnion backends. */
    distributed_pool_t *pool = nullptr;

    /** Spawned by @c main only for the `taskflow_*` backends. */
    tf::Executor *taskflow = nullptr;

    /** Force-accumulation graph, built once and re-run every step. */
    std::optional<tf::Taskflow> force_pass = std::nullopt;

    /** Position-update graph, built once and re-run every step. */
    std::optional<tf::Taskflow> apply_pass = std::nullopt;
};

/** Pre-split across threads vs work-stolen. */
enum class schedule_t : unsigned int { static_k, dynamic_k };

/** One shared body array vs one position replica per memory domain. */
enum class placement_t : unsigned int { shared_k, replicated_k };

/** Runs @p body over `[0, n)`, statically pre-split or work-stolen per compile-time schedule. */
template <schedule_t schedule_, typename body_type_>
static void for_n_scheduled(distributed_pool_t &pool, std::size_t const n, body_type_ body) noexcept {
    if constexpr (schedule_ == schedule_t::static_k) pool.for_n(n, body);
    else pool.for_n_dynamic(n, body);
}

/**
 *  @brief One simulation step, specialized over the schedule and placement axes at compile time.
 *
 *  The all-to-all sweep cannot be sharded - every body reads every other - so the only locality to
 *  win is the read side: replicate the positions once per step, then keep the quadratic loop
 *  node-local. The four ForkUnion backends are the four instantiations of this one body.
 */
template <schedule_t schedule_, placement_t placement_>
static void run(nbody_context_t &c) noexcept {

    std::size_t const n = c.bodies.size();
    std::span<body_t> const bodies = c.bodies;
    std::span<vector3_t> const forces = c.forces;

    if constexpr (placement_ == placement_t::replicated_k) refresh_replicas(c.topology, *c.pool, c.replicas, bodies);

    // Force pass: all-to-all, reading the shared array or the thread's node-local replica.
    auto calc = [&](std::size_t const task, fu::thread_in_domain_t at) noexcept {
        vector3_t f {0.0, 0.0, 0.0};
        if constexpr (placement_ == placement_t::replicated_k) {
            auto const local = c.replicas.on_memory_domain(
                c.topology.local_memory_of(static_cast<fu::compute_domain_index_t>(at.compute_domain)));
            body_t const body_i = local[task];
            f = net_force(body_i, local.data(), n);
        }
        else { f = net_force(bodies[task], bodies.data(), n); }
        forces[task] = f;
    };
    // Apply pass: integrate the canonical body by its force - identical for both placements.
    auto integrate = [&](std::size_t const task, fu::thread_in_domain_t) noexcept {
        apply_force(bodies[task], forces[task]);
    };

    for_n_scheduled<schedule_>(*c.pool, n, calc);
    for_n_scheduled<schedule_>(*c.pool, n, integrate);
}

#if defined(_OPENMP)

/** The OpenMP baselines - the same all-to-all sweep under an `omp parallel for`. */
static void run_openmp_static(nbody_context_t &c) noexcept {
    std::size_t const n = c.bodies.size();
    std::span<body_t> const bodies = c.bodies;
    std::span<vector3_t> const forces = c.forces;
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { forces[i] = net_force(bodies[i], bodies.data(), n); }
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) apply_force(bodies[i], forces[i]);
}
static void run_openmp_dynamic(nbody_context_t &c) noexcept {
    std::size_t const n = c.bodies.size();
    std::span<body_t> const bodies = c.bodies;
    std::span<vector3_t> const forces = c.forces;
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t i = 0; i < n; ++i) { forces[i] = net_force(bodies[i], bodies.data(), n); }
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t i = 0; i < n; ++i) apply_force(bodies[i], forces[i]);
}
#endif

/**
 *  @brief The Taskflow baselines - the same all-to-all sweep under @c tf::for_each_index.
 *
 *  The executor is spawned once by @c main, and the two task graphs are built once on the first
 *  step and re-run thereafter - exactly how Taskflow is meant to be used. Rebuilding a
 *  @c tf::Taskflow every step, or spawning a fresh @c tf::Executor per dispatch, would measure
 *  graph construction, not the dispatch this benchmark isolates. The graphs capture the @c bodies
 *  and @c forces spans, whose pointers never move, so one build stays valid.
 */
template <typename partitioner_>
static void run_taskflow(nbody_context_t &c, partitioner_ partitioner) noexcept {
    tf::Executor &executor = *c.taskflow;
    if (!c.force_pass) {
        std::size_t const n = c.bodies.size();
        std::span<body_t> const bodies = c.bodies;
        std::span<vector3_t> const forces = c.forces;
        c.force_pass.emplace();
        c.force_pass->for_each_index(
            std::size_t(0), n, std::size_t(1),
            [=](std::size_t i) noexcept { forces[i] = net_force(bodies[i], bodies.data(), n); }, partitioner);
        c.apply_pass.emplace();
        c.apply_pass->for_each_index(
            std::size_t(0), n, std::size_t(1), [=](std::size_t i) noexcept { apply_force(bodies[i], forces[i]); },
            partitioner);
    }
    executor.run(*c.force_pass).wait();
    executor.run(*c.apply_pass).wait();
}
static void run_taskflow_static(nbody_context_t &c) noexcept { run_taskflow(c, tf::StaticPartitioner()); }
static void run_taskflow_dynamic(nbody_context_t &c) noexcept { run_taskflow(c, tf::DynamicPartitioner(1)); }

/** Which execution engine a backend runs on, so @c main builds exactly the resource it needs. */
enum class engine_t : unsigned int {

    /** Spawns the shared ForkUnion pool. */
    forkunion_k,

    /** Also allocates the per-node position replicas. */
    forkunion_replicated_k,

    /** Runs under `omp parallel for`, needing no pool object. */
    openmp_k,

    /** Runs on a reused @c tf::Executor, needing no pool object. */
    taskflow_k,
};

/** The dispatch table - a name, its per-step function, and the engine it runs on. */
struct backend_t {
    std::string_view name;
    void (*run)(nbody_context_t &) noexcept;
    engine_t engine;
};

static constexpr backend_t backends_k[] = {
    {"forkunion_static_shared", &run<schedule_t::static_k, placement_t::shared_k>, engine_t::forkunion_k},
    {"forkunion_dynamic_shared", &run<schedule_t::dynamic_k, placement_t::shared_k>, engine_t::forkunion_k},
    {"forkunion_static_replicated", &run<schedule_t::static_k, placement_t::replicated_k>,
     engine_t::forkunion_replicated_k},
    {"forkunion_dynamic_replicated", &run<schedule_t::dynamic_k, placement_t::replicated_k>,
     engine_t::forkunion_replicated_k},
#if defined(_OPENMP)
    {"openmp_static", run_openmp_static, engine_t::openmp_k},
    {"openmp_dynamic", run_openmp_dynamic, engine_t::openmp_k},
#endif
    {"taskflow_static", run_taskflow_static, engine_t::taskflow_k},
    {"taskflow_dynamic", run_taskflow_dynamic, engine_t::taskflow_k},
};

/** The grammar of @c FORKUNION_BACKEND, listing @c backends_k. */
static constexpr char backend_grammar_k[] = "one of forkunion_static_shared, forkunion_dynamic_shared, "  //
                                            "forkunion_static_replicated, forkunion_dynamic_replicated, " //
#if defined(_OPENMP)
                                            "openmp_static, openmp_dynamic, " //
#endif
                                            "taskflow_static, taskflow_dynamic";

/** The backend named @p name, or nothing when @c backends_k has none by that name. */
std::optional<backend_t> parse_backend(std::string_view name) noexcept {
    for (backend_t const &entry : backends_k)
        if (entry.name == name) return entry;
    return std::nullopt;
}

#pragma endregion Backends

/** Simulates @c settings.bodies bodies on @c settings.backend, and prints the timed row. */
int bench_nbody(settings_t const &settings, backend_t const &backend) {
    std::size_t const n = settings.bodies, threads = settings.threads;

    // Prepare bodies and forces - 2 memory allocations
    fu::dynamic_array<body_t> bodies;
    fu::dynamic_array<vector3_t> forces;
    if (failed(bodies.resize(n)) || failed(forces.resize(n))) {
        std::fprintf(stderr, "Failed to allocate %zu bodies\n", n);
        return EXIT_FAILURE;
    }

    // Seven counter-based draws per body: three position coordinates, three velocity components,
    // and one mass in [1e10, 1e15) - so every language starts from bit-identical bodies.
    std::uint64_t const key = fu::split_mix(settings.seed);
    for (std::size_t i = 0; i < n; ++i) {
        std::uint64_t const counter = static_cast<std::uint64_t>(i) * 7;
        bodies[i].position.x = random_unit(key, counter + 0);
        bodies[i].position.y = random_unit(key, counter + 1);
        bodies[i].position.z = random_unit(key, counter + 2);
        bodies[i].velocity.x = random_unit(key, counter + 3);
        bodies[i].velocity.y = random_unit(key, counter + 4);
        bodies[i].velocity.z = random_unit(key, counter + 5);
        bodies[i].mass = 1e10f + random_unit(key, counter + 6) * (1e15f - 1e10f);
    }

    std::span<body_t> const bodies_view {bodies.data(), n};
    std::span<vector3_t> const forces_view {forces.data(), n};

    // One pool serves every ForkUnion backend; the topology harvest and replicas only where needed.
    bool const needs_pool =
        backend.engine == engine_t::forkunion_k || backend.engine == engine_t::forkunion_replicated_k;
    fu::machine_topology_t topology;
    fu::replicated_array<body_t> replicas;
    std::optional<distributed_pool_t> pool; // ? Spawned for the ForkUnion backends
    std::optional<tf::Executor> taskflow;   // ? Spawned for the Taskflow backends
    if (needs_pool) {
        if (failed(topology.harvest())) {
            std::fprintf(stderr, "Failed to harvest the memory topology\n");
            return EXIT_FAILURE;
        }
        pool.emplace();
        if (failed(pool->spawn(topology, threads))) {
            std::fprintf(stderr, "Failed to spawn the thread pool\n");
            return EXIT_FAILURE;
        }
        if (backend.engine == engine_t::forkunion_replicated_k && failed(replicas.resize_uninitialized(topology, n))) {
            std::fprintf(stderr, "Failed to allocate per-domain body replicas\n");
            return EXIT_FAILURE;
        }
    }
    if (backend.engine == engine_t::taskflow_k) taskflow.emplace(threads);
#if defined(_OPENMP)
    omp_set_num_threads(static_cast<int>(threads));
#endif

    nbody_context_t context {bodies_view, forces_view, replicas, topology};
    if (pool) context.pool = &*pool;
    if (taskflow) context.taskflow = &*taskflow;
    // A fixed time budget beats a fixed iteration count: every backend runs the same wall-clock
    // window - long enough to amortize scheduling noise - and reports the rate it sustained, with
    // no per-backend iteration guessing. One call is one step: two dispatches over `n` bodies.
    loop_t loop(settings.warmup, settings.time_limit);
    for ([[maybe_unused]] std::size_t call : loop) backend.run(context);
    loop.rate("bodies", static_cast<double>(n));
    print(loop.row(backend.name));
    return EXIT_SUCCESS;
}

} // namespace ashvardanian::forkunion::bench

using namespace ashvardanian::forkunion::bench;

int main() {
    settings_t const settings = read_settings();
    backend_t const backend = env_parsed("FORKUNION_BACKEND", backends_k[0], parse_backend, backend_grammar_k);
    print(probe_machine());
    print(settings);
    return bench_nbody(settings, backend);
}
