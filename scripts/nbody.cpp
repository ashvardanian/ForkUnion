/**
 *  @brief Demo app: N-Body simulation with ForkUnion and OpenMP.
 *  @author Ash Vardanian
 *  @file nbody.cpp
 *
 *  To control the script, several environment variables are used:
 *
 *  - `NBODY_COUNT` - number of bodies in the simulation (default: number of threads).
 *  - `NBODY_SECONDS` - wall-clock budget per run, reporting the sustained rate - default 10.
 *  - `NBODY_ITERATIONS` - run an exact iteration count instead, when set.
 *  - `NBODY_BACKEND` - backend to use for the simulation (default: `forkunion_static_shared`).
 *  - `NBODY_THREADS` - number of threads to use for the simulation (default: number of hardware threads).
 *
 *  The backends include: `forkunion_{static,dynamic}_{shared,replicated}`, `openmp_{static,dynamic}`,
 *  and `taskflow_{static,dynamic}`.
 *  To compile and run on all cores in Linux:
 *
 *  @code{.sh}
 *  cmake -B build_release -D CMAKE_BUILD_TYPE=Release
 *  cmake --build build_release --config Release
 *  NBODY_COUNT=128 NBODY_THREADS=$(nproc) build_release/forkunion_nbody
 *  @endcode
 *
 *  Each backend runs a fixed wall-clock window - 10 seconds by default, enough to amortize
 *  scheduling noise - and reports the dispatch rate it sustained:
 *
 *  @code{.sh}
 *  NBODY_COUNT=512 NBODY_BACKEND=openmp_static build_release/forkunion_nbody
 *  NBODY_COUNT=512 NBODY_BACKEND=openmp_dynamic build_release/forkunion_nbody
 *  NBODY_COUNT=512 NBODY_BACKEND=forkunion_static_shared build_release/forkunion_nbody
 *  NBODY_COUNT=512 NBODY_BACKEND=forkunion_dynamic_shared build_release/forkunion_nbody
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
 *  NBODY_COUNT=512 NBODY_THREADS=$(sysctl -n hw.logicalcpu) \
 *    NBODY_BACKEND=forkunion_static_shared build_release/forkunion_nbody
 *  @endcode
 */
#include <cmath>   // `std::floor`
#include <cstring> // `std::memcpy`

#include <chrono>   // `std::chrono::steady_clock`
#include <optional> // `std::optional` - the executor, spawned only when chosen

// Clang generally defines `_OPENMP` when OpenMP, but compiling it is
// tricky and the header may not be available.
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

namespace fu = ashvardanian::forkunion;

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

/** @brief Net gravitational force on @p bi over @p n @p bodies, in eight explicit lanes. */
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

/**
 *  @brief The SplitMix64 avalanche behind every random draw - a pure function of the @p counter.
 *
 *  Deliberately not `std::mt19937` with `std::uniform_real_distribution`: the standard generators
 *  differ across languages - and the C++ distributions even across standard libraries - so no two
 *  harnesses would simulate the same system. Each draw is a pure function of its counter instead,
 *  and the bodies are bit-identical across the C++, Rust, and Zig ports of this hash.
 */
static inline std::uint64_t split_mix(std::uint64_t const counter) noexcept {
    std::uint64_t x = (counter + 1) * 0x9E37'79B9'7F4A'7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58'476D'1CE4'E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D0'49BB'1331'11EBull;
    return x ^ (x >> 31);
}

/** @brief One draw in `[0, 1)`: the top 24 bits scaled by 2^-24 - both steps exact in `f32`. */
static inline float random_unit(std::uint64_t const counter) noexcept {
    return static_cast<float>(split_mix(counter) >> 40) * (1.0f / 16777216.0f);
}

#pragma endregion Shared Logic

#pragma region Backends

/**
 *  @brief The one pool type behind every ForkUnion backend.
 *
 *  The distributed pool exists wherever we can harvest a topology and spawn POSIX threads onto it -
 *  Linux and Apple both. Only the @b memory placement is NUMA-specific, and `replicated_array`
 *  already falls back to a heap-backed replica where no NUMA API exists, so a machine with one
 *  memory domain sees the per-node replicas collapse to one.
 */
using distributed_pool_t = fu::distributed_pool<fu::preferred_yield_t, fu::preferred_cache_hints_t>;

/**
 *  @brief Copies canonical @p bodies into every per-domain replica.
 *
 *  Each replica is written by the cores local to its node so the pages first-touch there. Every
 *  compute domain sharing a memory domain cooperates on that node's one replica, partitioned
 *  across all its threads so no element is copied twice.
 */
void refresh_replicas(fu::machine_topology_t const &topology, distributed_pool_t &pool,
                      fu::replicated_array<body_t> &replicas, fu::span<body_t const> bodies) noexcept {
    std::size_t const n = bodies.size();
    pool.for_threads([&](fu::local_thread_t thread_index) noexcept {
        std::size_t const compute_domain = static_cast<std::size_t>(thread_index.compute_domain);
        fu::memory_domain_index_t const memory_domain =
            topology.local_memory_of(static_cast<fu::compute_domain_index_t>(compute_domain));

        // Locate this thread among every thread on its memory domain, and count them, so the node's whole
        // team splits [0, n) without overlap even when several compute domains share the node.
        std::size_t threads_on_memory_domain = 0, local_index_on_memory_domain = 0;
        for (std::size_t other = 0; other < pool.compute_domains_count(); ++other) {
            if (topology.local_memory_of(static_cast<fu::compute_domain_index_t>(other)) != memory_domain) continue;
            if (other < compute_domain) local_index_on_memory_domain += pool.threads_count(other);
            threads_on_memory_domain += pool.threads_count(other);
        }
        local_index_on_memory_domain += pool.thread_local_index(thread_index, compute_domain);

        fu::indexed_range_t const range =
            fu::indexed_split_t {n, threads_on_memory_domain}[local_index_on_memory_domain];
        if (range.count == 0) return; // ? A past-the-end slice when the node has more threads than `n` bodies
        fu::span<body_t> const replica = replicas.on_memory_domain(memory_domain);
        std::memcpy(&replica[range.first], &bodies[range.first], range.count * sizeof(body_t));
    });
}

/** @brief Everything a backend reads or writes for one simulation step; the harness owns the lifetimes. */
struct nbody_context_t {
    fu::span<body_t> bodies;                // ? Canonical positions, updated in place each step
    fu::span<vector3_t> forces;             // ? Scratch for the accumulated force per body
    fu::replicated_array<body_t> &replicas; // ? Per-node position replicas, filled only for `replicated_*`
    fu::machine_topology_t const &topology; // ? The compute-to-memory bridge for the replicated read
    distributed_pool_t *pool = nullptr;     // ? Spawned by `main` only for the ForkUnion backends
    tf::Executor *taskflow = nullptr;       // ? Spawned by `main` only for the `taskflow_*` backends
    std::optional<tf::Taskflow> force_pass; // ? Taskflow graphs are built once and re-run every step, the way
    std::optional<tf::Taskflow> apply_pass; // ? Taskflow is meant to be used - never rebuilt on the hot path
};

/** @brief Pre-split across threads vs work-stolen. */
enum class schedule_k : unsigned int { static_k, dynamic_k };
/** @brief One shared body array vs one position replica per memory domain. */
enum class placement_k : unsigned int { shared_k, replicated_k };

/** @brief Runs @p body over `[0, n)`, statically pre-split or work-stolen per the compile-time schedule. */
template <schedule_k schedule_, typename body_type_>
static void for_n_scheduled(distributed_pool_t &pool, std::size_t const n, body_type_ body) noexcept {
    if constexpr (schedule_ == schedule_k::static_k) pool.for_n(n, body);
    else
        pool.for_n_dynamic(n, body);
}

/**
 *  @brief One simulation step, specialized over the schedule and placement axes at compile time.
 *
 *  The all-to-all sweep cannot be sharded - every body reads every other - so the only locality to win
 *  is the read side: replicate the positions once per step, then keep the quadratic loop node-local.
 *  The four ForkUnion backends are the four instantiations of this one body.
 */
template <schedule_k schedule_, placement_k placement_>
static void run(nbody_context_t &c) noexcept {
    using local_prong_t = typename distributed_pool_t::prong_t;
    std::size_t const n = c.bodies.size();
    fu::span<body_t> const bodies = c.bodies;
    fu::span<vector3_t> const forces = c.forces;

    if constexpr (placement_ == placement_k::replicated_k) refresh_replicas(c.topology, *c.pool, c.replicas, bodies);

    // Force pass: all-to-all, reading the shared array or the thread's node-local replica.
    auto calc = [&](local_prong_t prong) noexcept {
        vector3_t f {0.0, 0.0, 0.0};
        if constexpr (placement_ == placement_k::replicated_k) {
            auto const local = c.replicas.on_memory_domain(
                c.topology.local_memory_of(static_cast<fu::compute_domain_index_t>(prong.compute_domain)));
            body_t const body_i = local[prong.task];
            f = net_force(body_i, local.data(), n);
        }
        else { f = net_force(bodies[prong.task], bodies.data(), n); }
        forces[prong.task] = f;
    };
    // Apply pass: integrate the canonical body by its force - identical for both placements.
    auto integrate = [&](local_prong_t prong) noexcept { apply_force(bodies[prong.task], forces[prong.task]); };

    for_n_scheduled<schedule_>(*c.pool, n, calc);
    for_n_scheduled<schedule_>(*c.pool, n, integrate);
}

#if defined(_OPENMP)
/** @brief The OpenMP baselines - the same all-to-all sweep under an `omp parallel for`. */
static void run_openmp_static(nbody_context_t &c) noexcept {
    std::size_t const n = c.bodies.size();
    fu::span<body_t> const bodies = c.bodies;
    fu::span<vector3_t> const forces = c.forces;
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { forces[i] = net_force(bodies[i], bodies.data(), n); }
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) apply_force(bodies[i], forces[i]);
}
static void run_openmp_dynamic(nbody_context_t &c) noexcept {
    std::size_t const n = c.bodies.size();
    fu::span<body_t> const bodies = c.bodies;
    fu::span<vector3_t> const forces = c.forces;
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t i = 0; i < n; ++i) { forces[i] = net_force(bodies[i], bodies.data(), n); }
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t i = 0; i < n; ++i) apply_force(bodies[i], forces[i]);
}
#endif

/**
 *  @brief The Taskflow baselines - the same all-to-all sweep under `tf::for_each_index`.
 *
 *  The executor is spawned once by `main`, and the two task graphs are built once on the first step
 *  and re-run thereafter - exactly how Taskflow is meant to be used. Rebuilding a `tf::Taskflow`
 *  every step, or spawning a fresh `tf::Executor` per dispatch, would measure graph construction,
 *  not the dispatch this benchmark isolates. The graphs capture the `bodies`/`forces` spans, whose
 *  pointers never move, so one build stays valid.
 */
template <typename partitioner_>
static void run_taskflow(nbody_context_t &c, partitioner_ partitioner) noexcept {
    tf::Executor &executor = *c.taskflow;
    if (!c.force_pass) {
        std::size_t const n = c.bodies.size();
        fu::span<body_t> const bodies = c.bodies;
        fu::span<vector3_t> const forces = c.forces;
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

/** @brief Which execution engine a backend runs on, so `main` builds exactly the resource it needs. */
enum class engine_t : unsigned int {
    forkunion_k,            // ? Spawns the shared ForkUnion pool
    forkunion_replicated_k, // ? Also allocates the per-node position replicas
    openmp_k,               // ? Runs under `omp parallel for`, needing no pool object
    taskflow_k,             // ? Runs on a reused `tf::Executor`, needing no pool object
};

/** @brief The dispatch table - a name, its per-step function, and the engine it runs on. */
struct backend_t {
    std::string_view name;
    void (*run)(nbody_context_t &) noexcept;
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
    static thread_local char buffer[256];
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
    std::size_t n = env_usize("NBODY_COUNT", 0);
    double const budget_seconds = env_double("NBODY_SECONDS", 10);   // ? The primary knob: a fixed window
    std::size_t const iterations = env_usize("NBODY_ITERATIONS", 0); // ? Overrides with an exact count when set
    std::string_view const backend = env_string("NBODY_BACKEND", "forkunion_static_shared");
    std::size_t threads = env_usize("NBODY_THREADS", 0);
    if (threads == 0) threads = fu::allowed_cores_count();
    if (n == 0) n = threads;

    // Prepare bodies and forces - 2 memory allocations
    fu::dynamic_array<body_t> bodies;
    fu::dynamic_array<vector3_t> forces;
    if (!bodies.try_resize(n) || !forces.try_resize(n)) {
        std::fprintf(stderr, "Failed to allocate %zu bodies\n", n);
        return EXIT_FAILURE;
    }

    // Seven counter-based draws per body: three position coordinates, three velocity components, and
    // one mass in [1e10, 1e15) - so every language starts from bit-identical bodies.
    for (std::size_t i = 0; i < n; ++i) {
        std::uint64_t const counter = static_cast<std::uint64_t>(i) * 7;
        bodies[i].position.x = random_unit(counter + 0);
        bodies[i].position.y = random_unit(counter + 1);
        bodies[i].position.z = random_unit(counter + 2);
        bodies[i].velocity.x = random_unit(counter + 3);
        bodies[i].velocity.y = random_unit(counter + 4);
        bodies[i].velocity.z = random_unit(counter + 5);
        bodies[i].mass = 1e10f + random_unit(counter + 6) * (1e15f - 1e10f);
    }

    fu::span<body_t> const bodies_view {bodies.data(), n};
    fu::span<vector3_t> const forces_view {forces.data(), n};

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
    fu::replicated_array<body_t> replicas;
    std::optional<distributed_pool_t> pool; // ? Spawned for the ForkUnion backends
    std::optional<tf::Executor> taskflow;   // ? Spawned for the Taskflow backends
    if (needs_pool) {
        if (!topology.try_harvest()) {
            std::fprintf(stderr, "Failed to harvest the memory topology\n");
            return EXIT_FAILURE;
        }
        pool.emplace();
        if (!pool->try_spawn(topology, threads)) {
            std::fprintf(stderr, "Failed to spawn the thread pool\n");
            return EXIT_FAILURE;
        }
        if (selected->engine == engine_t::forkunion_replicated_k && !replicas.try_resize_uninitialized(topology, n)) {
            std::fprintf(stderr, "Failed to allocate per-domain body replicas\n");
            return EXIT_FAILURE;
        }
    }
    if (selected->engine == engine_t::taskflow_k) taskflow.emplace(threads);
#if defined(_OPENMP)
    omp_set_num_threads(static_cast<int>(threads));
#endif

    nbody_context_t context {bodies_view, forces_view, replicas, topology};
    if (pool) context.pool = &*pool;
    if (taskflow) context.taskflow = &*taskflow;
    // A fixed time budget beats a fixed iteration count: every backend runs the same wall-clock
    // window - long enough to amortize scheduling noise - and reports the rate it sustained, with
    // no per-backend iteration guessing. `NBODY_ITERATIONS` forces an exact count instead.
    auto const started = std::chrono::steady_clock::now();
    std::size_t passes = 0;
    if (iterations > 0)
        for (; passes < iterations; ++passes) selected->run(context);
    else
        do {
            selected->run(context), ++passes;
        } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() < budget_seconds);
    double const total_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    double const microseconds_per_iteration = total_seconds / static_cast<double>(passes) * 1e6;
    // Per-iteration latency is the comparable unit - one `for_each` dispatch over `n` bodies. The total
    // wall time follows for reference, since a fast dispatch rounds to zero when printed in seconds.
    std::printf("%.*s: %zu bodies, %zu iters, %.2f us/iter (%.2f s total)\n", static_cast<int>(backend.size()),
                backend.data(), n, passes, microseconds_per_iteration, total_seconds);
    return EXIT_SUCCESS;
}
