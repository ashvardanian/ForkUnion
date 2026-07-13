/**
 *  @brief Demo app: N-Body simulation with ForkUnion and OpenMP.
 *  @author Ash Vardanian
 *  @file nbody.cpp
 *
 *  To control the script, several environment variables are used:
 *
 *  - `NBODY_COUNT` - number of bodies in the simulation (default: number of threads).
 *  - `NBODY_ITERATIONS` - number of iterations to run the simulation (default: 1000).
 *  - `NBODY_BACKEND` - backend to use for the simulation (default: `forkunion_static_shared`).
 *  - `NBODY_THREADS` - number of threads to use for the simulation (default: number of hardware threads).
 *
 *  The backends include: `openmp_static`, `openmp_dynamic`, `forkunion_static_shared`, `forkunion_dynamic_shared`,
 *  `forkunion_static_replicated`, and `forkunion_dynamic_replicated`.
 *  To compile and run on all cores in Linux:
 *
 *  @code{.sh}
 *  cmake -B build_release -D CMAKE_BUILD_TYPE=Release
 *  cmake --build build_release --config Release
 *  NBODY_COUNT=128 NBODY_THREADS=$(nproc) build_release/forkunion_nbody
 *  @endcode
 *
 *  The default profiling scheme is to 1M iterations for 128 particles on each backend:
 *
 *  @code{.sh}
 *  time NBODY_COUNT=128 NBODY_THREADS=$(nproc) NBODY_ITERATIONS=1000000 \
 *      NBODY_BACKEND=openmp_static build_release/forkunion_nbody
 *  time NBODY_COUNT=128 NBODY_THREADS=$(nproc) NBODY_ITERATIONS=1000000 \
 *      NBODY_BACKEND=openmp_dynamic build_release/forkunion_nbody
 *  time NBODY_COUNT=128 NBODY_THREADS=$(nproc) NBODY_ITERATIONS=1000000 \
 *      NBODY_BACKEND=forkunion_static_shared build_release/forkunion_nbody
 *  time NBODY_COUNT=128 NBODY_THREADS=$(nproc) NBODY_ITERATIONS=1000000 \
 *      NBODY_BACKEND=forkunion_dynamic_shared build_release/forkunion_nbody
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
 *  NBODY_COUNT=128 NBODY_THREADS=$(sysctl -n hw.logicalcpu) NBODY_ITERATIONS=1000000 \
 *    NBODY_BACKEND=forkunion_static_shared build_release/forkunion_nbody
 *  @endcode
 */
#include <chrono>  // `std::chrono::steady_clock`
#include <cstring> // `std::memcpy`
#include <random>  // `std::uniform_real_distribution`, `std::mt19937`

// Clang generally defines `_OPENMP` when OpenMP, but compiling it is
// tricky and the header may not be available.
#if defined(_OPENMP)
#if __has_include(<omp.h>)
#include <omp.h>
#else
#undef _OPENMP
#endif
#endif

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

inline void apply_force(body_t &bi, vector3_t const &f) noexcept {
    bi.velocity.x += f.x / bi.mass * dt_const;
    bi.velocity.y += f.y / bi.mass * dt_const;
    bi.velocity.z += f.z / bi.mass * dt_const;
    bi.position.x += bi.velocity.x * dt_const;
    bi.position.y += bi.velocity.y * dt_const;
    bi.position.z += bi.velocity.z * dt_const;
}

#pragma endregion Shared Logic

#pragma region Backends

/*  The distributed pool exists wherever we can harvest a topology and spawn POSIX threads onto it -
 *  Linux and Apple both. Only the @b memory placement is NUMA-specific, and `replicated_array` already
 *  falls back to a heap-backed replica where no NUMA API exists, so a machine with one memory domain sees
 *  the per-node replicas collapse to one. One pool serves every ForkUnion backend.  */
using distributed_pool_t = fu::distributed_pool<fu::preferred_yield_t>;

/*  Copies canonical `bodies` into every per-domain replica, each written by the cores local to its node so
 *  the pages first-touch there. Every compute domain sharing a memory domain cooperates on that node's one
 *  replica, partitioned across all its threads so no element is copied twice.  */
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
    distributed_pool_t &pool;               // ? One pool spawned for every ForkUnion backend
    fu::replicated_array<body_t> &replicas; // ? Per-node position replicas, filled only for `replicated_*`
    fu::machine_topology_t const &topology; // ? The compute-to-memory bridge for the replicated read
};

/** @brief Pre-split across threads vs work-stolen. */
enum class schedule_k : unsigned int { static_k, dynamic_k };
/** @brief One shared body array vs one position replica per memory domain. */
enum class placement_k : unsigned int { shared_k, replicated_k };

/** @brief Runs @p body over `[0, n)`, statically pre-split or work-stolen per the compile-time schedule. */
template <schedule_k schedule_, class body_type_>
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

    if constexpr (placement_ == placement_k::replicated_k) refresh_replicas(c.topology, c.pool, c.replicas, bodies);

    // Force pass: all-to-all, reading the shared array or the thread's node-local replica.
    auto calc = [&](local_prong_t prong) noexcept {
        vector3_t f {0.0, 0.0, 0.0};
        if constexpr (placement_ == placement_k::replicated_k) {
            auto const local = c.replicas.on_memory_domain(
                c.topology.local_memory_of(static_cast<fu::compute_domain_index_t>(prong.compute_domain)));
            body_t const body_i = local[prong.task];
            for (std::size_t j = 0; j < n; ++j) f += gravitational_force(body_i, local[j]);
        }
        else {
            for (std::size_t j = 0; j < n; ++j) f += gravitational_force(bodies[prong.task], bodies[j]);
        }
        forces[prong.task] = f;
    };
    // Apply pass: integrate the canonical body by its force - identical for both placements.
    auto integrate = [&](local_prong_t prong) noexcept { apply_force(bodies[prong.task], forces[prong.task]); };

    for_n_scheduled<schedule_>(c.pool, n, calc);
    for_n_scheduled<schedule_>(c.pool, n, integrate);
}

#if defined(_OPENMP)
/** @brief The OpenMP baselines - the same all-to-all sweep under an `omp parallel for`. */
static void run_openmp_static(nbody_context_t &c) noexcept {
    std::size_t const n = c.bodies.size();
    fu::span<body_t> const bodies = c.bodies;
    fu::span<vector3_t> const forces = c.forces;
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        vector3_t f {0.0, 0.0, 0.0};
        for (std::size_t j = 0; j < n; ++j) f += gravitational_force(bodies[i], bodies[j]);
        forces[i] = f;
    }
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) apply_force(bodies[i], forces[i]);
}
static void run_openmp_dynamic(nbody_context_t &c) noexcept {
    std::size_t const n = c.bodies.size();
    fu::span<body_t> const bodies = c.bodies;
    fu::span<vector3_t> const forces = c.forces;
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t i = 0; i < n; ++i) {
        vector3_t f {0.0, 0.0, 0.0};
        for (std::size_t j = 0; j < n; ++j) f += gravitational_force(bodies[i], bodies[j]);
        forces[i] = f;
    }
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t i = 0; i < n; ++i) apply_force(bodies[i], forces[i]);
}
#endif

/** @brief Which execution engine a backend runs on, so `main` builds exactly the resource it needs. */
enum class engine_t : unsigned int {
    forkunion_k,            // ? Spawns the shared ForkUnion pool
    forkunion_replicated_k, // ? Also allocates the per-node position replicas
    openmp_k,               // ? Runs under `omp parallel for`, needing no pool object
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

int main() {
    std::size_t n = env_usize("NBODY_COUNT", 0);
    std::size_t const iterations = env_usize("NBODY_ITERATIONS", 1000);
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

    // A fixed seed - the benchmark only needs a spread of positions, not entropy, and every backend must
    // start from the same bodies to be comparable.
    std::uniform_real_distribution<float> coordinate_distribution(0.f, 1.f);
    std::uniform_real_distribution<float> mass_distribution(1e20f, 1e25f);
    std::mt19937 random_gen(0x1234'5678u);
    for (std::size_t i = 0; i < n; ++i) {
        bodies[i].position.x = coordinate_distribution(random_gen);
        bodies[i].position.y = coordinate_distribution(random_gen);
        bodies[i].position.z = coordinate_distribution(random_gen);
        bodies[i].velocity.x = coordinate_distribution(random_gen);
        bodies[i].velocity.y = coordinate_distribution(random_gen);
        bodies[i].velocity.z = coordinate_distribution(random_gen);
        bodies[i].mass = mass_distribution(random_gen);
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
    distributed_pool_t pool;
    fu::replicated_array<body_t> replicas;
    if (needs_pool) {
        if (!topology.try_harvest()) {
            std::fprintf(stderr, "Failed to harvest the memory topology\n");
            return EXIT_FAILURE;
        }
        if (!pool.try_spawn(topology, threads)) {
            std::fprintf(stderr, "Failed to spawn the thread pool\n");
            return EXIT_FAILURE;
        }
        if (selected->engine == engine_t::forkunion_replicated_k && !replicas.try_resize_uninitialized(topology, n)) {
            std::fprintf(stderr, "Failed to allocate per-domain body replicas\n");
            return EXIT_FAILURE;
        }
    }
#if defined(_OPENMP)
    omp_set_num_threads(static_cast<int>(threads));
#endif

    nbody_context_t context {bodies_view, forces_view, pool, replicas, topology};
    auto const started = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) selected->run(context);
    double const seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() //
                           / static_cast<double>(iterations);
    std::printf("%.*s: %zu bodies, %zu iters in %.3f s\n", static_cast<int>(backend.size()), backend.data(), n,
                iterations, seconds);
    return EXIT_SUCCESS;
}
