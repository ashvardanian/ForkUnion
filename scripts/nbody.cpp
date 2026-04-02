/**
 *  @brief Demo app: N-Body simulation with ForkUnion and OpenMP.
 *  @author Ash Vardanian
 *  @file nbody.cpp
 *
 *  To control the script, several environment variables are used:
 *
 *  - `NBODY_COUNT` - number of bodies in the simulation (default: number of threads).
 *  - `NBODY_ITERATIONS` - number of iterations to run the simulation (default: 1000).
 *  - `NBODY_BACKEND` - backend to use for the simulation (default: `forkunion_static`).
 *  - `NBODY_THREADS` - number of threads to use for the simulation (default: number of hardware threads).
 *
 *  The backends include: `forkunion_static`, `forkunion_dynamic`, `openmp_static`, and `openmp_dynamic`.
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
 *      NBODY_BACKEND=forkunion_static build_release/forkunion_nbody
 *  time NBODY_COUNT=128 NBODY_THREADS=$(nproc) NBODY_ITERATIONS=1000000 \
 *      NBODY_BACKEND=forkunion_dynamic build_release/forkunion_nbody
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
 *    NBODY_BACKEND=forkunion_static build_release/forkunion_nbody
 *  @endcode
 */
#include <vector> // `std::vector`
#include <random> // `std::random_device`, `std::uniform_real_distribution`
#include <thread> // `std::thread::hardware_concurrency`
#include <span>   // `std::span`
#include <bit>    // `std::bit_cast`
#include <limits> // `std::numeric_limits`

// Clang generally defines `_OPENMP` when OpenMP, but compiling it is
/// tricky and the header may not be available.
#if defined(_OPENMP)
#if __has_include(<omp.h>)
#include <omp.h>
#else
#undef _OPENMP
#endif
#endif

#include <forkunion.hpp>

namespace fu = ashvardanian::forkunion;

#if defined(__GNUC__) || defined(__clang__)
#define _FU_RESTRICT __restrict__
#else
#define _FU_RESTRICT
#endif

#pragma region - Shared Logic

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
    std::uint32_t i = std::bit_cast<std::uint32_t>(x);
    i = 0x5f3759df - (i >> 1);
    float y = std::bit_cast<float>(i);
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

#pragma endregion - Shared Logic

#pragma region - Backends

void iteration_openmp_static(FU_MAYBE_UNUSED_ body_t *_FU_RESTRICT bodies,
                             FU_MAYBE_UNUSED_ vector3_t *_FU_RESTRICT forces, FU_MAYBE_UNUSED_ std::size_t n) noexcept {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        vector3_t f {0.0, 0.0, 0.0};
        for (std::size_t j = 0; j < n; ++j) f += gravitational_force(bodies[i], bodies[j]);
        forces[i] = f;
    }
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) apply_force(bodies[i], forces[i]);
#endif
}

void iteration_openmp_dynamic(FU_MAYBE_UNUSED_ body_t *_FU_RESTRICT bodies,
                              FU_MAYBE_UNUSED_ vector3_t *_FU_RESTRICT forces,
                              FU_MAYBE_UNUSED_ std::size_t n) noexcept {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t i = 0; i < n; ++i) {
        vector3_t f {0.0, 0.0, 0.0};
        for (std::size_t j = 0; j < n; ++j) f += gravitational_force(bodies[i], bodies[j]);
        forces[i] = f;
    }
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t i = 0; i < n; ++i) apply_force(bodies[i], forces[i]);
#endif
}

using pool_t = fu::basic_pool<std::allocator<std::thread>, fu::standard_yield_t>;

void iteration_forkunion_static(pool_t &pool, body_t *_FU_RESTRICT bodies, vector3_t *_FU_RESTRICT forces,
                                std::size_t n) noexcept {
    pool.for_n(n, [=](std::size_t i) noexcept {
        vector3_t f {0.0, 0.0, 0.0};
        for (std::size_t j = 0; j < n; ++j) f += gravitational_force(bodies[i], bodies[j]);
        forces[i] = f;
    });
    pool.for_n(n, [=](std::size_t i) noexcept { apply_force(bodies[i], forces[i]); });
}

void iteration_forkunion_dynamic(pool_t &pool, body_t *_FU_RESTRICT bodies, vector3_t *_FU_RESTRICT forces,
                                 std::size_t n) noexcept {
    pool.for_n_dynamic(n, [=](std::size_t i) noexcept {
        vector3_t f {0.0, 0.0, 0.0};
        for (std::size_t j = 0; j < n; ++j) f += gravitational_force(bodies[i], bodies[j]);
        forces[i] = f;
    });
    pool.for_n_dynamic(n, [=](std::size_t i) noexcept { apply_force(bodies[i], forces[i]); });
}

#if FU_ENABLE_NUMA
using linux_numa_bodies_allocator_t = fu::linux_numa_allocator<body_t>;
using linux_numa_bodies_t = std::vector<body_t, linux_numa_bodies_allocator_t>;
using linux_distributed_pool_t = fu::linux_distributed_pool<fu::standard_yield_t>;

std::vector<linux_numa_bodies_t> make_buffers_for_forkunion_numa(linux_distributed_pool_t &pool,
                                                                 std::size_t n) noexcept {
    fu::numa_topology_t const &topology = pool.topology();
    std::size_t const numa_nodes_count = topology.nodes_count();

    std::vector<linux_numa_bodies_t> result;
    for (std::size_t i = 0; i < numa_nodes_count; ++i) {
        fu::numa_node_id_t const node_id = topology.node(i).node_id;
        linux_numa_bodies_allocator_t allocator(node_id);
        linux_numa_bodies_t bodies(n, allocator);
        result.emplace_back(std::move(bodies));
    }

    return result;
}

void iteration_forkunion_numa_static(linux_distributed_pool_t &pool, body_t *_FU_RESTRICT bodies,
                                     vector3_t *_FU_RESTRICT forces, std::size_t n,
                                     body_t **_FU_RESTRICT bodies_numa_copies) noexcept {

    using colocated_prong_t = typename linux_distributed_pool_t::prong_t;

    // This is a quadratic complexity all-to-all interaction, and it's not clear how
    // it can "shard" to take advantage of NUMA locality, especially for a small `n` world.
    // Still, at least we can replicate the body positions onto every node just once per iteration,
    // to reduce the number of remote accesses, even if they are cached.
    pool.for_threads([&](auto thread_index) noexcept {
        std::size_t const numa_node_index = pool.thread_colocation(thread_index);
        std::size_t const threads_next_to_numa_node = pool.threads_count(numa_node_index);
        std::size_t const thread_local_index = pool.thread_local_index(thread_index, numa_node_index);

        fu::indexed_split_t const n_split {n, threads_next_to_numa_node};
        fu::indexed_range_t const n_subrange = n_split[thread_local_index];
        std::memcpy(                                                //
            &bodies_numa_copies[numa_node_index][n_subrange.first], //
            &bodies[n_subrange.first],                              //
            n_subrange.count * sizeof(body_t));
    });

    pool.for_n(n, [&](colocated_prong_t prong) noexcept {
        std::size_t const numa_node_index = prong.colocation;
        body_t const *numa_bodies = bodies_numa_copies[numa_node_index];
        body_t const body_i = numa_bodies[prong.task];

        vector3_t f {0.0, 0.0, 0.0};
        for (std::size_t j = 0; j < n; ++j) f += gravitational_force(body_i, numa_bodies[j]);
        forces[prong.task] = f;
    });
    pool.for_n(n, [=](std::size_t i) noexcept { apply_force(bodies[i], forces[i]); });
}

void iteration_forkunion_numa_dynamic(linux_distributed_pool_t &pool, body_t *_FU_RESTRICT bodies,
                                      vector3_t *_FU_RESTRICT forces, std::size_t n,
                                      body_t **_FU_RESTRICT bodies_numa_copies) noexcept {

    using colocated_prong_t = typename linux_distributed_pool_t::prong_t;

    // This expressions is same as in `iteration_forkunion_numa_static` static version:
    pool.for_threads([&](auto thread_index) noexcept {
        std::size_t const numa_node_index = pool.thread_colocation(thread_index);
        std::size_t const threads_next_to_numa_node = pool.threads_count(numa_node_index);
        std::size_t const thread_local_index = pool.thread_local_index(thread_index, numa_node_index);

        fu::indexed_split_t const n_split {n, threads_next_to_numa_node};
        fu::indexed_range_t const n_subrange = n_split[thread_local_index];
        std::memcpy(                                                //
            &bodies_numa_copies[numa_node_index][n_subrange.first], //
            &bodies[n_subrange.first],                              //
            n_subrange.count * sizeof(body_t));
    });

    // The rest only differs in the `for_n_dynamic` usage over `for_n`:
    pool.for_n_dynamic(n, [&](colocated_prong_t prong) noexcept {
        std::size_t const numa_node_index = prong.colocation;
        body_t const *numa_bodies = bodies_numa_copies[numa_node_index];
        body_t const body_i = numa_bodies[prong.task];

        vector3_t f {0.0, 0.0, 0.0};
        for (std::size_t j = 0; j < n; ++j) f += gravitational_force(body_i, numa_bodies[j]);
        forces[prong.task] = f;
    });
    pool.for_n_dynamic(n, [=](std::size_t i) noexcept { apply_force(bodies[i], forces[i]); });
}

#endif // FU_ENABLE_NUMA

#pragma endregion - Backends

int main(void) {
    std::printf("Welcome to the ForkUnion N-Body simulation!\n");

    // Helper function to safely get environment variables
    auto safe_getenv = [](char const *name) -> char const * {
#if defined(_MSC_VER)
        static thread_local char buffer[256];
        size_t required_size;
        errno_t err = getenv_s(&required_size, buffer, sizeof(buffer), name);
        return (err == 0 && required_size > 0) ? buffer : nullptr;
#else
        return std::getenv(name); // ! Windows doesn't like this
#endif
    };

    // Read env vars
    auto const n_str = safe_getenv("NBODY_COUNT");
    auto const iterations_str = safe_getenv("NBODY_ITERATIONS");
    auto const backend_str = safe_getenv("NBODY_BACKEND");
    auto const threads_str = safe_getenv("NBODY_THREADS");

    // Parse env vars and validate
    auto n_ull = std::strtoull(n_str ? n_str : "0", nullptr, 10);
    auto iterations_ull = std::strtoull(iterations_str ? iterations_str : "1000", nullptr, 10);
    auto threads_ull = std::strtoull(threads_str ? threads_str : "0", nullptr, 10);
    auto const size_max = std::numeric_limits<std::size_t>::max();

    std::size_t n = (n_ull > size_max) ? size_max : static_cast<std::size_t>(n_ull);
    std::size_t const iterations = (iterations_ull > size_max) ? size_max : static_cast<std::size_t>(iterations_ull);
    std::size_t threads = (threads_ull > size_max) ? size_max : static_cast<std::size_t>(threads_ull);
    std::string_view const backend = backend_str ? backend_str : "forkunion_static";
    if (threads == 0) threads = std::thread::hardware_concurrency();
    if (n == 0) n = threads;

    // Prepare bodies and forces - 2 memory allocations
    std::vector<body_t> bodies(n);
    std::vector<vector3_t> forces(n);

    // Random generators are quite slow, but let's hope this doesn't take too long
    std::uniform_real_distribution<float> coordinate_distribution(0.f, 1.f);
    std::uniform_real_distribution<float> mass_distribution(1e20f, 1e25f);
    std::random_device random_device;
    std::mt19937 random_gen(random_device());
    for (std::size_t i = 0; i < n; ++i) {
        bodies[i].position.x = coordinate_distribution(random_gen);
        bodies[i].position.y = coordinate_distribution(random_gen);
        bodies[i].position.z = coordinate_distribution(random_gen);
        bodies[i].velocity.x = coordinate_distribution(random_gen);
        bodies[i].velocity.y = coordinate_distribution(random_gen);
        bodies[i].velocity.z = coordinate_distribution(random_gen);
        bodies[i].mass = mass_distribution(random_gen);
    }

#if defined(_OPENMP)
    omp_set_num_threads(static_cast<int>(threads));
    if (backend == "openmp_static") {
        for (std::size_t i = 0; i < iterations; ++i) iteration_openmp_static(bodies.data(), forces.data(), n);
        return EXIT_SUCCESS;
    }
    if (backend == "openmp_dynamic") {
        for (std::size_t i = 0; i < iterations; ++i) iteration_openmp_dynamic(bodies.data(), forces.data(), n);
        return EXIT_SUCCESS;
    }
#endif

    // Every other configuration uses ForkUnion
    pool_t pool;
    if (backend == "forkunion_static") {
        if (!pool.try_spawn(threads)) {
            std::fprintf(stderr, "Failed to spawn thread pool\n");
            return EXIT_FAILURE;
        }
        for (std::size_t i = 0; i < iterations; ++i) //
            iteration_forkunion_static(pool, bodies.data(), forces.data(), n);
        return EXIT_SUCCESS;
    }
    if (backend == "forkunion_dynamic") {
        if (!pool.try_spawn(threads)) {
            std::fprintf(stderr, "Failed to spawn thread pool\n");
            return EXIT_FAILURE;
        }
        for (std::size_t i = 0; i < iterations; ++i) iteration_forkunion_dynamic(pool, bodies.data(), forces.data(), n);
        return EXIT_SUCCESS;
    }

#if FU_ENABLE_NUMA
    fu::numa_topology_t topology;
    if (!topology.try_harvest()) {
        std::fprintf(stderr, "Failed to harvest NUMA topology\n");
        return EXIT_FAILURE;
    }

    linux_distributed_pool_t numa_pool(std::move(topology));
    std::vector<linux_numa_bodies_t> bodies_numa_arrays = make_buffers_for_forkunion_numa(numa_pool, n);
    std::vector<body_t *> bodies_numa_buffers(bodies_numa_arrays.size());
    for (std::size_t i = 0; i < bodies_numa_arrays.size(); ++i) bodies_numa_buffers[i] = bodies_numa_arrays[i].data();

    if (backend == "forkunion_numa_static") {
        if (!numa_pool.try_spawn(threads)) {
            std::fprintf(stderr, "Failed to spawn NUMA thread pools\n");
            return EXIT_FAILURE;
        }
        for (std::size_t i = 0; i < iterations; ++i)
            iteration_forkunion_numa_static(numa_pool, bodies.data(), forces.data(), n, bodies_numa_buffers.data());
        return EXIT_SUCCESS;
    }
    if (backend == "forkunion_numa_dynamic") {
        if (!numa_pool.try_spawn(threads)) {
            std::fprintf(stderr, "Failed to spawn NUMA thread pools\n");
            return EXIT_FAILURE;
        }
        for (std::size_t i = 0; i < iterations; ++i)
            iteration_forkunion_numa_dynamic(numa_pool, bodies.data(), forces.data(), n, bodies_numa_buffers.data());
        return EXIT_SUCCESS;
    }
#endif // FU_ENABLE_NUMA

    std::fprintf(stderr, "Unsupported backend: %s\n", backend.data());
    return EXIT_FAILURE;
}
