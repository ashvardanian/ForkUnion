/**
 *  @brief Demo app: N-Body simulation driven through ForkUnion's @b C ABI from C++.
 *  @author Ash Vardanian
 *  @file nbody.cpp
 *
 *  This benchmark is C++ - for `std::vector`, RAII, and lambdas that keep state ownership clear - but
 *  it reaches ForkUnion only through the @b C header `<forkunion.h>`, not the C++ templates. That
 *  matters: the C ABI selects the busy-wait waiter at @b runtime, so on AArch64 it can reach
 *  `arm64_wfet_t`, which the compile-time C++ `preferred_yield_t` never can (no compiler defines a
 *  `WFxT` macro). A single run therefore reports `WFET` against `YIELD` on the same machine.
 *
 *  Environment variables:
 *  - `NBODY_COUNT` - number of bodies (default: the thread count).
 *  - `NBODY_ITERATIONS` - iterations per timed run (default: 1000).
 *  - `NBODY_BACKEND` - one of `openmp_static`, `openmp_dynamic`, `forkunion_static`,
 *    `forkunion_dynamic`, `forkunion_numa_static`, `forkunion_numa_dynamic` (default: `forkunion_static`).
 *  - `NBODY_THREADS` - thread count (default: the machine's logical core count).
 *  - `NBODY_WAITER` - force one waiter for the forkunion backends, by its capability name as printed in
 *    the banner: `x86_pause`, `x86_tpause`, `arm64_yield`, `arm64_wfet`, `risc5_pause`, `risc5_wrs`.
 *    Unset iterates every waiter the machine offers.
 *  - `NBODY_RUNS` - timed repetitions; the minimum is reported (default: 3).
 *
 *  @code{.sh}
 *  cmake -B build_release -D CMAKE_BUILD_TYPE=Release && cmake --build build_release
 *  NBODY_COUNT=128 NBODY_ITERATIONS=1000000 NBODY_BACKEND=forkunion_static build_release/forkunion_nbody
 *  @endcode
 */
#include <cstdint> // `std::uint32_t`
#include <cstdio>  // `std::printf`
#include <cstdlib> // `std::getenv`, `std::strtoull`
#include <cstring> // `std::memcpy`
#include <chrono>  // `std::chrono::steady_clock`
#include <limits>  // `std::numeric_limits`
#include <string>  // `std::string`
#include <utility> // `std::move`, `std::exchange`
#include <vector>  // `std::vector`

#if defined(_OPENMP) && __has_include(<omp.h>)
#include <omp.h>
#define NBODY_HAS_OPENMP 1
#else
#define NBODY_HAS_OPENMP 0
#endif

#include <forkunion.h> // The @b C ABI - not `<forkunion.hpp>`

#pragma region Shared Logic

static constexpr float g_const = 6.674e-11f;
static constexpr float dt_const = 0.01f;
static constexpr float softening_const = 1e-9f;

struct vector3_t {
    float x = 0, y = 0, z = 0;

    vector3_t &operator+=(vector3_t const &other) noexcept {
        x += other.x, y += other.y, z += other.z;
        return *this;
    }
};

struct body_t {
    vector3_t position;
    vector3_t velocity;
    float mass = 0;
};

static inline float fast_rsqrt(float x) noexcept {
    std::uint32_t i;
    std::memcpy(&i, &x, sizeof(i));
    i = 0x5f3759df - (i >> 1);
    float y;
    std::memcpy(&y, &i, sizeof(y));
    return y * (1.5f - x * 0.5f * y * y);
}

static inline vector3_t gravitational_force(body_t const &bi, body_t const &bj) noexcept {
    float dx = bj.position.x - bi.position.x;
    float dy = bj.position.y - bi.position.y;
    float dz = bj.position.z - bi.position.z;
    float l2_squared = dx * dx + dy * dy + dz * dz + softening_const;
    float l2_reciprocal = fast_rsqrt(l2_squared);
    float l2_cube_reciprocal = l2_reciprocal * l2_reciprocal * l2_reciprocal;
    float mag = g_const * bi.mass * bj.mass * l2_cube_reciprocal;
    return {mag * dx, mag * dy, mag * dz};
}

static inline void apply_force(body_t &bi, vector3_t const &f) noexcept {
    bi.velocity.x += f.x / bi.mass * dt_const;
    bi.velocity.y += f.y / bi.mass * dt_const;
    bi.velocity.z += f.z / bi.mass * dt_const;
    bi.position.x += bi.velocity.x * dt_const;
    bi.position.y += bi.velocity.y * dt_const;
    bi.position.z += bi.velocity.z * dt_const;
}

/** A seeded, deterministic splitmix64, so every timed run does identical work. */
struct rng_t {
    std::uint64_t state = 0x123456789abcdefULL;
    std::uint64_t next() noexcept {
        std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    float in(float lo, float hi) noexcept {
        float unit = static_cast<float>(next() >> 40) * (1.0f / 16777216.0f);
        return lo + unit * (hi - lo);
    }
};

static std::vector<body_t> make_bodies(std::size_t n) {
    std::vector<body_t> bodies(n);
    rng_t rng;
    for (body_t &body : bodies) {
        body.position = {rng.in(0.f, 1.f), rng.in(0.f, 1.f), rng.in(0.f, 1.f)};
        body.velocity = {rng.in(0.f, 1.f), rng.in(0.f, 1.f), rng.in(0.f, 1.f)};
        body.mass = rng.in(1e20f, 1e25f);
    }
    return bodies;
}

#pragma endregion Shared Logic

#pragma region C ABI RAII Wrappers

/** Owns a `fu_pool_t`, and dispatches lambdas through the C callback ABI via a captureless trampoline. */
struct pool_t {
    fu_pool_t *handle = nullptr;

    pool_t(char const *name, fu_capabilities_t allowed) noexcept : handle(fu_pool_new(name, allowed)) {}
    ~pool_t() noexcept {
        if (handle) fu_pool_delete(handle);
    }
    pool_t(pool_t const &) = delete;
    pool_t &operator=(pool_t const &) = delete;

    explicit operator bool() const noexcept { return handle != nullptr; }
    bool spawn(std::size_t threads) noexcept { return fu_pool_spawn(handle, threads, fu_caller_inclusive_k) != 0; }
    std::size_t threads_count() const noexcept { return fu_pool_threads_count(handle); }

    /** The lambda receives `(task, compute_domain)`; `thread` is dropped, unused by every backend here. */
    template <typename fork_type_>
    void for_n(std::size_t n, bool dynamic, fork_type_ const &fork) const noexcept {
        fu_for_prongs_t trampoline = [](void *context, std::size_t task, std::size_t, std::size_t compute_domain) {
            (*static_cast<fork_type_ const *>(context))(task, compute_domain);
        };
        void *context = const_cast<void *>(static_cast<void const *>(&fork));
        if (dynamic) fu_pool_for_n_dynamic(handle, n, trampoline, context);
        else
            fu_pool_for_n(handle, n, trampoline, context);
    }

    /** The lambda receives `(thread, compute_domain)`. */
    template <typename fork_type_>
    void for_threads(fork_type_ const &fork) const noexcept {
        fu_for_threads_t trampoline = [](void *context, std::size_t thread, std::size_t compute_domain) {
            (*static_cast<fork_type_ const *>(context))(thread, compute_domain);
        };
        fu_pool_for_threads(handle, trampoline, const_cast<void *>(static_cast<void const *>(&fork)));
    }
};

/** Owns one `fu_allocate_in` block, bound to a memory domain, freed with `fu_free_in`. */
struct numa_buffer_t {
    body_t *data = nullptr;
    std::size_t memory_domain = 0;
    std::size_t bytes = 0;

    numa_buffer_t() = default;
    numa_buffer_t(std::size_t memory_domain, std::size_t count) noexcept
        : data(static_cast<body_t *>(fu_allocate_in(memory_domain, count * sizeof(body_t)))),
          memory_domain(memory_domain), bytes(count * sizeof(body_t)) {}
    ~numa_buffer_t() noexcept {
        if (data) fu_free_in(memory_domain, data, bytes);
    }
    numa_buffer_t(numa_buffer_t &&other) noexcept
        : data(std::exchange(other.data, nullptr)), memory_domain(other.memory_domain),
          bytes(std::exchange(other.bytes, 0)) {}
    numa_buffer_t &operator=(numa_buffer_t &&other) noexcept {
        std::swap(data, other.data), std::swap(memory_domain, other.memory_domain), std::swap(bytes, other.bytes);
        return *this;
    }
    numa_buffer_t(numa_buffer_t const &) = delete;
    numa_buffer_t &operator=(numa_buffer_t const &) = delete;
};

#pragma endregion C ABI RAII Wrappers

#pragma region Backends

static void iteration_forkunion(pool_t const &pool, bool dynamic, std::vector<body_t> &bodies,
                                std::vector<vector3_t> &forces) noexcept {
    std::size_t const n = bodies.size();
    body_t const *bodies_data = bodies.data();
    vector3_t *forces_data = forces.data();
    pool.for_n(n, dynamic, [=](std::size_t task, std::size_t) noexcept {
        vector3_t f;
        for (std::size_t j = 0; j < n; ++j) f += gravitational_force(bodies_data[task], bodies_data[j]);
        forces_data[task] = f;
    });
    body_t *bodies_mut = bodies.data();
    pool.for_n(n, dynamic,
               [=](std::size_t task, std::size_t) noexcept { apply_force(bodies_mut[task], forces_data[task]); });
}

/**
 *  Per-memory-domain replicas plus the map from a pool thread to the domain it fills and its rank
 *  among that domain's threads - the C-ABI substitute for the C++ pool's `thread_local_index`.
 */
struct numa_replicas_t {
    /** One replica of the bodies per memory domain. */
    std::vector<numa_buffer_t> replicas;
    /** The memory domain each pool thread fills, indexed by thread. */
    std::vector<std::size_t> thread_memory_domain;
    /** The rank of each thread within its memory domain, indexed by thread. */
    std::vector<std::size_t> thread_local_index;
    /** The number of threads mapped to each memory domain. */
    std::vector<std::size_t> memory_domain_thread_count;
    /** Whether the replicas and the thread map were built successfully. */
    bool ok = false;

    numa_replicas_t() = default;
    numa_replicas_t(pool_t const &pool, std::size_t n) {
        std::size_t const memory_domains = fu_memory_domains_count();
        std::size_t const threads = pool.threads_count();
        if (memory_domains == 0 || threads == 0) return;

        for (std::size_t m = 0; m < memory_domains; ++m) {
            replicas.emplace_back(m, n);
            if (!replicas.back().data) return; // ! Allocation failed
        }
        thread_memory_domain.resize(threads);
        thread_local_index.resize(threads);
        memory_domain_thread_count.assign(memory_domains, 0);

        // The parallel pass writes only this thread's own slot, so no two threads touch the same
        // element. The counter below is indexed by @b memory @b domain, which many threads share, so
        // it cannot be incremented here without a race - that ranking happens serially afterwards.
        pool.for_threads([&](std::size_t thread, std::size_t compute_domain) noexcept {
            thread_memory_domain[thread] = fu_local_memory_of(compute_domain);
        });
        // The serial pass ranks each thread within its memory domain and counts each domain's threads.
        for (std::size_t thread = 0; thread < threads; ++thread) {
            std::size_t const memory_domain = thread_memory_domain[thread];
            thread_local_index[thread] = memory_domain_thread_count[memory_domain]++;
        }
        ok = true;
    }
};

/** Fair-chunk split, reproducing `indexed_split::operator[]` from types.hpp. */
static inline void fair_chunk(std::size_t total, std::size_t parts, std::size_t i, std::size_t &first,
                              std::size_t &count) noexcept {
    std::size_t const quotient = total / parts, remainder = total % parts;
    first = quotient * i + (i < remainder ? i : remainder);
    count = quotient + (i < remainder ? 1u : 0u);
}

static void iteration_forkunion_numa(pool_t const &pool, bool dynamic, numa_replicas_t &numa,
                                     std::vector<body_t> &bodies, std::vector<vector3_t> &forces) noexcept {
    std::size_t const n = bodies.size();
    body_t const *source = bodies.data();

    // Replicate the canonical bodies onto every memory domain's replica.
    pool.for_threads([&](std::size_t thread, std::size_t) noexcept {
        std::size_t const memory_domain = numa.thread_memory_domain[thread];
        std::size_t const parts = numa.memory_domain_thread_count[memory_domain];
        if (parts == 0) return;
        std::size_t first, count;
        fair_chunk(n, parts, numa.thread_local_index[thread], first, count);
        std::memcpy(numa.replicas[memory_domain].data + first, source + first, count * sizeof(body_t));
    });

    vector3_t *forces_data = forces.data();
    numa_buffer_t const *replicas = numa.replicas.data();
    pool.for_n(n, dynamic, [=](std::size_t task, std::size_t compute_domain) noexcept {
        body_t const *nearby = replicas[fu_local_memory_of(compute_domain)].data;
        body_t const body_i = nearby[task];
        vector3_t f;
        for (std::size_t j = 0; j < n; ++j) f += gravitational_force(body_i, nearby[j]);
        forces_data[task] = f;
    });
    body_t *bodies_mut = bodies.data();
    pool.for_n(n, dynamic,
               [=](std::size_t task, std::size_t) noexcept { apply_force(bodies_mut[task], forces_data[task]); });
}

#if NBODY_HAS_OPENMP
static void iteration_openmp(bool dynamic, std::vector<body_t> &bodies, std::vector<vector3_t> &forces) noexcept {
    std::size_t const n = bodies.size();
    body_t *bodies_data = bodies.data();
    vector3_t *forces_data = forces.data();
    if (dynamic) {
#pragma omp parallel for schedule(dynamic, 1)
        for (std::size_t i = 0; i < n; ++i) {
            vector3_t f;
            for (std::size_t j = 0; j < n; ++j) f += gravitational_force(bodies_data[i], bodies_data[j]);
            forces_data[i] = f;
        }
#pragma omp parallel for schedule(dynamic, 1)
        for (std::size_t i = 0; i < n; ++i) apply_force(bodies_data[i], forces_data[i]);
    }
    else {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            vector3_t f;
            for (std::size_t j = 0; j < n; ++j) f += gravitational_force(bodies_data[i], bodies_data[j]);
            forces_data[i] = f;
        }
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) apply_force(bodies_data[i], forces_data[i]);
    }
}
#endif

#pragma endregion Backends

#pragma region Harness

static std::size_t parse_size(char const *value, std::size_t fallback) noexcept {
    if (!value || !*value) return fallback;
    unsigned long long parsed = std::strtoull(value, nullptr, 10);
    std::size_t const size_max = std::numeric_limits<std::size_t>::max();
    return parsed > size_max ? size_max : static_cast<std::size_t>(parsed);
}

/** The two axes an `NBODY_BACKEND` name encodes, parsed once rather than re-scanned at each site. */
struct backend_t {
    bool dynamic;
    bool numa;
};
static backend_t parse_backend(std::string const &name) noexcept {
    return {name.find("dynamic") != std::string::npos, name.find("numa") != std::string::npos};
}

/**
 *  Runs `iterations` of the chosen backend `runs` times, restoring the pristine bodies before each,
 *  and returns the minimum elapsed seconds.
 */
static double time_backend(pool_t const &pool, bool dynamic, bool is_numa, numa_replicas_t *numa,
                           std::vector<body_t> const &pristine, std::vector<body_t> &bodies,
                           std::vector<vector3_t> &forces, std::size_t iterations, std::size_t runs) noexcept {
    double best = 0.0;
    for (std::size_t r = 0; r < runs; ++r) {
        bodies = pristine;
        auto const start = std::chrono::steady_clock::now();
        for (std::size_t it = 0; it < iterations; ++it) {
            if (is_numa) iteration_forkunion_numa(pool, dynamic, *numa, bodies, forces);
            else
                iteration_forkunion(pool, dynamic, bodies, forces);
        }
        double const elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (r == 0 || elapsed < best) best = elapsed;
    }
    return best;
}

static bool run_forkunion_waiter(backend_t backend, char const *label, fu_capabilities_t mask, std::size_t threads,
                                 std::vector<body_t> const &pristine, std::vector<body_t> &bodies,
                                 std::vector<vector3_t> &forces, std::size_t iterations, std::size_t runs) {
    if (backend.numa) mask = static_cast<fu_capabilities_t>(mask | fu_capability_place_memory_on_domain_k);

    pool_t pool("nbody", mask);
    if (!pool || !pool.spawn(threads)) {
        std::fprintf(stderr, "  %-14s spawn failed\n", label);
        return false;
    }

    numa_replicas_t numa(backend.numa ? numa_replicas_t(pool, bodies.size()) : numa_replicas_t());
    if (backend.numa && !numa.ok) {
        std::fprintf(stderr, "  %-14s NUMA replica setup failed\n", label);
        return false;
    }

    double const best =
        time_backend(pool, backend.dynamic, backend.numa, &numa, pristine, bodies, forces, iterations, runs);
    double const per_interaction = best / static_cast<double>(iterations) / static_cast<double>(bodies.size()) /
                                   static_cast<double>(bodies.size()) * 1e9;
    std::printf("  %-14s %8.3f s   (%.2f ns/interaction)\n", label, best, per_interaction);
    return true;
}

#pragma endregion Harness

int main() {
    std::printf("Welcome to the ForkUnion N-Body simulation (C ABI from C++)!\n");
    char const *const runtime_string = fu_runtime_capabilities_string();
    if (!runtime_string) {
        std::fprintf(stderr, "Thread pool not supported on this platform\n");
        return EXIT_FAILURE;
    }
    std::printf("Compiled with: %s\n", fu_comptime_capabilities_string());
    std::printf("Running on:    %s\n", runtime_string);

    std::size_t threads = parse_size(std::getenv("NBODY_THREADS"), fu_logical_cores_count());
    if (threads == 0) threads = 4;
    std::size_t n = parse_size(std::getenv("NBODY_COUNT"), threads);
    if (n == 0) n = threads;
    std::size_t const iterations = parse_size(std::getenv("NBODY_ITERATIONS"), 1000);
    std::size_t const runs = parse_size(std::getenv("NBODY_RUNS"), 3);
    std::string const backend = std::getenv("NBODY_BACKEND") ? std::getenv("NBODY_BACKEND") : "forkunion_static";

    std::vector<body_t> const pristine = make_bodies(n);
    std::vector<body_t> bodies = pristine;
    std::vector<vector3_t> forces(n);

    if (backend.rfind("openmp", 0) == 0) {
#if NBODY_HAS_OPENMP
        bool const dynamic = parse_backend(backend).dynamic;
        omp_set_num_threads(static_cast<int>(threads));
        double best = 0.0;
        for (std::size_t r = 0; r < runs; ++r) {
            bodies = pristine;
            auto const start = std::chrono::steady_clock::now();
            for (std::size_t it = 0; it < iterations; ++it) iteration_openmp(dynamic, bodies, forces);
            double const elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            if (r == 0 || elapsed < best) best = elapsed;
        }
        std::printf("  %-14s %8.3f s\n", backend.c_str(), best);
        return EXIT_SUCCESS;
#else
        std::fprintf(stderr, "OpenMP backend requested, but this build has no OpenMP\n");
        return EXIT_FAILURE;
#endif
    }

    if (backend.rfind("forkunion", 0) != 0) {
        std::fprintf(stderr, "Unsupported backend: %s\n", backend.c_str());
        return EXIT_FAILURE;
    }

    backend_t const backend_flags = parse_backend(backend);
    if (backend_flags.numa && !(fu_comptime_capabilities() & fu_capability_colocate_pools_on_domain_k)) {
        std::fprintf(stderr, "NUMA backend requested, but this build has no colocated pools\n");
        return EXIT_FAILURE;
    }

    std::printf("Backend %s, %zu bodies, %zu iterations, %zu threads, min of %zu runs:\n", backend.c_str(), n,
                iterations, threads, runs);

    fu_capabilities_t const available =
        static_cast<fu_capabilities_t>(fu_runtime_capabilities() & fu_capability_any_yield_k);

    // Build the list of waiters to time: one forced entry if `NBODY_WAITER` is set, else every one
    // the machine offers. There are at most six waiter bits, so a fixed array needs no bound check.
    struct waiter_run_t {
        fu_capabilities_t mask;
        char const *label;
    };
    waiter_run_t waiter_runs[6];
    std::size_t waiter_runs_count = 0;

    if (char const *const forced_name = std::getenv("NBODY_WAITER")) {
        fu_capabilities_t const mask = static_cast<fu_capabilities_t>(fu_capability_named(forced_name) & available);
        if (!mask) {
            std::fprintf(stderr, "Requested waiter '%s' is not available on this machine\n", forced_name);
            return EXIT_FAILURE;
        }
        waiter_runs[waiter_runs_count++] = {mask, forced_name};
    }
    else
        for (unsigned bit = 1; bit; bit <<= 1) {
            fu_capabilities_t const one = static_cast<fu_capabilities_t>(bit);
            if (available & one) waiter_runs[waiter_runs_count++] = {one, fu_capability_name(one)};
        }

    bool all_ok = true;
    for (std::size_t i = 0; i < waiter_runs_count; ++i)
        all_ok &= run_forkunion_waiter(backend_flags, waiter_runs[i].label, waiter_runs[i].mask, threads, pristine,
                                       bodies, forces, iterations, runs);
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
