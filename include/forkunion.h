/**
 *  @brief  Low-latency OpenMP-style NUMA-aware cross-platform fine-grained parallelism library.
 *  @file   forkunion.h
 *  @author Ash Vardanian
 *  @date   June 17, 2025
 *
 *  ForkUnion provides a minimalistic cross-platform thread-pool implementation and Parallel Algorithms,
 *  avoiding dynamic memory allocations, exceptions, system calls, and heavy Compare-And-Swap instructions.
 *  The library leverages the "weak memory model" to allow Arm and IBM Power CPUs to aggressively optimize
 *  execution at runtime. It also aggressively tests against overflows on smaller index types, and is safe
 *  to use even with the maximal `size_t` values. It's compatible with C 99 and later.
 *
 *  @code{.c}
 *  #include <stdio.h> // `printf`
 *  #include <stdlib.h> // `EXIT_FAILURE`, `EXIT_SUCCESS`
 *  #include <forkunion.h> // `fu_pool_t`
 *
 *  struct print_args_context_t {
 *      size_t argc; // ? Number of arguments
 *      char **argv; // ? Array of arguments
 *  };
 *
 *  void print_arg(void *context_punned, size_t task_index, size_t thread_index, size_t compute_domain_index) {
 *      print_args_context_t *context = (print_args_context_t *)context_punned;
 *      printf(
 *          "Printing argument # %zu from thread # %zu at compute_domain # %zu: %s\n",
 *          task_index, context->argc, thread_index, compute_domain_index, context->argv[task_index]);
 *  }
 *
 *  int main(int argc, char *argv[]) {
 *      char const *caps = fu_runtime_capabilities_string();
 *      if (!caps) return EXIT_FAILURE; // ! Thread pool is not supported
 *      printf("ForkUnion capabilities: %s\n", caps);
 *
 *      fu_pool_t *pool = fu_pool_new("forkunion_demo");
 *      if (!pool) return EXIT_FAILURE; // ! Failed to create a thread pool
 *
 *      size_t threads = fu_logical_cores_count();
 *      if (!fu_pool_spawn(pool, threads, fu_caller_inclusive_k)) return EXIT_FAILURE; // ! Can't spawn
 *
 *      print_args_context_t context = {argc, argv};
 *      fu_pool_for_n(pool, argc, &print_arg, &context);
 *      fu_pool_delete(pool);
 *      return EXIT_SUCCESS;
 *  }
 *  @endcode
 *
 *  Unlike the C++ version, the C header wraps the best-fit pre-compiled platform-specific instantiation
 *  of C++ templates. It also uses a singleton state to store the NUMA topology and other OS/machine specs.
 *  Under the hood, the `fu_pool_t` maps to a `basic_pool` or `distributed_pool`.
 *  For advanced usage, prefer the core C++ library.
 *
 *  The next layer of logic is for basic index-addressable tasks. It includes basic parallel loops:
 *
 *  - `fu_pool_for_n` - for iterating over a range of similar duration tasks, addressable by an index.
 *  - `fu_pool_for_n_dynamic` - for unevenly distributed tasks, where each task may take a different time.
 *  - `fu_pool_for_slices` - for iterating over a range of similar duration tasks, addressable by a slice.
 *
 *  On Linux, when NUMA and PThreads are available, the library can also leverage @b NUMA-aware
 *  memory allocations and pin threads to specific physical cores to increase memory locality.
 *  It should reduce memory access latency by around 35% on average, compared to remote accesses.
 *  @sa `fu_memory_domains_count`, `fu_allocate_at_least_in`, `fu_free_in`.
 *
 *  On heterogeneous chips, cores with a different @b "Quality-of-Service" (QoS) may be combined.
 *  A typical example is laptop/desktop chips, having 1 NUMA node, but 3 tiers of CPU cores:
 *  performance, efficiency, and power-saving cores. Each group will have vastly different speed,
 *  so considering them equal in tasks scheduling is a bad idea... and separating them automatically
 *  isn't feasible either. It's up to the user to isolate those groups into individual pools.
 *  @sa `fu_compute_levels_count`
 *
 *  On x86, Arm, and RISC-V architectures, depending on the CPU features available, the library also
 *  exposes cheaper @b "busy-waiting" mechanisms, such as `tpause`, `wfet`, & `yield` instructions.
 *  @sa `fu_runtime_capabilities_string`
 *
 *  Minimum version of C 99 is needed to allow for `size_t` and other standard types.
 *  This significantly reduces complexity compared to the C++ templated version.
 *  @see https://en.cppreference.com/w/c/language/arithmetic_types
 */
#pragma once
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h> // `size_t`, `bool`

/** @brief Returns the major version component of the ForkUnion library. */
int fu_version_major(void);
/** @brief Returns the minor version component of the ForkUnion library. */
int fu_version_minor(void);
/** @brief Returns the patch version component of the ForkUnion library. */
int fu_version_patch(void);

#pragma region Types

/** @brief Boolean type: 0 for false, non-zero for true. */
typedef int fu_bool_t;
/** @brief Opaque, cross-platform thread-pool handle. */
typedef void *fu_pool_t;
/** @brief Type-punned pointer to a user-defined callback context. */
typedef void *fu_lambda_context_t;

/**
 *  @brief Callback type for thread-level operations.
 *  @param[in] context Type-punned pointer to user-defined context data.
 *  @param[in] thread The thread index in [0, threads_count).
 *  @param[in] compute_domain The compute-domain index in [0, `fu_compute_domains_count()`).
 */
typedef void (*fu_for_threads_t)(fu_lambda_context_t context, size_t thread, size_t compute_domain);

/**
 *  @brief Callback type for task-level operations receiving individual indices.
 *  @param[in] context Type-punned pointer to user-defined context data.
 *  @param[in] task The task index in [0, n).
 *  @param[in] thread The thread index in [0, threads_count).
 *  @param[in] compute_domain The compute-domain index in [0, `fu_compute_domains_count()`).
 */
typedef void (*fu_for_prongs_t)(fu_lambda_context_t context, size_t task, size_t thread, size_t compute_domain);

/**
 *  @brief Callback type for slice-level operations receiving ranges of tasks.
 *  @param[in] context Type-punned pointer to user-defined context data.
 *  @param[in] first The first task index in the slice.
 *  @param[in] count The number of tasks in the slice.
 *  @param[in] thread The thread index in [0, threads_count).
 *  @param[in] compute_domain The compute-domain index in [0, `fu_compute_domains_count()`).
 */
typedef void (*fu_for_slices_t)(fu_lambda_context_t context, size_t first, size_t count, size_t thread,
                                size_t compute_domain);

/**
 *  @brief Defines the in- and exclusivity of the calling thread for the executing task.
 *  @sa `fu_caller_inclusive_k` and `fu_caller_exclusive_k`
 *
 *  This enum affects how the join is performed. If the caller is inclusive, 1/Nth of the call
 *  will be executed by the calling thread (as opposed to workers) and the join will happen
 *  inside of the calling scope.
 */
typedef enum fu_caller_exclusivity_t {
    /** The calling thread participates in the workload. */
    fu_caller_inclusive_k,
    /** The calling thread only coordinates, doesn't execute tasks. */
    fu_caller_exclusive_k,
} fu_caller_exclusivity_t;

#pragma endregion Types

#pragma region Metadata

/**
 *  @brief Describes all the special library features, both those compiled in and those found here.
 *  @sa `fu_comptime_capabilities` and `fu_runtime_capabilities`
 *
 *  Two questions share one bit-space, and the names say which is which. An unmarked bit is a fact
 *  about @b this @b machine: `fu_capability_huge_pages_k` means the kernel is offering them. A bit
 *  marked `comptime` is a fact about @b this @b build: `fu_capability_comptime_huge_pages_k` means
 *  we compiled the code that would ask for them.
 *
 *  Neither implies the other. A binary carrying `fu_capability_comptime_numa_memory_k` runs
 *  perfectly well on a single-node box, where `fu_capability_numa_aware_k` never appears; and a
 *  machine with four NUMA nodes reports none of them to a build that left the topology out.
 */
typedef enum fu_capabilities_t {
    fu_capabilities_unknown_k = 0,

    /** The `PAUSE` spin hint, on every x86 since the Pentium 4. */
    fu_capability_x86_pause_k = 1 << 1,
    /** `TPAUSE` sleeps the core until a deadline, rather than spinning. Needs the `WAITPKG` feature. */
    fu_capability_x86_tpause_k = 1 << 2,
    /** The `YIELD` hint, on every AArch64. Releases the pipeline to a sibling hardware thread. */
    fu_capability_arm64_yield_k = 1 << 3,
    /** `WFET` sleeps the core until a deadline or an event. Needs `FEAT_WFxT`. */
    fu_capability_arm64_wfet_k = 1 << 4,
    /** The `PAUSE` spin hint, from the `Zihintpause` extension. */
    fu_capability_risc5_pause_k = 1 << 5,
    /** `WRS.STO` sleeps the hart until a reservation breaks or a timeout. Needs the `Zawrs` extension. */
    fu_capability_risc5_wrs_k = 1 << 7,

    /** Pinned to a single compute domain. */
    fu_capability_compute_domain_k = 1 << 6,

    /** NUMA-aware memory allocations. */
    fu_capability_numa_aware_k = 1 << 10,
    /** Reducing TLB pressure with huge pages. */
    fu_capability_huge_pages_k = 1 << 11,
    /** ... doing the same "transparently". */
    fu_capability_huge_pages_transparent_k = 1 << 12,

    /** Can spawn OS threads directly, rather than through the C++ standard library. `FU_WITH_THREADS`. */
    fu_capability_comptime_threads_k = 1 << 16,
    /** Can enumerate this machine's cores, compute domains, and memory domains. `FU_WITH_TOPOLOGY`. */
    fu_capability_comptime_topology_k = 1 << 17,
    /** Can see which cores share a cache, so a domain is cut at a cluster. `FU_WITH_TOPOLOGY_CACHES`. */
    fu_capability_comptime_topology_caches_k = 1 << 18,
    /** Can read inter-domain distance, bandwidth, and latency. `FU_WITH_TOPOLOGY_METRICS`. */
    fu_capability_comptime_topology_metrics_k = 1 << 19,
    /** Can bind a thread to a set of cores, and have the kernel honour it. `FU_WITH_THREAD_PINNING`. */
    fu_capability_comptime_thread_pinning_k = 1 << 20,
    /** Can hint which class of core a thread runs on, at creation. `FU_WITH_THREAD_QOS`. */
    fu_capability_comptime_thread_qos_k = 1 << 21,
    /** Can change another thread's scheduling class, to sleep or wake it. `FU_WITH_THREAD_SCHED_CLASS`. */
    fu_capability_comptime_thread_sched_class_k = 1 << 22,
    /** Can place pages on a chosen memory domain. `FU_WITH_NUMA_MEMORY`. */
    fu_capability_comptime_numa_memory_k = 1 << 23,
    /** Can request pages larger than the base page. `FU_WITH_HUGE_PAGES`. */
    fu_capability_comptime_huge_pages_k = 1 << 24,
    /** `fu_pool_spawn_in` and the distributed pool exist. `FU_WITH_COLOCATED_POOLS`. */
    fu_capability_comptime_colocated_pools_k = 1 << 25,
} fu_capabilities_t;

/**
 *  @brief Which kernel facilities this build of ForkUnion was compiled to use.
 *
 *  Consult it before reaching for a domain-aware API. `fu_pool_spawn_in`, `fu_allocate_in`, and the
 *  rest of that surface fail on a build without `fu_capability_comptime_colocated_pools_k`, and a C,
 *  Rust, or Zig caller has no other way to tell that apart from a machine with one compute domain.
 */
fu_capabilities_t fu_comptime_capabilities(void);

/**
 *  @brief The set `fu_comptime_capabilities` bits, comma-separated, like "threads,topology".
 *  @retval "none" if nothing beyond the portable STL pool was compiled in. Never `NULL`.
 */
char const *fu_comptime_capabilities_string(void);

/**
 *  @brief Which features this machine turned out to offer, probing the CPU and the memory system.
 *
 *  These affect performance rather than availability:
 *  - `fu_capability_numa_aware_k` pools reduce memory access latency by ~35% on multi-socket servers.
 *  - `fu_capability_x86_tpause_k` and `fu_capability_arm64_wfet_k` cut power draw during busy-waits.
 */
fu_capabilities_t fu_runtime_capabilities(void);

/**
 *  @brief The set `fu_runtime_capabilities` bits, comma-separated, like "arm64_yield,numa_aware".
 *  @retval `NULL` if the thread pool is not supported on the current platform.
 *  @retval "none" if this machine offers nothing beyond a plain, portable thread pool.
 */
char const *fu_runtime_capabilities_string(void);

/**
 *  @brief Returns the number of logical cores in a given compute domain.
 *  @param[in] compute_domain_index Target compute domain, in [0, `fu_compute_domains_count()`).
 *  @retval Number of cores backing that compute domain, or 0 if the index is out of range.
 *
 *  Use this to size a per-compute-domain pool (`fu_pool_spawn_on`), or to weight work across
 *  compute domains of differing core counts (e.g. performance vs efficiency cores).
 *  @sa `fu_compute_domains_count`, `fu_pool_spawn_on`.
 */
size_t fu_logical_cores_count_in(size_t compute_domain_index);

/**
 *  @brief Describes the number of logical CPU cores available on the system.
 *  @retval 0 if the thread pool is not supported on the current platform or detection failed.
 *  @retval 1-N where N is the number of logical cores detected by the OS.
 *
 *  On x86 systems with hyper-threading enabled, this will be 2x the number of physical cores.
 *  On ARM big.LITTLE architectures, this includes both performance and efficiency cores.
 *  The returned value is suitable for passing to `fu_pool_spawn` for maximum utilization.
 *
 *  When in doubt about optimal thread count:
 *  - CPU-bound tasks: use `fu_logical_cores_count()`
 *  - Memory-bound tasks: consider `fu_memory_domains_count() * cores_per_node`
 *  - I/O-bound tasks: consider 2-4x `fu_logical_cores_count()`
 */
size_t fu_logical_cores_count(void);

/**
 *  @brief Returns the number of compute domains (bindable clusters of cores).
 *  @retval 0 if the thread pool is not supported on the current platform.
 *  @retval 1 on most desktop, laptop, or IoT platforms with a single core cluster.
 *  @retval 2+ on multi-socket servers or heterogeneous chips (per NUMA node and QoS class).
 *
 *  A @b compute @b domain is a set of cores sharing one Quality-of-Service class (performance,
 *  efficiency, ...) and locality. It is the unit a pool binds to and the index a worker
 *  callback receives. Compute domains are one axis of the topology; @b memory @b domains
 *  (`fu_memory_domains_count`) are the other. The two are bridged by `fu_local_memory_of`.
 *  @sa `fu_logical_cores_count_in`, `fu_pool_spawn_on`, `fu_memory_domains_count`.
 */
size_t fu_compute_domains_count(void);

/**
 *  @brief Returns the performance level of a given compute domain.
 *  @param[in] compute_domain_index Target compute domain, in [0, `fu_compute_domains_count()`).
 *  @retval A level ordinal where @b higher @b is @b more @b performant (0 = most efficient), or 0
 *  if the index is out of range. Homogeneous systems report level 0 for every compute domain.
 *
 *  Distinguishes performance vs efficiency cores (Intel P/E, ARM big.LITTLE). @note The compute
 *  ordinal grows with performance, while the memory-domain level (`fu_memory_level_in`) grows with
 *  @b distance - both match their native hardware conventions, so they run opposite ways by design.
 *  @sa `fu_compute_levels_count`, `fu_compute_domains_count`.
 */
size_t fu_compute_level_in(size_t compute_domain_index);

/**
 *  @brief Returns the number of distinct compute performance levels across all compute domains.
 *  @retval 0 if unsupported, 1 on homogeneous cores, 2-3 with heterogeneous cores (P/E, big.LITTLE).
 *  @note May be smaller than `fu_compute_domains_count()` - several domains can share one level,
 *  as when equally-fast cores are split across cache clusters, or across NUMA nodes.
 *  @sa `fu_compute_level_in`.
 */
size_t fu_compute_levels_count(void);

/**
 *  @brief Returns the relative throughput of @b one core in a given compute domain.
 *  @param[in] compute_domain_index Target compute domain, in [0, `fu_compute_domains_count()`).
 *  @retval A magnitude on the Linux `cpu_capacity` scale (1024 = fastest core present), or 0 when
 *  the index is out of range or the platform publishes no per-core throughput rating.
 *
 *  This is the number to weight work by; `fu_compute_level_in` is a dense ordinal and must never be
 *  divided by. When this reports 0, weigh compute domains by `fu_logical_cores_count_in` instead.
 *  @sa `fu_compute_level_in`, `fu_logical_cores_count_in`.
 */
size_t fu_compute_capacity_in(size_t compute_domain_index);

/**
 *  @brief Returns the bytes of deepest cache private to a given compute domain's cores.
 *  @param[in] compute_domain_index Target compute domain, in [0, `fu_compute_domains_count()`).
 *  @retval Cache bytes shared within the domain, or 0 if the index is out of range or unknown.
 *
 *  Sizes a cache-resident chunk - a different question from how @b many chunks a domain deserves.
 *  Domains may sustain identical throughput yet back onto very differently sized caches, so neither
 *  number can be inferred from the other.
 *  @sa `fu_compute_capacity_in`.
 */
size_t fu_compute_cache_bytes_in(size_t compute_domain_index);

/**
 *  @brief Returns the number of memory domains (distinct allocation targets).
 *  @retval 0 if unsupported, 1 on uniform-memory systems, 2+ on NUMA / tiered-memory systems.
 *
 *  A @b memory @b domain is a bank of memory with a capacity and a performance @b level
 *  (`fu_memory_level_in`; lower = faster: HBM < DDR < CXL). It is the unit the allocator targets.
 *  A memory domain may be @b cpuless (CXL expander, GPU-attached HBM) and may be local to
 *  @b several compute domains (performance and efficiency cores sharing one DDR controller).
 *  @sa `fu_volume_ram_in`, `fu_memory_level_in`, `fu_local_memory_of`, `fu_allocate_in`.
 */
size_t fu_memory_domains_count(void);

/**
 *  @brief Returns the performance level of a given memory domain.
 *  @param[in] memory_domain_index Target memory domain, in [0, `fu_memory_domains_count()`).
 *  @retval A level ordinal where @b lower @b is @b faster (0 = fastest, e.g. HBM), or 0 if the index
 *  is out of range. Uniform-memory systems report level 0 for every memory domain.
 *
 *  Ranks memory by access speed independently of compute (HBM < DDR < CXL/PMEM), following the Linux
 *  memory-tiering abstract-distance convention. @note Runs opposite to `fu_compute_level_in`, where
 *  higher is faster - each direction matches its own hardware source.
 *  @sa `fu_memory_domains_count`, `fu_volume_ram_in`.
 */
size_t fu_memory_level_in(size_t memory_domain_index);

/**
 *  @brief Returns the number of distinct memory tiers across all memory domains.
 *  @retval 0 if unsupported, 1 on single-tier systems, 2+ when HBM / DDR / CXL are mixed.
 *  @note The memory-axis twin of `fu_compute_levels_count`; several memory domains may share a tier.
 *  @sa `fu_memory_level_in`, `fu_memory_domains_count`.
 */
size_t fu_memory_levels_count(void);

/**
 *  @brief Returns the memory domain nearest to a given compute domain.
 *  @param[in] compute_domain_index Target compute domain, in [0, `fu_compute_domains_count()`).
 *  @retval The index of that compute domain's primary (lowest-distance) memory domain, or 0 if
 *  the compute-domain index is out of range.
 *
 *  The convenience bridge for the common "run here, allocate near here" pattern: pass the result
 *  to `fu_allocate_in`. For the full cost picture use `fu_memory_distance`.
 *  @sa `fu_memory_distance`, `fu_allocate_in`.
 */
size_t fu_local_memory_of(size_t compute_domain_index);

/**
 *  @brief Returns the relative access distance from a compute domain to a memory domain.
 *  @param[in] compute_domain_index Initiator compute domain, in [0, `fu_compute_domains_count()`).
 *  @param[in] memory_domain_index Target memory domain, in [0, `fu_memory_domains_count()`).
 *  @retval A relative distance where @b 10 means local (SLIT convention); larger is farther;
 *  0 means unknown or an out-of-range index.
 *
 *  A scalar summary of the (initiator -> target) cost. Per-edge bandwidth and latency come from
 *  `fu_memory_bandwidth` and `fu_memory_latency`.
 *  @sa `fu_local_memory_of`, `fu_memory_bandwidth`, `fu_memory_latency`.
 */
size_t fu_memory_distance(size_t compute_domain_index, size_t memory_domain_index);

/**
 *  @brief Returns the HMAT read bandwidth from a compute domain to a memory domain.
 *  @param[in] compute_domain_index Initiator compute domain, in [0, `fu_compute_domains_count()`).
 *  @param[in] memory_domain_index Target memory domain, in [0, `fu_memory_domains_count()`).
 *  @retval Peak read bandwidth in MB/s, or 0 when the machine exposes no ACPI HMAT table.
 *  @sa `fu_memory_latency`, `fu_memory_distance`.
 */
size_t fu_memory_bandwidth(size_t compute_domain_index, size_t memory_domain_index);

/**
 *  @brief Returns the HMAT read latency from a compute domain to a memory domain.
 *  @param[in] compute_domain_index Initiator compute domain, in [0, `fu_compute_domains_count()`).
 *  @param[in] memory_domain_index Target memory domain, in [0, `fu_memory_domains_count()`).
 *  @retval Read latency in nanoseconds, or 0 when the machine exposes no ACPI HMAT table.
 *  @sa `fu_memory_bandwidth`, `fu_memory_distance`.
 */
size_t fu_memory_latency(size_t compute_domain_index, size_t memory_domain_index);

/**
 *  @brief Returns the RAM volume (bytes) of a given memory domain.
 *  @param[in] memory_domain_index Target memory domain, in [0, `fu_memory_domains_count()`).
 *  @retval Number of bytes of RAM in that memory domain, regardless of page size; 0 if out of range.
 *  @sa `fu_volume_ram`, `fu_allocate_in`.
 */
size_t fu_volume_ram_in(size_t memory_domain_index);

/**
 *  @brief Returns the total RAM volume (bytes) across all memory domains.
 *  @retval Number of bytes of RAM installed, regardless of page size.
 *  @sa `fu_volume_ram_in`.
 */
size_t fu_volume_ram(void);

/**
 *  @brief Returns the huge-page volume (bytes) available in a given memory domain.
 *  @param[in] memory_domain_index Target memory domain, in [0, `fu_memory_domains_count()`).
 *  @retval Bytes backed by free huge pages in that memory domain; 0 if out of range or unavailable.
 *
 *  Huge pages reduce TLB pressure by mapping memory in larger units than the base page.
 *  @sa `fu_huge_pages_count_in`, `fu_allocate_at_least_in`.
 */
size_t fu_volume_huge_pages_in(size_t memory_domain_index);

/**
 *  @brief Returns the total huge-page volume (bytes) across all memory domains.
 *  @retval Number of bytes backed by free huge pages, or 0 if huge pages are unavailable.
 *  @sa `fu_volume_huge_pages_in`, `fu_huge_pages_count`.
 */
size_t fu_volume_huge_pages(void);

/**
 *  @brief Returns the number of free huge pages in a given memory domain.
 *  @param[in] memory_domain_index Target memory domain, in [0, `fu_memory_domains_count()`).
 *  @retval Count of free huge pages (summed across page sizes) in that memory domain; 0 if
 *  out of range or unavailable.
 *  @sa `fu_volume_huge_pages_in`, `fu_huge_pages_count`.
 */
size_t fu_huge_pages_count_in(size_t memory_domain_index);

/**
 *  @brief Returns the total number of free huge pages across all memory domains.
 *  @retval Count of free huge pages of any size, or 0 if huge pages are unavailable.
 *  @sa `fu_volume_huge_pages`, `fu_huge_pages_count_in`.
 */
size_t fu_huge_pages_count(void);

#pragma endregion Metadata

#pragma region Memory

/**
 *  @brief Allocates memory in a given memory domain with the largest suitable page size.
 *  @param[in] memory_domain_index Target memory domain, in [0, `fu_memory_domains_count()`).
 *  @param[in] minimum_bytes Minimum number of bytes to allocate, must be > 0.
 *  @param[out] allocated_bytes Receives the actual allocation size (>= @p minimum_bytes), must not be NULL.
 *  @param[out] bytes_per_page Receives the page size used for the allocation, must not be NULL.
 *  @retval Pointer to allocated memory, or NULL if allocation failed.
 *  @note This API is @b thread-safe and can be called from any thread.
 *
 *  Prefers the largest available huge-page size to minimize TLB pressure; the actual size may
 *  exceed the request due to page alignment, so always read @p allocated_bytes. Pair with
 *  `fu_local_memory_of` to allocate near the compute domain a thread runs on.
 *  @code{.c}
 *  size_t memory_domain = fu_local_memory_of(compute_domain);
 *  void *pointer = NULL; size_t actual_bytes = 0, page = 0;
 *  if ((pointer = fu_allocate_at_least_in(memory_domain, 1u << 20, &actual_bytes, &page)))
 *      fu_free_in(memory_domain, pointer, actual_bytes);
 *  @endcode
 *  @sa `fu_free_in`, `fu_local_memory_of`, `fu_memory_domains_count`.
 */
void *fu_allocate_at_least_in(                        //
    size_t memory_domain_index, size_t minimum_bytes, //
    size_t *allocated_bytes, size_t *bytes_per_page);

/**
 *  @brief Allocates exactly the requested number of bytes in a given memory domain.
 *  @param[in] memory_domain_index Target memory domain, in [0, `fu_memory_domains_count()`).
 *  @param[in] bytes Number of bytes to allocate, must be > 0.
 *  @retval Pointer to allocated memory, or NULL if allocation failed.
 *  @note This API is @b thread-safe. Unlike `fu_allocate_at_least_in`, it does not over-allocate
 *  for page optimization — use it for standard-allocator compatibility.
 *  @sa `fu_free_in`, `fu_allocate_at_least_in`.
 */
void *fu_allocate_in(size_t memory_domain_index, size_t bytes);

/**
 *  @brief Releases memory allocated in a given memory domain.
 *  @param[in] memory_domain_index The memory domain the memory was allocated in.
 *  @param[in] pointer Pointer to the memory to release, must not be NULL.
 *  @param[in] bytes Number of bytes to release; must match the `allocated_bytes` from allocation.
 *  @note This API is @b thread-safe. A mismatched @p bytes is undefined behavior.
 *  @sa `fu_allocate_at_least_in`, `fu_allocate_in`.
 */
void fu_free_in(size_t memory_domain_index, void *pointer, size_t bytes);

#pragma endregion Memory

#pragma region Lifetime

/**
 *  @brief Creates a new thread pool instance.
 *  @param[in] name Optional name for the thread pool, may be NULL.
 *  @retval Non-NULL pointer to an opaque thread pool handle on success.
 *  @retval NULL if creation failed due to insufficient memory or platform limitations.
 *  @note This API is @b thread-safe and can be called from any thread.
 *
 *  The returned pool is initially empty (no worker threads) and must be configured
 *  with `fu_pool_spawn` before use. Multiple pools can coexist.
 *  @sa `fu_pool_delete` for cleanup, `fu_pool_spawn` for initialization.
 */
fu_pool_t *fu_pool_new(char const *name);

/**
 *  @brief Destroys a thread pool and releases all associated resources.
 *  @param[in] pool Thread pool handle, may be NULL (no-op).
 *  @note This API is @b thread-safe but must not be called concurrently with other pool operations.
 *
 *  After calling this function, the pool handle becomes invalid and must not be used.
 *  Any pending or running tasks must complete before destruction.
 *  @sa `fu_pool_terminate` for forceful shutdown, `fu_pool_new` for creation.
 */
void fu_pool_delete(fu_pool_t *pool);

/**
 *  @brief Creates worker threads and initializes the thread pool for use.
 *  @param[in] pool Thread pool handle, must not be NULL.
 *  @param[in] threads The number of threads to create, must be > 0.
 *  @param[in] exclusivity Whether the calling thread participates in task execution.
 *  @retval 1 if the thread pool was successfully initialized and is ready to use.
 *  @retval 0 if initialization failed due to resource limits, invalid parameters, or system errors.
 *  @note This API is @b not thread-safe and should only be called once per pool.
 *
 *  This is the de-facto @b constructor for the thread pool. You must call this function
 *  before using any parallel operations. The function can only be called once per pool
 *  instance, or after `fu_pool_terminate`.
 *
 *  Thread creation behavior:
 *  - `fu_caller_inclusive_k`: Creates `(threads-1)` workers, calling thread participates
 *  - `fu_caller_exclusive_k`: Creates @p `threads` workers, calling thread only coordinates
 *
 *  @code{.c}
 *  fu_pool_t *pool = fu_pool_new();
 *  if (pool && fu_pool_spawn(pool, fu_logical_cores_count(), fu_caller_inclusive_k)) {
 *      ... // Dispatch some parallel tasks
 *      fu_pool_delete(pool);
 *  }
 *  @endcode
 *  @sa `fu_pool_terminate` for shutdown, `fu_logical_cores_count` for optimal thread count.
 */
fu_bool_t fu_pool_spawn(fu_pool_t *pool, size_t threads, fu_caller_exclusivity_t exclusivity);

/**
 *  @brief Spawns a pool pinned to a single compute domain.
 *  @param[in] pool Thread pool handle, must not be NULL.
 *  @param[in] compute_domain_index Target compute domain, in [0, `fu_compute_domains_count()`).
 *  @param[in] threads The number of threads to create, must be > 0.
 *  @param[in] exclusivity Whether the calling thread participates in task execution.
 *  @retval 1 on success; 0 on failure or if @p compute_domain_index is out of range.
 *  @note This API is @b not thread-safe and should only be called once per pool.
 *
 *  Placement lives here, not in creation: `fu_pool_new` allocates the handle, and this binds it
 *  to one compute domain, so its threads and its NUMA-local allocations stay on that domain.
 *  Spawn one pool per compute domain and coordinate them with the generation-token API. Contrast
 *  `fu_pool_spawn`, which spans @b all compute domains. On builds without NUMA, only compute
 *  domain 0 is valid.
 *  @sa `fu_pool_spawn`, `fu_compute_domains_count`, `fu_logical_cores_count_in`.
 */
fu_bool_t fu_pool_spawn_on(fu_pool_t *pool, size_t compute_domain_index, size_t threads,
                           fu_caller_exclusivity_t exclusivity);

/**
 *  @brief Returns whether the calling thread participates in task execution.
 *  @param[in] pool Thread pool handle, must not be NULL and initialized.
 *  @retval `fu_caller_inclusive_k` if the calling thread contributes a slice of the work.
 *  @retval `fu_caller_exclusive_k` if the calling thread only coordinates.
 *  @note This API is @b not synchronized.
 *
 *  This reflects the exclusivity passed to the most recent `fu_pool_spawn` - the pool
 *  itself is the single source of truth, so this stays correct across `fu_pool_terminate`
 *  and re-spawning with a different mode. It also determines the completion contract:
 *  on `fu_caller_inclusive_k` pools the caller owes a slice that only runs inside
 *  `fu_pool_unsafe_join`, so `fu_pool_is_complete` cannot be reached by polling alone.
 *  @sa `fu_pool_spawn` for setting the mode, `fu_pool_is_complete` for the polling contract.
 */
fu_caller_exclusivity_t fu_pool_caller_exclusivity(fu_pool_t *pool);

/**
 *  @brief Returns the number of distinct thread compute_domains in the pool.
 *  @param[in] pool Thread pool handle, must not be NULL.
 *  @retval 0 if the pool is not initialized.
 *  @retval 1 on systems without NUMA or QoS heterogeneity.
 *  @retval 2-N on systems with multiple NUMA nodes or QoS levels.
 *  @note This API is @b not synchronized.
 *
 *  A compute_domain represents a group of threads sharing the same memory domain
 *  and performance characteristics. This information is useful for:
 *  - Understanding the system's memory topology
 *  - Optimizing memory allocation strategies
 *  - Load balancing across heterogeneous cores
 *  @sa `fu_pool_threads_count_in` for per-compute_domain thread counts.
 */
size_t fu_pool_compute_domains_count(fu_pool_t *pool);

/**
 *  @brief Returns the number of threads in a specific compute_domain.
 *  @param[in] pool Thread pool handle, must not be NULL.
 *  @param[in] compute_domain_index Index of the compute_domain, must be < `fu_pool_compute_domains_count(pool)`.
 *  @retval 0 if the pool is not initialized or compute_domain_index is invalid.
 *  @retval 1-N where N is the number of threads in the specified compute_domain.
 *  @note This API is @b not synchronized and doesn't validate bounds.
 *
 *  Different compute_domains may have different thread counts depending on:
 *  - NUMA node core counts (different sockets may have different core counts)
 *  - QoS level availability (P-cores vs E-cores)
 *  - User-specified thread distribution
 *  @sa `fu_pool_compute_domains_count` for valid compute_domain indices.
 */
size_t fu_pool_threads_count_in(fu_pool_t *pool, size_t compute_domain_index);

/**
 *  @brief Returns the total number of threads in the pool.
 *  @param[in] pool Thread pool handle, must not be NULL.
 *  @retval 0 if the pool is not initialized.
 *  @retval 1-N where N is the number of threads specified in `fu_pool_spawn`.
 *  @note This API is @b not synchronized.
 *
 *  This count includes the calling thread if `fu_caller_inclusive_k` was used
 *  during spawning. The returned value represents the maximum parallelism
 *  available for task execution.
 *  @sa `fu_pool_spawn` for thread count specification.
 */
size_t fu_pool_threads_count(fu_pool_t *pool);

/**
 *  @brief Converts a global thread index to a local thread index within a compute_domain.
 *  @param[in] pool Thread pool handle, must not be NULL.
 *  @param[in] global_thread_index The global thread index to convert.
 *  @param[in] compute_domain_index Index of the compute_domain, must be < `fu_pool_compute_domains_count(pool)`.
 *  @retval Local thread index within the specified compute_domain.
 */
size_t fu_pool_locate_thread_in(fu_pool_t *pool, size_t global_thread_index, size_t compute_domain_index);

/**
 *  @brief Transitions worker threads to a power-saving sleep state.
 *  @param[in] pool Thread pool handle, must not be NULL.
 *  @param[in] micros Wake-up check interval in microseconds, must be > 0.
 *  @note This API is @b not thread-safe and should only be called between task batches.
 *
 *  This function places worker threads into a low-power sleep state when no work
 *  is available for extended periods. Threads will periodically check for new work
 *  at the specified interval.
 *
 *  Use cases:
 *  - Batch processing with long idle periods between jobs
 *  - Background services where latency is not critical
 *  - Power-constrained environments (mobile, embedded)
 *
 *  Trade-offs:
 *  - Reduces power consumption and thermal load
 *  - Increases task startup latency by up to `micros` microseconds
 *  - May affect OS scheduler decisions for other processes
 *
 *  On Linux, this also informs the scheduler to de-prioritize sleeping threads.
 *  @sa Subsequent parallel operations will automatically wake sleeping threads.
 */
void fu_pool_sleep(fu_pool_t *pool, size_t micros);

/**
 *  @brief Stops all worker threads and resets the pool to uninitialized state.
 *  @param[in] pool Thread pool handle, must not be NULL.
 *  @note This API is @b not thread-safe and should only be called when no tasks are running.
 *
 *  This function gracefully stops all worker threads and deallocates internal resources
 *  while preserving the pool handle for potential reuse. Unlike `fu_pool_delete`, the
 *  pool can be re-spawned with `fu_pool_spawn` after termination.
 *
 *  Termination sequence:
 *  1. Signals all worker threads to stop
 *  2. Waits for threads to complete their current tasks
 *  3. Joins all worker threads
 *  4. Releases thread-related resources
 *  5. Resets internal state for potential re-spawn
 *
 *  When and how @b NOT to use this function:
 *  - As a synchronization point between concurrent tasks
 *  - While any parallel operations are in progress
 *
 *  When and how to use this function:
 *  - To restart the pool with a different thread count
 *  - As cleanup in error recovery scenarios
 *  - To temporarily disable parallelism without destroying the pool
 *  @sa `fu_pool_spawn` for reinitialization, `fu_pool_delete` for permanent cleanup.
 */
void fu_pool_terminate(fu_pool_t *pool);

#pragma endregion Lifetime

#pragma region Primary API

/**
 *  @brief Executes a callback function in parallel on all threads.
 *  @param[in] pool Thread pool handle, must not be NULL and initialized.
 *  @param[in] callback Function to execute on each thread, must not be NULL.
 *  @param[in] context User-defined context passed to the callback, may be NULL.
 *  @note This API blocks until all threads complete execution.
 *
 *  This is equivalent to OpenMP's `#pragma omp parallel` directive. Each thread
 *  executes the callback exactly once with its unique thread index and compute_domain.
 *
 *  The callback receives:
 *  - `context`: User-provided data (shared across all threads)
 *  - `thread`: Thread index in [0, threads_count)
 *  - `compute_domain`: NUMA node & QoS level identifier
 *
 *  Synchronization guarantee: This function returns only after all threads have
 *  completed their callback execution. No additional synchronization is needed.
 *
 *  @code{.c}
 *  void hello_world(void *ctx, size_t thread, size_t compute_domain) {
 *      printf("Hello from thread %zu in compute_domain %zu\n", thread, compute_domain);
 *  }
 *  fu_pool_for_threads(pool, hello_world, NULL);
 *  @endcode
 *  @sa `fu_pool_unsafe_for_threads` for non-blocking execution.
 */
void fu_pool_for_threads(fu_pool_t *pool, fu_for_threads_t callback, fu_lambda_context_t context);

/**
 *  @brief Distributes `n` tasks in slices, providing range information to callbacks.
 *  @param[in] pool Thread pool handle, must not be NULL and initialized.
 *  @param[in] n Total number of tasks to split across threads, may be 0 (no-op).
 *  @param[in] callback Function to execute for each slice, must not be NULL if n > 0.
 *  @param[in] context User-defined context passed to the callback, may be NULL.
 *  @note This API blocks until all slices are processed.
 *
 *  This function splits the task range into contiguous slices and provides each
 *  thread with both the starting index and count of tasks to process. This is
 *  particularly useful when the callback can optimize for processing contiguous
 *  ranges rather than individual elements.
 *
 *  Slicing strategy:
 *  - Tasks [0, n) are divided into approximately equal-sized contiguous ranges
 *  - Each thread receives exactly one slice (first_index, count)
 *  - Threads with no work receive empty slices (count = 0)
 *  - Maximum cache locality due to sequential access patterns
 *
 *  The callback receives:
 *  - `context`: User-provided data (shared across all slices)
 *  - `first`: Starting task index for this slice
 *  - `count`: Number of tasks in this slice (may be 0)
 *  - `thread`: Thread index processing this slice
 *  - `compute_domain`: NUMA node & QoS level of the executing thread
 *
 *  Use cases:
 *  - Vectorized operations that benefit from contiguous data access
 *  - Memory copying or initialization operations
 *  - Algorithms with significant per-slice setup costs
 *  - SIMD operations that process multiple elements simultaneously
 *
 *  @code{.c}
 *  void process_slice(void *array, size_t first, size_t count, size_t thread, size_t compute_domain) {
 *      float *data = (float*)array;
 *      for (size_t i = 0; i < count; ++i) {
 *          data[first + i] = sqrt(data[first + i]);
 *      }
 *  }
 *  fu_pool_for_slices(pool, array_length, process_slice, my_float_array);
 *  @endcode
 *  @sa `fu_pool_for_n` for individual task processing.
 */
void fu_pool_for_slices(fu_pool_t *pool, size_t n, fu_for_slices_t callback, fu_lambda_context_t context);

/**
 *  @brief Distributes `n` similar-duration tasks across all threads.
 *  @param[in] pool Thread pool handle, must not be NULL and initialized.
 *  @param[in] n Number of tasks to execute, may be 0 (no-op).
 *  @param[in] callback Function to execute for each task, must not be NULL if n > 0.
 *  @param[in] context User-defined context passed to the callback, may be NULL.
 *  @note This API blocks until all tasks complete execution.
 *
 *  This function is designed for "balanced" workloads where all tasks have roughly
 *  the same execution time. Tasks are distributed in contiguous chunks across threads
 *  to maximize cache locality and minimize coordination overhead.
 *
 *  Distribution strategy:
 *  - Tasks are split into (approximately) equal-sized chunks per thread
 *  - Each thread processes a contiguous range of task indices
 *  - Load balancing assumes uniform task duration
 *
 *  The callback receives:
 *  - `context`: User-provided data (shared across all tasks)
 *  - `task`: Task index in [0, n)
 *  - `thread`: Thread index executing this task
 *  - `compute_domain`: NUMA node & QoS level of the executing thread
 *
 *  @code{.c}
 *  void process_element(void *array, size_t i, size_t thread, size_t compute_domain) {
 *      int *data = (int*)array;
 *      data[i] = data[i] * 2; // Double each element
 *  }
 *  fu_pool_for_n(pool, array_size, process_element, my_array);
 *  @endcode
 *  @sa `fu_pool_for_n_dynamic` for unbalanced workloads, `fu_pool_for_slices` for range-based processing.
 */
void fu_pool_for_n(fu_pool_t *pool, size_t n, fu_for_prongs_t callback, fu_lambda_context_t context);

/**
 *  @brief Distributes `n` variable-duration tasks using dynamic work-stealing.
 *  @param[in] pool Thread pool handle, must not be NULL and initialized.
 *  @param[in] n Number of tasks to execute, may be 0 (no-op).
 *  @param[in] callback Function to execute for each task, must not be NULL if n > 0.
 *  @param[in] context User-defined context passed to the callback, may be NULL.
 *  @note This API blocks until all tasks complete execution.
 *
 *  This function is designed for "unbalanced" workloads where tasks may have vastly
 *  different execution times. It uses a work-stealing approach where threads dynamically
 *  claim tasks from a shared queue, ensuring optimal load balancing.
 *
 *  Work-stealing strategy:
 *  - Each thread initially gets one task assigned statically
 *  - Remaining tasks are distributed via atomic counter increments
 *  - Fast threads automatically pick up more work as they complete tasks
 *  - Slower threads are not penalized by early load-balancing decisions
 *
 *  This approach is ideal for:
 *  - Tasks with unpredictable execution times
 *  - Recursive algorithms with variable depth
 *  - Processing heterogeneous data structures
 *  - Algorithms with data-dependent complexity
 *
 *  The callback receives the same parameters as `fu_pool_for_n`, but task
 *  indices may be processed out of order depending on thread scheduling.
 *
 *  @code{.c}
 *  void process_variable_work(void *ctx, size_t task, size_t thread, size_t compute_domain) {
 *      complex_computation(task); // May take 1ms or 100ms
 *  }
 *  fu_pool_for_n_dynamic(pool, task_count, process_variable_work, context);
 *  @endcode
 *  @sa `fu_pool_for_n` for balanced workloads with predictable task duration.
 */
void fu_pool_for_n_dynamic(fu_pool_t *pool, size_t n, fu_for_prongs_t callback, fu_lambda_context_t context);

#pragma endregion Primary API

#pragma region Flexible API

/**
 *  @brief Token identifying one dispatch on one pool; always an @b odd number.
 *
 *  Every generation advances the pool's internal epoch by exactly two: once at dispatch
 *  and once when the last contributor finishes. Odd epochs are in-flight, even are idle.
 */
typedef size_t fu_generation_t;

/**
 *  @brief Executes a callback in parallel on all threads without blocking.
 *  @param[in] pool Thread pool handle, must not be NULL and initialized.
 *  @param[in] callback Function to execute on each thread, must not be NULL.
 *  @param[in] context User-defined context passed to the callback, may be NULL.
 *  @return A generation token identifying this dispatch, to pass to `fu_pool_unsafe_join`.
 *  @note This API returns immediately without waiting for completion.
 *
 *  This is the non-blocking variant of `fu_pool_for_threads`. The function
 *  initiates parallel execution but returns immediately, allowing the calling
 *  thread to perform other work while tasks execute in the background.
 *
 *  It can be used to implement higher-level concurrency patterns in other
 *  programming languages.
 *
 *  Critical requirements:
 *  - Must call `fu_pool_unsafe_join()` before pool destruction or next operation
 *  - Must ensure callback and context remain valid until join completes
 *  - Cannot call other pool operations until current operation finishes
 *
 *  On `fu_caller_inclusive_k` pools the calling thread owes one slice of the work,
 *  which only runs inside `fu_pool_unsafe_join` - so the pool can't reach completion
 *  until the caller joins.
 *
 *  @code{.c}
 *  fu_generation_t generation = fu_pool_unsafe_for_threads(pool, my_callback, my_context);
 *  prepare_next_batch();
 *  fu_pool_unsafe_join(pool, generation);
 *  @endcode
 *  @sa `fu_pool_unsafe_join` for synchronization, `fu_pool_for_threads` for blocking variant.
 */
fu_generation_t fu_pool_unsafe_for_threads(fu_pool_t *pool, fu_for_threads_t callback, fu_lambda_context_t context);

/**
 *  @brief Returns whether the given generation has completed.
 *  @param[in] pool Thread pool handle, must not be NULL and initialized.
 *  @param[in] generation The generation token returned by `fu_pool_unsafe_for_threads`.
 *  @return Non-zero if complete, zero if threads are still working.
 *  @note This is a non-blocking check that can be used for polling.
 *
 *  A non-zero result also guarantees the visibility of every contributor's writes.
 *  On `fu_caller_inclusive_k` pools this can only turn non-zero once `fu_pool_unsafe_join`
 *  contributes the calling thread's slice: the poll-then-join pattern below is reserved
 *  for `fu_caller_exclusive_k` pools.
 */
fu_bool_t fu_pool_is_complete(fu_pool_t *pool, fu_generation_t generation);

/**
 *  @brief Blocks the calling thread until the given generation completes.
 *  @param[in] pool Thread pool handle, must not be NULL and initialized.
 *  @param[in] generation The generation token returned by `fu_pool_unsafe_for_threads`.
 *
 *  This function provides the synchronization point for all non-blocking pool
 *  operations. It ensures that:
 *  - All contributors complete their slices, including the calling thread's own
 *    slice on `fu_caller_inclusive_k` pools, which is executed here
 *  - Memory writes from worker threads are visible to the calling thread
 *  - The pool is ready for the next operation
 *
 *  The call is idempotent: joining an already-joined or stale generation returns
 *  immediately.
 *
 *  @code{.c}
 *  fu_generation_t generation = fu_pool_unsafe_for_threads(pool, my_callback, my_context);
 *  while (!fu_pool_is_complete(pool, generation)) { do_other_work(); } // ! Exclusive pools only
 *  fu_pool_unsafe_join(pool, generation);
 *  @endcode
 *  @sa `fu_pool_unsafe_for_threads` for the entry point, `fu_pool_for_threads` for blocking execution.
 */
void fu_pool_unsafe_join(fu_pool_t *pool, fu_generation_t generation);

#pragma endregion Flexible API

#ifdef __cplusplus
} // extern "C"
#endif
