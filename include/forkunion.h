/**
 *  @brief Low-latency OpenMP-style NUMA-aware cross-platform fine-grained parallelism library.
 *  @file forkunion.h
 *  @author Ash Vardanian
 *  @date June 17, 2025
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
 *      char capabilities[256];
 *      fu_name_capabilities(fu_runtime_capabilities(), capabilities, sizeof(capabilities));
 *      printf("ForkUnion capabilities: %s\n", capabilities);
 *
 *      fu_topology_t topology = fu_topology_new();
 *      fu_pool_t pool = fu_pool_new("forkunion_demo", fu_capabilities_all_k);
 *      if (!pool) return EXIT_FAILURE; // ! Failed to create a thread pool
 *
 *      size_t threads = fu_logical_cores_count(topology);
 *      if (!fu_pool_spawn(topology, pool, threads, fu_caller_inclusive_k)) return EXIT_FAILURE; // ! Can't spawn
 *
 *      print_args_context_t context = {argc, argv};
 *      fu_pool_for_n(pool, argc, &print_arg, &context);
 *      fu_pool_delete(pool);
 *      fu_topology_delete(topology);
 *      return EXIT_SUCCESS;
 *  }
 *  @endcode
 *
 *  The C header wraps the best-fit pre-compiled instantiation of the C++ templates. The machine
 *  topology lives in an explicit `fu_topology_t` handle you build and thread through the API.
 *  Under the hood, a `fu_pool_t` is a `flat_pool`, a `colocated_pool`, or a `distributed_pool`.
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
 *  @sa `fu_memory_domains_count`, `fu_allocate_at_least_on_domain_id`, `fu_free_on_domain_id`.
 *
 *  On heterogeneous chips, cores with a different @b "Quality-of-Service", or QoS, may be combined.
 *  A typical example is laptop/desktop chips, having 1 NUMA node, but 3 tiers of CPU cores:
 *  performance, efficiency, and power-saving cores. Each group will have vastly different speed,
 *  so considering them equal in tasks scheduling is a bad idea... and separating them automatically
 *  isn't feasible either. It's up to the user to isolate those groups into individual pools.
 *  @sa `fu_compute_levels_count`
 *
 *  On x86, Arm, and RISC-V architectures, depending on the CPU features available, the library also
 *  exposes cheaper @b "busy-waiting" mechanisms, such as `tpause`, `wfet`, & `yield` instructions.
 *  @sa `fu_runtime_capabilities`
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
/** @brief Opaque, cross-platform handle for the machine topology; immutable once constructed. */
typedef void *fu_topology_t;
/** @brief Opaque, cross-platform thread-pool handle, either flat, colocated, or distributed. */
typedef void *fu_pool_t;
/** @brief Opaque handle for the measured memory fabric - latencies, bandwidths, tiers, distances. */
typedef void *fu_fabric_t;
/** @brief Type-punned pointer to a user-defined callback context. */
typedef void *fu_lambda_context_t;
/** @brief An OS memory-domain id - a NUMA node - the allocators key off; -1 when there is none. */
typedef int fu_memory_domain_id_t;

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
 *  will be executed by the calling thread rather than the workers, and the join will happen
 *  inside the calling scope.
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
 *  One bit per facility, and two accessors ask two questions of the same bit.
 *  `fu_comptime_capabilities` reports whether the code was @b built: a set
 *  `fu_capability_place_huge_pages_on_domain_k` means we compiled the path that asks for them.
 *  `fu_runtime_capabilities` reports whether the machine @b offers it now.
 *
 *  Neither implies the other. A binary that built `fu_capability_place_memory_on_domain_k` runs
 *  perfectly well on a single-node box, where the runtime accessor never sets that bit; and a
 *  machine with four NUMA nodes reports none of them to a build that left the topology out.
 */
typedef enum fu_capabilities_t {
    fu_capabilities_unknown_k = 0,

    /** The `PAUSE` spin hint, on every x86 since the Pentium 4. */
    fu_capability_x86_pause_k = 1 << 0,
    /** `TPAUSE` sleeps the core until a deadline, rather than spinning. Needs the `WAITPKG` feature. */
    fu_capability_x86_tpause_k = 1 << 1,
    /** The `YIELD` hint, on every AArch64. Releases the pipeline to a sibling hardware thread. */
    fu_capability_arm64_yield_k = 1 << 2,
    /** `WFET` sleeps the core until a deadline or an event. Needs `FEAT_WFxT`. */
    fu_capability_arm64_wfet_k = 1 << 3,
    /** The `PAUSE` spin hint, from the `Zihintpause` extension. */
    fu_capability_risc5_pause_k = 1 << 4,
    /** `WRS.STO` sleeps the hart until a reservation breaks or a timeout. Needs the `Zawrs` extension. */
    fu_capability_risc5_wrs_k = 1 << 5,

    /** Own the raw OS thread handle instead of a `std::thread`. Built: `FU_WITH_OS_THREADS`. */
    fu_capability_os_threads_k = 1 << 6,
    /** Enumerate this machine's cores, compute domains, and memory domains. Built: `FU_WITH_TOPOLOGY`. */
    fu_capability_topology_k = 1 << 7,
    /** Bind a thread to a set of cores, choosing where it runs. Built: `FU_WITH_PLACE_THREADS_BY_AFFINITY`. */
    fu_capability_place_threads_by_affinity_k = 1 << 8,
    /** Steer a thread onto a class of core at creation. Built: `FU_WITH_PLACE_THREADS_BY_CORE_CLASS`. */
    fu_capability_place_threads_by_core_class_k = 1 << 9,
    /** Reclass a thread's scheduler to sleep or wake it. Built: `FU_WITH_RESCHEDULE_THREADS_BY_CLASS`. */
    fu_capability_reschedule_threads_by_class_k = 1 << 10,
    /** Place a buffer's pages on a chosen memory domain. Built: `FU_WITH_PLACE_MEMORY_ON_DOMAIN`. */
    fu_capability_place_memory_on_domain_k = 1 << 11,
    /** Place larger-than-base pages on a chosen memory domain. Built: `FU_WITH_PLACE_HUGE_PAGES_ON_DOMAIN`. */
    fu_capability_place_huge_pages_on_domain_k = 1 << 12,
    /** The kernel promotes base pages to huge pages on its own. A passive, runtime-only observation. */
    fu_capability_huge_transparent_pages_k = 1 << 13,
    /** The domain-aware `colocated_pool` and `distributed_pool` are compiled. Built:
     *  `FU_WITH_COLOCATE_POOLS_ON_DOMAIN`. */
    fu_capability_colocate_pools_on_domain_k = 1 << 14,
    /** `CLDEMOTE` moves a just-written line toward the shared LLC and retains it. Reporting-only:
     *  the emitter is chosen at compile time by `FU_WITH_DEMOTE_CACHE_LINES`, never dispatched. */
    fu_capability_x86_cldemote_k = 1 << 15,
    /** `DC CVAC` cleans a dirty line to the coherency point - AArch64's nearest demote. Set where
     *  EL0 execution is known-legal, i.e. Linux, which sets `SCTLR_EL1.UCI`. */
    fu_capability_arm64_dc_cvac_k = 1 << 16,
    /** The kernel enabled user-mode Zicbom cache-block management, attested through `hwprobe` -
     *  the hook for a future runtime-dispatched `cbo.clean`; nothing emits it yet. */
    fu_capability_risc5_zicbom_k = 1 << 17,

    /** Composite mask of every busy-wait waiter bit above, to enumerate the ones a machine offers. */
    fu_capability_any_yield_k = fu_capability_x86_pause_k | fu_capability_x86_tpause_k | fu_capability_arm64_yield_k |
                                fu_capability_arm64_wfet_k | fu_capability_risc5_pause_k | fu_capability_risc5_wrs_k,

    /** All-ones allow-mask: pass to `fu_pool_new` to disable capability filtering. */
    fu_capabilities_all_k = ~0,
} fu_capabilities_t;

/**
 *  @brief Which kernel facilities this build of ForkUnion was compiled to use.
 *  @sa fu_runtime_capabilities
 */
fu_capabilities_t fu_comptime_capabilities(void);

/**
 *  @brief Which features this machine turned out to offer, probing the CPU and the memory system.
 *  @sa fu_comptime_capabilities
 */
fu_capabilities_t fu_runtime_capabilities(void);

/**
 *  @brief Writes @p capabilities as a comma-separated name list such as "arm64_yield,arm64_wfet".
 *  @param[out] name_buffer Destination, always null-terminated; the list is truncated to fit.
 *  @param[in] name_buffer_length Size of @p name_buffer in bytes.
 *  @retval Bytes written, excluding the null terminator.
 */
size_t fu_name_capabilities(fu_capabilities_t capabilities, char *name_buffer, size_t name_buffer_length);

/**
 *  @brief Harvests the machine topology - cores, compute and memory domains - into a handle.
 *  @retval An opaque topology handle for the pool and metadata queries, or NULL on failure.
 *
 *  Build it once and thread it through `fu_pool_spawn` and the topology queries below.
 */
fu_topology_t fu_topology_new(void);

/**
 *  @brief Frees a topology handle from `fu_topology_new`.
 *  @param[in] topology Topology handle, may be NULL - a no-op.
 *
 *  Pools never retain the topology, so it may be freed as soon as the last spawn returns.
 */
void fu_topology_delete(fu_topology_t topology);

/**
 *  @brief Returns the number of logical cores in a given compute domain.
 *  @param[in] compute_domain_index Target compute domain, in [0, `fu_compute_domains_count()`).
 *  @retval Number of cores backing that compute domain, or 0 if the index is out of range.
 *
 *  Use this to size a per-compute-domain pool for @ref fu_pool_spawn_on, or to weight work across
 *  compute domains of differing core counts, such as performance versus efficiency cores.
 *  @sa `fu_compute_domains_count`, `fu_pool_spawn_on`.
 */
size_t fu_logical_cores_count_in(fu_topology_t, size_t compute_domain_index);

/**
 *  @brief The number of logical cores the OS exposes - hyper-threads and every core class included.
 *  @retval 0 if detection failed; else the count, suitable as the thread count for `fu_pool_spawn`.
 *  @sa `fu_logical_cores_count_in` for the per-compute-domain count.
 */
size_t fu_logical_cores_count(fu_topology_t);

/**
 *  @brief The number of compute domains: bindable clusters of same-QoS, co-located cores.
 *  @retval 0 if unsupported, 1 on a uniform machine, 2+ per NUMA node and QoS class such as P/E or big.LITTLE.
 *
 *  A compute domain is the unit a pool binds to and the index a worker callback receives. It is one
 *  axis of the topology; memory domains are the other, bridged by `fu_local_memory_of`.
 *  @sa `fu_logical_cores_count_in`, `fu_pool_spawn_on`, `fu_memory_domains_count`.
 */
size_t fu_compute_domains_count(fu_topology_t);

/**
 *  @brief Returns the performance level of a given compute domain.
 *  @param[in] compute_domain_index Target compute domain, in [0, `fu_compute_domains_count()`).
 *  @retval A level ordinal where @b higher @b is @b more @b performant, 0 being the most efficient; or 0
 *  if the index is out of range. Homogeneous systems report level 0 for every compute domain.
 *
 *  Distinguishes performance vs efficiency cores, as in Intel P/E or ARM big.LITTLE. @note The compute
 *  ordinal grows with performance, while the memory tier from @ref fu_fabric_memory_level_in grows as
 *  media slow down - both match their native hardware conventions, so they run opposite ways by design.
 *  @sa `fu_compute_levels_count`, `fu_compute_domains_count`.
 */
size_t fu_compute_level_in(fu_topology_t, size_t compute_domain_index);

/**
 *  @brief Returns the number of distinct compute performance levels across all compute domains.
 *  @retval 0 if unsupported, 1 on homogeneous cores, 2-3 with heterogeneous cores such as P/E or big.LITTLE.
 *  @note May be smaller than `fu_compute_domains_count()` - several domains can share one level,
 *  as when equally-fast cores are split across cache clusters, or across NUMA nodes.
 *  @sa `fu_compute_level_in`.
 */
size_t fu_compute_levels_count(fu_topology_t);

/**
 *  @brief Returns the relative throughput of @b one core in a given compute domain.
 *  @param[in] compute_domain_index Target compute domain, in [0, `fu_compute_domains_count()`).
 *  @retval A magnitude on the Linux `cpu_capacity` scale where 1024 is the fastest core present, or 0 when
 *  the index is out of range or the platform publishes no per-core throughput rating.
 *
 *  This is the number to weight work by; `fu_compute_level_in` is a dense ordinal and must never be
 *  divided by. When this reports 0, weigh compute domains by `fu_logical_cores_count_in` instead.
 *  @sa `fu_compute_level_in`, `fu_logical_cores_count_in`.
 */
size_t fu_compute_capacity_in(fu_topology_t, size_t compute_domain_index);

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
size_t fu_compute_cache_bytes_in(fu_topology_t, size_t compute_domain_index);

/**
 *  @brief Returns the number of memory domains - the distinct allocation targets.
 *  @retval 0 if unsupported, 1 on uniform-memory systems, 2+ on NUMA / tiered-memory systems.
 *
 *  A @b memory @b domain is a bank of memory with its own capacity and access cost. It is the unit
 *  the allocator targets. A memory domain may be @b cpuless, as with a CXL expander or GPU-attached
 *  HBM, and may be local to @b several compute domains, as when performance and efficiency cores
 *  share one DDR controller. Its performance - tiers, latencies, bandwidths, distances - is not
 *  the topology's to declare: a `fu_fabric_t` measures it in-process.
 *  @sa `fu_volume_ram_in`, `fu_local_memory_of`, `fu_allocate_on_domain_id`, `fu_fabric_harvest`.
 */
size_t fu_memory_domains_count(fu_topology_t);

/**
 *  @brief Returns the memory domain nearest to a given compute domain.
 *  @param[in] compute_domain_index Target compute domain, in [0, `fu_compute_domains_count()`).
 *  @retval The index of that compute domain's nearest memory domain, or 0 if
 *  the compute-domain index is out of range.
 *
 *  The convenience bridge for the common "run here, allocate near here" pattern: pass the result
 *  to `fu_allocate_on_domain_id`. For the full cost picture use `fu_fabric_memory_distance`.
 *  @sa `fu_fabric_memory_distance`, `fu_allocate_on_domain_id`.
 */
size_t fu_local_memory_of(fu_topology_t, size_t compute_domain_index);

/**
 *  @brief Returns the RAM volume in bytes of a given memory domain.
 *  @param[in] memory_domain_index Target memory domain, in [0, `fu_memory_domains_count()`).
 *  @retval Number of bytes of RAM in that memory domain, regardless of page size; 0 if out of range.
 *  @sa `fu_volume_ram`, `fu_allocate_on_domain_id`.
 */
size_t fu_volume_ram_in(fu_topology_t, size_t memory_domain_index);

/**
 *  @brief Returns the total RAM volume in bytes across all memory domains.
 *  @retval Number of bytes of RAM installed, regardless of page size.
 *  @sa `fu_volume_ram_in`.
 */
size_t fu_volume_ram(fu_topology_t);

/**
 *  @brief Returns the huge-page volume in bytes available in a given memory domain.
 *  @param[in] memory_domain_index Target memory domain, in [0, `fu_memory_domains_count()`).
 *  @retval Bytes backed by free huge pages in that memory domain; 0 if out of range or unavailable.
 *
 *  Huge pages reduce TLB pressure by mapping memory in larger units than the base page.
 *  @sa `fu_huge_pages_count_in`, `fu_allocate_at_least_on_domain_id`.
 */
size_t fu_volume_huge_pages_in(fu_topology_t, size_t memory_domain_index);

/**
 *  @brief Returns the total huge-page volume in bytes across all memory domains.
 *  @retval Number of bytes backed by free huge pages, or 0 if huge pages are unavailable.
 *  @sa `fu_volume_huge_pages_in`, `fu_huge_pages_count`.
 */
size_t fu_volume_huge_pages(fu_topology_t);

/**
 *  @brief Returns the number of free huge pages in a given memory domain.
 *  @param[in] memory_domain_index Target memory domain, in [0, `fu_memory_domains_count()`).
 *  @retval Count of free huge pages across all page sizes in that memory domain; 0 if
 *  out of range or unavailable.
 *  @sa `fu_volume_huge_pages_in`, `fu_huge_pages_count`.
 */
size_t fu_huge_pages_count_in(fu_topology_t, size_t memory_domain_index);

/**
 *  @brief Returns the total number of free huge pages across all memory domains.
 *  @retval Count of free huge pages of any size, or 0 if huge pages are unavailable.
 *  @sa `fu_volume_huge_pages`, `fu_huge_pages_count_in`.
 */
size_t fu_huge_pages_count(fu_topology_t);

#pragma endregion Metadata

#pragma region Memory

/**
 *  @brief Resolves a memory domain's dense index to the OS id the allocators take.
 *  @param[in] topology Machine topology from `fu_topology_new`.
 *  @param[in] memory_domain_index Target memory domain, in [0, `fu_memory_domains_count()`).
 *  @retval The OS memory-domain id - a NUMA node - to hand to `fu_allocate_on_domain_id`, or -1 if out of range.
 *
 *  The allocators key off the OS id rather than the topology, so an allocation can outlive the handle.
 *  Look the id up once - typically near a compute domain via `fu_local_memory_of` - then allocate and
 *  free with it alone.
 *  @sa `fu_allocate_on_domain_id`, `fu_local_memory_of`, `fu_memory_domains_count`.
 */
fu_memory_domain_id_t fu_memory_domain_id_at_index(fu_topology_t topology, size_t memory_domain_index);

/**
 *  @brief Allocates memory in @p memory_domain_id with the largest suitable page size.
 *  @param[in] memory_domain_id Target memory domain, from `fu_memory_domain_id_at_index`.
 *  @param[in] minimum_bytes Minimum number of bytes to allocate, must be > 0.
 *  @param[out] allocated_bytes Receives the actual allocation size - at least @p minimum_bytes - must not be NULL.
 *  @param[out] bytes_per_page Receives the page size used for the allocation, must not be NULL.
 *  @retval Pointer to allocated memory, or NULL if allocation failed.
 *  @note This API is @b thread-safe and can be called from any thread.
 *  @note The pointer is aligned to at least the cache-line default, so over-aligned element types up
 *  to that width need no caller-side padding.
 *
 *  Prefers the largest available huge-page size to minimize TLB pressure; the actual size may exceed
 *  the request due to page alignment, so always read @p allocated_bytes. No topology handle is needed -
 *  the id fully identifies the domain.
 *  @code{.c}
 *  fu_memory_domain_id_t domain = fu_memory_domain_id_at_index(topology, fu_local_memory_of(topology, compute_domain));
 *  void *pointer = NULL; size_t actual_bytes = 0, page = 0;
 *  if ((pointer = fu_allocate_at_least_on_domain_id(domain, 1u << 20, &actual_bytes, &page)))
 *      fu_free_on_domain_id(domain, pointer, actual_bytes);
 *  @endcode
 *  @sa `fu_free_on_domain_id`, `fu_memory_domain_id_at_index`.
 */
void *fu_allocate_at_least_on_domain_id(fu_memory_domain_id_t memory_domain_id, size_t minimum_bytes,
                                        size_t *allocated_bytes, size_t *bytes_per_page);

/**
 *  @brief Allocates exactly @p bytes in @p memory_domain_id.
 *  @param[in] memory_domain_id Target memory domain, from `fu_memory_domain_id_at_index`.
 *  @param[in] bytes Number of bytes to allocate, must be > 0.
 *  @retval Pointer to allocated memory, or NULL if allocation failed.
 *  @note This API is @b thread-safe. Unlike `fu_allocate_at_least_on_domain_id`, it does not over-allocate for
 *  page optimization - use it for standard-allocator compatibility.
 *  @note The pointer is aligned to at least the cache-line default, matching `fu_allocate_at_least_on_domain_id`.
 *  @sa `fu_free_on_domain_id`, `fu_allocate_at_least_on_domain_id`.
 */
void *fu_allocate_on_domain_id(fu_memory_domain_id_t memory_domain_id, size_t bytes);

/**
 *  @brief Releases memory allocated in @p memory_domain_id.
 *  @param[in] memory_domain_id The memory domain the memory was allocated in.
 *  @param[in] pointer Pointer to the memory to release, must not be NULL.
 *  @param[in] bytes Number of bytes to release; must match the `allocated_bytes` from allocation.
 *  @note This API is @b thread-safe. A mismatched @p bytes is undefined behavior.
 *  @sa `fu_allocate_at_least_on_domain_id`, `fu_allocate_on_domain_id`.
 */
void fu_free_on_domain_id(fu_memory_domain_id_t memory_domain_id, void *pointer, size_t bytes);

/**
 *  @brief Allocates one @b symmetric mapping - `bytes_per_domain` striped across every memory domain.
 *  @param[in] topology Machine topology from `fu_topology_new`.
 *  @param[in] bytes_per_domain Minimum usable bytes per domain slice; the slice stride is page-rounded up from it.
 *  @param[out] stride_bytes Receives the page-aligned byte distance between slices, must not be NULL.
 *  @param[out] memory_domains_count Receives the number of domain slices, must not be NULL.
 *  @param[out] total_bytes Receives the whole mapping size to hand back to `fu_free_symmetric`, must not be NULL.
 *  @param[out] bytes_per_page Receives the page size used, may be NULL.
 *  @retval Base pointer of the mapping, or NULL if allocation failed.
 *  @note This API is @b thread-safe. Slice @b `d` begins at `base + d * *stride_bytes` and is bound to its
 *  own memory domain; a machine with no NUMA API collapses the mapping to a single heap-backed slice.
 *
 *  Prefers the largest available huge-page size to minimize TLB pressure, so the stride may exceed
 *  @p bytes_per_domain - always read @p stride_bytes.
 *  @code{.c}
 *  size_t stride = 0, domains = 0, total = 0, page = 0;
 *  void *base = fu_allocate_symmetric(topology, 1u << 20, &stride, &domains, &total, &page);
 *  if (base) fu_free_symmetric(base, total);
 *  @endcode
 *  @sa `fu_free_symmetric`, `fu_local_memory_of`, `fu_memory_domains_count`.
 */
void *fu_allocate_symmetric(fu_topology_t topology, size_t bytes_per_domain, size_t *stride_bytes,
                            size_t *memory_domains_count, size_t *total_bytes, size_t *bytes_per_page);

/**
 *  @brief Releases a symmetric mapping from `fu_allocate_symmetric`.
 *  @param[in] base Base pointer returned by `fu_allocate_symmetric`, must not be NULL.
 *  @param[in] total_bytes The `total_bytes` the allocation reported; a mismatch is undefined behavior.
 *  @note This API is @b thread-safe. The size is required - the Linux backing unmaps the whole range.
 *  @sa `fu_allocate_symmetric`.
 */
void fu_free_symmetric(void *base, size_t total_bytes);

#pragma endregion Memory

#pragma region Lifetime

/**
 *  @brief Creates an empty pool constrained to an allow-mask of capabilities.
 *  @param[in] name Optional pool name, may be NULL.
 *  @param[in] allowed Allow-mask of `fu_capabilities_t`; pass `fu_capabilities_all_k` for no filtering.
 *  @retval An opaque pool handle, or NULL on allocation failure.
 *  @note Thread-safe.
 *
 *  The waiter, and whether a spawn spans domains, are chosen from `fu_runtime_capabilities() & allowed`,
 *  so clearing bits forces a lesser waiter or the flat non-NUMA pool - handy for benchmarking. Creation
 *  takes no topology: @b placement is decided at spawn, where you hand over the `fu_topology_t`. The
 *  pool starts empty; call `fu_pool_spawn` before use.
 *  @sa `fu_pool_spawn`, `fu_pool_capabilities`, `fu_pool_delete`.
 */
fu_pool_t fu_pool_new(char const *name, fu_capabilities_t allowed);

/**
 *  @brief Destroys a pool and frees its resources; the handle is invalid afterward.
 *  @param[in] pool Pool handle, may be NULL - a no-op. Pending tasks must have completed.
 *  @note Must not run concurrently with other operations on @p pool.
 *  @sa `fu_pool_terminate` to stop workers but keep the handle.
 */
void fu_pool_delete(fu_pool_t pool);

/**
 *  @brief The capabilities an initialized pool actually uses - a subset of comptime & runtime.
 *  @param[in] pool Pool handle, must not be NULL.
 */
fu_capabilities_t fu_pool_capabilities(fu_pool_t pool);

/**
 *  @brief Spawns @p threads workers across the @b whole machine, readying the pool for dispatch.
 *  @param[in] topology Machine topology from `fu_topology_new`; read only during this call, so it may
 *             be freed as soon as this returns.
 *  @param[in] pool Pool handle, must not be NULL.
 *  @param[in] threads Worker count, must be > 0. For an inclusive pool the caller counts as one of them.
 *  @param[in] exclusivity Whether the calling thread also executes tasks.
 *  @retval 1 on success, 0 on failure.
 *  @note Not thread-safe; call once per pool, or again after `fu_pool_terminate`.
 *
 *  Covers every compute domain. If a prior @ref fu_pool_spawn_on left the pool pinned to a single
 *  domain, this rebuilds it as a whole-machine pool before spawning.
 *  @sa `fu_pool_spawn_on` to pin to one compute domain, `fu_logical_cores_count` for the count.
 */
fu_bool_t fu_pool_spawn(fu_topology_t topology, fu_pool_t pool, size_t threads, fu_caller_exclusivity_t exclusivity);

/**
 *  @brief Spawns @p threads workers pinned to a single compute domain.
 *  @param[in] topology Machine topology from `fu_topology_new`; read only during this call.
 *  @param[in] pool Pool handle, must not be NULL.
 *  @param[in] compute_domain_index Target compute domain, in [0, `fu_compute_domains_count()`).
 *  @param[in] threads Worker count, must be > 0.
 *  @param[in] exclusivity Whether the calling thread also executes tasks.
 *  @retval 1 on success; 0 on failure or an out-of-range domain.
 *  @note Not thread-safe; call once per pool.
 *
 *  Placement lives here, not in creation: @ref fu_pool_new allocates the handle, and this binds it to
 *  a single compute domain, so its workers and their memory-domain-local allocations stay on that
 *  domain. Spawn one pool per compute domain and coordinate them with the generation-token API;
 *  @ref fu_pool_spawn instead spans @b all compute domains. Without NUMA, only compute domain 0 is valid.
 *  @sa `fu_pool_spawn`, `fu_compute_domains_count`, `fu_logical_cores_count_in`.
 */
fu_bool_t fu_pool_spawn_on(fu_topology_t topology, fu_pool_t pool, size_t compute_domain_index, size_t threads,
                           fu_caller_exclusivity_t exclusivity);

/**
 *  @brief Whether the calling thread executes a slice of each dispatch.
 *  @param[in] pool Pool handle, must not be NULL and initialized.
 *  @retval `fu_caller_inclusive_k` if the caller runs a slice, `fu_caller_exclusive_k` if it only coordinates.
 *  @note Not synchronized. Reflects the most recent `fu_pool_spawn`, so it survives a re-spawn.
 *
 *  On inclusive pools the caller's slice runs only inside `fu_pool_unsafe_join`, so
 *  `fu_pool_is_complete` cannot be reached by polling alone.
 *  @sa `fu_pool_spawn`, `fu_pool_is_complete`.
 */
fu_caller_exclusivity_t fu_pool_caller_exclusivity(fu_pool_t pool);

/**
 *  @brief The number of compute domains the pool's workers span.
 *  @param[in] pool Pool handle, must not be NULL.
 *  @retval 0 if uninitialized, 1 without NUMA/QoS heterogeneity, 2+ across NUMA nodes or QoS levels.
 *  @note Not synchronized.
 *  @sa `fu_pool_threads_count_in`.
 */
size_t fu_pool_compute_domains_count(fu_pool_t pool);

/**
 *  @brief The worker count in one compute domain of the pool.
 *  @param[in] pool Pool handle, must not be NULL.
 *  @param[in] compute_domain_index In [0, `fu_pool_compute_domains_count(pool)`); not bounds-checked.
 *  @retval Threads in that domain, or 0 if uninitialized.
 *  @note Not synchronized.
 *  @sa `fu_pool_compute_domains_count`.
 */
size_t fu_pool_threads_count_in(fu_pool_t pool, size_t compute_domain_index);

/**
 *  @brief The total worker count, including the caller on an inclusive pool.
 *  @param[in] pool Pool handle, must not be NULL.
 *  @retval 0 if uninitialized, else the count from `fu_pool_spawn`.
 *  @note Not synchronized.
 */
size_t fu_pool_threads_count(fu_pool_t pool);

/**
 *  @brief Converts a global thread index to a local thread index within a compute_domain.
 *  @param[in] pool Thread pool handle, must not be NULL.
 *  @param[in] global_thread_index The global thread index to convert.
 *  @param[in] compute_domain_index Index of the compute_domain, must be < `fu_pool_compute_domains_count(pool)`.
 *  @retval Local thread index within the specified compute_domain.
 */
size_t fu_pool_locate_thread_in(fu_pool_t pool, size_t global_thread_index, size_t compute_domain_index);

/**
 *  @brief Parks idle workers in a low-power sleep, re-checking for work every @p micros.
 *  @param[in] pool Pool handle, must not be NULL.
 *  @param[in] micros Wake-up poll interval in microseconds, must be > 0.
 *  @note Not thread-safe; call between task batches. The next dispatch wakes the workers.
 *
 *  Trades up to @p micros of startup latency for lower power draw during long idle periods; on Linux
 *  it also de-prioritizes the sleeping threads with the scheduler.
 */
void fu_pool_sleep(fu_pool_t pool, size_t micros);

/**
 *  @brief Stops the workers and clears the pool, keeping the handle for a re-spawn.
 *  @param[in] pool Pool handle, must not be NULL.
 *  @note Not thread-safe; call only when no dispatch is in flight, never as a sync point.
 *
 *  Joins every worker and releases their resources; call `fu_pool_spawn` again, e.g. to change the
 *  thread count. Use `fu_pool_delete` for permanent teardown.
 *  @sa `fu_pool_spawn`, `fu_pool_delete`.
 */
void fu_pool_terminate(fu_pool_t pool);

#pragma endregion Lifetime

#pragma region Fabric

/**
 *  @brief Creates an empty, unharvested memory-fabric handle.
 *  @retval An opaque fabric handle, or NULL on allocation failure.
 *
 *  Completes the library's pipeline: build a `fu_topology_t` first, spawn a `fu_pool_t` on it,
 *  then harvest the fabric through that pool's pinned workers with @ref fu_fabric_harvest. Before
 *  a harvest every query on the handle answers 0, and `fu_fabric_memory_levels_count` answers 1.
 *  @sa `fu_fabric_harvest`, `fu_fabric_delete`.
 */
fu_fabric_t fu_fabric_new(void);

/**
 *  @brief Destroys a fabric and frees its observations; the handle is invalid afterward.
 *  @param[in] fabric Fabric handle, may be NULL - a no-op.
 *  @note Must not run concurrently with other operations on @p fabric.
 */
void fu_fabric_delete(fu_fabric_t fabric);

/**
 *  @brief Measures the memory fabric through the pool's pinned workers, rebuilding @p fabric.
 *  @param[in] topology Read only; may be freed once this returns - the fabric snapshots what it needs.
 *  @param[in] pool Pool handle, must not be NULL and spawned across the machine via `fu_pool_spawn`.
 *  @param[out] fabric Receives the observations, replacing any previous harvest; must not be NULL,
 *              and a failed harvest leaves it empty, never half-written.
 *  @retval 1 on success; 0 on allocation failure or a pool that spans no memory domains - flat,
 *  pinned to a single compute domain, or terminated.
 *  @note Not thread-safe: dispatches on the pool and rebuilds the fabric, so call it between task
 *        batches and do not query @p fabric concurrently. Expect seconds of runtime on large fabrics.
 *
 *  Pointer-chases every reachable edge for latency and streams it with each initiator domain's
 *  full worker set for bandwidth - in-process, with no ACPI tables or OS-specific interfaces.
 *  @b Cpuless memory domains, like CXL expanders, stay unwalked: portable first-touch cannot
 *  place pages there, so their edges answer 0 and they share one tier past the slowest observed.
 */
fu_bool_t fu_fabric_harvest(fu_topology_t topology, fu_pool_t pool, fu_fabric_t fabric);

/**
 *  @brief Returns the measured read latency from a compute domain to a memory domain.
 *  @param[in] compute_domain_index Initiator compute domain, in [0, `fu_compute_domains_count()`).
 *  @param[in] memory_domain_index Target memory domain, in [0, `fu_memory_domains_count()`).
 *  @retval Dependent-load latency in nanoseconds - the best recording of the edge; 0 before a
 *  harvest, for an edge no worker could reach, or an out-of-range index.
 */
size_t fu_fabric_memory_latency(fu_fabric_t, size_t compute_domain_index, size_t memory_domain_index);

/**
 *  @brief Returns the measured read bandwidth from a compute domain to a memory domain.
 *  @param[in] compute_domain_index Initiator compute domain, in [0, `fu_compute_domains_count()`).
 *  @param[in] memory_domain_index Target memory domain, in [0, `fu_memory_domains_count()`).
 *  @retval Saturated read bandwidth in MB/s, streamed by all the initiator domain's workers at
 *  once - the best recording of the edge; 0 before a harvest, for an edge no worker could reach,
 *  or an out-of-range index.
 */
size_t fu_fabric_memory_bandwidth(fu_fabric_t, size_t compute_domain_index, size_t memory_domain_index);

/**
 *  @brief Returns the relative access distance from a compute domain to a memory domain.
 *  @param[in] compute_domain_index Initiator compute domain, in [0, `fu_compute_domains_count()`).
 *  @param[in] memory_domain_index Target memory domain, in [0, `fu_memory_domains_count()`).
 *  @retval A relative distance where @b 10 means local per the SLIT convention; larger is farther;
 *  0 means an out-of-range index or an unharvested fabric.
 *
 *  The measured latency ratio to the initiator's local domain, clamped so the local domain always
 *  carries the row's minimum; unwalked edges fall back to the 10-local / 20-remote convention.
 */
size_t fu_fabric_memory_distance(fu_fabric_t, size_t compute_domain_index, size_t memory_domain_index);

/**
 *  @brief Returns the derived speed class of a given memory domain, independent of any initiator.
 *  @param[in] memory_domain_index Target memory domain, in [0, `fu_memory_domains_count()`).
 *  @retval A tier ordinal where @b lower @b is @b faster, 0 being the fastest such as HBM; or 0
 *  if the index is out of range or the fabric is unharvested.
 *
 *  Keyed by the best bandwidth any initiator sustains to the pool, ties split by the best
 *  latency, so HBM < DDR < CXL/PMEM; boundaries are measurement-derived and can shift between
 *  harvests. @note Runs opposite to `fu_compute_level_in`.
 */
size_t fu_fabric_memory_level_in(fu_fabric_t, size_t memory_domain_index);

/**
 *  @brief Returns the number of distinct derived memory tiers across all memory domains.
 *  @retval 1 on single-tier systems and before a harvest, 2+ when HBM / DDR / CXL are mixed.
 *  @note The memory-axis twin of `fu_compute_levels_count`; several memory domains may share a tier.
 */
size_t fu_fabric_memory_levels_count(fu_fabric_t);

#pragma endregion Fabric

#pragma region Primary API

/**
 *  @brief Runs @p callback once on every worker, blocking until all return - OpenMP's @b `parallel`.
 *  @param[in] pool Pool handle, must not be NULL and initialized.
 *  @param[in] callback Runs once per worker, must not be NULL.
 *  @param[in] context Shared context passed to every call, may be NULL.
 *  @note Returns only after every worker completes; no extra synchronization is needed.
 *
 *  The callback receives @p context, its `thread` index in [0, threads_count), and the
 *  `compute_domain` - the NUMA node and QoS level - it runs on.
 *  @code{.c}
 *  void hello(void *ctx, size_t thread, size_t compute_domain) {
 *      printf("Hello from thread %zu on compute_domain %zu\n", thread, compute_domain);
 *  }
 *  fu_pool_for_threads(pool, hello, NULL);
 *  @endcode
 *  @sa `fu_pool_unsafe_for_threads` for the non-blocking form.
 */
void fu_pool_for_threads(fu_pool_t pool, fu_for_threads_t callback, fu_lambda_context_t context);

/**
 *  @brief Splits [0, @p n) into @b contiguous slices, one per worker, and blocks until all finish.
 *  @param[in] pool Pool handle, must not be NULL and initialized.
 *  @param[in] n Total tasks, may be 0 - a no-op.
 *  @param[in] callback Runs once per slice, must not be NULL if @p n > 0.
 *  @param[in] context Shared context, may be NULL.
 *
 *  One contiguous range per worker - ideal for @b vectorized or per-slice-setup work where sequential
 *  access matters, since each worker touches its data in order and only sets up once. Idle workers
 *  receive an empty slice with `count` == 0. The callback receives @p context, the slice's `first`
 *  index and its `count` of tasks, its `thread` index, and its `compute_domain`.
 *  @code{.c}
 *  void normalize(void *array, size_t first, size_t count, size_t thread, size_t compute_domain) {
 *      float *data = (float *)array;
 *      for (size_t i = first; i != first + count; ++i) data[i] = sqrtf(data[i]); // One sequential sweep
 *  }
 *  fu_pool_for_slices(pool, length, normalize, values);
 *  @endcode
 *  @sa `fu_pool_for_n` for per-index dispatch.
 */
void fu_pool_for_slices(fu_pool_t pool, size_t n, fu_for_slices_t callback, fu_lambda_context_t context);

/**
 *  @brief Runs @p callback for each of @p n @b similar-cost tasks, blocking until done.
 *  @param[in] pool Pool handle, must not be NULL and initialized.
 *  @param[in] n Task count, may be 0 - a no-op.
 *  @param[in] callback Runs once per task, must not be NULL if @p n > 0.
 *  @param[in] context Shared context, may be NULL.
 *
 *  Tasks split into equal @b contiguous chunks per worker, keeping coordination near zero - the right
 *  choice when every task costs about the same, so a static split stays balanced. The callback
 *  receives @p context, the `task` index in [0, @p n), its `thread` index, and its `compute_domain`.
 *  @code{.c}
 *  void double_each(void *array, size_t i, size_t thread, size_t compute_domain) {
 *      int *data = (int *)array;
 *      data[i] = data[i] * 2; // Uniform O(1) work per index
 *  }
 *  fu_pool_for_n(pool, length, double_each, array);
 *  @endcode
 *  @sa `fu_pool_for_n_dynamic` for uneven workloads, `fu_pool_for_slices` for range callbacks.
 */
void fu_pool_for_n(fu_pool_t pool, size_t n, fu_for_prongs_t callback, fu_lambda_context_t context);

/**
 *  @brief Runs @p callback for each of @p n @b uneven-cost tasks via work-stealing, blocking until done.
 *  @param[in] pool Pool handle, must not be NULL and initialized.
 *  @param[in] n Task count, may be 0 - a no-op.
 *  @param[in] callback Runs once per task, must not be NULL if @p n > 0.
 *  @param[in] context Shared context, may be NULL.
 *
 *  Each worker starts with one task, then claims more from a shared cursor as it finishes, so fast
 *  workers absorb the slack and one slow task never stalls the rest - the right choice when task cost
 *  is unpredictable or data-dependent. Same callback shape as @ref fu_pool_for_n, but task order is
 *  @b not deterministic.
 *  @code{.c}
 *  void variable_work(void *ctx, size_t task, size_t thread, size_t compute_domain) {
 *      complex_computation(task); // May take 1us or 1ms
 *  }
 *  fu_pool_for_n_dynamic(pool, count, variable_work, ctx);
 *  @endcode
 *  @sa `fu_pool_for_n` for balanced workloads.
 */
void fu_pool_for_n_dynamic(fu_pool_t pool, size_t n, fu_for_prongs_t callback, fu_lambda_context_t context);

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
 *  @brief Dispatches @p callback on every worker @b without blocking; returns a generation token.
 *  @param[in] pool Pool handle, must not be NULL and initialized.
 *  @param[in] callback Runs on each thread, must not be NULL.
 *  @param[in] context Shared context, must stay valid until the join, may be NULL.
 *  @return A generation token to pass to `fu_pool_unsafe_join`.
 *  @note Returns immediately. Call no other pool operation until you join this generation.
 *
 *  On @b inclusive pools the caller's own slice runs inside `fu_pool_unsafe_join`, so the pool cannot
 *  complete until the caller joins.
 *  @code{.c}
 *  fu_generation_t generation = fu_pool_unsafe_for_threads(pool, callback, context);
 *  prepare_next_batch();
 *  fu_pool_unsafe_join(pool, generation);
 *  @endcode
 *  @sa `fu_pool_unsafe_join`, `fu_pool_for_threads` for the blocking form.
 */
fu_generation_t fu_pool_unsafe_for_threads(fu_pool_t pool, fu_for_threads_t callback, fu_lambda_context_t context);

/**
 *  @brief Whether @p generation has finished - a @b non-blocking poll.
 *  @param[in] pool Pool handle, must not be NULL and initialized.
 *  @param[in] generation Token from `fu_pool_unsafe_for_threads`.
 *  @return Non-zero once complete, which also publishes every contributor's writes.
 *  @note On @b inclusive pools this turns non-zero only after `fu_pool_unsafe_join` runs the caller's
 *  slice, so poll-then-join is for @b exclusive pools only.
 */
fu_bool_t fu_pool_is_complete(fu_pool_t pool, fu_generation_t generation);

/**
 *  @brief Blocks until @p generation completes, running the caller's slice on an @b inclusive pool.
 *  @param[in] pool Pool handle, must not be NULL and initialized.
 *  @param[in] generation Token from `fu_pool_unsafe_for_threads`.
 *
 *  The synchronization point for the non-blocking API: on return every contributor is done, their
 *  writes are visible, and the pool is ready for the next dispatch. @b Idempotent on a stale or
 *  already-joined generation.
 *  @code{.c}
 *  fu_generation_t generation = fu_pool_unsafe_for_threads(pool, callback, context);
 *  while (!fu_pool_is_complete(pool, generation)) { do_other_work(); } // ! Exclusive pools only
 *  fu_pool_unsafe_join(pool, generation);
 *  @endcode
 *  @sa `fu_pool_unsafe_for_threads`, `fu_pool_for_threads`.
 */
void fu_pool_unsafe_join(fu_pool_t pool, fu_generation_t generation);

#pragma endregion Flexible API

#ifdef __cplusplus
} // extern "C"
#endif
