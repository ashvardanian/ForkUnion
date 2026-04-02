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
 *  void print_arg(void *context_punned, size_t task_index, size_t thread_index, size_t colocation_index) {
 *      print_args_context_t *context = (print_args_context_t *)context_punned;
 *      printf(
 *          "Printing argument # %zu from thread # %zu at colocation # %zu: %s\n",
 *          task_index, context->argc, thread_index, colocation_index, context->argv[task_index]);
 *  }
 *
 *  int main(int argc, char *argv[]) {
 *      char const *caps = fu_capabilities_string();
 *      if (!caps) return EXIT_FAILURE; // ! Thread pool is not supported
 *      printf("ForkUnion capabilities: %s\n", caps);
 *
 *      fu_pool_t *pool = fu_pool_new("forkunion_demo");
 *      if (!pool) return EXIT_FAILURE; // ! Failed to create a thread pool
 *
 *      size_t threads = fu_count_logical_cores();
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
 *  Under the hood, the `fu_pool_t` maps to a `basic_pool` or `linux_distributed_pool`.
 *  For advanced usage, prefer the core C++ library.
 *
 *  ------------------------------------------------------------------------------------------------
 *
 *  The next layer of logic is for basic index-addressable tasks. It includes basic parallel loops:
 *
 *  - `fu_pool_for_n` - for iterating over a range of similar duration tasks, addressable by an index.
 *  - `fu_pool_for_n_dynamic` - for unevenly distributed tasks, where each task may take a different time.
 *  - `fu_pool_for_slices` - for iterating over a range of similar duration tasks, addressable by a slice.
 *
 *  ------------------------------------------------------------------------------------------------
 *
 *  On Linux, when NUMA and PThreads are available, the library can also leverage @b NUMA-aware
 *  memory allocations and pin threads to specific physical cores to increase memory locality.
 *  It should reduce memory access latency by around 35% on average, compared to remote accesses.
 *  @sa `fu_count_numa_nodes`, `fu_allocate_at_least`, `fu_free`.
 *
 *  On heterogeneous chips, cores with a different @b "Quality-of-Service" (QoS) may be combined.
 *  A typical example is laptop/desktop chips, having 1 NUMA node, but 3 tiers of CPU cores:
 *  performance, efficiency, and power-saving cores. Each group will have vastly different speed,
 *  so considering them equal in tasks scheduling is a bad idea... and separating them automatically
 *  isn't feasible either. It's up to the user to isolate those groups into individual pools.
 *  @sa `fu_count_quality_levels`
 *
 *  On x86, Arm, and RISC-V architectures, depending on the CPU features available, the library also
 *  exposes cheaper @b "busy-waiting" mechanisms, such as `tpause`, `wfet`, & `yield` instructions.
 *  @sa `fu_capabilities_string`
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

int fu_version_major(void); // ? Returns the major version of the ForkUnion library
int fu_version_minor(void); // ? Returns the minor version of the ForkUnion library
int fu_version_patch(void); // ? Returns the patch version of the ForkUnion library
int fu_enabled_numa(void);  // ? Checks if the library was compiled with NUMA support

#pragma region - Types

typedef int fu_bool_t;             // ? A simple boolean type, 0 for false, 1 for true
typedef void *fu_pool_t;           // ? A simple cross-platform opaque wrapper
typedef void *fu_lambda_context_t; // ? Type-punned pointer to the user-defined context

/**
 *  @brief Callback type for thread-level operations.
 *  @param[in] context Type-punned pointer to user-defined context data.
 *  @param[in] thread The thread index in [0, threads_count).
 *  @param[in] colocation The colocation index (NUMA node & QoS level) in [0, colocations_count).
 */
typedef void (*fu_for_threads_t)(fu_lambda_context_t context, size_t thread, size_t colocation);

/**
 *  @brief Callback type for task-level operations receiving individual indices.
 *  @param[in] context Type-punned pointer to user-defined context data.
 *  @param[in] task The task index in [0, n).
 *  @param[in] thread The thread index in [0, threads_count).
 *  @param[in] colocation The colocation index (NUMA node & QoS level) in [0, colocations_count).
 */
typedef void (*fu_for_prongs_t)(fu_lambda_context_t context, size_t task, size_t thread, size_t colocation);

/**
 *  @brief Callback type for slice-level operations receiving ranges of tasks.
 *  @param[in] context Type-punned pointer to user-defined context data.
 *  @param[in] first The first task index in the slice.
 *  @param[in] count The number of tasks in the slice.
 *  @param[in] thread The thread index in [0, threads_count).
 *  @param[in] colocation The colocation index (NUMA node & QoS level) in [0, colocations_count).
 */
typedef void (*fu_for_slices_t)(fu_lambda_context_t context, size_t first, size_t count, size_t thread,
                                size_t colocation);

/**
 *  @brief Defines the in- and exclusivity of the calling thread for the executing task.
 *  @sa `fu_caller_inclusive_k` and `fu_caller_exclusive_k`
 *
 *  This enum affects how the join is performed. If the caller is inclusive, 1/Nth of the call
 *  will be executed by the calling thread (as opposed to workers) and the join will happen
 *  inside of the calling scope.
 */
typedef enum fu_caller_exclusivity_t {
    fu_caller_inclusive_k, // ? The calling thread participates in the workload
    fu_caller_exclusive_k, // ? The calling thread only coordinates, doesn't execute tasks
} fu_caller_exclusivity_t;

#pragma endregion - Types

#pragma region - Metadata

/**
 *  @brief Describes available OS+CPU capabilities used by the thread pools.
 *  @retval `NULL`, if the thread pool is not supported on the current platform.
 *  @retval "serial" for the default C++ STL-powered thread pool without NUMA awareness.
 *  @retval "numa" for the NUMA-aware thread pool on Linux-based systems.
 *  @retval "numa+x86_pause" for the NUMA-aware pool with `pause` instruction on x86.
 *  @retval "numa+arm64_yield" for the NUMA-aware pool with `yield` instruction on AArch64.
 *  @retval "numa+x86_tpause" for the NUMA-aware pool with `tpause` instruction with "waitpkg" CPU feature.
 *  @retval "numa+arm64_wfet" for the NUMA-aware pool with `wfet` instruction on AArch64.
 *  @retval "numa+risc5_pause" for the NUMA-aware pool with `pause` instruction on RISC-V.
 *
 *  The string describes both the memory topology awareness and the CPU-specific optimizations
 *  available for busy-waiting. These capabilities directly affect performance characteristics:
 *  - Basic "serial" pools are suitable for single-NUMA-node systems or when portability is key.
 *  - "numa" pools reduce memory access latency by ~35% on multi-socket servers.
 *  - CPU-specific extensions like "tpause" and "wfet" reduce power consumption during busy-waits.
 */
char const *fu_capabilities_string(void);

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
 *  - CPU-bound tasks: use `fu_count_logical_cores()`
 *  - Memory-bound tasks: consider `fu_count_numa_nodes() * cores_per_node`
 *  - I/O-bound tasks: consider 2-4x `fu_count_logical_cores()`
 */
size_t fu_count_logical_cores(void);

/**
 *  @brief Describes the maximum number of individually addressable thread groups.
 *  @retval 0 if the thread pool is not supported on the current platform.
 *  @retval 1 on most desktop, laptop, or IoT platforms with unified memory.
 *  @retval 2-8 on typical dual-socket servers or heterogeneous mobile chips.
 *  @retval 4-32 is a typical range on high-end cloud servers with multiple sockets.
 *
 *  A "colocation" represents a group of threads that share the same:
 *  - NUMA memory domain (fast local memory access)
 *  - Quality-of-Service level (performance vs efficiency cores)
 *  - Cache hierarchy (L3 cache sharing)
 *
 *  The total may be as large as the product of NUMA nodes and QoS levels.
 *  Understanding colocations helps optimize memory allocation and task distribution.
 *  @sa `fu_count_numa_nodes`, `fu_count_quality_levels`, `fu_allocate_at_least`.
 */
size_t fu_count_colocations(void);

/**
 *  @brief Describes the number of NUMA (Non-Uniform Memory Access) nodes.
 *  @retval 0 if NUMA is not supported or detection failed.
 *  @retval 1 on systems with uniform memory access (UMA).
 *  @retval 2-8 on typical multi-socket servers.
 *  @retval 8+ on high-end systems with complex topologies.
 *
 *  NUMA nodes represent distinct memory domains with different access latencies.
 *  Memory allocated on the local NUMA node is typically 2-3x faster to access
 *  than remote NUMA node memory. For optimal performance, allocate memory and
 *  schedule tasks on the same NUMA node.
 *  @sa `fu_allocate_at_least`, `fu_free`.
 */
size_t fu_count_numa_nodes(void);

/**
 *  @brief Describes the number of distinct Quality-of-Service levels.
 *  @retval 0 if QoS detection is not supported.
 *  @retval 1 on systems with homogeneous cores.
 *  @retval 2-3 on systems with heterogeneous cores (e.g., ARM big.LITTLE, Intel P+E cores).
 *
 *  Different QoS levels may have vastly different performance characteristics.
 *  Consider creating separate thread pools for different workload types.
 */
size_t fu_count_quality_levels(void);

/**
 *  @brief Returns the total volume of any pages (huge or regular) available across all NUMA nodes.
 *  @retval Number of bytes of memory pages available across all NUMA nodes.
 */
size_t fu_volume_any_pages(void);

/**
 *  @brief Returns the volume of any pages (huge or regular) available on the specified NUMA node.
 *  @param[in] numa_node_index The index of the NUMA node to query, in [0, numa_nodes_count).
 *  @retval 0 if the NUMA node index is invalid or if no memory is available.
 *  @retval Number of bytes of memory pages available on the specified NUMA node.
 *
 *  This function queries the operating system for the total amount of memory pages
 *  (both huge pages and regular pages) available for allocation on the specified NUMA node.
 *  This is useful for determining how much memory can be allocated for vector storage
 *  based on a percentage of available memory.
 */
size_t fu_volume_any_pages_in(size_t numa_node_index);

/**
 *  @brief Describes the number of different huge page sizes supported.
 *  @param[in] numa_node_index The index of the NUMA node to allocate memory on, in [0, numa_nodes_count).
 *  @retval 0 if huge pages are not supported or not available.
 *  @retval 1-4 on systems with huge page support (typically 2MB, 1GB sizes).
 *
 *  Huge pages reduce TLB (Translation Lookaside Buffer) pressure by using larger
 *  page sizes than the standard 4KB. This can significantly improve performance
 *  for memory-intensive applications by reducing page table overhead.
 *
 *  Common huge page sizes:
 *  - 2MB pages: Standard huge pages on x86-64 and AArch64
 *  - 1GB pages: Gigantic pages for very large allocations
 *  - 16KB/64KB: Base page sizes on some ARM configurations
 *  @sa `fu_allocate_at_least` for NUMA-aware allocation with huge page support.
 */
size_t fu_volume_huge_pages_in(size_t numa_node_index);

#pragma endregion - Metadata

#pragma region - Memory

/**
 *  @brief Allocates memory on a specific NUMA node with optimal page size selection.
 *  @param[in] numa_node_index The index of the NUMA node to allocate memory on, in [0, numa_nodes_count).
 *  @param[in] minimum_bytes Minimum number of bytes to allocate, must be > 0.
 *  @param[out] allocated_pointer Pointer to store the address of the allocated memory, must not be NULL.
 *  @param[out] bytes_per_page Pointer to store the size of the RAM pages used for allocation, must not be NULL.
 *  @retval Pointer to allocated memory, or NULL if allocation failed.
 *  @note This API is @b thread-safe and can be called from any thread.
 *
 *  This function attempts to allocate memory with the largest available page size
 *  to minimize TLB pressure. The actual allocation size may be larger than requested
 *  due to page alignment requirements. Always check `allocated_bytes` for the actual size.
 *
 *  Memory allocation strategy:
 *  - Attempts 1 GB huge pages for allocations >= 2 GB
 *  - Attempts 2 MB huge pages for allocations >= 4 MB
 *  - Falls back to standard (typically 4KB) pages for smaller allocations
 *  - Always aligns to page boundaries for optimal performance
 *
 *  @code{.c}
 *  void *ptr = NULL;
 *  size_t actual_bytes = 0;
 *  if (fu_allocate_at_least(0, 1024 * 1024, &ptr, &actual_bytes)) {
 *      ... // Do some work with the allocated memory
 *      fu_free(0, ptr, actual_bytes);
 *  }
 *  @endcode
 *  @sa `fu_free` for deallocation, `fu_count_numa_nodes` for valid node indices.
 */
void *fu_allocate_at_least(                       //
    size_t numa_node_index, size_t minimum_bytes, //
    size_t *allocated_bytes, size_t *bytes_per_page);

/**
 *  @brief Releases memory allocated on a specific NUMA node.
 *  @param[in] numa_node_index The index of the NUMA node where the memory was allocated.
 *  @param[in] pointer Pointer to the memory to be released, must not be NULL.
 *  @param[in] bytes Number of bytes to release, must match the value from `allocated_bytes`.
 *  @note This API is @b thread-safe and can be called from any thread.
 *
 *  The `bytes` parameter must exactly match the `allocated_bytes` value returned
 *  by `fu_allocate_at_least`. Mismatched sizes may result in undefined behavior
 *  or memory corruption.
 *
 *  @sa `fu_allocate_at_least` for allocation.
 */
void fu_free(size_t numa_node_index, void *pointer, size_t bytes);

/**
 *  @brief Allocates exactly the requested amount of memory on a specific NUMA node.
 *  @param[in] numa_node_index The index of the NUMA node to allocate memory on.
 *  @param[in] bytes Number of bytes to allocate, must be > 0.
 *  @retval Pointer to allocated memory, or NULL if allocation failed.
 *  @note This API is @b thread-safe and can be called from any thread.
 *
 *  This function allocates exactly `bytes` of memory with the specified alignment.
 *  Unlike `fu_allocate_at_least`, this function doesn't over-allocate for page optimization.
 *  Use this for compatibility with standard allocator interfaces.
 */
void *fu_allocate(size_t numa_node_index, size_t bytes);

#pragma endregion - Memory

#pragma region - Lifetime

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
 *  if (pool && fu_pool_spawn(pool, fu_count_logical_cores(), fu_caller_inclusive_k)) {
 *      ... // Dispatch some parallel tasks
 *      fu_pool_delete(pool);
 *  }
 *  @endcode
 *  @sa `fu_pool_terminate` for shutdown, `fu_count_logical_cores` for optimal thread count.
 */
fu_bool_t fu_pool_spawn(fu_pool_t *pool, size_t threads, fu_caller_exclusivity_t exclusivity);

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

/**
 *  @brief Returns the number of distinct thread colocations in the pool.
 *  @param[in] pool Thread pool handle, must not be NULL.
 *  @retval 0 if the pool is not initialized.
 *  @retval 1 on systems without NUMA or QoS heterogeneity.
 *  @retval 2-N on systems with multiple NUMA nodes or QoS levels.
 *  @note This API is @b not synchronized.
 *
 *  A colocation represents a group of threads sharing the same memory domain
 *  and performance characteristics. This information is useful for:
 *  - Understanding the system's memory topology
 *  - Optimizing memory allocation strategies
 *  - Load balancing across heterogeneous cores
 *  @sa `fu_pool_count_threads_in` for per-colocation thread counts.
 */
size_t fu_pool_count_colocations(fu_pool_t *pool);

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
size_t fu_pool_count_threads(fu_pool_t *pool);

/**
 *  @brief Returns the number of threads in a specific colocation.
 *  @param[in] pool Thread pool handle, must not be NULL.
 *  @param[in] colocation_index Index of the colocation, must be < `fu_pool_count_colocations(pool)`.
 *  @retval 0 if the pool is not initialized or colocation_index is invalid.
 *  @retval 1-N where N is the number of threads in the specified colocation.
 *  @note This API is @b not synchronized and doesn't validate bounds.
 *
 *  Different colocations may have different thread counts depending on:
 *  - NUMA node core counts (different sockets may have different core counts)
 *  - QoS level availability (P-cores vs E-cores)
 *  - User-specified thread distribution
 *  @sa `fu_pool_count_colocations` for valid colocation indices.
 */
size_t fu_pool_count_threads_in(fu_pool_t *pool, size_t colocation_index);

/**
 *  @brief Converts a global thread index to a local thread index within a colocation.
 *  @param[in] pool Thread pool handle, must not be NULL.
 *  @param[in] global_thread_index The global thread index to convert.
 *  @param[in] colocation_index Index of the colocation, must be < `fu_pool_count_colocations(pool)`.
 *  @retval Local thread index within the specified colocation.
 */
size_t fu_pool_locate_thread_in(fu_pool_t *pool, size_t global_thread_index, size_t colocation_index);

#pragma endregion - Lifetime

#pragma region - Primary API

/**
 *  @brief Executes a callback function in parallel on all threads.
 *  @param[in] pool Thread pool handle, must not be NULL and initialized.
 *  @param[in] callback Function to execute on each thread, must not be NULL.
 *  @param[in] context User-defined context passed to the callback, may be NULL.
 *  @note This API blocks until all threads complete execution.
 *
 *  This is equivalent to OpenMP's `#pragma omp parallel` directive. Each thread
 *  executes the callback exactly once with its unique thread index and colocation.
 *
 *  The callback receives:
 *  - `context`: User-provided data (shared across all threads)
 *  - `thread`: Thread index in [0, threads_count)
 *  - `colocation`: NUMA node & QoS level identifier
 *
 *  Synchronization guarantee: This function returns only after all threads have
 *  completed their callback execution. No additional synchronization is needed.
 *
 *  @code{.c}
 *  void hello_world(void *ctx, size_t thread, size_t colocation) {
 *      printf("Hello from thread %zu in colocation %zu\n", thread, colocation);
 *  }
 *  fu_pool_for_threads(pool, hello_world, NULL);
 *  @endcode
 *  @sa `fu_pool_unsafe_for_threads` for non-blocking execution.
 */
void fu_pool_for_threads(fu_pool_t *pool, fu_for_threads_t callback, fu_lambda_context_t context);

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
 *  - `colocation`: NUMA node & QoS level of the executing thread
 *
 *  @code{.c}
 *  void process_element(void *array, size_t i, size_t thread, size_t colocation) {
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
 *  void process_variable_work(void *ctx, size_t task, size_t thread, size_t colocation) {
 *      complex_computation(task); // May take 1ms or 100ms
 *  }
 *  fu_pool_for_n_dynamic(pool, task_count, process_variable_work, context);
 *  @endcode
 *  @sa `fu_pool_for_n` for balanced workloads with predictable task duration.
 */
void fu_pool_for_n_dynamic(fu_pool_t *pool, size_t n, fu_for_prongs_t callback, fu_lambda_context_t context);

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
 *  - `colocation`: NUMA node & QoS level of the executing thread
 *
 *  Use cases:
 *  - Vectorized operations that benefit from contiguous data access
 *  - Memory copying or initialization operations
 *  - Algorithms with significant per-slice setup costs
 *  - SIMD operations that process multiple elements simultaneously
 *
 *  @code{.c}
 *  void process_slice(void *array, size_t first, size_t count, size_t thread, size_t colocation) {
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

#pragma endregion - Primary API

#pragma region - Unsafe API

/**
 *  @brief Executes a callback in parallel on all threads without blocking.
 *  @param[in] pool Thread pool handle, must not be NULL and initialized.
 *  @param[in] callback Function to execute on each thread, must not be NULL.
 *  @param[in] context User-defined context passed to the callback, may be NULL.
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
 *  The "unsafe" designation indicates:
 *  - No automatic lifetime management of callback/context
 *  - No protection against concurrent pool operations
 *  - Manual synchronization responsibility
 *
 *  @code{.c}
 *  fu_pool_unsafe_for_threads(pool, my_callback, my_context);  // Start parallel work
 *  prepare_next_batch();                                       // Do other work while tasks execute
 *  fu_pool_unsafe_join(pool);                                  // Wait for completion before proceeding
 *  @endcode
 *  @sa `fu_pool_unsafe_join` for synchronization, `fu_pool_for_threads` for blocking variant.
 */
void fu_pool_unsafe_for_threads(fu_pool_t *pool, fu_for_threads_t callback, fu_lambda_context_t context);

/**
 *  @brief Blocks the calling thread until the current parallel operation completes.
 *  @param[in] pool Thread pool handle, must not be NULL and initialized.
 *  @note This API must be called after the `fu_pool_unsafe_for_threads` operation.
 *
 *  This function provides the synchronization point for all non-blocking pool
 *  operations. It ensures that:
 *  - All worker threads complete their current tasks
 *  - Memory writes from worker threads are visible to the calling thread
 *  - The pool is ready for the next operation
 *
 *  Synchronization behavior:
 *  - If `fu_caller_inclusive_k` was used: executes the calling thread's portion first
 *  - Waits for all worker threads using efficient busy-waiting
 *  - Provides full memory synchronization (acquire-release semantics)
 *
 *  This function is mandatory after any `unsafe_` operation and before:
 *  - Starting a new parallel operation
 *  - Calling `fu_pool_terminate` or `fu_pool_delete`
 *  - Accessing results produced by the parallel operation
 *
 *  @code{.c}
 *  fu_pool_unsafe_for_n(pool, count, process_data, context);
 *  setup_next_iteration(void);
 *  fu_pool_unsafe_join(pool);
 *  use_processed_data(void);
 *  @endcode
 *  @sa `fu_pool_unsafe_for_threads` for the entry point, `fu_pool_for_threads` for blocking execution.
 */
void fu_pool_unsafe_join(fu_pool_t *pool);

#pragma endregion - Flexible API

#ifdef __cplusplus
} // extern "C"
#endif