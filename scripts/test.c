#include <stdio.h>     // `printf`, `fprintf`
#include <stdlib.h>    // `EXIT_FAILURE`, `EXIT_SUCCESS`
#include <stdatomic.h> // `atomic_size_t`, `atomic_fetch_add`
#include <stdbool.h>   // `bool`, `true`, `false`
#include <string.h>    // `memset`

#include <forkunion.h>

static const size_t default_parallel_tasks_k = 10000; // 10K

/**
 *  The machine topology, probed once in `main` and threaded through every spawn and query below. The C
 *  ABI reads it only during a call, so a single handle serves the whole suite.
 */
static fu_topology_t machine_topology = NULL;

/** Creates a pool from @p mask and spawns a default-sized crew, or returns NULL on failure. */
static fu_pool_t spawn_default_pool(char const *name, fu_capabilities_t mask, fu_caller_exclusivity_t mode) {
    fu_pool_t pool = fu_pool_new(name, mask);
    if (!pool) return NULL;
    size_t threads = fu_logical_cores_count(machine_topology);
    if (threads == 0) threads = 4;
    if (fu_pool_spawn(machine_topology, pool, threads, mode)) return pool;
    fu_pool_delete(pool);
    return NULL;
}

static bool test_try_spawn_zero(fu_capabilities_t mask) {
    fu_pool_t pool = fu_pool_new("test_zero", mask);
    bool result = !fu_pool_spawn(machine_topology, pool, 0u, fu_caller_inclusive_k);
    fu_pool_delete(pool);
    return result;
}

static bool test_try_spawn_success(fu_capabilities_t mask) {
    fu_pool_t pool = spawn_default_pool("test_spawn", mask, fu_caller_inclusive_k);
    if (!pool) return false;
    fu_pool_delete(pool);
    return true;
}

/** Context for the `for_threads` test. */
struct for_threads_context_t {
    atomic_bool *visited;
};

static void for_threads_callback(void *context_punned, size_t thread, size_t compute_domain) {
    (void)compute_domain;
    struct for_threads_context_t *context = (struct for_threads_context_t *)context_punned;
    atomic_store(&context->visited[thread], true);
}

static bool test_for_threads(fu_capabilities_t mask) {
    fu_pool_t pool = spawn_default_pool("test_for_threads", mask, fu_caller_inclusive_k);
    if (!pool) return false;

    size_t threads_count = fu_pool_threads_count(pool);
    atomic_bool *visited = calloc(threads_count, sizeof(atomic_bool));
    struct for_threads_context_t context = {.visited = visited};

    fu_pool_for_threads(pool, for_threads_callback, &context);

    bool result = true;
    for (size_t i = 0; i < threads_count; ++i) {
        if (!atomic_load(&visited[i])) {
            result = false;
            break;
        }
    }

    free(visited);
    fu_pool_delete(pool);
    return result;
}

static bool test_caller_exclusivity_query(fu_capabilities_t mask) {
    fu_pool_t pool = fu_pool_new("test_exclusivity", mask);
    if (!pool) return false;

    size_t threads = fu_logical_cores_count(machine_topology);
    if (threads == 0) threads = 4;

    /* The pool is the single source of truth, even after a re-spawn with a different mode. */
    bool result = true;
    if (!fu_pool_spawn(machine_topology, pool, threads, fu_caller_inclusive_k)) result = false;
    else if (fu_pool_caller_exclusivity(pool) != fu_caller_inclusive_k)
        result = false;
    else {
        fu_pool_terminate(pool);
        if (!fu_pool_spawn(machine_topology, pool, threads, fu_caller_exclusive_k)) result = false;
        else if (fu_pool_caller_exclusivity(pool) != fu_caller_exclusive_k)
            result = false;
    }

    fu_pool_delete(pool);
    return result;
}

static bool test_per_compute_domain_pool(fu_capabilities_t mask) {
    size_t compute_domains = fu_compute_domains_count(machine_topology);
    if (compute_domains == 0) return false;

    /* Spawn one pool per compute domain, sized to that domain's core count. */
    bool result = true;
    for (size_t compute_domain = 0; compute_domain < compute_domains && result; ++compute_domain) {
        size_t cores = fu_logical_cores_count_in(machine_topology, compute_domain);
        if (cores == 0) cores = 2; /* Non-NUMA build reports via hardware_concurrency */

        fu_pool_t pool = fu_pool_new("compute_domain", mask);
        if (!pool) {
            result = false;
            break;
        }
        if (!fu_pool_spawn_on(machine_topology, pool, compute_domain, cores, fu_caller_exclusive_k)) result = false;
        else if (fu_pool_threads_count(pool) == 0)
            result = false;
        fu_pool_delete(pool);
    }

    /* Out-of-range compute domain must fail cleanly, not crash. */
    fu_pool_t out_of_range = fu_pool_new("bad", mask);
    if (out_of_range &&
        fu_pool_spawn_on(machine_topology, out_of_range, compute_domains + 100, 2, fu_caller_exclusive_k))
        result = false;
    fu_pool_delete(out_of_range);
    return result;
}

static bool test_generation_polling(fu_capabilities_t mask) {
    /* Polling before join is the caller-exclusive pattern: no caller slice is owed. */
    fu_pool_t pool = spawn_default_pool("test_generation", mask, fu_caller_exclusive_k);
    if (!pool) return false;

    size_t threads_count = fu_pool_threads_count(pool);
    atomic_bool *visited = calloc(threads_count, sizeof(atomic_bool));
    struct for_threads_context_t context = {.visited = visited};

    fu_generation_t generation = fu_pool_unsafe_for_threads(pool, for_threads_callback, &context);

    bool result = true;
    if ((generation & 1u) == 0) result = false; /* Tokens are always odd */
    else {
        while (!fu_pool_is_complete(pool, generation)) { /* Spin until the workers finish */
        }
        fu_pool_unsafe_join(pool, generation);
        for (size_t i = 0; i < threads_count; ++i)
            if (!atomic_load(&visited[i])) {
                result = false;
                break;
            }
    }

    free(visited);
    fu_pool_delete(pool);
    return result;
}

/** Context for uncomfortable input size test */
struct uncomfortable_context_t {
    size_t input_size;
    atomic_bool out_of_bounds;
};

static void uncomfortable_callback(void *context_punned, size_t task, size_t thread, size_t compute_domain) {
    (void)thread;
    (void)compute_domain;
    struct uncomfortable_context_t *context = (struct uncomfortable_context_t *)context_punned;
    if (task >= context->input_size) atomic_store(&context->out_of_bounds, true);
}

static bool test_uncomfortable_input_size(fu_capabilities_t mask) {
    fu_pool_t pool = spawn_default_pool("test_uncomfortable", mask, fu_caller_inclusive_k);
    if (!pool) return false;

    size_t threads_count = fu_pool_threads_count(pool);
    size_t max_input_size = threads_count * 3;

    for (size_t input_size = 0; input_size <= max_input_size; ++input_size) {
        struct uncomfortable_context_t context = {.input_size = input_size, .out_of_bounds = false};

        fu_pool_for_n(pool, input_size, uncomfortable_callback, &context);

        if (atomic_load(&context.out_of_bounds)) {
            fu_pool_delete(pool);
            return false;
        }
    }

    fu_pool_delete(pool);
    return true;
}

/** Aligned visit structure for cache-line alignment */
struct aligned_visit_t {
    _Alignas(64) size_t task;
};

/* Comparator for qsort */
static int compare_visits(const void *a, const void *b) {
    const struct aligned_visit_t *va = (const struct aligned_visit_t *)a;
    const struct aligned_visit_t *vb = (const struct aligned_visit_t *)b;
    if (va->task < vb->task) return -1;
    if (va->task > vb->task) return 1;
    return 0;
}

static bool contains_iota(struct aligned_visit_t *visited, size_t size) {
    qsort(visited, size, sizeof(struct aligned_visit_t), compare_visits);

    for (size_t i = 0; i < size; ++i)
        if (visited[i].task != i) return false;
    return true;
}

/** Context for for_n test */
struct for_n_context_t {
    atomic_size_t counter;
    struct aligned_visit_t *visited;
};

static void for_n_callback(void *context_punned, size_t task, size_t thread, size_t compute_domain) {
    (void)thread;
    (void)compute_domain;
    struct for_n_context_t *context = (struct for_n_context_t *)context_punned;

    size_t count_populated = atomic_fetch_add(&context->counter, 1);
    context->visited[count_populated].task = task;
}

static bool test_for_n(fu_capabilities_t mask) {
    fu_pool_t pool = spawn_default_pool("test_for_n", mask, fu_caller_inclusive_k);
    if (!pool) return false;

    struct aligned_visit_t *visited = calloc(default_parallel_tasks_k, sizeof(struct aligned_visit_t));
    struct for_n_context_t context = {.counter = 0, .visited = visited};

    fu_pool_for_n(pool, default_parallel_tasks_k, for_n_callback, &context);

    bool result =
        (atomic_load(&context.counter) == default_parallel_tasks_k) && contains_iota(visited, default_parallel_tasks_k);

    if (result) {
        // Test repeated calls
        atomic_store(&context.counter, 0);
        fu_pool_for_n(pool, default_parallel_tasks_k, for_n_callback, &context);

        result = (atomic_load(&context.counter) == default_parallel_tasks_k) &&
                 contains_iota(visited, default_parallel_tasks_k);
    }

    free(visited);
    fu_pool_delete(pool);
    return result;
}

static bool test_for_n_dynamic(fu_capabilities_t mask) {
    fu_pool_t pool = spawn_default_pool("test_for_n_dynamic", mask, fu_caller_inclusive_k);
    if (!pool) return false;

    struct aligned_visit_t *visited = calloc(default_parallel_tasks_k, sizeof(struct aligned_visit_t));
    struct for_n_context_t context = {.counter = 0, .visited = visited};

    fu_pool_for_n_dynamic(pool, default_parallel_tasks_k, for_n_callback, &context);

    bool result =
        (atomic_load(&context.counter) == default_parallel_tasks_k) && contains_iota(visited, default_parallel_tasks_k);

    if (result) {
        // Test repeated calls
        atomic_store(&context.counter, 0);
        fu_pool_for_n_dynamic(pool, default_parallel_tasks_k, for_n_callback, &context);

        result = (atomic_load(&context.counter) == default_parallel_tasks_k) &&
                 contains_iota(visited, default_parallel_tasks_k);
    }

    free(visited);
    fu_pool_delete(pool);
    return result;
}

static void oversubscribed_callback(void *context_punned, size_t task, size_t thread, size_t compute_domain) {
    (void)thread;
    (void)compute_domain;
    struct for_n_context_t *context = (struct for_n_context_t *)context_punned;

    // Perform some weird amount of work, that is not very different between consecutive tasks
    static _Thread_local volatile size_t some_local_work = 0;
    for (size_t i = 0; i != task % 3; ++i) some_local_work = some_local_work + i * i;

    size_t count_populated = atomic_fetch_add(&context->counter, 1);
    context->visited[count_populated].task = task;
}

static bool test_oversubscribed_threads(fu_capabilities_t mask) {
    const size_t oversubscription = 3;

    fu_pool_t pool = fu_pool_new("test_oversubscribed", mask);
    if (!pool) return false;

    size_t threads = fu_logical_cores_count(machine_topology);
    if (threads == 0) threads = 4;

    if (!fu_pool_spawn(machine_topology, pool, threads * oversubscription, fu_caller_inclusive_k)) {
        fu_pool_delete(pool);
        return false;
    }

    struct aligned_visit_t *visited = calloc(default_parallel_tasks_k, sizeof(struct aligned_visit_t));
    struct for_n_context_t context = {.counter = 0, .visited = visited};

    fu_pool_for_n_dynamic(pool, default_parallel_tasks_k, oversubscribed_callback, &context);

    bool result =
        (atomic_load(&context.counter) == default_parallel_tasks_k) && contains_iota(visited, default_parallel_tasks_k);

    free(visited);
    fu_pool_delete(pool);
    return result;
}

/** Context for the `for_slices` test. */
struct for_slices_context_t {
    atomic_uint *executions;
    size_t n;
    atomic_bool bounds_violated;
    atomic_bool empty_slice;
};

static void for_slices_callback(void *context_punned, size_t first, size_t count, size_t thread,
                                size_t compute_domain) {
    (void)thread;
    (void)compute_domain;
    struct for_slices_context_t *context = (struct for_slices_context_t *)context_punned;
    if (count == 0) atomic_store(&context->empty_slice, true);
    if (first + count > context->n) {
        atomic_store(&context->bounds_violated, true);
        return;
    }
    for (size_t i = 0; i != count; ++i) atomic_fetch_add(&context->executions[first + i], 1);
}

static bool test_for_slices(fu_capabilities_t mask) {
    fu_pool_t pool = spawn_default_pool("test_for_slices", mask, fu_caller_inclusive_k);
    if (!pool) return false;

    size_t const n = 1000;
    atomic_uint *executions = calloc(n, sizeof(atomic_uint));
    struct for_slices_context_t context = {.executions = executions, .n = n};

    fu_pool_for_slices(pool, n, for_slices_callback, &context);

    bool result = !atomic_load(&context.bounds_violated) && !atomic_load(&context.empty_slice);
    for (size_t i = 0; i < n && result; ++i) result = atomic_load(&executions[i]) == 1;

    free(executions);
    fu_pool_delete(pool);
    return result;
}

static bool test_sleep_wake(fu_capabilities_t mask) {
    fu_pool_t pool = spawn_default_pool("test_sleep", mask, fu_caller_inclusive_k);
    if (!pool) return false;

    struct aligned_visit_t *visited = calloc(default_parallel_tasks_k, sizeof(struct aligned_visit_t));
    struct for_n_context_t context = {.counter = 0, .visited = visited};

    /* Each batch naps the workers, then the dispatch itself must wake them - exactly once per task. */
    bool result = true;
    for (size_t batch = 0; batch != 2 && result; ++batch) {
        fu_pool_sleep(pool, 100);
        atomic_store(&context.counter, 0);
        fu_pool_for_n(pool, default_parallel_tasks_k, for_n_callback, &context);
        result = atomic_load(&context.counter) == default_parallel_tasks_k &&
                 contains_iota(visited, default_parallel_tasks_k);
    }

    free(visited);
    fu_pool_delete(pool);
    return result;
}

/* The C mirror of the C++ topology invariants: counts agree, cores partition, levels stay in range. */
static bool test_topology_introspection(fu_capabilities_t mask) {
    (void)mask;
    size_t const cores = fu_logical_cores_count(machine_topology);
    size_t const memory_domains = fu_memory_domains_count(machine_topology);
    size_t const compute_domains = fu_compute_domains_count(machine_topology);
    size_t const compute_levels = fu_compute_levels_count(machine_topology);
    if (cores == 0 || memory_domains == 0 || compute_domains == 0 || compute_levels == 0) return false;
    if (compute_levels > compute_domains) return false;

    /* Every core belongs to exactly one compute domain, and every level ordinal stays in range. */
    size_t cores_across_domains = 0;
    for (size_t i = 0; i < compute_domains; ++i) {
        size_t const domain_cores = fu_logical_cores_count_in(machine_topology, i);
        if (domain_cores == 0) return false;
        if (fu_compute_level_in(machine_topology, i) >= compute_levels) return false;
        cores_across_domains += domain_cores;
    }
    if (cores_across_domains != cores) return false;

    /* Per-domain RAM never exceeds the machine, and the ids handed out are valid allocator inputs. */
    size_t const total_ram = fu_volume_ram(machine_topology);
    for (size_t i = 0; i < memory_domains; ++i) {
        if (fu_memory_domain_id_at_index(machine_topology, i) < 0) return false;
        if (total_ram != 0 && fu_volume_ram_in(machine_topology, i) > total_ram) return false;
    }
    return true;
}

/* Per-domain worker counts must sum to the pool total, and global ids must localize contiguously. */
static bool test_pool_domain_accounting(fu_capabilities_t mask) {
    fu_pool_t pool = spawn_default_pool("test_accounting", mask, fu_caller_inclusive_k);
    if (!pool) return false;

    bool result = true;
    size_t const threads = fu_pool_threads_count(pool);
    size_t const domains = fu_pool_compute_domains_count(pool);
    if (domains == 0) result = false;

    size_t prefix = 0;
    for (size_t domain = 0; domain < domains && result; ++domain) {
        size_t const local_threads = fu_pool_threads_count_in(pool, domain);
        if (local_threads == 0) {
            result = false;
            break;
        }
        /* Workers are numbered contiguously per domain, so the prefix boundaries must localize to 0. */
        if (fu_pool_locate_thread_in(pool, prefix, domain) != 0) result = false;
        if (fu_pool_locate_thread_in(pool, prefix + local_threads - 1, domain) != local_threads - 1) result = false;
        prefix += local_threads;
    }
    if (prefix != threads) result = false;

    fu_pool_delete(pool);
    return result;
}

/* The allocators fall back to the heap where no placement exists, so the round-trip must succeed everywhere. */
static bool test_allocations_on_domains(fu_capabilities_t mask) {
    (void)mask;
    size_t const domains = fu_memory_domains_count(machine_topology);
    if (domains == 0) return false;

    size_t const bytes = 1u << 20;
    for (size_t index = 0; index < domains; ++index) {
        fu_memory_domain_id_t const domain_id = fu_memory_domain_id_at_index(machine_topology, index);

        unsigned char *plain = fu_allocate_on_domain_id(domain_id, bytes);
        if (!plain) return false;
        memset(plain, 0x5A, bytes);
        bool const plain_ok = plain[0] == 0x5A && plain[bytes - 1] == 0x5A;
        fu_free_on_domain_id(domain_id, plain, bytes);
        if (!plain_ok) return false;

        size_t allocated = 0, page = 0;
        unsigned char *sized = fu_allocate_at_least_on_domain_id(domain_id, bytes, &allocated, &page);
        if (!sized) return false;
        bool const sized_ok = allocated >= bytes;
        memset(sized, 0x5A, allocated);
        fu_free_on_domain_id(domain_id, sized, allocated);
        if (!sized_ok) return false;
    }

    size_t stride = 0, mapped_domains = 0, total = 0, page = 0;
    unsigned char *base = fu_allocate_symmetric(machine_topology, 4096, &stride, &mapped_domains, &total, &page);
    if (!base) return false;
    bool result = mapped_domains == domains && stride >= 4096 && total == mapped_domains * stride;
    for (size_t d = 0; d < mapped_domains && result; ++d) {
        unsigned char *slice = base + d * stride;
        memset(slice, (int)(d + 1), 4096);
        result = slice[0] == (unsigned char)(d + 1) && slice[4095] == (unsigned char)(d + 1);
    }
    fu_free_symmetric(base, total);
    return result;
}

/* GCC nested functions extension test */
#if defined(__GNUC__) && !defined(__clang__)

static bool test_gcc_nested_functions(fu_capabilities_t mask) {
    fu_pool_t pool = spawn_default_pool("test_gcc_nested", mask, fu_caller_inclusive_k);
    if (!pool) return false;

    atomic_size_t counter = 0;
    size_t num_tasks = 100;

    /* GCC nested function - captures local variables */
    void nested_callback(void *context, size_t task, size_t thread, size_t compute_domain) {
        (void)context;
        (void)thread;
        (void)compute_domain;
        atomic_fetch_add(&counter, 1);
        if (task % 20 == 0) printf("  GCC nested: Task %zu\n", task);
    }

    fu_pool_for_n(pool, num_tasks, nested_callback, NULL);

    bool result = atomic_load(&counter) == num_tasks;
    fu_pool_delete(pool);
    return result;
}

#endif // defined(__GNUC__) && !defined(__clang__)

/* Clang blocks extension test */
#if defined(__clang__) && defined(__BLOCKS__)

#include <Block.h>

typedef void (^task_block_t)(void *, size_t, size_t, size_t);

struct block_wrapper {
    task_block_t block;
};

static void block_callback_wrapper(void *context_punned, size_t task, size_t thread, size_t compute_domain) {
    struct block_wrapper *wrapper = (struct block_wrapper *)context_punned;
    wrapper->block(NULL, task, thread, compute_domain);
}

static bool test_clang_blocks(fu_capabilities_t mask) {
    fu_pool_t pool = spawn_default_pool("test_clang_blocks", mask, fu_caller_inclusive_k);
    if (!pool) return false;

    __block atomic_size_t counter = 0;
    size_t num_tasks = 100;

    /* Clang block - captures local variables with __block */
    task_block_t my_block = ^(void *ctx, size_t task, size_t thread, size_t compute_domain) {
      (void)ctx;
      (void)thread;
      (void)compute_domain;
      atomic_fetch_add(&counter, 1);
      if (task % 20 == 0) printf("  Clang block: Task %zu\n", task);
    };

    task_block_t heap_block = Block_copy(my_block);
    struct block_wrapper wrapper = {.block = heap_block};

    fu_pool_for_n(pool, num_tasks, block_callback_wrapper, &wrapper);

    Block_release(heap_block);

    bool result = atomic_load(&counter) == num_tasks;
    fu_pool_delete(pool);
    return result;
}

#endif // defined(__clang__) && defined(__BLOCKS__)

/** Runs every unit test under one capability @p mask, accumulating the tallies for `main`'s verdict. */
static void run_battery(fu_capabilities_t mask, size_t *passes_out, size_t *failures_out) {
    static struct {
        char const *name;
        bool (*function)(fu_capabilities_t);
    } const unit_tests[] = {
        {"`try_spawn` zero threads", test_try_spawn_zero},
        {"`try_spawn` normal", test_try_spawn_success},
        {"`caller_exclusivity` query", test_caller_exclusivity_query},
        {"`fu_pool_spawn_on` per-compute-domain", test_per_compute_domain_pool},
        {"`for_threads` dispatch", test_for_threads},
        {"`generation` polling", test_generation_polling},
        {"`for_n` for uncomfortable input size", test_uncomfortable_input_size},
        {"`for_n` static scheduling", test_for_n},
        {"`for_slices` slice scheduling", test_for_slices},
        {"`for_n_dynamic` dynamic scheduling", test_for_n_dynamic},
        {"`for_n_dynamic` oversubscribed threads", test_oversubscribed_threads},
        {"`sleep` and wake exactly-once", test_sleep_wake},
        {"topology introspection invariants", test_topology_introspection},
        {"per-compute-domain pool accounting", test_pool_domain_accounting},
        {"allocations on every memory domain", test_allocations_on_domains},
#if defined(__GNUC__) && !defined(__clang__)
        {"GCC nested functions extension", test_gcc_nested_functions},
#endif
#if defined(__clang__) && defined(__BLOCKS__)
        {"Clang blocks extension", test_clang_blocks},
#endif
    };

    char mask_name[256];
    fu_name_capabilities(mask, mask_name, sizeof(mask_name));

    for (size_t i = 0; i < sizeof(unit_tests) / sizeof(unit_tests[0]); ++i) {
        printf("Running %s for `%s`... ", unit_tests[i].name, mask_name);
        bool const ok = unit_tests[i].function(mask);
        printf(ok ? "PASS\n" : "FAIL\n");
        *passes_out += ok, *failures_out += !ok;
    }
}

int main(void) {
    printf("Welcome to the ForkUnion library test suite (C API)!\n");

    fu_capabilities_t const comptime_mask = fu_comptime_capabilities();
    fu_capabilities_t const runtime_mask = fu_runtime_capabilities();
    if (!(comptime_mask & fu_capability_os_threads_k)) {
        fprintf(stderr, "Thread pool not supported on this platform\n");
        return EXIT_FAILURE;
    }

    machine_topology = fu_topology_new();
    if (!machine_topology) {
        fprintf(stderr, "Failed to probe machine topology\n");
        return EXIT_FAILURE;
    }

    char comptime_mask_name[256], runtime_mask_name[256];
    fu_name_capabilities(comptime_mask, comptime_mask_name, sizeof(comptime_mask_name));
    fu_name_capabilities(runtime_mask, runtime_mask_name, sizeof(runtime_mask_name));
    printf("Compiled with:      %s\n", comptime_mask_name);
    printf("Running on:         %s\n", runtime_mask_name);
    printf("Logical cores:      %zu\n", fu_logical_cores_count(machine_topology));
    printf("Memory domains:     %zu\n", fu_memory_domains_count(machine_topology));
    printf("Compute domains:    %zu\n", fu_compute_domains_count(machine_topology));

    fu_capabilities_t const yield_variants[] = {
        fu_capabilities_unknown_k,   fu_capability_x86_pause_k,  fu_capability_x86_tpause_k,
        fu_capability_arm64_yield_k, fu_capability_arm64_wfet_k, fu_capability_risc5_pause_k,
        fu_capability_risc5_wrs_k,
    };

    fu_capabilities_t const topology_variants[] = {
        fu_capability_os_threads_k,
        fu_capability_os_threads_k | fu_capability_topology_k | fu_capability_place_memory_on_domain_k,
    };

    size_t passes = 0;
    size_t failures = 0;
    for (size_t i = 0; i < sizeof(yield_variants) / sizeof(*yield_variants); ++i) {
        fu_capabilities_t yield_variant = yield_variants[i];

        /* A yield the machine lacks would be silently narrowed by `fu_pool_new`, so the battery
         * would mislabel which variant it exercised. Facility bits only need to be compiled in -
         * the runtime mask never carries them, and the library narrows the rest per pool. */
        if ((yield_variant & runtime_mask) != yield_variant) continue;
        for (size_t j = 0; j < sizeof(topology_variants) / sizeof(*topology_variants); ++j) {
            fu_capabilities_t topology_variant = topology_variants[j];
            if ((topology_variant & comptime_mask) != topology_variant) continue;
            run_battery(yield_variant | topology_variant, &passes, &failures);
        }
    }

    fu_topology_delete(machine_topology);

    if (passes + failures == 0) {
        fprintf(stderr, "No capability combination was runnable - the filter is broken\n");
        return EXIT_FAILURE;
    }
    if (failures > 0) {
        fprintf(stderr, "%zu/%zu test runs failed\n", failures, failures + passes);
        return EXIT_FAILURE;
    }

    printf("All %zu test runs passed\n", passes);
    return EXIT_SUCCESS;
}
