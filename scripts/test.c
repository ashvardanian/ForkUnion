#include <stdio.h>     // `printf`, `fprintf`
#include <stdlib.h>    // `EXIT_FAILURE`, `EXIT_SUCCESS`
#include <stdatomic.h> // `atomic_size_t`, `atomic_fetch_add`
#include <stdbool.h>   // `bool`, `true`, `false`
#include <string.h>    // `memset`

#include <forkunion.h>

static const size_t default_parallel_tasks_k = 10000; // 10K

/** Creates a pool from @p mask and spawns a default-sized crew, or returns NULL on failure. */
static fu_pool_t *spawn_default_pool(char const *name, fu_capabilities_t mask, fu_caller_exclusivity_t mode) {
    fu_pool_t *pool = fu_pool_new(name, mask);
    if (!pool) return NULL;
    size_t threads = fu_logical_cores_count();
    if (threads == 0) threads = 4;
    if (fu_pool_spawn(pool, threads, mode)) return pool;
    fu_pool_delete(pool);
    return NULL;
}

static bool test_try_spawn_zero(fu_capabilities_t mask) {
    fu_pool_t *pool = fu_pool_new("test_zero", mask);
    bool result = !fu_pool_spawn(pool, 0u, fu_caller_inclusive_k);
    fu_pool_delete(pool);
    return result;
}

static bool test_try_spawn_success(fu_capabilities_t mask) {
    fu_pool_t *pool = spawn_default_pool("test_spawn", mask, fu_caller_inclusive_k);
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
    fu_pool_t *pool = spawn_default_pool("test_for_threads", mask, fu_caller_inclusive_k);
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
    fu_pool_t *pool = fu_pool_new("test_exclusivity", mask);
    if (!pool) return false;

    size_t threads = fu_logical_cores_count();
    if (threads == 0) threads = 4;

    /* The pool is the single source of truth, even after a re-spawn with a different mode. */
    bool result = true;
    if (!fu_pool_spawn(pool, threads, fu_caller_inclusive_k)) result = false;
    else if (fu_pool_caller_exclusivity(pool) != fu_caller_inclusive_k)
        result = false;
    else {
        fu_pool_terminate(pool);
        if (!fu_pool_spawn(pool, threads, fu_caller_exclusive_k)) result = false;
        else if (fu_pool_caller_exclusivity(pool) != fu_caller_exclusive_k)
            result = false;
    }

    fu_pool_delete(pool);
    return result;
}

static bool test_per_compute_domain_pool(fu_capabilities_t mask) {
    size_t compute_domains = fu_compute_domains_count();
    if (compute_domains == 0) return false;

    /* Spawn one pool per compute domain, sized to that domain's core count. */
    bool result = true;
    for (size_t compute_domain = 0; compute_domain < compute_domains && result; ++compute_domain) {
        size_t cores = fu_logical_cores_count_in(compute_domain);
        if (cores == 0) cores = 2; /* Non-NUMA build reports via hardware_concurrency */

        fu_pool_t *pool = fu_pool_new("compute_domain", mask);
        if (!pool) {
            result = false;
            break;
        }
        if (!fu_pool_spawn_on(pool, compute_domain, cores, fu_caller_exclusive_k)) result = false;
        else if (fu_pool_threads_count(pool) == 0)
            result = false;
        fu_pool_delete(pool);
    }

    /* Out-of-range compute domain must fail cleanly, not crash. */
    fu_pool_t *out_of_range = fu_pool_new("bad", mask);
    if (out_of_range && fu_pool_spawn_on(out_of_range, compute_domains + 100, 2, fu_caller_exclusive_k)) result = false;
    fu_pool_delete(out_of_range);
    return result;
}

static bool test_generation_polling(fu_capabilities_t mask) {
    /* Polling before join is the caller-exclusive pattern: no caller slice is owed. */
    fu_pool_t *pool = spawn_default_pool("test_generation", mask, fu_caller_exclusive_k);
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
    fu_pool_t *pool = spawn_default_pool("test_uncomfortable", mask, fu_caller_inclusive_k);
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
    fu_pool_t *pool = spawn_default_pool("test_for_n", mask, fu_caller_inclusive_k);
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
    fu_pool_t *pool = spawn_default_pool("test_for_n_dynamic", mask, fu_caller_inclusive_k);
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

    fu_pool_t *pool = fu_pool_new("test_oversubscribed", mask);
    if (!pool) return false;

    size_t threads = fu_logical_cores_count();
    if (threads == 0) threads = 4;

    if (!fu_pool_spawn(pool, threads * oversubscription, fu_caller_inclusive_k)) {
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

/* GCC nested functions extension test */
#if defined(__GNUC__) && !defined(__clang__)

static bool test_gcc_nested_functions(fu_capabilities_t mask) {
    fu_pool_t *pool = spawn_default_pool("test_gcc_nested", mask, fu_caller_inclusive_k);
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
    fu_pool_t *pool = spawn_default_pool("test_clang_blocks", mask, fu_caller_inclusive_k);
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

/**
 *  @brief Runs the whole unit-test battery once, building every pool from @p mask.
 *  @return Tests that failed; adds the number run to `*ran`.
 */
static size_t run_battery(fu_capabilities_t mask, char const *combo, size_t *ran) {
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
        {"`for_n_dynamic` dynamic scheduling", test_for_n_dynamic},
        {"`for_n_dynamic` oversubscribed threads", test_oversubscribed_threads},
#if defined(__GNUC__) && !defined(__clang__)
        {"GCC nested functions extension", test_gcc_nested_functions},
#endif
#if defined(__clang__) && defined(__BLOCKS__)
        {"Clang blocks extension", test_clang_blocks},
#endif
    };

    printf("== capability combo: %s ==\n", combo);
    size_t failed = 0;
    for (size_t i = 0; i < sizeof(unit_tests) / sizeof(unit_tests[0]); ++i, ++*ran) {
        printf("Running %s... ", unit_tests[i].name);
        bool const ok = unit_tests[i].function(mask);
        printf(ok ? "PASS\n" : "FAIL\n");
        failed += !ok;
    }
    return failed;
}

int main(void) {
    printf("Welcome to the ForkUnion library test suite (C API)!\n");

    char const *caps = fu_runtime_capabilities_string();
    if (!caps) {
        fprintf(stderr, "Thread pool not supported on this platform\n");
        return EXIT_FAILURE;
    }

    printf("Compiled with: %s\n", fu_comptime_capabilities_string());
    printf("Running on:    %s\n", caps);
    printf("Logical cores: %zu\n", fu_logical_cores_count());
    printf("NUMA nodes: %zu\n", fu_memory_domains_count());
    printf("ComputeDomains: %zu\n", fu_compute_domains_count());

    // Run the battery under the default pool, then under each waiter the machine offers - flat, and
    // (where colocated pools exist) NUMA-gated too. Combo names come straight from the enum.
    fu_capabilities_t const waiters = fu_runtime_capabilities() & fu_capability_any_yield_k;
    bool const has_numa = (fu_comptime_capabilities() & fu_capability_colocate_pools_on_domain_k) != 0;

    size_t ran = 0, failed = 0;
    failed += run_battery(fu_capabilities_all_k, "default", &ran);
    for (unsigned bit = 1; bit != 0; bit <<= 1) {
        fu_capabilities_t const waiter = (fu_capabilities_t)bit;
        if (!(waiters & waiter)) continue;
        failed += run_battery(waiter, fu_capability_name(waiter), &ran);
        if (has_numa) {
            char distributed[64];
            snprintf(distributed, sizeof(distributed), "%s (distributed)", fu_capability_name(waiter));
            failed +=
                run_battery((fu_capabilities_t)(waiter | fu_capability_place_memory_on_domain_k), distributed, &ran);
        }
    }

    if (failed > 0) {
        fprintf(stderr, "%zu/%zu test runs failed\n", failed, ran);
        return EXIT_FAILURE;
    }

    printf("All %zu test runs passed\n", ran);
    return EXIT_SUCCESS;
}
