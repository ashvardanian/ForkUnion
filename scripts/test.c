#include <stdio.h>     // `printf`, `fprintf`
#include <stdlib.h>    // `EXIT_FAILURE`, `EXIT_SUCCESS`
#include <stdatomic.h> // `atomic_size_t`, `atomic_fetch_add`
#include <stdbool.h>   // `bool`, `true`, `false`
#include <string.h>    // `memset`

#include <fork_union.h>

/* Constants */
static const size_t default_parallel_tasks_k = 10000; // 10K

/* Test helpers */
static bool test_try_spawn_zero(void) {
    fu_pool_t *pool = fu_pool_new("test_zero");
    bool result = !fu_pool_spawn(pool, 0u, fu_caller_inclusive_k);
    fu_pool_delete(pool);
    return result;
}

static bool test_try_spawn_success(void) {
    fu_pool_t *pool = fu_pool_new("test_spawn");
    if (!pool) return false;

    size_t threads = fu_count_logical_cores();
    if (threads == 0) threads = 4;

    bool result = fu_pool_spawn(pool, threads, fu_caller_inclusive_k);
    fu_pool_delete(pool);
    return result;
}

/* Context for for_threads test */
struct for_threads_context {
    atomic_bool *visited;
};

static void for_threads_callback(void *context_punned, size_t thread, size_t colocation) {
    (void)colocation;
    struct for_threads_context *context = (struct for_threads_context *)context_punned;
    atomic_store(&context->visited[thread], true);
}

static bool test_for_threads(void) {
    fu_pool_t *pool = fu_pool_new("test_for_threads");
    if (!pool) return false;

    size_t threads = fu_count_logical_cores();
    if (threads == 0) threads = 4;

    if (!fu_pool_spawn(pool, threads, fu_caller_inclusive_k)) {
        fu_pool_delete(pool);
        return false;
    }

    size_t threads_count = fu_pool_count_threads(pool);
    atomic_bool *visited = calloc(threads_count, sizeof(atomic_bool));
    struct for_threads_context context = {.visited = visited};

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

/* Context for uncomfortable input size test */
struct uncomfortable_context {
    size_t input_size;
    atomic_bool out_of_bounds;
};

static void uncomfortable_callback(void *context_punned, size_t task, size_t thread, size_t colocation) {
    (void)thread;
    (void)colocation;
    struct uncomfortable_context *context = (struct uncomfortable_context *)context_punned;
    if (task >= context->input_size) atomic_store(&context->out_of_bounds, true);
}

static bool test_uncomfortable_input_size(void) {
    fu_pool_t *pool = fu_pool_new("test_uncomfortable");
    if (!pool) return false;

    size_t threads = fu_count_logical_cores();
    if (threads == 0) threads = 4;

    if (!fu_pool_spawn(pool, threads, fu_caller_inclusive_k)) {
        fu_pool_delete(pool);
        return false;
    }

    size_t threads_count = fu_pool_count_threads(pool);
    size_t max_input_size = threads_count * 3;

    for (size_t input_size = 0; input_size <= max_input_size; ++input_size) {
        struct uncomfortable_context context = {.input_size = input_size, .out_of_bounds = false};

        fu_pool_for_n(pool, input_size, uncomfortable_callback, &context);

        if (atomic_load(&context.out_of_bounds)) {
            fu_pool_delete(pool);
            return false;
        }
    }

    fu_pool_delete(pool);
    return true;
}

/* Aligned visit structure for cache-line alignment */
struct aligned_visit {
    _Alignas(64) size_t task;
};

/* Comparator for qsort */
static int compare_visits(const void *a, const void *b) {
    const struct aligned_visit *va = (const struct aligned_visit *)a;
    const struct aligned_visit *vb = (const struct aligned_visit *)b;
    if (va->task < vb->task) return -1;
    if (va->task > vb->task) return 1;
    return 0;
}

static bool contains_iota(struct aligned_visit *visited, size_t size) {
    qsort(visited, size, sizeof(struct aligned_visit), compare_visits);

    for (size_t i = 0; i < size; ++i)
        if (visited[i].task != i) return false;
    return true;
}

/* Context for for_n test */
struct for_n_context {
    atomic_size_t counter;
    struct aligned_visit *visited;
};

static void for_n_callback(void *context_punned, size_t task, size_t thread, size_t colocation) {
    (void)thread;
    (void)colocation;
    struct for_n_context *context = (struct for_n_context *)context_punned;

    size_t count_populated = atomic_fetch_add(&context->counter, 1);
    context->visited[count_populated].task = task;
}

static bool test_for_n(void) {
    fu_pool_t *pool = fu_pool_new("test_for_n");
    if (!pool) return false;

    size_t threads = fu_count_logical_cores();
    if (threads == 0) threads = 4;

    if (!fu_pool_spawn(pool, threads, fu_caller_inclusive_k)) {
        fu_pool_delete(pool);
        return false;
    }

    struct aligned_visit *visited = calloc(default_parallel_tasks_k, sizeof(struct aligned_visit));
    struct for_n_context context = {.counter = 0, .visited = visited};

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

static bool test_for_n_dynamic(void) {
    fu_pool_t *pool = fu_pool_new("test_for_n_dynamic");
    if (!pool) return false;

    size_t threads = fu_count_logical_cores();
    if (threads == 0) threads = 4;

    if (!fu_pool_spawn(pool, threads, fu_caller_inclusive_k)) {
        fu_pool_delete(pool);
        return false;
    }

    struct aligned_visit *visited = calloc(default_parallel_tasks_k, sizeof(struct aligned_visit));
    struct for_n_context context = {.counter = 0, .visited = visited};

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

static void oversubscribed_callback(void *context_punned, size_t task, size_t thread, size_t colocation) {
    (void)thread;
    (void)colocation;
    struct for_n_context *context = (struct for_n_context *)context_punned;

    // Perform some weird amount of work, that is not very different between consecutive tasks
    static _Thread_local volatile size_t some_local_work = 0;
    for (size_t i = 0; i != task % 3; ++i) some_local_work = some_local_work + i * i;

    size_t count_populated = atomic_fetch_add(&context->counter, 1);
    context->visited[count_populated].task = task;
}

static bool test_oversubscribed_threads(void) {
    const size_t oversubscription = 3;

    fu_pool_t *pool = fu_pool_new("test_oversubscribed");
    if (!pool) return false;

    size_t threads = fu_count_logical_cores();
    if (threads == 0) threads = 4;

    if (!fu_pool_spawn(pool, threads * oversubscription, fu_caller_inclusive_k)) {
        fu_pool_delete(pool);
        return false;
    }

    struct aligned_visit *visited = calloc(default_parallel_tasks_k, sizeof(struct aligned_visit));
    struct for_n_context context = {.counter = 0, .visited = visited};

    fu_pool_for_n_dynamic(pool, default_parallel_tasks_k, oversubscribed_callback, &context);

    bool result =
        (atomic_load(&context.counter) == default_parallel_tasks_k) && contains_iota(visited, default_parallel_tasks_k);

    free(visited);
    fu_pool_delete(pool);
    return result;
}

/* GCC nested functions extension test */
#if defined(__GNUC__) && !defined(__clang__)

static bool test_gcc_nested_functions(void) {
    fu_pool_t *pool = fu_pool_new("test_gcc_nested");
    if (!pool) return false;

    size_t threads = fu_count_logical_cores();
    if (threads == 0) threads = 4;

    if (!fu_pool_spawn(pool, threads, fu_caller_inclusive_k)) {
        fu_pool_delete(pool);
        return false;
    }

    atomic_size_t counter = 0;
    size_t num_tasks = 100;

    /* GCC nested function - captures local variables */
    void nested_callback(void *context, size_t task, size_t thread, size_t colocation) {
        (void)context;
        (void)thread;
        (void)colocation;
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

static void block_callback_wrapper(void *context_punned, size_t task, size_t thread, size_t colocation) {
    struct block_wrapper *wrapper = (struct block_wrapper *)context_punned;
    wrapper->block(NULL, task, thread, colocation);
}

static bool test_clang_blocks(void) {
    fu_pool_t *pool = fu_pool_new("test_clang_blocks");
    if (!pool) return false;

    size_t threads = fu_count_logical_cores();
    if (threads == 0) threads = 4;

    if (!fu_pool_spawn(pool, threads, fu_caller_inclusive_k)) {
        fu_pool_delete(pool);
        return false;
    }

    __block atomic_size_t counter = 0;
    size_t num_tasks = 100;

    /* Clang block - captures local variables with __block */
    task_block_t my_block = ^(void *ctx, size_t task, size_t thread, size_t colocation) {
      (void)ctx;
      (void)thread;
      (void)colocation;
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

int main(void) {
    printf("Welcome to the Fork Union library test suite (C API)!\n");

    char const *caps = fu_capabilities_string();
    if (!caps) {
        fprintf(stderr, "Thread pool not supported on this platform\n");
        return EXIT_FAILURE;
    }

    printf("Capabilities: %s\n", caps);
    printf("Logical cores: %zu\n", fu_count_logical_cores());
    printf("NUMA nodes: %zu\n", fu_count_numa_nodes());
    printf("Colocations: %zu\n", fu_count_colocations());

    printf("\nStarting unit tests...\n");

    typedef bool (*test_func_t)(void);
    struct {
        char const *name;
        test_func_t function;
    } const unit_tests[] = {
        {"`try_spawn` zero threads", test_try_spawn_zero},
        {"`try_spawn` normal", test_try_spawn_success},
        {"`for_threads` dispatch", test_for_threads},
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

    size_t const total_unit_tests = sizeof(unit_tests) / sizeof(unit_tests[0]);
    size_t failed_unit_tests = 0;

    for (size_t i = 0; i < total_unit_tests; ++i) {
        printf("Running %s... ", unit_tests[i].name);
        bool const ok = unit_tests[i].function();
        if (ok) printf("PASS\n");
        else
            printf("FAIL\n");
        failed_unit_tests += !ok;
    }

    if (failed_unit_tests > 0) {
        fprintf(stderr, "%zu/%zu unit tests failed\n", failed_unit_tests, total_unit_tests);
        return EXIT_FAILURE;
    }

    printf("All %zu unit tests passed\n", total_unit_tests);

    return EXIT_SUCCESS;
}
