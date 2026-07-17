/**
 *  @brief  Regression test: a `fork()`-ed child must degrade to serial, never block.
 *  @file   fork.c
 *  @author Ash Vardanian
 *
 *  Python's `multiprocessing` defaults to the `fork` start method on Linux, so any process that has touched a pool
 *  before forking lands here. Only the forking thread survives, but the pool's memory - including its thread count -
 *  is copied intact, so a child that trusts it waits on workers that no longer exist.
 *
 *  The child must therefore see zero threads and still complete every task on the caller, while the parent's own pool
 *  stays untouched.
 */
#include <stdio.h>  // `printf`
#include <stdlib.h> // `EXIT_FAILURE`

#if defined(_WIN32) // FU-ALLOW: standalone C test, includes only the C ABI header, no FU_ON_* in scope
int main(void) {
    puts("No `fork()` on Windows - skipping");
    return EXIT_SUCCESS;
}
#else
#include <sys/wait.h> // `waitpid`
#include <unistd.h>   // `fork`, `usleep`

#include <forkunion.h>

enum { tasks_k = 8, threads_k = 4, child_timeout_deciseconds_k = 100 };

static void mark(fu_lambda_context_t context, size_t task, size_t thread, size_t compute_domain) {
    (void)thread, (void)compute_domain;
    ((int *)context)[task] = 1;
}

int main(void) {
    fu_topology_t topology = fu_topology_new();
    fu_pool_t pool = fu_pool_new("fork-test", fu_capabilities_all_k);
    if (!topology || !pool) return EXIT_FAILURE;
    if (!fu_pool_spawn(topology, pool, threads_k, fu_caller_exclusive_k)) return EXIT_FAILURE;

    int parent_marks[tasks_k] = {0};
    fu_pool_for_n(pool, tasks_k, &mark, parent_marks);
    if (fu_pool_threads_count(pool) != threads_k) return EXIT_FAILURE;

    pid_t const child = fork();
    if (child == 0) {
        // The workers stayed in the parent, so the count must not claim otherwise.
        if (fu_pool_threads_count(pool) != 0) _exit(EXIT_FAILURE);

        // Each of these blocked forever before the `pthread_atfork` handler existed.
        int marks[tasks_k] = {0};
        fu_pool_for_n(pool, tasks_k, &mark, marks);
        for (size_t i = 0; i != tasks_k; ++i)
            if (!marks[i]) _exit(EXIT_FAILURE);

        int dynamic_marks[tasks_k] = {0};
        fu_pool_for_n_dynamic(pool, tasks_k, &mark, dynamic_marks);
        for (size_t i = 0; i != tasks_k; ++i)
            if (!dynamic_marks[i]) _exit(EXIT_FAILURE);

        int thread_marks[tasks_k] = {0};
        fu_pool_for_threads(pool, (fu_for_threads_t)&mark, thread_marks);
        _exit(EXIT_SUCCESS);
    }
    if (child < 0) return EXIT_FAILURE;

    // Poll rather than block: a regression here is a hang, and a hung test should fail, not stall CI forever.
    int status = 0;
    for (int i = 0; i != child_timeout_deciseconds_k; ++i) {
        if (waitpid(child, &status, WNOHANG) == child) goto reaped;
        usleep(100000);
    }
    puts("Child hung in a dispatch - `fork()` handling regressed");
    kill(child, SIGKILL), waitpid(child, &status, 0);
    return EXIT_FAILURE;

reaped:
    if (!WIFEXITED(status) || WEXITSTATUS(status) != EXIT_SUCCESS) return EXIT_FAILURE;

    // Forking must not disturb the parent: its workers are still its own.
    int again[tasks_k] = {0};
    fu_pool_for_n(pool, tasks_k, &mark, again);
    for (size_t i = 0; i != tasks_k; ++i)
        if (!again[i]) return EXIT_FAILURE;
    if (fu_pool_threads_count(pool) != threads_k) return EXIT_FAILURE;

    fu_pool_delete(pool);
    fu_topology_delete(topology);
    puts("Fork test passed");
    return EXIT_SUCCESS;
}
#endif
