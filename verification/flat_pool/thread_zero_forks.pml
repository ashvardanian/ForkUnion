/**
 *  @file verification/flat_pool/thread_zero_forks.pml
 *  @author Ash Vardanian
 *  @date November 9, 2026
 *  @brief Thread 0 of a caller-exclusive pool forking from inside its slice: twice, the second fork
 *      nesting a third, while the dispatcher only polls its token and then dispatches again.
 *
 *  The dispatcher hands the pool a fork whose slice on thread 0 forks twice through the same pool,
 *  the second fork's slice forking once more. Each fork waits out the other slices, re-arms the
 *  count and steps the epoch by two, runs thread 0's slice in place and waits again; thread 0 then
 *  counts its own unit down. The dispatcher polls the token rule, reads every result, dispatches
 *  once more and reads that dispatch's results.
 *
 *  - Every fork's slices run once, and each fork reads both workers' results of it.
 *  - The token reads complete only once thread 0's slice ended, and tokens never repeat.
 *
 *  @budget 4096
 *  @verify pass sc,rc11
 *  @verify fail rc11 fork_publish_order=order_relaxed: relaxed, a worker that picked up a fork from
 *      thread 0 reads the @c fork_state from before it
 *  @verify fail sc wait_for_other_slices=false: thread 0 re-arms while a worker that picked up the
 *      dispatch has yet to read its @c fork_state, and that worker runs the fork in its place
 *  @verify fail sc completion_from_current_epoch=false: the completion step steps the epoch the
 *      last contributor picked up, behind the forks, so the next dispatch reuses a fork's token
 */
#define thread_count 4
#define location_count 10
#define history_depth 18
#define through_c_shim false
#include <flat_pool/protocol.pml>

/** The thread that dispatches, past the pool's three, and the fork of its second dispatch. */
#define dispatcher_thread 3
#define plain_fork 1

/** A worker of the caller-exclusive pool, thread 0 among them: contributes to every dispatch until
 *  die, and forks from its slice of @c forking_fork. */
proctype worker(byte t) { serve(t) }

/** The dispatcher: hands out @c forking_fork, polls its token, reads every result, then dispatches
 *  once more and lets the workers go. */
proctype dispatcher(byte t) {
    int observed, generation, second_generation, seen_dispatched;
    bool exchanged, complete;
    dispatch(t, forking_fork, generation, observed, exchanged);
    assert(generation != 0);
    await_token(t, generation, observed, seen_dispatched, complete);
    check_results(t, generation + 6, observed);
    dispatch(t, plain_fork, second_generation, observed, exchanged);
    assert(second_generation > generation + 6);
    await_token(t, second_generation, observed, seen_dispatched, complete);
    check_results(t, second_generation, observed);
    signal_die(t, observed)
}

init { atomic { run worker(0); run worker(1); run worker(2); run dispatcher(dispatcher_thread) } }
