/**
 *  @file verification/flat_pool/plain.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Two generations on a caller-inclusive pool with two workers, and what each join sees.
 *
 *  After each join the dispatcher reads both workers' results of that generation; after the last,
 *  it stores die.
 *
 *  @verify pass sc,rc11
 *  @verify fail rc11 decrement_order=order_release: the countdown's decrements are @c acq_rel, so
 *      each contributor's writes chain into the last one, whose release step publishes them; with
 *      release alone, the last contributor acquires nothing of its peers, and the join that follows
 *      reads a stale result
 *  @verify fail rc11 dispatch_order=order_relaxed: @c unsafe_for_threads steps the epoch with a
 *      release; relaxed, a worker that picked up the dispatch reads an older @c fork_state
 *  @verify fail rc11 pickup_order=order_relaxed: @c worker_loop_ loads the epoch with an acquire;
 *      relaxed, the worker reads the @c fork_state from before the dispatch it picked up
 *  @verify fail rc11 worker_completion_order=order_relaxed: a worker that is the last contributor
 *      steps the epoch with a release; relaxed, the join that sees the step reads stale results
 *  @verify fail rc11 join_order=order_relaxed: @c unsafe_join waits for the completion step with
 *      an acquire; relaxed, it sees the step and reads a stale result
 *  @verify stuck sc recheck=false: the monitored wait re-checks the epoch after arming; without
 *      it, the completion step lands between the join's load and its arming, wakes nobody, and the
 *      uncapped join waits for it forever
 */
#define thread_count 3
#define location_count 7
#define history_depth 9
#define through_c_shim false
#include "protocol.pml"

/** A worker the spawn starts: contributes to every dispatch until die. */
proctype worker(byte t) {
    start(t);
    serve(t)
}

/** The caller: spawns, dispatches and joins every generation, then lets the workers go. */
proctype dispatcher(byte t) {
    int round, observed, generation, seen_epoch, before, seen_fork, seen_callback;
    bool exchanged;
    spawn(t);
    for (round : 1 .. generations) {
        dispatch(t, round, generation, observed, exchanged);
        join(t, generation, seen_epoch, seen_fork, seen_callback, before, observed);
        check_results(t, generation, observed)
    };
    signal_die(t, observed)
}

init { atomic { run dispatcher(caller) } }
