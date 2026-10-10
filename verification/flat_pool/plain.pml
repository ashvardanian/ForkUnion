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
 */
#define thread_count 3
#define location_count 8
#define history_depth 9
#define through_c_shim false
#include "protocol.pml"

/** A worker the spawn starts: contributes to every dispatch until die. */
proctype worker(byte t) { serve(t) }

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
    store(t, mood, order_release, die)
}

init { atomic { run dispatcher(caller) } }
