/**
 *  @file verification/flat_pool/respawn.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief A re-spawn: after @c terminate, two fresh workers and one more generation.
 *
 *  The fresh workers run as threads 3 and 4 and reuse the result words of the first two. A
 *  @c broadcast_join kept alive across a @c terminate and a @c spawn would alias after one
 *  re-spawn rather than after 2^bits epochs, since @c terminate resets the epoch; it asserts every
 *  dispatch joined, so nothing outlives it by contract.
 *
 *  @verify pass rc11
 *  @verify stuck sc join_before_reset=false: @c terminate waits for every worker to leave before it
 *      resets the mood and the epoch; resetting them first, a worker that has not read die yet
 *      finds the pool back in grind at epoch zero, and waits for a dispatch forever
 */
#define thread_count 5
#define location_count 8
#define history_depth 13
#define through_c_shim false
#include "protocol.pml"

/** A worker the spawn starts: contributes to every dispatch until die. */
proctype worker(byte t) { serve(t) }

/** The caller: spawns, dispatches and joins every generation, terminates and spawns again before
 *  the last one, and terminates. */
proctype dispatcher(byte t) {
    int round, observed, generation, seen_epoch, before, seen_fork, seen_callback, exited_before_reset;
    bool exchanged;
    spawn(t);
    for (round : 1 .. generations + 1) {
        if
        :: round == generations + 1 ->
            terminate(t, observed, exited_before_reset);
            (workers_exited == exited_before_reset);
            spawn(t)
        :: else
        fi;
        dispatch(t, round, generation, observed, exchanged);
        join(t, generation, seen_epoch, seen_fork, seen_callback, before, observed);
        check_results(t, generation, observed)
    };
    terminate(t, observed, exited_before_reset)
}

init { atomic { run dispatcher(caller) } }
