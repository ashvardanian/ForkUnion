/**
 *  @file verification/flat_pool/contenders.pml
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief Two threads dispatching on one pool at once: one is admitted, and the other is refused or
 *      waits its turn.
 *
 *  The caller spawns the pool and dispatches fork 1; a rival thread holding the same pool
 *  dispatches fork 2 at any moment after the spawn. Each joins only an admitted dispatch and then
 *  reads both workers' results; a refused one returns the zero token and joins nothing. Once both
 *  finished, the caller lets the workers go.
 *
 *  No fork is torn: a refused dispatch writes nothing, an admitted one writes its fork only once
 *  the other's contributors all counted down, and every contributor reads the fork of the
 *  generation it runs, and runs it once. An admitted dispatch that follows another waits out the
 *  completion step of the other, so every token is odd.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc claim=false: @c unsafe_for_threads claims the countdown by exchanging it from
 *      zero, and only then writes the fork; with the plain store, the second dispatch overwrites
 *      the @c fork_state while the running generation's slices still run, and then resets the
 *      countdown under them
 *  @verify fail sc completion_wait=false: the last contributor counts down to zero one step before
 *      it steps the epoch to even, and the next dispatch waits that step out; without the wait, a
 *      dispatch admitted in between steps the still odd epoch to even and receives an even token
 *  @verify stuck rc11 claim_order=order_relaxed: the claim acquires the countdown's release
 *      sequence, which carries the running generation's odd epoch to the next dispatch; relaxed,
 *      its completion wait reads an even epoch from before that generation, and its store sends
 *      the epoch back to that generation's token, so a worker already holding the token never
 *      sees a new one, the countdown never reaches zero, and the join waits forever
 */
#define thread_count 4
#define location_count 7
#define history_depth 9
#define through_c_shim false
#include "protocol.pml"

/** The thread that dispatches beside the caller. */
#define rival 3

bool rival_finished;

/** A worker the spawn starts: contributes to every dispatch until die. */
proctype worker(byte t) {
    start(t);
    serve(t)
}

/** The caller: spawns, dispatches fork 1 and joins it if admitted, then lets the workers go once
 *  the rival finished. */
proctype dispatcher(byte t) {
    int observed, generation, seen_epoch, before, seen_fork, seen_callback;
    bool exchanged;
    spawn(t);
    dispatch(t, 1, generation, observed, exchanged);
    if
    :: generation != 0 ->
        join(t, generation, seen_epoch, seen_fork, seen_callback, before, observed);
        check_results(t, generation, observed)
    :: else
    fi;
    (rival_finished);
    atomic { store(t, mood, order_release, die); wake(mood) }
}

/** A second holder of the pool: dispatches fork 2 after the spawn, and joins it if admitted. */
proctype contender(byte t) {
    int observed, generation, seen_epoch, before, seen_fork, seen_callback;
    bool exchanged;
    start(t);
    dispatch(t, 2, generation, observed, exchanged);
    if
    :: generation != 0 ->
        join(t, generation, seen_epoch, seen_fork, seen_callback, before, observed);
        check_results(t, generation, observed)
    :: else
    fi;
    rival_finished = true
}

init { atomic { run dispatcher(caller); run contender(rival) } }
