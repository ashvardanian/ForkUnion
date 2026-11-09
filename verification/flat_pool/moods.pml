/**
 *  @file verification/flat_pool/moods.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief The moods: a sleep between dispatches, the wake-up from it, and @c terminate.
 *
 *  The dispatcher may nap, sleep, or do both in either order between dispatches, storing the mood
 *  from the thread that spawned, as the header allows; the dispatch stores it back to grind, a
 *  worker seeing a parked mood with the epoch still waits on the mood word with no cap, so only a
 *  store to that word wakes it, and @c terminate stores die after the last join, waits for the
 *  workers, and resets the mood and the epoch.
 *
 *  - No slice runs after die, and no worker is left waiting on the pool.
 *
 *  @budget 4096
 *  @verify pass sc,rc11
 *  @verify stuck sc wait_cap=false: a worker that read grind before die and arms after the
 *      re-release re-checks an unmoved epoch, so it never notices die, and @c terminate waits
 *      on it forever
 *  @verify stuck sc nap_wakes_kernel=false: a @c sleep then a @c nap leaves a worker in the kernel,
 *      where the store wakes nobody, and the dispatch reads nap and skips its wake
 */
#define thread_count 3
#define location_count 10
#define history_depth 13
#define through_c_shim false
#include <flat_pool/protocol.pml>

/** A worker the spawn starts: contributes to every dispatch until die. */
proctype worker(byte t) {
    start(t);
    serve(t)
}

/** The caller: spawns, may sleep before each dispatch, joins every generation, and terminates. */
proctype dispatcher(byte t) {
    int round, observed, generation, seen_epoch, before, seen_fork, seen_callback;
    bool exchanged;
    spawn(t);
    for (round : 1 .. generations) {
        // Park, between the tasks and from the thread that runs them: `flat_pool::nap` and `sleep`
        if
        :: nap_workers(t, observed)
        :: sleep_workers(t)
        :: nap_workers(t, observed); sleep_workers(t)
        :: sleep_workers(t); nap_workers(t, observed)
        :: skip
        fi;
        dispatch(t, round, generation, observed, exchanged);
        join(t, generation, seen_epoch, seen_fork, seen_callback, before, observed);
        check_results(t, generation, observed)
    };
    terminate(t, observed)
}

init { atomic { run dispatcher(caller) } }
