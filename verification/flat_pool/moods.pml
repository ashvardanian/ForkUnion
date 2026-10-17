/**
 *  @file verification/flat_pool/moods.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief The moods: a sleep between dispatches, the wake-up from it, and @c terminate.
 *
 *  The dispatcher may sleep between dispatches, storing chill from the thread that spawned, as the
 *  header allows; the dispatch exchanges it back to grind, a worker seeing chill with the epoch
 *  still naps once and looks again, and @c terminate stores die after the last join, waits for the
 *  workers, and resets the mood and the epoch. No slice runs after die, and no worker is left
 *  waiting on the pool.
 *
 *  @verify pass sc,rc11
 *  @verify stuck sc wait_cap=false: @c terminate re-releases the epoch after die, which wakes a
 *      worker already asleep on it, but the write keeps the epoch's value; a worker that read
 *      grind before die and arms after the re-release re-checks an unmoved epoch, so waiting on
 *      the epoch alone, as an uncapped monitor would, it never notices die, and @c terminate waits
 *      on it forever; the cap bounds that window to one timeout
 *  @verify pass sc strong_wake=false: a @c flat_pool worker looks at the epoch before the mood, so
 *      a chill that a spuriously failed exchange leaves behind costs it naps and never a dispatch
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

/** The caller: spawns, may sleep before each dispatch, joins every generation, and terminates. */
proctype dispatcher(byte t) {
    int round, observed, generation, seen_epoch, before, seen_fork, seen_callback;
    bool exchanged;
    spawn(t);
    for (round : 1 .. generations) {
        // Sleep, between the tasks and from the thread that runs them: `flat_pool::sleep`
        if
        :: atomic { store(t, mood, order_release, chill); wake(mood) }
        :: skip
        fi;
        dispatch(t, round, generation, observed, exchanged);
        join(t, generation, seen_epoch, seen_fork, seen_callback, before, observed);
        check_results(t, generation, observed)
    };
    terminate(t, observed)
}

init { atomic { run dispatcher(caller) } }
