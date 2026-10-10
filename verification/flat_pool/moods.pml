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
 *  @verify stuck sc wait_cap=false: the workers re-check the mood only under a capped wait, which
 *      bounds a @c sleep or a @c terminate notice to one timeout; waiting on the epoch alone, as
 *      an uncapped monitor would, a worker never notices die, and @c terminate waits on it forever
 *  @verify fail rc11 chill_at_spawn=false: @c colocated_pool::spawn stores chill before it writes
 *      the handles, and grind with a release after; without the chill, a fresh worker leaves its
 *      startup wait at once and reads no handle
 *  @verify stuck sc strong_wake=false: the dispatch wakes a chilled pool with one strong exchange
 *      back to grind, the same instruction on x86; a weak one may fail spuriously on load-linked
 *      architectures, and a colocated worker still in its startup wait after a @c sleep leaves it
 *      only once the mood stops being chill, so the join waits on that worker forever
 */
#define thread_count 3
#define location_count 8
#define history_depth 9
#define through_c_shim false
#include "protocol.pml"

/** A worker the spawn starts: contributes to every dispatch until die. */
proctype worker(byte t) { serve(t) }

/** The caller: spawns, may sleep before each dispatch, joins every generation, and terminates. */
proctype dispatcher(byte t) {
    int round, observed, generation, seen_epoch, before, seen_fork, seen_callback, exited_before_reset;
    bool exchanged;
    spawn(t);
    for (round : 1 .. generations) {
        // Sleep, between the tasks and from the thread that runs them: `flat_pool::sleep`
        if
        :: store(t, mood, order_release, chill)
        :: skip
        fi;
        dispatch(t, round, generation, observed, exchanged);
        join(t, generation, seen_epoch, seen_fork, seen_callback, before, observed);
        check_results(t, generation, observed)
    };
    terminate(t, observed, exited_before_reset)
}

init { atomic { run dispatcher(caller) } }
