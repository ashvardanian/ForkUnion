/**
 *  @file verification/colocated_pool.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Spin model of @c colocated_pool from `include/forkunion/distributed.hpp`: its spawn and
 *      the startup wait before a worker finds its handle.
 *
 *  The spawn stores chill with a release, starts the workers, writes their handles, and stores
 *  grind with a release. A fresh worker waits out the chill with an uncapped wait, since only the
 *  mood moves during the startup, finds its handle, and then runs the worker loop, which
 *  @c colocated_pool shares with @c flat_pool together with the dispatch and the join, so they
 *  come from `flat_pool/protocol.pml`. The dispatcher may sleep between dispatches, storing chill
 *  from the thread that spawned, joins every generation, and terminates.
 *
 *  Every worker leaves its startup wait and reads the handles the spawn wrote.
 *
 *  @verify pass sc,rc11
 *  @verify fail rc11 chill_at_spawn=false: @c colocated_pool::spawn stores chill before it writes
 *      the handles, and grind with a release after; without the chill, a fresh worker leaves its
 *      startup wait at once and reads no handle
 *  @verify stuck sc strong_wake=false: the dispatch wakes a chilled pool with one strong exchange
 *      back to grind, the same instruction on x86; a weak one may fail spuriously on load-linked
 *      architectures, and a worker still in its startup wait after a @c sleep leaves it only once
 *      the mood stops being chill, so the join waits on that worker forever
 *  @verify stuck sc recheck=false: every monitored wait re-checks its word after arming, the
 *      startup wait the mood as the join the epoch; without it, a write that lands between a
 *      waiter's load and its arming wakes nobody, and the uncapped waiter waits forever, the
 *      worker for a grind or the join for a completion step that already happened
 */
#define thread_count 3
#define location_count 10
#define history_depth 9
#define through_c_shim false
#include "flat_pool/protocol.pml"

/** The handles the spawn writes, past the words of the flat pool. */
#define handles 9

/** The knob: the header's choice, overridden by a `@verify` line. */
#ifndef chill_at_spawn
#define chill_at_spawn true
#endif

/** Chill, the workers, their handles, then grind with a release: @c colocated_pool::spawn in
 *  `distributed.hpp`. */
inline spawn_colocated(t) {
    if
    :: chill_at_spawn -> atomic { store(t, mood, order_release, chill); wake(mood) }
    :: else
    fi;
    store(t, spawned, order_release, 1);
    run worker(1);
    run worker(2);
    store(t, handles, order_relaxed, 1);
    atomic { store(t, mood, order_release, grind); wake(mood) }
}

/** A worker: acquires the spawn, waits out the chill, finds its handle, and runs the worker loop:
 *  @c colocated_pool::worker_loop_body_ in `distributed.hpp`. */
proctype worker(byte t) {
    int startup_mood, seen_handles;
    start(t);
    do
    :: load(t, mood, order_acquire, startup_mood);
       if
       :: startup_mood != chill -> break
       :: else -> wait_on(t, mood, chill, false)
       fi
    od;
    if
    :: startup_mood == grind ->
        load(t, handles, order_relaxed, seen_handles);
        assert(seen_handles == 1)
    :: else
    fi;
    serve(t)
}

/** The caller: spawns, may sleep before each dispatch, joins every generation, and terminates. */
proctype dispatcher(byte t) {
    int round, observed, generation, seen_epoch, before, seen_fork, seen_callback;
    bool exchanged;
    spawn_colocated(t);
    for (round : 1 .. generations) {
        // Sleep, between the tasks and from the thread that runs them: `colocated_pool::sleep`
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
