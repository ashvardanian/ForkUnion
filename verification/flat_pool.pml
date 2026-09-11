/**
 *  `flat_pool` from `include/forkunion/flat.hpp`: the epoch clock, odd while a fork_state is in
 *  flight and even when idle, the threads_to_sync that names the last contributor, and the mood
 *  that puts workers to sleep and lets them go.
 *
 *  A dispatcher runs two generations on a caller-inclusive pool with two workers. Each
 *  dispatch writes the fork_state, resets the threads_to_sync relaxed, and steps the epoch with a release;
 *  every contributor reads the fork_state, writes its result relaxed, decrements the threads_to_sync with
 *  `acq_rel`, and the one that took it to zero steps the epoch again with a release. The
 *  dispatcher contributes its own slice inside the join, then waits for the completion step.
 *  The workers wait on the epoch, checking the mood only while it stands still, under a capped
 *  wait. The dispatcher spawns the workers as the colocated pool does: chill first, then the
 *  handles, then grind with a release, which the workers wait for before they read a handle;
 *  the spawn itself is a released word a fresh worker acquires, since a new thread's view is
 *  its creator's.
 *
 *  Invariants:
 *  - every worker runs each generation's slice exactly once, and reads the fork_state of that
 *    generation;
 *  - the threads_to_sync never goes below zero, and a dispatch finds it at zero and the epoch even;
 *  - a completed join sees every contributor's result: the `acq_rel` decrements chain each
 *    contributor's writes into the last one, whose release step publishes them at once.
 *    `-Dwithout_decrement_acquire` weakens the decrements to release, and the last contributor
 *    no longer acquires its peers: the join reads a stale result;
 *  - `-Dscenario=moods`: the dispatcher may sleep between dispatches, storing chill from the
 *    thread that spawned, as the header allows; the dispatch exchanges it back to grind, a
 *    worker seeing chill with the epoch still naps once and looks again, and
 *    `terminate` stores die after the last join, waits for the workers, and resets the mood and
 *    the epoch. No slice runs after die and no worker is left waiting. `-Dwithout_wait_cap`
 *    has the workers wait on the epoch alone, as an uncapped monitor would, and die is never
 *    noticed; `-Dwithout_chill_at_spawn` skips the chill before the handles are written, and a
 *    worker reads no handle; `-Dwithout_strong_wake` lets the dispatch's exchange fail
 *    spuriously, as a weak one may, and a worker still at its startup wait after a sleep is
 *    never released, so the join waits for it forever;
 *  - `-Dscenario=respawn`: after `terminate`, two fresh workers and one more generation.
 *    `-Dwithout_join_before_reset` resets the mood and the epoch before the old workers left,
 *    and one of them contributes to the new generation;
 *  - `-Dscenario=polling`: a poller holds the first generation's token, handed over with a
 *    release, polls `is_complete` with acquire loads, reads every result once it turns true,
 *    and then joins its stale token, which must return at once. `-Dwithout_complete_acquire`
 *    reads a stale result; `-Depoch_modulus=2` wraps the epoch so the stale token names the
 *    live generation, and the stale join contributes a slice that is not its own.
 */
#include "weak_memory.pml"

// The knob's values are integers, so a typo fails the range check below.
#define plain 1
#define moods 2
#define respawn 3
#define polling 4
#ifndef scenario
#define scenario plain
#endif
#if scenario < plain || scenario > polling
#error "scenario is plain, moods, respawn or polling"
#endif

// The threads, by role, apart from the processes that play them; the workers name themselves
// as they are spawned, 1 and 2 first, then 3 and 4 after a respawn.
#define dispatcher_thread 0
#define poller_thread 3

// The words, by the pool member each stands for, then one result word per worker cell, which
// a respawned worker reuses, the spawn edge, the handles, and the token handed to the poller.
#define epoch 0
#define threads_to_sync 1
#define mood 2
#define fork_state 3
#define result(worker) (4 + ((worker) - 1) % 2)
#define spawned 6
#define handles 7
#define token 8

// The scenario's sizes, the moods the pool moves through, and the epoch's width: 2^64 in the
// header, `epoch_modulus` here, small enough to wrap on request, as `generation_modulus` is in
// the gate's model.
#ifndef generations
#define generations 2
#endif
#define rounds (generations + (scenario == respawn))
#define contributors 3 // the caller's slice and the two workers'
#define grind 0
#define chill 1
#define die 2
#ifndef epoch_modulus
#define epoch_modulus 16
#endif
#if epoch_modulus % 2 != 0
#error "epoch_modulus is even, so odd epochs stay dispatches"
#endif
#define stepped(value) (((value) + 1) % epoch_modulus)

#ifdef without_decrement_acquire
#define decrement_order order_release
#else
#define decrement_order order_acq_rel
#endif

#ifdef without_complete_acquire
#define complete_order order_relaxed
#else
#define complete_order order_acquire
#endif

byte workers_started;
byte workers_exited;
byte spawns;      // how many times the pool was spawned
bool terminated;  // die was stored and not yet reset
byte in_flight;   // the round being dispatched, which every slice must read as its fork_state
byte ran[5 * (generations + 2)]; // slices run, per contributor and round

// one slice: the fork_state, the result, the decrement, and the completion step for the last one
inline contribute(t, generation) {
    assert(!terminated);
    load(t, fork_state, order_relaxed, seen_fork);
    assert(seen_fork == in_flight);
    atomic {
        ran[(t) * (generations + 2) + seen_fork]++;
        assert(ran[(t) * (generations + 2) + seen_fork] == 1)
    };
    if
    :: (t) != dispatcher_thread -> store(t, result(t), order_relaxed, generation)
    :: else
    fi;
    read_modify_write(t, threads_to_sync, decrement_order, before, before - 1);
    assert(before > 0);
    if
    :: before == 1 -> read_modify_write(t, epoch, order_release, observed, stepped(observed))
    :: else
    fi
}

// spawn as the colocated pool does: chill, the handles, then grind with a release, and the
// workers themselves, which acquire the spawn: distributed.hpp:395, 411-426, 484
inline spawn() {
#ifndef without_chill_at_spawn
    store(dispatcher_thread, mood, order_release, chill);
#endif
    spawns++;
    store(dispatcher_thread, spawned, order_release, spawns);
    run worker();
    run worker();
    store(dispatcher_thread, handles, order_relaxed, spawns);
    store(dispatcher_thread, mood, order_release, grind)
}

// terminate: die with a release after the last join, the joins, then the resets: flat.hpp:375-392
inline terminate() {
    load(dispatcher_thread, threads_to_sync, order_acquire, observed);
    assert(observed == 0);
    load(dispatcher_thread, epoch, order_acquire, observed);
    assert((observed % 2) == 0);
    atomic { store(dispatcher_thread, mood, order_release, die); terminated = true };
#ifndef without_join_before_reset
    (workers_exited == 2 * spawns);
#endif
    atomic { store(dispatcher_thread, mood, order_relaxed, grind); terminated = false };
    store(dispatcher_thread, epoch, order_relaxed, 0);
    exited_before_reset = workers_exited
}

active proctype dispatcher() {
    int round, observed, generation, seen_epoch, before, seen_fork, exited_before_reset;
    bool exchanged;
    spawn();
    for (round : 1 .. rounds) {
#if scenario == respawn
        if
        :: round == generations + 1 -> terminate(); (workers_exited == exited_before_reset); spawn()
        :: else
        fi;
#endif
#if scenario == moods
        // sleep, between the tasks and from the thread that runs them: flat.hpp:410
        if
        :: store(dispatcher_thread, mood, order_release, chill)
        :: skip
        fi;
#endif
        // unsafe_for_threads: one dispatch in flight, fully joined before the next
        load(dispatcher_thread, threads_to_sync, order_acquire, observed);
        assert(observed == 0);
        load(dispatcher_thread, epoch, order_relaxed, observed);
        assert((observed % 2) == 0);
        in_flight = round;
        store(dispatcher_thread, fork_state, order_relaxed, round);
        store(dispatcher_thread, threads_to_sync, order_relaxed, contributors);
        // the wake-up from a chill: one exchange back to grind, relaxed, and strong: flat.hpp:500-503
        observed = chill;
#ifdef without_strong_wake
        if
        :: compare_exchange(dispatcher_thread, mood, order_relaxed, observed, grind, exchanged)
        :: skip
        fi;
#else
        compare_exchange(dispatcher_thread, mood, order_relaxed, observed, grind, exchanged);
#endif
        read_modify_write(dispatcher_thread, epoch, order_release, observed, stepped(observed));
        generation = stepped(observed);
#if scenario == polling
        if :: round == 1 -> store(dispatcher_thread, token, order_release, generation) :: else fi;
#endif
        // unsafe_join: the caller's slice, then the wait for the completion step
        load(dispatcher_thread, epoch, order_acquire, seen_epoch);
        if
        :: seen_epoch == generation -> contribute(dispatcher_thread, generation)
        :: else
        fi;
        (newest_value(epoch) != generation);
        do
        :: load(dispatcher_thread, epoch, order_acquire, seen_epoch);
           if
           :: seen_epoch != generation -> break
           :: else
           fi
        od;
        // is_complete: a true result synchronizes with every contributor
        load(dispatcher_thread, result(1), order_relaxed, observed);
        assert(observed == generation);
        load(dispatcher_thread, result(2), order_relaxed, observed);
        assert(observed == generation)
    };
#if scenario == plain || scenario == polling
    store(dispatcher_thread, mood, order_release, die)
#else
    terminate()
#endif
}

proctype worker() {
    byte me;
    int last_epoch, new_epoch, seen_mood, seen_spawn, seen_handles, before, observed, seen_fork;
    atomic { me = workers_started + 1; workers_started++ };
    // the thread's creation: its view is its creator's at the spawn
    do
    :: load(me, spawned, order_acquire, seen_spawn);
       if :: seen_spawn == spawns -> break :: else fi
    od;
    // the colocated startup: wait out the chill uncapped, then find the handle: distributed.hpp:827-829, 839-850
    (newest_value(mood) != chill);
    do
    :: load(me, mood, order_acquire, seen_mood);
       if :: seen_mood != chill -> break :: else -> (newest_value(mood) != chill) fi
    od;
    load(me, handles, order_relaxed, seen_handles);
    assert(seen_handles == spawns);
    last_epoch = 0;
    do
#ifdef without_wait_cap
    :: (newest_value(epoch) != last_epoch) ->
#else
    :: (newest_value(epoch) != last_epoch || newest_value(mood) != grind) ->
#endif
       load(me, epoch, order_acquire, new_epoch);
       if
       :: new_epoch == last_epoch ->
           // the epoch stands still: the mood decides between waiting, napping and leaving: flat.hpp:626-630
           load(me, mood, order_acquire, seen_mood);
           if
           :: seen_mood == die -> break
           :: seen_mood == chill -> skip // the nap, then another look
           :: else
           fi
       :: new_epoch != last_epoch && (new_epoch % 2) == 1 -> contribute(me, new_epoch); last_epoch = new_epoch
       :: else -> last_epoch = new_epoch
       fi
    od;
    workers_exited++
}

#if scenario == polling
// a second holder of the first token: is_complete polled with acquire, the results, then the idempotent join: flat.hpp:515-517, 524-526
active proctype poller() {
    int watched, seen_epoch, seen_result, before, observed, seen_fork;
    (newest_value(token) != 0);
    do
    :: load(poller_thread, token, order_acquire, watched);
       if :: watched != 0 -> break :: else fi
    od;
    (newest_value(epoch) != watched);
    do
    :: load(poller_thread, epoch, complete_order, seen_epoch);
       if :: seen_epoch != watched -> break :: else -> (newest_value(epoch) != watched) fi
    od;
    load(poller_thread, result(1), order_relaxed, seen_result);
    assert(seen_result >= watched);
    load(poller_thread, result(2), order_relaxed, seen_result);
    assert(seen_result >= watched);
    // unsafe_join on the stale token: returns at once, unless the epoch wrapped onto it
    load(poller_thread, epoch, order_acquire, seen_epoch);
    if
    :: seen_epoch == watched -> contribute(poller_thread, watched)
    :: else
    fi
}
#endif
