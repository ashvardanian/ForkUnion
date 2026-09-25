/**
 *  @file verification/flat_pool.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Spin model of @c flat_pool from `include/forkunion/flat.hpp`: the epoch clock, the
 *      @c threads_to_sync that names the last contributor, and the mood.
 *
 *  The epoch is odd while a @c fork_state is in flight and even when idle, and the mood puts
 *  workers to sleep and lets them go.
 *
 *  A dispatcher runs two generations on a caller-inclusive pool with two workers. Each dispatch
 *  writes the @c fork_state, resets the @c threads_to_sync relaxed, and steps the epoch with a
 *  release; every contributor reads the @c fork_state, writes its result relaxed, decrements the
 *  @c threads_to_sync with @c acq_rel, and the one that took it to zero steps the epoch again with
 *  a release. The dispatcher contributes its own slice inside the join, then waits for the
 *  completion step. The workers wait on the epoch, checking the mood only while it stands still,
 *  under a capped wait. The dispatcher spawns the workers as the colocated pool does: chill first,
 *  then the handles, then grind with a release, which the workers wait for before they read a
 *  handle; the spawn itself is a released word a fresh worker acquires, since a new thread's view
 *  is its creator's.
 *
 *  Every worker runs each generation's slice exactly once, and reads the @c fork_state of that
 *  generation. The @c threads_to_sync never goes below zero, and a dispatch finds it at zero and
 *  the epoch even.
 *
 *  A completed join sees every contributor's result: the @c acq_rel decrements chain each
 *  contributor's writes into the last one, whose release step publishes them at once.
 *  `-Dwithout_decrement_acquire` weakens the decrements to release, and the last contributor no
 *  longer acquires its peers: the join reads a stale result.
 *
 *  Under `-Dscenario=moods`, the dispatcher may sleep between dispatches, storing chill from the
 *  thread that spawned, as the header allows; the dispatch exchanges it back to grind, a worker
 *  seeing chill with the epoch still naps once and looks again, and @c terminate stores die after
 *  the last join, waits for the workers, and resets the mood and the epoch. No slice runs after die
 *  and no worker is left waiting. `-Dwithout_wait_cap` has the workers wait on the epoch alone, as
 *  an uncapped monitor would, and die is never noticed; `-Dwithout_chill_at_spawn` skips the chill
 *  before the handles are written, and a worker reads no handle; `-Dwithout_strong_wake` lets the
 *  dispatch's exchange fail spuriously, as a weak one may, and a worker still at its startup wait
 *  after a sleep is never released, so the join waits for it forever.
 *
 *  Under `-Dscenario=respawn`, after @c terminate, two fresh workers and one more generation.
 *  `-Dwithout_join_before_reset` resets the mood and the epoch before the old workers left, and one
 *  of them contributes to the new generation.
 *
 *  Under `-Dscenario=polling`, a poller holds the first generation's token, handed over with a
 *  release, polls @c is_complete with acquire loads, reads every result once it turns true, and
 *  then joins its stale token, which must return at once. `-Dwithout_complete_acquire` reads a
 *  stale result; `-Depoch_modulus=2` wraps the epoch so the stale token names the live generation,
 *  and the stale join contributes a slice that is not its own.
 *
 *  Under `-Dscenario=redispatch`, every dispatch goes through the C shim, which stores the caller's
 *  callback in the opaque pool's plain slot before the pool's own dispatch, and clears it after the
 *  pool's join; every contributor reads the slot through the trampoline and runs the callback its
 *  generation published. `-Dwithout_join_before_dispatch` dispatches the next generation before
 *  joining this one, which the C header forbids: the slot changes under the running workers, and
 *  the dispatch finds the countdown above zero.
 *
 *  Under `-Dscenario=stale_join`, the C shim again, and the first generation's token is joined
 *  once more while the second generation runs, the stale join the C header calls idempotent; the
 *  shim clears the slot only for the generation it was published with, so the stale join returns
 *  at once. `-Dwithout_generation_check` reads any set slot as the joined one: the stale join
 *  clears it, the live join then returns without the caller's slice, and a worker trampolines into
 *  the cleared slot.
 */
#include "weak_memory.pml"

/** The knob's values are integers, so a typo fails the range check below. */
#define plain 1
#define moods 2
#define respawn 3
#define polling 4
#define redispatch 5
#define stale_join 6
#ifndef scenario
#define scenario plain
#endif
#if scenario < plain || scenario > stale_join
#error "scenario is plain, moods, respawn, polling, redispatch or stale_join"
#endif
#define through_c_shim (scenario == redispatch || scenario == stale_join)

/** The threads, by role, apart from the processes that play them; the workers name themselves as
 *  they are spawned, 1 and 2 first, then 3 and 4 after a respawn. */
#define dispatcher_thread 0
#define poller_thread 3

/** The words, by the pool member each stands for, then one result word per worker cell, which a
 *  respawned worker reuses, the spawn edge, the handles, and either the token handed to the poller
 *  or the C shim's callback slot, @c opaque_pool_t::current_callback, which never meet. */
#define epoch 0
#define threads_to_sync 1
#define mood 2
#define fork_state 3
#define result(worker) (4 + ((worker) - 1) % 2)
#define spawned 6
#define handles 7
#define token 8
#define callback 8

/** The scenario's sizes, the moods the pool moves through, and the epoch's width: 2^64 in the
 *  header, @c epoch_modulus here, small enough to wrap on request, as @c generation_modulus is in
 *  the gate's model. */
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

/** One slice: the @c fork_state, the result, the decrement, and the completion step for
 *  the last one. */
inline contribute(t, generation) {
    assert(!terminated);
    load(t, fork_state, order_relaxed, seen_fork);
    assert(seen_fork == in_flight);
#if through_c_shim
    // The trampoline reads the slot on every contributor, and round r publishes callback r:
    // `opaque_pool_t::operator()` in `c/forkunion.cpp`
    load(t, callback, order_relaxed, seen_callback);
    assert(seen_callback == ((generation) + 1) / 2);
#endif
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

/** Spawns as the colocated pool does: chill, the handles, then grind with a release, and the
 *  workers themselves, which acquire the spawn: @c colocated_pool::spawn in `distributed.hpp`. */
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

/** Die with a release after the last join, the joins, then the resets:
 *  @c flat_pool::terminate in `flat.hpp`. */
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

/** The caller's slice, then the wait for the completion step:
 *  @c flat_pool::unsafe_join in `flat.hpp`. */
inline join(joined) {
    load(dispatcher_thread, epoch, order_acquire, seen_epoch);
    if
    :: seen_epoch == joined -> contribute(dispatcher_thread, joined)
    :: else
    fi;
    (newest_value(epoch) != joined);
    do
    :: load(dispatcher_thread, epoch, order_acquire, seen_epoch);
       if
       :: seen_epoch != joined -> break
       :: else
       fi
    od
}

/** Whether a C join runs the pool's join: only for a set slot, and only for the generation it
 *  was published with. */
#ifdef without_generation_check
#define joins_the_slot(joined) (seen_callback != 0)
#else
#define joins_the_slot(joined) (seen_callback != 0 && (joined) == published_generation)
#endif

/** An empty slot or another generation's token returns at once, else the pool's join, then the
 *  clear: @c fu_pool_unsafe_join in `c/forkunion.cpp`. */
inline c_join(joined) {
    load(dispatcher_thread, callback, order_relaxed, seen_callback);
    if
    :: joins_the_slot(joined) -> join(joined); store(dispatcher_thread, callback, order_relaxed, 0)
    :: else
    fi
}

/** The join and what a completed join sees: every worker's result of this generation. */
inline join_and_check() {
#if through_c_shim
    c_join(generation);
#else
    join(generation);
#endif
    load(dispatcher_thread, result(1), order_relaxed, observed);
    assert(observed == generation);
    load(dispatcher_thread, result(2), order_relaxed, observed);
    assert(observed == generation)
}

/** The caller: spawns, dispatches and joins every round, then lets the workers go. */
active proctype dispatcher() {
    int round, observed, generation, seen_epoch, before, seen_fork, exited_before_reset, seen_callback, stale_generation, published_generation;
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
        // Sleep, between the tasks and from the thread that runs them: `flat_pool::sleep`
        if
        :: store(dispatcher_thread, mood, order_release, chill)
        :: skip
        fi;
#endif
#if through_c_shim
        // The slot first, then the pool's own dispatch: `fu_pool_unsafe_for_threads`
        store(dispatcher_thread, callback, order_relaxed, round);
#endif
        // One dispatch in flight, fully joined before the next: `flat_pool::unsafe_for_threads`
        load(dispatcher_thread, threads_to_sync, order_acquire, observed);
        assert(observed == 0);
        load(dispatcher_thread, epoch, order_relaxed, observed);
        assert((observed % 2) == 0);
        in_flight = round;
        store(dispatcher_thread, fork_state, order_relaxed, round);
        store(dispatcher_thread, threads_to_sync, order_relaxed, contributors);
        // The wake-up from a chill: one exchange back to grind, relaxed, and strong, also in
        // `flat_pool::unsafe_for_threads`
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
#if through_c_shim
        published_generation = generation;
#endif
#if scenario == polling
        if :: round == 1 -> store(dispatcher_thread, token, order_release, generation) :: else fi;
#endif
#if scenario == stale_join
        // The first token, joined once more while the second generation runs
        if :: round == 1 -> stale_generation = generation :: else -> c_join(stale_generation) fi;
#endif
#ifdef without_join_before_dispatch
        if :: round == rounds -> join_and_check() :: else fi
#else
        join_and_check()
#endif
    };
#if scenario == plain || scenario == polling || through_c_shim
    store(dispatcher_thread, mood, order_release, die)
#else
    terminate()
#endif
}

/** A worker: acquires the spawn, waits out the chill, then contributes to every
 *  dispatch until die. */
proctype worker() {
    byte me;
    int last_epoch, new_epoch, seen_mood, seen_spawn, seen_handles, before, observed, seen_fork, seen_callback;
    atomic { me = workers_started + 1; workers_started++ };
    // The thread's creation: its view is its creator's at the spawn
    do
    :: load(me, spawned, order_acquire, seen_spawn);
       if :: seen_spawn == spawns -> break :: else fi
    od;
    // The colocated startup, waiting out the chill uncapped, then finding the handle:
    // `colocated_pool::_worker_loop_body` in `distributed.hpp`
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
           // The epoch stands still: the mood decides between waiting, napping and leaving,
           // as in `flat_pool::_worker_loop`
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

/** A second holder of the first token: @c flat_pool::is_complete polled with acquire, the results,
 *  then the idempotent @c flat_pool::unsafe_join. */
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
    // The join on the stale token returns at once, unless the epoch wrapped onto it
    load(poller_thread, epoch, order_acquire, seen_epoch);
    if
    :: seen_epoch == watched -> contribute(poller_thread, watched)
    :: else
    fi
}
#endif
