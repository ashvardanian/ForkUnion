/**
 *  @file verification/flat_pool/protocol.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Spin model of @c flat_pool from `include/forkunion/flat.hpp`: the epoch clock, the
 *      @c threads_to_sync that names the last contributor, and the mood.
 *
 *  The epoch is odd while a @c fork_state is in flight and even when idle, and the mood puts
 *  workers to sleep and lets them go.
 *
 *  A dispatcher runs @c generations generations, two by default, on a caller-inclusive pool with
 *  two workers. Each dispatch writes the @c fork_state, resets the @c threads_to_sync relaxed, and
 *  steps the epoch with a release; every contributor reads the @c fork_state, writes its result
 *  relaxed, decrements the @c threads_to_sync with @c acq_rel, and the one that took it to zero
 *  steps the epoch again with a release. The dispatcher contributes its own slice inside the join,
 *  then waits for the completion step. The workers wait on the epoch, checking the mood only while
 *  it stands still, under a capped wait. The dispatcher spawns the workers as the colocated pool
 *  does: chill first, then the handles, then grind with a release, which the workers wait for
 *  before they read a handle; the spawn itself is a released word a fresh worker acquires, since a
 *  new thread's view is its creator's.
 *
 *  Every worker runs each generation's slice exactly once, and reads the @c fork_state of that
 *  generation. The @c threads_to_sync never goes below zero, and a dispatch finds it at zero and
 *  the epoch even. A completed join sees every contributor's result: the @c acq_rel decrements
 *  chain each contributor's writes into the last one, whose release step publishes them at once.
 *
 *  Every scenario beside this file defines its shape, @c through_c_shim for dispatches the C shim
 *  wraps, and the @c worker the spawn runs.
 */
#include "../weak_memory.pml"

/** The caller's thread, which plays the dispatcher and contributes without a result word of its
 *  own; the workers are spawned as threads 1 and 2, then 3 and 4 after a respawn. */
#define caller 0

/** The words, by the pool member each stands for, then one result word per worker cell, which a
 *  respawned worker reuses, the spawn edge, the handles, the C shim's callback slot
 *  @c opaque_pool_t::current_callback, and the token handed to the poller. */
#define epoch 0
#define threads_to_sync 1
#define mood 2
#define fork_state 3
#define result(worker) (4 + ((worker) - 1) % 2)
#define spawned 6
#define handles 7
#define current_callback 8
#define token 9

/** The contributors, the caller's slice and the two workers', and the moods the pool moves
 *  through. */
#define contributors 3
#define grind 0
#define chill 1
#define die 2

/** The knobs: the header's choices, the generations, and the epoch's width, which is 2^64 in the
 *  header and small enough here to wrap on request, as @c generation_modulus is in the gate's
 *  model; each is overridden by a `@verify` line. */
#ifndef generations
#define generations 2
#endif
#ifndef epoch_modulus
#define epoch_modulus 16
#endif
#if epoch_modulus % 2 != 0
#error "epoch_modulus is even, so odd epochs stay dispatches"
#endif
#ifndef decrement_order
#define decrement_order order_acq_rel
#endif
#ifndef complete_order
#define complete_order order_acquire
#endif
#ifndef chill_at_spawn
#define chill_at_spawn true
#endif
#ifndef wait_cap
#define wait_cap true
#endif
#ifndef strong_wake
#define strong_wake true
#endif
#ifndef join_before_reset
#define join_before_reset true
#endif
#ifndef join_before_dispatch
#define join_before_dispatch true
#endif
#ifndef generation_check
#define generation_check true
#endif

#define stepped(value) (((value) + 1) % epoch_modulus)

byte workers_exited;
byte spawns;                     // how many times the pool was spawned
bool terminated;                 // die was stored and not yet reset
byte in_flight;                  // the round being dispatched, which every slice must read as its fork_state
byte current_generation;         // `opaque_pool_t::current_generation`, which only the caller touches
byte ran[5 * (generations + 2)]; // slices run, per contributor and round

/** One slice: the @c fork_state, the result, the decrement, and the completion step for the last
 *  one, over the caller's @p seen_fork, @p seen_callback, @p before and @p observed. */
inline contribute(t, generation, seen_fork, seen_callback, before, observed) {
    assert(!terminated);
    load(t, fork_state, order_relaxed, seen_fork);
    assert(seen_fork == in_flight);
    // The trampoline reads the slot on every contributor, and round r publishes callback r:
    // `opaque_pool_t::operator()` in `c/forkunion.cpp`
    if
    :: through_c_shim ->
        load(t, current_callback, order_relaxed, seen_callback);
        assert(seen_callback == ((generation) + 1) / 2)
    :: else
    fi;
    atomic {
        ran[(t) * (generations + 2) + seen_fork]++;
        assert(ran[(t) * (generations + 2) + seen_fork] == 1)
    };
    if
    :: (t) != caller -> store(t, result(t), order_relaxed, generation)
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
inline spawn(t) {
    if
    :: chill_at_spawn -> store(t, mood, order_release, chill)
    :: else
    fi;
    spawns++;
    store(t, spawned, order_release, spawns);
    run worker(2 * spawns - 1);
    run worker(2 * spawns);
    store(t, handles, order_relaxed, spawns);
    store(t, mood, order_release, grind)
}

/** Die with a release after the last join, the joins, then the resets, as @c flat_pool::terminate
 *  in `flat.hpp` does, over the caller's @p observed and @p exited_before_reset. */
inline terminate(t, observed, exited_before_reset) {
    // The header's two asserts read seq_cst; acquire is the strongest order the module spells.
    load(t, threads_to_sync, order_acquire, observed);
    assert(observed == 0);
    load(t, epoch, order_acquire, observed);
    assert((observed % 2) == 0);
    atomic { store(t, mood, order_release, die); terminated = true };
    if
    :: join_before_reset -> (workers_exited == 2 * spawns)
    :: else
    fi;
    atomic { store(t, mood, order_relaxed, grind); terminated = false };
    store(t, epoch, order_relaxed, 0);
    exited_before_reset = workers_exited
}

/** One dispatch in flight, fully joined before the next, with the wake-up from a chill as one
 *  relaxed and strong exchange back to grind: @c flat_pool::unsafe_for_threads in `flat.hpp`,
 *  over the caller's @p observed and @p exchanged. */
inline dispatch(t, round, generation, observed, exchanged) {
    load(t, threads_to_sync, order_acquire, observed);
    assert(observed == 0);
    load(t, epoch, order_relaxed, observed);
    assert((observed % 2) == 0);
    in_flight = round;
    store(t, fork_state, order_relaxed, round);
    store(t, threads_to_sync, order_relaxed, contributors);
    observed = chill;
    if
    :: compare_exchange(t, mood, order_relaxed, observed, grind, exchanged)
    :: !strong_wake -> skip
    fi;
    read_modify_write(t, epoch, order_release, observed, stepped(observed));
    generation = stepped(observed)
}

/** The caller's slice, then the wait for the completion step: @c flat_pool::unsafe_join in
 *  `flat.hpp`, over the caller's @p seen_epoch and the slice's scratch. */
inline join(t, joined, seen_epoch, seen_fork, seen_callback, before, observed) {
    load(t, epoch, order_acquire, seen_epoch);
    if
    :: seen_epoch == joined -> contribute(t, joined, seen_fork, seen_callback, before, observed)
    :: else
    fi;
    (newest_value(epoch) != joined);
    do
    :: load(t, epoch, order_acquire, seen_epoch);
       if
       :: seen_epoch != joined -> break
       :: else
       fi
    od
}

/** An empty slot or another generation's token returns at once, else the pool's join and then the
 *  clear, as @c fu_pool_unsafe_join in `c/forkunion.cpp` does, over the join's scratch. */
inline c_join(t, joined, seen_epoch, seen_fork, seen_callback, before, observed) {
    load(t, current_callback, order_relaxed, seen_callback);
    if
    :: seen_callback != 0 && (!generation_check || (joined) == current_generation) ->
        join(t, joined, seen_epoch, seen_fork, seen_callback, before, observed);
        store(t, current_callback, order_relaxed, 0)
    :: else
    fi
}

/** What a completed join sees: every worker's result of this @p generation, read into the caller's
 *  @p observed. */
inline check_results(t, generation, observed) {
    load(t, result(1), order_relaxed, observed);
    assert(observed == generation);
    load(t, result(2), order_relaxed, observed);
    assert(observed == generation)
}

/** A worker: acquires the spawn, waits out the chill, then contributes to every dispatch until
 *  die. */
inline serve(t) {
    int last_epoch, new_epoch, seen_mood, seen_spawn, seen_handles, before, observed, seen_fork, seen_callback;
    // The thread's creation: its view is its creator's at the spawn
    do
    :: load(t, spawned, order_acquire, seen_spawn);
       if :: seen_spawn == spawns -> break :: else fi
    od;
    // The colocated startup, waiting out the chill uncapped, then finding the handle:
    // `colocated_pool::_worker_loop_body` in `distributed.hpp`
    (newest_value(mood) != chill);
    do
    :: load(t, mood, order_acquire, seen_mood);
       if :: seen_mood != chill -> break :: else -> (newest_value(mood) != chill) fi
    od;
    load(t, handles, order_relaxed, seen_handles);
    assert(seen_handles == spawns);
    do
    :: (newest_value(epoch) != last_epoch || (wait_cap && newest_value(mood) != grind)) ->
       load(t, epoch, order_acquire, new_epoch);
       if
       :: new_epoch == last_epoch ->
           // The epoch stands still: the mood decides between waiting, napping and leaving,
           // as in `flat_pool::_worker_loop`
           load(t, mood, order_acquire, seen_mood);
           if
           :: seen_mood == die -> break
           :: seen_mood == chill -> skip // the nap, then another look
           :: else
           fi
       :: new_epoch != last_epoch && (new_epoch % 2) == 1 ->
           contribute(t, new_epoch, seen_fork, seen_callback, before, observed);
           last_epoch = new_epoch
       :: else -> last_epoch = new_epoch
       fi
    od;
    workers_exited++
}
