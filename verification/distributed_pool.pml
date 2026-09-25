/**
 *  @file verification/distributed_pool.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Spin model of @c distributed_pool from `include/forkunion/distributed.hpp`: one dispatch
 *      fanned over every colocation's own epoch and countdown, in lockstep.
 *
 *  One generation token names every colocation; the join runs the caller's slice on its own
 *  colocation first and waits for the others after; @c is_complete is true only once every
 *  colocation stepped past the token.
 *
 *  A dispatcher on a caller-inclusive pool of two colocations, the first with the caller and one
 *  worker, the second with one worker, and a poller; one round under views, two under
 *  `-Dmemory=sequential -Dgenerations=2`, where the state space allows it. Each round the
 *  dispatcher resets the second colocation's countdown and steps its epoch with a release, then
 *  the first's, and asserts the two tokens agree; the join contributes the caller's slice to the
 *  first colocation, waits for its completion step, then waits for the second's; the poller reads
 *  both epochs with acquire until both differ from the token, and then reads every result. The
 *  workers are the colocated loop: they wait on their epoch, contribute, count down with
 *  @c acq_rel, and the last one steps the epoch with a release; the fork word is dropped, since
 *  `flat_pool.pml` covers it.
 *
 *  Lockstep: both dispatches return the same token, and after the join both epochs are past it.
 *  `-Dwithout_dispatch_before_join` joins the second colocation before dispatching to it: the join
 *  returns stale, and the round ends with that colocation still running.
 *
 *  Completion: a poller that saw both epochs past the token reads every worker's result of that
 *  round, the acquire on each epoch carrying its colocation's contributors.
 *
 *  Under `-Dscenario=overlap`, the second colocation's workers start their slices only once the
 *  caller has contributed its own, a remote colocation that lags the caller. The join as written
 *  passes, since the caller's slice runs before any remote wait; `-Dwithout_caller_first_join`
 *  waits for the second colocation first, and both sides wait on each other forever, which Spin
 *  reports as an invalid end state.
 *
 *  Under `-Dscenario=locked`, every slice takes @c spin_mutex around one read and one write of a
 *  tally, the lock an acquire exchange retried after a wait on the flag, the unlock a release
 *  store; the joiner reads the tally of every contributor. `-Dwithout_lock_acquire` and
 *  `-Dwithout_unlock_release` each lose an increment.
 */
#include "weak_memory.pml"

/** The knob's values are integers, so a typo fails the range check below. */
#define plain 1
#define overlap 2
#define locked 3
#ifndef scenario
#define scenario plain
#endif
#if scenario < plain || scenario > locked
#error "scenario is plain, overlap or locked"
#endif

/** The threads, by role, apart from the processes that play them; the workers name themselves,
 *  worker 1 on the first colocation and worker 2 on the second. */
#define dispatcher_thread 0
#define poller_thread 3
#define colocation_of(worker) ((worker) - 1)

/** The words, by the member each stands for: each colocation's clock and countdown, one result word
 *  per worker, and either the token handed to the poller, or the mutex's flag and the tally under
 *  it, which the locked scenario has instead of a poller. */
#define epoch(colocation) (colocation)
#define threads_to_sync(colocation) (2 + (colocation))
#define result(worker) (3 + (worker))
#define token 7
#define flag 7
#define tally 8

/** The scenario's sizes: the rounds, and the contributors, the caller counted among those of
 *  the first colocation. */
#ifndef generations
#define generations 1
#endif
#define contributors 3
#define round_of(generation) (((generation) + 1) / 2)

#ifdef without_lock_acquire
#define lock_order order_relaxed
#else
#define lock_order order_acquire
#endif

#ifdef without_unlock_release
#define unlock_order order_relaxed
#else
#define unlock_order order_release
#endif

byte workers_started;
byte workers_exited;
bool terminated;
bool caller_contributed; // the caller's slice of the current round ran
byte ran[3 * (generations + 1)]; // slices run, per contributor and round

/** One increment of the tally under @c spin_mutex, whose @c spin_mutex::lock and
 *  @c spin_mutex::unlock live in `types.hpp`. */
inline count_under_lock(t) {
    do
    :: read_modify_write_if(t, flag, lock_order, seen_flag == 0, seen_flag, 1);
       if
       :: seen_flag == 0 -> break
       :: else -> (newest_value(flag) == 0)
       fi
    od;
    load(t, tally, order_relaxed, seen_tally);
    store(t, tally, order_relaxed, seen_tally + 1);
    store(t, flag, unlock_order, 0)
}

/** One slice: the result or the tally, the countdown, and the completion step by the last one,
 *  as @c colocated_pool::unsafe_join and @c colocated_pool::_worker_loop_body in
 *  `distributed.hpp` run it. */
inline contribute(t, colocation, generation) {
    atomic {
        ran[(t) * (generations + 1) + round_of(generation)]++;
        assert(ran[(t) * (generations + 1) + round_of(generation)] == 1)
    };
#if scenario == locked
    count_under_lock(t);
#else
    if
    :: (t) != dispatcher_thread -> store(t, result(t), order_relaxed, generation)
    :: else
    fi;
#endif
    read_modify_write(t, threads_to_sync(colocation), order_acq_rel, before, before - 1);
    assert(before > 0);
    if
    :: before == 1 -> read_modify_write(t, epoch(colocation), order_release, observed, observed + 1)
    :: else
    fi
}

/** The dispatch on one colocation: the countdown reset relaxed, the epoch stepped with a release,
 *  as @c colocated_pool::unsafe_for_threads in `distributed.hpp` does. */
inline dispatch(colocation, contributing, generation) {
    store(dispatcher_thread, threads_to_sync(colocation), order_relaxed, contributing);
    read_modify_write(dispatcher_thread, epoch(colocation), order_release, observed, observed + 1);
    generation = observed + 1
}

/** The join on one colocation: stale tokens return at once, else the wait for the completion step,
 *  as in @c colocated_pool::unsafe_join in `distributed.hpp`. */
inline join(colocation, generation) {
    load(dispatcher_thread, epoch(colocation), order_acquire, seen_epoch);
    if
    :: seen_epoch == generation ->
        (newest_value(epoch(colocation)) != generation);
        do
        :: load(dispatcher_thread, epoch(colocation), order_acquire, seen_epoch);
           if
           :: seen_epoch != generation -> break
           :: else
           fi
        od
    :: else
    fi
}

/** The caller: dispatches to every colocation, joins them caller first, and checks the round. */
active proctype dispatcher() {
    int round, observed, generation, remote_generation, seen_epoch, before, seen_flag, seen_tally;
    for (round : 1 .. generations) {
        caller_contributed = false;
        // Every remote colocation first, then the caller's, and the tokens agree:
        // `distributed_pool::unsafe_for_threads` in `distributed.hpp`
#ifdef without_dispatch_before_join
        dispatch(0, 2, generation);
#else
        dispatch(1, 1, remote_generation);
        dispatch(0, 2, generation);
        assert(remote_generation == generation);
#endif
#if scenario != locked
        // The token, handed to the poller as a `broadcast_join` hands it, with a synchronization
        store(dispatcher_thread, token, order_release, generation);
#endif
        // The caller's colocation first, its slice inside, then the rest:
        // `distributed_pool::unsafe_join` in `distributed.hpp`
#ifdef without_caller_first_join
        join(1, generation);
#endif
        load(dispatcher_thread, epoch(0), order_acquire, seen_epoch);
        if
        :: seen_epoch == generation ->
            contribute(dispatcher_thread, 0, generation);
            caller_contributed = true
        :: else
        fi;
        join(0, generation);
        join(1, generation);
#ifdef without_dispatch_before_join
        dispatch(1, 1, remote_generation);
#endif
        assert(newest_value(epoch(0)) == generation + 1 && newest_value(epoch(1)) == generation + 1);
#if scenario == locked
        load(dispatcher_thread, tally, order_relaxed, seen_tally);
        assert(seen_tally == contributors * round)
#endif
    };
    terminated = true;
    (workers_exited == 2)
}

/** A worker: the colocated loop on its own colocation's epoch, until the caller terminates. */
active [2] proctype worker() {
    byte me, colocation;
    int last_epoch, new_epoch, before, observed, seen_flag, seen_tally;
    atomic { me = workers_started + 1; workers_started++ };
    colocation = colocation_of(me);
    last_epoch = 0;
    do
    :: (newest_value(epoch(colocation)) != last_epoch || terminated) ->
        load(me, epoch(colocation), order_acquire, new_epoch);
        if
        :: new_epoch == last_epoch -> if :: terminated -> break :: else fi
        :: new_epoch != last_epoch && (new_epoch % 2) == 1 ->
#if scenario == overlap
            if :: colocation == 1 -> (caller_contributed) :: else fi;
#endif
            contribute(me, colocation, new_epoch);
            last_epoch = new_epoch
        :: else -> last_epoch = new_epoch
        fi
    od;
    workers_exited++
}

#if scenario != locked

/** The poll over every colocation, then the results of the round it watched:
 *  @c distributed_pool::is_complete in `distributed.hpp`. */
active proctype poller() {
    int seen_first, seen_second, seen_result, watched;
    byte contributor;
    (newest_value(token) != 0);
    do
    :: load(poller_thread, token, order_acquire, watched);
       if :: watched != 0 -> break :: else fi
    od;
    // The poll waits for a step it has not seen, so a stuck pool is a stuck poller and not
    // a spinning one
    do
    :: (newest_value(epoch(0)) != watched && newest_value(epoch(1)) != watched) ->
       load(poller_thread, epoch(0), order_acquire, seen_first);
       load(poller_thread, epoch(1), order_acquire, seen_second);
       if
       :: seen_first != watched && seen_second != watched -> break
       :: else
       fi
    od;
    for (contributor : 1 .. 2) {
        load(poller_thread, result(contributor), order_relaxed, seen_result);
        assert(seen_result >= watched)
    }
}
#endif
