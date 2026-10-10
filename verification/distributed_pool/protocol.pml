/**
 *  @file verification/distributed_pool/protocol.pml
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
 *  worker, the second with one worker. Each round the dispatcher resets the second colocation's
 *  countdown and steps its epoch with a release, then the first's, and asserts the two tokens
 *  agree; the join contributes the caller's slice to the first colocation, waits for its
 *  completion step, then waits for the second's. The workers are the colocated loop: they wait on
 *  their epoch, contribute, count down with @c acq_rel, and the last one steps the epoch with a
 *  release; the fork word is dropped, since `flat_pool/` covers it.
 *
 *  Lockstep: both dispatches return the same token, and after the join both epochs are past it.
 *
 *  Every scenario beside this file defines its shape, @c tally_under_lock for slices that count
 *  under @c spin_mutex instead of writing a result, and @c remote_lags for a second colocation
 *  that starts its slices only once the caller contributed its own.
 */
#include "../weak_memory.pml"

/** The caller's thread, which plays the dispatcher and contributes to the first colocation without
 *  a result word of its own. */
#define caller 0

/** The words, by the member each stands for: each colocation's clock and countdown, one result word
 *  per worker, the token handed to the poller, and the mutex's flag and the tally under it. */
#define epoch(colocation) (colocation)
#define threads_to_sync(colocation) (2 + (colocation))
#define result(worker) (3 + (worker))
#define token 6
#define flag 7
#define tally 8

/** The contributors, the caller counted among those of the first colocation, and the round a
 *  generation belongs to. */
#define contributors 3
#define round_of(generation) (((generation) + 1) / 2)

/** The knobs: the header's choices and the rounds, each overridden by a `@verify` line. */
#ifndef generations
#define generations 1
#endif
#ifndef dispatch_before_join
#define dispatch_before_join true
#endif
#ifndef caller_first_join
#define caller_first_join true
#endif
#ifndef lock_order
#define lock_order order_acquire
#endif
#ifndef unlock_order
#define unlock_order order_release
#endif

byte workers_exited;
bool terminated;
bool caller_contributed;         // the caller's slice of the current round ran
byte ran[3 * (generations + 1)]; // slices run, per contributor and round

/** One increment of the tally under @c spin_mutex, whose @c spin_mutex::lock and
 *  @c spin_mutex::unlock live in `types.hpp`, over the caller's @p seen_flag and @p seen_tally. */
inline count_under_lock(t, seen_flag, seen_tally) {
    do
    :: read_modify_write(t, flag, lock_order, seen_flag, 1); // the exchange, unconditional as in the header
       if
       :: seen_flag == 0 -> break
       :: else -> (newest_value(flag) == 0)
       fi
    od;
    load(t, tally, order_relaxed, seen_tally);
    store(t, tally, order_relaxed, seen_tally + 1);
    store(t, flag, unlock_order, 0)
}

/** One slice: the result or the tally, the countdown, and the completion step by the last one, as
 *  @c colocated_pool::unsafe_join and @c colocated_pool::_worker_loop_body in `distributed.hpp`
 *  run it, over the caller's scratch. */
inline contribute(t, colocation, generation, before, observed, seen_flag, seen_tally) {
    atomic {
        ran[(t) * (generations + 1) + round_of(generation)]++;
        assert(ran[(t) * (generations + 1) + round_of(generation)] == 1)
    };
    if
    :: tally_under_lock -> count_under_lock(t, seen_flag, seen_tally)
    :: !tally_under_lock && (t) != caller -> store(t, result(t), order_relaxed, generation)
    :: else
    fi;
    read_modify_write(t, threads_to_sync(colocation), order_acq_rel, before, before - 1);
    assert(before > 0);
    if
    :: before == 1 -> read_modify_write(t, epoch(colocation), order_release, observed, observed + 1)
    :: else
    fi
}

/** The dispatch on one colocation: the countdown reset relaxed, the epoch stepped with a release,
 *  as @c colocated_pool::unsafe_for_threads in `distributed.hpp` does. */
inline dispatch(t, colocation, contributing, generation, observed) {
    store(t, threads_to_sync(colocation), order_relaxed, contributing);
    read_modify_write(t, epoch(colocation), order_release, observed, observed + 1);
    generation = observed + 1
}

/** The join on one colocation: stale tokens return at once, else the wait for the completion step,
 *  as in @c colocated_pool::unsafe_join in `distributed.hpp`. */
inline join(t, colocation, generation, seen_epoch) {
    load(t, epoch(colocation), order_acquire, seen_epoch);
    if
    :: seen_epoch == generation ->
        (newest_value(epoch(colocation)) != generation);
        do
        :: load(t, epoch(colocation), order_acquire, seen_epoch);
           if
           :: seen_epoch != generation -> break
           :: else
           fi
        od
    :: else
    fi
}

/** The caller: dispatches to every colocation, joins them caller first, and checks every round. */
inline dispatch_and_join(t) {
    int round, observed, generation, remote_generation, seen_epoch, before, seen_flag, seen_tally;
    for (round : 1 .. generations) {
        caller_contributed = false;
        // Every remote colocation first, then the caller's, and the tokens agree:
        // `distributed_pool::unsafe_for_threads` in `distributed.hpp`
        if
        :: dispatch_before_join ->
            dispatch(t, 1, 1, remote_generation, observed);
            dispatch(t, 0, 2, generation, observed);
            assert(remote_generation == generation)
        :: else -> dispatch(t, 0, 2, generation, observed)
        fi;
        // The token, handed to the poller as a `broadcast_join` hands it, with a synchronization
        if
        :: !tally_under_lock -> store(t, token, order_release, generation)
        :: else
        fi;
        // The caller's colocation first, its slice inside, then the rest:
        // `distributed_pool::unsafe_join` in `distributed.hpp`
        if
        :: !caller_first_join -> join(t, 1, generation, seen_epoch)
        :: else
        fi;
        load(t, epoch(0), order_acquire, seen_epoch);
        if
        :: seen_epoch == generation ->
            contribute(t, 0, generation, before, observed, seen_flag, seen_tally);
            caller_contributed = true
        :: else
        fi;
        join(t, 0, generation, seen_epoch);
        join(t, 1, generation, seen_epoch);
        if
        :: !dispatch_before_join -> dispatch(t, 1, 1, remote_generation, observed)
        :: else
        fi;
        assert(newest_value(epoch(0)) == generation + 1 && newest_value(epoch(1)) == generation + 1);
        if
        :: tally_under_lock ->
            load(t, tally, order_relaxed, seen_tally);
            assert(seen_tally == contributors * round)
        :: else
        fi
    };
    terminated = true;
    (workers_exited == 2)
}

/** A worker: the colocated loop on its own @p colocation's epoch, until the caller terminates. */
inline serve(t, colocation) {
    int last_epoch, new_epoch, before, observed, seen_flag, seen_tally;
    do
    :: (newest_value(epoch(colocation)) != last_epoch || terminated) ->
        load(t, epoch(colocation), order_acquire, new_epoch);
        if
        :: new_epoch == last_epoch -> if :: terminated -> break :: else fi
        :: new_epoch != last_epoch && (new_epoch % 2) == 1 ->
            if :: remote_lags && colocation == 1 -> (caller_contributed) :: else fi;
            contribute(t, colocation, new_epoch, before, observed, seen_flag, seen_tally);
            last_epoch = new_epoch
        :: else -> last_epoch = new_epoch
        fi
    od;
    workers_exited++
}

/** The poll over every colocation, then the results of the round it watched:
 *  @c distributed_pool::is_complete in `distributed.hpp`. */
inline poll(t) {
    int seen_first, seen_second, seen_result, watched;
    byte contributor;
    (newest_value(token) != 0);
    do
    :: load(t, token, order_acquire, watched);
       if :: watched != 0 -> break :: else fi
    od;
    // The poll waits for a step it has not seen, so a stuck pool is a stuck poller and not
    // a spinning one
    do
    :: (newest_value(epoch(0)) != watched && newest_value(epoch(1)) != watched) ->
       load(t, epoch(0), order_acquire, seen_first);
       load(t, epoch(1), order_acquire, seen_second);
       if
       :: seen_first != watched && seen_second != watched -> break
       :: else
       fi
    od;
    for (contributor : 1 .. 2) {
        load(t, result(contributor), order_relaxed, seen_result);
        assert(seen_result >= watched)
    }
}
