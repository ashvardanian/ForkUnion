/**
 *  @file verification/distributed_pool/protocol.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Spin model of @c distributed_pool from `include/forkunion/distributed.hpp`: one dispatch
 *      fanned over every colocation's own epoch and countdown, in lockstep.
 *
 *  One generation token names every colocation; a dispatch claims every colocation before it
 *  publishes on any; the join runs the caller's slice on its own colocation first and waits for
 *  the others after; @c is_complete is true only once every colocation stepped past the token.
 *  A dispatcher on a caller-inclusive pool of two colocations, the first with the caller and one
 *  worker, the second with one worker. The second colocation counts one extra unit, a hold that
 *  the first colocation's thread 0 ends when its slice does, and on a caller-exclusive pool that
 *  thread 0 may fork from inside its slice.
 *
 *  - Lockstep: both colocations return the same token, and after the join both epochs are past it.
 *  - A refused dispatch published on no colocation.
 *  - Every contributor runs each generation once.
 *  - The second colocation completes only when thread 0's slice releases its hold, and no
 *    completion step fires while thread 0 forks from its slice.
 */
#include <weak_memory.pml>
#include <monitor_wait.pml>

/** The caller's thread, which plays the dispatcher and contributes to the first colocation without
 *  a result word of its own; on a caller-exclusive pool, the first colocation's first worker. */
#define caller 0

/** The words, by the member each stands for: each colocation's clock and countdown, one result
 *  word per worker, the token handed to the poller, and each colocation's
 *  @c dispatched_generation. The fork word is dropped, since <tt>flat_pool/</tt> covers it. */
#define epoch(colocation) (colocation)
#define threads_to_sync(colocation) (2 + (colocation))
#define result(worker) (3 + (worker))
#define token 6
#define dispatched_generation(colocation) (7 + (colocation))

/** The contributors of each colocation, the caller counted among those of the first, the round a
 *  generation belongs to, and the rounds the ghosts track. */
#define local_contributors 2
#define remote_contributors 1
#define round_of(generation) (((generation) + 1) / 2)
#define rounds 6

/** The knobs: the header's choices and the rounds, each overridden by a `@verify` line. */
#ifndef generations
#define generations 1
#endif
#ifndef claim_first
#define claim_first true
#endif
#ifndef dispatch_before_join
#define dispatch_before_join true
#endif
#ifndef caller_first_join
#define caller_first_join true
#endif
#ifndef hold_elsewhere
#define hold_elsewhere true
#endif
#ifndef other_slices_order
#define other_slices_order order_acquire
#endif

byte workers_exited;
bool terminated;                 // the ghost mood the capped waits notice: die was stored
bool caller_contributed;         // the caller's slice of the current round ran
byte ran[thread_count * rounds]; // slices run, per contributor and round
bool held;                       // the dispatch owes the second colocation the release of its hold
bool forking;                    // thread 0 runs the slice it forks from, not yet counted down
bool fork_next;                  // the next dispatch has thread 0 fork from its slice
byte forking_round;              // the round of that dispatch, once published, or zero

/** One slice in @p slot, over the caller's scratch: the result, the forks of slot 0 for the
 *  dispatch that asks for them, the end of the second colocation's hold by slot 0, whichever thread
 *  runs it, the countdown, which never reads 1 on a held colocation or while thread 0 forks, and
 *  the completion step by the last one, from the current epoch if @p from_current and else from
 *  @p generation, as @c colocated_pool::unsafe_join, @c colocated_pool::worker_loop_body_ and
 *  @c distributed_pool::call_as_lambda_then_end_holds_ in <tt>distributed.hpp</tt> run it. */
inline contribute(t, slot, colocation, generation, from_current, before, observed, fork_generation) {
    atomic {
        ran[(t) * rounds + round_of(generation)]++;
        assert(ran[(t) * rounds + round_of(generation)] == 1)
    };
    if
    :: (t) == 1 || (t) == 2 -> store(t, result(t), order_relaxed, generation)
    :: else
    fi;
    if
    :: (slot) == 0 && forking_round != 0 && round_of(generation) == forking_round ->
        forks_from_thread_zero(t, fork_generation, observed)
    :: else
    fi;
    if
    :: (slot) == 0 -> caller_contributed = true
    :: else
    fi;
    if
    :: (slot) == 0 && held -> end_hold(t, observed)
    :: else
    fi;
    atomic {
        read_modify_write(t, threads_to_sync(colocation), order_acq_rel, before, before - 1);
        assert(before != 1 || !forking);
        assert(colocation == 0 || before != 1 || !held);
        wake(threads_to_sync(colocation))
    };
    assert(before > 0);
    if
    :: before == 1 ->
        if
        :: from_current -> load(t, epoch(colocation), order_relaxed, observed)
        :: else -> observed = generation
        fi;
        atomic { store(t, epoch(colocation), order_release, observed + 1); wake(epoch(colocation)) }
    :: else
    fi
}

/** Exchanges one colocation's countdown from zero to @p contributing, into @p claimed:
 *  @c colocated_pool::claim_ in `distributed.hpp`. */
inline claim_colocation(t, colocation, contributing, observed, claimed) {
    observed = 0;
    compare_exchange(t, threads_to_sync(colocation), order_acquire, observed, contributing, claimed)
}

/** Waits for an even epoch on one claimed colocation, records the next as the dispatch in flight,
 *  and stores it with a release, as its only writer, returning the token:
 *  @c colocated_pool::publish_claimed_ in <tt>distributed.hpp</tt>. */
inline publish_colocation(t, colocation, generation, observed) {
    do
    :: load(t, epoch(colocation), order_acquire, observed);
       if
       :: (observed % 2) == 0 -> break
       :: else -> wait_on(t, epoch(colocation), observed, false)
       fi
    od;
    generation = observed + 1;
    store(t, dispatched_generation(colocation), order_relaxed, generation);
    atomic {
        store(t, epoch(colocation), order_release, generation);
        if
        :: colocation == 0 && fork_next -> forking_round = round_of(generation); fork_next = false
        :: else
        fi;
        wake(epoch(colocation))
    }
}

/** Every colocation claimed before any is published, the remote one with its hold unless
 *  @c hold_elsewhere is off, the claimed ones handed back if another refuses, then the remote one
 *  published before the caller's, and the tokens agree:
 *  @c distributed_pool::unsafe_for_threads in `distributed.hpp`, over the caller's scratch.
 *  Without @c claim_first, each colocation is claimed and published in turn, the remote one first;
 *  without @c dispatch_before_join, only the caller's is, and the join dispatches the remote one.
 *  @p published counts the colocations this dispatch stepped. If either refuses, it stores zero
 *  back into the one it claimed, with a release, and returns the zero token. */
inline dispatch(t, generation, remote_generation, observed, claimed, published) {
    published = 0;
    if
    :: claim_first ->
        claim_colocation(t, 0, local_contributors, observed, claimed);
        if
        :: claimed && dispatch_before_join ->
            atomic {
                claim_colocation(t, 1, remote_contributors + hold_elsewhere, observed, claimed);
                if :: claimed -> held = true :: else fi
            };
            if
            :: !claimed -> store(t, threads_to_sync(0), order_release, 0)
            :: else
            fi
        :: else
        fi;
        if
        :: claimed ->
            if
            :: dispatch_before_join ->
                publish_colocation(t, 1, remote_generation, observed);
                published++
            :: else
            fi;
            publish_colocation(t, 0, generation, observed);
            published++;
            assert(!dispatch_before_join || remote_generation == generation)
        :: else -> generation = 0
        fi
    :: else ->
        atomic {
            claim_colocation(t, 1, remote_contributors + hold_elsewhere, observed, claimed);
            if :: claimed -> held = true :: else fi
        };
        if
        :: claimed ->
            publish_colocation(t, 1, remote_generation, observed);
            published++;
            claim_colocation(t, 0, local_contributors, observed, claimed)
        :: else
        fi;
        if
        :: claimed ->
            publish_colocation(t, 0, generation, observed);
            published++;
            assert(remote_generation == generation)
        :: else -> generation = 0
        fi
    fi;
    assert(generation != 0 || published == 0)
}

/** The wait for one colocation's completion step: stale tokens return at once, as in
 *  @c colocated_pool::unsafe_join in `distributed.hpp`. */
inline join(t, colocation, generation, seen_epoch) {
    load(t, epoch(colocation), order_acquire, seen_epoch);
    if
    :: seen_epoch == generation ->
        do
        :: load(t, epoch(colocation), order_acquire, seen_epoch);
           if
           :: seen_epoch != generation -> break
           :: else -> wait_on(t, epoch(colocation), generation, false)
           fi
        od
    :: else
    fi
}

/** The caller's colocation first, its slice inside, then the rest, unless @c caller_first_join is
 *  off: @c distributed_pool::unsafe_join in `distributed.hpp`, over the caller's scratch. */
inline join_all(t, generation, seen_epoch, before, observed) {
    if
    :: !caller_first_join -> join(t, 1, generation, seen_epoch)
    :: else
    fi;
    load(t, epoch(0), order_acquire, seen_epoch);
    if
    :: seen_epoch == generation ->
        // The caller of an inclusive pool forks nothing from its slice
        contribute(t, 0, 0, generation, false, before, observed, seen_epoch)
    :: else
    fi;
    join(t, 0, generation, seen_epoch);
    join(t, 1, generation, seen_epoch)
}

/** Die, then each colocation's current epoch re-released, whose write wakes the workers asleep
 *  on it: @c colocated_pool::signal_die_ in `distributed.hpp`, over the caller's @p observed. */
inline signal_die(t, observed) {
    terminated = true;
    load(t, epoch(0), order_relaxed, observed);
    atomic { store(t, epoch(0), order_release, observed); wake(epoch(0)) };
    load(t, epoch(1), order_relaxed, observed);
    atomic { store(t, epoch(1), order_release, observed); wake(epoch(1)) }
}

/** The caller: dispatches to every colocation, joins them caller first, and checks every round. */
inline dispatch_and_join(t) {
    int round, observed, generation, remote_generation, seen_epoch, before;
    byte published;
    bool claimed;
    for (round : 1 .. generations) {
        caller_contributed = false;
        dispatch(t, generation, remote_generation, observed, claimed, published);
        // The token, handed to the poller as a `broadcast_join` hands it, with a synchronization
        store(t, token, order_release, generation);
        join_all(t, generation, seen_epoch, before, observed);
        if
        :: !dispatch_before_join ->
            claim_colocation(t, 1, remote_contributors, observed, claimed);
            publish_colocation(t, 1, remote_generation, observed)
        :: else
        fi;
        assert(newest_value(epoch(0)) == generation + 1 && newest_value(epoch(1)) == generation + 1)
    };
    signal_die(t, observed);
    (workers_exited == 2)
}

/** Thread 0 waits, over its @p observed, until one unit remains in a colocation's countdown, its
 *  own or the hold, with loads ordered by @c other_slices_order:
 *  @c colocated_pool::wait_for_other_slices_ in <tt>distributed.hpp</tt>. */
inline await_other_slices(t, colocation, observed) {
    do
    :: load(t, threads_to_sync(colocation), other_slices_order, observed);
       if
       :: observed == 1 -> break
       :: else -> wait_on(t, threads_to_sync(colocation), observed, false)
       fi
    od
}

/** Thread 0 re-arms one held colocation into @p fork_generation, as @c colocated_pool::rearm_held_
 *  in <tt>distributed.hpp</tt> does: it stores the countdown to @p contributing relaxed, then releases the
 *  epoch two past it. */
inline rearm_held(t, colocation, contributing, fork_generation, observed) {
    store(t, threads_to_sync(colocation), order_relaxed, contributing);
    load(t, epoch(colocation), order_relaxed, observed);
    fork_generation = observed + 2;
    atomic { store(t, epoch(colocation), order_release, fork_generation); wake(epoch(colocation)) }
}

/** Thread 0's fork from its slice, into @p fork_generation: every colocation's other slices waited
 *  out, both re-armed with the first last, then its own slice in place, never counted down, as the
 *  first half of @c distributed_pool::fork_from_thread_zero_ in <tt>distributed.hpp</tt> runs them. */
inline fork_in_place(t, fork_generation, observed) {
    await_other_slices(t, 0, observed);
    await_other_slices(t, 1, observed);
    rearm_held(t, 1, remote_contributors + hold_elsewhere, fork_generation, observed);
    rearm_held(t, 0, local_contributors, fork_generation, observed);
    atomic {
        ran[(t) * rounds + round_of(fork_generation)]++;
        assert(ran[(t) * rounds + round_of(fork_generation)] == 1)
    }
}

/** The second half: every colocation's other slices waited out, after which the fork that made
 *  @p fork_generation reads both workers' results of it. */
inline fork_ended(t, fork_generation, observed) {
    await_other_slices(t, 0, observed);
    await_other_slices(t, 1, observed);
    load(t, result(1), order_relaxed, observed);
    assert(observed == fork_generation);
    load(t, result(2), order_relaxed, observed);
    assert(observed == fork_generation)
}

/** Thread 0's slice of the dispatch that asks for forks: a fork, then a second whose slice forks
 *  once more, each through @c distributed_pool::unsafe_for_threads, whose join does nothing. */
inline forks_from_thread_zero(t, fork_generation, observed) {
    forking = true;
    fork_in_place(t, fork_generation, observed);
    fork_ended(t, fork_generation, observed);
    fork_in_place(t, fork_generation, observed);
    fork_in_place(t, fork_generation, observed);
    fork_ended(t, fork_generation, observed);
    fork_ended(t, fork_generation, observed);
    forking = false
}

/** Thread 0 ends the second colocation's hold once its worker's slice ended: the countdown, then
 *  the even epoch, both released: @c colocated_pool::release_hold_ in <tt>distributed.hpp</tt>. */
inline end_hold(t, observed) {
    await_other_slices(t, 1, observed);
    load(t, epoch(1), order_relaxed, observed);
    atomic { store(t, threads_to_sync(1), order_release, 0); held = false };
    atomic { store(t, epoch(1), order_release, observed + 1); wake(epoch(1)) }
}

/** Whether @p generation ended on one colocation by the time it showed the epoch read into
 *  @p observed, into @p complete: @c colocated_pool::completed_at_ in <tt>distributed.hpp</tt>. */
inline completed_at(t, colocation, generation, observed, seen_dispatched, complete) {
    load(t, dispatched_generation(colocation), order_relaxed, seen_dispatched);
    complete = (observed % 2) == 0 || seen_dispatched != (generation)
}

/** The join of a caller-exclusive pool on one colocation, which polls the token rule until
 *  @p generation reads complete: @c colocated_pool::unsafe_join in <tt>distributed.hpp</tt>. */
inline await_token(t, colocation, generation, observed, seen_dispatched, complete) {
    do
    :: load(t, epoch(colocation), order_acquire, observed);
       completed_at(t, colocation, generation, observed, seen_dispatched, complete);
       if
       :: complete -> break
       :: else -> wait_on(t, epoch(colocation), observed, false)
       fi
    od
}

/** What a dispatcher whose token reads complete sees: both workers' results of @p generation or
 *  later, read into its @p observed. */
inline check_results(t, generation, observed) {
    load(t, result(1), order_relaxed, observed);
    assert(observed >= generation);
    load(t, result(2), order_relaxed, observed);
    assert(observed >= generation)
}

/** A worker: the colocated loop on its own @p colocation's epoch, until the caller terminates. */
inline serve(t, colocation) {
    int last_epoch, new_epoch, before, observed, fork_generation;
    do
    :: load(t, epoch(colocation), order_acquire, new_epoch);
       if
       :: new_epoch == last_epoch ->
           if
           :: terminated -> break
           :: else -> wait_on(t, epoch(colocation), last_epoch, terminated)
           fi
       :: else ->
           if
           :: (new_epoch % 2) == 1 ->
               if :: remote_lags && colocation == 1 -> (caller_contributed) :: else fi;
               contribute(t, (t), colocation, new_epoch, true, before, observed, fork_generation)
           :: else
           fi;
           last_epoch = new_epoch
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
