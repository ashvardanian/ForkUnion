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
 *
 *  A dispatcher on a caller-inclusive pool of two colocations, the first with the caller and one
 *  worker, the second with one worker. Each dispatch claims the first colocation's countdown and
 *  then the second's, each by exchanging it from zero with an acquire; if either refuses, it
 *  stores zero back into the one it claimed, with a release, and returns the zero token. Once it
 *  holds both, it publishes on the second colocation and then the first, each by waiting for an
 *  even epoch and stepping it with a release, and asserts the two tokens agree. The join
 *  contributes the caller's slice to the first colocation, waits for its completion step, then
 *  waits for the second's. The workers are the colocated loop: they wait on their epoch,
 *  contribute, count down with @c acq_rel, and the last one steps the epoch with a release; the
 *  fork word is dropped, since `flat_pool/` covers it, and the mood is a ghost the capped waits
 *  notice. Every wait is the @c wait_on of `monitor_wait.pml`.
 *
 *  Lockstep: both colocations return the same token, and after the join both epochs are past it.
 *  A refused dispatch published on no colocation. Every contributor runs each generation once.
 *
 *  Every scenario beside this file defines its shape, and @c remote_lags for a second colocation
 *  that starts its slices only once the caller contributed its own.
 */
#include "../weak_memory.pml"
#include "../monitor_wait.pml"

/** The caller's thread, which plays the dispatcher and contributes to the first colocation without
 *  a result word of its own. */
#define caller 0

/** The words, by the member each stands for: each colocation's clock and countdown, one result word
 *  per worker, and the token handed to the poller. */
#define epoch(colocation) (colocation)
#define threads_to_sync(colocation) (2 + (colocation))
#define result(worker) (3 + (worker))
#define token 6

/** The contributors of each colocation, the caller counted among those of the first, and the round
 *  a generation belongs to. */
#define local_contributors 2
#define remote_contributors 1
#define round_of(generation) (((generation) + 1) / 2)

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

byte workers_exited;
bool terminated;                 // the ghost mood: die was stored
bool caller_contributed;         // the caller's slice of the current round ran
byte ran[4 * (generations + 2)]; // slices run, per contributor and round

/** One slice: the result, the countdown, and the completion step by the last one, as
 *  @c colocated_pool::unsafe_join and @c colocated_pool::worker_loop_body_ in `distributed.hpp`
 *  run it, over the caller's scratch. */
inline contribute(t, colocation, generation, before, observed) {
    atomic {
        ran[(t) * (generations + 2) + round_of(generation)]++;
        assert(ran[(t) * (generations + 2) + round_of(generation)] == 1)
    };
    if
    :: (t) == 1 || (t) == 2 -> store(t, result(t), order_relaxed, generation)
    :: else
    fi;
    read_modify_write(t, threads_to_sync(colocation), order_acq_rel, before, before - 1);
    assert(before > 0);
    if
    :: before == 1 -> atomic { store(t, epoch(colocation), order_release, generation + 1); wake(epoch(colocation)) }
    :: else
    fi
}

/** Exchanges one colocation's countdown from zero to @p contributing, into @p claimed:
 *  @c colocated_pool::claim_ in `distributed.hpp`. */
inline claim_colocation(t, colocation, contributing, observed, claimed) {
    observed = 0;
    compare_exchange(t, threads_to_sync(colocation), order_acquire, observed, contributing, claimed)
}

/** Waits for an even epoch on one claimed colocation and stores the next with a release, as its
 *  only writer, returning the token: @c colocated_pool::publish_ in `distributed.hpp`. */
inline publish_colocation(t, colocation, generation, observed) {
    do
    :: load(t, epoch(colocation), order_acquire, observed);
       if
       :: (observed % 2) == 0 -> break
       :: else -> wait_on(t, epoch(colocation), observed, false)
       fi
    od;
    generation = observed + 1;
    atomic { store(t, epoch(colocation), order_release, generation); wake(epoch(colocation)) }
}

/** Every colocation claimed before any is published, the claimed ones handed back if another
 *  refuses, then the remote one published before the caller's, and the tokens agree:
 *  @c distributed_pool::unsafe_for_threads in `distributed.hpp`, over the caller's scratch.
 *  Without @c claim_first, each colocation is claimed and published in turn, the remote one first;
 *  without @c dispatch_before_join, only the caller's is, and the join dispatches the remote one.
 *  @p published counts the colocations this dispatch stepped. */
inline dispatch(t, generation, remote_generation, observed, claimed, published) {
    published = 0;
    if
    :: claim_first ->
        claim_colocation(t, 0, local_contributors, observed, claimed);
        if
        :: claimed && dispatch_before_join ->
            claim_colocation(t, 1, remote_contributors, observed, claimed);
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
        claim_colocation(t, 1, remote_contributors, observed, claimed);
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
        contribute(t, 0, generation, before, observed);
        caller_contributed = true
    :: else
    fi;
    join(t, 0, generation, seen_epoch);
    join(t, 1, generation, seen_epoch)
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
    terminated = true;
    (workers_exited == 2)
}

/** A worker: the colocated loop on its own @p colocation's epoch, until the caller terminates. */
inline serve(t, colocation) {
    int last_epoch, new_epoch, before, observed;
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
               contribute(t, colocation, new_epoch, before, observed)
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
