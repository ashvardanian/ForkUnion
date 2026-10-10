/**
 *  @file verification/flat_pool/polling.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief A poller holding the first token: @c is_complete, the results, and the stale join.
 *
 *  The dispatcher hands the first token over with a release. The poller polls @c is_complete with
 *  acquire loads, reads every result once it turns true, and then joins its stale token, which
 *  must return at once.
 *
 *  @verify pass sc
 *  @verify pass rc11 generations=1
 *  @verify fail rc11 generations=1 complete_order=order_relaxed: @c is_complete acquires the epoch;
 *      relaxed, the poller that saw the completion step reads a stale result
 *  @verify fail sc epoch_modulus=2: the epoch's width aliases only at the debug widths, as the
 *      header prices; modulo 2, the stale token names the live generation, and the stale join
 *      contributes a slice that is not its own, taking the countdown past zero
 */
#define thread_count 4
#define location_count 10
#define history_depth 5
#define through_c_shim false
#include "protocol.pml"

/** A worker the spawn starts: contributes to every dispatch until die. */
proctype worker(byte t) { serve(t) }

/** The caller: spawns, dispatches every generation, hands the first token over, joins, and lets the
 *  workers go. */
proctype dispatcher(byte t) {
    int round, observed, generation, seen_epoch, before, seen_fork, seen_callback;
    bool exchanged;
    spawn(t);
    for (round : 1 .. generations) {
        dispatch(t, round, generation, observed, exchanged);
        if :: round == 1 -> store(t, token, order_release, generation) :: else fi;
        join(t, generation, seen_epoch, seen_fork, seen_callback, before, observed);
        check_results(t, generation, observed)
    };
    store(t, mood, order_release, die)
}

/** A second holder of the first token: @c flat_pool::is_complete polled, the results, then the
 *  idempotent @c flat_pool::unsafe_join. */
proctype poller(byte t) {
    int watched, seen_epoch, seen_result, before, observed, seen_fork, seen_callback;
    (newest_value(token) != 0);
    do
    :: load(t, token, order_acquire, watched);
       if :: watched != 0 -> break :: else fi
    od;
    (newest_value(epoch) != watched);
    do
    :: load(t, epoch, complete_order, seen_epoch);
       if :: seen_epoch != watched -> break :: else -> (newest_value(epoch) != watched) fi
    od;
    load(t, result(1), order_relaxed, seen_result);
    assert(seen_result >= watched);
    load(t, result(2), order_relaxed, seen_result);
    assert(seen_result >= watched);
    // The join on the stale token returns at once, unless the epoch wrapped onto it
    load(t, epoch, order_acquire, seen_epoch);
    if
    :: seen_epoch == watched -> contribute(t, watched, seen_fork, seen_callback, before, observed)
    :: else
    fi
}

init { atomic { run dispatcher(caller); run poller(3) } }
