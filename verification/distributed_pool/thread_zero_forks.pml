/**
 *  @file verification/distributed_pool/thread_zero_forks.pml
 *  @author Ash Vardanian
 *  @date November 9, 2026
 *  @brief Thread 0 of a caller-exclusive pool of two colocations forking from inside its slice:
 *      twice, the second fork nesting a third, while the dispatcher only polls its token.
 *
 *  Thread 0 and worker 1 share the first colocation, worker 2 has the second, whose countdown keeps
 *  an extra unit, the dispatch's hold. Each fork waits out both colocations' other slices, re-arms
 *  both with the first last, runs thread 0's slice in place and waits again; once its slice ends,
 *  thread 0 releases the hold and counts its own unit down. The dispatcher polls both tokens, reads
 *  every result, and dispatches once more.
 *
 *  - Both colocations idle at one between the forks, and the second completes only by its hold.
 *  - Each fork's results reach thread 0, every result reaches the dispatcher, and tokens advance
 *    in lockstep and never repeat.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc hold_elsewhere=false: the second colocation counts only its worker, whose slice
 *      takes its countdown to zero under the dispatch and steps its epoch behind thread 0's back
 *  @verify fail rc11 other_slices_order=order_relaxed: relaxed, thread 0 ends the second
 *      colocation's hold without acquiring its worker's slice, and a dispatcher whose token reads
 *      complete there reads that worker's result stale
 */
#define thread_count 4
#define location_count 9
#define history_depth 13
#define remote_lags false
#include <distributed_pool/protocol.pml>

/** The thread that dispatches, past the pool's three. */
#define dispatcher_thread 3

/** A worker on its own colocation, thread 0 among them, until the dispatcher terminates. */
proctype worker(byte t; byte colocation) { serve(t, colocation) }

/** The dispatcher: a dispatch whose slice on thread 0 forks, both tokens polled and every result
 *  read, then a plain dispatch polled likewise, and the workers let go. */
proctype dispatcher(byte t) {
    int observed, generation, remote_generation, second_generation, seen_dispatched;
    byte published;
    bool claimed, complete;
    fork_next = true;
    dispatch(t, generation, remote_generation, observed, claimed, published);
    assert(generation != 0);
    await_token(t, 0, generation, observed, seen_dispatched, complete);
    await_token(t, 1, generation, observed, seen_dispatched, complete);
    check_results(t, generation + 6, observed);
    dispatch(t, second_generation, remote_generation, observed, claimed, published);
    assert(second_generation > generation + 6);
    await_token(t, 0, second_generation, observed, seen_dispatched, complete);
    await_token(t, 1, second_generation, observed, seen_dispatched, complete);
    check_results(t, second_generation, observed);
    signal_die(t, observed)
}

init { atomic { run worker(0, 0); run worker(1, 0); run worker(2, 1); run dispatcher(dispatcher_thread) } }
