/**
 *  @file verification/flat_pool/redispatch.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Every dispatch through the C shim, which publishes its callback in a plain slot.
 *
 *  The shim stores the callback in @c opaque_pool_t::current_callback before the pool's dispatch,
 *  and clears it after the pool's join. Every contributor reads the slot through the trampoline,
 *  and runs the callback that its generation published.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc join_before_dispatch=false: the C header requires each dispatch joined before
 *      the next; dispatching the next generation first, the slot changes under the running
 *      workers, and the dispatch finds the countdown above zero
 */
#define thread_count 3
#define location_count 9
#define history_depth 9
#define through_c_shim true
#include "protocol.pml"

/** A worker the spawn starts: contributes to every dispatch until die. */
proctype worker(byte t) { serve(t) }

/** The caller: spawns, then publishes the slot, dispatches and joins every generation through the
 *  shim, and lets the workers go. */
proctype dispatcher(byte t) {
    int round, observed, generation, seen_epoch, before, seen_fork, seen_callback;
    bool exchanged;
    spawn(t);
    for (round : 1 .. generations) {
        // The slot first, then the pool's own dispatch: `fu_pool_unsafe_for_threads`
        store(t, current_callback, order_relaxed, round);
        dispatch(t, round, generation, observed, exchanged);
        current_generation = generation;
        if
        :: join_before_dispatch || round == generations ->
            c_join(t, generation, seen_epoch, seen_fork, seen_callback, before, observed);
            check_results(t, generation, observed)
        :: else
        fi
    };
    store(t, mood, order_release, die)
}

init { atomic { run dispatcher(caller) } }
