/**
 *  @file verification/flat_pool/stale_join.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief The first token joined once more through the C shim while the second generation runs.
 *
 *  This is the stale join the C header calls idempotent. The shim claims its callback slot from
 *  empty before the pool's dispatch, and clears it only for the generation it was published with,
 *  so the stale join returns at once.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc generation_check=false: @c fu_pool_unsafe_join clears the callback slot only
 *      for the generation it published; on any token, the stale join clears it during the live
 *      dispatch, the live join then returns at once and reads the workers' results before they
 *      land, and a worker calls through the cleared slot
 */
#define thread_count 3
#define location_count 8
#define history_depth 9
#define through_c_shim true
#include "protocol.pml"

/** A worker the spawn starts: contributes to every dispatch until die. */
proctype worker(byte t) {
    start(t);
    serve(t)
}

/** The caller: spawns, then dispatches and joins every generation through the shim, the second one
 *  after a stale join of the first token, and lets the workers go. */
proctype dispatcher(byte t) {
    int round, observed, generation, seen_epoch, before, seen_fork, seen_callback, stale_generation;
    bool exchanged;
    spawn(t);
    for (round : 1 .. generations) {
        c_dispatch(t, round, generation, observed, exchanged);
        // The first token, joined once more while the second generation runs
        if
        :: round == 1 -> stale_generation = generation
        :: else -> c_join(t, stale_generation, seen_epoch, seen_fork, seen_callback, before, observed)
        fi;
        c_join(t, generation, seen_epoch, seen_fork, seen_callback, before, observed);
        check_results(t, generation, observed)
    };
    atomic { store(t, mood, order_release, die); wake(mood) }
}

init { atomic { run dispatcher(caller) } }
