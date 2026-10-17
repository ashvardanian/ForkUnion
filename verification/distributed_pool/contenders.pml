/**
 *  @file verification/distributed_pool/contenders.pml
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief Two threads dispatching on one pool of two colocations at once: a dispatch runs on every
 *      colocation or on none.
 *
 *  The caller and a rival thread holding the same pool each make one dispatch and join it if it
 *  was admitted. A dispatch refused by either colocation hands back the countdown it claimed on
 *  the other, so the next dispatch is admitted once the running one completes; once both
 *  finished, the caller lets the workers go.
 *
 *  A refused dispatch published on no colocation, the two tokens of an admitted one agree, and
 *  every contributor runs each generation once.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc claim_first=false: @c distributed_pool::unsafe_for_threads claims every
 *      colocation before it publishes on any; claiming and publishing each in turn, the rival
 *      publishes on the remote colocation, finds the caller's still held and is refused, and
 *      leaves its fork running there with nobody to join it
 */
#define thread_count 4
#define location_count 8
#define history_depth 7
#define remote_lags false
#include "protocol.pml"

/** The thread that dispatches beside the caller, and the count of finished holders, past the words
 *  of the pool, whose @c acq_rel steps are the program's own hand-off before @c terminate. */
#define rival 3
#define finished 7

/** A holder of the pool: one dispatch, joined caller first if admitted; the second to finish lets
 *  the workers go. */
proctype dispatcher(byte t) {
    int observed, generation, remote_generation, seen_epoch, before;
    byte published;
    bool claimed;
    dispatch(t, generation, remote_generation, observed, claimed, published);
    if
    :: generation != 0 -> join_all(t, generation, seen_epoch, before, observed)
    :: else
    fi;
    read_modify_write(t, finished, order_acq_rel, before, before + 1);
    if :: before == 1 -> signal_die(t, observed) :: else fi
}

/** A worker on its own colocation, until the caller terminates. */
proctype worker(byte t; byte colocation) { serve(t, colocation) }

init { atomic { run dispatcher(caller); run dispatcher(rival); run worker(1, 0); run worker(2, 1) } }
