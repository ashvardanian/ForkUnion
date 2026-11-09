/**
 *  @file verification/flat_pool/thread_zero_contenders.pml
 *  @author Ash Vardanian
 *  @date November 9, 2026
 *  @brief A rival dispatching at any moment while thread 0 forks from inside its slice: thread 0's
 *      own unit keeps the pool claimed across its forks.
 *
 *  The dispatcher hands the pool a fork whose slice on thread 0 forks once, and polls its token; a
 *  rival holding the same pool makes one plain dispatch at any moment after that one went out,
 *  since a rival before it is the plain race of <tt>contenders.pml</tt>, and polls its own token if
 *  admitted. The rival lands only after thread 0 counted down, and the second holder to finish lets
 *  the workers go.
 *
 *  - No dispatch is admitted and no completion step fires while thread 0 forks from its slice.
 *  - Every contributor runs each fork once.
 *
 *  @budget 16384: five threads with a fork from thread 0 take 46M states under rc11
 *  @verify pass sc,rc11
 */
#define thread_count 5
#define location_count 11
#define history_depth 12
#define through_c_shim false
#include <flat_pool/protocol.pml>

/** The dispatcher and the rival, past the pool's three threads, the rival's fork, and the count of
 *  finished holders, past the words of the flat pool, whose @c acq_rel steps are the program's own
 *  hand-off before @c terminate. */
#define dispatcher_thread 3
#define rival 4
#define rival_fork 1
#define finished 10

/** A worker of the caller-exclusive pool, thread 0 among them: contributes to every dispatch until
 *  die, and forks from its slice of @c forking_once. */
proctype worker(byte t) { serve(t) }

/** The second holder to finish lets the workers go, over the caller's @p observed. */
inline finish(t, observed) {
    read_modify_write(t, finished, order_acq_rel, observed, observed + 1);
    if :: observed == 1 -> signal_die(t, observed) :: else fi
}

/** A holder of the pool: one dispatch of @p fork, the rival's once the dispatcher's went out, its
 *  token polled if admitted. The polls start once the newest words read complete, since they are
 *  reads of the holder's own, which <tt>thread_zero_forks.pml</tt> checks from the dispatch on. */
proctype holder(byte t; byte fork) {
    int observed, generation, seen_dispatched;
    bool exchanged, complete;
    if :: fork == rival_fork -> (newest_value(dispatched_generation) != 0) :: else fi;
    dispatch(t, fork, generation, observed, exchanged);
    if
    :: generation != 0 ->
        ((newest_value(epoch) % 2) == 0 || newest_value(dispatched_generation) != generation);
        await_token(t, generation, observed, seen_dispatched, complete)
    :: else
    fi;
    finish(t, observed)
}

init {
    atomic {
        run worker(0); run worker(1); run worker(2);
        run holder(dispatcher_thread, forking_once); run holder(rival, rival_fork)
    }
}
