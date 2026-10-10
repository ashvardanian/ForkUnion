/**
 *  @file verification/distributed_pool/plain.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief The lockstep dispatch over two colocations, beside a poller that reads every result.
 *
 *  The dispatcher hands the token to the poller with a release, as a @c broadcast_join hands it;
 *  the poller reads both epochs with acquire until both differ from the token, and then reads every
 *  result. One round under @c rc11, and two under @c sc, where the state space allows it.
 *
 *  Completion: a poller that saw both epochs past the token reads every worker's result of that
 *  round, the acquire on each epoch carrying its colocation's contributors.
 *
 *  @verify pass sc generations=2
 *  @verify pass rc11
 *  @verify fail sc dispatch_before_join=false: @c unsafe_for_threads dispatches every remote
 *      colocation before the join; joining the second colocation before dispatching to it, the
 *      join returns stale, the round ends with that colocation still running, and the poller reads
 *      its worker's result unwritten
 */
#define thread_count 4
#define location_count 7
#define history_depth 4
#define tally_under_lock false
#define remote_lags false
#include "protocol.pml"

/** The caller: dispatches to both colocations, joins them caller first, and checks every round. */
proctype dispatcher(byte t) { dispatch_and_join(t) }

/** A worker on its own colocation, until the caller terminates. */
proctype worker(byte t; byte colocation) { serve(t, colocation) }

/** A second holder of the token: polls both epochs, then reads every result. */
proctype poller(byte t) { poll(t) }

init { atomic { run dispatcher(caller); run worker(1, 0); run worker(2, 1); run poller(3) } }
