/**
 *  @file verification/distributed_pool/overlap.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief A remote colocation lagging the caller: its worker waits for the caller's own slice.
 *
 *  The cast is `plain.pml`'s, and the second colocation's worker starts its slice only once the
 *  caller has contributed its own. The join as written passes, since the caller's slice runs
 *  before it waits on any remote colocation.
 *
 *  @verify pass sc generations=2
 *  @verify pass rc11
 *  @verify stuck sc caller_first_join=false: @c unsafe_join runs the caller's slice on its own
 *      colocation before it waits for any other; waiting for the second colocation first, the join
 *      waits on a worker that waits on the caller's slice, and the two wait on each other forever
 */
#define thread_count 4
#define location_count 7
#define history_depth 4
#define tally_under_lock false
#define remote_lags true
#include "protocol.pml"

/** The caller: dispatches to both colocations, joins them caller first, and checks every round. */
proctype dispatcher(byte t) { dispatch_and_join(t) }

/** A worker on its own colocation, until the caller terminates. */
proctype worker(byte t; byte colocation) { serve(t, colocation) }

/** A second holder of the token: polls both epochs, then reads every result. */
proctype poller(byte t) { poll(t) }

init { atomic { run dispatcher(caller); run worker(1, 0); run worker(2, 1); run poller(3) } }
