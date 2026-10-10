/**
 *  @file verification/distributed_pool/locked.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Every slice counts on a tally under @c spin_mutex, and the joiner reads every count.
 *
 *  Each slice takes the mutex around one read and one write of the tally. The lock is an acquire
 *  exchange retried after a wait on the flag, and the unlock a release store. No poller runs, and
 *  no token is handed over.
 *
 *  @verify pass rc11
 *  @verify fail rc11 lock_order=order_relaxed: @c spin_mutex::lock acquires with its exchange;
 *      relaxed, the next holder reads the tally from before the last holder's increment, and that
 *      increment is lost
 *  @verify fail rc11 unlock_order=order_relaxed: @c spin_mutex::unlock releases with its store;
 *      relaxed, the next holder reads the tally from before the last holder's increment, and that
 *      increment is lost
 */
#define thread_count 3
#define location_count 9
#define history_depth 10
#define tally_under_lock true
#define remote_lags false
#include "protocol.pml"

/** The caller: dispatches to both colocations, joins them caller first, and checks every round. */
proctype dispatcher(byte t) { dispatch_and_join(t) }

/** A worker on its own colocation, until the caller terminates. */
proctype worker(byte t; byte colocation) { serve(t, colocation) }

init { atomic { run dispatcher(caller); run worker(1, 0); run worker(2, 1) } }
