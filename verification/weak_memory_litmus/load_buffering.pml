/**
 *  @file verification/weak_memory_litmus/load_buffering.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Load buffering, forbidden in RC11 and here: a view model makes no promises.
 *
 *  Each side reads one location relaxed and then writes the other; neither may read the write the
 *  other makes after its own read. The checker plays no thread and touches no location.
 *
 *  @verify pass sc,rc11,far
 */
#define thread_count 2
#define location_count 2
#define history_depth 2
#include "protocol.pml"

int left_seen, right_seen;
byte finished;

/** Reads the data, then raises the flag. */
proctype left(byte t) {
    load(t, data, order_relaxed, left_seen);
    store(t, flag, order_relaxed, 1);
    finished++
}

/** Reads the flag, then writes the data. */
proctype right(byte t) {
    load(t, flag, order_relaxed, right_seen);
    store(t, data, order_relaxed, 1);
    finished++
}

/** Once both sides finished, checks that neither read the other's later write. */
proctype checker() {
    (finished == 2);
    assert(!(left_seen == 1 && right_seen == 1))
}

init { atomic { run left(0); run right(1); run checker() } }
