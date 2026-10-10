/**
 *  @file verification/weak_memory_litmus/store_buffering.pml
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief Store buffering, allowed in RC11 and here: the negative control of `load_buffering.pml`.
 *
 *  Each side writes one location relaxed and then reads the other; both may read the initial
 *  value, as no order ties a write to a later read of another location. Load buffering passes
 *  under every model only if the views refuse it rather than miss it, and this shape, its mirror,
 *  shows they find the outcome the memory model admits; the checker touches no location.
 *
 *  @verify pass sc
 *  @verify fail rc11,far: a relaxed write carries no view to a read of another location; both
 *      sides read zero, as RC11 allows and sequential consistency forbids
 */
#define thread_count 2
#define location_count 2
#define history_depth 2
#include "protocol.pml"

int left_seen, right_seen;
byte finished;

/** Writes the data, then reads the flag. */
proctype left(byte t) {
    store(t, data, order_relaxed, 1);
    load(t, flag, order_relaxed, left_seen);
    finished++
}

/** Writes the flag, then reads the data. */
proctype right(byte t) {
    store(t, flag, order_relaxed, 1);
    load(t, data, order_relaxed, right_seen);
    finished++
}

/** Once both sides finished, checks that at least one read the other's write. */
proctype checker() {
    (finished == 2);
    assert(!(left_seen == 0 && right_seen == 0))
}

init { atomic { run left(0); run right(1); run checker() } }
