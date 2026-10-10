/**
 *  @file verification/weak_memory_litmus/coherence.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Coherence: two relaxed reads of one location never go backwards.
 *
 *  @verify pass sc,rc11,far
 */
#define thread_count 2
#define location_count 1
#define history_depth 3
#include "protocol.pml"

/** Writes the data twice, relaxed. */
proctype writer(byte t) {
    store(t, data, order_relaxed, 1);
    store(t, data, order_relaxed, 2)
}

/** Reads the data twice, relaxed. */
proctype reader(byte t) {
    int first, second;
    load(t, data, order_relaxed, first);
    load(t, data, order_relaxed, second);
    assert(!(first == 2 && second == 1))
}

init { atomic { run writer(0); run reader(1) } }
