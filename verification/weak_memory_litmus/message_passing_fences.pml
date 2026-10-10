/**
 *  @file verification/weak_memory_litmus/message_passing_fences.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Message passing over relaxed accesses, between a release fence and an acquire fence.
 *
 *  The writer fences before it raises the flag, and the reader fences after it reads the flag.
 *
 *  @verify pass sc,rc11,far
 */
#define thread_count 2
#define location_count 2
#define history_depth 2
#include "protocol.pml"

/** Writes the data, fences, then raises the flag relaxed. */
proctype writer(byte t) {
    store(t, data, order_relaxed, 1);
    fence_release(t);
    store(t, flag, order_relaxed, 1)
}

/** Reads the flag relaxed, fences, then reads the data relaxed. */
proctype reader(byte t) {
    int seen_flag, seen_data;
    load(t, flag, order_relaxed, seen_flag);
    fence_acquire(t);
    load(t, data, order_relaxed, seen_data);
    assert(!(seen_flag == 1 && seen_data == 0))
}

init { atomic { run writer(0); run reader(1) } }
