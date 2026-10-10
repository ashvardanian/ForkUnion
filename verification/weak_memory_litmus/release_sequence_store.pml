/**
 *  @file verification/weak_memory_litmus/release_sequence_store.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief A release sequence broken by a third thread's relaxed store.
 *
 *  The reader that acquires the overwritten flag acquires nothing of the writer's.
 *
 *  @verify pass sc
 *  @verify fail rc11,far: only a read-modify-write continues a release sequence; the relaxed store
 *      ends it, and the reader that acquires the store sees the data unwritten
 */
#define thread_count 3
#define location_count 2
#define history_depth 3
#include "protocol.pml"

/** Writes the data, then raises the flag with a release. */
proctype writer(byte t) { send(t, order_release) }

/** Overwrites the raised flag with a relaxed store. */
proctype bumper(byte t) {
    (newest_value(flag) == 1);
    store(t, flag, order_relaxed, 2)
}

/** Reads the overwritten flag with an acquire, then the data relaxed. */
proctype reader(byte t) { receive(t, order_acquire, order_relaxed, 2) }

init { atomic { run writer(0); run bumper(1); run reader(2) } }
