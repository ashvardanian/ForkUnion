/**
 *  @file verification/weak_memory_litmus/message_passing_release_acquire.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Message passing over a release store of the flag and an acquire load of it.
 *
 *  @verify pass sc,rc11,far
 */
#define thread_count 2
#define location_count 2
#define history_depth 2
#include "protocol.pml"

/** Writes the data, then raises the flag with a release. */
proctype writer(byte t) { send(t, order_release) }

/** Reads the flag with an acquire, then the data relaxed. */
proctype reader(byte t) { receive(t, order_acquire, order_relaxed, 1) }

init { atomic { run writer(0); run reader(1) } }
