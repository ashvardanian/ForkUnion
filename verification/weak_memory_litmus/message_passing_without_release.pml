/**
 *  @file verification/weak_memory_litmus/message_passing_without_release.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Message passing over a relaxed flag, whose store carries nothing.
 *
 *  @verify pass sc
 *  @verify fail rc11,far: a relaxed store carries no view; the reader sees the flag raised and the
 *      data unwritten, as RC11 allows
 */
#define thread_count 2
#define location_count 2
#define history_depth 2
#include "protocol.pml"

/** Writes the data, then raises the flag relaxed. */
proctype writer(byte t) { send(t, order_relaxed) }

/** Reads the flag, then the data, both relaxed. */
proctype reader(byte t) { receive(t, order_relaxed, order_relaxed, 1) }

init { atomic { run writer(0); run reader(1) } }
