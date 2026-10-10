/**
 *  @file verification/weak_memory_litmus/release_sequence_read_modify_write.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief A release sequence continued by a third thread's relaxed read-modify-write.
 *
 *  The reader that acquires the bumped flag still acquires the writer's release before it.
 *
 *  @verify pass sc,rc11,far
 */
#define thread_count 3
#define location_count 2
#define history_depth 3
#include "protocol.pml"

/** Writes the data, then raises the flag with a release. */
proctype writer(byte t) { send(t, order_release) }

/** Bumps the raised flag with a relaxed read-modify-write. */
proctype bumper(byte t) {
    int observed;
    (newest_value(flag) == 1);
    read_modify_write(t, flag, order_relaxed, observed, observed + 1)
}

/** Reads the bumped flag with an acquire, then the data relaxed. */
proctype reader(byte t) { receive(t, order_acquire, order_relaxed, 2) }

init { atomic { run writer(0); run bumper(1); run reader(2) } }
