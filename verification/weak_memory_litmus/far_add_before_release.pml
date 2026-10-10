/**
 *  @file verification/weak_memory_litmus/far_add_before_release.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief A relaxed no-return add before a release store: carried in C++, posted under far memory.
 *
 *  @verify pass sc,rc11
 *  @verify fail far: a relaxed no-return add is posted to the far cache and no release carries it;
 *      the reader acquires the flag and the data, and still sees the add unlanded
 */
#define thread_count 2
#define location_count 2
#define history_depth 2
#include "protocol.pml"

/** Adds to the data relaxed with no return, then raises the flag with a release. */
proctype writer(byte t) {
    add_no_return(t, data, order_relaxed, 1);
    store(t, flag, order_release, 1);
    landed(t)
}

/** Reads the flag and then the data, both with an acquire. */
proctype reader(byte t) { receive(t, order_acquire, order_acquire, 1) }

init { atomic { run writer(0); run reader(1) } }
