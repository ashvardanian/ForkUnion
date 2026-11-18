/**
 *  @file verification/wait_until_changed.pml
 *  @author Ash Vardanian
 *  @date November 18, 2026
 *  @brief Spin model of the waiters' @c wait_until_changed from @c capabilities.hpp: a loop on a
 *      64-bit word whose writer moves its high half alone.
 *
 *  The waiter loads the word and leaves once it moved; otherwise it waits uncapped on the value it
 *  read. The writer's store clears every monitor armed on the word, so the @c notify_waiters after
 *  it wakes nobody and is no step here. The word is scaled down: its low half is eight bits.
 *
 *  - The waiter leaves its loop having read the written word.
 *
 *  @verify pass sc,rc11
 *  @verify stuck sc recheck_mask=255: a re-check of the low half alone reads the moved word as the
 *      one it observed, and the uncapped waiter waits forever
 */
#define thread_count 2
#define location_count 1
#define history_depth 2
#include <weak_memory.pml>
#include <monitor_wait.pml>

/** The word, and the value whose high half alone differs from zero. */
#define generation 0
#define raised 256

/** Stores the raised word with a release, which wakes every wait armed on it. */
proctype writer(byte t) { atomic { store(t, generation, order_release, raised); wake(generation) } }

/** Loops on the word until it moved, waiting uncapped on the value it read. */
proctype waiter(byte t) {
    int seen;
    do
    :: load(t, generation, order_acquire, seen);
       if
       :: seen != 0 -> break
       :: else -> wait_on(t, generation, seen, false)
       fi
    od;
    assert(seen == raised)
}

init { atomic { run writer(0); run waiter(1) } }
