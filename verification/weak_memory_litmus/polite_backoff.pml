/**
 *  @file verification/weak_memory_litmus/polite_backoff.pml
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief Polite back-off: two threads, each lowering its claim on seeing the other's, can take
 *      turns with no end.
 *
 *  Every round appends writes, which a history of @c history_depth cannot hold forever, so a
 *  livelock that keeps writing is searched for under @c sc, whose words keep no history.
 *
 *  @verify livelock sc: both raise, both see the other, both lower, and the round repeats
 */
#define thread_count 2
#define location_count 2
#define history_depth 2
#include <weak_memory.pml>

/** The words: each thread's claim, raised while it wants in. */
#define claim(t) (t)

/** Raises its claim and enters if the other's is down, else lowers it and retries. */
proctype polite(byte t) {
    int seen_claim;
    do
    :: store(t, claim(t), order_relaxed, 1);
       load(t, claim(1 - t), order_relaxed, seen_claim);
       if
       :: seen_claim == 0 -> break
       :: else -> store(t, claim(t), order_relaxed, 0)
       fi
    od
}

init { atomic { run polite(0); run polite(1) } }
