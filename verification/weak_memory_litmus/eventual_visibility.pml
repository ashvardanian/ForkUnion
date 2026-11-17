/**
 *  @file verification/weak_memory_litmus/eventual_visibility.pml
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief Eventual visibility: a relaxed spin on the flag ends once the flag is raised.
 *
 *  With staleness unbounded, the spinner may re-read the initial flag forever, an execution C++
 *  rules out. The search for cycles bounds it and finds none, and without the bound it finds that
 *  one, which proves the bound is what tells the two apart.
 *
 *  @verify pass sc,rc11,far
 *  @verify livelock rc11 staleness_bound=0: unbounded, the spinner re-reads the flag's initial
 *      value forever and never sees it raised
 */
#define thread_count 2
#define location_count 2
#define history_depth 2
#include <weak_memory_litmus/protocol.pml>

/** Raises the flag, relaxed. */
proctype raiser(byte t) { store(t, flag, order_relaxed, 1) }

/** Re-reads the flag, relaxed, until it reads raised. */
proctype spinner(byte t) {
    int seen_flag;
    do
    :: load(t, flag, order_relaxed, seen_flag);
       if
       :: seen_flag == 1 -> break
       :: else
       fi
    od
}

init { atomic { run raiser(0); run spinner(1) } }
