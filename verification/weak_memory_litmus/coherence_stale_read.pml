/**
 *  @file verification/weak_memory_litmus/coherence_stale_read.pml
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief A stale read of one location, allowed in RC11 and here: the coherence shape's negative
 *      control, which the memory model must flag.
 *
 *  The writer writes the data twice, relaxed, and the reader starts only after it finished, by a
 *  ghost that carries no view, then reads the data once. Coherence orders the reads of one location
 *  and promises nothing about how recent they are, so the reader may still read the first write.
 *  `coherence.pml` passes under every model only if the views refuse its outcome rather than never
 *  read an old write, and this shape shows they do read one; the ghost touches no location.
 *
 *  @verify pass sc
 *  @verify fail rc11,far: a reader that synchronized with nothing may read any write from its view
 *      onward; it reads the first write after the second one landed, which RC11 allows and
 *      sequential consistency forbids
 */
#define thread_count 2
#define location_count 1
#define history_depth 3
#include "protocol.pml"

bool written;

/** Writes the data twice, relaxed, then raises the ghost. */
proctype writer(byte t) {
    store(t, data, order_relaxed, 1);
    store(t, data, order_relaxed, 2);
    written = true
}

/** Once the ghost is raised, reads the data relaxed. */
proctype reader(byte t) {
    int seen;
    (written);
    load(t, data, order_relaxed, seen);
    assert(seen == 2)
}

init { atomic { run writer(0); run reader(1) } }
