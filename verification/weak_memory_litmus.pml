/**
 *  @file verification/weak_memory_litmus.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Calibration of `weak_memory.pml`: the classic shapes, one per `-Dshape=` name, each
 *      asserting the outcome the C++ model forbids.
 *
 *  A shape whose outcome the model admits fails its assertion, and `check.sh` expects exactly the
 *  failures RC11 would produce; `-Dshape=` picks one. Threads are 0, 1 and 2; locations are @c data
 *  and @c flag.
 */
#include "weak_memory.pml"

/** The threads, by position: the shapes are symmetric, so the processes name the roles. */
#define first_thread 0
#define second_thread 1
#define third_thread 2

/** The words. */
#define data 0
#define flag 1

/** The shapes, one integer each, so `-Dshape=` names one and a typo fails the range check. */
#define message_passing_without_release 1
#define message_passing_release_acquire 2
#define message_passing_fences 3
#define release_sequence_read_modify_write 4
#define release_sequence_store 5
#define far_add_before_release 6
#define coherence 7
#define load_buffering 8
#ifndef shape
#error "shape names one of the eight above"
#endif
#if shape < message_passing_without_release || shape > load_buffering
#error "shape names one of the eight above"
#endif

/*  The flag's store carries nothing: the reader may see the flag and stale data. */
#if shape == message_passing_without_release
active proctype writer() { store(first_thread, data, order_relaxed, 1); store(first_thread, flag, order_relaxed, 1) }
active proctype reader() {
    int seen_flag, seen_data;
    load(second_thread, flag, order_relaxed, seen_flag);
    load(second_thread, data, order_relaxed, seen_data);
    assert(!(seen_flag == 1 && seen_data == 0))
}
#endif

#if shape == message_passing_release_acquire
active proctype writer() { store(first_thread, data, order_relaxed, 1); store(first_thread, flag, order_release, 1) }
active proctype reader() {
    int seen_flag, seen_data;
    load(second_thread, flag, order_acquire, seen_flag);
    load(second_thread, data, order_relaxed, seen_data);
    assert(!(seen_flag == 1 && seen_data == 0))
}
#endif

#if shape == message_passing_fences
active proctype writer() { store(first_thread, data, order_relaxed, 1); fence_release(first_thread); store(first_thread, flag, order_relaxed, 1) }
active proctype reader() {
    int seen_flag, seen_data;
    load(second_thread, flag, order_relaxed, seen_flag);
    fence_acquire(second_thread);
    load(second_thread, data, order_relaxed, seen_data);
    assert(!(seen_flag == 1 && seen_data == 0))
}
#endif

/*  A relaxed read-modify-write by a third thread extends the release sequence. */
#if shape == release_sequence_read_modify_write
active proctype writer() { store(first_thread, data, order_relaxed, 1); store(first_thread, flag, order_release, 1) }
active proctype bumper() {
    int observed;
    (newest_value(flag) == 1);
    read_modify_write(third_thread, flag, order_relaxed, observed, observed + 1)
}
active proctype reader() {
    int seen_flag, seen_data;
    load(second_thread, flag, order_acquire, seen_flag);
    load(second_thread, data, order_relaxed, seen_data);
    assert(!(seen_flag == 2 && seen_data == 0))
}
#endif

/*  A relaxed store by a third thread breaks it: the reader of 2 acquires nothing. */
#if shape == release_sequence_store
active proctype writer() { store(first_thread, data, order_relaxed, 1); store(first_thread, flag, order_release, 1) }
active proctype bumper() { (newest_value(flag) == 1); store(third_thread, flag, order_relaxed, 2) }
active proctype reader() {
    int seen_flag, seen_data;
    load(second_thread, flag, order_acquire, seen_flag);
    load(second_thread, data, order_relaxed, seen_data);
    assert(!(seen_flag == 2 && seen_data == 0))
}
#endif

/*  A relaxed no-return add followed by a release store: carried in C++, posted under far memory. */
#if shape == far_add_before_release
active proctype writer() { add_no_return(first_thread, data, order_relaxed, 1); store(first_thread, flag, order_release, 1); landed(first_thread) }
active proctype reader() {
    int seen_flag, seen_data;
    load(second_thread, flag, order_acquire, seen_flag);
    load(second_thread, data, order_acquire, seen_data);
    assert(!(seen_flag == 1 && seen_data == 0))
}
#endif

/*  Two reads of one location never go backwards. */
#if shape == coherence
active proctype writer() { store(first_thread, data, order_relaxed, 1); store(first_thread, data, order_relaxed, 2) }
active proctype reader() {
    int first, second;
    load(second_thread, data, order_relaxed, first);
    load(second_thread, data, order_relaxed, second);
    assert(!(first == 2 && second == 1))
}
#endif

/*  Forbidden in RC11 and here: a view model makes no promises. */
#if shape == load_buffering
int left_seen, right_seen;
byte finished;
active proctype left() { load(first_thread, data, order_relaxed, left_seen); store(first_thread, flag, order_relaxed, 1); finished++ }
active proctype right() { load(second_thread, flag, order_relaxed, right_seen); store(second_thread, data, order_relaxed, 1); finished++ }
active proctype checker() { (finished == 2); assert(!(left_seen == 1 && right_seen == 1)) }
#endif
