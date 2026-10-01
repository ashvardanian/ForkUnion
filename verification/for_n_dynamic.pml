/**
 *  @file verification/for_n_dynamic.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Spin model of @c invoke_for_n_dynamic from `include/forkunion/types.hpp`, behind
 *      @c flat_pool::for_n_dynamic in `flat.hpp`: one private cursor per thread, published by the
 *      caller before the dispatch.
 *
 *  Each thread drains its own slice with relaxed adds, runs one static prong from the trailing
 *  tasks, then walks its neighbours' slices in a coprime order and drains them too, one task per
 *  claim; the join sees every task's effect.
 *
 *  A dispatcher and two workers on a caller-exclusive pool, over @c tasks tasks, five by default:
 *  three dynamic ones split into the slices [0, 2) and [2, 3), and two prongs, one per thread. The
 *  dispatcher writes each slice's end plain and its cursor with a release, resets the countdown,
 *  and steps the epoch with a release; each worker acquires the epoch, runs its prong, and drains
 *  its own slice and then the other's: a relaxed probe of the cursor against the end, then relaxed
 *  adds until the task read is at or past the end. Every task adds one to a shared result relaxed.
 *  The countdown and the completion step are the pool's, proven in `flat_pool.pml`; the dispatcher
 *  waits for the completion and reads the result after an acquire.
 *
 *  Every task runs exactly once, and the result the join reads is the task count.
 *  `-Dwithout_dispatch_release` steps the epoch relaxed and a worker reads a slice's end from
 *  before the dispatch, skipping its tasks; `-Dwithout_completion_acquire` reads the result relaxed
 *  after the wait and sees a stale sum; `-Dwithout_claim_guard` runs the task a claim returns past
 *  the end, and the cursor runs into the next slice; `-Dpublish_relaxed` publishes the cursors
 *  relaxed and passes, since the epoch's release carries them.
 *
 *  No cursor passes `max(tasks, threads)`, the bound @c invoke_for_n_dynamic documents: a visit
 *  overshoots by at most one, and the probe skips a drained slice with no add at all.
 *  `-Dwithout_probe` adds on every visit, `-Dwithout_single_visit` walks every slice twice; each
 *  passes alone and the pair fails, which is the docblock's argument that either the probe or the
 *  once-per-slice walk bounds the overshoot.
 *
 *  `-Dtasks=2` is the regime with no dynamic task, every slice empty, and only the prongs.
 *
 *  No dispatch is in flight when @c invoke_for_n_dynamic::reset_slices_ rewrites the cursors, which
 *  it does in the invoker's constructor, before @c flat_pool::unsafe_for_threads can check
 *  anything. `-Dscenario=nested` has the first worker's prong call @c for_n_dynamic on its own
 *  pool, the re-entry the pool's docblock forbids: the nested constructor rewinds the cursors under
 *  the live dispatch, and its tasks run twice.
 */
#include "weak_memory.pml"

/** The knob's values are integers, so a typo fails the range check below. */
#define plain 1
#define nested 2
#ifndef scenario
#define scenario plain
#endif
#if scenario < plain || scenario > nested
#error "scenario is plain or nested"
#endif

/** The threads, by role, apart from the processes that play them; the workers name themselves. */
#define dispatcher_thread 0
#define threads 2

/** The words, by the member each stands for: the pool's clock and countdown, each thread's cursor
 *  and end, and the result. */
#define epoch 0
#define threads_to_sync 1
#define next(slice) (2 + (slice))
#define end(slice) (4 + (slice))
#define result 6

/** The sizes: the tasks, the dynamic ones ahead of the trailing prongs, the split of the dynamic
 *  ones as @c indexed_split makes it, and the cursor bound the docblock proves. */
#ifndef tasks
#define tasks 5
#endif
#if tasks > threads
#define dynamic_tasks (tasks - threads)
#define cursor_bound tasks
#else
#define dynamic_tasks 0
#define cursor_bound threads
#endif
#define slice_split ((dynamic_tasks + 1) / 2)
#define generation 1

#ifdef without_dispatch_release
#define dispatch_order order_relaxed
#else
#define dispatch_order order_release
#endif

#ifdef without_completion_acquire
#define completion_order order_relaxed
#else
#define completion_order order_acquire
#endif

#ifdef publish_relaxed
#define publish_order order_relaxed
#else
#define publish_order order_release
#endif

#ifdef without_single_visit
#define visits 2
#else
#define visits 1
#endif

byte workers_started;
byte ran[tasks];

/** One task: its effect on the shared result, and the ghost that counts its runs. */
inline perform(t, task) {
    ran[task]++;
    read_modify_write(t, result, order_relaxed, summed_before, summed_before + 1)
}

/** The relaxed probe, then one task per relaxed add until the claim is past the end, as
 *  @c drain_claim in `types.hpp` does. */
inline drain(t, slice) {
    load(t, next(slice), order_relaxed, cursor);
    load(t, end(slice), order_relaxed, slice_last);
    if
#ifndef without_probe
    :: cursor >= slice_last
#endif
    :: else ->
        do
        :: read_modify_write(t, next(slice), order_relaxed, cursor, cursor + 1);
           assert(cursor + 1 <= cursor_bound);
#ifndef without_claim_guard
           if :: cursor >= slice_last -> break :: else fi;
#endif
           if :: cursor >= tasks -> break :: else fi;
           perform(t, cursor)
        od
    fi
}

/** With no dispatch in flight, each end plain and each cursor released, as
 *  @c invoke_for_n_dynamic::reset_slices_ in `types.hpp` does. */
inline reset_slices(t) {
    assert((newest_value(epoch) % 2) == 0);
    store(t, end(0), order_relaxed, slice_split);
    store(t, next(0), publish_order, 0);
    store(t, end(1), order_relaxed, dynamic_tasks);
    store(t, next(1), publish_order, slice_split)
}

/** The caller: publishes the slices, dispatches, joins, and checks every task ran once. */
active proctype dispatcher() {
    byte task;
    int observed, seen, summed_before;
    reset_slices(dispatcher_thread);
    // The countdown reset relaxed and the epoch stepped, as `flat_pool::unsafe_for_threads` does
    store(dispatcher_thread, threads_to_sync, order_relaxed, threads);
    read_modify_write(dispatcher_thread, epoch, dispatch_order, observed, observed + 1);
    // The wait for the completion step, then the results: `flat_pool::unsafe_join` in `flat.hpp`
    (newest_value(epoch) == generation + 1);
    do
    :: load(dispatcher_thread, epoch, completion_order, seen);
       if :: seen != generation -> break :: else fi
    od;
    load(dispatcher_thread, result, order_relaxed, summed_before);
    assert(summed_before == tasks);
    for (task : 0 .. tasks - 1) { assert(ran[task] == 1) }
}

/** A worker: acquires the dispatch, runs its prong, drains every slice, and counts down. */
active [2] proctype worker() {
    byte me, slice, visit;
    int seen_epoch, cursor, slice_last, summed_before, before, observed;
    atomic { me = workers_started + 1; workers_started++ };
    // The worker loop's acquire of the dispatch: `flat_pool::_worker_loop` in `flat.hpp`
    (newest_value(epoch) == generation);
    do
    :: load(me, epoch, order_acquire, seen_epoch);
       if :: seen_epoch == generation -> break :: else fi
    od;
    // The static prong, then the slices in coprime order, its own first:
    // `invoke_for_n_dynamic::operator()` in `types.hpp`
    if
    :: dynamic_tasks + (me - 1) < tasks -> perform(me, dynamic_tasks + (me - 1))
    :: else
    fi;
#if scenario == nested
    // The prong's task dispatches on its own pool, and the nested invoker's constructor runs here
    if :: me == 1 -> reset_slices(me) :: else fi;
#endif
    for (visit : 1 .. visits) {
        drain(me, me - 1);
        drain(me, 2 - me)
    };
    // The countdown with `acq_rel`, and the completion step by the last contributor:
    // `flat_pool::_worker_loop` in `flat.hpp`
    read_modify_write(me, threads_to_sync, order_acq_rel, before, before - 1);
    if
    :: before == 1 -> read_modify_write(me, epoch, order_release, observed, observed + 1)
    :: else
    fi
}
