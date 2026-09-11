/**
 *  `for_n_dynamic` from `include/forkunion/types.hpp`, on `flat_pool` from `flat.hpp`: one
 *  private cursor per thread, published by the caller before the dispatch; each thread drains
 *  its own slice with relaxed adds, runs one static prong from the trailing tasks, then walks
 *  its neighbours' slices in a coprime order and drains them too, one task per claim; the
 *  join sees every task's effect.
 *
 *  A dispatcher and two workers on a caller-exclusive pool, over `tasks` tasks, five by
 *  default: three dynamic ones split into the slices [0, 2) and [2, 3), and two prongs, one
 *  per thread. The dispatcher writes each slice's end plain and its cursor with a release,
 *  resets the countdown, and steps the epoch with a release; each worker acquires the epoch,
 *  runs its prong, and drains its own slice and then the other's: a relaxed probe of the
 *  cursor against the end, then relaxed adds until the task read is at or past the end. Every
 *  task adds one to a shared result relaxed. The countdown and the completion step are the
 *  pool's, proven in `flat_pool.pml`; the dispatcher waits for the completion and reads the
 *  result after an acquire.
 *
 *  Invariants:
 *  - every task runs exactly once, and the result the join reads is the task count.
 *    `-Dwithout_dispatch_release` steps the epoch relaxed and a worker reads a slice's end
 *    from before the dispatch, skipping its tasks; `-Dwithout_completion_acquire` reads the
 *    result relaxed after the wait and sees a stale sum; `-Dwithout_claim_guard` runs the task
 *    a claim returns past the end, and the cursor runs into the next slice; `-Dpublish_relaxed`
 *    publishes the cursors relaxed and passes, since the epoch's release carries them;
 *  - no cursor passes `max(tasks, threads)`, the bound `invoke_for_n_dynamic` documents: a
 *    visit overshoots by at most one, and the probe skips a drained slice with no add at all.
 *    `-Dwithout_probe` adds on every visit, `-Dwithout_single_visit` walks every slice twice;
 *    each passes alone and the pair fails, which is the docblock's argument that either the
 *    probe or the once-per-slice walk bounds the overshoot;
 *  - `-Dtasks=2` is the regime with no dynamic task, every slice empty, and only the prongs.
 */
#include "weak_memory.pml"

// The threads, by role, apart from the processes that play them; the workers name themselves.
#define dispatcher_thread 0
#define threads 2

// The words, by the member each stands for: the pool's clock and countdown, each thread's cursor and end, and the result.
#define epoch 0
#define threads_to_sync 1
#define next(slice) (2 + (slice))
#define end(slice) (4 + (slice))
#define result 6

// The sizes: the tasks, the dynamic ones ahead of the trailing prongs, the split of the dynamic
// ones as `indexed_split` makes it, and the cursor bound the docblock proves.
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

// one task: its effect on the shared result, and the ghost that counts its runs
inline perform(t, task) {
    ran[task]++;
    read_modify_write(t, result, order_relaxed, summed_before, summed_before + 1)
}

// drain_claim: the relaxed probe, then one task per relaxed add until the claim is past the end: types.hpp:1993-2003
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

active proctype dispatcher() {
    byte task;
    int observed, seen, summed_before;
    // reset_slices_: each end plain and each cursor released, before the dispatch: types.hpp:2088-2099
    store(dispatcher_thread, end(0), order_relaxed, slice_split);
    store(dispatcher_thread, next(0), publish_order, 0);
    store(dispatcher_thread, end(1), order_relaxed, dynamic_tasks);
    store(dispatcher_thread, next(1), publish_order, slice_split);
    // unsafe_for_threads: the countdown reset relaxed, the epoch stepped: flat.hpp:494, 504
    store(dispatcher_thread, threads_to_sync, order_relaxed, threads);
    read_modify_write(dispatcher_thread, epoch, dispatch_order, observed, observed + 1);
    // unsafe_join on a caller-exclusive pool: the wait for the completion step, then the results: flat.hpp:541-544
    (newest_value(epoch) == generation + 1);
    do
    :: load(dispatcher_thread, epoch, completion_order, seen);
       if :: seen != generation -> break :: else fi
    od;
    load(dispatcher_thread, result, order_relaxed, summed_before);
    assert(summed_before == tasks);
    for (task : 0 .. tasks - 1) { assert(ran[task] == 1) }
}

active [2] proctype worker() {
    byte me, slice, visit;
    int seen_epoch, cursor, slice_last, summed_before, before, observed;
    atomic { me = workers_started + 1; workers_started++ };
    // the worker loop's acquire of the dispatch: flat.hpp:622
    (newest_value(epoch) == generation);
    do
    :: load(me, epoch, order_acquire, seen_epoch);
       if :: seen_epoch == generation -> break :: else fi
    od;
    // operator(): the static prong, then the slices in coprime order, its own first: types.hpp:2070-2084
    if
    :: dynamic_tasks + (me - 1) < tasks -> perform(me, dynamic_tasks + (me - 1))
    :: else
    fi;
    for (visit : 1 .. visits) {
        drain(me, me - 1);
        drain(me, 2 - me)
    };
    // the countdown with acq_rel, and the completion step by the last contributor: flat.hpp:637-643
    read_modify_write(me, threads_to_sync, order_acq_rel, before, before - 1);
    if
    :: before == 1 -> read_modify_write(me, epoch, order_release, observed, observed + 1)
    :: else
    fi
}
