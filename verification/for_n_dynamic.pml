/**
 *  @file verification/for_n_dynamic.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Spin model of @c invoke_for_n_dynamic from `include/forkunion/types.hpp`.
 *
 *  Behind @c flat_pool::for_n_dynamic in `flat.hpp`: one private cursor per thread, published by
 *  the caller before the dispatch. Each thread drains its own slice with relaxed adds, runs one
 *  static prong from the trailing tasks, then walks its neighbours' slices in a coprime order and
 *  drains them too, one task per claim; the join sees every task's effect.
 *
 *  A dispatcher and two workers on a caller-exclusive pool, over @c tasks tasks, five by default:
 *  three dynamic ones split into the slices [0, 2) and [2, 3), and two prongs, one per thread. The
 *  dispatcher writes each slice's end plain and its cursor with a release, resets the countdown,
 *  and steps the epoch with a release; each worker acquires the epoch, runs its prong, and drains
 *  its own slice and then the other's: a relaxed probe of the cursor against the end, then relaxed
 *  adds until the task read is at or past the end. Every task adds one to a shared result relaxed.
 *  The countdown and the completion step are the pool's, proven in `flat_pool/`; the dispatcher
 *  waits for the completion and reads the result after an acquire.
 *
 *  Every task runs exactly once, and the result the join reads is the task count.
 *
 *  No cursor passes `max(tasks, threads)`, the bound @c invoke_for_n_dynamic documents, which is
 *  @c tasks here, since every thread has a prong: a visit overshoots by at most one, and the probe
 *  skips a drained slice with no add at all.
 *
 *  No dispatch is in flight when @c invoke_for_n_dynamic::reset_slices_ rewrites the cursors, which
 *  it does in the invoker's constructor, before @c flat_pool::unsafe_for_threads checks anything.
 *
 *  @verify pass sc,rc11
 *  @verify pass rc11 tasks=2: the regime with no dynamic task, where every slice is empty and only
 *      the prongs run
 *  @verify pass rc11 publish_order=order_relaxed: the dispatch's release on the epoch carries every
 *      cursor and end to every worker, so the release on each cursor in @c reset_slices_ is
 *      redundant, though the header keeps it
 *  @verify fail rc11 dispatch_order=order_relaxed: the dispatch steps the epoch with a release;
 *      relaxed, a worker reads a slice's end from before the dispatch and skips its tasks
 *  @verify fail rc11 completion_order=order_relaxed: the join acquires the completion step;
 *      relaxed, it reads the result after the wait and sees a stale sum
 *  @verify fail sc claim_guard=false: @c drain_claim stops at a claim at or past the slice's end;
 *      without the check, it runs the task the claim returns, and the cursor runs into the next
 *      slice and past the bound
 *  @verify pass sc probe=false: without the probe, the walk still visits each slice once, which
 *      bounds the overshoot on its own
 *  @verify pass sc visits=2: walking every slice twice, the probe still skips a drained slice,
 *      which bounds the overshoot on its own
 *  @verify fail sc probe=false visits=2: the relaxed probe skips a drained slice with no add, and
 *      the walk visits each slice once, and the docblock argues either one bounds the overshoot;
 *      without both, every visit adds past the end, and a cursor passes `max(tasks, threads)`
 *  @verify fail sc nested=true: the pool's docblock forbids re-entry; a prong that calls
 *      @c for_n_dynamic on its own pool runs the nested invoker's constructor with the epoch odd,
 *      which rewinds the cursors under the live dispatch, and the tasks behind them run twice
 */
#define thread_count 3
#define location_count 7
#define history_depth 6
#include "weak_memory.pml"

/** The workers beside the caller, and the generation the one dispatch runs. */
#define threads 2
#define generation 1

/** The words, by the member each stands for: the pool's clock and countdown, each thread's cursor
 *  and end, and the result. */
#define epoch 0
#define threads_to_sync 1
#define next(slice) (2 + (slice))
#define end(slice) (4 + (slice))
#define result 6

/** The knobs: the header's choices and the tasks, each overridden by a `@verify` line. */
#ifndef tasks
#define tasks 5
#endif
#if tasks < threads
#error "tasks counts one prong per thread at least"
#endif
#ifndef dispatch_order
#define dispatch_order order_release
#endif
#ifndef completion_order
#define completion_order order_acquire
#endif
#ifndef publish_order
#define publish_order order_release
#endif
#ifndef claim_guard
#define claim_guard true
#endif
#ifndef probe
#define probe true
#endif
#ifndef visits
#define visits 1
#endif
#ifndef nested
#define nested false
#endif

/** The dynamic tasks ahead of the trailing prongs, and their split as @c indexed_split makes it. */
#define dynamic_tasks (tasks - threads)
#define slice_split ((dynamic_tasks + 1) / 2)

byte ran[tasks]; // each task's runs

/** One task: its effect on the shared result through @p sum, and the ghost that counts its runs. */
inline perform(t, task, sum) {
    ran[task]++;
    read_modify_write(t, result, order_relaxed, sum, sum + 1)
}

/** The relaxed probe, then one task per relaxed add until the claim is past the end, as
 *  @c drain_claim in `types.hpp` does, over the caller's @p cursor, @p slice_last and @p sum. */
inline drain(t, slice, cursor, slice_last, sum) {
    load(t, next(slice), order_relaxed, cursor);
    load(t, end(slice), order_relaxed, slice_last);
    if
    :: probe && cursor >= slice_last
    :: else ->
        do
        :: read_modify_write(t, next(slice), order_relaxed, cursor, cursor + 1);
           assert(cursor + 1 <= tasks);
           if :: claim_guard && cursor >= slice_last -> break :: else fi;
           if :: cursor >= tasks -> break :: else fi;
           perform(t, cursor, sum)
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
proctype dispatcher(byte t) {
    byte task;
    int observed, seen, summed_before;
    reset_slices(t);
    // The countdown reset relaxed and the epoch stepped, as `flat_pool::unsafe_for_threads` does
    store(t, threads_to_sync, order_relaxed, threads);
    read_modify_write(t, epoch, dispatch_order, observed, observed + 1);
    // The wait for the completion step, then the results: `flat_pool::unsafe_join` in `flat.hpp`
    (newest_value(epoch) == generation + 1);
    do
    :: load(t, epoch, completion_order, seen);
       if :: seen != generation -> break :: else fi
    od;
    load(t, result, order_relaxed, summed_before);
    assert(summed_before == tasks);
    for (task : 0 .. tasks - 1) { assert(ran[task] == 1) }
}

/** A worker on slice `t - 1`: acquires the dispatch, runs its prong, drains every slice, and counts
 *  down. */
proctype worker(byte t) {
    byte visit;
    int seen_epoch, cursor, slice_last, sum, before, observed;
    // The worker loop's acquire of the dispatch: `flat_pool::_worker_loop` in `flat.hpp`
    (newest_value(epoch) == generation);
    do
    :: load(t, epoch, order_acquire, seen_epoch);
       if :: seen_epoch == generation -> break :: else fi
    od;
    // The static prong, then the slices in coprime order, its own first:
    // `invoke_for_n_dynamic::operator()` in `types.hpp`
    perform(t, dynamic_tasks + t - 1, sum);
    // A nested prong dispatches on its own pool, and the nested invoker's constructor runs here
    if :: nested && t == 1 -> reset_slices(t) :: else fi;
    for (visit : 1 .. visits) {
        drain(t, t - 1, cursor, slice_last, sum);
        drain(t, 2 - t, cursor, slice_last, sum)
    };
    // The countdown with `acq_rel`, and the completion step by the last contributor:
    // `flat_pool::_worker_loop` in `flat.hpp`
    read_modify_write(t, threads_to_sync, order_acq_rel, before, before - 1);
    if
    :: before == 1 -> read_modify_write(t, epoch, order_release, observed, observed + 1)
    :: else
    fi
}

init { atomic { run dispatcher(0); run worker(1); run worker(2) } }
