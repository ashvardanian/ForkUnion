/**
 *  @file verification/for_n_dynamic.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Spin model of @c invoke_for_n_dynamic from `include/forkunion/types.hpp`.
 *
 *  Behind @c flat_pool::for_n_dynamic in `flat.hpp`: one private cursor per thread, rewound by the
 *  caller once its claim admitted the dispatch. Each thread runs one static prong from the trailing
 *  tasks, drains its own slice with relaxed adds, then walks its neighbours' slices in a coprime
 *  order and drains them too, one task per claim; the join sees every task's effect.
 *
 *  A dispatcher and three workers on a caller-exclusive pool, over @c tasks tasks, four by default:
 *  one dynamic one in the slice [0, 1), the other two slices empty, and three prongs, one per
 *  thread. With fewer tasks than threads every slice is empty, and only the first threads have a
 *  prong. The dispatcher claims the countdown from zero, writes each slice's end and cursor
 *  relaxed, since the epoch's release carries them, and stores the epoch with a release; each
 *  worker acquires the epoch, runs its prong, and drains its own slice and then the others: a
 *  relaxed probe of the cursor against the end, then relaxed adds until the task read is at or
 *  past the end. Every task adds one to a shared result relaxed. The countdown and the completion
 *  step are the pool's, proven in `flat_pool/`; the dispatcher waits for the completion and reads
 *  the result after an acquire.
 *  Every wait is the @c wait_on of `monitor_wait.pml`.
 *
 *  Every worker reads the ends the dispatch published, every task runs exactly once, and the
 *  result the join reads is the task count.
 *
 *  No cursor passes `max(tasks, threads)`, the bound @c invoke_for_n_dynamic documents: a visit
 *  overshoots by at most one, and the probe skips a drained slice with no add at all.
 *
 *  A prong that calls @c for_n_dynamic on its own pool is refused, since the countdown is above
 *  zero until every thread counted down, and the refused call writes nothing, since the pool
 *  rewinds the cursors only after its claim succeeds, in @c flat_pool::unsafe_for_threads.
 *
 *  @verify pass sc,rc11
 *  @verify pass rc11 tasks=3: the regime with no dynamic task, where every slice is empty and only
 *      the prongs run
 *  @verify pass sc,rc11 tasks=2: fewer tasks than threads, where the third thread has no prong and
 *      the cursors overshoot to @c threads, past @c tasks
 *  @verify fail rc11 dispatch_order=order_relaxed: the dispatch stores the epoch with a release;
 *      relaxed, a worker reads a slice's end from before the dispatch, which skips its tasks
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
 *  @verify pass sc nested=true: a nested @c for_n_dynamic is refused, and the pool rewinds the
 *      cursors only once its claim admitted a dispatch, so the refused call writes nothing
 *  @verify fail sc nested=true rewind_after_claim=false: rewinding before the claim, the refused
 *      nested call rewinds the cursors under the live dispatch, and the tasks behind them run twice
 */
#define thread_count 4
#define location_count 9
#define history_depth 6
#include "weak_memory.pml"
#include "monitor_wait.pml"

/** The workers beside the caller, and the generation the one dispatch runs. */
#define threads 3
#define generation 1

/** The words, by the member each stands for: the pool's clock and countdown, each thread's cursor
 *  and end, and the result. */
#define epoch 0
#define threads_to_sync 1
#define next(slice) (2 + (slice))
#define end(slice) (5 + (slice))
#define result 8

/** The knobs: the header's choices and the tasks, each overridden by a `@verify` line. */
#ifndef tasks
#define tasks 4
#endif
#ifndef dispatch_order
#define dispatch_order order_release
#endif
#ifndef completion_order
#define completion_order order_acquire
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
#ifndef rewind_after_claim
#define rewind_after_claim true
#endif

/** The dynamic tasks ahead of the trailing prongs, their split as @c indexed_split makes it, and
 *  the bound no cursor passes, spelled with comparisons as factors, since Spin parses no
 *  conditional nested in another. */
#define dynamic_tasks ((tasks - threads) * (tasks > threads))
#define slice_larger(slice) ((slice) < dynamic_tasks % threads)
#define slice_first(slice) \
    ((dynamic_tasks / threads) * (slice) + (slice) * slice_larger(slice) + (dynamic_tasks % threads) * !slice_larger(slice))
#define slice_end(slice) (slice_first(slice) + dynamic_tasks / threads + slice_larger(slice))
#define bound (tasks * (tasks > threads) + threads * (tasks <= threads))

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
    assert(slice_last == slice_end(slice));
    if
    :: probe && cursor >= slice_last
    :: else ->
        do
        :: read_modify_write(t, next(slice), order_relaxed, cursor, cursor + 1);
           assert(cursor + 1 <= bound);
           if :: claim_guard && cursor >= slice_last -> break :: else fi;
           if :: cursor >= tasks -> break :: else fi;
           perform(t, cursor, sum)
        od
    fi
}

/** Each end and each cursor relaxed, as @c invoke_for_n_dynamic::rewind in `types.hpp` does. */
inline rewind(t) {
    store(t, end(0), order_relaxed, slice_end(0));
    store(t, next(0), order_relaxed, slice_first(0));
    store(t, end(1), order_relaxed, slice_end(1));
    store(t, next(1), order_relaxed, slice_first(1));
    store(t, end(2), order_relaxed, slice_end(2));
    store(t, next(2), order_relaxed, slice_first(2))
}

/** The caller: claims, rewinds the slices, dispatches, joins, and checks every task ran once. */
proctype dispatcher(byte t) {
    byte task;
    int observed, seen, summed_before;
    bool claimed;
    if :: !rewind_after_claim -> rewind(t) :: else fi;
    // The claim from zero, the rewind and the epoch step of `flat_pool::unsafe_for_threads`
    compare_exchange(t, threads_to_sync, order_acquire, observed, threads, claimed);
    assert(claimed);
    if :: rewind_after_claim -> rewind(t) :: else fi;
    atomic { store(t, epoch, dispatch_order, generation); wake(epoch) };
    // The wait for the completion step, then the results: `flat_pool::unsafe_join` in `flat.hpp`
    do
    :: load(t, epoch, completion_order, seen);
       if :: seen != generation -> break :: else -> wait_on(t, epoch, seen, false) fi
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
    bool claimed;
    // The worker loop's acquire of the dispatch: `flat_pool::worker_loop_` in `flat.hpp`
    do
    :: load(t, epoch, order_acquire, seen_epoch);
       if :: seen_epoch == generation -> break :: else -> wait_on(t, epoch, seen_epoch, false) fi
    od;
    // The static prong, then the slices in coprime order, its own first:
    // `invoke_for_n_dynamic::operator()` in `types.hpp`
    if :: dynamic_tasks + t - 1 < tasks -> perform(t, dynamic_tasks + t - 1, sum) :: else fi;
    // A nested prong dispatches on its own pool, and is refused at the claim
    if
    :: nested && t == 1 ->
        if :: !rewind_after_claim -> rewind(t) :: else fi;
        observed = 0;
        compare_exchange(t, threads_to_sync, order_acquire, observed, threads, claimed);
        assert(!claimed)
    :: else
    fi;
    for (visit : 1 .. visits) {
        drain(t, (t - 1) % threads, cursor, slice_last, sum);
        drain(t, t % threads, cursor, slice_last, sum);
        drain(t, (t + 1) % threads, cursor, slice_last, sum)
    };
    // The countdown with `acq_rel`, and the completion step by the last contributor:
    // `flat_pool::worker_loop_` in `flat.hpp`
    read_modify_write(t, threads_to_sync, order_acq_rel, before, before - 1);
    if
    :: before == 1 -> atomic { store(t, epoch, order_release, generation + 1); wake(epoch) }
    :: else
    fi
}

init { atomic { run dispatcher(0); run worker(1); run worker(2); run worker(3) } }
