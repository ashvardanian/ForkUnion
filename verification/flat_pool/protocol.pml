/**
 *  @file verification/flat_pool/protocol.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Spin model of @c flat_pool from `include/forkunion/flat.hpp`: the epoch clock, the
 *      @c threads_to_sync that admits one dispatch and names the last contributor, and the mood.
 *
 *  The epoch is odd while a @c fork_state is in flight and even when idle, and the mood puts
 *  workers to sleep and lets them go. A dispatcher runs @c generations generations, two by default,
 *  on a caller-inclusive pool with two workers. Each epoch store has one writer, so neither step
 *  needs a read-modify-write. On a caller-exclusive pool, thread 0 may fork from its slice, still
 *  holding its own unit. Every wait is the @c wait_on of <tt>monitor_wait.pml</tt>.
 *
 *  - No fork is torn.
 *  - Every contributor reads the @c fork_state its generation published and runs each one exactly
 *    once.
 *  - The @c threads_to_sync never goes below zero, and every token a join receives is odd.
 *  - A completed join sees every contributor's result.
 *  - No completion step fires and no dispatch is admitted while thread 0 forks from its slice.
 *  - A token reads complete exactly once its dispatch, forks from thread 0 included, ended.
 */
#include <weak_memory.pml>
#include <monitor_wait.pml>

/** The caller's thread, which plays the dispatcher and contributes without a result word of its
 *  own; the workers are threads 1 and 2. On a caller-exclusive pool, thread 0 is a worker, again
 *  without a result word. */
#define caller 0

/** The words, by the pool member each stands for, then one result word per worker, the spawn edge,
 *  the C shim's callback slot @c opaque_pool_t::current_callback, the poller's token, and
 *  @c dispatched_generation, the generation of the dispatch in flight. */
#define epoch 0
#define threads_to_sync 1
#define mood 2
#define fork_state 3
#define result(worker) (3 + (worker))
#define spawned 6
#define current_callback 7
#define token 8
#define dispatched_generation 9

/** The contributors, the caller's slice and the two workers', and the moods the pool moves
 *  through. */
#define contributors 3
#define grind 0
#define nap 1
#define die 2
#define sleep 3

/** The fork values past every round a dispatcher publishes: a dispatch of @c forking_fork has
 *  thread 0 fork twice from its slice, the second fork's slice nesting a third, and one of
 *  @c forking_once has it fork once. */
#define forking_fork (generations + 1)
#define first_fork (generations + 2)
#define second_fork (generations + 3)
#define nested_fork (generations + 4)
#define forking_once (generations + 5)
#define fork_values (generations + 6)

/** Not a word but the kernel's queue of threads blocked on @c mood: a store to the word wakes a
 *  monitor, while only an explicit @c os_wake_all_on_word wakes a thread in this queue. */
#define kernel_queue location_count

/** The knobs: the header's choices, the generations, and the epoch's width, which is 2^64 in the
 *  header and small enough here to wrap on request, as @c generation_modulus is in the gate's
 *  model; each is overridden by a `@verify` line. */
#ifndef generations
#define generations 2
#endif
#ifndef epoch_modulus
#define epoch_modulus 16
#endif
#if epoch_modulus % 2 != 0
#error "epoch_modulus is even, so odd epochs stay dispatches"
#endif
#ifndef claim
#define claim true
#endif
#ifndef claim_order
#define claim_order order_acquire
#endif
#ifndef completion_wait
#define completion_wait true
#endif
#ifndef dispatch_order
#define dispatch_order order_release
#endif
#ifndef pickup_order
#define pickup_order order_acquire
#endif
#ifndef decrement_order
#define decrement_order order_acq_rel
#endif
#ifndef worker_completion_order
#define worker_completion_order order_release
#endif
#ifndef caller_completion_order
#define caller_completion_order order_release
#endif
#ifndef join_order
#define join_order order_acquire
#endif
#ifndef complete_order
#define complete_order order_acquire
#endif
#ifndef wait_cap
#define wait_cap true
#endif
#ifndef wake_order
#define wake_order order_release
#endif
#ifndef nap_wakes_kernel
#define nap_wakes_kernel true
#endif
#ifndef generation_check
#define generation_check true
#endif
#ifndef fork_publish_order
#define fork_publish_order order_release
#endif
#ifndef wait_for_other_slices
#define wait_for_other_slices true
#endif
#ifndef completion_from_current_epoch
#define completion_from_current_epoch true
#endif

#define stepped(value) (((value) + 1) % epoch_modulus)
#define round_of(generation) (((generation) + 1) / 2)
#define rounds (epoch_modulus / 2 + 1)

byte workers_exited;
bool terminated;                     // die was stored and not yet reset
byte current_generation;             // `opaque_pool_t::current_generation`, which only its holder touches
byte forked[rounds];                 // the fork each generation published, by its round
byte running;                        // contributors of the live generation yet to count down
byte ran[thread_count * fork_values]; // slices run, per contributor and fork
bool forking;                        // thread 0 runs the slice it forks from, not yet counted down

/** The decrement after a slice, which never reads 1 while thread 0 forks from its slice, waking a
 *  thread 0 that waits on the countdown, and the completion step by the last contributor, released
 *  with @p completion, from the current epoch if @p from_current and else from @p generation, over
 *  the caller's @p before and @p observed. */
inline count_down(t, generation, completion, from_current, before, observed) {
    atomic {
        read_modify_write(t, threads_to_sync, decrement_order, before, before - 1);
        assert(before != 1 || !forking);
        running--;
        wake(threads_to_sync)
    };
    assert(before > 0);
    if
    :: before == 1 ->
        if
        :: from_current -> load(t, epoch, order_relaxed, observed)
        :: else -> observed = generation
        fi;
        atomic { store(t, epoch, completion, stepped(observed)); wake(epoch) }
    :: else
    fi
}

/** One slice: the @c fork_state, the result of a worker, thread 0's forks for @c forking_fork and
 *  @c forking_once, and the count-down, over the caller's @p seen_fork, @p seen_callback,
 *  @p before, @p observed and @p fork_generation. */
inline contribute(t, generation, by_worker, completion, from_current, seen_fork, seen_callback, before, observed, fork_generation) {
    assert(!terminated);
    load(t, fork_state, order_relaxed, seen_fork);
    assert(seen_fork == forked[round_of(generation)]);
    // The trampoline reads the slot on every contributor, and fork r went out as callback r:
    // `opaque_pool_t::operator()` in `src/forkunion.cpp`
    if
    :: through_c_shim ->
        load(t, current_callback, order_relaxed, seen_callback);
        assert(seen_callback == seen_fork)
    :: else
    fi;
    atomic {
        ran[(t) * fork_values + seen_fork]++;
        assert(ran[(t) * fork_values + seen_fork] == 1)
    };
    if
    :: by_worker -> store(t, result(t), order_relaxed, generation)
    :: else
    fi;
    if
    :: (t) == 0 && seen_fork == forking_fork -> forks_from_thread_zero(t, fork_generation, observed)
    :: (t) == 0 && seen_fork == forking_once -> fork_once_from_thread_zero(t, fork_generation, observed)
    :: else
    fi;
    count_down(t, generation, completion, from_current, before, observed)
}

/** The spawn edge a fresh thread starts from: its view is its creator's at the spawn. */
inline start(t) {
    int seen_spawn;
    (newest_value(spawned) == 1);
    do
    :: load(t, spawned, order_acquire, seen_spawn);
       if :: seen_spawn == 1 -> break :: else fi
    od
}

/** Stores grind with a release, then starts the workers: @c flat_pool::spawn in `flat.hpp`. */
inline spawn(t) {
    atomic { store(t, mood, order_release, grind); wake(mood) };
    store(t, spawned, order_release, 1);
    run worker(1);
    run worker(2)
}

/** Die with a release, then the epoch's current value re-released, whose write wakes the workers
 *  waiting on it, then the kernel's queue: @c flat_pool::signal_die_ in <tt>flat.hpp</tt>, over the
 *  caller's @p observed. */
inline signal_die(t, observed) {
    atomic { store(t, mood, order_release, die); terminated = true; wake(mood) };
    load(t, epoch, order_relaxed, observed);
    atomic { store(t, epoch, order_release, observed); wake(epoch) };
    wake(kernel_queue)
}

/** Die after the last join, the joins, then the resets, as @c flat_pool::terminate in `flat.hpp`
 *  does, over the caller's @p observed. */
inline terminate(t, observed) {
    // The header's two asserts read seq_cst; acquire is the strongest order the module spells.
    load(t, threads_to_sync, order_acquire, observed);
    assert(observed == 0);
    load(t, epoch, order_acquire, observed);
    assert((observed % 2) == 0);
    signal_die(t, observed);
    (workers_exited == 2);
    atomic { store(t, mood, order_relaxed, grind); terminated = false; wake(mood) };
    atomic { store(t, epoch, order_relaxed, 0); wake(epoch) }
}

/** The mood each dispatcher read, for its wake after the epoch store. */
byte parked_mood[thread_count];

/** Thread @p t blocks in the kernel while @c mood holds @p observed: @c os_wait_on_word, whose
 *  period only ever adds an early return, so the model leaves it out. */
inline kernel_wait(t, observed) {
    atomic {
        armed[t] = kernel_queue + 1;
        event[t] = recheck && newest_value(mood) != (observed);
        (event[t]);
        armed[t] = 0;
        event[t] = false
    }
}

/** Parks on the monitor from the dispatching thread, waking any kernel waiters a @c sleep left:
 *  @c flat_pool::nap in <tt>flat.hpp</tt>, over the caller's @p observed. */
inline nap_workers(t, observed) {
    load(t, mood, order_relaxed, observed);
    atomic { store(t, mood, order_release, nap); wake(mood) };
    if
    :: observed == sleep && nap_wakes_kernel -> wake(kernel_queue)
    :: else
    fi
}

/** Parks in the kernel from the dispatching thread: @c flat_pool::sleep in <tt>flat.hpp</tt>. */
inline sleep_workers(t) {
    atomic { store(t, mood, order_release, sleep); wake(mood) }
}

/** Stores the odd @p generation, released with @p order, after the wake-up from a parked mood as a
 *  relaxed read and a releasing store back to grind, then wakes the kernel's queue if the mood was
 *  sleep: @c flat_pool::step_epoch_and_wake_ in <tt>flat.hpp</tt>, over the caller's @p observed. The
 *  ghost records the fork the generation carries and the @p contributing count-downs it awaits. */
inline step_epoch_and_wake(t, generation, order, round, contributing, observed) {
    load(t, mood, order_relaxed, observed);
    parked_mood[t] = observed;
    if
    :: observed != grind -> atomic { store(t, mood, wake_order, grind); wake(mood) }
    :: else
    fi;
    atomic {
        store(t, epoch, order, generation);
        forked[round_of(generation)] = round;
        running = contributing;
        wake(epoch)
    };
    if
    :: parked_mood[t] == sleep -> wake(kernel_queue)
    :: else
    fi
}

/** The wait for an even epoch, the next one recorded as the dispatch in flight, then stored with a
 *  release, since the claim leaves the dispatcher its only writer: @c flat_pool::publish_claimed_
 *  in <tt>flat.hpp</tt>, over the caller's @p observed. */
inline publish_claimed(t, generation, round, contributing, observed) {
    if
    :: completion_wait ->
        do
        :: load(t, epoch, order_acquire, observed);
           if
           :: (observed % 2) == 0 -> break
           :: else -> wait_on(t, epoch, observed, false)
           fi
        od
    :: else -> load(t, epoch, order_acquire, observed)
    fi;
    generation = stepped(observed);
    store(t, dispatched_generation, order_relaxed, generation);
    step_epoch_and_wake(t, generation, dispatch_order, round, contributing, observed)
}

/** The claim from zero, refused with the zero token, then the fork and its publication:
 *  @c flat_pool::unsafe_for_threads in `flat.hpp`, over the caller's @p observed and
 *  @p exchanged. */
inline dispatch(t, round, generation, observed, exchanged) {
    // Without the claim, the plain store of the countdown follows the fork, as it did before it
    if
    :: claim ->
        observed = 0;
        compare_exchange(t, threads_to_sync, claim_order, observed, contributors, exchanged)
    :: else -> exchanged = true
    fi;
    if
    :: exchanged ->
        assert(running == 0);
        store(t, fork_state, order_relaxed, round);
        if
        :: !claim -> store(t, threads_to_sync, order_relaxed, contributors)
        :: else
        fi;
        publish_claimed(t, generation, round, contributors, observed)
    :: else -> generation = 0
    fi
}

/** The wait for the completion step of the generation @p joined, into the caller's
 *  @p seen_epoch. */
inline await_completion(t, joined, seen_epoch) {
    do
    :: load(t, epoch, join_order, seen_epoch);
       if
       :: seen_epoch != joined -> break
       :: else -> wait_on(t, epoch, joined, false)
       fi
    od
}

/** The token's parity, the caller's slice, then the wait for the completion step, as
 *  @c flat_pool::unsafe_join in `flat.hpp` does, over the caller's @p seen_epoch and scratch. */
inline join(t, joined, seen_epoch, seen_fork, seen_callback, before, observed) {
    assert((joined % 2) == 1);
    load(t, epoch, order_acquire, seen_epoch);
    if
    :: seen_epoch == joined ->
        // The caller of an inclusive pool is no thread 0, so its slice never forks
        contribute(t, joined, false, caller_completion_order, false, seen_fork, seen_callback, before, observed, seen_epoch);
        await_completion(t, joined, seen_epoch)
    :: else
    fi
}

/** The slot claimed from empty before the pool, handed back if the pool refuses, as
 *  @c fu_pool_unsafe_for_threads in <tt>src/forkunion.cpp</tt> does, over the dispatch's scratch. */
inline c_dispatch(t, round, generation, observed, exchanged) {
    observed = 0;
    compare_exchange(t, current_callback, order_acquire, observed, round, exchanged);
    if
    :: exchanged ->
        dispatch(t, round, generation, observed, exchanged);
        if
        :: generation == 0 -> store(t, current_callback, order_release, 0)
        :: else -> current_generation = generation
        fi
    :: else -> generation = 0
    fi
}

/** An empty slot or another generation's token returns at once, else the pool's join and
 *  then the clear, released last, as @c fu_pool_unsafe_join in <tt>src/forkunion.cpp</tt> does,
 *  over its scratch. */
inline c_join(t, joined, seen_epoch, seen_fork, seen_callback, before, observed) {
    load(t, current_callback, order_relaxed, seen_callback);
    if
    :: seen_callback != 0 && (!generation_check || (joined) == current_generation) ->
        join(t, joined, seen_epoch, seen_fork, seen_callback, before, observed);
        store(t, current_callback, order_release, 0)
    :: else
    fi
}

/** What a completed join sees: every worker's result of this @p generation, or of a later one that
 *  another dispatcher ran since, read into the caller's @p observed. */
inline check_results(t, generation, observed) {
    load(t, result(1), order_relaxed, observed);
    assert(observed >= generation);
    load(t, result(2), order_relaxed, observed);
    assert(observed >= generation)
}

/** Thread 0 waits, over its @p observed, until only its own unit remains in the countdown, unless
 *  @c wait_for_other_slices is off: @c flat_pool::wait_for_other_slices_ in <tt>flat.hpp</tt>. */
inline await_other_slices(t, observed) {
    if
    :: wait_for_other_slices ->
        do
        :: load(t, threads_to_sync, order_acquire, observed);
           if
           :: observed == 1 -> break
           :: else -> wait_on(t, threads_to_sync, observed, false)
           fi
        od
    :: else
    fi
}

/** Thread 0 re-arms the pool its unit holds for the fork @p value, into @p fork_generation: the
 *  wait, the fork, the countdown stored relaxed, still counting thread 0's unit, and the epoch two
 *  past stored with @c fork_publish_order by the mood steps of a publication, then its own slice in
 *  place, never counted down, as the first half of @c flat_pool::fork_from_thread_zero_ in
 *  <tt>flat.hpp</tt> runs them. */
inline fork_in_place(t, value, fork_generation, observed) {
    await_other_slices(t, observed);
    store(t, fork_state, order_relaxed, value);
    store(t, threads_to_sync, order_relaxed, contributors);
    load(t, epoch, order_relaxed, observed);
    fork_generation = (observed + 2) % epoch_modulus;
    step_epoch_and_wake(t, fork_generation, fork_publish_order, value, contributors, observed);
    atomic {
        ran[(t) * fork_values + value]++;
        assert(ran[(t) * fork_values + value] == 1)
    }
}

/** The second half: the wait for the other slices, after which the fork that made
 *  @p fork_generation reads both workers' results of it. */
inline fork_ended(t, fork_generation, observed) {
    await_other_slices(t, observed);
    load(t, result(1), order_relaxed, observed);
    assert(observed == fork_generation);
    load(t, result(2), order_relaxed, observed);
    assert(observed == fork_generation)
}

/** Thread 0's slice of @c forking_fork: a fork, then a second whose slice forks once more, each
 *  through @c flat_pool::unsafe_for_threads, whose join does nothing. */
inline forks_from_thread_zero(t, fork_generation, observed) {
    forking = true;
    fork_in_place(t, first_fork, fork_generation, observed);
    fork_ended(t, fork_generation, observed);
    fork_in_place(t, second_fork, fork_generation, observed);
    fork_in_place(t, nested_fork, fork_generation, observed);
    fork_ended(t, fork_generation, observed);
    fork_ended(t, fork_generation, observed);
    forking = false
}

/** Thread 0's slice of @c forking_once: a single fork. */
inline fork_once_from_thread_zero(t, fork_generation, observed) {
    forking = true;
    fork_in_place(t, first_fork, fork_generation, observed);
    fork_ended(t, fork_generation, observed);
    forking = false
}

/** Whether @p generation ended by the time the pool showed the epoch read into @p observed, into
 *  @p complete, over the caller's @p seen_dispatched: nothing runs, or another dispatch does. This
 *  is @c flat_pool::completed_at_ in <tt>flat.hpp</tt>. */
inline completed_at(t, generation, observed, seen_dispatched, complete) {
    load(t, dispatched_generation, order_relaxed, seen_dispatched);
    complete = (observed % 2) == 0 || seen_dispatched != (generation)
}

/** The join of a caller-exclusive pool, which polls the token rule until @p generation reads
 *  complete: @c flat_pool::unsafe_join in <tt>flat.hpp</tt>, over the caller's scratch. */
inline await_token(t, generation, observed, seen_dispatched, complete) {
    do
    :: load(t, epoch, order_acquire, observed);
       completed_at(t, generation, observed, seen_dispatched, complete);
       if
       :: complete -> break
       :: else -> wait_on(t, epoch, observed, false)
       fi
    od
}

/** The worker loop of @c flat_pool::worker_loop_ in `flat.hpp`, until die: waits on the epoch,
 *  looks at the mood only while the epoch stands still, parks on the mood word by the monitor at
 *  nap and in the kernel at sleep, and contributes to every odd epoch. */
inline serve(t) {
    int last_epoch, new_epoch, seen_mood, before, observed, seen_fork, seen_callback, fork_generation;
    do
    :: load(t, epoch, pickup_order, new_epoch);
       if
       :: new_epoch == last_epoch ->
           load(t, mood, order_acquire, seen_mood);
           if
           :: seen_mood == die -> break
           :: seen_mood == nap -> wait_on(t, mood, nap, false)
           :: seen_mood == sleep -> kernel_wait(t, sleep)
           :: else -> wait_on(t, epoch, last_epoch, wait_cap && newest_value(mood) != grind)
           fi
       :: else ->
           if
           :: (new_epoch % 2) == 1 ->
               contribute(t, new_epoch, (t) != 0, worker_completion_order, completion_from_current_epoch, seen_fork, seen_callback, before, observed, fork_generation)
           :: else
           fi;
           last_epoch = new_epoch
       fi
    od;
    workers_exited++
}
