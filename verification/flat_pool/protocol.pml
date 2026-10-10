/**
 *  @file verification/flat_pool/protocol.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Spin model of @c flat_pool from `include/forkunion/flat.hpp`: the epoch clock, the
 *      @c threads_to_sync that admits one dispatch and names the last contributor, and the mood.
 *
 *  The epoch is odd while a @c fork_state is in flight and even when idle, and the mood puts
 *  workers to sleep and lets them go.
 *
 *  A dispatcher runs @c generations generations, two by default, on a caller-inclusive pool with
 *  two workers. Each dispatch claims the @c threads_to_sync by exchanging it from zero with an
 *  acquire, and one that finds it above zero is refused with the zero token and writes nothing.
 *  An admitted dispatch writes the @c fork_state, waits until the epoch is even, which the last
 *  contributor of the previous generation makes it just after its decrement, and stores the next
 *  epoch with a release. Every contributor reads the @c fork_state, writes its result relaxed,
 *  decrements the @c threads_to_sync with @c acq_rel, and the one that took it to zero stores the
 *  epoch after its generation with a release. Each epoch store has one writer, so neither step
 *  needs a read-modify-write. The dispatcher contributes its own slice inside the join, then waits
 *  for the completion step. The workers wait on the epoch, checking the mood only while it stands
 *  still, under a capped wait. The spawn stores grind with a release and starts the workers, whose
 *  first step acquires the spawn, since a new thread's view is its creator's.
 *
 *  Every wait is the @c wait_on of `monitor_wait.pml`, and every write to the epoch or the mood
 *  wakes the threads armed on it, as any store to a monitored line does.
 *
 *  No fork is torn: a dispatch writes the @c fork_state only once every contributor of the last
 *  generation counted down, and every contributor reads the @c fork_state its generation
 *  published and runs each one exactly once. The @c threads_to_sync never goes below zero, and
 *  every token a join receives is odd. A completed join sees every contributor's result: the
 *  @c acq_rel decrements chain each contributor's writes into the last one, whose release step
 *  publishes them at once.
 *
 *  Every scenario beside this file defines its shape, @c through_c_shim for dispatches the C shim
 *  wraps, and the @c worker the spawn runs.
 */
#include "../weak_memory.pml"
#include "../monitor_wait.pml"

/** The caller's thread, which plays the dispatcher and contributes without a result word of its
 *  own; the workers are threads 1 and 2. */
#define caller 0

/** The words, by the pool member each stands for, then one result word per worker, the spawn edge,
 *  the C shim's callback slot @c opaque_pool_t::current_callback, and the poller's token. */
#define epoch 0
#define threads_to_sync 1
#define mood 2
#define fork_state 3
#define result(worker) (3 + (worker))
#define spawned 6
#define current_callback 7
#define token 8

/** The contributors, the caller's slice and the two workers', and the moods the pool moves
 *  through. */
#define contributors 3
#define grind 0
#define chill 1
#define die 2

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
#ifndef strong_wake
#define strong_wake true
#endif
#ifndef generation_check
#define generation_check true
#endif

#define stepped(value) (((value) + 1) % epoch_modulus)
#define round_of(generation) (((generation) + 1) / 2)

byte workers_exited;
bool terminated;                     // die was stored and not yet reset
byte current_generation;             // `opaque_pool_t::current_generation`, which only its holder touches
byte forked[epoch_modulus / 2 + 1];  // the fork each generation published, by its round
byte running;                        // contributors of the live generation yet to count down
byte ran[thread_count * (generations + 3)]; // slices run, per contributor and fork

/** One slice: the @c fork_state, the result of a worker, the decrement, and the completion step
 *  for the last one, released with @p completion, over the caller's @p seen_fork,
 *  @p seen_callback, @p before and @p observed. */
inline contribute(t, generation, by_worker, completion, seen_fork, seen_callback, before, observed) {
    assert(!terminated);
    load(t, fork_state, order_relaxed, seen_fork);
    assert(seen_fork == forked[round_of(generation)]);
    // The trampoline reads the slot on every contributor, and fork r went out as callback r:
    // `opaque_pool_t::operator()` in `c/forkunion.cpp`
    if
    :: through_c_shim ->
        load(t, current_callback, order_relaxed, seen_callback);
        assert(seen_callback == seen_fork)
    :: else
    fi;
    atomic {
        ran[(t) * (generations + 3) + seen_fork]++;
        assert(ran[(t) * (generations + 3) + seen_fork] == 1)
    };
    if
    :: by_worker -> store(t, result(t), order_relaxed, generation)
    :: else
    fi;
    atomic { read_modify_write(t, threads_to_sync, decrement_order, before, before - 1); running-- };
    assert(before > 0);
    if
    :: before == 1 -> atomic { store(t, epoch, completion, stepped(generation)); wake(epoch) }
    :: else
    fi
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

/** Die with a release after the last join, the joins, then the resets, as @c flat_pool::terminate
 *  in `flat.hpp` does, over the caller's @p observed. */
inline terminate(t, observed) {
    // The header's two asserts read seq_cst; acquire is the strongest order the module spells.
    load(t, threads_to_sync, order_acquire, observed);
    assert(observed == 0);
    load(t, epoch, order_acquire, observed);
    assert((observed % 2) == 0);
    atomic { store(t, mood, order_release, die); terminated = true; wake(mood) };
    (workers_exited == 2);
    atomic { store(t, mood, order_relaxed, grind); terminated = false; wake(mood) };
    atomic { store(t, epoch, order_relaxed, 0); wake(epoch) }
}

/** The claim from zero, refused with the zero token, then the fork, the wait for an even epoch,
 *  the wake-up from a chill as a relaxed read and a strong exchange back to grind, and the next
 *  epoch stored with a release, since the claim leaves the dispatcher its only writer:
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
        load(t, mood, order_relaxed, observed);
        if
        :: observed == chill ->
            if
            :: atomic { compare_exchange(t, mood, order_relaxed, observed, grind, exchanged); if :: exchanged -> wake(mood) :: else fi }
            :: !strong_wake -> skip
            fi
        :: else
        fi;
        atomic {
            store(t, epoch, dispatch_order, generation);
            forked[round_of(generation)] = round;
            running = contributors;
            wake(epoch)
        }
    :: else -> generation = 0
    fi
}

/** The token's parity, the caller's slice, then the wait for the completion step, as
 *  @c flat_pool::unsafe_join in `flat.hpp` does, over the caller's @p seen_epoch and scratch. */
inline join(t, joined, seen_epoch, seen_fork, seen_callback, before, observed) {
    assert((joined % 2) == 1);
    load(t, epoch, order_acquire, seen_epoch);
    if
    :: seen_epoch == joined ->
        contribute(t, joined, false, caller_completion_order, seen_fork, seen_callback, before, observed);
        do
        :: load(t, epoch, join_order, seen_epoch);
           if
           :: seen_epoch != joined -> break
           :: else -> wait_on(t, epoch, joined, false)
           fi
        od
    :: else
    fi
}

/** The slot claimed from empty before the pool, handed back if the pool refuses, as
 *  @c fu_pool_unsafe_for_threads in `c/forkunion.cpp` does, over the dispatch's scratch. */
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

/** An empty slot or another generation's token returns at once, else the pool's join and then the
 *  clear, released last, as @c fu_pool_unsafe_join in `c/forkunion.cpp` does, over its scratch. */
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

/** The worker loop of @c flat_pool::worker_loop_ in `flat.hpp`, until die: waits on the epoch,
 *  looks at the mood only while the epoch stands still, naps while chill, and contributes to
 *  every odd epoch. */
inline serve(t) {
    int last_epoch, new_epoch, seen_mood, before, observed, seen_fork, seen_callback;
    do
    :: load(t, epoch, pickup_order, new_epoch);
       if
       :: new_epoch == last_epoch ->
           load(t, mood, order_acquire, seen_mood);
           if
           :: seen_mood == die -> break
           :: seen_mood == chill -> skip // the nap, then another look
           :: else -> wait_on(t, epoch, last_epoch, wait_cap && newest_value(mood) != grind)
           fi
       :: else ->
           if
           :: (new_epoch % 2) == 1 ->
               contribute(t, new_epoch, true, worker_completion_order, seen_fork, seen_callback, before, observed)
           :: else
           fi;
           last_epoch = new_epoch
       fi
    od;
    workers_exited++
}
