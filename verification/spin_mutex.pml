/**
 *  @file verification/spin_mutex.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Spin model of @c spin_mutex from `include/forkunion/types.hpp`: @c lock, @c lock_if_free
 *      and @c unlock over one flag.
 *
 *  @c lock exchanges the flag to held with an acquire, and while it finds it held, waits on the
 *  flag with relaxed loads and a capped monitored wait; @c lock_if_free makes one such exchange and
 *  reports whether it took the flag; @c unlock stores it free with a release. Two threads take the
 *  lock and a third tries it once, each holder adding one to a tally with a relaxed read and a
 *  relaxed write, and once all three are done the checker takes the lock and reads the tally.
 *
 *  One holder at a time, and no increment is lost: the checker reads a tally of every holder.
 *
 *  @verify pass sc,rc11
 *  @verify fail rc11 lock_order=order_relaxed: @c lock acquires with its exchange; relaxed, the
 *      next holder reads the tally from before the last holder's increment, which is lost
 *  @verify fail rc11 lock_if_free_order=order_relaxed: @c lock_if_free acquires with its exchange;
 *      relaxed, a holder it admits reads the tally from before the last holder's increment, which
 *      is then lost
 *  @verify fail rc11 unlock_order=order_relaxed: @c unlock releases with its store; relaxed, the
 *      next holder reads the tally from before the last holder's increment, which is lost
 *  @verify pass sc recheck=false: the wait in @c lock is capped, so a release that lands between
 *      its load and its arming costs one timeout and never the lock
 */
#define thread_count 4
#define location_count 2
#define history_depth 12
#include "weak_memory.pml"
#include "monitor_wait.pml"

/** The words: the flag, held while it reads one, and the tally its holders add to. */
#define flag 0
#define tally 1

/** The knobs: the header's choices, each overridden by a `@verify` line. */
#ifndef lock_order
#define lock_order order_acquire
#endif
#ifndef lock_if_free_order
#define lock_if_free_order order_acquire
#endif
#ifndef unlock_order
#define unlock_order order_release
#endif

byte holders;  // threads between a lock and its unlock
byte admitted; // holders so far, the tally's due
byte finished; // threads done with the mutex

/** The exchange, retried after a capped wait while the flag reads held: @c spin_mutex::lock, over
 *  the caller's @p seen_flag. */
inline lock(t, seen_flag) {
    do
    :: atomic { read_modify_write(t, flag, lock_order, seen_flag, 1); wake(flag) };
       if
       :: seen_flag == 0 -> break
       :: else ->
           do
           :: load(t, flag, order_relaxed, seen_flag);
              if
              :: seen_flag == 0 -> break
              :: else -> wait_on(t, flag, 1, newest_value(flag) == 0)
              fi
           od
       fi
    od
}

/** One exchange, holding the flag only if it read free: @c spin_mutex::lock_if_free, over the
 *  caller's @p seen_flag. */
inline lock_if_free(t, seen_flag) {
    atomic { read_modify_write(t, flag, lock_if_free_order, seen_flag, 1); wake(flag) }
}

/** The release store of a free flag: @c spin_mutex::unlock. */
inline unlock(t) {
    atomic { store(t, flag, unlock_order, 0); wake(flag) }
}

/** One increment of the tally under the held flag, over the caller's @p seen_tally. */
inline increment(t, seen_tally) {
    atomic { holders++; admitted++; assert(holders == 1) };
    load(t, tally, order_relaxed, seen_tally);
    store(t, tally, order_relaxed, seen_tally + 1);
    holders--
}

/** A thread that takes the lock, adds one, and lets it go. */
proctype locker(byte t) {
    int seen_flag, seen_tally;
    lock(t, seen_flag);
    increment(t, seen_tally);
    unlock(t);
    finished++
}

/** A thread that tries the lock once, and adds one only if it got it. */
proctype trier(byte t) {
    int seen_flag, seen_tally;
    lock_if_free(t, seen_flag);
    if
    :: seen_flag == 0 ->
        increment(t, seen_tally);
        unlock(t)
    :: else
    fi;
    finished++
}

/** Once every other thread is done, takes the lock and reads the tally. */
proctype checker(byte t) {
    int seen_flag, seen_tally;
    (finished == 3);
    lock(t, seen_flag);
    load(t, tally, order_relaxed, seen_tally);
    assert(seen_tally == admitted);
    unlock(t)
}

init { atomic { run locker(0); run locker(1); run trier(2); run checker(3) } }
