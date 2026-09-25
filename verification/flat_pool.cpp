/**
 *  @file verification/flat_pool.cpp
 *  @author Ash Vardanian
 *  @date September 9, 2026
 *  @brief GenMC client for @c flat_pool's fork-join protocol in `include/forkunion/flat.hpp`: the
 *      epoch clock, the countdown, and what a completed join sees.
 *
 *  Everything here is spelled line for line over @c std::atomic. The pool itself spawns through
 *  @c std::thread, which rides on the platform's @c pthread_create that GenMC does not intercept,
 *  so its words stand alone here.
 *
 *  A dispatcher runs one generation on a caller-inclusive pool with two workers, contributes its
 *  own slice inside the join, and after the completion step reads every worker's result.
 *  `-Dwithout_decrement_acquire` weakens the @c acq_rel decrements to release: the last contributor
 *  drops the acquire on its peers, and the join reads a stale result. The companion Promela model
 *  covers the same scenario in `flat_pool.pml`.
 */
#include <cstddef> // `std::size_t`

#include <atomic> // `std::atomic` - the epoch and the countdown

#include "genmc.hpp"

#ifdef without_decrement_acquire
constexpr std::memory_order decrement_order_k = std::memory_order_release;
#else
constexpr std::memory_order decrement_order_k = std::memory_order_acq_rel;
#endif

constexpr std::size_t contributors_k = 3;
std::atomic<std::size_t> threads_to_sync {0};
std::atomic<std::size_t> epoch {0};
std::size_t fork_state = 0;
std::size_t results[contributors_k] = {0, 0, 0};

/** One slice: the fork state, the result, the decrement, and the completion step for
 *  the last one. */
void contribute(std::size_t thread_index, std::size_t generation) noexcept {
    verify(fork_state == generation);
    results[thread_index] = generation;
    std::size_t const before_decrement = threads_to_sync.fetch_sub(1, decrement_order_k);
    verify(before_decrement > 0);
    if (before_decrement == 1) epoch.fetch_add(1, std::memory_order_release);
}

/** A worker: waits for the dispatch with acquire, then contributes the slice named by @p slot. */
void *work(void *slot) noexcept {
    std::size_t const thread_index = *static_cast<std::size_t *>(slot);
    std::size_t new_epoch;
    while ((new_epoch = epoch.load(std::memory_order_acquire)) == 0) {}
    if (new_epoch & 1) contribute(thread_index, new_epoch);
    return nullptr;
}

int main() {
    std::size_t const slots[contributors_k] = {0, 1, 2};
    thread_t const workers[2] = {spawn(work, const_cast<std::size_t *>(&slots[1])),
                                 spawn(work, const_cast<std::size_t *>(&slots[2]))};

    // The fork, the countdown reset, the release step of the epoch: `flat_pool::unsafe_for_threads`
    verify(threads_to_sync.load(std::memory_order_acquire) == 0);
    fork_state = 1;
    threads_to_sync.store(contributors_k, std::memory_order_relaxed);
    std::size_t const generation = epoch.fetch_add(1, std::memory_order_release) + 1;

    // The caller's slice, then the wait for the completion step: `flat_pool::unsafe_join`
    if (epoch.load(std::memory_order_acquire) == generation) contribute(0, generation);
    while (epoch.load(std::memory_order_acquire) == generation) {}

    // A true `flat_pool::is_complete` synchronizes with every contributor
    verify(results[1] == generation);
    verify(results[2] == generation);
    join(workers[0]);
    join(workers[1]);
    return 0;
}
