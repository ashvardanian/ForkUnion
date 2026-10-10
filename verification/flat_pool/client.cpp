/**
 *  @file verification/flat_pool/client.cpp
 *  @author Ash Vardanian
 *  @date September 9, 2026
 *  @brief GenMC client for the fork-join words of @c flat_pool in `include/forkunion/flat.hpp`.
 *
 *  The epoch clock, the countdown, and what a completed join sees, each spelled line for line over
 *  @c std::atomic. The pool itself spawns through @c std::thread, which rides on the platform's
 *  @c pthread_create that GenMC does not intercept, so its words stand alone here.
 *
 *  A dispatcher runs one generation on a caller-inclusive pool with two workers, contributes its
 *  own slice inside the join, and after the completion step reads every worker's result. The same
 *  scenario is `plain.pml`.
 *
 *  @verify pass sc,rc11,imm plain
 *  @verify fail rc11,imm plain_decrement_order_release: the countdown's decrements are @c acq_rel,
 *      so the last contributor acquires its peers' results before its release step publishes
 *      them; with release alone, the join reads a result racing with its write, under IMM as under
 *      RC11
 *  @verify pass sc plain_decrement_order_release
 */
#include <cstddef> // `std::size_t`

#include <atomic> // `std::atomic` - the epoch and the countdown

#include "genmc.hpp"

/** The header's choices; each mutant entry changes one. */
struct knobs_t {
    std::memory_order decrement_order = std::memory_order_acq_rel;
};

constexpr knobs_t header_k {};
constexpr knobs_t decrement_order_release_k {.decrement_order = std::memory_order_release};

constexpr std::size_t contributors_k = 3;
std::atomic<std::size_t> threads_to_sync {0};
std::atomic<std::size_t> epoch {0};
std::size_t fork_state = 0;
std::size_t results[contributors_k] = {0, 0, 0};

/** One slice: the fork state, the result, the decrement, and the completion step for
 *  the last one. */
template <knobs_t const &knobs_>
void contribute(std::size_t thread_index, std::size_t generation) noexcept {
    verify(fork_state == generation);
    results[thread_index] = generation;
    std::size_t const before_decrement = threads_to_sync.fetch_sub(1, knobs_.decrement_order);
    verify(before_decrement > 0);
    if (before_decrement == 1) epoch.fetch_add(1, std::memory_order_release);
}

/** A worker: waits for the dispatch with acquire, then contributes the slice named by @p slot. */
template <knobs_t const &knobs_>
void *work(void *slot) noexcept {
    std::size_t const thread_index = *static_cast<std::size_t *>(slot);
    std::size_t new_epoch;
    while ((new_epoch = epoch.load(std::memory_order_acquire)) == 0) {}
    if (new_epoch & 1) contribute<knobs_>(thread_index, new_epoch);
    return nullptr;
}

/** The caller: dispatches one generation, contributes its slice inside the join, and reads every
 *  worker's result once the completion step lands. */
template <knobs_t const &knobs_>
int dispatch_and_join() noexcept {
    std::size_t const slots[contributors_k] = {0, 1, 2};
    thread_t const workers[2] = {spawn(work<knobs_>, const_cast<std::size_t *>(&slots[1])),
                                 spawn(work<knobs_>, const_cast<std::size_t *>(&slots[2]))};

    // The fork, the countdown reset, the release step of the epoch: `flat_pool::unsafe_for_threads`
    verify(threads_to_sync.load(std::memory_order_acquire) == 0);
    fork_state = 1;
    threads_to_sync.store(contributors_k, std::memory_order_relaxed);
    std::size_t const generation = epoch.fetch_add(1, std::memory_order_release) + 1;

    // The caller's slice, then the wait for the completion step: `flat_pool::unsafe_join`
    if (epoch.load(std::memory_order_acquire) == generation) contribute<knobs_>(0, generation);
    while (epoch.load(std::memory_order_acquire) == generation) {}

    // A true `flat_pool::is_complete` synchronizes with every contributor
    verify(results[1] == generation);
    verify(results[2] == generation);
    join(workers[0]);
    join(workers[1]);
    return 0;
}

extern "C" int plain() { return dispatch_and_join<header_k>(); }
extern "C" int plain_decrement_order_release() { return dispatch_and_join<decrement_order_release_k>(); }
