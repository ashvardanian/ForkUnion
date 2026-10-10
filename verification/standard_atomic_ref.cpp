/**
 *  @file verification/standard_atomic_ref.cpp
 *  @author Ash Vardanian
 *  @date September 9, 2026
 *  @brief GenMC client for the portable loops of @c standard_atomic_ref in `forkunion/atomics.hpp`.
 *
 *  Two admitters run through the free conditional add loop, @c atomic_fetch_add_if_at_most,
 *  against a ceiling of one, and two @c fetch_max calls race. Each thread writes a payload of its
 *  own before its call, which releases. The ceiling is never crossed, every admission is counted,
 *  and the maximum is the maximum. A refused call acquires the value that refused it, so the
 *  refused admitter reads the admitted one's payload, and a raiser refused by a higher maximum
 *  reads the payload of the raiser that set it. Two of each, since every read-first loop
 *  multiplies the executions GenMC explores.
 *
 *  @verify pass sc,rc11,imm standard_atomic_ref
 *  @verify fail rc11,imm standard_atomic_ref_bounded_refusal_order_relaxed: the loop of
 *      @c atomic_fetch_add_if_at_most reads the word, and re-reads it on a failed exchange, with an
 *      acquire; relaxed, the refused admitter learns the ceiling is reached without the admitted
 *      one's writes, and reads its payload racing with the write
 *  @verify fail rc11,imm standard_atomic_ref_max_refusal_order_relaxed: the loop of @c fetch_max
 *      reads the word, and re-reads it on a failed exchange, with an acquire; relaxed, the raiser
 *      refused by a higher maximum reads the payload of the raiser that set it, racing the write
 */
#include <cstdint> // `std::uint32_t` - the words the loops move

#include <atomic> // `std::atomic_ref`, `std::memory_order`

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the portable loops under test

#include "genmc.hpp"

namespace fu = ashvardanian::forkunion;

/** The header's choices; each mutant entry changes one, and runs the loop spelled here instead. */
struct knobs_t {
    std::memory_order bounded_refusal_order = std::memory_order_acquire;
    std::memory_order max_refusal_order = std::memory_order_acquire;
};

constexpr knobs_t header_k {};
constexpr knobs_t bounded_refusal_order_relaxed_k {.bounded_refusal_order = std::memory_order_relaxed};
constexpr knobs_t max_refusal_order_relaxed_k {.max_refusal_order = std::memory_order_relaxed};

constexpr std::uint32_t ceiling_k = 1;
std::uint32_t admitted = 0;
std::uint32_t admissions[2] = {0, 0};
std::uint32_t highest = 0;
std::uint32_t payloads[4] = {0, 0, 0, 0}; // plain: the admitters' two, then the raisers'

/** Admits one through @c fetch_add_if_at_most against the ceiling, after writing its payload, and
 *  records in its admission slot whether it got in; refused, it reads the other's payload. */
template <knobs_t const &knobs_>
void *admit(void *index) noexcept {
    std::uint32_t const self = *static_cast<std::uint32_t *>(index);
    payloads[self] = 1;
    std::uint32_t observed;
    if constexpr (knobs_.bounded_refusal_order == header_k.bounded_refusal_order)
        observed = fu::standard_atomic_ref<std::uint32_t>(admitted).fetch_add_if_at_most(1, ceiling_k,
                                                                                         std::memory_order_release);
    else {
        // `atomic_fetch_add_if_at_most`'s portable loop, line for line
        std::atomic_ref<std::uint32_t> const reference(admitted);
        observed = reference.load(knobs_.bounded_refusal_order);
        while (observed <= ceiling_k && ceiling_k - observed >= 1 &&
               !reference.compare_exchange_weak(observed, observed + 1, std::memory_order_release,
                                                knobs_.bounded_refusal_order)) {}
    }
    admissions[self] = observed + 1 <= ceiling_k;
    if (!admissions[self]) verify(payloads[1 - self] == 1);
    return nullptr;
}

/** Raises the shared maximum to `5 + 2 * index` through @c fetch_max, after writing its payload;
 *  refused by a higher maximum, it reads the payload of the raiser that set it. */
template <knobs_t const &knobs_>
void *raise(void *index) noexcept {
    std::uint32_t const self = *static_cast<std::uint32_t *>(index);
    std::uint32_t const operand = 5 + 2 * self;
    payloads[2 + self] = 1;
    std::uint32_t observed;
    if constexpr (knobs_.max_refusal_order == header_k.max_refusal_order)
        observed = fu::standard_atomic_ref<std::uint32_t>(highest).fetch_max(operand, std::memory_order_release);
    else {
        // `standard_atomic_ref::fetch_max`'s portable loop, line for line
        std::atomic_ref<std::uint32_t> const reference(highest);
        observed = reference.load(knobs_.max_refusal_order);
        while (observed < operand && !reference.compare_exchange_weak(observed, operand, std::memory_order_release,
                                                                      knobs_.max_refusal_order)) {}
    }
    if (observed > operand) verify(payloads[2 + 1 - self] == 1);
    return nullptr;
}

template <knobs_t const &knobs_>
int admit_and_raise() noexcept {
    std::uint32_t indices[2] = {0, 1};
    thread_t admitters[2] = {spawn(admit<knobs_>, &indices[0]), spawn(admit<knobs_>, &indices[1])};
    join(admitters[0]);
    join(admitters[1]);
    verify(admitted <= ceiling_k);
    verify(admitted == admissions[0] + admissions[1]);

    thread_t raisers[2] = {spawn(raise<knobs_>, &indices[0]), spawn(raise<knobs_>, &indices[1])};
    join(raisers[0]);
    join(raisers[1]);
    verify(highest == 7);
    return 0;
}

extern "C" int standard_atomic_ref() { return admit_and_raise<header_k>(); }
extern "C" int standard_atomic_ref_bounded_refusal_order_relaxed() {
    return admit_and_raise<bounded_refusal_order_relaxed_k>();
}
extern "C" int standard_atomic_ref_max_refusal_order_relaxed() {
    return admit_and_raise<max_refusal_order_relaxed_k>();
}
