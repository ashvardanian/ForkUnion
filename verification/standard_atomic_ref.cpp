/**
 *  @brief GenMC client for the portable `standard_atomic_ref` loops in `include/forkunion/atomics.hpp`:
 *      two admitters through `fetch_add_if_at_most` against a ceiling of one, and two racing
 *      `fetch_max` calls. The ceiling is never crossed, every admission is counted, and the
 *      maximum is the maximum. Two of each: the read-first loops multiply GenMC's executions.

 *  @author Ash Vardanian
 *  @file verification/standard_atomic_ref.cpp
 *  @date September 9, 2026
 * */
#include <cstdint> // `std::uint32_t` - the words the loops move

#include <atomic> // `std::memory_order`

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the portable loops under test

#include "genmc.hpp"

namespace fu = ashvardanian::forkunion;

constexpr std::uint32_t ceiling_k = 1;
std::uint32_t admitted = 0;
std::uint32_t admissions[2] = {0, 0};
std::uint32_t highest = 0;

void *admit(void *slot) noexcept {
    std::uint32_t const observed =
        fu::standard_atomic_ref<std::uint32_t>(admitted).fetch_add_if_at_most(1, ceiling_k, std::memory_order_relaxed);
    *static_cast<std::uint32_t *>(slot) = observed + 1 <= ceiling_k;
    return nullptr;
}

void *raise(void *operand) noexcept {
    fu::standard_atomic_ref<std::uint32_t>(highest).fetch_max(*static_cast<std::uint32_t *>(operand),
                                                              std::memory_order_relaxed);
    return nullptr;
}

int main() {
    thread_t admitters[2] = {spawn(admit, &admissions[0]), spawn(admit, &admissions[1])};
    join(admitters[0]);
    join(admitters[1]);
    verify(admitted <= ceiling_k);
    verify(admitted == admissions[0] + admissions[1]);

    std::uint32_t five = 5, seven = 7;
    thread_t raisers[2] = {spawn(raise, &five), spawn(raise, &seven)};
    join(raisers[0]);
    join(raisers[1]);
    verify(highest == 7);
    return 0;
}
