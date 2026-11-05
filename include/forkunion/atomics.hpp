/**
 *  @file include/forkunion/atomics.hpp
 *  @author Ash Vardanian
 *  @date September 5, 2026
 *  @brief @c std::atomic_ref replacements that spell every access as the one instruction a CPU
 *      generation has for it, independent of the translation unit's target flags.
 *
 *  @c standard_atomic_ref is the in-house @c std::atomic_ref: the standard operations verbatim plus
 *  the ones the standard lacks - @c fetch_max and @c fetch_min ahead of C++26, for floats too,
 *  no-return @c add, @c sub, @c set_bits, @c clear_bits, @c flip_bits, @c max and @c min, and the
 *  conditional @c fetch_add_if_at_most, @c fetch_add_if_at_least and @c fetch_sub_if_at_least -
 *  spelled portably via compare-exchange. Instruction-set references share that interface and
 *  replace both the portable loops and the compiler's flag-dependent lowering with instructions:
 *
 *  - @c x86_cmpccxadd_atomic_ref: the conditional adds as one @c cmpccxadd; every other operation
 *    is the @c lock-prefixed instruction the compiler emits anyway.
 *  - @c x86_raoint_atomic_ref: the relaxed no-return forms as RAO-INT @c aadd, @c aand, @c aor,
 *    which Intel specifies to execute at the shared cache rather than pull the line.
 *
 *  @c arm64_lse_atomic_ref swaps in Armv8.1 @c swp, @c cas, @c ldadd, @c ldclr, @c ldset, @c ldeor,
 *  @c ldsmax & kin - one instruction per read-modify-write, and the no-return @c st* forms, which a
 *  core may perform near like @c ldadd or far at the interconnect, as its design and firmware pick;
 *  @c ldar and @c stlr for ordered loads & stores. Measured on an M5 Pro, where @c stadd runs near,
 *  against the exclusive loops of a baseline build: 5x uncontended, 3-4x with 18 threads on a word.
 *
 *  - @c arm64_rcpc_atomic_ref: the above with Armv8.3 @c ldapr for acquiring loads - RCpc, which
 *    needn't wait for the core's earlier release stores the way RCsc @c ldar may.
 *  - @c riscv64_atomic_ref: the base A extension - @c amoswap, @c amoadd, @c amoand, @c amoor,
 *    @c amomax, @c amomin & the unsigned twins, with @c x0 as the destination for the no-return
 *    forms - and @c lr and @c sc loops for compare-exchange and byte exchanges; fences around loads
 *    & stores.
 *  - @c riscv64_zacas_atomic_ref: compare-exchange as one @c amocas.
 *  - @c riscv64_zabha_atomic_ref: the byte and halfword @c amo forms, @c amocas among them.
 *  - @c wasm_atomic_ref: the standard operations, with @c memory.atomic.notify after each releasing
 *    write to a 32- or 64-bit word, as no WebAssembly store wakes a @c memory.atomic.wait sleeper.
 *
 *  Only the widths & operations the indexes use are spelled: byte exchanges & compare-exchanges for
 *  flags, 32-bit forms for node ids, 64-bit forms for counters & packed words. The Arm references
 *  spell all four widths, as LSE has byte & halfword forms of every operation, and so does the
 *  @c Zabha one. Float extrema ride the integer ones on their bits where an ISA has those. Every
 *  reference assembles in a baseline translation unit: on Arm the extension is named in the
 *  assembly text - <tt>.arch_extension</tt> - the RISC-V base atomics are the A-extension mnemonics every
 *  @c rv64gc toolchain assembles, and the x86 and RISC-V extension instructions are raw bytes; the
 *  runtime capability bit decides whether it may run. Where inline assembly is unavailable - MSVC -
 *  the Arm64 references are spelled over @c __ldar, @c __stlr, @c __ldapr, @c __swp*, @c __cas* and
 *  the @c _Interlocked* arithmetic, which stays inline only under <tt>/arch:armv8.1</tt>; the x86 and
 *  RISC-V references lack intrinsics for their instructions, so those targets keep
 *  @c standard_atomic_ref. The @c preferred_atomic_ref at the bottom is the newest reference this
 *  unit may run with no runtime probe, reading <tt>FORKUNION_TARGET_\<CAPABILITY\></tt> alone - the
 *  compilation target's promise in a unit that dispatches nothing - for compile-time callers, not
 *  per CPU class.
 *
 *  The header needs the library's @c std::atomic_ref and @c std::bit_cast, so it is empty without
 *  them - a C++20 language mode over a library that still lacks them sees nothing here.
 */
#pragma once
#include <cstdint> // `std::uint8_t`, `std::uint16_t`, `std::uint32_t`, `std::uint64_t`, their signed twins

#include <atomic>   // `std::memory_order`
#include <bit>      // `std::bit_cast`
#include <concepts> // `std::same_as`
#include <limits>   // `std::numeric_limits`
#include <type_traits> // `std::conditional_t`, `std::is_arithmetic_v`, `std::is_trivially_copyable_v`, `std::make_signed_t`

#include "types.hpp" // `capabilities_t`, the bits a reference needs admitted, and the target macros

/*  Gated on a macro `types.hpp` defines, so it follows rather than joins the third-party group. */
#if FORKUNION_HAS_ARM64_ATOMIC_INTRINSICS_
#include <intrin.h> // `__ldar64`, `__stlr64`, `__ldapr64`, `__swpal64`, `__casal64`, `_Interlocked*`
#endif

namespace ashvardanian {
namespace forkunion {

#if defined(__cpp_lib_atomic_ref) && defined(__cpp_lib_bit_cast)

/** A word the instructions move: trivially copyable, one of the four widths, and described by a
 *  @c std::numeric_limits specialization, built-ins and wrappers alike. */
template <typename value_type_>
concept atomic_scalar =
    std::numeric_limits<value_type_>::is_specialized && std::is_trivially_copyable_v<value_type_> &&
    (sizeof(value_type_) == 1 || sizeof(value_type_) == 2 || sizeof(value_type_) == 4 || sizeof(value_type_) == 8);

/** The words arithmetic read-modify-writes apply to: integers other than @c bool, as the standard's
 *  integral @c atomic_ref specialization draws the line, and wrappers whose limits say integer. */
template <typename value_type_>
concept atomic_integer =
    atomic_scalar<value_type_> && std::numeric_limits<value_type_>::is_integer && !std::same_as<value_type_, bool>;

/** The floats the extremal read-modify-writes also apply to, where a NaN on either side leaves the
 *  winner unspecified: IEC 559 words of 32 or 64 bits, such as @c float, @c double and NumKong's
 *  @c f32_t and @c f64_t. */
template <typename value_type_>
concept atomic_float = atomic_scalar<value_type_> && std::numeric_limits<value_type_>::is_iec559 &&
                       (sizeof(value_type_) == 4 || sizeof(value_type_) == 8);

/** The unsigned word a value of one width travels through the instructions as. */
template <typename value_type_>
using atomic_word =
    std::conditional_t<sizeof(value_type_) == 1, std::uint8_t,
                       std::conditional_t<sizeof(value_type_) == 2, std::uint16_t,
                                          std::conditional_t<sizeof(value_type_) == 4, std::uint32_t, std::uint64_t>>>;

/** Whether an order carries acquire or release semantics - the two bits every ISA's mnemonics
 *  encode. */
constexpr bool acquires(std::memory_order order) noexcept {
    return order == std::memory_order_acquire || order == std::memory_order_consume ||
           order == std::memory_order_acq_rel || order == std::memory_order_seq_cst;
}
constexpr bool releases(std::memory_order order) noexcept {
    return order == std::memory_order_release || order == std::memory_order_acq_rel ||
           order == std::memory_order_seq_cst;
}

/** The failure order the standard derives from a single compare-exchange order. */
constexpr std::memory_order failure_order(std::memory_order success) noexcept {
    if (success == std::memory_order_acq_rel) return std::memory_order_acquire;
    if (success == std::memory_order_release) return std::memory_order_relaxed;
    return success;
}

/** @p order raised from relaxed to acquire, for a loop whose refusal acquires: GCC refuses a
 *  relaxed success beside an acquiring failure. */
constexpr std::memory_order acquiring(std::memory_order order) noexcept {
    return order == std::memory_order_relaxed || order == std::memory_order_consume ? std::memory_order_acquire : order;
}

/** A reference spelling the verbs past the standard's - every reference in this header does,
 *  @c std::atomic_ref does not - so code holding either posts through the functions below. */
template <typename reference_type_, typename value_type_>
concept extended_atomic_ref = requires(reference_type_ reference, value_type_ value) {
    reference.add(value, std::memory_order_relaxed);
    reference.sub(value, std::memory_order_relaxed);
    reference.set_bits(value, std::memory_order_relaxed);
    reference.clear_bits(value, std::memory_order_relaxed);
    reference.flip_bits(value, std::memory_order_relaxed);
    reference.fetch_test_and_set_bit(0u, std::memory_order_relaxed);
    reference.fetch_test_and_reset_bit(0u, std::memory_order_relaxed);
    reference.max(value, std::memory_order_relaxed);
    reference.min(value, std::memory_order_relaxed);
    reference.fetch_add_if_at_most(value, value, std::memory_order_relaxed);
    reference.fetch_add_if_at_least(value, value, std::memory_order_relaxed);
    reference.fetch_sub_if_at_least(value, value, std::memory_order_relaxed);
};

/** The no-return read-modify-writes for any reference: the reference's own verb where it spells
 *  one, the standard's returning form with the result discarded otherwise. */
template <typename reference_type_, typename value_type_>
void atomic_add(reference_type_ reference, value_type_ operand, std::memory_order order) noexcept {
    if constexpr (extended_atomic_ref<reference_type_, value_type_>) reference.add(operand, order);
    else reference.fetch_add(operand, order);
}
template <typename reference_type_, typename value_type_>
void atomic_sub(reference_type_ reference, value_type_ operand, std::memory_order order) noexcept {
    if constexpr (extended_atomic_ref<reference_type_, value_type_>) reference.sub(operand, order);
    else reference.fetch_sub(operand, order);
}
template <typename reference_type_, typename value_type_>
void atomic_set_bits(reference_type_ reference, value_type_ bits, std::memory_order order) noexcept {
    if constexpr (extended_atomic_ref<reference_type_, value_type_>) reference.set_bits(bits, order);
    else reference.fetch_or(bits, order);
}
template <typename reference_type_, typename value_type_>
void atomic_clear_bits(reference_type_ reference, value_type_ bits, std::memory_order order) noexcept {
    if constexpr (extended_atomic_ref<reference_type_, value_type_>) reference.clear_bits(bits, order);
    else reference.fetch_and(static_cast<value_type_>(~bits), order);
}
template <typename reference_type_, typename value_type_>
void atomic_flip_bits(reference_type_ reference, value_type_ bits, std::memory_order order) noexcept {
    if constexpr (extended_atomic_ref<reference_type_, value_type_>) reference.flip_bits(bits, order);
    else reference.fetch_xor(bits, order);
}

/** The single-bit claims for any reference: set or clear bit @p bit, below the word's width, and
 *  return whether it was set before - the reference's own verb where it spells one, the standard's
 *  @c fetch_or or @c fetch_and of the mask otherwise. */
template <typename reference_type_>
bool atomic_fetch_test_and_set_bit(reference_type_ reference, unsigned bit, std::memory_order order) noexcept {
    // The type the load returns: the one spelling `std::atomic_ref` and these references share
    using value_t = std::remove_cv_t<decltype(reference.load(std::memory_order_relaxed))>;
    if constexpr (extended_atomic_ref<reference_type_, value_t>) return reference.fetch_test_and_set_bit(bit, order);
    else return (reference.fetch_or(static_cast<value_t>(value_t {1} << bit), order) >> bit) & 1;
}
template <typename reference_type_>
bool atomic_fetch_test_and_reset_bit(reference_type_ reference, unsigned bit, std::memory_order order) noexcept {
    using value_t = std::remove_cv_t<decltype(reference.load(std::memory_order_relaxed))>;
    if constexpr (extended_atomic_ref<reference_type_, value_t>) return reference.fetch_test_and_reset_bit(bit, order);
    else return (reference.fetch_and(static_cast<value_t>(~(value_t {1} << bit)), order) >> bit) & 1;
}

/** The no-return extrema for any reference, integers and floats alike: the reference's own verb
 *  where it spells one, a read-first compare-exchange loop otherwise. */
template <typename reference_type_, typename value_type_>
void atomic_max(reference_type_ reference, value_type_ operand, std::memory_order order) noexcept {
    if constexpr (requires { reference.max(operand, order); }) reference.max(operand, order);
    else {
        value_type_ observed = reference.load(std::memory_order_relaxed);
        while (observed < operand &&
               !reference.compare_exchange_weak(observed, operand, order, std::memory_order_relaxed)) {}
    }
}
template <typename reference_type_, typename value_type_>
void atomic_min(reference_type_ reference, value_type_ operand, std::memory_order order) noexcept {
    if constexpr (requires { reference.min(operand, order); }) reference.min(operand, order);
    else {
        value_type_ observed = reference.load(std::memory_order_relaxed);
        while (observed > operand &&
               !reference.compare_exchange_weak(observed, operand, order, std::memory_order_relaxed)) {}
    }
}

/** The least word whose sum with @p operand stays at least @p floor, into @p bound - clamped to the
 *  type's minimum when every word qualifies. False when none does: a negative operand pushing the
 *  bound past the maximum. */
template <typename value_type_>
constexpr bool fetch_add_if_at_least_bound(value_type_ operand, value_type_ floor, value_type_ &bound) noexcept {
    using limits_t = std::numeric_limits<value_type_>;
    if constexpr (limits_t::is_signed) {
        if (operand < value_type_ {} && floor > limits_t::max() + operand) return false;
        bound = operand > value_type_ {} && floor < limits_t::min() + operand
                    ? limits_t::min()
                    : static_cast<value_type_>(floor - operand);
    }
    else bound = floor < operand ? value_type_ {} : static_cast<value_type_>(floor - operand);
    return true;
}

/** The bounded read-modify-writes for any reference: adds @p operand only if the sum stays at most
 *  @p limit or at least @p floor, or subtracts it only if the difference stays at least @p floor,
 *  returning the value held before. One @c cmpccxadd on Intel; elsewhere a read-first
 *  compare-exchange loop, so a word that already refuses returns without writing and losers never
 *  take the line. */
template <typename reference_type_, typename value_type_>
value_type_ atomic_fetch_add_if_at_most(reference_type_ reference, value_type_ operand, value_type_ limit,
                                        std::memory_order order) noexcept {
    if constexpr (extended_atomic_ref<reference_type_, value_type_>)
        return reference.fetch_add_if_at_most(operand, limit, order);
    else {
        value_type_ observed = reference.load(std::memory_order_acquire);
        while (observed <= limit && limit - observed >= operand &&
               !reference.compare_exchange_weak(observed, observed + operand, order, std::memory_order_acquire)) {}
        return observed;
    }
}
template <typename reference_type_, typename value_type_>
value_type_ atomic_fetch_add_if_at_least(reference_type_ reference, value_type_ operand, value_type_ floor,
                                         std::memory_order order) noexcept {
    if constexpr (extended_atomic_ref<reference_type_, value_type_>)
        return reference.fetch_add_if_at_least(operand, floor, order);
    else {
        value_type_ bound;
        value_type_ observed = reference.load(std::memory_order_acquire);
        if (!fetch_add_if_at_least_bound(operand, floor, bound)) return observed;
        while (observed >= bound &&
               !reference.compare_exchange_weak(observed, static_cast<value_type_>(observed + operand), order,
                                                std::memory_order_acquire)) {}
        return observed;
    }
}
template <typename reference_type_, typename value_type_>
value_type_ atomic_fetch_sub_if_at_least(reference_type_ reference, value_type_ operand, value_type_ floor,
                                         std::memory_order order) noexcept {
    if constexpr (extended_atomic_ref<reference_type_, value_type_>)
        return reference.fetch_sub_if_at_least(operand, floor, order);
    else {
        value_type_ observed = reference.load(std::memory_order_acquire);
        while (observed >= floor && observed - floor >= operand &&
               !reference.compare_exchange_weak(observed, observed - operand, order, std::memory_order_acquire)) {}
        return observed;
    }
}

/**
 *  @brief @c std::atomic_ref with the operations the standard lacks, spelled portably over
 *      compare-exchange loops.
 *
 *  Compare-exchange loops handle the conditional and extremal read-modify-writes; the no-return
 *  ones discard the result, which the compiler lowers to the store-only instruction where the
 *  target has it. The instruction-set references below share this interface and replace the loops
 *  with @c ldsmax, @c stadd, @c cmpccxadd and kin.
 *
 *  Fences can't stand in for any of these: a fence orders accesses, it doesn't make a
 *  read-compare-write atomic. What makes the loops cheap is reading first - a word that already
 *  satisfies the condition returns without writing, so losers never take the line.
 */
template <typename value_type_>
struct standard_atomic_ref {
    using value_t = value_type_;
    using word_t = atomic_word<value_type_>;

    /** The runtime bits a reference needs admitted before it may run - none here. */
    static constexpr capabilities_t capabilities_k = capabilities_unknown_k;

    explicit standard_atomic_ref(value_type_ &word) noexcept : word_(&word) {
        assert(reinterpret_cast<std::uintptr_t>(&word) % std::atomic_ref<value_type_>::required_alignment == 0 &&
               "std::atomic_ref needs the word at its required alignment");
    }

    value_type_ load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return reference_().load(order);
    }
    void store(value_type_ desired, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        reference_().store(desired, order);
    }
    value_type_ exchange(value_type_ desired, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return reference_().exchange(desired, order);
    }

    bool compare_exchange_strong(value_type_ &expected, value_type_ desired, std::memory_order success,
                                 std::memory_order failure) const noexcept {
        return reference_().compare_exchange_strong(expected, desired, success, failure);
    }
    bool compare_exchange_weak(value_type_ &expected, value_type_ desired, std::memory_order success,
                               std::memory_order failure) const noexcept {
        return reference_().compare_exchange_weak(expected, desired, success, failure);
    }
    bool compare_exchange_strong(value_type_ &expected, value_type_ desired,
                                 std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return reference_().compare_exchange_strong(expected, desired, order);
    }
    bool compare_exchange_weak(value_type_ &expected, value_type_ desired,
                               std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return reference_().compare_exchange_weak(expected, desired, order);
    }

    /** Built-ins use the standard's arithmetic, integer wrappers a loop over their bits. */
    value_type_ fetch_add(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        if constexpr (std::is_arithmetic_v<value_type_>) return reference_().fetch_add(operand, order);
        else return fetch_update_bits_(operand, order, [](word_t observed, word_t bits) { return observed + bits; });
    }
    value_type_ fetch_sub(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        if constexpr (std::is_arithmetic_v<value_type_>) return reference_().fetch_sub(operand, order);
        else return fetch_update_bits_(operand, order, [](word_t observed, word_t bits) { return observed - bits; });
    }
    value_type_ fetch_and(value_type_ mask, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        if constexpr (std::is_arithmetic_v<value_type_>) return reference_().fetch_and(mask, order);
        else return fetch_update_bits_(mask, order, [](word_t observed, word_t bits) { return observed & bits; });
    }
    value_type_ fetch_or(value_type_ bits, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        if constexpr (std::is_arithmetic_v<value_type_>) return reference_().fetch_or(bits, order);
        else return fetch_update_bits_(bits, order, [](word_t observed, word_t set) { return observed | set; });
    }
    value_type_ fetch_xor(value_type_ bits, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        if constexpr (std::is_arithmetic_v<value_type_>) return reference_().fetch_xor(bits, order);
        else return fetch_update_bits_(bits, order, [](word_t observed, word_t flipped) { return observed ^ flipped; });
    }

    /** C++26's @c fetch_max: the value held before, whether or not the operand replaced it. */
    value_type_ fetch_max(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> || atomic_float<value_type_>
    {
        std::atomic_ref<value_type_> const reference = reference_();
        value_type_ observed = reference.load(std::memory_order_acquire);
        while (observed < operand &&
               !reference.compare_exchange_weak(observed, operand, acquiring(order), std::memory_order_acquire)) {}
        return observed;
    }
    value_type_ fetch_min(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> || atomic_float<value_type_>
    {
        std::atomic_ref<value_type_> const reference = reference_();
        value_type_ observed = reference.load(std::memory_order_acquire);
        while (observed > operand &&
               !reference.compare_exchange_weak(observed, operand, acquiring(order), std::memory_order_acquire)) {}
        return observed;
    }

    /** Adds @p operand only if the sum stays at most @p limit - one @c cmpccxadd on Intel, a
     *  read-first compare-exchange loop elsewhere. Returns the value held before; the caller learns
     *  the outcome from <tt>observed + operand <= limit</tt>. An operand past the limit, or a floor the
     *  operand cannot be taken from, merely observes the word - nothing is required of either,
     *  beyond a signed sum or difference that stays representable. */
    value_type_ fetch_add_if_at_most(value_type_ operand, value_type_ limit,
                                     std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        // A plain `std::atomic_ref`, or the free verb would post straight back through this one.
        return atomic_fetch_add_if_at_most(reference_(), operand, limit, order);
    }

    /** Adds @p operand only if the sum stays at least @p floor - one @c cmpccxadd on Intel. Returns
     *  the value held before; the outcome is <tt>observed + operand >= floor</tt>. A <tt>floor - operand</tt>
     *  below the type's minimum admits every word, one past its maximum none. */
    value_type_ fetch_add_if_at_least(value_type_ operand, value_type_ floor,
                                      std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        // A plain `std::atomic_ref`, or the free verb would post straight back through this one.
        return atomic_fetch_add_if_at_least(reference_(), operand, floor, order);
    }

    /** Subtracts @p operand only if the difference stays at least @p floor - the semaphore acquire.
     *  Returns the value held before; the outcome is <tt>observed >= floor + operand</tt>. */
    value_type_ fetch_sub_if_at_least(value_type_ operand, value_type_ floor,
                                      std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        // A plain `std::atomic_ref`, or the free verb would post straight back through this one.
        return atomic_fetch_sub_if_at_least(reference_(), operand, floor, order);
    }

    /** No-return read-modify-writes: the op is posted, nothing is waited for - @c stadd, @c stclr,
     *  @c stset on Arm, <tt>lock add</tt> on x86, the remote @c aadd family with RAO-INT. Only relaxed and
     *  release orders exist for them, so release is the default: with no value returned there is
     *  nothing to acquire. */
    void add(value_type_ operand, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        fetch_add(operand, order);
    }
    void sub(value_type_ operand, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        fetch_sub(operand, order);
    }
    void set_bits(value_type_ bits, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        fetch_or(bits, order);
    }
    void clear_bits(value_type_ bits, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        fetch_and(std::bit_cast<value_type_>(static_cast<word_t>(~std::bit_cast<word_t>(bits))), order);
    }
    void flip_bits(value_type_ bits, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        fetch_xor(bits, order);
    }

    /** Sets or clears bit @p bit, below the word's width, returning whether it was set before: a
     *  claim exactly one of several racing callers wins. @c ldset and @c ldclr on Arm, @c amoor and
     *  @c amoand on RISC-V, @c lock @c bts and @c lock @c btr on x86. */
    bool fetch_test_and_set_bit(unsigned bit, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        word_t const mask = static_cast<word_t>(word_t {1} << bit);
        return (std::bit_cast<word_t>(fetch_or(std::bit_cast<value_type_>(mask), order)) >> bit) & 1;
    }
    bool fetch_test_and_reset_bit(unsigned bit, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        word_t const mask = static_cast<word_t>(~(word_t {1} << bit));
        return (std::bit_cast<word_t>(fetch_and(std::bit_cast<value_type_>(mask), order)) >> bit) & 1;
    }
    void max(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> || atomic_float<value_type_>
    {
        fetch_max(operand, order);
    }
    void min(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> || atomic_float<value_type_>
    {
        fetch_min(operand, order);
    }

  private:
    value_type_ *word_;

    /** A @c std::atomic_ref wraps just a pointer, so one per operation costs nothing. */
    std::atomic_ref<value_type_> reference_() const noexcept { return std::atomic_ref<value_type_>(*word_); }

    /** An integer wrapper's read-modify-write, which @c std::atomic_ref spells only for built-ins:
     *  @p combine applied to the bits in a compare-exchange loop, so no storage is punned. */
    template <typename combine_type_>
    value_type_ fetch_update_bits_(value_type_ operand, std::memory_order order, combine_type_ combine) const noexcept
        requires atomic_integer<value_type_>
    {
        std::atomic_ref<value_type_> const reference = reference_();
        word_t const operand_bits = std::bit_cast<word_t>(operand);
        value_type_ observed = reference.load(std::memory_order_relaxed);
        while (!reference.compare_exchange_weak(
            observed,
            std::bit_cast<value_type_>(static_cast<word_t>(combine(std::bit_cast<word_t>(observed), operand_bits))),
            order, std::memory_order_relaxed)) {}
        return observed;
    }
};

/**
 *  @brief Read-only access to a word other threads may be writing: the loads of @ref
 *      standard_atomic_ref, reached down a path that may not write.
 *
 *  @c const is a permission on the path, never a promise that the word holds still - a reader
 *  holding a const handle still shares the storage a writer mutates. This is the one reference that
 *  cannot reach its word without a cast: @c std::atomic_ref binds no const word before P3323, and
 *  no other standard spelling loads atomically from a plain word. Every ISA reference below binds
 *  its own const pointer and needs none.
 */
template <typename value_type_>
struct standard_atomic_ref<value_type_ const> {
    using value_t = value_type_ const;
    static constexpr capabilities_t capabilities_k = capabilities_unknown_k;

    explicit standard_atomic_ref(value_type_ const &word) noexcept : word_(&word) {
        assert(reinterpret_cast<std::uintptr_t>(&word) % std::atomic_ref<value_type_>::required_alignment == 0 &&
               "std::atomic_ref needs the word at its required alignment");
    }

    /** A temporary would die before the reference does. */
    standard_atomic_ref(value_type_ const &&) = delete;

    value_type_ load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return std::atomic_ref<value_type_>(const_cast<value_type_ &>(*word_)).load(order);
    }

  private:
    value_type_ const *word_;
};

/*  @c cmpccxadd: compares the word against @c bound - flags from `word - bound` - and adds
 *  @c addend only when the condition holds; the register handed as @c bound receives what the word
 *  held. Spelled as bytes, since binutils before 2.40 and LLVM before 16 have no mnemonic: the VEX
 *  form in map 0F38, opcode E0 plus the condition, @c W set for the 64-bit forms; the word in
 *  @c rax, the compare-and-return register in @c rcx, the addend in @c rdx. */
#if FORKUNION_TARGET_X86_CMPCCXADD
#pragma region x86 CMPCCXADD

inline std::uint32_t x86_cmpbexadd_u32(std::uint32_t *word, std::uint32_t bound, std::uint32_t addend) noexcept {
    __asm__ __volatile__(".byte 0xc4, 0xe2, 0x69, 0xe6, 0x08" // ? `cmpbexadd %edx, %ecx, (%rax)`
                         : "+c"(bound)
                         : "d"(addend), "a"(word)
                         : "memory", "cc");
    return bound;
}
inline std::uint64_t x86_cmpbexadd_u64(std::uint64_t *word, std::uint64_t bound, std::uint64_t addend) noexcept {
    __asm__ __volatile__(".byte 0xc4, 0xe2, 0xe9, 0xe6, 0x08" // ? `cmpbexadd %rdx, %rcx, (%rax)`
                         : "+c"(bound)
                         : "d"(addend), "a"(word)
                         : "memory", "cc");
    return bound;
}
inline std::int32_t x86_cmplexadd_i32(std::int32_t *word, std::int32_t bound, std::int32_t addend) noexcept {
    __asm__ __volatile__(".byte 0xc4, 0xe2, 0x69, 0xee, 0x08" // ? `cmplexadd %edx, %ecx, (%rax)`
                         : "+c"(bound)
                         : "d"(addend), "a"(word)
                         : "memory", "cc");
    return bound;
}
inline std::int64_t x86_cmplexadd_i64(std::int64_t *word, std::int64_t bound, std::int64_t addend) noexcept {
    __asm__ __volatile__(".byte 0xc4, 0xe2, 0xe9, 0xee, 0x08" // ? `cmplexadd %rdx, %rcx, (%rax)`
                         : "+c"(bound)
                         : "d"(addend), "a"(word)
                         : "memory", "cc");
    return bound;
}
inline std::uint32_t x86_cmpaexadd_u32(std::uint32_t *word, std::uint32_t bound, std::uint32_t addend) noexcept {
    __asm__ __volatile__(".byte 0xc4, 0xe2, 0x69, 0xe3, 0x08" // ? `cmpaexadd %edx, %ecx, (%rax)`
                         : "+c"(bound)
                         : "d"(addend), "a"(word)
                         : "memory", "cc");
    return bound;
}
inline std::uint64_t x86_cmpaexadd_u64(std::uint64_t *word, std::uint64_t bound, std::uint64_t addend) noexcept {
    __asm__ __volatile__(".byte 0xc4, 0xe2, 0xe9, 0xe3, 0x08" // ? `cmpaexadd %rdx, %rcx, (%rax)`
                         : "+c"(bound)
                         : "d"(addend), "a"(word)
                         : "memory", "cc");
    return bound;
}
inline std::int32_t x86_cmpgexadd_i32(std::int32_t *word, std::int32_t bound, std::int32_t addend) noexcept {
    __asm__ __volatile__(".byte 0xc4, 0xe2, 0x69, 0xed, 0x08" // ? `cmpgexadd %edx, %ecx, (%rax)`
                         : "+c"(bound)
                         : "d"(addend), "a"(word)
                         : "memory", "cc");
    return bound;
}
inline std::int64_t x86_cmpgexadd_i64(std::int64_t *word, std::int64_t bound, std::int64_t addend) noexcept {
    __asm__ __volatile__(".byte 0xc4, 0xe2, 0xe9, 0xed, 0x08" // ? `cmpgexadd %rdx, %rcx, (%rax)`
                         : "+c"(bound)
                         : "d"(addend), "a"(word)
                         : "memory", "cc");
    return bound;
}

/*  @c lock @c bts and @c lock @c btr: the claimed bit's old value lands in the carry flag. Spelled
 *  as bytes like the rest of this region, so no assembler dialect matters: the word in @c rax, the
 *  bit index in @c rcx. */
inline bool x86_lock_bts_u32(std::uint32_t *word, std::uint32_t bit) noexcept {
    bool was_set;
    __asm__ __volatile__(".byte 0xf0, 0x0f, 0xab, 0x08" // ? `lock bts %ecx, (%rax)`
                         : "=@ccc"(was_set)
                         : "c"(bit), "a"(word)
                         : "memory");
    return was_set;
}
inline bool x86_lock_bts_u64(std::uint64_t *word, std::uint64_t bit) noexcept {
    bool was_set;
    __asm__ __volatile__(".byte 0xf0, 0x48, 0x0f, 0xab, 0x08" // ? `lock bts %rcx, (%rax)`
                         : "=@ccc"(was_set)
                         : "c"(bit), "a"(word)
                         : "memory");
    return was_set;
}
inline bool x86_lock_btr_u32(std::uint32_t *word, std::uint32_t bit) noexcept {
    bool was_set;
    __asm__ __volatile__(".byte 0xf0, 0x0f, 0xb3, 0x08" // ? `lock btr %ecx, (%rax)`
                         : "=@ccc"(was_set)
                         : "c"(bit), "a"(word)
                         : "memory");
    return was_set;
}
inline bool x86_lock_btr_u64(std::uint64_t *word, std::uint64_t bit) noexcept {
    bool was_set;
    __asm__ __volatile__(".byte 0xf0, 0x48, 0x0f, 0xb3, 0x08" // ? `lock btr %rcx, (%rax)`
                         : "=@ccc"(was_set)
                         : "c"(bit), "a"(word)
                         : "memory");
    return was_set;
}

/**
 *  @brief The standard reference with the conditional adds as one @c cmpccxadd - every other
 *      operation is already the @c lock-prefixed instruction the compiler emits.
 *  @sa capability_x86_cmpccxadd_k, the admitting bit; x86_raoint_atomic_ref, the same for RAO-INT.
 *
 *  Intel cores from the 2024 E-core Xeons on; <tt>CPUID.(7,1):EAX[7]</tt> says so at runtime. The standard
 *  reference hides its pointer, so the word's address is kept alongside.
 */
template <typename value_type_>
struct x86_cmpccxadd_atomic_ref {
    using value_t = value_type_;
    using word_t = atomic_word<value_type_>;
    using signed_word_t = std::make_signed_t<word_t>;
    static constexpr signed_word_t signed_bits_(word_t bits) noexcept { return std::bit_cast<signed_word_t>(bits); }
    static constexpr capabilities_t capabilities_k = capability_x86_cmpccxadd_k;

    explicit x86_cmpccxadd_atomic_ref(value_type_ &word) noexcept : word_(&word) {
        assert(reinterpret_cast<std::uintptr_t>(&word) % sizeof(value_type_) == 0 &&
               "Atomics need a naturally aligned word");
    }

    value_type_ load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return portable_().load(order);
    }
    void store(value_type_ desired, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        portable_().store(desired, order);
    }
    value_type_ exchange(value_type_ desired, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return portable_().exchange(desired, order);
    }

    bool compare_exchange_strong(value_type_ &expected, value_type_ desired, std::memory_order success,
                                 std::memory_order failure) const noexcept {
        return portable_().compare_exchange_strong(expected, desired, success, failure);
    }
    bool compare_exchange_weak(value_type_ &expected, value_type_ desired, std::memory_order success,
                               std::memory_order failure) const noexcept {
        return portable_().compare_exchange_weak(expected, desired, success, failure);
    }
    bool compare_exchange_strong(value_type_ &expected, value_type_ desired,
                                 std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return portable_().compare_exchange_strong(expected, desired, order);
    }
    bool compare_exchange_weak(value_type_ &expected, value_type_ desired,
                               std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return portable_().compare_exchange_weak(expected, desired, order);
    }

    value_type_ fetch_add(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return portable_().fetch_add(operand, order);
    }
    value_type_ fetch_sub(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return portable_().fetch_sub(operand, order);
    }
    value_type_ fetch_and(value_type_ mask, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return portable_().fetch_and(mask, order);
    }
    value_type_ fetch_or(value_type_ bits, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return portable_().fetch_or(bits, order);
    }
    value_type_ fetch_xor(value_type_ bits, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return portable_().fetch_xor(bits, order);
    }
    value_type_ fetch_max(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> || atomic_float<value_type_>
    {
        return portable_().fetch_max(operand, order);
    }
    value_type_ fetch_min(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> || atomic_float<value_type_>
    {
        return portable_().fetch_min(operand, order);
    }

    void add(value_type_ operand, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        portable_().add(operand, order);
    }
    void sub(value_type_ operand, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        portable_().sub(operand, order);
    }
    void set_bits(value_type_ bits, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        portable_().set_bits(bits, order);
    }
    void clear_bits(value_type_ bits, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        portable_().clear_bits(bits, order);
    }
    void flip_bits(value_type_ bits, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        portable_().flip_bits(bits, order);
    }
    bool fetch_test_and_set_bit(unsigned bit, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        if constexpr (sizeof(value_type_) == 4) return x86_lock_bts_u32(reinterpret_cast<word_t *>(word_), bit);
        else if constexpr (sizeof(value_type_) == 8) return x86_lock_bts_u64(reinterpret_cast<word_t *>(word_), bit);
        else return portable_().fetch_test_and_set_bit(bit, order);
    }
    bool fetch_test_and_reset_bit(unsigned bit, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        if constexpr (sizeof(value_type_) == 4) return x86_lock_btr_u32(reinterpret_cast<word_t *>(word_), bit);
        else if constexpr (sizeof(value_type_) == 8) return x86_lock_btr_u64(reinterpret_cast<word_t *>(word_), bit);
        else return portable_().fetch_test_and_reset_bit(bit, order);
    }
    void max(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> || atomic_float<value_type_>
    {
        portable_().max(operand, order);
    }
    void min(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> || atomic_float<value_type_>
    {
        portable_().min(operand, order);
    }

    value_type_ fetch_add_if_at_most(value_type_ operand, value_type_ limit,
                                     std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> && (sizeof(value_type_) == 4 || sizeof(value_type_) == 8)
    {
        // ? `limit - operand` could wrap below for a negative operand
        if constexpr (std::numeric_limits<value_type_>::is_signed)
            if (operand < 0) return portable_().fetch_add_if_at_most(operand, limit, order);
        // Adds while `word <= limit - operand`: below-or-equal unsigned, less-or-equal signed.
        if (operand > limit) return portable_().load(std::memory_order_acquire); // ? Nothing could be admitted
        word_t const bound = std::bit_cast<word_t>(static_cast<value_type_>(limit - operand));
        word_t const addend = std::bit_cast<word_t>(operand);
        if constexpr (std::numeric_limits<value_type_>::is_signed && sizeof(value_type_) == 4)
            return std::bit_cast<value_type_>(
                x86_cmplexadd_i32(reinterpret_cast<std::int32_t *>(word_), signed_bits_(bound), signed_bits_(addend)));
        else if constexpr (std::numeric_limits<value_type_>::is_signed)
            return std::bit_cast<value_type_>(
                x86_cmplexadd_i64(reinterpret_cast<std::int64_t *>(word_), signed_bits_(bound), signed_bits_(addend)));
        else if constexpr (sizeof(value_type_) == 4)
            return std::bit_cast<value_type_>(
                x86_cmpbexadd_u32(reinterpret_cast<std::uint32_t *>(word_), bound, addend));
        else
            return std::bit_cast<value_type_>(
                x86_cmpbexadd_u64(reinterpret_cast<std::uint64_t *>(word_), bound, addend));
    }
    value_type_ fetch_add_if_at_least(value_type_ operand, value_type_ floor,
                                      std::memory_order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> && (sizeof(value_type_) == 4 || sizeof(value_type_) == 8)
    {
        // Adds while `word >= floor - operand`: above-or-equal unsigned, at-least signed.
        value_type_ floor_bound;
        if (!fetch_add_if_at_least_bound(operand, floor, floor_bound))
            return portable_().load(std::memory_order_acquire); // ? No word could reach the floor
        word_t const bound = std::bit_cast<word_t>(floor_bound);
        word_t const addend = std::bit_cast<word_t>(operand);
        if constexpr (std::numeric_limits<value_type_>::is_signed && sizeof(value_type_) == 4)
            return std::bit_cast<value_type_>(
                x86_cmpgexadd_i32(reinterpret_cast<std::int32_t *>(word_), signed_bits_(bound), signed_bits_(addend)));
        else if constexpr (std::numeric_limits<value_type_>::is_signed)
            return std::bit_cast<value_type_>(
                x86_cmpgexadd_i64(reinterpret_cast<std::int64_t *>(word_), signed_bits_(bound), signed_bits_(addend)));
        else if constexpr (sizeof(value_type_) == 4)
            return std::bit_cast<value_type_>(
                x86_cmpaexadd_u32(reinterpret_cast<std::uint32_t *>(word_), bound, addend));
        else
            return std::bit_cast<value_type_>(
                x86_cmpaexadd_u64(reinterpret_cast<std::uint64_t *>(word_), bound, addend));
    }
    value_type_ fetch_sub_if_at_least(value_type_ operand, value_type_ floor,
                                      std::memory_order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> && (sizeof(value_type_) == 4 || sizeof(value_type_) == 8)
    {
        // Subtracts while `word >= floor + operand`: above-or-equal unsigned, at-least signed.
        value_type_ const sum = static_cast<value_type_>(floor + operand);
        if (sum < floor)
            return portable_().load(std::memory_order_acquire); // ? The bound wrapped: nothing could be taken
        word_t const bound = std::bit_cast<word_t>(sum);
        word_t const negated = static_cast<word_t>(word_t {0} - std::bit_cast<word_t>(operand));
        if constexpr (std::numeric_limits<value_type_>::is_signed && sizeof(value_type_) == 4)
            return std::bit_cast<value_type_>(
                x86_cmpgexadd_i32(reinterpret_cast<std::int32_t *>(word_), signed_bits_(bound), signed_bits_(negated)));
        else if constexpr (std::numeric_limits<value_type_>::is_signed)
            return std::bit_cast<value_type_>(
                x86_cmpgexadd_i64(reinterpret_cast<std::int64_t *>(word_), signed_bits_(bound), signed_bits_(negated)));
        else if constexpr (sizeof(value_type_) == 4)
            return std::bit_cast<value_type_>(
                x86_cmpaexadd_u32(reinterpret_cast<std::uint32_t *>(word_), bound, negated));
        else
            return std::bit_cast<value_type_>(
                x86_cmpaexadd_u64(reinterpret_cast<std::uint64_t *>(word_), bound, negated));
    }

  private:
    /** The portable floor, built on demand: it holds this same pointer and nothing else. */
    standard_atomic_ref<value_type_> portable_() const noexcept { return standard_atomic_ref<value_type_>(*word_); }

    value_type_ *word_;
};

/**
 *  @brief Read-only access through @ref x86_cmpccxadd_atomic_ref, which specializes no load.
 *  @sa standard_atomic_ref<value_type_ const>
 */
template <typename value_type_>
struct x86_cmpccxadd_atomic_ref<value_type_ const> {
    using value_t = value_type_ const;
    static constexpr capabilities_t capabilities_k = capability_x86_cmpccxadd_k;

    explicit x86_cmpccxadd_atomic_ref(value_type_ const &word) noexcept : portable_(word) {}

    /** A temporary would die before the reference does. */
    x86_cmpccxadd_atomic_ref(value_type_ const &&) = delete;

    value_type_ load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return portable_.load(order);
    }

  private:
    standard_atomic_ref<value_type_ const> portable_;
};

#pragma endregion x86 CMPCCXADD

#endif // FORKUNION_TARGET_X86_CMPCCXADD

/*  RAO-INT: the remote, no-return forms - weakly ordered like write-combining stores, which only
 *  SFENCE or MFENCE order and a C++ release fence never emits on x86, so only the relaxed callers
 *  take them; a release order stays on the @c lock-prefixed base. Bytes for the same reason: map
 *  0F38 opcode FC, the operation picked by the legacy prefix - none for add, 66 for and, F2 for or,
 *  F3 for xor - `REX.W` for the 64-bit forms; the word in @c rax, the operand in @c rcx. */
#if FORKUNION_TARGET_X86_RAOINT
#pragma region x86 RAOINT

inline void x86_aadd_u32(std::uint32_t *word, std::uint32_t operand) noexcept {
    __asm__ __volatile__(".byte 0x0f, 0x38, 0xfc, 0x08"
                         :
                         : "c"(operand), "a"(word)
                         : "memory"); // ? `aadd %ecx, (%rax)`
}
inline void x86_aadd_u64(std::uint64_t *word, std::uint64_t operand) noexcept {
    __asm__ __volatile__(".byte 0x48, 0x0f, 0x38, 0xfc, 0x08"
                         :
                         : "c"(operand), "a"(word)
                         : "memory"); // ? `aadd %rcx, (%rax)`
}
inline void x86_aand_u32(std::uint32_t *word, std::uint32_t mask) noexcept {
    __asm__ __volatile__(".byte 0x66, 0x0f, 0x38, 0xfc, 0x08"
                         :
                         : "c"(mask), "a"(word)
                         : "memory"); // ? `aand %ecx, (%rax)`
}
inline void x86_aand_u64(std::uint64_t *word, std::uint64_t mask) noexcept {
    __asm__ __volatile__(".byte 0x66, 0x48, 0x0f, 0x38, 0xfc, 0x08"
                         :
                         : "c"(mask), "a"(word)
                         : "memory"); // ? `aand %rcx, (%rax)`
}
inline void x86_aor_u32(std::uint32_t *word, std::uint32_t bits) noexcept {
    __asm__ __volatile__(".byte 0xf2, 0x0f, 0x38, 0xfc, 0x08"
                         :
                         : "c"(bits), "a"(word)
                         : "memory"); // ? `aor %ecx, (%rax)`
}
inline void x86_aor_u64(std::uint64_t *word, std::uint64_t bits) noexcept {
    __asm__ __volatile__(".byte 0xf2, 0x48, 0x0f, 0x38, 0xfc, 0x08"
                         :
                         : "c"(bits), "a"(word)
                         : "memory"); // ? `aor %rcx, (%rax)`
}
inline void x86_axor_u32(std::uint32_t *word, std::uint32_t bits) noexcept {
    __asm__ __volatile__(".byte 0xf3, 0x0f, 0x38, 0xfc, 0x08"
                         :
                         : "c"(bits), "a"(word)
                         : "memory"); // ? `axor %ecx, (%rax)`
}
inline void x86_axor_u64(std::uint64_t *word, std::uint64_t bits) noexcept {
    __asm__ __volatile__(".byte 0xf3, 0x48, 0x0f, 0x38, 0xfc, 0x08"
                         :
                         : "c"(bits), "a"(word)
                         : "memory"); // ? `axor %rcx, (%rax)`
}

/**
 *  @brief The above plus RAO-INT for the relaxed no-return forms, specified to execute at the
 *      shared cache: @c aadd, @c aand, @c aor, @c axor. An ordered one keeps the @c lock-prefixed
 *      instruction, already a full fence. <tt>CPUID.(7,1):EAX[3]</tt> says so at runtime.
 *  @sa capability_x86_raoint_k - the bit admitting it, on top of @c capability_x86_cmpccxadd_k.
 */
template <typename value_type_>
struct x86_raoint_atomic_ref {
    using value_t = value_type_;
    using word_t = atomic_word<value_type_>;
    static constexpr capabilities_t capabilities_k = capability_x86_cmpccxadd_k | capability_x86_raoint_k;

    explicit x86_raoint_atomic_ref(value_type_ &word) noexcept : word_(&word) {
        assert(reinterpret_cast<std::uintptr_t>(&word) % sizeof(value_type_) == 0 &&
               "Atomics need a naturally aligned word");
    }

    value_type_ load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return weaker_().load(order);
    }
    void store(value_type_ desired, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        weaker_().store(desired, order);
    }
    value_type_ exchange(value_type_ desired, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return weaker_().exchange(desired, order);
    }

    bool compare_exchange_strong(value_type_ &expected, value_type_ desired, std::memory_order success,
                                 std::memory_order failure) const noexcept {
        return weaker_().compare_exchange_strong(expected, desired, success, failure);
    }
    bool compare_exchange_weak(value_type_ &expected, value_type_ desired, std::memory_order success,
                               std::memory_order failure) const noexcept {
        return weaker_().compare_exchange_weak(expected, desired, success, failure);
    }
    bool compare_exchange_strong(value_type_ &expected, value_type_ desired,
                                 std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return weaker_().compare_exchange_strong(expected, desired, order);
    }
    bool compare_exchange_weak(value_type_ &expected, value_type_ desired,
                               std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return weaker_().compare_exchange_weak(expected, desired, order);
    }

    value_type_ fetch_add(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return weaker_().fetch_add(operand, order);
    }
    value_type_ fetch_sub(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return weaker_().fetch_sub(operand, order);
    }
    value_type_ fetch_and(value_type_ mask, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return weaker_().fetch_and(mask, order);
    }
    value_type_ fetch_or(value_type_ bits, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return weaker_().fetch_or(bits, order);
    }
    value_type_ fetch_xor(value_type_ bits, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return weaker_().fetch_xor(bits, order);
    }
    value_type_ fetch_max(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> || atomic_float<value_type_>
    {
        return weaker_().fetch_max(operand, order);
    }
    value_type_ fetch_min(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> || atomic_float<value_type_>
    {
        return weaker_().fetch_min(operand, order);
    }
    void max(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> || atomic_float<value_type_>
    {
        weaker_().max(operand, order);
    }
    void min(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> || atomic_float<value_type_>
    {
        weaker_().min(operand, order);
    }

    /** The conditional adds stay one @c cmpccxadd: RAO-INT spells no comparing form. */
    value_type_ fetch_add_if_at_most(value_type_ operand, value_type_ limit,
                                     std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> && (sizeof(value_type_) == 4 || sizeof(value_type_) == 8)
    {
        return weaker_().fetch_add_if_at_most(operand, limit, order);
    }
    value_type_ fetch_add_if_at_least(value_type_ operand, value_type_ floor,
                                      std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> && (sizeof(value_type_) == 4 || sizeof(value_type_) == 8)
    {
        return weaker_().fetch_add_if_at_least(operand, floor, order);
    }
    value_type_ fetch_sub_if_at_least(value_type_ operand, value_type_ floor,
                                      std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> && (sizeof(value_type_) == 4 || sizeof(value_type_) == 8)
    {
        return weaker_().fetch_sub_if_at_least(operand, floor, order);
    }

    void add(value_type_ operand, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_> && (sizeof(value_type_) == 4 || sizeof(value_type_) == 8)
    {
        if (order != std::memory_order_relaxed) return weaker_().add(operand, order);
        word_t *word = reinterpret_cast<word_t *>(word_);
        if constexpr (sizeof(value_type_) == 4) x86_aadd_u32(word, std::bit_cast<word_t>(operand));
        else x86_aadd_u64(word, std::bit_cast<word_t>(operand));
    }
    void sub(value_type_ operand, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_> && (sizeof(value_type_) == 4 || sizeof(value_type_) == 8)
    {
        add(std::bit_cast<value_type_>(static_cast<word_t>(word_t {0} - std::bit_cast<word_t>(operand))), order);
    }
    void set_bits(value_type_ bits, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_> && (sizeof(value_type_) == 4 || sizeof(value_type_) == 8)
    {
        if (order != std::memory_order_relaxed) return weaker_().set_bits(bits, order);
        word_t *word = reinterpret_cast<word_t *>(word_);
        if constexpr (sizeof(value_type_) == 4) x86_aor_u32(word, std::bit_cast<word_t>(bits));
        else x86_aor_u64(word, std::bit_cast<word_t>(bits));
    }
    void clear_bits(value_type_ bits, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_> && (sizeof(value_type_) == 4 || sizeof(value_type_) == 8)
    {
        if (order != std::memory_order_relaxed) return weaker_().clear_bits(bits, order);
        word_t *word = reinterpret_cast<word_t *>(word_);
        word_t const mask = static_cast<word_t>(~std::bit_cast<word_t>(bits));
        if constexpr (sizeof(value_type_) == 4) x86_aand_u32(word, mask);
        else x86_aand_u64(word, mask);
    }
    void flip_bits(value_type_ bits, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_> && (sizeof(value_type_) == 4 || sizeof(value_type_) == 8)
    {
        if (order != std::memory_order_relaxed) return weaker_().flip_bits(bits, order);
        word_t *word = reinterpret_cast<word_t *>(word_);
        if constexpr (sizeof(value_type_) == 4) x86_axor_u32(word, std::bit_cast<word_t>(bits));
        else x86_axor_u64(word, std::bit_cast<word_t>(bits));
    }
    bool fetch_test_and_set_bit(unsigned bit, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        return weaker_().fetch_test_and_set_bit(bit, order);
    }
    bool fetch_test_and_reset_bit(unsigned bit, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        return weaker_().fetch_test_and_reset_bit(bit, order);
    }

  private:
    /** The previous rung, built on demand: it holds this same pointer and nothing else. */
    x86_cmpccxadd_atomic_ref<value_type_> weaker_() const noexcept {
        return x86_cmpccxadd_atomic_ref<value_type_>(*word_);
    }

    value_type_ *word_;
};

/**
 *  @brief Read-only access through @ref x86_raoint_atomic_ref, which specializes no load.
 *  @sa standard_atomic_ref<value_type_ const>
 */
template <typename value_type_>
struct x86_raoint_atomic_ref<value_type_ const> {
    using value_t = value_type_ const;
    static constexpr capabilities_t capabilities_k = capability_x86_cmpccxadd_k | capability_x86_raoint_k;

    explicit x86_raoint_atomic_ref(value_type_ const &word) noexcept : portable_(word) {}

    /** A temporary would die before the reference does. */
    x86_raoint_atomic_ref(value_type_ const &&) = delete;

    value_type_ load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return portable_.load(order);
    }

  private:
    standard_atomic_ref<value_type_ const> portable_;
};

#pragma endregion x86 RAOINT

#endif // FORKUNION_TARGET_X86_RAOINT

#if FORKUNION_TARGET_ARM64_LSE

#pragma region Arm64 LSE

#if FORKUNION_HAS_INLINE_ASM_

/*  Loads: plain and acquire. */

inline std::uint8_t arm64_load_u8(std::uint8_t const *word) noexcept {
    std::uint8_t value;
    __asm__ __volatile__("ldrb %w0, [%1]" : "=r"(value) : "r"(word) : "memory");
    return value;
}
inline std::uint16_t arm64_load_u16(std::uint16_t const *word) noexcept {
    std::uint16_t value;
    __asm__ __volatile__("ldrh %w0, [%1]" : "=r"(value) : "r"(word) : "memory");
    return value;
}
inline std::uint32_t arm64_load_u32(std::uint32_t const *word) noexcept {
    std::uint32_t value;
    __asm__ __volatile__("ldr %w0, [%1]" : "=r"(value) : "r"(word) : "memory");
    return value;
}
inline std::uint64_t arm64_load_u64(std::uint64_t const *word) noexcept {
    std::uint64_t value;
    __asm__ __volatile__("ldr %x0, [%1]" : "=r"(value) : "r"(word) : "memory");
    return value;
}

inline std::uint8_t arm64_ldar_u8(std::uint8_t const *word) noexcept {
    std::uint8_t value;
    __asm__ __volatile__("ldarb %w0, [%1]" : "=r"(value) : "r"(word) : "memory");
    return value;
}
inline std::uint16_t arm64_ldar_u16(std::uint16_t const *word) noexcept {
    std::uint16_t value;
    __asm__ __volatile__("ldarh %w0, [%1]" : "=r"(value) : "r"(word) : "memory");
    return value;
}
inline std::uint32_t arm64_ldar_u32(std::uint32_t const *word) noexcept {
    std::uint32_t value;
    __asm__ __volatile__("ldar %w0, [%1]" : "=r"(value) : "r"(word) : "memory");
    return value;
}
inline std::uint64_t arm64_ldar_u64(std::uint64_t const *word) noexcept {
    std::uint64_t value;
    __asm__ __volatile__("ldar %x0, [%1]" : "=r"(value) : "r"(word) : "memory");
    return value;
}

/*  Stores: plain and release. */

inline void arm64_store_u8(std::uint8_t *word, std::uint8_t value) noexcept {
    __asm__ __volatile__("strb %w0, [%1]" : : "r"(value), "r"(word) : "memory");
}
inline void arm64_store_u16(std::uint16_t *word, std::uint16_t value) noexcept {
    __asm__ __volatile__("strh %w0, [%1]" : : "r"(value), "r"(word) : "memory");
}
inline void arm64_store_u32(std::uint32_t *word, std::uint32_t value) noexcept {
    __asm__ __volatile__("str %w0, [%1]" : : "r"(value), "r"(word) : "memory");
}
inline void arm64_store_u64(std::uint64_t *word, std::uint64_t value) noexcept {
    __asm__ __volatile__("str %x0, [%1]" : : "r"(value), "r"(word) : "memory");
}

inline void arm64_stlr_u8(std::uint8_t *word, std::uint8_t value) noexcept {
    __asm__ __volatile__("stlrb %w0, [%1]" : : "r"(value), "r"(word) : "memory");
}
inline void arm64_stlr_u16(std::uint16_t *word, std::uint16_t value) noexcept {
    __asm__ __volatile__("stlrh %w0, [%1]" : : "r"(value), "r"(word) : "memory");
}
inline void arm64_stlr_u32(std::uint32_t *word, std::uint32_t value) noexcept {
    __asm__ __volatile__("stlr %w0, [%1]" : : "r"(value), "r"(word) : "memory");
}
inline void arm64_stlr_u64(std::uint64_t *word, std::uint64_t value) noexcept {
    __asm__ __volatile__("stlr %x0, [%1]" : : "r"(value), "r"(word) : "memory");
}

/*  LSE read-modify-writes: @c ws is the operand, @c wt receives what the word held; the order picks
 *  the acquire/release flavor, and the switches fold away at every constant call site. */

inline std::uint8_t arm64_swp_u8(std::uint8_t *word, std::uint8_t desired, std::memory_order order) noexcept {
    std::uint8_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tswpb %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tswpab %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tswplb %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tswpalb %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::uint16_t arm64_swp_u16(std::uint16_t *word, std::uint16_t desired, std::memory_order order) noexcept {
    std::uint16_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tswph %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tswpah %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tswplh %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tswpalh %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::uint32_t arm64_swp_u32(std::uint32_t *word, std::uint32_t desired, std::memory_order order) noexcept {
    std::uint32_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tswp %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tswpa %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tswpl %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tswpal %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::uint64_t arm64_swp_u64(std::uint64_t *word, std::uint64_t desired, std::memory_order order) noexcept {
    std::uint64_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tswp %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tswpa %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tswpl %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tswpal %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::uint8_t arm64_ldadd_u8(std::uint8_t *word, std::uint8_t operand, std::memory_order order) noexcept {
    std::uint8_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tldaddb %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tldaddab %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tldaddlb %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tldaddalb %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::uint16_t arm64_ldadd_u16(std::uint16_t *word, std::uint16_t operand, std::memory_order order) noexcept {
    std::uint16_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tldaddh %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tldaddah %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tldaddlh %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tldaddalh %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::uint32_t arm64_ldadd_u32(std::uint32_t *word, std::uint32_t operand, std::memory_order order) noexcept {
    std::uint32_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tldadd %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tldadda %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tldaddl %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tldaddal %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::uint64_t arm64_ldadd_u64(std::uint64_t *word, std::uint64_t operand, std::memory_order order) noexcept {
    std::uint64_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tldadd %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tldadda %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tldaddl %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tldaddal %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

/** @c ldclr clears the operand's bits: a word's <tt>fetch_and(mask)</tt> is <tt>ldclr ~mask</tt>. */
inline std::uint8_t arm64_ldclr_u8(std::uint8_t *word, std::uint8_t bits, std::memory_order order) noexcept {
    std::uint8_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tldclrb %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(bits), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tldclrab %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(bits), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tldclrlb %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(bits), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tldclralb %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(bits), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::uint16_t arm64_ldclr_u16(std::uint16_t *word, std::uint16_t bits, std::memory_order order) noexcept {
    std::uint16_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tldclrh %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(bits), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tldclrah %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(bits), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tldclrlh %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(bits), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tldclralh %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(bits), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::uint64_t arm64_ldclr_u64(std::uint64_t *word, std::uint64_t bits, std::memory_order order) noexcept {
    std::uint64_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tldclr %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(bits), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tldclra %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(bits), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tldclrl %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(bits), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tldclral %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(bits), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::uint32_t arm64_ldclr_u32(std::uint32_t *word, std::uint32_t operand, std::memory_order order) noexcept {
    std::uint32_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tldclr %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tldclra %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tldclrl %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tldclral %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::uint8_t arm64_ldset_u8(std::uint8_t *word, std::uint8_t operand, std::memory_order order) noexcept {
    std::uint8_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tldsetb %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tldsetab %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tldsetlb %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tldsetalb %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::uint16_t arm64_ldset_u16(std::uint16_t *word, std::uint16_t operand, std::memory_order order) noexcept {
    std::uint16_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tldseth %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tldsetah %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tldsetlh %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tldsetalh %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::uint32_t arm64_ldset_u32(std::uint32_t *word, std::uint32_t operand, std::memory_order order) noexcept {
    std::uint32_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tldset %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tldseta %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tldsetl %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tldsetal %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::uint64_t arm64_ldset_u64(std::uint64_t *word, std::uint64_t operand, std::memory_order order) noexcept {
    std::uint64_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tldset %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tldseta %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tldsetl %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tldsetal %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::uint8_t arm64_ldeor_u8(std::uint8_t *word, std::uint8_t operand, std::memory_order order) noexcept {
    std::uint8_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tldeorb %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tldeorab %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tldeorlb %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tldeoralb %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::uint16_t arm64_ldeor_u16(std::uint16_t *word, std::uint16_t operand, std::memory_order order) noexcept {
    std::uint16_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tldeorh %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tldeorah %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tldeorlh %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tldeoralh %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::uint32_t arm64_ldeor_u32(std::uint32_t *word, std::uint32_t operand, std::memory_order order) noexcept {
    std::uint32_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tldeor %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tldeora %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tldeorl %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tldeoral %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::uint64_t arm64_ldeor_u64(std::uint64_t *word, std::uint64_t operand, std::memory_order order) noexcept {
    std::uint64_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tldeor %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tldeora %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tldeorl %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tldeoral %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

/*  No-return forms: nothing comes back, so only the plain and release flavors exist. */

inline void arm64_stadd_u8(std::uint8_t *word, std::uint8_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tstaddb %w0, [%1]" : : "r"(operand), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tstaddlb %w0, [%1]" : : "r"(operand), "r"(word) : "memory");
}
inline void arm64_stadd_u16(std::uint16_t *word, std::uint16_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tstaddh %w0, [%1]" : : "r"(operand), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tstaddlh %w0, [%1]" : : "r"(operand), "r"(word) : "memory");
}
inline void arm64_stadd_u32(std::uint32_t *word, std::uint32_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tstadd %w0, [%1]" : : "r"(operand), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tstaddl %w0, [%1]" : : "r"(operand), "r"(word) : "memory");
}
inline void arm64_stadd_u64(std::uint64_t *word, std::uint64_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tstadd %x0, [%1]" : : "r"(operand), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tstaddl %x0, [%1]" : : "r"(operand), "r"(word) : "memory");
}
inline void arm64_stclr_u8(std::uint8_t *word, std::uint8_t bits, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tstclrb %w0, [%1]" : : "r"(bits), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tstclrlb %w0, [%1]" : : "r"(bits), "r"(word) : "memory");
}
inline void arm64_stclr_u16(std::uint16_t *word, std::uint16_t bits, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tstclrh %w0, [%1]" : : "r"(bits), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tstclrlh %w0, [%1]" : : "r"(bits), "r"(word) : "memory");
}
inline void arm64_stclr_u64(std::uint64_t *word, std::uint64_t bits, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tstclr %x0, [%1]" : : "r"(bits), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tstclrl %x0, [%1]" : : "r"(bits), "r"(word) : "memory");
}
inline void arm64_stset_u8(std::uint8_t *word, std::uint8_t bits, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tstsetb %w0, [%1]" : : "r"(bits), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tstsetlb %w0, [%1]" : : "r"(bits), "r"(word) : "memory");
}
inline void arm64_stset_u16(std::uint16_t *word, std::uint16_t bits, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tstseth %w0, [%1]" : : "r"(bits), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tstsetlh %w0, [%1]" : : "r"(bits), "r"(word) : "memory");
}
inline void arm64_stset_u64(std::uint64_t *word, std::uint64_t bits, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tstset %x0, [%1]" : : "r"(bits), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tstsetl %x0, [%1]" : : "r"(bits), "r"(word) : "memory");
}

inline void arm64_stclr_u32(std::uint32_t *word, std::uint32_t bits, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tstclr %w0, [%1]" : : "r"(bits), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tstclrl %w0, [%1]" : : "r"(bits), "r"(word) : "memory");
}
inline void arm64_stset_u32(std::uint32_t *word, std::uint32_t bits, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tstset %w0, [%1]" : : "r"(bits), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tstsetl %w0, [%1]" : : "r"(bits), "r"(word) : "memory");
}
inline void arm64_steor_u8(std::uint8_t *word, std::uint8_t bits, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tsteorb %w0, [%1]" : : "r"(bits), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tsteorlb %w0, [%1]" : : "r"(bits), "r"(word) : "memory");
}
inline void arm64_steor_u16(std::uint16_t *word, std::uint16_t bits, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tsteorh %w0, [%1]" : : "r"(bits), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tsteorlh %w0, [%1]" : : "r"(bits), "r"(word) : "memory");
}
inline void arm64_steor_u32(std::uint32_t *word, std::uint32_t bits, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tsteor %w0, [%1]" : : "r"(bits), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tsteorl %w0, [%1]" : : "r"(bits), "r"(word) : "memory");
}
inline void arm64_steor_u64(std::uint64_t *word, std::uint64_t bits, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tsteor %x0, [%1]" : : "r"(bits), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tsteorl %x0, [%1]" : : "r"(bits), "r"(word) : "memory");
}

/*  LSE maxima & minima - signed & unsigned differ by instruction, the width by register and the
 *  byte & halfword suffix. */

inline std::uint8_t arm64_ldumax_u8(std::uint8_t *word, std::uint8_t operand, std::memory_order order) noexcept {
    std::uint8_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tldumaxb %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tldumaxab %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tldumaxlb %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tldumaxalb %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::uint16_t arm64_ldumax_u16(std::uint16_t *word, std::uint16_t operand, std::memory_order order) noexcept {
    std::uint16_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tldumaxh %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tldumaxah %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tldumaxlh %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tldumaxalh %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::uint32_t arm64_ldumax_u32(std::uint32_t *word, std::uint32_t operand, std::memory_order order) noexcept {
    std::uint32_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tldumax %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tldumaxa %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tldumaxl %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tldumaxal %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::uint64_t arm64_ldumax_u64(std::uint64_t *word, std::uint64_t operand, std::memory_order order) noexcept {
    std::uint64_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tldumax %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tldumaxa %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tldumaxl %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tldumaxal %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::int8_t arm64_ldsmax_i8(std::int8_t *word, std::int8_t operand, std::memory_order order) noexcept {
    std::int8_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tldsmaxb %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tldsmaxab %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tldsmaxlb %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tldsmaxalb %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::int16_t arm64_ldsmax_i16(std::int16_t *word, std::int16_t operand, std::memory_order order) noexcept {
    std::int16_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tldsmaxh %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tldsmaxah %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tldsmaxlh %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tldsmaxalh %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::int32_t arm64_ldsmax_i32(std::int32_t *word, std::int32_t operand, std::memory_order order) noexcept {
    std::int32_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tldsmax %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tldsmaxa %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tldsmaxl %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tldsmaxal %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::int64_t arm64_ldsmax_i64(std::int64_t *word, std::int64_t operand, std::memory_order order) noexcept {
    std::int64_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tldsmax %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tldsmaxa %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tldsmaxl %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tldsmaxal %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::uint8_t arm64_ldumin_u8(std::uint8_t *word, std::uint8_t operand, std::memory_order order) noexcept {
    std::uint8_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tlduminb %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tlduminab %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tlduminlb %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tlduminalb %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::uint16_t arm64_ldumin_u16(std::uint16_t *word, std::uint16_t operand, std::memory_order order) noexcept {
    std::uint16_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tlduminh %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tlduminah %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tlduminlh %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tlduminalh %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::uint32_t arm64_ldumin_u32(std::uint32_t *word, std::uint32_t operand, std::memory_order order) noexcept {
    std::uint32_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tldumin %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tldumina %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tlduminl %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tlduminal %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::uint64_t arm64_ldumin_u64(std::uint64_t *word, std::uint64_t operand, std::memory_order order) noexcept {
    std::uint64_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tldumin %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tldumina %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tlduminl %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tlduminal %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::int8_t arm64_ldsmin_i8(std::int8_t *word, std::int8_t operand, std::memory_order order) noexcept {
    std::int8_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tldsminb %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tldsminab %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tldsminlb %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tldsminalb %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::int16_t arm64_ldsmin_i16(std::int16_t *word, std::int16_t operand, std::memory_order order) noexcept {
    std::int16_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tldsminh %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tldsminah %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tldsminlh %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tldsminalh %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::int32_t arm64_ldsmin_i32(std::int32_t *word, std::int32_t operand, std::memory_order order) noexcept {
    std::int32_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tldsmin %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tldsmina %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tldsminl %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tldsminal %w1, %w0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

inline std::int64_t arm64_ldsmin_i64(std::int64_t *word, std::int64_t operand, std::memory_order order) noexcept {
    std::int64_t observed;
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tldsmin %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tldsmina %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tldsminl %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tldsminal %x1, %x0, [%2]"
                             : "=r"(observed)
                             : "r"(operand), "r"(word)
                             : "memory");
        break;
    }
    return observed;
}

/*  No-return maxima & minima: @c stsmax & kin, posted like @c stadd. */

inline void arm64_stumax_u8(std::uint8_t *word, std::uint8_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tstumaxb %w0, [%1]" : : "r"(operand), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tstumaxlb %w0, [%1]" : : "r"(operand), "r"(word) : "memory");
}
inline void arm64_stumax_u16(std::uint16_t *word, std::uint16_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tstumaxh %w0, [%1]" : : "r"(operand), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tstumaxlh %w0, [%1]" : : "r"(operand), "r"(word) : "memory");
}
inline void arm64_stumax_u32(std::uint32_t *word, std::uint32_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tstumax %w0, [%1]" : : "r"(operand), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tstumaxl %w0, [%1]" : : "r"(operand), "r"(word) : "memory");
}
inline void arm64_stumax_u64(std::uint64_t *word, std::uint64_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tstumax %x0, [%1]" : : "r"(operand), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tstumaxl %x0, [%1]" : : "r"(operand), "r"(word) : "memory");
}
inline void arm64_stsmax_i8(std::int8_t *word, std::int8_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tstsmaxb %w0, [%1]" : : "r"(operand), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tstsmaxlb %w0, [%1]" : : "r"(operand), "r"(word) : "memory");
}
inline void arm64_stsmax_i16(std::int16_t *word, std::int16_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tstsmaxh %w0, [%1]" : : "r"(operand), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tstsmaxlh %w0, [%1]" : : "r"(operand), "r"(word) : "memory");
}
inline void arm64_stsmax_i32(std::int32_t *word, std::int32_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tstsmax %w0, [%1]" : : "r"(operand), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tstsmaxl %w0, [%1]" : : "r"(operand), "r"(word) : "memory");
}
inline void arm64_stsmax_i64(std::int64_t *word, std::int64_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tstsmax %x0, [%1]" : : "r"(operand), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tstsmaxl %x0, [%1]" : : "r"(operand), "r"(word) : "memory");
}
inline void arm64_stumin_u8(std::uint8_t *word, std::uint8_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tstuminb %w0, [%1]" : : "r"(operand), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tstuminlb %w0, [%1]" : : "r"(operand), "r"(word) : "memory");
}
inline void arm64_stumin_u16(std::uint16_t *word, std::uint16_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tstuminh %w0, [%1]" : : "r"(operand), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tstuminlh %w0, [%1]" : : "r"(operand), "r"(word) : "memory");
}
inline void arm64_stumin_u32(std::uint32_t *word, std::uint32_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tstumin %w0, [%1]" : : "r"(operand), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tstuminl %w0, [%1]" : : "r"(operand), "r"(word) : "memory");
}
inline void arm64_stumin_u64(std::uint64_t *word, std::uint64_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tstumin %x0, [%1]" : : "r"(operand), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tstuminl %x0, [%1]" : : "r"(operand), "r"(word) : "memory");
}
inline void arm64_stsmin_i8(std::int8_t *word, std::int8_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tstsminb %w0, [%1]" : : "r"(operand), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tstsminlb %w0, [%1]" : : "r"(operand), "r"(word) : "memory");
}
inline void arm64_stsmin_i16(std::int16_t *word, std::int16_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tstsminh %w0, [%1]" : : "r"(operand), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tstsminlh %w0, [%1]" : : "r"(operand), "r"(word) : "memory");
}
inline void arm64_stsmin_i32(std::int32_t *word, std::int32_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tstsmin %w0, [%1]" : : "r"(operand), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tstsminl %w0, [%1]" : : "r"(operand), "r"(word) : "memory");
}
inline void arm64_stsmin_i64(std::int64_t *word, std::int64_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tstsmin %x0, [%1]" : : "r"(operand), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tstsminl %x0, [%1]" : : "r"(operand), "r"(word) : "memory");
}

/** @c cas compares with @p expected and returns what the word held; equality means it swapped. */
inline std::uint8_t arm64_cas_u8(std::uint8_t *word, std::uint8_t expected, std::uint8_t desired,
                                 std::memory_order order) noexcept {
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tcasb %w0, %w1, [%2]"
                             : "+r"(expected)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tcasab %w0, %w1, [%2]"
                             : "+r"(expected)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tcaslb %w0, %w1, [%2]"
                             : "+r"(expected)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tcasalb %w0, %w1, [%2]"
                             : "+r"(expected)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    }
    return expected;
}

inline std::uint16_t arm64_cas_u16(std::uint16_t *word, std::uint16_t expected, std::uint16_t desired,
                                   std::memory_order order) noexcept {
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tcash %w0, %w1, [%2]"
                             : "+r"(expected)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tcasah %w0, %w1, [%2]"
                             : "+r"(expected)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tcaslh %w0, %w1, [%2]"
                             : "+r"(expected)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tcasalh %w0, %w1, [%2]"
                             : "+r"(expected)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    }
    return expected;
}

inline std::uint32_t arm64_cas_u32(std::uint32_t *word, std::uint32_t expected, std::uint32_t desired,
                                   std::memory_order order) noexcept {
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tcas %w0, %w1, [%2]"
                             : "+r"(expected)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tcasa %w0, %w1, [%2]"
                             : "+r"(expected)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tcasl %w0, %w1, [%2]"
                             : "+r"(expected)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tcasal %w0, %w1, [%2]"
                             : "+r"(expected)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    }
    return expected;
}

inline std::uint64_t arm64_cas_u64(std::uint64_t *word, std::uint64_t expected, std::uint64_t desired,
                                   std::memory_order order) noexcept {
    switch (order) {
    case std::memory_order_relaxed:
        __asm__ __volatile__(".arch_extension lse\n\tcas %x0, %x1, [%2]"
                             : "+r"(expected)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    case std::memory_order_consume:
    case std::memory_order_acquire:
        __asm__ __volatile__(".arch_extension lse\n\tcasa %x0, %x1, [%2]"
                             : "+r"(expected)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    case std::memory_order_release:
        __asm__ __volatile__(".arch_extension lse\n\tcasl %x0, %x1, [%2]"
                             : "+r"(expected)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    default:
        __asm__ __volatile__(".arch_extension lse\n\tcasal %x0, %x1, [%2]"
                             : "+r"(expected)
                             : "r"(desired), "r"(word)
                             : "memory");
        break;
    }
    return expected;
}

#endif // FORKUNION_HAS_INLINE_ASM_

/*  Loads: plain and acquire. `__iso_volatile_load*` is the plain load no optimizer may fold away,
 *  spelled over the signed widths; `__ldar*` takes the unsigned ones directly. */
#if FORKUNION_HAS_ARM64_ATOMIC_INTRINSICS_
inline std::uint8_t arm64_load_u8(std::uint8_t const *word) noexcept {
    return static_cast<std::uint8_t>(__iso_volatile_load8(reinterpret_cast<char const volatile *>(word)));
}
inline std::uint16_t arm64_load_u16(std::uint16_t const *word) noexcept {
    return static_cast<std::uint16_t>(__iso_volatile_load16(reinterpret_cast<short const volatile *>(word)));
}
inline std::uint32_t arm64_load_u32(std::uint32_t const *word) noexcept {
    return static_cast<std::uint32_t>(__iso_volatile_load32(reinterpret_cast<int const volatile *>(word)));
}
inline std::uint64_t arm64_load_u64(std::uint64_t const *word) noexcept {
    return static_cast<std::uint64_t>(__iso_volatile_load64(reinterpret_cast<__int64 const volatile *>(word)));
}

inline std::uint8_t arm64_ldar_u8(std::uint8_t const *word) noexcept { return __ldar8(word); }
inline std::uint16_t arm64_ldar_u16(std::uint16_t const *word) noexcept { return __ldar16(word); }
inline std::uint32_t arm64_ldar_u32(std::uint32_t const *word) noexcept { return __ldar32(word); }
inline std::uint64_t arm64_ldar_u64(std::uint64_t const *word) noexcept { return __ldar64(word); }

/*  Stores: plain and release. */

inline void arm64_store_u8(std::uint8_t *word, std::uint8_t value) noexcept {
    __iso_volatile_store8(reinterpret_cast<char volatile *>(word), static_cast<char>(value));
}
inline void arm64_store_u16(std::uint16_t *word, std::uint16_t value) noexcept {
    __iso_volatile_store16(reinterpret_cast<short volatile *>(word), static_cast<short>(value));
}
inline void arm64_store_u32(std::uint32_t *word, std::uint32_t value) noexcept {
    __iso_volatile_store32(reinterpret_cast<int volatile *>(word), static_cast<int>(value));
}
inline void arm64_store_u64(std::uint64_t *word, std::uint64_t value) noexcept {
    __iso_volatile_store64(reinterpret_cast<__int64 volatile *>(word), static_cast<__int64>(value));
}

inline void arm64_stlr_u8(std::uint8_t *word, std::uint8_t value) noexcept { __stlr8(word, value); }
inline void arm64_stlr_u16(std::uint16_t *word, std::uint16_t value) noexcept { __stlr16(word, value); }
inline void arm64_stlr_u32(std::uint32_t *word, std::uint32_t value) noexcept { __stlr32(word, value); }
inline void arm64_stlr_u64(std::uint64_t *word, std::uint64_t value) noexcept { __stlr64(word, value); }

/*  LSE exchanges: the intrinsic's name carries the acquire/release flavor the mnemonic's suffix
 *  does, so the order picks the spelling and the switches fold away at every constant call site. */

inline std::uint8_t arm64_swp_u8(std::uint8_t *word, std::uint8_t desired, std::memory_order order) noexcept {
    switch (order) {
    case std::memory_order_relaxed: return __swp8(word, desired);
    case std::memory_order_consume:
    case std::memory_order_acquire: return __swpa8(word, desired);
    case std::memory_order_release: return __swpl8(word, desired);
    default: return __swpal8(word, desired);
    }
}
inline std::uint16_t arm64_swp_u16(std::uint16_t *word, std::uint16_t desired, std::memory_order order) noexcept {
    switch (order) {
    case std::memory_order_relaxed: return __swp16(word, desired);
    case std::memory_order_consume:
    case std::memory_order_acquire: return __swpa16(word, desired);
    case std::memory_order_release: return __swpl16(word, desired);
    default: return __swpal16(word, desired);
    }
}
inline std::uint32_t arm64_swp_u32(std::uint32_t *word, std::uint32_t desired, std::memory_order order) noexcept {
    switch (order) {
    case std::memory_order_relaxed: return __swp32(word, desired);
    case std::memory_order_consume:
    case std::memory_order_acquire: return __swpa32(word, desired);
    case std::memory_order_release: return __swpl32(word, desired);
    default: return __swpal32(word, desired);
    }
}
inline std::uint64_t arm64_swp_u64(std::uint64_t *word, std::uint64_t desired, std::memory_order order) noexcept {
    switch (order) {
    case std::memory_order_relaxed: return __swp64(word, desired);
    case std::memory_order_consume:
    case std::memory_order_acquire: return __swpa64(word, desired);
    case std::memory_order_release: return __swpl64(word, desired);
    default: return __swpal64(word, desired);
    }
}

/** @c cas compares with @p expected and returns what the word held; equality means it swapped. */
inline std::uint8_t arm64_cas_u8(std::uint8_t *word, std::uint8_t expected, std::uint8_t desired,
                                 std::memory_order order) noexcept {
    switch (order) {
    case std::memory_order_relaxed: return __cas8(word, expected, desired);
    case std::memory_order_consume:
    case std::memory_order_acquire: return __casa8(word, expected, desired);
    case std::memory_order_release: return __casl8(word, expected, desired);
    default: return __casal8(word, expected, desired);
    }
}
inline std::uint16_t arm64_cas_u16(std::uint16_t *word, std::uint16_t expected, std::uint16_t desired,
                                   std::memory_order order) noexcept {
    switch (order) {
    case std::memory_order_relaxed: return __cas16(word, expected, desired);
    case std::memory_order_consume:
    case std::memory_order_acquire: return __casa16(word, expected, desired);
    case std::memory_order_release: return __casl16(word, expected, desired);
    default: return __casal16(word, expected, desired);
    }
}
inline std::uint32_t arm64_cas_u32(std::uint32_t *word, std::uint32_t expected, std::uint32_t desired,
                                   std::memory_order order) noexcept {
    switch (order) {
    case std::memory_order_relaxed: return __cas32(word, expected, desired);
    case std::memory_order_consume:
    case std::memory_order_acquire: return __casa32(word, expected, desired);
    case std::memory_order_release: return __casl32(word, expected, desired);
    default: return __casal32(word, expected, desired);
    }
}
inline std::uint64_t arm64_cas_u64(std::uint64_t *word, std::uint64_t expected, std::uint64_t desired,
                                   std::memory_order order) noexcept {
    switch (order) {
    case std::memory_order_relaxed: return __cas64(word, expected, desired);
    case std::memory_order_consume:
    case std::memory_order_acquire: return __casa64(word, expected, desired);
    case std::memory_order_release: return __casl64(word, expected, desired);
    default: return __casal64(word, expected, desired);
    }
}

/*  LSE arithmetic: @c ldadd, @c ldclr, @c ldset and @c ldeor have no intrinsic of their own, so the
 *  `_Interlocked*` family carries them - one instruction under `/arch:armv8.1`, and a call into the
 *  CRT without it, which @c FORKUNION_HAS_ARM64_ATOMIC_INTRINSICS_ has already refused. */

inline std::uint8_t arm64_ldadd_u8(std::uint8_t *word, std::uint8_t operand, std::memory_order order) noexcept {
    char volatile *target = reinterpret_cast<char volatile *>(word);
    char const addend = static_cast<char>(operand);
    switch (order) {
    case std::memory_order_relaxed: return static_cast<std::uint8_t>(_InterlockedExchangeAdd8_nf(target, addend));
    case std::memory_order_consume:
    case std::memory_order_acquire: return static_cast<std::uint8_t>(_InterlockedExchangeAdd8_acq(target, addend));
    case std::memory_order_release: return static_cast<std::uint8_t>(_InterlockedExchangeAdd8_rel(target, addend));
    default: return static_cast<std::uint8_t>(_InterlockedExchangeAdd8(target, addend));
    }
}
inline std::uint16_t arm64_ldadd_u16(std::uint16_t *word, std::uint16_t operand, std::memory_order order) noexcept {
    short volatile *target = reinterpret_cast<short volatile *>(word);
    short const addend = static_cast<short>(operand);
    switch (order) {
    case std::memory_order_relaxed: return static_cast<std::uint16_t>(_InterlockedExchangeAdd16_nf(target, addend));
    case std::memory_order_consume:
    case std::memory_order_acquire: return static_cast<std::uint16_t>(_InterlockedExchangeAdd16_acq(target, addend));
    case std::memory_order_release: return static_cast<std::uint16_t>(_InterlockedExchangeAdd16_rel(target, addend));
    default: return static_cast<std::uint16_t>(_InterlockedExchangeAdd16(target, addend));
    }
}
inline std::uint32_t arm64_ldadd_u32(std::uint32_t *word, std::uint32_t operand, std::memory_order order) noexcept {
    long volatile *target = reinterpret_cast<long volatile *>(word);
    long const addend = static_cast<long>(operand);
    switch (order) {
    case std::memory_order_relaxed: return static_cast<std::uint32_t>(_InterlockedExchangeAdd_nf(target, addend));
    case std::memory_order_consume:
    case std::memory_order_acquire: return static_cast<std::uint32_t>(_InterlockedExchangeAdd_acq(target, addend));
    case std::memory_order_release: return static_cast<std::uint32_t>(_InterlockedExchangeAdd_rel(target, addend));
    default: return static_cast<std::uint32_t>(_InterlockedExchangeAdd(target, addend));
    }
}
inline std::uint64_t arm64_ldadd_u64(std::uint64_t *word, std::uint64_t operand, std::memory_order order) noexcept {
    __int64 volatile *target = reinterpret_cast<__int64 volatile *>(word);
    __int64 const addend = static_cast<__int64>(operand);
    switch (order) {
    case std::memory_order_relaxed: return static_cast<std::uint64_t>(_InterlockedExchangeAdd64_nf(target, addend));
    case std::memory_order_consume:
    case std::memory_order_acquire: return static_cast<std::uint64_t>(_InterlockedExchangeAdd64_acq(target, addend));
    case std::memory_order_release: return static_cast<std::uint64_t>(_InterlockedExchangeAdd64_rel(target, addend));
    default: return static_cast<std::uint64_t>(_InterlockedExchangeAdd64(target, addend));
    }
}

/** @c ldclr clears bits; @c _InterlockedAnd keeps the complement, so the mask flips. */
inline std::uint8_t arm64_ldclr_u8(std::uint8_t *word, std::uint8_t bits, std::memory_order order) noexcept {
    char volatile *target = reinterpret_cast<char volatile *>(word);
    char const kept = static_cast<char>(~bits);
    switch (order) {
    case std::memory_order_relaxed: return static_cast<std::uint8_t>(_InterlockedAnd8_nf(target, kept));
    case std::memory_order_consume:
    case std::memory_order_acquire: return static_cast<std::uint8_t>(_InterlockedAnd8_acq(target, kept));
    case std::memory_order_release: return static_cast<std::uint8_t>(_InterlockedAnd8_rel(target, kept));
    default: return static_cast<std::uint8_t>(_InterlockedAnd8(target, kept));
    }
}
inline std::uint16_t arm64_ldclr_u16(std::uint16_t *word, std::uint16_t bits, std::memory_order order) noexcept {
    short volatile *target = reinterpret_cast<short volatile *>(word);
    short const kept = static_cast<short>(~bits);
    switch (order) {
    case std::memory_order_relaxed: return static_cast<std::uint16_t>(_InterlockedAnd16_nf(target, kept));
    case std::memory_order_consume:
    case std::memory_order_acquire: return static_cast<std::uint16_t>(_InterlockedAnd16_acq(target, kept));
    case std::memory_order_release: return static_cast<std::uint16_t>(_InterlockedAnd16_rel(target, kept));
    default: return static_cast<std::uint16_t>(_InterlockedAnd16(target, kept));
    }
}
inline std::uint32_t arm64_ldclr_u32(std::uint32_t *word, std::uint32_t bits, std::memory_order order) noexcept {
    long volatile *target = reinterpret_cast<long volatile *>(word);
    long const kept = static_cast<long>(~bits);
    switch (order) {
    case std::memory_order_relaxed: return static_cast<std::uint32_t>(_InterlockedAnd_nf(target, kept));
    case std::memory_order_consume:
    case std::memory_order_acquire: return static_cast<std::uint32_t>(_InterlockedAnd_acq(target, kept));
    case std::memory_order_release: return static_cast<std::uint32_t>(_InterlockedAnd_rel(target, kept));
    default: return static_cast<std::uint32_t>(_InterlockedAnd(target, kept));
    }
}
inline std::uint64_t arm64_ldclr_u64(std::uint64_t *word, std::uint64_t bits, std::memory_order order) noexcept {
    __int64 volatile *target = reinterpret_cast<__int64 volatile *>(word);
    __int64 const kept = static_cast<__int64>(~bits);
    switch (order) {
    case std::memory_order_relaxed: return static_cast<std::uint64_t>(_InterlockedAnd64_nf(target, kept));
    case std::memory_order_consume:
    case std::memory_order_acquire: return static_cast<std::uint64_t>(_InterlockedAnd64_acq(target, kept));
    case std::memory_order_release: return static_cast<std::uint64_t>(_InterlockedAnd64_rel(target, kept));
    default: return static_cast<std::uint64_t>(_InterlockedAnd64(target, kept));
    }
}

inline std::uint8_t arm64_ldset_u8(std::uint8_t *word, std::uint8_t operand, std::memory_order order) noexcept {
    char volatile *target = reinterpret_cast<char volatile *>(word);
    char const bits = static_cast<char>(operand);
    switch (order) {
    case std::memory_order_relaxed: return static_cast<std::uint8_t>(_InterlockedOr8_nf(target, bits));
    case std::memory_order_consume:
    case std::memory_order_acquire: return static_cast<std::uint8_t>(_InterlockedOr8_acq(target, bits));
    case std::memory_order_release: return static_cast<std::uint8_t>(_InterlockedOr8_rel(target, bits));
    default: return static_cast<std::uint8_t>(_InterlockedOr8(target, bits));
    }
}
inline std::uint16_t arm64_ldset_u16(std::uint16_t *word, std::uint16_t operand, std::memory_order order) noexcept {
    short volatile *target = reinterpret_cast<short volatile *>(word);
    short const bits = static_cast<short>(operand);
    switch (order) {
    case std::memory_order_relaxed: return static_cast<std::uint16_t>(_InterlockedOr16_nf(target, bits));
    case std::memory_order_consume:
    case std::memory_order_acquire: return static_cast<std::uint16_t>(_InterlockedOr16_acq(target, bits));
    case std::memory_order_release: return static_cast<std::uint16_t>(_InterlockedOr16_rel(target, bits));
    default: return static_cast<std::uint16_t>(_InterlockedOr16(target, bits));
    }
}
inline std::uint32_t arm64_ldset_u32(std::uint32_t *word, std::uint32_t operand, std::memory_order order) noexcept {
    long volatile *target = reinterpret_cast<long volatile *>(word);
    long const bits = static_cast<long>(operand);
    switch (order) {
    case std::memory_order_relaxed: return static_cast<std::uint32_t>(_InterlockedOr_nf(target, bits));
    case std::memory_order_consume:
    case std::memory_order_acquire: return static_cast<std::uint32_t>(_InterlockedOr_acq(target, bits));
    case std::memory_order_release: return static_cast<std::uint32_t>(_InterlockedOr_rel(target, bits));
    default: return static_cast<std::uint32_t>(_InterlockedOr(target, bits));
    }
}
inline std::uint64_t arm64_ldset_u64(std::uint64_t *word, std::uint64_t operand, std::memory_order order) noexcept {
    __int64 volatile *target = reinterpret_cast<__int64 volatile *>(word);
    __int64 const bits = static_cast<__int64>(operand);
    switch (order) {
    case std::memory_order_relaxed: return static_cast<std::uint64_t>(_InterlockedOr64_nf(target, bits));
    case std::memory_order_consume:
    case std::memory_order_acquire: return static_cast<std::uint64_t>(_InterlockedOr64_acq(target, bits));
    case std::memory_order_release: return static_cast<std::uint64_t>(_InterlockedOr64_rel(target, bits));
    default: return static_cast<std::uint64_t>(_InterlockedOr64(target, bits));
    }
}

inline std::uint8_t arm64_ldeor_u8(std::uint8_t *word, std::uint8_t operand, std::memory_order order) noexcept {
    char volatile *target = reinterpret_cast<char volatile *>(word);
    char const bits = static_cast<char>(operand);
    switch (order) {
    case std::memory_order_relaxed: return static_cast<std::uint8_t>(_InterlockedXor8_nf(target, bits));
    case std::memory_order_consume:
    case std::memory_order_acquire: return static_cast<std::uint8_t>(_InterlockedXor8_acq(target, bits));
    case std::memory_order_release: return static_cast<std::uint8_t>(_InterlockedXor8_rel(target, bits));
    default: return static_cast<std::uint8_t>(_InterlockedXor8(target, bits));
    }
}
inline std::uint16_t arm64_ldeor_u16(std::uint16_t *word, std::uint16_t operand, std::memory_order order) noexcept {
    short volatile *target = reinterpret_cast<short volatile *>(word);
    short const bits = static_cast<short>(operand);
    switch (order) {
    case std::memory_order_relaxed: return static_cast<std::uint16_t>(_InterlockedXor16_nf(target, bits));
    case std::memory_order_consume:
    case std::memory_order_acquire: return static_cast<std::uint16_t>(_InterlockedXor16_acq(target, bits));
    case std::memory_order_release: return static_cast<std::uint16_t>(_InterlockedXor16_rel(target, bits));
    default: return static_cast<std::uint16_t>(_InterlockedXor16(target, bits));
    }
}
inline std::uint32_t arm64_ldeor_u32(std::uint32_t *word, std::uint32_t operand, std::memory_order order) noexcept {
    long volatile *target = reinterpret_cast<long volatile *>(word);
    long const bits = static_cast<long>(operand);
    switch (order) {
    case std::memory_order_relaxed: return static_cast<std::uint32_t>(_InterlockedXor_nf(target, bits));
    case std::memory_order_consume:
    case std::memory_order_acquire: return static_cast<std::uint32_t>(_InterlockedXor_acq(target, bits));
    case std::memory_order_release: return static_cast<std::uint32_t>(_InterlockedXor_rel(target, bits));
    default: return static_cast<std::uint32_t>(_InterlockedXor(target, bits));
    }
}
inline std::uint64_t arm64_ldeor_u64(std::uint64_t *word, std::uint64_t operand, std::memory_order order) noexcept {
    __int64 volatile *target = reinterpret_cast<__int64 volatile *>(word);
    __int64 const bits = static_cast<__int64>(operand);
    switch (order) {
    case std::memory_order_relaxed: return static_cast<std::uint64_t>(_InterlockedXor64_nf(target, bits));
    case std::memory_order_consume:
    case std::memory_order_acquire: return static_cast<std::uint64_t>(_InterlockedXor64_acq(target, bits));
    case std::memory_order_release: return static_cast<std::uint64_t>(_InterlockedXor64_rel(target, bits));
    default: return static_cast<std::uint64_t>(_InterlockedXor64(target, bits));
    }
}

/*  No-return forms: nothing spells @c stadd, @c stclr, @c stset or @c steor, so the returning
 *  instruction runs and its answer is dropped, a round trip the posted form skips. */

inline void arm64_stadd_u8(std::uint8_t *word, std::uint8_t operand, std::memory_order order) noexcept {
    [[maybe_unused]] std::uint8_t const observed = arm64_ldadd_u8(word, operand, order);
}
inline void arm64_stadd_u16(std::uint16_t *word, std::uint16_t operand, std::memory_order order) noexcept {
    [[maybe_unused]] std::uint16_t const observed = arm64_ldadd_u16(word, operand, order);
}
inline void arm64_stadd_u32(std::uint32_t *word, std::uint32_t operand, std::memory_order order) noexcept {
    [[maybe_unused]] std::uint32_t const observed = arm64_ldadd_u32(word, operand, order);
}
inline void arm64_stadd_u64(std::uint64_t *word, std::uint64_t operand, std::memory_order order) noexcept {
    [[maybe_unused]] std::uint64_t const observed = arm64_ldadd_u64(word, operand, order);
}
inline void arm64_stclr_u8(std::uint8_t *word, std::uint8_t bits, std::memory_order order) noexcept {
    [[maybe_unused]] std::uint8_t const observed = arm64_ldclr_u8(word, bits, order);
}
inline void arm64_stclr_u16(std::uint16_t *word, std::uint16_t bits, std::memory_order order) noexcept {
    [[maybe_unused]] std::uint16_t const observed = arm64_ldclr_u16(word, bits, order);
}
inline void arm64_stclr_u32(std::uint32_t *word, std::uint32_t bits, std::memory_order order) noexcept {
    [[maybe_unused]] std::uint32_t const observed = arm64_ldclr_u32(word, bits, order);
}
inline void arm64_stclr_u64(std::uint64_t *word, std::uint64_t bits, std::memory_order order) noexcept {
    [[maybe_unused]] std::uint64_t const observed = arm64_ldclr_u64(word, bits, order);
}
inline void arm64_stset_u8(std::uint8_t *word, std::uint8_t bits, std::memory_order order) noexcept {
    [[maybe_unused]] std::uint8_t const observed = arm64_ldset_u8(word, bits, order);
}
inline void arm64_stset_u16(std::uint16_t *word, std::uint16_t bits, std::memory_order order) noexcept {
    [[maybe_unused]] std::uint16_t const observed = arm64_ldset_u16(word, bits, order);
}
inline void arm64_stset_u32(std::uint32_t *word, std::uint32_t bits, std::memory_order order) noexcept {
    [[maybe_unused]] std::uint32_t const observed = arm64_ldset_u32(word, bits, order);
}
inline void arm64_stset_u64(std::uint64_t *word, std::uint64_t bits, std::memory_order order) noexcept {
    [[maybe_unused]] std::uint64_t const observed = arm64_ldset_u64(word, bits, order);
}
inline void arm64_steor_u8(std::uint8_t *word, std::uint8_t bits, std::memory_order order) noexcept {
    [[maybe_unused]] std::uint8_t const observed = arm64_ldeor_u8(word, bits, order);
}
inline void arm64_steor_u16(std::uint16_t *word, std::uint16_t bits, std::memory_order order) noexcept {
    [[maybe_unused]] std::uint16_t const observed = arm64_ldeor_u16(word, bits, order);
}
inline void arm64_steor_u32(std::uint32_t *word, std::uint32_t bits, std::memory_order order) noexcept {
    [[maybe_unused]] std::uint32_t const observed = arm64_ldeor_u32(word, bits, order);
}
inline void arm64_steor_u64(std::uint64_t *word, std::uint64_t bits, std::memory_order order) noexcept {
    [[maybe_unused]] std::uint64_t const observed = arm64_ldeor_u64(word, bits, order);
}

/*  LSE maxima & minima: nothing spells @c ldsmax & kin either, and no `_Interlocked*` computes
 *  them, so a @c cas loop stands in. It swaps on every pass, an unchanged word included, so the
 *  ordered flavors stay the read-modify-write their callers were promised. */

inline std::uint8_t arm64_ldumax_u8(std::uint8_t *word, std::uint8_t operand, std::memory_order order) noexcept {
    std::uint8_t observed = arm64_load_u8(word);
    while (true) {
        std::uint8_t const wanted = observed < operand ? operand : observed;
        std::uint8_t const seen = arm64_cas_u8(word, observed, wanted, order);
        if (seen == observed) return observed;
        observed = seen;
    }
}
inline std::uint16_t arm64_ldumax_u16(std::uint16_t *word, std::uint16_t operand, std::memory_order order) noexcept {
    std::uint16_t observed = arm64_load_u16(word);
    while (true) {
        std::uint16_t const wanted = observed < operand ? operand : observed;
        std::uint16_t const seen = arm64_cas_u16(word, observed, wanted, order);
        if (seen == observed) return observed;
        observed = seen;
    }
}
inline std::uint32_t arm64_ldumax_u32(std::uint32_t *word, std::uint32_t operand, std::memory_order order) noexcept {
    std::uint32_t observed = arm64_load_u32(word);
    while (true) {
        std::uint32_t const wanted = observed < operand ? operand : observed;
        std::uint32_t const seen = arm64_cas_u32(word, observed, wanted, order);
        if (seen == observed) return observed;
        observed = seen;
    }
}
inline std::uint64_t arm64_ldumax_u64(std::uint64_t *word, std::uint64_t operand, std::memory_order order) noexcept {
    std::uint64_t observed = arm64_load_u64(word);
    while (true) {
        std::uint64_t const wanted = observed < operand ? operand : observed;
        std::uint64_t const seen = arm64_cas_u64(word, observed, wanted, order);
        if (seen == observed) return observed;
        observed = seen;
    }
}
inline std::int8_t arm64_ldsmax_i8(std::int8_t *word, std::int8_t operand, std::memory_order order) noexcept {
    std::uint8_t *target = reinterpret_cast<std::uint8_t *>(word);
    std::int8_t observed = static_cast<std::int8_t>(arm64_load_u8(target));
    while (true) {
        std::int8_t const wanted = observed < operand ? operand : observed;
        std::int8_t const seen = static_cast<std::int8_t>(
            arm64_cas_u8(target, static_cast<std::uint8_t>(observed), static_cast<std::uint8_t>(wanted), order));
        if (seen == observed) return observed;
        observed = seen;
    }
}
inline std::int16_t arm64_ldsmax_i16(std::int16_t *word, std::int16_t operand, std::memory_order order) noexcept {
    std::uint16_t *target = reinterpret_cast<std::uint16_t *>(word);
    std::int16_t observed = static_cast<std::int16_t>(arm64_load_u16(target));
    while (true) {
        std::int16_t const wanted = observed < operand ? operand : observed;
        std::int16_t const seen = static_cast<std::int16_t>(
            arm64_cas_u16(target, static_cast<std::uint16_t>(observed), static_cast<std::uint16_t>(wanted), order));
        if (seen == observed) return observed;
        observed = seen;
    }
}
inline std::int32_t arm64_ldsmax_i32(std::int32_t *word, std::int32_t operand, std::memory_order order) noexcept {
    std::uint32_t *target = reinterpret_cast<std::uint32_t *>(word);
    std::int32_t observed = static_cast<std::int32_t>(arm64_load_u32(target));
    while (true) {
        std::int32_t const wanted = observed < operand ? operand : observed;
        std::int32_t const seen = static_cast<std::int32_t>(
            arm64_cas_u32(target, static_cast<std::uint32_t>(observed), static_cast<std::uint32_t>(wanted), order));
        if (seen == observed) return observed;
        observed = seen;
    }
}
inline std::int64_t arm64_ldsmax_i64(std::int64_t *word, std::int64_t operand, std::memory_order order) noexcept {
    std::uint64_t *target = reinterpret_cast<std::uint64_t *>(word);
    std::int64_t observed = static_cast<std::int64_t>(arm64_load_u64(target));
    while (true) {
        std::int64_t const wanted = observed < operand ? operand : observed;
        std::int64_t const seen = static_cast<std::int64_t>(
            arm64_cas_u64(target, static_cast<std::uint64_t>(observed), static_cast<std::uint64_t>(wanted), order));
        if (seen == observed) return observed;
        observed = seen;
    }
}

inline std::uint8_t arm64_ldumin_u8(std::uint8_t *word, std::uint8_t operand, std::memory_order order) noexcept {
    std::uint8_t observed = arm64_load_u8(word);
    while (true) {
        std::uint8_t const wanted = operand < observed ? operand : observed;
        std::uint8_t const seen = arm64_cas_u8(word, observed, wanted, order);
        if (seen == observed) return observed;
        observed = seen;
    }
}
inline std::uint16_t arm64_ldumin_u16(std::uint16_t *word, std::uint16_t operand, std::memory_order order) noexcept {
    std::uint16_t observed = arm64_load_u16(word);
    while (true) {
        std::uint16_t const wanted = operand < observed ? operand : observed;
        std::uint16_t const seen = arm64_cas_u16(word, observed, wanted, order);
        if (seen == observed) return observed;
        observed = seen;
    }
}
inline std::uint32_t arm64_ldumin_u32(std::uint32_t *word, std::uint32_t operand, std::memory_order order) noexcept {
    std::uint32_t observed = arm64_load_u32(word);
    while (true) {
        std::uint32_t const wanted = operand < observed ? operand : observed;
        std::uint32_t const seen = arm64_cas_u32(word, observed, wanted, order);
        if (seen == observed) return observed;
        observed = seen;
    }
}
inline std::uint64_t arm64_ldumin_u64(std::uint64_t *word, std::uint64_t operand, std::memory_order order) noexcept {
    std::uint64_t observed = arm64_load_u64(word);
    while (true) {
        std::uint64_t const wanted = operand < observed ? operand : observed;
        std::uint64_t const seen = arm64_cas_u64(word, observed, wanted, order);
        if (seen == observed) return observed;
        observed = seen;
    }
}
inline std::int8_t arm64_ldsmin_i8(std::int8_t *word, std::int8_t operand, std::memory_order order) noexcept {
    std::uint8_t *target = reinterpret_cast<std::uint8_t *>(word);
    std::int8_t observed = static_cast<std::int8_t>(arm64_load_u8(target));
    while (true) {
        std::int8_t const wanted = operand < observed ? operand : observed;
        std::int8_t const seen = static_cast<std::int8_t>(
            arm64_cas_u8(target, static_cast<std::uint8_t>(observed), static_cast<std::uint8_t>(wanted), order));
        if (seen == observed) return observed;
        observed = seen;
    }
}
inline std::int16_t arm64_ldsmin_i16(std::int16_t *word, std::int16_t operand, std::memory_order order) noexcept {
    std::uint16_t *target = reinterpret_cast<std::uint16_t *>(word);
    std::int16_t observed = static_cast<std::int16_t>(arm64_load_u16(target));
    while (true) {
        std::int16_t const wanted = operand < observed ? operand : observed;
        std::int16_t const seen = static_cast<std::int16_t>(
            arm64_cas_u16(target, static_cast<std::uint16_t>(observed), static_cast<std::uint16_t>(wanted), order));
        if (seen == observed) return observed;
        observed = seen;
    }
}
inline std::int32_t arm64_ldsmin_i32(std::int32_t *word, std::int32_t operand, std::memory_order order) noexcept {
    std::uint32_t *target = reinterpret_cast<std::uint32_t *>(word);
    std::int32_t observed = static_cast<std::int32_t>(arm64_load_u32(target));
    while (true) {
        std::int32_t const wanted = operand < observed ? operand : observed;
        std::int32_t const seen = static_cast<std::int32_t>(
            arm64_cas_u32(target, static_cast<std::uint32_t>(observed), static_cast<std::uint32_t>(wanted), order));
        if (seen == observed) return observed;
        observed = seen;
    }
}
inline std::int64_t arm64_ldsmin_i64(std::int64_t *word, std::int64_t operand, std::memory_order order) noexcept {
    std::uint64_t *target = reinterpret_cast<std::uint64_t *>(word);
    std::int64_t observed = static_cast<std::int64_t>(arm64_load_u64(target));
    while (true) {
        std::int64_t const wanted = operand < observed ? operand : observed;
        std::int64_t const seen = static_cast<std::int64_t>(
            arm64_cas_u64(target, static_cast<std::uint64_t>(observed), static_cast<std::uint64_t>(wanted), order));
        if (seen == observed) return observed;
        observed = seen;
    }
}

/*  No-return maxima & minima: the loops above with their answers dropped. */

inline void arm64_stumax_u8(std::uint8_t *word, std::uint8_t operand, std::memory_order order) noexcept {
    [[maybe_unused]] std::uint8_t const observed = arm64_ldumax_u8(word, operand, order);
}
inline void arm64_stumax_u16(std::uint16_t *word, std::uint16_t operand, std::memory_order order) noexcept {
    [[maybe_unused]] std::uint16_t const observed = arm64_ldumax_u16(word, operand, order);
}
inline void arm64_stumax_u32(std::uint32_t *word, std::uint32_t operand, std::memory_order order) noexcept {
    [[maybe_unused]] std::uint32_t const observed = arm64_ldumax_u32(word, operand, order);
}
inline void arm64_stumax_u64(std::uint64_t *word, std::uint64_t operand, std::memory_order order) noexcept {
    [[maybe_unused]] std::uint64_t const observed = arm64_ldumax_u64(word, operand, order);
}
inline void arm64_stsmax_i8(std::int8_t *word, std::int8_t operand, std::memory_order order) noexcept {
    [[maybe_unused]] std::int8_t const observed = arm64_ldsmax_i8(word, operand, order);
}
inline void arm64_stsmax_i16(std::int16_t *word, std::int16_t operand, std::memory_order order) noexcept {
    [[maybe_unused]] std::int16_t const observed = arm64_ldsmax_i16(word, operand, order);
}
inline void arm64_stsmax_i32(std::int32_t *word, std::int32_t operand, std::memory_order order) noexcept {
    [[maybe_unused]] std::int32_t const observed = arm64_ldsmax_i32(word, operand, order);
}
inline void arm64_stsmax_i64(std::int64_t *word, std::int64_t operand, std::memory_order order) noexcept {
    [[maybe_unused]] std::int64_t const observed = arm64_ldsmax_i64(word, operand, order);
}
inline void arm64_stumin_u8(std::uint8_t *word, std::uint8_t operand, std::memory_order order) noexcept {
    [[maybe_unused]] std::uint8_t const observed = arm64_ldumin_u8(word, operand, order);
}
inline void arm64_stumin_u16(std::uint16_t *word, std::uint16_t operand, std::memory_order order) noexcept {
    [[maybe_unused]] std::uint16_t const observed = arm64_ldumin_u16(word, operand, order);
}
inline void arm64_stumin_u32(std::uint32_t *word, std::uint32_t operand, std::memory_order order) noexcept {
    [[maybe_unused]] std::uint32_t const observed = arm64_ldumin_u32(word, operand, order);
}
inline void arm64_stumin_u64(std::uint64_t *word, std::uint64_t operand, std::memory_order order) noexcept {
    [[maybe_unused]] std::uint64_t const observed = arm64_ldumin_u64(word, operand, order);
}
inline void arm64_stsmin_i8(std::int8_t *word, std::int8_t operand, std::memory_order order) noexcept {
    [[maybe_unused]] std::int8_t const observed = arm64_ldsmin_i8(word, operand, order);
}
inline void arm64_stsmin_i16(std::int16_t *word, std::int16_t operand, std::memory_order order) noexcept {
    [[maybe_unused]] std::int16_t const observed = arm64_ldsmin_i16(word, operand, order);
}
inline void arm64_stsmin_i32(std::int32_t *word, std::int32_t operand, std::memory_order order) noexcept {
    [[maybe_unused]] std::int32_t const observed = arm64_ldsmin_i32(word, operand, order);
}
inline void arm64_stsmin_i64(std::int64_t *word, std::int64_t operand, std::memory_order order) noexcept {
    [[maybe_unused]] std::int64_t const observed = arm64_ldsmin_i64(word, operand, order);
}

#endif // FORKUNION_HAS_ARM64_ATOMIC_INTRINSICS_

/**
 *  @brief @c std::atomic_ref over Armv8.1 LSE: @c ldar and @c stlr for ordered loads & stores, one
 *      instruction per read-modify-write in the acquire/release flavor the order asks for.
 *  @sa capability_arm64_lse_k, the admitting bit; arm64_rcpc_atomic_ref, the same with RCpc loads.
 *
 *  Same discipline as the standard reference: the word is naturally aligned and never touched
 *  non-atomically while references exist. Widths & operations not spelled above fail to compile
 *  rather than fall back.
 */
template <typename value_type_>
struct arm64_lse_atomic_ref {
    static_assert(std::is_trivially_copyable_v<value_type_>, "Only trivially-copyable words are atomic");
    static_assert(sizeof(value_type_) == 1 || sizeof(value_type_) == 2 || sizeof(value_type_) == 4 ||
                      sizeof(value_type_) == 8,
                  "Only 8-, 16-, 32- and 64-bit words are spelled");
    using value_t = value_type_;
    using word_t = atomic_word<value_type_>;
    using signed_word_t = std::make_signed_t<word_t>;
    static constexpr signed_word_t signed_bits_(word_t bits) noexcept { return std::bit_cast<signed_word_t>(bits); }
    static constexpr capabilities_t capabilities_k = capability_arm64_lse_k;

    explicit arm64_lse_atomic_ref(value_type_ &word) noexcept : word_(reinterpret_cast<word_t *>(&word)) {
        assert(reinterpret_cast<std::uintptr_t>(&word) % sizeof(value_type_) == 0 &&
               "Atomics need a naturally aligned word");
    }

    value_type_ load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
        if (order == std::memory_order_relaxed) {
            if constexpr (sizeof(value_type_) == 1) return std::bit_cast<value_type_>(arm64_load_u8(word_));
            else if constexpr (sizeof(value_type_) == 2) return std::bit_cast<value_type_>(arm64_load_u16(word_));
            else if constexpr (sizeof(value_type_) == 4) return std::bit_cast<value_type_>(arm64_load_u32(word_));
            else return std::bit_cast<value_type_>(arm64_load_u64(word_));
        }
        if constexpr (sizeof(value_type_) == 1) return std::bit_cast<value_type_>(arm64_ldar_u8(word_));
        else if constexpr (sizeof(value_type_) == 2) return std::bit_cast<value_type_>(arm64_ldar_u16(word_));
        else if constexpr (sizeof(value_type_) == 4) return std::bit_cast<value_type_>(arm64_ldar_u32(word_));
        else return std::bit_cast<value_type_>(arm64_ldar_u64(word_));
    }

    void store(value_type_ desired, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        word_t const word = std::bit_cast<word_t>(desired);
        if (order == std::memory_order_relaxed) {
            if constexpr (sizeof(value_type_) == 1) arm64_store_u8(word_, word);
            else if constexpr (sizeof(value_type_) == 2) arm64_store_u16(word_, word);
            else if constexpr (sizeof(value_type_) == 4) arm64_store_u32(word_, word);
            else arm64_store_u64(word_, word);
        }
        else {
            if constexpr (sizeof(value_type_) == 1) arm64_stlr_u8(word_, word);
            else if constexpr (sizeof(value_type_) == 2) arm64_stlr_u16(word_, word);
            else if constexpr (sizeof(value_type_) == 4) arm64_stlr_u32(word_, word);
            else arm64_stlr_u64(word_, word);
        }
    }

    value_type_ exchange(value_type_ desired, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        word_t const word = std::bit_cast<word_t>(desired);
        if constexpr (sizeof(value_type_) == 1) return std::bit_cast<value_type_>(arm64_swp_u8(word_, word, order));
        else if constexpr (sizeof(value_type_) == 2)
            return std::bit_cast<value_type_>(arm64_swp_u16(word_, word, order));
        else if constexpr (sizeof(value_type_) == 4)
            return std::bit_cast<value_type_>(arm64_swp_u32(word_, word, order));
        else return std::bit_cast<value_type_>(arm64_swp_u64(word_, word, order));
    }

    /** One @c cas runs in the stronger of the two orders whatever the outcome; a failed compare is
     *  then an acquiring load of the observed word, which the standard permits. */
    bool compare_exchange_strong(value_type_ &expected, value_type_ desired, std::memory_order success,
                                 std::memory_order failure) const noexcept {
        word_t const wanted = std::bit_cast<word_t>(expected);
        word_t const word = std::bit_cast<word_t>(desired);
        std::memory_order const order = stronger_(success, failure);
        word_t observed;
        if constexpr (sizeof(value_type_) == 1) observed = arm64_cas_u8(word_, wanted, word, order);
        else if constexpr (sizeof(value_type_) == 2) observed = arm64_cas_u16(word_, wanted, word, order);
        else if constexpr (sizeof(value_type_) == 4) observed = arm64_cas_u32(word_, wanted, word, order);
        else observed = arm64_cas_u64(word_, wanted, word, order);
        expected = std::bit_cast<value_type_>(observed);
        return observed == wanted;
    }
    bool compare_exchange_weak(value_type_ &expected, value_type_ desired, std::memory_order success,
                               std::memory_order failure) const noexcept {
        return compare_exchange_strong(expected, desired, success, failure);
    }
    bool compare_exchange_strong(value_type_ &expected, value_type_ desired,
                                 std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return compare_exchange_strong(expected, desired, order, failure_order(order));
    }
    bool compare_exchange_weak(value_type_ &expected, value_type_ desired,
                               std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return compare_exchange_strong(expected, desired, order, failure_order(order));
    }

    value_type_ fetch_add(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        word_t const word = std::bit_cast<word_t>(operand);
        if constexpr (sizeof(value_type_) == 1) return std::bit_cast<value_type_>(arm64_ldadd_u8(word_, word, order));
        else if constexpr (sizeof(value_type_) == 2)
            return std::bit_cast<value_type_>(arm64_ldadd_u16(word_, word, order));
        else if constexpr (sizeof(value_type_) == 4)
            return std::bit_cast<value_type_>(arm64_ldadd_u32(word_, word, order));
        else return std::bit_cast<value_type_>(arm64_ldadd_u64(word_, word, order));
    }
    value_type_ fetch_sub(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        word_t const negated = static_cast<word_t>(word_t {0} - std::bit_cast<word_t>(operand));
        return fetch_add(std::bit_cast<value_type_>(negated), order);
    }
    value_type_ fetch_and(value_type_ mask, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        word_t const cleared = static_cast<word_t>(~std::bit_cast<word_t>(mask));
        if constexpr (sizeof(value_type_) == 1)
            return std::bit_cast<value_type_>(arm64_ldclr_u8(word_, cleared, order));
        else if constexpr (sizeof(value_type_) == 2)
            return std::bit_cast<value_type_>(arm64_ldclr_u16(word_, cleared, order));
        else if constexpr (sizeof(value_type_) == 4)
            return std::bit_cast<value_type_>(arm64_ldclr_u32(word_, cleared, order));
        else return std::bit_cast<value_type_>(arm64_ldclr_u64(word_, cleared, order));
    }
    value_type_ fetch_or(value_type_ bits, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        word_t const word = std::bit_cast<word_t>(bits);
        if constexpr (sizeof(value_type_) == 1) return std::bit_cast<value_type_>(arm64_ldset_u8(word_, word, order));
        else if constexpr (sizeof(value_type_) == 2)
            return std::bit_cast<value_type_>(arm64_ldset_u16(word_, word, order));
        else if constexpr (sizeof(value_type_) == 4)
            return std::bit_cast<value_type_>(arm64_ldset_u32(word_, word, order));
        else return std::bit_cast<value_type_>(arm64_ldset_u64(word_, word, order));
    }
    value_type_ fetch_xor(value_type_ bits, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        word_t const word = std::bit_cast<word_t>(bits);
        if constexpr (sizeof(value_type_) == 1) return std::bit_cast<value_type_>(arm64_ldeor_u8(word_, word, order));
        else if constexpr (sizeof(value_type_) == 2)
            return std::bit_cast<value_type_>(arm64_ldeor_u16(word_, word, order));
        else if constexpr (sizeof(value_type_) == 4)
            return std::bit_cast<value_type_>(arm64_ldeor_u32(word_, word, order));
        else return std::bit_cast<value_type_>(arm64_ldeor_u64(word_, word, order));
    }

    /** Ahead of C++26: one @c ldsmax and @c ldumax, the signedness picking the instruction. */
    value_type_ fetch_max(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        word_t const bits = std::bit_cast<word_t>(operand);
        if constexpr (std::numeric_limits<value_type_>::is_signed && sizeof(value_type_) == 1)
            return std::bit_cast<value_type_>(
                arm64_ldsmax_i8(reinterpret_cast<std::int8_t *>(word_), signed_bits_(bits), order));
        else if constexpr (std::numeric_limits<value_type_>::is_signed && sizeof(value_type_) == 2)
            return std::bit_cast<value_type_>(
                arm64_ldsmax_i16(reinterpret_cast<std::int16_t *>(word_), signed_bits_(bits), order));
        else if constexpr (std::numeric_limits<value_type_>::is_signed && sizeof(value_type_) == 4)
            return std::bit_cast<value_type_>(
                arm64_ldsmax_i32(reinterpret_cast<std::int32_t *>(word_), signed_bits_(bits), order));
        else if constexpr (std::numeric_limits<value_type_>::is_signed)
            return std::bit_cast<value_type_>(
                arm64_ldsmax_i64(reinterpret_cast<std::int64_t *>(word_), signed_bits_(bits), order));
        else if constexpr (sizeof(value_type_) == 1)
            return std::bit_cast<value_type_>(arm64_ldumax_u8(word_, bits, order));
        else if constexpr (sizeof(value_type_) == 2)
            return std::bit_cast<value_type_>(arm64_ldumax_u16(word_, bits, order));
        else if constexpr (sizeof(value_type_) == 4)
            return std::bit_cast<value_type_>(arm64_ldumax_u32(word_, bits, order));
        else return std::bit_cast<value_type_>(arm64_ldumax_u64(word_, bits, order));
    }
    value_type_ fetch_min(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        word_t const bits = std::bit_cast<word_t>(operand);
        if constexpr (std::numeric_limits<value_type_>::is_signed && sizeof(value_type_) == 1)
            return std::bit_cast<value_type_>(
                arm64_ldsmin_i8(reinterpret_cast<std::int8_t *>(word_), signed_bits_(bits), order));
        else if constexpr (std::numeric_limits<value_type_>::is_signed && sizeof(value_type_) == 2)
            return std::bit_cast<value_type_>(
                arm64_ldsmin_i16(reinterpret_cast<std::int16_t *>(word_), signed_bits_(bits), order));
        else if constexpr (std::numeric_limits<value_type_>::is_signed && sizeof(value_type_) == 4)
            return std::bit_cast<value_type_>(
                arm64_ldsmin_i32(reinterpret_cast<std::int32_t *>(word_), signed_bits_(bits), order));
        else if constexpr (std::numeric_limits<value_type_>::is_signed)
            return std::bit_cast<value_type_>(
                arm64_ldsmin_i64(reinterpret_cast<std::int64_t *>(word_), signed_bits_(bits), order));
        else if constexpr (sizeof(value_type_) == 1)
            return std::bit_cast<value_type_>(arm64_ldumin_u8(word_, bits, order));
        else if constexpr (sizeof(value_type_) == 2)
            return std::bit_cast<value_type_>(arm64_ldumin_u16(word_, bits, order));
        else if constexpr (sizeof(value_type_) == 4)
            return std::bit_cast<value_type_>(arm64_ldumin_u32(word_, bits, order));
        else return std::bit_cast<value_type_>(arm64_ldumin_u64(word_, bits, order));
    }

    /** Floats through the integer forms on their bits: a non-negative operand orders like a signed
     *  integer, a negative one like an unsigned integer reversed - one instruction still. */
    value_type_ fetch_max(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_float<value_type_>
    {
        word_t const bits = std::bit_cast<word_t>(operand);
        if (signed_bits_(bits) >= 0)
            return std::bit_cast<value_type_>(signed_word_().fetch_max(signed_bits_(bits), order));
        return std::bit_cast<value_type_>(unsigned_word_().fetch_min(bits, order));
    }
    value_type_ fetch_min(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_float<value_type_>
    {
        word_t const bits = std::bit_cast<word_t>(operand);
        if (signed_bits_(bits) >= 0)
            return std::bit_cast<value_type_>(signed_word_().fetch_min(signed_bits_(bits), order));
        return std::bit_cast<value_type_>(unsigned_word_().fetch_max(bits, order));
    }

    /** No-return forms: @c stadd, @c stclr, @c stset - posted, nothing waited for. */
    void add(value_type_ operand, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        word_t const word = std::bit_cast<word_t>(operand);
        if constexpr (sizeof(value_type_) == 1) arm64_stadd_u8(word_, word, order);
        else if constexpr (sizeof(value_type_) == 2) arm64_stadd_u16(word_, word, order);
        else if constexpr (sizeof(value_type_) == 4) arm64_stadd_u32(word_, word, order);
        else arm64_stadd_u64(word_, word, order);
    }
    void sub(value_type_ operand, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        add(std::bit_cast<value_type_>(static_cast<word_t>(word_t {0} - std::bit_cast<word_t>(operand))), order);
    }
    void set_bits(value_type_ bits, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        word_t const word = std::bit_cast<word_t>(bits);
        if constexpr (sizeof(value_type_) == 1) arm64_stset_u8(word_, word, order);
        else if constexpr (sizeof(value_type_) == 2) arm64_stset_u16(word_, word, order);
        else if constexpr (sizeof(value_type_) == 4) arm64_stset_u32(word_, word, order);
        else arm64_stset_u64(word_, word, order);
    }
    void clear_bits(value_type_ bits, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        word_t const word = std::bit_cast<word_t>(bits);
        if constexpr (sizeof(value_type_) == 1) arm64_stclr_u8(word_, word, order);
        else if constexpr (sizeof(value_type_) == 2) arm64_stclr_u16(word_, word, order);
        else if constexpr (sizeof(value_type_) == 4) arm64_stclr_u32(word_, word, order);
        else arm64_stclr_u64(word_, word, order);
    }
    void flip_bits(value_type_ bits, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        word_t const word = std::bit_cast<word_t>(bits);
        if constexpr (sizeof(value_type_) == 1) arm64_steor_u8(word_, word, order);
        else if constexpr (sizeof(value_type_) == 2) arm64_steor_u16(word_, word, order);
        else if constexpr (sizeof(value_type_) == 4) arm64_steor_u32(word_, word, order);
        else arm64_steor_u64(word_, word, order);
    }
    bool fetch_test_and_set_bit(unsigned bit, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        word_t const mask = static_cast<word_t>(word_t {1} << bit);
        return (std::bit_cast<word_t>(fetch_or(std::bit_cast<value_type_>(mask), order)) >> bit) & 1;
    }
    bool fetch_test_and_reset_bit(unsigned bit, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        word_t const mask = static_cast<word_t>(~(word_t {1} << bit));
        return (std::bit_cast<word_t>(fetch_and(std::bit_cast<value_type_>(mask), order)) >> bit) & 1;
    }
    void max(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        word_t const bits = std::bit_cast<word_t>(operand);
        if constexpr (std::numeric_limits<value_type_>::is_signed && sizeof(value_type_) == 1)
            arm64_stsmax_i8(reinterpret_cast<std::int8_t *>(word_), signed_bits_(bits), order);
        else if constexpr (std::numeric_limits<value_type_>::is_signed && sizeof(value_type_) == 2)
            arm64_stsmax_i16(reinterpret_cast<std::int16_t *>(word_), signed_bits_(bits), order);
        else if constexpr (std::numeric_limits<value_type_>::is_signed && sizeof(value_type_) == 4)
            arm64_stsmax_i32(reinterpret_cast<std::int32_t *>(word_), signed_bits_(bits), order);
        else if constexpr (std::numeric_limits<value_type_>::is_signed)
            arm64_stsmax_i64(reinterpret_cast<std::int64_t *>(word_), signed_bits_(bits), order);
        else if constexpr (sizeof(value_type_) == 1) arm64_stumax_u8(word_, bits, order);
        else if constexpr (sizeof(value_type_) == 2) arm64_stumax_u16(word_, bits, order);
        else if constexpr (sizeof(value_type_) == 4) arm64_stumax_u32(word_, bits, order);
        else arm64_stumax_u64(word_, bits, order);
    }
    void min(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        word_t const bits = std::bit_cast<word_t>(operand);
        if constexpr (std::numeric_limits<value_type_>::is_signed && sizeof(value_type_) == 1)
            arm64_stsmin_i8(reinterpret_cast<std::int8_t *>(word_), signed_bits_(bits), order);
        else if constexpr (std::numeric_limits<value_type_>::is_signed && sizeof(value_type_) == 2)
            arm64_stsmin_i16(reinterpret_cast<std::int16_t *>(word_), signed_bits_(bits), order);
        else if constexpr (std::numeric_limits<value_type_>::is_signed && sizeof(value_type_) == 4)
            arm64_stsmin_i32(reinterpret_cast<std::int32_t *>(word_), signed_bits_(bits), order);
        else if constexpr (std::numeric_limits<value_type_>::is_signed)
            arm64_stsmin_i64(reinterpret_cast<std::int64_t *>(word_), signed_bits_(bits), order);
        else if constexpr (sizeof(value_type_) == 1) arm64_stumin_u8(word_, bits, order);
        else if constexpr (sizeof(value_type_) == 2) arm64_stumin_u16(word_, bits, order);
        else if constexpr (sizeof(value_type_) == 4) arm64_stumin_u32(word_, bits, order);
        else arm64_stumin_u64(word_, bits, order);
    }
    void max(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_float<value_type_>
    {
        word_t const bits = std::bit_cast<word_t>(operand);
        if (signed_bits_(bits) >= 0) signed_word_().max(signed_bits_(bits), order);
        else unsigned_word_().min(bits, order);
    }
    void min(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_float<value_type_>
    {
        word_t const bits = std::bit_cast<word_t>(operand);
        if (signed_bits_(bits) >= 0) signed_word_().min(signed_bits_(bits), order);
        else unsigned_word_().max(bits, order);
    }

    /** The conditional forms: Arm has no @c cmpccxadd, so a read-first @c cas loop - one acquiring
     *  load, then one instruction per attempt. */
    value_type_ fetch_add_if_at_most(value_type_ operand, value_type_ limit,
                                     std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        value_type_ observed = load(std::memory_order_acquire);
        while (observed <= limit && limit - observed >= operand &&
               !compare_exchange_strong(observed, observed + operand, order, std::memory_order_acquire)) {}
        return observed;
    }
    value_type_ fetch_add_if_at_least(value_type_ operand, value_type_ floor,
                                      std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        value_type_ bound;
        value_type_ observed = load(std::memory_order_acquire);
        if (!fetch_add_if_at_least_bound(operand, floor, bound)) return observed;
        while (observed >= bound && !compare_exchange_strong(observed, static_cast<value_type_>(observed + operand),
                                                             order, std::memory_order_acquire)) {}
        return observed;
    }
    value_type_ fetch_sub_if_at_least(value_type_ operand, value_type_ floor,
                                      std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        value_type_ observed = load(std::memory_order_acquire);
        while (observed >= floor && observed - floor >= operand &&
               !compare_exchange_strong(observed, observed - operand, order, std::memory_order_acquire)) {}
        return observed;
    }

  private:
    word_t *word_;

    /** The same word as integers, for floats ordered through their bits. */
    arm64_lse_atomic_ref<signed_word_t> signed_word_() const noexcept {
        return arm64_lse_atomic_ref<signed_word_t>(*reinterpret_cast<signed_word_t *>(word_));
    }
    arm64_lse_atomic_ref<word_t> unsigned_word_() const noexcept { return arm64_lse_atomic_ref<word_t>(*word_); }

    static constexpr std::memory_order stronger_(std::memory_order success, std::memory_order failure) noexcept {
        bool const acquire = acquires(success) || acquires(failure);
        bool const release = releases(success);
        if (acquire && release) return std::memory_order_acq_rel;
        if (acquire) return std::memory_order_acquire;
        if (release) return std::memory_order_release;
        return std::memory_order_relaxed;
    }
};

/**
 *  @brief Read-only access through @ref arm64_lse_atomic_ref: the same @c LDR and @c LDAR over a
 *      const pointer, so no cast exists.
 *  @sa standard_atomic_ref<value_type_ const>
 */
template <typename value_type_>
struct arm64_lse_atomic_ref<value_type_ const> {
    using value_t = value_type_ const;
    using word_t = atomic_word<value_type_>;
    static constexpr capabilities_t capabilities_k = capability_arm64_lse_k;

    explicit arm64_lse_atomic_ref(value_type_ const &word) noexcept : word_(reinterpret_cast<word_t const *>(&word)) {
        assert(reinterpret_cast<std::uintptr_t>(&word) % sizeof(value_type_) == 0 &&
               "Atomics need a naturally aligned word");
    }

    /** A temporary would die before the reference does. */
    arm64_lse_atomic_ref(value_type_ const &&) = delete;

    value_type_ load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
        if (order == std::memory_order_relaxed) {
            if constexpr (sizeof(value_type_) == 1) return std::bit_cast<value_type_>(arm64_load_u8(word_));
            else if constexpr (sizeof(value_type_) == 2) return std::bit_cast<value_type_>(arm64_load_u16(word_));
            else if constexpr (sizeof(value_type_) == 4) return std::bit_cast<value_type_>(arm64_load_u32(word_));
            else return std::bit_cast<value_type_>(arm64_load_u64(word_));
        }
        if constexpr (sizeof(value_type_) == 1) return std::bit_cast<value_type_>(arm64_ldar_u8(word_));
        else if constexpr (sizeof(value_type_) == 2) return std::bit_cast<value_type_>(arm64_ldar_u16(word_));
        else if constexpr (sizeof(value_type_) == 4) return std::bit_cast<value_type_>(arm64_ldar_u32(word_));
        else return std::bit_cast<value_type_>(arm64_ldar_u64(word_));
    }

  private:
    word_t const *word_;
};

#pragma endregion Arm64 LSE

#endif // FORKUNION_TARGET_ARM64_LSE

/*  RCpc acquire loads: ordered against later loads and stores, not against earlier stores. */
#if FORKUNION_TARGET_ARM64_RCPC
#pragma region Arm64 RCpc

#if FORKUNION_HAS_INLINE_ASM_

inline std::uint8_t arm64_ldapr_u8(std::uint8_t const *word) noexcept {
    std::uint8_t value;
    __asm__ __volatile__(".arch_extension rcpc\n\tldaprb %w0, [%1]" : "=r"(value) : "r"(word) : "memory");
    return value;
}
inline std::uint16_t arm64_ldapr_u16(std::uint16_t const *word) noexcept {
    std::uint16_t value;
    __asm__ __volatile__(".arch_extension rcpc\n\tldaprh %w0, [%1]" : "=r"(value) : "r"(word) : "memory");
    return value;
}
inline std::uint32_t arm64_ldapr_u32(std::uint32_t const *word) noexcept {
    std::uint32_t value;
    __asm__ __volatile__(".arch_extension rcpc\n\tldapr %w0, [%1]" : "=r"(value) : "r"(word) : "memory");
    return value;
}
inline std::uint64_t arm64_ldapr_u64(std::uint64_t const *word) noexcept {
    std::uint64_t value;
    __asm__ __volatile__(".arch_extension rcpc\n\tldapr %x0, [%1]" : "=r"(value) : "r"(word) : "memory");
    return value;
}

#endif // FORKUNION_HAS_INLINE_ASM_

#if FORKUNION_HAS_ARM64_ATOMIC_INTRINSICS_

inline std::uint8_t arm64_ldapr_u8(std::uint8_t const *word) noexcept { return __ldapr8(word); }
inline std::uint16_t arm64_ldapr_u16(std::uint16_t const *word) noexcept { return __ldapr16(word); }
inline std::uint32_t arm64_ldapr_u32(std::uint32_t const *word) noexcept { return __ldapr32(word); }
inline std::uint64_t arm64_ldapr_u64(std::uint64_t const *word) noexcept { return __ldapr64(word); }

#endif // FORKUNION_HAS_ARM64_ATOMIC_INTRINSICS_

/**
 *  @brief Armv8.3 RCpc on top of LSE: acquiring loads are @c ldapr. Sequentially-consistent loads
 *      keep @c ldar - the only form that composes with @c stlr into a total order.
 *  @sa capability_arm64_rcpc_k - the bit admitting it, on top of @c capability_arm64_lse_k.
 */
template <typename value_type_>
struct arm64_rcpc_atomic_ref {
    using value_t = value_type_;
    using word_t = atomic_word<value_type_>;
    static constexpr capabilities_t capabilities_k = capability_arm64_lse_k | capability_arm64_rcpc_k;

    explicit arm64_rcpc_atomic_ref(value_type_ &word) noexcept : word_(reinterpret_cast<word_t *>(&word)) {
        assert(reinterpret_cast<std::uintptr_t>(&word) % sizeof(value_type_) == 0 &&
               "Atomics need a naturally aligned word");
    }

    value_type_ load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
        if (order != std::memory_order_acquire && order != std::memory_order_consume) return weaker_().load(order);
        if constexpr (sizeof(value_type_) == 1) return std::bit_cast<value_type_>(arm64_ldapr_u8(word_));
        else if constexpr (sizeof(value_type_) == 2) return std::bit_cast<value_type_>(arm64_ldapr_u16(word_));
        else if constexpr (sizeof(value_type_) == 4) return std::bit_cast<value_type_>(arm64_ldapr_u32(word_));
        else return std::bit_cast<value_type_>(arm64_ldapr_u64(word_));
    }
    void store(value_type_ desired, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        weaker_().store(desired, order);
    }
    value_type_ exchange(value_type_ desired, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return weaker_().exchange(desired, order);
    }

    bool compare_exchange_strong(value_type_ &expected, value_type_ desired, std::memory_order success,
                                 std::memory_order failure) const noexcept {
        return weaker_().compare_exchange_strong(expected, desired, success, failure);
    }
    bool compare_exchange_weak(value_type_ &expected, value_type_ desired, std::memory_order success,
                               std::memory_order failure) const noexcept {
        return weaker_().compare_exchange_weak(expected, desired, success, failure);
    }
    bool compare_exchange_strong(value_type_ &expected, value_type_ desired,
                                 std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return weaker_().compare_exchange_strong(expected, desired, order);
    }
    bool compare_exchange_weak(value_type_ &expected, value_type_ desired,
                               std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return weaker_().compare_exchange_weak(expected, desired, order);
    }

    value_type_ fetch_add(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        return weaker_().fetch_add(operand, order);
    }
    value_type_ fetch_sub(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        return weaker_().fetch_sub(operand, order);
    }
    value_type_ fetch_and(value_type_ mask, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        return weaker_().fetch_and(mask, order);
    }
    value_type_ fetch_or(value_type_ bits, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        return weaker_().fetch_or(bits, order);
    }
    value_type_ fetch_xor(value_type_ bits, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        return weaker_().fetch_xor(bits, order);
    }
    value_type_ fetch_max(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> || atomic_float<value_type_>
    {
        return weaker_().fetch_max(operand, order);
    }
    value_type_ fetch_min(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> || atomic_float<value_type_>
    {
        return weaker_().fetch_min(operand, order);
    }

    void add(value_type_ operand, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        weaker_().add(operand, order);
    }
    void sub(value_type_ operand, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        weaker_().sub(operand, order);
    }
    void set_bits(value_type_ bits, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        weaker_().set_bits(bits, order);
    }
    void clear_bits(value_type_ bits, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        weaker_().clear_bits(bits, order);
    }
    void flip_bits(value_type_ bits, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        weaker_().flip_bits(bits, order);
    }
    bool fetch_test_and_set_bit(unsigned bit, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        return weaker_().fetch_test_and_set_bit(bit, order);
    }
    bool fetch_test_and_reset_bit(unsigned bit, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        return weaker_().fetch_test_and_reset_bit(bit, order);
    }
    void max(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> || atomic_float<value_type_>
    {
        weaker_().max(operand, order);
    }
    void min(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> || atomic_float<value_type_>
    {
        weaker_().min(operand, order);
    }

    value_type_ fetch_add_if_at_most(value_type_ operand, value_type_ limit,
                                     std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        return weaker_().fetch_add_if_at_most(operand, limit, order);
    }
    value_type_ fetch_add_if_at_least(value_type_ operand, value_type_ floor,
                                      std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        return weaker_().fetch_add_if_at_least(operand, floor, order);
    }
    value_type_ fetch_sub_if_at_least(value_type_ operand, value_type_ floor,
                                      std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        return weaker_().fetch_sub_if_at_least(operand, floor, order);
    }

  private:
    /** The previous rung, built on demand: it holds this same pointer and nothing else. */
    arm64_lse_atomic_ref<value_type_> weaker_() const noexcept {
        return arm64_lse_atomic_ref<value_type_>(*reinterpret_cast<value_type_ *>(word_));
    }

    word_t *word_;
};

/**
 *  @brief Read-only access through @ref arm64_rcpc_atomic_ref: @c LDAPR for the acquiring orders,
 *      the LSE loads for the rest.
 *  @sa standard_atomic_ref<value_type_ const>
 */
template <typename value_type_>
struct arm64_rcpc_atomic_ref<value_type_ const> {
    using value_t = value_type_ const;
    using word_t = atomic_word<value_type_>;
    static constexpr capabilities_t capabilities_k = capability_arm64_lse_k | capability_arm64_rcpc_k;

    explicit arm64_rcpc_atomic_ref(value_type_ const &word) noexcept : word_(&word) {
        assert(reinterpret_cast<std::uintptr_t>(&word) % sizeof(value_type_) == 0 &&
               "Atomics need a naturally aligned word");
    }

    /** A temporary would die before the reference does. */
    arm64_rcpc_atomic_ref(value_type_ const &&) = delete;

    value_type_ load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
        if (order != std::memory_order_acquire && order != std::memory_order_consume) return weaker_().load(order);
        word_t const *word = reinterpret_cast<word_t const *>(word_);
        if constexpr (sizeof(value_type_) == 1) return std::bit_cast<value_type_>(arm64_ldapr_u8(word));
        else if constexpr (sizeof(value_type_) == 2) return std::bit_cast<value_type_>(arm64_ldapr_u16(word));
        else if constexpr (sizeof(value_type_) == 4) return std::bit_cast<value_type_>(arm64_ldapr_u32(word));
        else return std::bit_cast<value_type_>(arm64_ldapr_u64(word));
    }

  private:
    /** The previous rung, built on demand: it holds this same pointer and nothing else. */
    arm64_lse_atomic_ref<value_type_ const> weaker_() const noexcept {
        return arm64_lse_atomic_ref<value_type_ const>(*word_);
    }

    value_type_ const *word_;
};

#pragma endregion Arm64 RCpc

#endif // FORKUNION_TARGET_ARM64_RCPC

/*  Loads & stores carry their order as fences, per the RISC-V mapping: acquire is `fence r,rw`
 *  after the load, release `fence rw,w` before the store, sequential consistency both. */
#if FORKUNION_TARGET_RISCV64_ATOMIC
#pragma region RISCV64 A

inline std::uint8_t riscv64_lbu(std::uint8_t const *word) noexcept {
    std::uint8_t value;
    __asm__ __volatile__("lbu %0, 0(%1)" : "=r"(value) : "r"(word) : "memory");
    return value;
}
inline std::uint16_t riscv64_lhu(std::uint16_t const *word) noexcept {
    std::uint16_t value;
    __asm__ __volatile__("lhu %0, 0(%1)" : "=r"(value) : "r"(word) : "memory");
    return value;
}
inline std::uint32_t riscv64_lw(std::uint32_t const *word) noexcept {
    std::uint32_t value;
    __asm__ __volatile__("lw %0, 0(%1)" : "=r"(value) : "r"(word) : "memory");
    return value;
}
inline std::uint64_t riscv64_ld(std::uint64_t const *word) noexcept {
    std::uint64_t value;
    __asm__ __volatile__("ld %0, 0(%1)" : "=r"(value) : "r"(word) : "memory");
    return value;
}
inline void riscv64_sb(std::uint8_t *word, std::uint8_t value) noexcept {
    __asm__ __volatile__("sb %0, 0(%1)" : : "r"(value), "r"(word) : "memory");
}
inline void riscv64_sh(std::uint16_t *word, std::uint16_t value) noexcept {
    __asm__ __volatile__("sh %0, 0(%1)" : : "r"(value), "r"(word) : "memory");
}
inline void riscv64_sw(std::uint32_t *word, std::uint32_t value) noexcept {
    __asm__ __volatile__("sw %0, 0(%1)" : : "r"(value), "r"(word) : "memory");
}
inline void riscv64_sd(std::uint64_t *word, std::uint64_t value) noexcept {
    __asm__ __volatile__("sd %0, 0(%1)" : : "r"(value), "r"(word) : "memory");
}
inline void riscv64_fence_r_rw() noexcept { __asm__ __volatile__("fence r, rw" : : : "memory"); }
inline void riscv64_fence_rw_w() noexcept { __asm__ __volatile__("fence rw, w" : : : "memory"); }
inline void riscv64_fence_rw_rw() noexcept { __asm__ __volatile__("fence rw, rw" : : : "memory"); }

/*  Atomic memory operations of the base A extension: @c rd receives what the word held; `.aqrl`
 *  serves every order but relaxed - a strengthening the ISA prices at nothing on the fast path. */

inline std::uint32_t riscv64_amoswap_w(std::uint32_t *word, std::uint32_t desired, std::memory_order order) noexcept {
    std::uint32_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoswap.w %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(desired) : "memory");
    else __asm__ __volatile__("amoswap.w.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(desired) : "memory");
    return observed;
}
inline std::uint64_t riscv64_amoswap_d(std::uint64_t *word, std::uint64_t desired, std::memory_order order) noexcept {
    std::uint64_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoswap.d %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(desired) : "memory");
    else __asm__ __volatile__("amoswap.d.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(desired) : "memory");
    return observed;
}
inline std::uint32_t riscv64_amoadd_w(std::uint32_t *word, std::uint32_t operand, std::memory_order order) noexcept {
    std::uint32_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoadd.w %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amoadd.w.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    return observed;
}
inline std::uint64_t riscv64_amoadd_d(std::uint64_t *word, std::uint64_t operand, std::memory_order order) noexcept {
    std::uint64_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoadd.d %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amoadd.d.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    return observed;
}
inline std::uint32_t riscv64_amoand_w(std::uint32_t *word, std::uint32_t mask, std::memory_order order) noexcept {
    std::uint32_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoand.w %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(mask) : "memory");
    else __asm__ __volatile__("amoand.w.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(mask) : "memory");
    return observed;
}
inline std::uint64_t riscv64_amoand_d(std::uint64_t *word, std::uint64_t mask, std::memory_order order) noexcept {
    std::uint64_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoand.d %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(mask) : "memory");
    else __asm__ __volatile__("amoand.d.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(mask) : "memory");
    return observed;
}
inline std::uint32_t riscv64_amoor_w(std::uint32_t *word, std::uint32_t bits, std::memory_order order) noexcept {
    std::uint32_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoor.w %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(bits) : "memory");
    else __asm__ __volatile__("amoor.w.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(bits) : "memory");
    return observed;
}
inline std::uint64_t riscv64_amoor_d(std::uint64_t *word, std::uint64_t bits, std::memory_order order) noexcept {
    std::uint64_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoor.d %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(bits) : "memory");
    else __asm__ __volatile__("amoor.d.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(bits) : "memory");
    return observed;
}
inline std::uint32_t riscv64_amoxor_w(std::uint32_t *word, std::uint32_t bits, std::memory_order order) noexcept {
    std::uint32_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoxor.w %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(bits) : "memory");
    else __asm__ __volatile__("amoxor.w.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(bits) : "memory");
    return observed;
}
inline std::uint64_t riscv64_amoxor_d(std::uint64_t *word, std::uint64_t bits, std::memory_order order) noexcept {
    std::uint64_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoxor.d %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(bits) : "memory");
    else __asm__ __volatile__("amoxor.d.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(bits) : "memory");
    return observed;
}
inline std::uint32_t riscv64_amomaxu_w(std::uint32_t *word, std::uint32_t operand, std::memory_order order) noexcept {
    std::uint32_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amomaxu.w %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amomaxu.w.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    return observed;
}
inline std::uint64_t riscv64_amomaxu_d(std::uint64_t *word, std::uint64_t operand, std::memory_order order) noexcept {
    std::uint64_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amomaxu.d %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amomaxu.d.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    return observed;
}
inline std::int32_t riscv64_amomax_w(std::int32_t *word, std::int32_t operand, std::memory_order order) noexcept {
    std::int32_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amomax.w %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amomax.w.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    return observed;
}
inline std::int64_t riscv64_amomax_d(std::int64_t *word, std::int64_t operand, std::memory_order order) noexcept {
    std::int64_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amomax.d %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amomax.d.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    return observed;
}
inline std::uint32_t riscv64_amominu_w(std::uint32_t *word, std::uint32_t operand, std::memory_order order) noexcept {
    std::uint32_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amominu.w %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amominu.w.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    return observed;
}
inline std::uint64_t riscv64_amominu_d(std::uint64_t *word, std::uint64_t operand, std::memory_order order) noexcept {
    std::uint64_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amominu.d %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amominu.d.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    return observed;
}
inline std::int32_t riscv64_amomin_w(std::int32_t *word, std::int32_t operand, std::memory_order order) noexcept {
    std::int32_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amomin.w %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amomin.w.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    return observed;
}
inline std::int64_t riscv64_amomin_d(std::int64_t *word, std::int64_t operand, std::memory_order order) noexcept {
    std::int64_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amomin.d %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amomin.d.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    return observed;
}

/*  No-return forms are the same operations with @c x0 as the destination - the ISA's own hint that
 *  nothing waits for the value; release is `.rl`, nothing to acquire. */

inline void riscv64_amoadd_w_x0(std::uint32_t *word, std::uint32_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoadd.w zero, %1, (%0)" : : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amoadd.w.rl zero, %1, (%0)" : : "r"(word), "r"(operand) : "memory");
}
inline void riscv64_amoadd_d_x0(std::uint64_t *word, std::uint64_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoadd.d zero, %1, (%0)" : : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amoadd.d.rl zero, %1, (%0)" : : "r"(word), "r"(operand) : "memory");
}
inline void riscv64_amoand_w_x0(std::uint32_t *word, std::uint32_t mask, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoand.w zero, %1, (%0)" : : "r"(word), "r"(mask) : "memory");
    else __asm__ __volatile__("amoand.w.rl zero, %1, (%0)" : : "r"(word), "r"(mask) : "memory");
}
inline void riscv64_amoand_d_x0(std::uint64_t *word, std::uint64_t mask, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoand.d zero, %1, (%0)" : : "r"(word), "r"(mask) : "memory");
    else __asm__ __volatile__("amoand.d.rl zero, %1, (%0)" : : "r"(word), "r"(mask) : "memory");
}
inline void riscv64_amoor_w_x0(std::uint32_t *word, std::uint32_t bits, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoor.w zero, %1, (%0)" : : "r"(word), "r"(bits) : "memory");
    else __asm__ __volatile__("amoor.w.rl zero, %1, (%0)" : : "r"(word), "r"(bits) : "memory");
}
inline void riscv64_amoor_d_x0(std::uint64_t *word, std::uint64_t bits, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoor.d zero, %1, (%0)" : : "r"(word), "r"(bits) : "memory");
    else __asm__ __volatile__("amoor.d.rl zero, %1, (%0)" : : "r"(word), "r"(bits) : "memory");
}
inline void riscv64_amoxor_w_x0(std::uint32_t *word, std::uint32_t bits, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoxor.w zero, %1, (%0)" : : "r"(word), "r"(bits) : "memory");
    else __asm__ __volatile__("amoxor.w.rl zero, %1, (%0)" : : "r"(word), "r"(bits) : "memory");
}
inline void riscv64_amoxor_d_x0(std::uint64_t *word, std::uint64_t bits, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoxor.d zero, %1, (%0)" : : "r"(word), "r"(bits) : "memory");
    else __asm__ __volatile__("amoxor.d.rl zero, %1, (%0)" : : "r"(word), "r"(bits) : "memory");
}
inline void riscv64_amomaxu_w_x0(std::uint32_t *word, std::uint32_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amomaxu.w zero, %1, (%0)" : : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amomaxu.w.rl zero, %1, (%0)" : : "r"(word), "r"(operand) : "memory");
}
inline void riscv64_amomaxu_d_x0(std::uint64_t *word, std::uint64_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amomaxu.d zero, %1, (%0)" : : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amomaxu.d.rl zero, %1, (%0)" : : "r"(word), "r"(operand) : "memory");
}
inline void riscv64_amomax_w_x0(std::int32_t *word, std::int32_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amomax.w zero, %1, (%0)" : : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amomax.w.rl zero, %1, (%0)" : : "r"(word), "r"(operand) : "memory");
}
inline void riscv64_amomax_d_x0(std::int64_t *word, std::int64_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amomax.d zero, %1, (%0)" : : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amomax.d.rl zero, %1, (%0)" : : "r"(word), "r"(operand) : "memory");
}
inline void riscv64_amominu_w_x0(std::uint32_t *word, std::uint32_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amominu.w zero, %1, (%0)" : : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amominu.w.rl zero, %1, (%0)" : : "r"(word), "r"(operand) : "memory");
}
inline void riscv64_amominu_d_x0(std::uint64_t *word, std::uint64_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amominu.d zero, %1, (%0)" : : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amominu.d.rl zero, %1, (%0)" : : "r"(word), "r"(operand) : "memory");
}
inline void riscv64_amomin_w_x0(std::int32_t *word, std::int32_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amomin.w zero, %1, (%0)" : : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amomin.w.rl zero, %1, (%0)" : : "r"(word), "r"(operand) : "memory");
}
inline void riscv64_amomin_d_x0(std::int64_t *word, std::int64_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amomin.d zero, %1, (%0)" : : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amomin.d.rl zero, %1, (%0)" : : "r"(word), "r"(operand) : "memory");
}

/*  Load-reserved / store-conditional loops: the reservation must live inside one assembly block.
 *  `lr.w` sign-extends, so the 32-bit comparands travel sign-extended too. */

inline std::uint32_t riscv64_lr_sc_cas_w(std::uint32_t *word, std::uint32_t expected, std::uint32_t desired) noexcept {
    std::int64_t const wanted = static_cast<std::int32_t>(expected);
    std::int64_t observed, failed;
    __asm__ __volatile__("1:\n\t"                                       //
                         "lr.w.aqrl %[observed], (%[word])\n\t"         //
                         "bne %[observed], %[wanted], 2f\n\t"           //
                         "sc.w.rl %[failed], %[desired], (%[word])\n\t" //
                         "bnez %[failed], 1b\n"                         //
                         "2:"
                         : [observed] "=&r"(observed), [failed] "=&r"(failed)
                         : [word] "r"(word), [wanted] "r"(wanted), [desired] "r"(desired)
                         : "memory");
    return static_cast<std::uint32_t>(observed);
}
inline std::uint64_t riscv64_lr_sc_cas_d(std::uint64_t *word, std::uint64_t expected, std::uint64_t desired) noexcept {
    std::uint64_t observed, failed;
    __asm__ __volatile__("1:\n\t"                                       //
                         "lr.d.aqrl %[observed], (%[word])\n\t"         //
                         "bne %[observed], %[expected], 2f\n\t"         //
                         "sc.d.rl %[failed], %[desired], (%[word])\n\t" //
                         "bnez %[failed], 1b\n"                         //
                         "2:"
                         : [observed] "=&r"(observed), [failed] "=&r"(failed)
                         : [word] "r"(word), [expected] "r"(expected), [desired] "r"(desired)
                         : "memory");
    return observed;
}

/** A byte exchange without @c Zabha: the aligned word is reserved, the byte masked in. */
inline std::uint8_t riscv64_lr_sc_swap_b(std::uint8_t *byte, std::uint8_t desired) noexcept {
    std::uintptr_t const address = reinterpret_cast<std::uintptr_t>(byte);
    std::uint32_t *word = reinterpret_cast<std::uint32_t *>(address & ~std::uintptr_t {3});
    unsigned const shift = static_cast<unsigned>(address & 3) * 8;
    std::uint32_t const keep = ~(std::uint32_t {0xFF} << shift);
    std::uint32_t const placed = std::uint32_t {desired} << shift;
    std::uint32_t observed, merged, failed;
    __asm__ __volatile__("1:\n\t"                                      //
                         "lr.w.aqrl %[observed], (%[word])\n\t"        //
                         "and %[merged], %[observed], %[keep]\n\t"     //
                         "or %[merged], %[merged], %[placed]\n\t"      //
                         "sc.w.rl %[failed], %[merged], (%[word])\n\t" //
                         "bnez %[failed], 1b"
                         : [observed] "=&r"(observed), [merged] "=&r"(merged), [failed] "=&r"(failed)
                         : [word] "r"(word), [keep] "r"(keep), [placed] "r"(placed)
                         : "memory");
    return static_cast<std::uint8_t>(observed >> shift);
}

/**
 *  @brief @c std::atomic_ref over the RISC-V base A extension: one @c amo* per read-modify-write,
 *      @c x0 as the destination for the no-return forms, @c lr and @c sc loops for compare-exchange
 *      and byte exchanges, fences around the ordered loads & stores.
 *  @sa capability_riscv64_atomic_k, the admitting bit; riscv64_zacas_atomic_ref, using @c amocas.
 */
template <typename value_type_>
struct riscv64_atomic_ref {
    static_assert(std::is_trivially_copyable_v<value_type_>, "Only trivially-copyable words are atomic");
    static_assert(sizeof(value_type_) == 1 || sizeof(value_type_) == 2 || sizeof(value_type_) == 4 ||
                      sizeof(value_type_) == 8,
                  "Only 8-, 16-, 32- and 64-bit words are spelled");
    using value_t = value_type_;
    using word_t = atomic_word<value_type_>;
    using signed_word_t = std::make_signed_t<word_t>;
    static constexpr signed_word_t signed_bits_(word_t bits) noexcept { return std::bit_cast<signed_word_t>(bits); }
    static constexpr capabilities_t capabilities_k = capability_riscv64_atomic_k;

    explicit riscv64_atomic_ref(value_type_ &word) noexcept : word_(reinterpret_cast<word_t *>(&word)) {
        assert(reinterpret_cast<std::uintptr_t>(&word) % sizeof(value_type_) == 0 &&
               "Atomics need a naturally aligned word");
    }

    value_type_ load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
        if (order == std::memory_order_seq_cst) riscv64_fence_rw_rw();
        word_t word;
        if constexpr (sizeof(value_type_) == 1) word = riscv64_lbu(word_);
        else if constexpr (sizeof(value_type_) == 2) word = riscv64_lhu(word_);
        else if constexpr (sizeof(value_type_) == 4) word = riscv64_lw(word_);
        else word = riscv64_ld(word_);
        if (acquires(order)) riscv64_fence_r_rw();
        return std::bit_cast<value_type_>(word);
    }
    void store(value_type_ desired, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        if (releases(order)) riscv64_fence_rw_w();
        word_t const word = std::bit_cast<word_t>(desired);
        if constexpr (sizeof(value_type_) == 1) riscv64_sb(word_, word);
        else if constexpr (sizeof(value_type_) == 2) riscv64_sh(word_, word);
        else if constexpr (sizeof(value_type_) == 4) riscv64_sw(word_, word);
        else riscv64_sd(word_, word);
        if (order == std::memory_order_seq_cst) riscv64_fence_rw_rw();
    }

    value_type_ exchange(value_type_ desired, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        static_assert(sizeof(value_type_) != 2, "Zabha spells 16-bit exchange");
        word_t const word = std::bit_cast<word_t>(desired);
        if constexpr (sizeof(value_type_) == 1) return std::bit_cast<value_type_>(riscv64_lr_sc_swap_b(word_, word));
        else if constexpr (sizeof(value_type_) == 4)
            return std::bit_cast<value_type_>(riscv64_amoswap_w(word_, word, order));
        else return std::bit_cast<value_type_>(riscv64_amoswap_d(word_, word, order));
    }

    bool compare_exchange_strong(value_type_ &expected, value_type_ desired, std::memory_order,
                                 std::memory_order) const noexcept {
        static_assert(sizeof(value_type_) >= 4, "Zabha spells sub-word compare-exchange");
        word_t const wanted = std::bit_cast<word_t>(expected);
        word_t observed;
        if constexpr (sizeof(value_type_) == 4)
            observed = riscv64_lr_sc_cas_w(word_, wanted, std::bit_cast<word_t>(desired));
        else observed = riscv64_lr_sc_cas_d(word_, wanted, std::bit_cast<word_t>(desired));
        expected = std::bit_cast<value_type_>(observed);
        return observed == wanted;
    }
    bool compare_exchange_weak(value_type_ &expected, value_type_ desired, std::memory_order success,
                               std::memory_order failure) const noexcept {
        return compare_exchange_strong(expected, desired, success, failure);
    }
    bool compare_exchange_strong(value_type_ &expected, value_type_ desired,
                                 std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return compare_exchange_strong(expected, desired, order, failure_order(order));
    }
    bool compare_exchange_weak(value_type_ &expected, value_type_ desired,
                               std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return compare_exchange_strong(expected, desired, order, failure_order(order));
    }

    value_type_ fetch_add(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        static_assert(sizeof(value_type_) >= 4, "Zabha spells sub-word fetch-add");
        word_t const word = std::bit_cast<word_t>(operand);
        if constexpr (sizeof(value_type_) == 4) return std::bit_cast<value_type_>(riscv64_amoadd_w(word_, word, order));
        else return std::bit_cast<value_type_>(riscv64_amoadd_d(word_, word, order));
    }
    value_type_ fetch_sub(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        word_t const negated = static_cast<word_t>(word_t {0} - std::bit_cast<word_t>(operand));
        return fetch_add(std::bit_cast<value_type_>(negated), order);
    }
    value_type_ fetch_and(value_type_ mask, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        static_assert(sizeof(value_type_) >= 4, "Zabha spells sub-word fetch-and");
        word_t const word = std::bit_cast<word_t>(mask);
        if constexpr (sizeof(value_type_) == 4) return std::bit_cast<value_type_>(riscv64_amoand_w(word_, word, order));
        else return std::bit_cast<value_type_>(riscv64_amoand_d(word_, word, order));
    }
    value_type_ fetch_or(value_type_ bits, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        static_assert(sizeof(value_type_) >= 4, "Zabha spells sub-word fetch-or");
        word_t const word = std::bit_cast<word_t>(bits);
        if constexpr (sizeof(value_type_) == 4) return std::bit_cast<value_type_>(riscv64_amoor_w(word_, word, order));
        else return std::bit_cast<value_type_>(riscv64_amoor_d(word_, word, order));
    }
    value_type_ fetch_xor(value_type_ bits, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        static_assert(sizeof(value_type_) >= 4, "Zabha spells sub-word fetch-xor");
        word_t const word = std::bit_cast<word_t>(bits);
        if constexpr (sizeof(value_type_) == 4) return std::bit_cast<value_type_>(riscv64_amoxor_w(word_, word, order));
        else return std::bit_cast<value_type_>(riscv64_amoxor_d(word_, word, order));
    }

    /** Native in the base A extension: @c amomax and @c amomaxu, @c amomin and @c amominu. */
    value_type_ fetch_max(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        static_assert(sizeof(value_type_) >= 4, "Zabha spells sub-word maxima");
        word_t const bits = std::bit_cast<word_t>(operand);
        if constexpr (std::numeric_limits<value_type_>::is_signed && sizeof(value_type_) == 4)
            return std::bit_cast<value_type_>(
                riscv64_amomax_w(reinterpret_cast<std::int32_t *>(word_), signed_bits_(bits), order));
        else if constexpr (std::numeric_limits<value_type_>::is_signed)
            return std::bit_cast<value_type_>(
                riscv64_amomax_d(reinterpret_cast<std::int64_t *>(word_), signed_bits_(bits), order));
        else if constexpr (sizeof(value_type_) == 4)
            return std::bit_cast<value_type_>(riscv64_amomaxu_w(word_, bits, order));
        else return std::bit_cast<value_type_>(riscv64_amomaxu_d(word_, bits, order));
    }
    value_type_ fetch_min(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        static_assert(sizeof(value_type_) >= 4, "Zabha spells sub-word minima");
        word_t const bits = std::bit_cast<word_t>(operand);
        if constexpr (std::numeric_limits<value_type_>::is_signed && sizeof(value_type_) == 4)
            return std::bit_cast<value_type_>(
                riscv64_amomin_w(reinterpret_cast<std::int32_t *>(word_), signed_bits_(bits), order));
        else if constexpr (std::numeric_limits<value_type_>::is_signed)
            return std::bit_cast<value_type_>(
                riscv64_amomin_d(reinterpret_cast<std::int64_t *>(word_), signed_bits_(bits), order));
        else if constexpr (sizeof(value_type_) == 4)
            return std::bit_cast<value_type_>(riscv64_amominu_w(word_, bits, order));
        else return std::bit_cast<value_type_>(riscv64_amominu_d(word_, bits, order));
    }

    /** Floats through the integer forms on their bits: a non-negative operand orders like a signed
     *  integer, a negative one like an unsigned integer reversed - one @c amo still. */
    value_type_ fetch_max(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_float<value_type_>
    {
        word_t const bits = std::bit_cast<word_t>(operand);
        if (signed_bits_(bits) >= 0)
            return std::bit_cast<value_type_>(signed_word_().fetch_max(signed_bits_(bits), order));
        return std::bit_cast<value_type_>(unsigned_word_().fetch_min(bits, order));
    }
    value_type_ fetch_min(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_float<value_type_>
    {
        word_t const bits = std::bit_cast<word_t>(operand);
        if (signed_bits_(bits) >= 0)
            return std::bit_cast<value_type_>(signed_word_().fetch_min(signed_bits_(bits), order));
        return std::bit_cast<value_type_>(unsigned_word_().fetch_max(bits, order));
    }

    /** No-return forms: the same operations into @c x0. */
    void add(value_type_ operand, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        static_assert(sizeof(value_type_) >= 4, "Zabha spells sub-word no-return add");
        word_t const word = std::bit_cast<word_t>(operand);
        if constexpr (sizeof(value_type_) == 4) riscv64_amoadd_w_x0(word_, word, order);
        else riscv64_amoadd_d_x0(word_, word, order);
    }
    void sub(value_type_ operand, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        add(std::bit_cast<value_type_>(static_cast<word_t>(word_t {0} - std::bit_cast<word_t>(operand))), order);
    }
    void set_bits(value_type_ bits, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        static_assert(sizeof(value_type_) >= 4, "Zabha spells sub-word no-return set");
        word_t const word = std::bit_cast<word_t>(bits);
        if constexpr (sizeof(value_type_) == 4) riscv64_amoor_w_x0(word_, word, order);
        else riscv64_amoor_d_x0(word_, word, order);
    }
    void clear_bits(value_type_ bits, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        static_assert(sizeof(value_type_) >= 4, "Zabha spells sub-word no-return clear");
        word_t const mask = static_cast<word_t>(~std::bit_cast<word_t>(bits));
        if constexpr (sizeof(value_type_) == 4) riscv64_amoand_w_x0(word_, mask, order);
        else riscv64_amoand_d_x0(word_, mask, order);
    }
    void flip_bits(value_type_ bits, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        static_assert(sizeof(value_type_) >= 4, "Zabha spells sub-word no-return flip");
        word_t const word = std::bit_cast<word_t>(bits);
        if constexpr (sizeof(value_type_) == 4) riscv64_amoxor_w_x0(word_, word, order);
        else riscv64_amoxor_d_x0(word_, word, order);
    }
    bool fetch_test_and_set_bit(unsigned bit, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        word_t const mask = static_cast<word_t>(word_t {1} << bit);
        return (std::bit_cast<word_t>(fetch_or(std::bit_cast<value_type_>(mask), order)) >> bit) & 1;
    }
    bool fetch_test_and_reset_bit(unsigned bit, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        word_t const mask = static_cast<word_t>(~(word_t {1} << bit));
        return (std::bit_cast<word_t>(fetch_and(std::bit_cast<value_type_>(mask), order)) >> bit) & 1;
    }
    void max(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        static_assert(sizeof(value_type_) >= 4, "Zabha spells sub-word no-return maxima");
        word_t const bits = std::bit_cast<word_t>(operand);
        if constexpr (std::numeric_limits<value_type_>::is_signed && sizeof(value_type_) == 4)
            riscv64_amomax_w_x0(reinterpret_cast<std::int32_t *>(word_), signed_bits_(bits), order);
        else if constexpr (std::numeric_limits<value_type_>::is_signed)
            riscv64_amomax_d_x0(reinterpret_cast<std::int64_t *>(word_), signed_bits_(bits), order);
        else if constexpr (sizeof(value_type_) == 4) riscv64_amomaxu_w_x0(word_, bits, order);
        else riscv64_amomaxu_d_x0(word_, bits, order);
    }
    void min(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        static_assert(sizeof(value_type_) >= 4, "Zabha spells sub-word no-return minima");
        word_t const bits = std::bit_cast<word_t>(operand);
        if constexpr (std::numeric_limits<value_type_>::is_signed && sizeof(value_type_) == 4)
            riscv64_amomin_w_x0(reinterpret_cast<std::int32_t *>(word_), signed_bits_(bits), order);
        else if constexpr (std::numeric_limits<value_type_>::is_signed)
            riscv64_amomin_d_x0(reinterpret_cast<std::int64_t *>(word_), signed_bits_(bits), order);
        else if constexpr (sizeof(value_type_) == 4) riscv64_amominu_w_x0(word_, bits, order);
        else riscv64_amominu_d_x0(word_, bits, order);
    }
    void max(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_float<value_type_>
    {
        word_t const bits = std::bit_cast<word_t>(operand);
        if (signed_bits_(bits) >= 0) signed_word_().max(signed_bits_(bits), order);
        else unsigned_word_().min(bits, order);
    }
    void min(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_float<value_type_>
    {
        word_t const bits = std::bit_cast<word_t>(operand);
        if (signed_bits_(bits) >= 0) signed_word_().min(signed_bits_(bits), order);
        else unsigned_word_().max(bits, order);
    }

    /** The conditional forms: a read-first compare-exchange loop, or @c amocas where available. */
    value_type_ fetch_add_if_at_most(value_type_ operand, value_type_ limit,
                                     std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        value_type_ observed = load(std::memory_order_acquire);
        while (observed <= limit && limit - observed >= operand &&
               !compare_exchange_strong(observed, observed + operand, order, std::memory_order_acquire)) {}
        return observed;
    }
    value_type_ fetch_add_if_at_least(value_type_ operand, value_type_ floor,
                                      std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        value_type_ bound;
        value_type_ observed = load(std::memory_order_acquire);
        if (!fetch_add_if_at_least_bound(operand, floor, bound)) return observed;
        while (observed >= bound && !compare_exchange_strong(observed, static_cast<value_type_>(observed + operand),
                                                             order, std::memory_order_acquire)) {}
        return observed;
    }
    value_type_ fetch_sub_if_at_least(value_type_ operand, value_type_ floor,
                                      std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        value_type_ observed = load(std::memory_order_acquire);
        while (observed >= floor && observed - floor >= operand &&
               !compare_exchange_strong(observed, observed - operand, order, std::memory_order_acquire)) {}
        return observed;
    }

  private:
    word_t *word_;

    /** The same word as integers, for floats ordered through their bits. */
    riscv64_atomic_ref<signed_word_t> signed_word_() const noexcept {
        return riscv64_atomic_ref<signed_word_t>(*reinterpret_cast<signed_word_t *>(word_));
    }
    riscv64_atomic_ref<word_t> unsigned_word_() const noexcept { return riscv64_atomic_ref<word_t>(*word_); }
};

/**
 *  @brief Read-only access through @ref riscv64_atomic_ref: the same loads over a const pointer, so
 *      no cast exists.
 *  @sa standard_atomic_ref<value_type_ const>
 */
template <typename value_type_>
struct riscv64_atomic_ref<value_type_ const> {
    using value_t = value_type_ const;
    using word_t = atomic_word<value_type_>;
    static constexpr capabilities_t capabilities_k = capability_riscv64_atomic_k;

    explicit riscv64_atomic_ref(value_type_ const &word) noexcept : word_(reinterpret_cast<word_t const *>(&word)) {
        assert(reinterpret_cast<std::uintptr_t>(&word) % sizeof(value_type_) == 0 &&
               "Atomics need a naturally aligned word");
    }

    /** A temporary would die before the reference does. */
    riscv64_atomic_ref(value_type_ const &&) = delete;

    value_type_ load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
        if (order == std::memory_order_seq_cst) riscv64_fence_rw_rw();
        word_t word;
        if constexpr (sizeof(value_type_) == 1) word = riscv64_lbu(word_);
        else if constexpr (sizeof(value_type_) == 2) word = riscv64_lhu(word_);
        else if constexpr (sizeof(value_type_) == 4) word = riscv64_lw(word_);
        else word = riscv64_ld(word_);
        if (acquires(order)) riscv64_fence_r_rw();
        return std::bit_cast<value_type_>(word);
    }

  private:
    word_t const *word_;
};

#pragma endregion RISCV64 A

#endif // FORKUNION_TARGET_RISCV64_ATOMIC

/*  @c Zacas: compare-and-swap as one instruction; the comparand register receives what the word
 *  held. Assemblers disagree on how to name the extension inline - @c zacas, @c zacas1p0, or not at
 *  all - and `.insn` is a directive older LLVM lacks, so the two are whole words with the registers
 *  pinned: the AMO opcode, funct5 `00101`, both @c aq and @c rl set, @c a0 as the comparand, @c a1
 *  as the address, @c a2 as the desired value. A baseline @c rv64gc build then assembles them and
 *  the runtime bit decides. */
#if FORKUNION_TARGET_RISCV64_ZACAS
#pragma region RISCV64 Zacas

inline std::uint32_t riscv64_amocas_w(std::uint32_t *word, std::uint32_t expected, std::uint32_t desired) noexcept {
    register std::int64_t observed __asm__("a0") = static_cast<std::int32_t>(expected);
    register std::uint32_t *address __asm__("a1") = word;
    register std::uint32_t value __asm__("a2") = desired;
    __asm__ __volatile__(".4byte 0x2ec5a52f"
                         : "+r"(observed)
                         : "r"(address), "r"(value)
                         : "memory"); // ? `amocas.w.aqrl a0, a2, (a1)`
    return static_cast<std::uint32_t>(observed);
}
inline std::uint64_t riscv64_amocas_d(std::uint64_t *word, std::uint64_t expected, std::uint64_t desired) noexcept {
    register std::uint64_t observed __asm__("a0") = expected;
    register std::uint64_t *address __asm__("a1") = word;
    register std::uint64_t value __asm__("a2") = desired;
    __asm__ __volatile__(".4byte 0x2ec5b52f"
                         : "+r"(observed)
                         : "r"(address), "r"(value)
                         : "memory"); // ? `amocas.d.aqrl a0, a2, (a1)`
    return observed;
}

/**
 *  @brief @c Zacas atop the base: one @c amocas compare-exchange instead of an
 *      @c lr and @c sc loop.
 *  @sa capability_riscv64_zacas_k, the admitting bit;
 *      riscv64_atomic_ref, the reference it composes.
 */
template <typename value_type_>
struct riscv64_zacas_atomic_ref {
    using value_t = value_type_;
    using word_t = atomic_word<value_type_>;
    static constexpr capabilities_t capabilities_k = capability_riscv64_atomic_k | capability_riscv64_zacas_k;

    explicit riscv64_zacas_atomic_ref(value_type_ &word) noexcept : word_(reinterpret_cast<word_t *>(&word)) {
        assert(reinterpret_cast<std::uintptr_t>(&word) % sizeof(value_type_) == 0 &&
               "Atomics need a naturally aligned word");
    }

    value_type_ load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return weaker_().load(order);
    }
    void store(value_type_ desired, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        weaker_().store(desired, order);
    }
    value_type_ exchange(value_type_ desired, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return weaker_().exchange(desired, order);
    }

    bool compare_exchange_strong(value_type_ &expected, value_type_ desired, std::memory_order,
                                 std::memory_order) const noexcept {
        static_assert(sizeof(value_type_) >= 4, "Byte and halfword compare-exchange need Zabha");
        word_t const wanted = std::bit_cast<word_t>(expected);
        word_t observed;
        if constexpr (sizeof(value_type_) == 4)
            observed = riscv64_amocas_w(word_, wanted, std::bit_cast<word_t>(desired));
        else observed = riscv64_amocas_d(word_, wanted, std::bit_cast<word_t>(desired));
        expected = std::bit_cast<value_type_>(observed);
        return observed == wanted;
    }
    bool compare_exchange_weak(value_type_ &expected, value_type_ desired, std::memory_order success,
                               std::memory_order failure) const noexcept {
        return compare_exchange_strong(expected, desired, success, failure);
    }
    bool compare_exchange_strong(value_type_ &expected, value_type_ desired,
                                 std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return compare_exchange_strong(expected, desired, order, failure_order(order));
    }
    bool compare_exchange_weak(value_type_ &expected, value_type_ desired,
                               std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return compare_exchange_strong(expected, desired, order, failure_order(order));
    }

    value_type_ fetch_add(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        return weaker_().fetch_add(operand, order);
    }
    value_type_ fetch_sub(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        return weaker_().fetch_sub(operand, order);
    }
    value_type_ fetch_and(value_type_ mask, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        return weaker_().fetch_and(mask, order);
    }
    value_type_ fetch_or(value_type_ bits, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        return weaker_().fetch_or(bits, order);
    }
    value_type_ fetch_xor(value_type_ bits, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        return weaker_().fetch_xor(bits, order);
    }
    value_type_ fetch_max(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> || atomic_float<value_type_>
    {
        return weaker_().fetch_max(operand, order);
    }
    value_type_ fetch_min(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> || atomic_float<value_type_>
    {
        return weaker_().fetch_min(operand, order);
    }

    void add(value_type_ operand, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        weaker_().add(operand, order);
    }
    void sub(value_type_ operand, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        weaker_().sub(operand, order);
    }
    void set_bits(value_type_ bits, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        weaker_().set_bits(bits, order);
    }
    void clear_bits(value_type_ bits, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        weaker_().clear_bits(bits, order);
    }
    void flip_bits(value_type_ bits, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        weaker_().flip_bits(bits, order);
    }
    bool fetch_test_and_set_bit(unsigned bit, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        return weaker_().fetch_test_and_set_bit(bit, order);
    }
    bool fetch_test_and_reset_bit(unsigned bit, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        return weaker_().fetch_test_and_reset_bit(bit, order);
    }
    void max(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> || atomic_float<value_type_>
    {
        weaker_().max(operand, order);
    }
    void min(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> || atomic_float<value_type_>
    {
        weaker_().min(operand, order);
    }

    value_type_ fetch_add_if_at_most(value_type_ operand, value_type_ limit,
                                     std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        return weaker_().fetch_add_if_at_most(operand, limit, order);
    }
    value_type_ fetch_add_if_at_least(value_type_ operand, value_type_ floor,
                                      std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        return weaker_().fetch_add_if_at_least(operand, floor, order);
    }
    value_type_ fetch_sub_if_at_least(value_type_ operand, value_type_ floor,
                                      std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        return weaker_().fetch_sub_if_at_least(operand, floor, order);
    }

  private:
    /** The previous rung, built on demand: it holds this same pointer and nothing else. */
    riscv64_atomic_ref<value_type_> weaker_() const noexcept {
        return riscv64_atomic_ref<value_type_>(*reinterpret_cast<value_type_ *>(word_));
    }

    word_t *word_;
};

/**
 *  @brief Read-only access through @ref riscv64_zacas_atomic_ref, which specializes no load.
 *  @sa standard_atomic_ref<value_type_ const>
 */
template <typename value_type_>
struct riscv64_zacas_atomic_ref<value_type_ const> {
    using value_t = value_type_ const;
    static constexpr capabilities_t capabilities_k = capability_riscv64_atomic_k | capability_riscv64_zacas_k;

    explicit riscv64_zacas_atomic_ref(value_type_ const &word) noexcept : portable_(word) {}

    /** A temporary would die before the reference does. */
    riscv64_zacas_atomic_ref(value_type_ const &&) = delete;

    value_type_ load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return portable_.load(order);
    }

  private:
    riscv64_atomic_ref<value_type_ const> portable_;
};

#pragma endregion RISCV64 Zacas

#endif // FORKUNION_TARGET_RISCV64_ZACAS

/*  @c Zabha: the byte and halfword @c amo forms - and with @c Zacas, which this rung extends, the
 *  byte and halfword @c amocas. Raw words for the reason @c amocas is one, with the same registers
 *  pinned - @c a0 the comparand and the result, @c a1 the address, @c a2 the operand - funct3
 *  @c 000 for bytes and @c 001 for halfwords, both @c aq and @c rl set. */
#if FORKUNION_TARGET_RISCV64_ZABHA
#pragma region RISCV64 Zabha

inline constexpr std::uint32_t riscv64_amoswap_b_k = 0x0ec5852f; // ? `amoswap.b.aqrl a0, a2, (a1)`
inline constexpr std::uint32_t riscv64_amoswap_h_k = 0x0ec5952f; // ? `amoswap.h.aqrl a0, a2, (a1)`
inline constexpr std::uint32_t riscv64_amocas_b_k = 0x2ec5852f;  // ? `amocas.b.aqrl a0, a2, (a1)`
inline constexpr std::uint32_t riscv64_amocas_h_k = 0x2ec5952f;  // ? `amocas.h.aqrl a0, a2, (a1)`
inline constexpr std::uint32_t riscv64_amoadd_b_k = 0x06c5852f;  // ? `amoadd.b.aqrl a0, a2, (a1)`
inline constexpr std::uint32_t riscv64_amoadd_h_k = 0x06c5952f;  // ? `amoadd.h.aqrl a0, a2, (a1)`
inline constexpr std::uint32_t riscv64_amoand_b_k = 0x66c5852f;  // ? `amoand.b.aqrl a0, a2, (a1)`
inline constexpr std::uint32_t riscv64_amoand_h_k = 0x66c5952f;  // ? `amoand.h.aqrl a0, a2, (a1)`
inline constexpr std::uint32_t riscv64_amoor_b_k = 0x46c5852f;   // ? `amoor.b.aqrl a0, a2, (a1)`
inline constexpr std::uint32_t riscv64_amoor_h_k = 0x46c5952f;   // ? `amoor.h.aqrl a0, a2, (a1)`
inline constexpr std::uint32_t riscv64_amoxor_b_k = 0x26c5852f;  // ? `amoxor.b.aqrl a0, a2, (a1)`
inline constexpr std::uint32_t riscv64_amoxor_h_k = 0x26c5952f;  // ? `amoxor.h.aqrl a0, a2, (a1)`
inline constexpr std::uint32_t riscv64_amomax_b_k = 0xa6c5852f;  // ? `amomax.b.aqrl a0, a2, (a1)`
inline constexpr std::uint32_t riscv64_amomax_h_k = 0xa6c5952f;  // ? `amomax.h.aqrl a0, a2, (a1)`
inline constexpr std::uint32_t riscv64_amomaxu_b_k = 0xe6c5852f; // ? `amomaxu.b.aqrl a0, a2, (a1)`
inline constexpr std::uint32_t riscv64_amomaxu_h_k = 0xe6c5952f; // ? `amomaxu.h.aqrl a0, a2, (a1)`
inline constexpr std::uint32_t riscv64_amomin_b_k = 0x86c5852f;  // ? `amomin.b.aqrl a0, a2, (a1)`
inline constexpr std::uint32_t riscv64_amomin_h_k = 0x86c5952f;  // ? `amomin.h.aqrl a0, a2, (a1)`
inline constexpr std::uint32_t riscv64_amominu_b_k = 0xc6c5852f; // ? `amominu.b.aqrl a0, a2, (a1)`
inline constexpr std::uint32_t riscv64_amominu_h_k = 0xc6c5952f; // ? `amominu.h.aqrl a0, a2, (a1)`

/** One pinned @c amo word from the table above: @c a0 carries the comparand in and what the word
 *  held out, sign-extended. */
template <std::uint32_t instruction_>
inline std::uint64_t riscv64_zabha_amo(void *word, std::uint64_t operand, std::uint64_t comparand) noexcept {
    register std::uint64_t observed __asm__("a0") = comparand;
    register void *address __asm__("a1") = word;
    register std::uint64_t value __asm__("a2") = operand;
    __asm__ __volatile__(".4byte %3" : "+r"(observed) : "r"(address), "r"(value), "i"(instruction_) : "memory");
    return observed;
}

/**
 *  @brief @c Zabha atop @c Zacas: every read-modify-write of a byte or halfword as one @c amo,
 *      where the lower rungs loop over the aligned word or refuse to compile.
 *  @sa capability_riscv64_zabha_k, the admitting bit; riscv64_zacas_atomic_ref, the reference it
 *      composes for 32- and 64-bit words.
 *
 *  Every byte and halfword @c amo runs @c aqrl whatever the order, as the @c amocas words do.
 */
template <typename value_type_>
struct riscv64_zabha_atomic_ref {
    static_assert(std::is_trivially_copyable_v<value_type_>, "Only trivially-copyable words are atomic");
    static_assert(sizeof(value_type_) == 1 || sizeof(value_type_) == 2 || sizeof(value_type_) == 4 ||
                      sizeof(value_type_) == 8,
                  "Only 8-, 16-, 32- and 64-bit words are spelled");
    using value_t = value_type_;
    using word_t = atomic_word<value_type_>;
    using signed_word_t = std::make_signed_t<word_t>;
    static constexpr signed_word_t signed_bits_(word_t bits) noexcept { return std::bit_cast<signed_word_t>(bits); }
    static constexpr capabilities_t capabilities_k =
        capability_riscv64_atomic_k | capability_riscv64_zacas_k | capability_riscv64_zabha_k;

    explicit riscv64_zabha_atomic_ref(value_type_ &word) noexcept : word_(reinterpret_cast<word_t *>(&word)) {
        assert(reinterpret_cast<std::uintptr_t>(&word) % sizeof(value_type_) == 0 &&
               "Atomics need a naturally aligned word");
    }

    value_type_ load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return weaker_().load(order);
    }
    void store(value_type_ desired, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        weaker_().store(desired, order);
    }
    value_type_ exchange(value_type_ desired, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        if constexpr (sizeof(value_type_) >= 4) return weaker_().exchange(desired, order);
        else return amo_<riscv64_amoswap_b_k, riscv64_amoswap_h_k>(desired);
    }

    bool compare_exchange_strong(value_type_ &expected, value_type_ desired, std::memory_order success,
                                 std::memory_order failure) const noexcept {
        if constexpr (sizeof(value_type_) >= 4)
            return weaker_().compare_exchange_strong(expected, desired, success, failure);
        else {
            word_t const wanted = std::bit_cast<word_t>(expected);
            std::uint64_t const comparand = static_cast<std::uint64_t>(signed_bits_(wanted));
            word_t const observed = static_cast<word_t>(
                riscv64_zabha_amo<sizeof(value_type_) == 1 ? riscv64_amocas_b_k : riscv64_amocas_h_k>(
                    word_, std::bit_cast<word_t>(desired), comparand));
            expected = std::bit_cast<value_type_>(observed);
            return observed == wanted;
        }
    }
    bool compare_exchange_weak(value_type_ &expected, value_type_ desired, std::memory_order success,
                               std::memory_order failure) const noexcept {
        return compare_exchange_strong(expected, desired, success, failure);
    }
    bool compare_exchange_strong(value_type_ &expected, value_type_ desired,
                                 std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return compare_exchange_strong(expected, desired, order, failure_order(order));
    }
    bool compare_exchange_weak(value_type_ &expected, value_type_ desired,
                               std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return compare_exchange_strong(expected, desired, order, failure_order(order));
    }

    value_type_ fetch_add(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        if constexpr (sizeof(value_type_) >= 4) return weaker_().fetch_add(operand, order);
        else return amo_<riscv64_amoadd_b_k, riscv64_amoadd_h_k>(operand);
    }
    value_type_ fetch_sub(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        word_t const negated = static_cast<word_t>(word_t {0} - std::bit_cast<word_t>(operand));
        return fetch_add(std::bit_cast<value_type_>(negated), order);
    }
    value_type_ fetch_and(value_type_ mask, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        if constexpr (sizeof(value_type_) >= 4) return weaker_().fetch_and(mask, order);
        else return amo_<riscv64_amoand_b_k, riscv64_amoand_h_k>(mask);
    }
    value_type_ fetch_or(value_type_ bits, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        if constexpr (sizeof(value_type_) >= 4) return weaker_().fetch_or(bits, order);
        else return amo_<riscv64_amoor_b_k, riscv64_amoor_h_k>(bits);
    }
    value_type_ fetch_xor(value_type_ bits, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        if constexpr (sizeof(value_type_) >= 4) return weaker_().fetch_xor(bits, order);
        else return amo_<riscv64_amoxor_b_k, riscv64_amoxor_h_k>(bits);
    }
    value_type_ fetch_max(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> || atomic_float<value_type_>
    {
        if constexpr (sizeof(value_type_) >= 4) return weaker_().fetch_max(operand, order);
        else if constexpr (std::numeric_limits<value_type_>::is_signed)
            return amo_<riscv64_amomax_b_k, riscv64_amomax_h_k>(operand);
        else return amo_<riscv64_amomaxu_b_k, riscv64_amomaxu_h_k>(operand);
    }
    value_type_ fetch_min(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> || atomic_float<value_type_>
    {
        if constexpr (sizeof(value_type_) >= 4) return weaker_().fetch_min(operand, order);
        else if constexpr (std::numeric_limits<value_type_>::is_signed)
            return amo_<riscv64_amomin_b_k, riscv64_amomin_h_k>(operand);
        else return amo_<riscv64_amominu_b_k, riscv64_amominu_h_k>(operand);
    }

    /** No-return forms: the @c x0 words for 32 and 64 bits, the returning ones for the rest. */
    void add(value_type_ operand, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        if constexpr (sizeof(value_type_) >= 4) weaker_().add(operand, order);
        else fetch_add(operand, order);
    }
    void sub(value_type_ operand, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        if constexpr (sizeof(value_type_) >= 4) weaker_().sub(operand, order);
        else fetch_sub(operand, order);
    }
    void set_bits(value_type_ bits, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        if constexpr (sizeof(value_type_) >= 4) weaker_().set_bits(bits, order);
        else fetch_or(bits, order);
    }
    void clear_bits(value_type_ bits, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        if constexpr (sizeof(value_type_) >= 4) weaker_().clear_bits(bits, order);
        else fetch_and(std::bit_cast<value_type_>(static_cast<word_t>(~std::bit_cast<word_t>(bits))), order);
    }
    void flip_bits(value_type_ bits, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        if constexpr (sizeof(value_type_) >= 4) weaker_().flip_bits(bits, order);
        else fetch_xor(bits, order);
    }
    bool fetch_test_and_set_bit(unsigned bit, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        word_t const mask = static_cast<word_t>(word_t {1} << bit);
        return (std::bit_cast<word_t>(fetch_or(std::bit_cast<value_type_>(mask), order)) >> bit) & 1;
    }
    bool fetch_test_and_reset_bit(unsigned bit, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        word_t const mask = static_cast<word_t>(~(word_t {1} << bit));
        return (std::bit_cast<word_t>(fetch_and(std::bit_cast<value_type_>(mask), order)) >> bit) & 1;
    }
    void max(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> || atomic_float<value_type_>
    {
        if constexpr (sizeof(value_type_) >= 4) weaker_().max(operand, order);
        else fetch_max(operand, order);
    }
    void min(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> || atomic_float<value_type_>
    {
        if constexpr (sizeof(value_type_) >= 4) weaker_().min(operand, order);
        else fetch_min(operand, order);
    }

    /** The conditional forms: a read-first loop over the one-instruction compare-exchange. */
    value_type_ fetch_add_if_at_most(value_type_ operand, value_type_ limit,
                                     std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        value_type_ observed = load(std::memory_order_acquire);
        while (observed <= limit && limit - observed >= operand &&
               !compare_exchange_strong(observed, static_cast<value_type_>(observed + operand), order,
                                        std::memory_order_acquire)) {}
        return observed;
    }
    value_type_ fetch_add_if_at_least(value_type_ operand, value_type_ floor,
                                      std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        value_type_ bound;
        value_type_ observed = load(std::memory_order_acquire);
        if (!fetch_add_if_at_least_bound(operand, floor, bound)) return observed;
        while (observed >= bound && !compare_exchange_strong(observed, static_cast<value_type_>(observed + operand),
                                                             order, std::memory_order_acquire)) {}
        return observed;
    }
    value_type_ fetch_sub_if_at_least(value_type_ operand, value_type_ floor,
                                      std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        value_type_ observed = load(std::memory_order_acquire);
        while (observed >= floor && observed - floor >= operand &&
               !compare_exchange_strong(observed, static_cast<value_type_>(observed - operand), order,
                                        std::memory_order_acquire)) {}
        return observed;
    }

  private:
    /** The previous rung, built on demand: it holds this same pointer and nothing else. */
    riscv64_zacas_atomic_ref<value_type_> weaker_() const noexcept {
        return riscv64_zacas_atomic_ref<value_type_>(*reinterpret_cast<value_type_ *>(word_));
    }

    /** One byte or halfword @c amo: signed operands travel sign-extended, the answer comes back
     *  truncated. */
    template <std::uint32_t byte_instruction_, std::uint32_t halfword_instruction_>
    value_type_ amo_(value_type_ operand) const noexcept {
        std::uint64_t register_operand;
        word_t const bits = std::bit_cast<word_t>(operand);
        if constexpr (std::numeric_limits<value_type_>::is_signed)
            register_operand = static_cast<std::uint64_t>(static_cast<std::int64_t>(signed_bits_(bits)));
        else register_operand = bits;
        constexpr std::uint32_t instruction_k = sizeof(value_type_) == 1 ? byte_instruction_ : halfword_instruction_;
        return std::bit_cast<value_type_>(
            static_cast<word_t>(riscv64_zabha_amo<instruction_k>(word_, register_operand, std::uint64_t {0})));
    }

    word_t *word_;
};

/**
 *  @brief Read-only access through @ref riscv64_zabha_atomic_ref, which specializes no load.
 *  @sa standard_atomic_ref<value_type_ const>
 */
template <typename value_type_>
struct riscv64_zabha_atomic_ref<value_type_ const> {
    using value_t = value_type_ const;
    static constexpr capabilities_t capabilities_k =
        capability_riscv64_atomic_k | capability_riscv64_zacas_k | capability_riscv64_zabha_k;

    explicit riscv64_zabha_atomic_ref(value_type_ const &word) noexcept : portable_(word) {}

    /** A temporary would die before the reference does. */
    riscv64_zabha_atomic_ref(value_type_ const &&) = delete;

    value_type_ load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return portable_.load(order);
    }

  private:
    riscv64_atomic_ref<value_type_ const> portable_;
};

#pragma endregion RISCV64 Zabha

#endif // FORKUNION_TARGET_RISCV64_ZABHA

/*  @c memory.atomic.notify: wakes the threads @c memory.atomic.wait sleeps on one 4-byte-aligned
 *  address, all of them for a count of `~0u`. No store wakes a sleeper in WebAssembly, so the
 *  reference below notifies after each of its releasing writes. */
#if FORKUNION_TARGET_WASM_WAIT
#pragma region WebAssembly

inline void wasm_memory_atomic_notify(void *word) noexcept {
    __builtin_wasm_memory_atomic_notify(static_cast<int *>(word), ~0u);
}

/**
 *  @brief The standard operations for WebAssembly threads, each store and read-modify-write ordered
 *      release, acq_rel or seq_cst on a 32- or 64-bit word followed by @c memory.atomic.notify,
 *      waking every thread @c wasm_wait_t sleeps on that word.
 *
 *  A compare-exchange notifies only when it wrote. Acquire and relaxed operations notify nothing,
 *  and neither do 1- and 2-byte words: @c wasm_wait_t spins on those, and @c memory.atomic.notify
 *  needs a 4-byte-aligned address.
 *
 *  @sa wasm_wait_t, the waiter it wakes.
 */
template <typename value_type_>
struct wasm_atomic_ref {
    using value_t = value_type_;
    static constexpr capabilities_t capabilities_k = capabilities_unknown_k;

    explicit wasm_atomic_ref(value_type_ &word) noexcept : word_(&word) {}

    value_type_ load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return weaker_().load(order);
    }
    void store(value_type_ desired, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        weaker_().store(desired, order);
        notify_(order);
    }
    value_type_ exchange(value_type_ desired, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        value_type_ const observed = weaker_().exchange(desired, order);
        notify_(order);
        return observed;
    }

    bool compare_exchange_strong(value_type_ &expected, value_type_ desired, std::memory_order success,
                                 std::memory_order failure) const noexcept {
        bool const exchanged = weaker_().compare_exchange_strong(expected, desired, success, failure);
        if (exchanged) notify_(success);
        return exchanged;
    }
    bool compare_exchange_weak(value_type_ &expected, value_type_ desired, std::memory_order success,
                               std::memory_order failure) const noexcept {
        bool const exchanged = weaker_().compare_exchange_weak(expected, desired, success, failure);
        if (exchanged) notify_(success);
        return exchanged;
    }
    bool compare_exchange_strong(value_type_ &expected, value_type_ desired,
                                 std::memory_order order = std::memory_order_seq_cst) const noexcept {
        bool const exchanged = weaker_().compare_exchange_strong(expected, desired, order);
        if (exchanged) notify_(order);
        return exchanged;
    }
    bool compare_exchange_weak(value_type_ &expected, value_type_ desired,
                               std::memory_order order = std::memory_order_seq_cst) const noexcept {
        bool const exchanged = weaker_().compare_exchange_weak(expected, desired, order);
        if (exchanged) notify_(order);
        return exchanged;
    }

    value_type_ fetch_add(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        value_type_ const observed = weaker_().fetch_add(operand, order);
        notify_(order);
        return observed;
    }
    value_type_ fetch_sub(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        value_type_ const observed = weaker_().fetch_sub(operand, order);
        notify_(order);
        return observed;
    }
    value_type_ fetch_and(value_type_ mask, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        value_type_ const observed = weaker_().fetch_and(mask, order);
        notify_(order);
        return observed;
    }
    value_type_ fetch_or(value_type_ bits, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        value_type_ const observed = weaker_().fetch_or(bits, order);
        notify_(order);
        return observed;
    }
    value_type_ fetch_xor(value_type_ bits, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        value_type_ const observed = weaker_().fetch_xor(bits, order);
        notify_(order);
        return observed;
    }
    value_type_ fetch_max(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> || atomic_float<value_type_>
    {
        value_type_ const observed = weaker_().fetch_max(operand, order);
        notify_(order);
        return observed;
    }
    value_type_ fetch_min(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> || atomic_float<value_type_>
    {
        value_type_ const observed = weaker_().fetch_min(operand, order);
        notify_(order);
        return observed;
    }

    value_type_ fetch_add_if_at_most(value_type_ operand, value_type_ limit,
                                     std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        value_type_ const observed = weaker_().fetch_add_if_at_most(operand, limit, order);
        notify_(order);
        return observed;
    }
    value_type_ fetch_add_if_at_least(value_type_ operand, value_type_ floor,
                                      std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        value_type_ const observed = weaker_().fetch_add_if_at_least(operand, floor, order);
        notify_(order);
        return observed;
    }
    value_type_ fetch_sub_if_at_least(value_type_ operand, value_type_ floor,
                                      std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        value_type_ const observed = weaker_().fetch_sub_if_at_least(operand, floor, order);
        notify_(order);
        return observed;
    }

    void add(value_type_ operand, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        weaker_().add(operand, order);
        notify_(order);
    }
    void sub(value_type_ operand, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        weaker_().sub(operand, order);
        notify_(order);
    }
    void set_bits(value_type_ bits, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        weaker_().set_bits(bits, order);
        notify_(order);
    }
    void clear_bits(value_type_ bits, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        weaker_().clear_bits(bits, order);
        notify_(order);
    }
    void flip_bits(value_type_ bits, std::memory_order order = std::memory_order_release) const noexcept
        requires atomic_integer<value_type_>
    {
        weaker_().flip_bits(bits, order);
        notify_(order);
    }
    bool fetch_test_and_set_bit(unsigned bit, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        bool const was_set = weaker_().fetch_test_and_set_bit(bit, order);
        notify_(order);
        return was_set;
    }
    bool fetch_test_and_reset_bit(unsigned bit, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        bool const was_set = weaker_().fetch_test_and_reset_bit(bit, order);
        notify_(order);
        return was_set;
    }
    void max(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> || atomic_float<value_type_>
    {
        weaker_().max(operand, order);
        notify_(order);
    }
    void min(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> || atomic_float<value_type_>
    {
        weaker_().min(operand, order);
        notify_(order);
    }

  private:
    /** The previous rung, built on demand: it holds this same pointer and nothing else. */
    standard_atomic_ref<value_type_> weaker_() const noexcept { return standard_atomic_ref<value_type_>(*word_); }

    /** Wakes the word's sleepers after a write ordered @p order, if that order releases. */
    void notify_(std::memory_order order) const noexcept {
        if ((sizeof(value_type_) == 4 || sizeof(value_type_) == 8) && releases(order)) wasm_memory_atomic_notify(word_);
    }

    value_type_ *word_;
};

/**
 *  @brief Read-only access through @ref wasm_atomic_ref, whose loads notify nothing.
 *  @sa standard_atomic_ref<value_type_ const>
 */
template <typename value_type_>
struct wasm_atomic_ref<value_type_ const> {
    using value_t = value_type_ const;
    static constexpr capabilities_t capabilities_k = capabilities_unknown_k;

    explicit wasm_atomic_ref(value_type_ const &word) noexcept : portable_(word) {}

    /** A temporary would die before the reference does. */
    wasm_atomic_ref(value_type_ const &&) = delete;

    value_type_ load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return portable_.load(order);
    }

  private:
    standard_atomic_ref<value_type_ const> portable_;
};

#pragma endregion WebAssembly

#endif // FORKUNION_TARGET_WASM_WAIT

/** The newest reference this translation unit may run with no runtime probe, reading each rung's
 *  <tt>FORKUNION_TARGET_\<CAPABILITY\></tt> alone - in a unit that dispatches nothing the compilation
 *  target's promise, so the pick can never be illegal there. In a unit that dispatches at runtime -
 *  one with the probe verdicts or @c FORKUNION_RUNTIME_DISPATCH - the bit is what the toolchain
 *  builds and the alias resolves to the newest buildable rung, so such a unit names its reference
 *  per CPU class instead. */
#if FORKUNION_TARGET_ARM64_RCPC
template <typename value_type_>
using preferred_atomic_ref = arm64_rcpc_atomic_ref<value_type_>;
#elif FORKUNION_TARGET_ARM64_LSE
template <typename value_type_>
using preferred_atomic_ref = arm64_lse_atomic_ref<value_type_>;
#elif FORKUNION_TARGET_X86_RAOINT
template <typename value_type_>
using preferred_atomic_ref = x86_raoint_atomic_ref<value_type_>;
#elif FORKUNION_TARGET_X86_CMPCCXADD
template <typename value_type_>
using preferred_atomic_ref = x86_cmpccxadd_atomic_ref<value_type_>;
#elif FORKUNION_TARGET_RISCV64_ZABHA
template <typename value_type_>
using preferred_atomic_ref = riscv64_zabha_atomic_ref<value_type_>;
#elif FORKUNION_TARGET_RISCV64_ZACAS
template <typename value_type_>
using preferred_atomic_ref = riscv64_zacas_atomic_ref<value_type_>;
#elif FORKUNION_TARGET_RISCV64_ATOMIC
template <typename value_type_>
using preferred_atomic_ref = riscv64_atomic_ref<value_type_>;
#elif FORKUNION_TARGET_WASM_WAIT
template <typename value_type_>
using preferred_atomic_ref = wasm_atomic_ref<value_type_>;
#else
template <typename value_type_>
using preferred_atomic_ref = standard_atomic_ref<value_type_>;
#endif

#endif // __cpp_lib_atomic_ref && __cpp_lib_bit_cast

} // namespace forkunion
} // namespace ashvardanian
