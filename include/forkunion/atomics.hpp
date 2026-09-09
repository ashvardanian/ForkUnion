/**
 *  @brief `std::atomic_ref` replacements that spell every access as the one instruction a CPU
 *      generation has for it, independent of the translation unit's target flags.
 *
 *  @author Ash Vardanian
 *  @file include/forkunion/atomics.hpp
 *  @date September 5, 2026
 *
 *  `standard_atomic_ref` is the in-house `std::atomic_ref`: the standard operations verbatim plus
 *  the ones the standard lacks - `fetch_max`/`fetch_min` ahead of C++26, the no-return `add`/`sub`/
 *  `set`/`clear`, and the conditional `fetch_add_if_at_most`/`fetch_sub_if_at_least` - spelled
 *  portably over compare-exchange. The instruction-set references share that interface and replace
 *  the loops and the compiler's flag-dependent lowering with instructions:
 *
 *  - `x86_cmpccxadd_atomic_ref`: the conditional adds as one `cmpccxadd`; every other operation
 *    is the `lock`-prefixed instruction the compiler emits anyway.
 *  - `x86_raoint_atomic_ref`: the relaxed no-return forms as RAO-INT `aadd`/`aand`/`aor`, executed
 *    at the shared cache rather than pulling the line.
 *  - `arm64_lse_atomic_ref`: Armv8.1 `swp`, `cas`, `ldadd`, `ldclr`, `ldset`, `ldeor`, `ldsmax`
 *    & kin - one instruction per read-modify-write, the no-return `st*` forms posted without a
 *    round trip; `ldar`/`stlr` for the ordered loads & stores. Measured on an M5 Pro against the
 *    exclusive loops a baseline build gets: 5x uncontended, 3-4x with 18 threads on one word.
 *  - `arm64_rcpc_atomic_ref`: the above with Armv8.3 `ldapr` for acquiring loads - RCpc, which
 *    needn't wait for the core's earlier release stores the way RCsc `ldar` may.
 *  - `risc5_atomic_ref`: the base A extension - `amoswap`, `amoadd`, `amoand`, `amoor`, `amomax`,
 *    `amomin` & the unsigned twins, with `x0` as the destination for the no-return forms - and
 *    `lr`/`sc` loops for compare-exchange and byte exchanges; fences around loads & stores.
 *  - `risc5_zacas_atomic_ref`: compare-exchange as one `amocas`.
 *
 *  Only the widths & operations the indexes use are spelled: byte exchanges & compare-exchanges
 *  for flags, 32-bit forms for node ids, 64-bit forms for counters & packed words. Every reference
 *  assembles in a baseline translation unit: on Arm the extension is named in the assembly text
 *  - `.arch_extension` - the RISC-V base atomics are the A-extension mnemonics every `rv64gc`
 *  toolchain assembles, and the x86 and RISC-V extension instructions are raw bytes; the runtime
 *  capability bit decides whether it may run. Where inline assembly is unavailable - MSVC - only
 *  `standard_atomic_ref` remains. `preferred_atomic_ref`, at the bottom, is the newest reference
 *  the build target guarantees through its predefined macros - the `preferred_yield_t` rule - for
 *  callers picking at compile time rather than per CPU class.
 *
 *  The header needs the library's `std::atomic_ref` and `std::bit_cast`, so it is empty without
 *  them - a C++17 translation unit including the umbrella sees nothing here.
 */
#pragma once
#include <cstdint> // `std::uint8_t`, `std::uint32_t`, `std::uint64_t`, their signed twins

#include <atomic>      // `std::memory_order`
#include <bit>         // `std::bit_cast`
#include <concepts>    // `std::integral`, `std::signed_integral`, `std::same_as`
#include <type_traits> // `std::conditional_t`, `std::is_trivially_copyable_v`

#include "types.hpp" // `capabilities_t`, the bits a reference needs admitted, and the target macros

namespace ashvardanian {
namespace forkunion {

#if defined(__cpp_lib_atomic_ref) && defined(__cpp_lib_bit_cast)

/** The words arithmetic read-modify-writes apply to: integers other than `bool`, as the
 *  standard's integral `atomic_ref` specialization draws the line - the instructions add bits. */
template <typename value_type_>
concept atomic_integer = std::integral<value_type_> && !std::same_as<value_type_, bool>;

/** The unsigned word a value of one width travels through the instructions as. */
template <typename value_type_>
using atomic_word = std::conditional_t<sizeof(value_type_) == 1, std::uint8_t,
                                       std::conditional_t<sizeof(value_type_) == 4, std::uint32_t, std::uint64_t>>;

/** Whether an order carries acquire or release semantics - the two bits every ISA's mnemonics encode. */
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

/**
 *  @brief `std::atomic_ref` with the operations the standard lacks, spelled the portable way:
 *      compare-exchange loops for the conditional and extremal read-modify-writes, discarded
 *      results for the no-return ones - which the compiler lowers to the store-only instruction
 *      where the target has it. The instruction-set references below share this interface and
 *      replace the loops with `ldsmax`, `stadd`, `cmpccxadd` and kin.
 *
 *  Fences can't stand in for any of these: a fence orders accesses, it doesn't make a
 *  read-compare-write atomic. What makes the loops cheap is reading first - a word that already
 *  satisfies the condition returns without writing, so losers never take the line.
 */
template <typename value_type_>
struct standard_atomic_ref : public std::atomic_ref<value_type_> {
    using base_t = std::atomic_ref<value_type_>;
    using base_t::base_t;
    /** The runtime bits a reference needs admitted before it may run - none here. */
    static constexpr capabilities_t capabilities_k = capabilities_unknown_k;

    /** C++26's `fetch_max`: the value held before, whether or not the operand replaced it. */
    value_type_ fetch_max(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        value_type_ observed = base_t::load(std::memory_order_acquire);
        while (observed < operand &&
               !base_t::compare_exchange_weak(observed, operand, order, std::memory_order_acquire)) {}
        return observed;
    }
    value_type_ fetch_min(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        value_type_ observed = base_t::load(std::memory_order_acquire);
        while (observed > operand &&
               !base_t::compare_exchange_weak(observed, operand, order, std::memory_order_acquire)) {}
        return observed;
    }

    /** Adds @p operand only if the sum stays at most @p limit - one `cmpccxadd` on Intel, a
     *  read-first compare-exchange loop elsewhere. Returns the value held before; the caller
     *  learns the outcome from `observed + operand <= limit`. An operand past the limit, or a
     *  floor the operand cannot be taken from, merely observes the word - nothing is required of
     *  either, beyond a signed sum or difference that stays representable. */
    value_type_ fetch_add_if_at_most(value_type_ operand, value_type_ limit,
                                     std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        value_type_ observed = base_t::load(std::memory_order_acquire);
        while (observed <= limit && limit - observed >= operand &&
               !base_t::compare_exchange_weak(observed, observed + operand, order, std::memory_order_acquire)) {}
        return observed;
    }
    /** Subtracts @p operand only if the difference stays at least @p floor - the semaphore
     *  acquire. Returns the value held before; the outcome is `observed >= floor + operand`. */
    value_type_ fetch_sub_if_at_least(value_type_ operand, value_type_ floor,
                                      std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        value_type_ observed = base_t::load(std::memory_order_acquire);
        while (observed >= floor && observed - floor >= operand &&
               !base_t::compare_exchange_weak(observed, observed - operand, order, std::memory_order_acquire)) {}
        return observed;
    }

    /** No-return read-modify-writes: the op is posted, nothing is waited for - `stadd`, `stclr`,
     *  `stset` on Arm, `lock add` on x86, the remote `aadd` family with RAO-INT. Only relaxed
     *  and release orders exist for them: with no value returned there is nothing to acquire. */
    void add(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        (void)base_t::fetch_add(operand, order);
    }
    void sub(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        (void)base_t::fetch_sub(operand, order);
    }
    void set(value_type_ bits, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        (void)base_t::fetch_or(bits, order);
    }
    void clear(value_type_ bits, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        (void)base_t::fetch_and(static_cast<value_type_>(~bits), order);
    }
};

#if FU_TARGET_X86_CMPCCXADD

#pragma region x86 CMPCCXADD

/*  `cmpccxadd`: compares the word against `bound` - flags from `word - bound` - and adds `addend`
 *  only when the condition holds; the register handed as `bound` receives what the word held.
 *  Spelled as bytes, since binutils before 2.40 and LLVM before 16 have no mnemonic: the VEX form
 *  in map 0F38, opcode E0 plus the condition, `W` set for the 64-bit forms; the word in `rax`, the
 *  compare-and-return register in `rcx`, the addend in `rdx`. */

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

/**
 *  @brief The standard reference with the conditional adds as one `cmpccxadd` - every other
 *      operation is already the `lock`-prefixed instruction the compiler emits. Intel cores from
 *      the 2024 E-core Xeons on; `CPUID.(7,1):EAX[7]` says so at runtime. The standard reference
 *      hides its pointer, so the word's address is kept alongside.
 *  @sa `capability_x86_cmpccxadd_k` - the bit admitting it; `x86_raoint_atomic_ref` - the same with RAO-INT.
 */
template <typename value_type_>
struct x86_cmpccxadd_atomic_ref : public standard_atomic_ref<value_type_> {
    using base_t = standard_atomic_ref<value_type_>;
    using word_t = atomic_word<value_type_>;
    static constexpr capabilities_t capabilities_k = capability_x86_cmpccxadd_k;

    explicit x86_cmpccxadd_atomic_ref(value_type_ &word) noexcept : base_t(word), word_(&word) {}

    value_type_ fetch_add_if_at_most(value_type_ operand, value_type_ limit,
                                     std::memory_order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> && (sizeof(value_type_) == 4 || sizeof(value_type_) == 8)
    {
        // Adds while `word <= limit - operand`: below-or-equal unsigned, less-or-equal signed.
        if (operand > limit) return base_t::load(std::memory_order_acquire); // ? Nothing could be admitted
        value_type_ const bound = static_cast<value_type_>(limit - operand);
        if constexpr (std::signed_integral<value_type_> && sizeof(value_type_) == 4)
            return x86_cmplexadd_i32(reinterpret_cast<std::int32_t *>(word_), bound, operand);
        else if constexpr (std::signed_integral<value_type_>)
            return x86_cmplexadd_i64(reinterpret_cast<std::int64_t *>(word_), bound, operand);
        else if constexpr (sizeof(value_type_) == 4)
            return x86_cmpbexadd_u32(reinterpret_cast<std::uint32_t *>(word_), bound, operand);
        else return x86_cmpbexadd_u64(reinterpret_cast<std::uint64_t *>(word_), bound, operand);
    }
    value_type_ fetch_sub_if_at_least(value_type_ operand, value_type_ floor,
                                      std::memory_order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> && (sizeof(value_type_) == 4 || sizeof(value_type_) == 8)
    {
        // Subtracts while `word >= floor + operand`: above-or-equal unsigned, greater-or-equal signed.
        value_type_ const bound = static_cast<value_type_>(floor + operand);
        if (bound < floor)
            return base_t::load(std::memory_order_acquire); // ? The bound wrapped: nothing could be taken
        value_type_ const negated = static_cast<value_type_>(value_type_ {0} - operand);
        if constexpr (std::signed_integral<value_type_> && sizeof(value_type_) == 4)
            return x86_cmpgexadd_i32(reinterpret_cast<std::int32_t *>(word_), bound, negated);
        else if constexpr (std::signed_integral<value_type_>)
            return x86_cmpgexadd_i64(reinterpret_cast<std::int64_t *>(word_), bound, negated);
        else if constexpr (sizeof(value_type_) == 4)
            return x86_cmpaexadd_u32(reinterpret_cast<std::uint32_t *>(word_), bound, negated);
        else return x86_cmpaexadd_u64(reinterpret_cast<std::uint64_t *>(word_), bound, negated);
    }

  protected:
    value_type_ *word_;
};

#pragma endregion x86 CMPCCXADD

#endif // FU_TARGET_X86_CMPCCXADD

#if FU_TARGET_X86_RAOINT

#pragma region x86 RAOINT

/*  RAO-INT: the remote, no-return forms - weakly ordered, so only the relaxed callers take them.
 *  Bytes for the same reason: map 0F38 opcode FC, the operation picked by the legacy prefix - none
 *  for add, 66 for and, F2 for or - `REX.W` for the 64-bit forms; the word in `rax`, the operand
 *  in `rcx`. */

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

/**
 *  @brief The above plus RAO-INT for the relaxed no-return forms: `aadd`, `aand`, `aor` execute
 *      at the shared cache. Anything ordered keeps the `lock`-prefixed base, already a full
 *      fence. `CPUID.(7,1):EAX[3]` says so at runtime.
 *  @sa `capability_x86_raoint_k` - the bit admitting it, on top of `capability_x86_cmpccxadd_k`.
 */
template <typename value_type_>
struct x86_raoint_atomic_ref : public x86_cmpccxadd_atomic_ref<value_type_> {
    using base_t = x86_cmpccxadd_atomic_ref<value_type_>;
    using base_t::base_t;
    using typename base_t::word_t;
    static constexpr capabilities_t capabilities_k = capability_x86_cmpccxadd_k | capability_x86_raoint_k;

    void add(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> && (sizeof(value_type_) == 4 || sizeof(value_type_) == 8)
    {
        if (order != std::memory_order_relaxed) return base_t::add(operand, order);
        word_t *word = reinterpret_cast<word_t *>(this->word_);
        if constexpr (sizeof(value_type_) == 4) x86_aadd_u32(word, std::bit_cast<word_t>(operand));
        else x86_aadd_u64(word, std::bit_cast<word_t>(operand));
    }
    void sub(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> && (sizeof(value_type_) == 4 || sizeof(value_type_) == 8)
    {
        add(static_cast<value_type_>(value_type_ {0} - operand), order);
    }
    void set(value_type_ bits, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> && (sizeof(value_type_) == 4 || sizeof(value_type_) == 8)
    {
        if (order != std::memory_order_relaxed) return base_t::set(bits, order);
        word_t *word = reinterpret_cast<word_t *>(this->word_);
        if constexpr (sizeof(value_type_) == 4) x86_aor_u32(word, std::bit_cast<word_t>(bits));
        else x86_aor_u64(word, std::bit_cast<word_t>(bits));
    }
    void clear(value_type_ bits, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_> && (sizeof(value_type_) == 4 || sizeof(value_type_) == 8)
    {
        if (order != std::memory_order_relaxed) return base_t::clear(bits, order);
        word_t *word = reinterpret_cast<word_t *>(this->word_);
        word_t const mask = static_cast<word_t>(~std::bit_cast<word_t>(bits));
        if constexpr (sizeof(value_type_) == 4) x86_aand_u32(word, mask);
        else x86_aand_u64(word, mask);
    }
};

#pragma endregion x86 RAOINT

#endif // FU_TARGET_X86_RAOINT

#if FU_TARGET_ARM64_LSE

#pragma region Arm64 LSE

/*  Loads: plain and acquire. */

inline std::uint8_t arm64_ldr_u8(std::uint8_t const *word) noexcept {
    std::uint8_t value;
    __asm__ __volatile__("ldrb %w0, [%1]" : "=r"(value) : "r"(word) : "memory");
    return value;
}
inline std::uint32_t arm64_ldr_u32(std::uint32_t const *word) noexcept {
    std::uint32_t value;
    __asm__ __volatile__("ldr %w0, [%1]" : "=r"(value) : "r"(word) : "memory");
    return value;
}
inline std::uint64_t arm64_ldr_u64(std::uint64_t const *word) noexcept {
    std::uint64_t value;
    __asm__ __volatile__("ldr %x0, [%1]" : "=r"(value) : "r"(word) : "memory");
    return value;
}

inline std::uint8_t arm64_ldar_u8(std::uint8_t const *word) noexcept {
    std::uint8_t value;
    __asm__ __volatile__("ldarb %w0, [%1]" : "=r"(value) : "r"(word) : "memory");
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

inline void arm64_str_u8(std::uint8_t *word, std::uint8_t value) noexcept {
    __asm__ __volatile__("strb %w0, [%1]" : : "r"(value), "r"(word) : "memory");
}
inline void arm64_str_u32(std::uint32_t *word, std::uint32_t value) noexcept {
    __asm__ __volatile__("str %w0, [%1]" : : "r"(value), "r"(word) : "memory");
}
inline void arm64_str_u64(std::uint64_t *word, std::uint64_t value) noexcept {
    __asm__ __volatile__("str %x0, [%1]" : : "r"(value), "r"(word) : "memory");
}

inline void arm64_stlr_u8(std::uint8_t *word, std::uint8_t value) noexcept {
    __asm__ __volatile__("stlrb %w0, [%1]" : : "r"(value), "r"(word) : "memory");
}
inline void arm64_stlr_u32(std::uint32_t *word, std::uint32_t value) noexcept {
    __asm__ __volatile__("stlr %w0, [%1]" : : "r"(value), "r"(word) : "memory");
}
inline void arm64_stlr_u64(std::uint64_t *word, std::uint64_t value) noexcept {
    __asm__ __volatile__("stlr %x0, [%1]" : : "r"(value), "r"(word) : "memory");
}

/*  LSE read-modify-writes: `ws` is the operand, `wt` receives what the word held; the order picks
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

/** `ldclr` clears the operand's bits: a word's `fetch_and(mask)` is `ldclr ~mask`. */
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
inline void arm64_stclr_u64(std::uint64_t *word, std::uint64_t bits, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__(".arch_extension lse\n\tstclr %x0, [%1]" : : "r"(bits), "r"(word) : "memory");
    else __asm__ __volatile__(".arch_extension lse\n\tstclrl %x0, [%1]" : : "r"(bits), "r"(word) : "memory");
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

/*  LSE maxima & minima - signed and unsigned are different instructions, the width is the register. */

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

/** `cas` compares against `expected` and returns what the word held; equality means it swapped. */
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

/**
 *  @brief `std::atomic_ref` over Armv8.1 LSE: `ldar`/`stlr` for the ordered loads & stores, one
 *      instruction per read-modify-write in the acquire/release flavor the order asks for. Same
 *      discipline as the standard reference: the word is naturally aligned and never touched
 *      non-atomically while references exist. Widths & operations not spelled above fail to
 *      compile rather than fall back.
 *  @sa `capability_arm64_lse_k` - the bit admitting it; `arm64_rcpc_atomic_ref` - the same with RCpc loads.
 */
template <typename value_type_>
struct arm64_lse_atomic_ref {
    static_assert(std::is_trivially_copyable_v<value_type_>, "Only trivially-copyable words are atomic");
    static_assert(sizeof(value_type_) == 1 || sizeof(value_type_) == 4 || sizeof(value_type_) == 8,
                  "Only byte, 32-bit and 64-bit words are spelled");
    using value_t = value_type_;
    using word_t = atomic_word<value_type_>;
    static constexpr capabilities_t capabilities_k = capability_arm64_lse_k;

    explicit arm64_lse_atomic_ref(value_type_ &word) noexcept : word_(reinterpret_cast<word_t *>(&word)) {}

    value_type_ load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
        if (order == std::memory_order_relaxed) {
            if constexpr (sizeof(value_type_) == 1) return std::bit_cast<value_type_>(arm64_ldr_u8(word_));
            else if constexpr (sizeof(value_type_) == 4) return std::bit_cast<value_type_>(arm64_ldr_u32(word_));
            else return std::bit_cast<value_type_>(arm64_ldr_u64(word_));
        }
        if constexpr (sizeof(value_type_) == 1) return std::bit_cast<value_type_>(arm64_ldar_u8(word_));
        else if constexpr (sizeof(value_type_) == 4) return std::bit_cast<value_type_>(arm64_ldar_u32(word_));
        else return std::bit_cast<value_type_>(arm64_ldar_u64(word_));
    }

    void store(value_type_ desired, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        word_t const word = std::bit_cast<word_t>(desired);
        if (order == std::memory_order_relaxed) {
            if constexpr (sizeof(value_type_) == 1) arm64_str_u8(word_, word);
            else if constexpr (sizeof(value_type_) == 4) arm64_str_u32(word_, word);
            else arm64_str_u64(word_, word);
        }
        else {
            if constexpr (sizeof(value_type_) == 1) arm64_stlr_u8(word_, word);
            else if constexpr (sizeof(value_type_) == 4) arm64_stlr_u32(word_, word);
            else arm64_stlr_u64(word_, word);
        }
    }

    value_type_ exchange(value_type_ desired, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        word_t const word = std::bit_cast<word_t>(desired);
        if constexpr (sizeof(value_type_) == 1) return std::bit_cast<value_type_>(arm64_swp_u8(word_, word, order));
        else if constexpr (sizeof(value_type_) == 4)
            return std::bit_cast<value_type_>(arm64_swp_u32(word_, word, order));
        else return std::bit_cast<value_type_>(arm64_swp_u64(word_, word, order));
    }

    /** One `cas` runs in the stronger of the two orders whatever the outcome; a failed compare
     *  is then an acquiring load of the observed word, which the standard permits. */
    bool compare_exchange_strong(value_type_ &expected, value_type_ desired, std::memory_order success,
                                 std::memory_order failure) const noexcept {
        word_t const wanted = std::bit_cast<word_t>(expected);
        word_t const word = std::bit_cast<word_t>(desired);
        std::memory_order const order = stronger_(success, failure);
        word_t observed;
        if constexpr (sizeof(value_type_) == 1) observed = arm64_cas_u8(word_, wanted, word, order);
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
        static_assert(sizeof(value_type_) != 1, "Byte fetch-add is not spelled yet");
        word_t const word = std::bit_cast<word_t>(operand);
        if constexpr (sizeof(value_type_) == 4) return std::bit_cast<value_type_>(arm64_ldadd_u32(word_, word, order));
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
        static_assert(sizeof(value_type_) != 1, "Byte fetch-and is not spelled yet");
        word_t const cleared = static_cast<word_t>(~std::bit_cast<word_t>(mask));
        if constexpr (sizeof(value_type_) == 4)
            return std::bit_cast<value_type_>(arm64_ldclr_u32(word_, cleared, order));
        else return std::bit_cast<value_type_>(arm64_ldclr_u64(word_, cleared, order));
    }
    value_type_ fetch_or(value_type_ bits, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        static_assert(sizeof(value_type_) != 1, "Byte fetch-or is not spelled yet");
        word_t const word = std::bit_cast<word_t>(bits);
        if constexpr (sizeof(value_type_) == 4) return std::bit_cast<value_type_>(arm64_ldset_u32(word_, word, order));
        else return std::bit_cast<value_type_>(arm64_ldset_u64(word_, word, order));
    }
    value_type_ fetch_xor(value_type_ bits, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        static_assert(sizeof(value_type_) != 1, "Byte fetch-xor is not spelled yet");
        word_t const word = std::bit_cast<word_t>(bits);
        if constexpr (sizeof(value_type_) == 4) return std::bit_cast<value_type_>(arm64_ldeor_u32(word_, word, order));
        else return std::bit_cast<value_type_>(arm64_ldeor_u64(word_, word, order));
    }

    /** Ahead of C++26: one `ldsmax`/`ldumax`, the signedness picking the instruction. */
    value_type_ fetch_max(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        static_assert(sizeof(value_type_) != 1, "Byte maxima are not spelled yet");
        if constexpr (std::signed_integral<value_type_> && sizeof(value_type_) == 4)
            return static_cast<value_type_>(arm64_ldsmax_i32(reinterpret_cast<std::int32_t *>(word_), operand, order));
        else if constexpr (std::signed_integral<value_type_>)
            return static_cast<value_type_>(arm64_ldsmax_i64(reinterpret_cast<std::int64_t *>(word_), operand, order));
        else if constexpr (sizeof(value_type_) == 4)
            return static_cast<value_type_>(arm64_ldumax_u32(word_, operand, order));
        else return static_cast<value_type_>(arm64_ldumax_u64(word_, operand, order));
    }
    value_type_ fetch_min(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        static_assert(sizeof(value_type_) != 1, "Byte minima are not spelled yet");
        if constexpr (std::signed_integral<value_type_> && sizeof(value_type_) == 4)
            return static_cast<value_type_>(arm64_ldsmin_i32(reinterpret_cast<std::int32_t *>(word_), operand, order));
        else if constexpr (std::signed_integral<value_type_>)
            return static_cast<value_type_>(arm64_ldsmin_i64(reinterpret_cast<std::int64_t *>(word_), operand, order));
        else if constexpr (sizeof(value_type_) == 4)
            return static_cast<value_type_>(arm64_ldumin_u32(word_, operand, order));
        else return static_cast<value_type_>(arm64_ldumin_u64(word_, operand, order));
    }

    /** No-return forms: `stadd`, `stclr`, `stset` - posted, nothing waited for. */
    void add(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        static_assert(sizeof(value_type_) != 1, "Byte no-return add is not spelled yet");
        word_t const word = std::bit_cast<word_t>(operand);
        if constexpr (sizeof(value_type_) == 4) arm64_stadd_u32(word_, word, order);
        else arm64_stadd_u64(word_, word, order);
    }
    void sub(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        add(std::bit_cast<value_type_>(static_cast<word_t>(word_t {0} - std::bit_cast<word_t>(operand))), order);
    }
    void set(value_type_ bits, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        static_assert(sizeof(value_type_) != 1, "Byte no-return set is not spelled yet");
        word_t const word = std::bit_cast<word_t>(bits);
        if constexpr (sizeof(value_type_) == 4) arm64_stset_u32(word_, word, order);
        else arm64_stset_u64(word_, word, order);
    }
    void clear(value_type_ bits, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        static_assert(sizeof(value_type_) != 1, "Byte no-return clear is not spelled yet");
        word_t const word = std::bit_cast<word_t>(bits);
        if constexpr (sizeof(value_type_) == 4) arm64_stclr_u32(word_, word, order);
        else arm64_stclr_u64(word_, word, order);
    }

    /** The conditional forms: Arm has no `cmpccxadd`, so a read-first `cas` loop - one
     *  acquiring load, then one instruction per attempt. */
    value_type_ fetch_add_if_at_most(value_type_ operand, value_type_ limit,
                                     std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        value_type_ observed = load(std::memory_order_acquire);
        while (observed <= limit && limit - observed >= operand &&
               !compare_exchange_strong(observed, observed + operand, order, std::memory_order_acquire)) {}
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

  protected:
    word_t *word_;

    static constexpr std::memory_order stronger_(std::memory_order success, std::memory_order failure) noexcept {
        bool const acquire = acquires(success) || acquires(failure);
        bool const release = releases(success);
        if (acquire && release) return std::memory_order_acq_rel;
        if (acquire) return std::memory_order_acquire;
        if (release) return std::memory_order_release;
        return std::memory_order_relaxed;
    }
};

#pragma endregion Arm64 LSE

#endif // FU_TARGET_ARM64_LSE

#if FU_TARGET_ARM64_RCPC

#pragma region Arm64 RCpc

/*  RCpc acquire loads: ordered against later loads and stores, not against earlier stores. */

inline std::uint8_t arm64_ldapr_u8(std::uint8_t const *word) noexcept {
    std::uint8_t value;
    __asm__ __volatile__(".arch_extension rcpc\n\tldaprb %w0, [%1]" : "=r"(value) : "r"(word) : "memory");
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

/**
 *  @brief Armv8.3 RCpc on top of LSE: acquiring loads are `ldapr`. Sequentially-consistent
 *      loads keep `ldar` - the only form that composes with `stlr` into a total order.
 *  @sa `capability_arm64_rcpc_k` - the bit admitting it, on top of `capability_arm64_lse_k`.
 */
template <typename value_type_>
struct arm64_rcpc_atomic_ref : public arm64_lse_atomic_ref<value_type_> {
    using base_t = arm64_lse_atomic_ref<value_type_>;
    using base_t::base_t;
    static constexpr capabilities_t capabilities_k = capability_arm64_lse_k | capability_arm64_rcpc_k;

    value_type_ load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
        if (order != std::memory_order_acquire && order != std::memory_order_consume) return base_t::load(order);
        if constexpr (sizeof(value_type_) == 1) return std::bit_cast<value_type_>(arm64_ldapr_u8(this->word_));
        else if constexpr (sizeof(value_type_) == 4) return std::bit_cast<value_type_>(arm64_ldapr_u32(this->word_));
        else return std::bit_cast<value_type_>(arm64_ldapr_u64(this->word_));
    }
};

#pragma endregion Arm64 RCpc

#endif // FU_TARGET_ARM64_RCPC

#if FU_TARGET_RISC5_ATOMIC

#pragma region RISC5 A

/*  Loads & stores carry their order as fences, per the RISC-V mapping: acquire is `fence r,rw`
 *  after the load, release `fence rw,w` before the store, sequential consistency both. */

inline std::uint8_t risc5_lbu(std::uint8_t const *word) noexcept {
    std::uint8_t value;
    __asm__ __volatile__("lbu %0, 0(%1)" : "=r"(value) : "r"(word) : "memory");
    return value;
}
inline std::uint32_t risc5_lw(std::uint32_t const *word) noexcept {
    std::uint32_t value;
    __asm__ __volatile__("lw %0, 0(%1)" : "=r"(value) : "r"(word) : "memory");
    return value;
}
inline std::uint64_t risc5_ld(std::uint64_t const *word) noexcept {
    std::uint64_t value;
    __asm__ __volatile__("ld %0, 0(%1)" : "=r"(value) : "r"(word) : "memory");
    return value;
}
inline void risc5_sb(std::uint8_t *word, std::uint8_t value) noexcept {
    __asm__ __volatile__("sb %0, 0(%1)" : : "r"(value), "r"(word) : "memory");
}
inline void risc5_sw(std::uint32_t *word, std::uint32_t value) noexcept {
    __asm__ __volatile__("sw %0, 0(%1)" : : "r"(value), "r"(word) : "memory");
}
inline void risc5_sd(std::uint64_t *word, std::uint64_t value) noexcept {
    __asm__ __volatile__("sd %0, 0(%1)" : : "r"(value), "r"(word) : "memory");
}
inline void risc5_fence_r_rw() noexcept { __asm__ __volatile__("fence r, rw" : : : "memory"); }
inline void risc5_fence_rw_w() noexcept { __asm__ __volatile__("fence rw, w" : : : "memory"); }
inline void risc5_fence_rw_rw() noexcept { __asm__ __volatile__("fence rw, rw" : : : "memory"); }

/*  Atomic memory operations of the base A extension: `rd` receives what the word held; `.aqrl`
 *  serves every order but relaxed - a strengthening the ISA prices at nothing on the fast path. */

inline std::uint32_t risc5_amoswap_w(std::uint32_t *word, std::uint32_t desired, std::memory_order order) noexcept {
    std::uint32_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoswap.w %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(desired) : "memory");
    else __asm__ __volatile__("amoswap.w.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(desired) : "memory");
    return observed;
}
inline std::uint64_t risc5_amoswap_d(std::uint64_t *word, std::uint64_t desired, std::memory_order order) noexcept {
    std::uint64_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoswap.d %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(desired) : "memory");
    else __asm__ __volatile__("amoswap.d.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(desired) : "memory");
    return observed;
}
inline std::uint32_t risc5_amoadd_w(std::uint32_t *word, std::uint32_t operand, std::memory_order order) noexcept {
    std::uint32_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoadd.w %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amoadd.w.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    return observed;
}
inline std::uint64_t risc5_amoadd_d(std::uint64_t *word, std::uint64_t operand, std::memory_order order) noexcept {
    std::uint64_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoadd.d %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amoadd.d.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    return observed;
}
inline std::uint32_t risc5_amoand_w(std::uint32_t *word, std::uint32_t mask, std::memory_order order) noexcept {
    std::uint32_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoand.w %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(mask) : "memory");
    else __asm__ __volatile__("amoand.w.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(mask) : "memory");
    return observed;
}
inline std::uint64_t risc5_amoand_d(std::uint64_t *word, std::uint64_t mask, std::memory_order order) noexcept {
    std::uint64_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoand.d %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(mask) : "memory");
    else __asm__ __volatile__("amoand.d.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(mask) : "memory");
    return observed;
}
inline std::uint32_t risc5_amoor_w(std::uint32_t *word, std::uint32_t bits, std::memory_order order) noexcept {
    std::uint32_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoor.w %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(bits) : "memory");
    else __asm__ __volatile__("amoor.w.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(bits) : "memory");
    return observed;
}
inline std::uint64_t risc5_amoor_d(std::uint64_t *word, std::uint64_t bits, std::memory_order order) noexcept {
    std::uint64_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoor.d %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(bits) : "memory");
    else __asm__ __volatile__("amoor.d.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(bits) : "memory");
    return observed;
}
inline std::uint32_t risc5_amoxor_w(std::uint32_t *word, std::uint32_t bits, std::memory_order order) noexcept {
    std::uint32_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoxor.w %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(bits) : "memory");
    else __asm__ __volatile__("amoxor.w.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(bits) : "memory");
    return observed;
}
inline std::uint64_t risc5_amoxor_d(std::uint64_t *word, std::uint64_t bits, std::memory_order order) noexcept {
    std::uint64_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoxor.d %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(bits) : "memory");
    else __asm__ __volatile__("amoxor.d.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(bits) : "memory");
    return observed;
}
inline std::uint32_t risc5_amomaxu_w(std::uint32_t *word, std::uint32_t operand, std::memory_order order) noexcept {
    std::uint32_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amomaxu.w %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amomaxu.w.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    return observed;
}
inline std::uint64_t risc5_amomaxu_d(std::uint64_t *word, std::uint64_t operand, std::memory_order order) noexcept {
    std::uint64_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amomaxu.d %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amomaxu.d.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    return observed;
}
inline std::int32_t risc5_amomax_w(std::int32_t *word, std::int32_t operand, std::memory_order order) noexcept {
    std::int32_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amomax.w %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amomax.w.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    return observed;
}
inline std::int64_t risc5_amomax_d(std::int64_t *word, std::int64_t operand, std::memory_order order) noexcept {
    std::int64_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amomax.d %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amomax.d.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    return observed;
}
inline std::uint32_t risc5_amominu_w(std::uint32_t *word, std::uint32_t operand, std::memory_order order) noexcept {
    std::uint32_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amominu.w %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amominu.w.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    return observed;
}
inline std::uint64_t risc5_amominu_d(std::uint64_t *word, std::uint64_t operand, std::memory_order order) noexcept {
    std::uint64_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amominu.d %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amominu.d.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    return observed;
}
inline std::int32_t risc5_amomin_w(std::int32_t *word, std::int32_t operand, std::memory_order order) noexcept {
    std::int32_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amomin.w %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amomin.w.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    return observed;
}
inline std::int64_t risc5_amomin_d(std::int64_t *word, std::int64_t operand, std::memory_order order) noexcept {
    std::int64_t observed;
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amomin.d %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amomin.d.aqrl %0, %2, (%1)" : "=r"(observed) : "r"(word), "r"(operand) : "memory");
    return observed;
}

/*  No-return forms are the same operations with `x0` as the destination - the ISA's own hint
 *  that nothing waits for the value; release is `.rl`, nothing to acquire. */

inline void risc5_amoadd_w_x0(std::uint32_t *word, std::uint32_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoadd.w zero, %1, (%0)" : : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amoadd.w.rl zero, %1, (%0)" : : "r"(word), "r"(operand) : "memory");
}
inline void risc5_amoadd_d_x0(std::uint64_t *word, std::uint64_t operand, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoadd.d zero, %1, (%0)" : : "r"(word), "r"(operand) : "memory");
    else __asm__ __volatile__("amoadd.d.rl zero, %1, (%0)" : : "r"(word), "r"(operand) : "memory");
}
inline void risc5_amoand_w_x0(std::uint32_t *word, std::uint32_t mask, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoand.w zero, %1, (%0)" : : "r"(word), "r"(mask) : "memory");
    else __asm__ __volatile__("amoand.w.rl zero, %1, (%0)" : : "r"(word), "r"(mask) : "memory");
}
inline void risc5_amoand_d_x0(std::uint64_t *word, std::uint64_t mask, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoand.d zero, %1, (%0)" : : "r"(word), "r"(mask) : "memory");
    else __asm__ __volatile__("amoand.d.rl zero, %1, (%0)" : : "r"(word), "r"(mask) : "memory");
}
inline void risc5_amoor_w_x0(std::uint32_t *word, std::uint32_t bits, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoor.w zero, %1, (%0)" : : "r"(word), "r"(bits) : "memory");
    else __asm__ __volatile__("amoor.w.rl zero, %1, (%0)" : : "r"(word), "r"(bits) : "memory");
}
inline void risc5_amoor_d_x0(std::uint64_t *word, std::uint64_t bits, std::memory_order order) noexcept {
    if (order == std::memory_order_relaxed)
        __asm__ __volatile__("amoor.d zero, %1, (%0)" : : "r"(word), "r"(bits) : "memory");
    else __asm__ __volatile__("amoor.d.rl zero, %1, (%0)" : : "r"(word), "r"(bits) : "memory");
}

/*  Load-reserved / store-conditional loops: the reservation must live inside one assembly block.
 *  `lr.w` sign-extends, so the 32-bit comparands travel sign-extended too. */

inline std::uint32_t risc5_lr_sc_cas_w(std::uint32_t *word, std::uint32_t expected, std::uint32_t desired) noexcept {
    std::int64_t const wanted = static_cast<std::int32_t>(expected);
    std::int64_t observed, failed;
    __asm__ __volatile__("1:\n\t"
                         "lr.w.aqrl %[observed], (%[word])\n\t"
                         "bne %[observed], %[wanted], 2f\n\t"
                         "sc.w.rl %[failed], %[desired], (%[word])\n\t"
                         "bnez %[failed], 1b\n"
                         "2:"
                         : [observed] "=&r"(observed), [failed] "=&r"(failed)
                         : [word] "r"(word), [wanted] "r"(wanted), [desired] "r"(desired)
                         : "memory");
    return static_cast<std::uint32_t>(observed);
}
inline std::uint64_t risc5_lr_sc_cas_d(std::uint64_t *word, std::uint64_t expected, std::uint64_t desired) noexcept {
    std::uint64_t observed, failed;
    __asm__ __volatile__("1:\n\t"
                         "lr.d.aqrl %[observed], (%[word])\n\t"
                         "bne %[observed], %[expected], 2f\n\t"
                         "sc.d.rl %[failed], %[desired], (%[word])\n\t"
                         "bnez %[failed], 1b\n"
                         "2:"
                         : [observed] "=&r"(observed), [failed] "=&r"(failed)
                         : [word] "r"(word), [expected] "r"(expected), [desired] "r"(desired)
                         : "memory");
    return observed;
}

/** A byte exchange without `Zabha`: the reservation covers the aligned word, the byte is masked in. */
inline std::uint8_t risc5_lr_sc_swap_b(std::uint8_t *byte, std::uint8_t desired) noexcept {
    std::uintptr_t const address = reinterpret_cast<std::uintptr_t>(byte);
    std::uint32_t *word = reinterpret_cast<std::uint32_t *>(address & ~std::uintptr_t {3});
    unsigned const shift = static_cast<unsigned>(address & 3) * 8;
    std::uint32_t const keep = ~(std::uint32_t {0xFF} << shift);
    std::uint32_t const placed = std::uint32_t {desired} << shift;
    std::uint32_t observed, merged, failed;
    __asm__ __volatile__("1:\n\t"
                         "lr.w.aqrl %[observed], (%[word])\n\t"
                         "and %[merged], %[observed], %[keep]\n\t"
                         "or %[merged], %[merged], %[placed]\n\t"
                         "sc.w.rl %[failed], %[merged], (%[word])\n\t"
                         "bnez %[failed], 1b"
                         : [observed] "=&r"(observed), [merged] "=&r"(merged), [failed] "=&r"(failed)
                         : [word] "r"(word), [keep] "r"(keep), [placed] "r"(placed)
                         : "memory");
    return static_cast<std::uint8_t>(observed >> shift);
}

/**
 *  @brief `std::atomic_ref` over the RISC-V base A extension: one `amo*` per read-modify-write,
 *      `x0` as the destination for the no-return forms, `lr`/`sc` loops for compare-exchange and
 *      byte exchanges, fences around the ordered loads & stores.
 *  @sa `capability_risc5_atomic_k` - the bit admitting it; `risc5_zacas_atomic_ref` - the same with `amocas`.
 */
template <typename value_type_>
struct risc5_atomic_ref {
    static_assert(std::is_trivially_copyable_v<value_type_>, "Only trivially-copyable words are atomic");
    static_assert(sizeof(value_type_) == 1 || sizeof(value_type_) == 4 || sizeof(value_type_) == 8,
                  "Only byte, 32-bit and 64-bit words are spelled");
    using value_t = value_type_;
    using word_t = atomic_word<value_type_>;
    static constexpr capabilities_t capabilities_k = capability_risc5_atomic_k;

    explicit risc5_atomic_ref(value_type_ &word) noexcept : word_(reinterpret_cast<word_t *>(&word)) {}

    value_type_ load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
        if (order == std::memory_order_seq_cst) risc5_fence_rw_rw();
        word_t word;
        if constexpr (sizeof(value_type_) == 1) word = risc5_lbu(word_);
        else if constexpr (sizeof(value_type_) == 4) word = risc5_lw(word_);
        else word = risc5_ld(word_);
        if (acquires(order)) risc5_fence_r_rw();
        return std::bit_cast<value_type_>(word);
    }
    void store(value_type_ desired, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        if (releases(order)) risc5_fence_rw_w();
        word_t const word = std::bit_cast<word_t>(desired);
        if constexpr (sizeof(value_type_) == 1) risc5_sb(word_, word);
        else if constexpr (sizeof(value_type_) == 4) risc5_sw(word_, word);
        else risc5_sd(word_, word);
        if (order == std::memory_order_seq_cst) risc5_fence_rw_rw();
    }

    value_type_ exchange(value_type_ desired, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        word_t const word = std::bit_cast<word_t>(desired);
        if constexpr (sizeof(value_type_) == 1) return std::bit_cast<value_type_>(risc5_lr_sc_swap_b(word_, word));
        else if constexpr (sizeof(value_type_) == 4)
            return std::bit_cast<value_type_>(risc5_amoswap_w(word_, word, order));
        else return std::bit_cast<value_type_>(risc5_amoswap_d(word_, word, order));
    }

    bool compare_exchange_strong(value_type_ &expected, value_type_ desired, std::memory_order,
                                 std::memory_order) const noexcept {
        static_assert(sizeof(value_type_) != 1, "Byte compare-exchange is not spelled yet");
        word_t const wanted = std::bit_cast<word_t>(expected);
        word_t observed;
        if constexpr (sizeof(value_type_) == 4)
            observed = risc5_lr_sc_cas_w(word_, wanted, std::bit_cast<word_t>(desired));
        else observed = risc5_lr_sc_cas_d(word_, wanted, std::bit_cast<word_t>(desired));
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
        static_assert(sizeof(value_type_) != 1, "Byte fetch-add is not spelled yet");
        word_t const word = std::bit_cast<word_t>(operand);
        if constexpr (sizeof(value_type_) == 4) return std::bit_cast<value_type_>(risc5_amoadd_w(word_, word, order));
        else return std::bit_cast<value_type_>(risc5_amoadd_d(word_, word, order));
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
        static_assert(sizeof(value_type_) != 1, "Byte fetch-and is not spelled yet");
        word_t const word = std::bit_cast<word_t>(mask);
        if constexpr (sizeof(value_type_) == 4) return std::bit_cast<value_type_>(risc5_amoand_w(word_, word, order));
        else return std::bit_cast<value_type_>(risc5_amoand_d(word_, word, order));
    }
    value_type_ fetch_or(value_type_ bits, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        static_assert(sizeof(value_type_) != 1, "Byte fetch-or is not spelled yet");
        word_t const word = std::bit_cast<word_t>(bits);
        if constexpr (sizeof(value_type_) == 4) return std::bit_cast<value_type_>(risc5_amoor_w(word_, word, order));
        else return std::bit_cast<value_type_>(risc5_amoor_d(word_, word, order));
    }
    value_type_ fetch_xor(value_type_ bits, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        static_assert(sizeof(value_type_) != 1, "Byte fetch-xor is not spelled yet");
        word_t const word = std::bit_cast<word_t>(bits);
        if constexpr (sizeof(value_type_) == 4) return std::bit_cast<value_type_>(risc5_amoxor_w(word_, word, order));
        else return std::bit_cast<value_type_>(risc5_amoxor_d(word_, word, order));
    }

    /** Native in the base A extension: `amomax`/`amomaxu`, `amomin`/`amominu`. */
    value_type_ fetch_max(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        static_assert(sizeof(value_type_) != 1, "Byte maxima are not spelled yet");
        if constexpr (std::signed_integral<value_type_> && sizeof(value_type_) == 4)
            return static_cast<value_type_>(risc5_amomax_w(reinterpret_cast<std::int32_t *>(word_), operand, order));
        else if constexpr (std::signed_integral<value_type_>)
            return static_cast<value_type_>(risc5_amomax_d(reinterpret_cast<std::int64_t *>(word_), operand, order));
        else if constexpr (sizeof(value_type_) == 4)
            return static_cast<value_type_>(risc5_amomaxu_w(word_, operand, order));
        else return static_cast<value_type_>(risc5_amomaxu_d(word_, operand, order));
    }
    value_type_ fetch_min(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        static_assert(sizeof(value_type_) != 1, "Byte minima are not spelled yet");
        if constexpr (std::signed_integral<value_type_> && sizeof(value_type_) == 4)
            return static_cast<value_type_>(risc5_amomin_w(reinterpret_cast<std::int32_t *>(word_), operand, order));
        else if constexpr (std::signed_integral<value_type_>)
            return static_cast<value_type_>(risc5_amomin_d(reinterpret_cast<std::int64_t *>(word_), operand, order));
        else if constexpr (sizeof(value_type_) == 4)
            return static_cast<value_type_>(risc5_amominu_w(word_, operand, order));
        else return static_cast<value_type_>(risc5_amominu_d(word_, operand, order));
    }

    /** No-return forms: the same operations into `x0`. */
    void add(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        static_assert(sizeof(value_type_) != 1, "Byte no-return add is not spelled yet");
        word_t const word = std::bit_cast<word_t>(operand);
        if constexpr (sizeof(value_type_) == 4) risc5_amoadd_w_x0(word_, word, order);
        else risc5_amoadd_d_x0(word_, word, order);
    }
    void sub(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        add(std::bit_cast<value_type_>(static_cast<word_t>(word_t {0} - std::bit_cast<word_t>(operand))), order);
    }
    void set(value_type_ bits, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        static_assert(sizeof(value_type_) != 1, "Byte no-return set is not spelled yet");
        word_t const word = std::bit_cast<word_t>(bits);
        if constexpr (sizeof(value_type_) == 4) risc5_amoor_w_x0(word_, word, order);
        else risc5_amoor_d_x0(word_, word, order);
    }
    void clear(value_type_ bits, std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        static_assert(sizeof(value_type_) != 1, "Byte no-return clear is not spelled yet");
        word_t const mask = static_cast<word_t>(~std::bit_cast<word_t>(bits));
        if constexpr (sizeof(value_type_) == 4) risc5_amoand_w_x0(word_, mask, order);
        else risc5_amoand_d_x0(word_, mask, order);
    }

    /** The conditional forms: a read-first compare-exchange loop, `amocas` where the extension is. */
    value_type_ fetch_add_if_at_most(value_type_ operand, value_type_ limit,
                                     std::memory_order order = std::memory_order_seq_cst) const noexcept
        requires atomic_integer<value_type_>
    {
        value_type_ observed = load(std::memory_order_acquire);
        while (observed <= limit && limit - observed >= operand &&
               !compare_exchange_strong(observed, observed + operand, order, std::memory_order_acquire)) {}
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

  protected:
    word_t *word_;
};

#pragma endregion RISC5 A

#endif // FU_TARGET_RISC5_ATOMIC

#if FU_TARGET_RISC5_ZACAS

#pragma region RISC5 Zacas

/*  `Zacas`: compare-and-swap as one instruction; the comparand register receives what the word
 *  held. Assemblers disagree on how to name the extension inline - `zacas`, `zacas1p0`, or not at
 *  all - and `.insn` is a directive older LLVM lacks, so the two are whole words with the registers
 *  pinned: the AMO opcode, funct5 `00101`, both `aq` and `rl` set, `a0` as the comparand, `a1` as
 *  the address, `a2` as the desired value. A baseline `rv64gc` build then assembles them and the
 *  runtime bit decides. */
inline std::uint32_t risc5_amocas_w(std::uint32_t *word, std::uint32_t expected, std::uint32_t desired) noexcept {
    register std::int64_t observed __asm__("a0") = static_cast<std::int32_t>(expected);
    register std::uint32_t *address __asm__("a1") = word;
    register std::uint32_t value __asm__("a2") = desired;
    __asm__ __volatile__(".4byte 0x2ec5a52f"
                         : "+r"(observed)
                         : "r"(address), "r"(value)
                         : "memory"); // ? `amocas.w.aqrl a0, a2, (a1)`
    return static_cast<std::uint32_t>(observed);
}
inline std::uint64_t risc5_amocas_d(std::uint64_t *word, std::uint64_t expected, std::uint64_t desired) noexcept {
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
 *  @brief `Zacas` on top of the base: compare-exchange as one `amocas` instead of an `lr`/`sc` loop.
 *  @sa `capability_risc5_zacas_k` - the bit admitting it; `risc5_atomic_ref` - the base it extends.
 */
template <typename value_type_>
struct risc5_zacas_atomic_ref : public risc5_atomic_ref<value_type_> {
    using base_t = risc5_atomic_ref<value_type_>;
    using base_t::base_t;
    using typename base_t::word_t;
    static constexpr capabilities_t capabilities_k = capability_risc5_atomic_k | capability_risc5_zacas_k;

    bool compare_exchange_strong(value_type_ &expected, value_type_ desired, std::memory_order,
                                 std::memory_order) const noexcept {
        static_assert(sizeof(value_type_) != 1, "Byte compare-exchange is not spelled yet");
        word_t const wanted = std::bit_cast<word_t>(expected);
        word_t observed;
        if constexpr (sizeof(value_type_) == 4)
            observed = risc5_amocas_w(this->word_, wanted, std::bit_cast<word_t>(desired));
        else observed = risc5_amocas_d(this->word_, wanted, std::bit_cast<word_t>(desired));
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
};

#pragma endregion RISC5 Zacas

#endif // FU_TARGET_RISC5_ZACAS

/**
 *  The newest reference the build target guarantees through its predefined macros - the
 *  `preferred_yield_t` rule: if the compiler says the instruction is there, running it can never
 *  be illegal; otherwise the standard reference, whose lowering the same flags decide. Callers
 *  dispatching per CPU class at runtime name the references directly instead.
 */
#if FU_TARGET_ARM64_RCPC && defined(__ARM_FEATURE_ATOMICS) && defined(__ARM_FEATURE_RCPC)
template <typename value_type_>
using preferred_atomic_ref = arm64_rcpc_atomic_ref<value_type_>;
#elif FU_TARGET_ARM64_LSE && defined(__ARM_FEATURE_ATOMICS)
template <typename value_type_>
using preferred_atomic_ref = arm64_lse_atomic_ref<value_type_>;
#elif FU_TARGET_X86_RAOINT && defined(__CMPCCXADD__) && defined(__RAOINT__)
template <typename value_type_>
using preferred_atomic_ref = x86_raoint_atomic_ref<value_type_>;
#elif FU_TARGET_X86_CMPCCXADD && defined(__CMPCCXADD__)
template <typename value_type_>
using preferred_atomic_ref = x86_cmpccxadd_atomic_ref<value_type_>;
#elif FU_TARGET_RISC5_ZACAS && defined(__riscv_zacas)
template <typename value_type_>
using preferred_atomic_ref = risc5_zacas_atomic_ref<value_type_>;
#elif FU_TARGET_RISC5_ATOMIC && defined(__riscv_atomic)
template <typename value_type_>
using preferred_atomic_ref = risc5_atomic_ref<value_type_>;
#else
template <typename value_type_>
using preferred_atomic_ref = standard_atomic_ref<value_type_>;
#endif

#endif // __cpp_lib_atomic_ref && __cpp_lib_bit_cast

} // namespace forkunion
} // namespace ashvardanian
