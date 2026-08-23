/**
 *  @brief Low-latency OpenMP-style NUMA-aware cross-platform fine-grained parallelism library.
 *  @file forkunion.cpp
 *  @author Ash Vardanian
 *  @date June 27, 2025
 */
#include <forkunion.h>   // C type aliases
#include <forkunion.hpp> // C++ core implementation

#include <utility>     // `std::in_place_type_t`
#include <algorithm>   // `std::max`
#include <new>         // placement `new` operator
#include <cstdint>     // `std::uint8_t`
#include <type_traits> // `std::aligned_storage`

namespace fu = ashvardanian::forkunion;

using thread_allocator_t = std::allocator<std::thread>;

/** @brief The concrete pool type for a shape, a waiter, and a cache-hints policy - the reverse map
 *      from a pool's `kind_k` and its stored capability bits. */
template <fu::pool_kind_t kind_, typename yield_type_, typename cache_hints_type_>
struct pool_for;
template <typename yield_type_, typename cache_hints_type_>
struct pool_for<fu::pool_kind_t::flat_k, yield_type_, cache_hints_type_> {
    using type = fu::flat_pool<thread_allocator_t, yield_type_, cache_hints_type_>;
};
template <typename yield_type_, typename cache_hints_type_>
struct pool_for<fu::pool_kind_t::colocated_k, yield_type_, cache_hints_type_> {
    using type = fu::colocated_pool<yield_type_, cache_hints_type_>;
};
template <typename yield_type_, typename cache_hints_type_>
struct pool_for<fu::pool_kind_t::distributed_k, yield_type_, cache_hints_type_> {
    using type = fu::distributed_pool<yield_type_, cache_hints_type_>;
};

/**
 *  @brief Custom variant implementation to avoid MSVC `std::variant` alignment issues.
 *
 *  MSVC cannot handle alignas > 64 when objects are passed by value in `std::variant`.
 *  This custom implementation uses a tagged union with manual type management.
 *  @see https://github.com/ashvardanian/ForkUnion/issues/26
 */
struct pool_variants_t {

    /** @brief The largest `sizeof` and the strictest `alignof` across @p types_, for the union's storage. */
    template <typename... types_>
    struct max_size_align {
        static constexpr std::size_t size_k = std::max({sizeof(types_)...});
        static constexpr std::size_t alignment_k = std::max({alignof(types_)...});
    };

    /*  Every (waiter, cache-hints) pair the `select_pool` cascade below may instantiate, spelled
     *  out explicitly per shape. A missed entry cannot silently under-size the storage: `construct`
     *  static-asserts every pool it places against these bounds, so drift fails the build.  */
    using pool_traits_t = max_size_align< //
#if FU_DETECT_ARCH_X86_64_
        fu::flat_pool<thread_allocator_t, fu::x86_pause_t, fu::standard_cache_hints_t>,  //
        fu::flat_pool<thread_allocator_t, fu::x86_tpause_t, fu::standard_cache_hints_t>, //
        fu::flat_pool<thread_allocator_t, fu::x86_tpause_t, fu::x86_cache_hints_t>,      //
#elif FU_DETECT_ARCH_ARM64_
        fu::flat_pool<thread_allocator_t, fu::arm64_yield_t, fu::preferred_cache_hints_t>, //
#if FU_DETECT_INLINE_ASM_SUPPORT_ // `WFET` is inline-assembly only
        fu::flat_pool<thread_allocator_t, fu::arm64_wfet_t, fu::preferred_cache_hints_t>, //
#endif
#elif FU_DETECT_INLINE_ASM_SUPPORT_ && FU_DETECT_ARCH_RISC5_
        fu::flat_pool<thread_allocator_t, fu::risc5_pause_t, fu::risc5_cache_hints_t>,   //
        fu::flat_pool<thread_allocator_t, fu::risc5_wrs_t, fu::risc5_cache_hints_t>,     //
        fu::flat_pool<thread_allocator_t, fu::risc5_wrs_t, fu::risc5_cbo_cache_hints_t>, //
#endif

        fu::colocated_pool<fu::standard_yield_t, fu::standard_cache_hints_t>,   // Single-compute-domain pools
        fu::distributed_pool<fu::standard_yield_t, fu::standard_cache_hints_t>, // Whole-machine pools
#if FU_DETECT_ARCH_X86_64_
        fu::colocated_pool<fu::x86_pause_t, fu::standard_cache_hints_t>,    //
        fu::colocated_pool<fu::x86_tpause_t, fu::standard_cache_hints_t>,   //
        fu::colocated_pool<fu::x86_tpause_t, fu::x86_cache_hints_t>,        //
        fu::distributed_pool<fu::x86_pause_t, fu::standard_cache_hints_t>,  //
        fu::distributed_pool<fu::x86_tpause_t, fu::standard_cache_hints_t>, //
        fu::distributed_pool<fu::x86_tpause_t, fu::x86_cache_hints_t>,      //
#elif FU_DETECT_ARCH_ARM64_
        fu::colocated_pool<fu::arm64_yield_t, fu::preferred_cache_hints_t>,   //
        fu::distributed_pool<fu::arm64_yield_t, fu::preferred_cache_hints_t>, //
#if FU_DETECT_INLINE_ASM_SUPPORT_ // `WFET` is inline-assembly only
        fu::colocated_pool<fu::arm64_wfet_t, fu::preferred_cache_hints_t>,   //
        fu::distributed_pool<fu::arm64_wfet_t, fu::preferred_cache_hints_t>, //
#endif
#elif FU_DETECT_INLINE_ASM_SUPPORT_ && FU_DETECT_ARCH_RISC5_
        fu::colocated_pool<fu::risc5_pause_t, fu::risc5_cache_hints_t>,     //
        fu::colocated_pool<fu::risc5_wrs_t, fu::risc5_cache_hints_t>,       //
        fu::colocated_pool<fu::risc5_wrs_t, fu::risc5_cbo_cache_hints_t>,   //
        fu::distributed_pool<fu::risc5_pause_t, fu::risc5_cache_hints_t>,   //
        fu::distributed_pool<fu::risc5_wrs_t, fu::risc5_cache_hints_t>,     //
        fu::distributed_pool<fu::risc5_wrs_t, fu::risc5_cbo_cache_hints_t>, //
#endif

        fu::flat_pool<thread_allocator_t, fu::standard_yield_t, fu::standard_cache_hints_t> //
        >;

    /** @brief Raw aligned storage holding the one live pool, reinterpreted per `kind_` and `capabilities_`. */
    alignas(pool_traits_t::alignment_k) std::uint8_t storage_[pool_traits_t::size_k];
    /** @brief The stored pool's shape, or `unknown_k` when the storage is empty - no pool spawned yet. */
    fu::pool_kind_t kind_ {fu::pool_kind_t::unknown_k};
    /** @brief The one busy-wait bit the stored pool uses; together with `kind_` it names the concrete type. */
    fu::capabilities_t capabilities_ {fu::capabilities_unknown_k};

    pool_variants_t() = default;
    ~pool_variants_t() = default;

    template <typename pool_type_, typename... args_types_>
    pool_variants_t(std::in_place_type_t<pool_type_>, args_types_ &&...args) noexcept {
        construct<pool_type_>(std::forward<args_types_>(args)...);
    }

    template <typename pool_type_, typename... args_types_>
    void construct(args_types_ &&...args) noexcept {
        // The drift guard: a pool type missing from `pool_traits_t`'s explicit list fails here at
        // compile time, instead of silently under-sizing the storage.
        static_assert(sizeof(pool_type_) <= pool_traits_t::size_k, "Add this pool to `pool_traits_t`");
        static_assert(alignof(pool_type_) <= pool_traits_t::alignment_k, "Add this pool to `pool_traits_t`");
        new (storage_) pool_type_(std::forward<args_types_>(args)...);
        kind_ = pool_type_::kind_k;
        // The waiter and the hints declare distinct bits, so their union names the combination
        // unambiguously, and `select_pool` decodes the exact type back - a deterministic round-trip.
        capabilities_ = static_cast<fu::capabilities_t>(pool_type_::micro_yield_t::capability_k |
                                                        pool_type_::cache_hints_t::capability_k);
    }
};

/** @brief Carries a concrete pool type into a generic action through overload resolution. */
template <typename pool_type_>
struct pool_type_tag_t {
    using type = pool_type_;
};

/** @brief Whether every capability bit the @p yield_type_ and @p hints_type_ declare is in @p bits. */
template <typename yield_type_, typename hints_type_>
static bool selects(fu::capabilities_t const bits) noexcept {
    auto const required = yield_type_::capability_k | hints_type_::capability_k;
    return (bits & required) == static_cast<fu::capabilities_t>(required);
}

/**
 *  @brief The one capability→type cascade: walks the silicon-real (waiter, cache-hints) pairs, most
 *      capable first, and invokes @p action with the tag of the first pair whose every declared
 *      bit is in @p bits; anything unexpected degrades to the nearest pair that only drops
 *      capabilities, down to the portable `(standard_yield_t, standard_cache_hints_t)` fallback.
 *
 *  Serves both directions - `construct_pool` passes the probed machine capabilities, `visit_kind`
 *  passes the bits stored at construction - so selection and decoding can never disagree.
 */
template <fu::pool_kind_t kind_, typename action_type_>
static auto select_pool(FU_MAYBE_UNUSED_ fu::capabilities_t const bits, action_type_ &&action) {
#if FU_DETECT_ARCH_X86_64_
    // WAITPKG ships in Tremont, Alder Lake, and Sapphire Rapids onward; CLDEMOTE only ever shipped
    // alongside it (Tremont, SPR, GNR - fused off on Alder/Raptor/Meteor client parts), so the
    // (pause + cldemote) cell has no silicon and is deliberately not offered.
    if (selects<fu::x86_tpause_t, fu::x86_cache_hints_t>(bits))
        return action(pool_type_tag_t<typename pool_for<kind_, fu::x86_tpause_t, fu::x86_cache_hints_t>::type> {});
    if (selects<fu::x86_tpause_t, fu::standard_cache_hints_t>(bits))
        return action(pool_type_tag_t<typename pool_for<kind_, fu::x86_tpause_t, fu::standard_cache_hints_t>::type> {});
    if (selects<fu::x86_pause_t, fu::standard_cache_hints_t>(bits))
        return action(pool_type_tag_t<typename pool_for<kind_, fu::x86_pause_t, fu::standard_cache_hints_t>::type> {});
#elif FU_DETECT_ARCH_ARM64_
    // `DC CVAC` legality is an OS property (`SCTLR_EL1.UCI`), so the hints half is decided at
    // compile time by `preferred_cache_hints_t` - the clean on Linux, a no-op elsewhere - and the
    // runtime axis stays the waiter alone.
#if FU_DETECT_INLINE_ASM_SUPPORT_ // `WFET` is inline-assembly only
    if (selects<fu::arm64_wfet_t, fu::preferred_cache_hints_t>(bits))
        return action(
            pool_type_tag_t<typename pool_for<kind_, fu::arm64_wfet_t, fu::preferred_cache_hints_t>::type> {});
#endif
    if (selects<fu::arm64_yield_t, fu::preferred_cache_hints_t>(bits))
        return action(
            pool_type_tag_t<typename pool_for<kind_, fu::arm64_yield_t, fu::preferred_cache_hints_t>::type> {});
#elif FU_DETECT_INLINE_ASM_SUPPORT_ && FU_DETECT_ARCH_RISC5_
    // RVA23 mandates Zawrs and Zicbom together, so the monitored waiter travels with the
    // `cbo.clean` demote where the kernel attested it; older parts keep the hint-space
    // `prefetch.w` promotion that can never fault.
    if (selects<fu::risc5_wrs_t, fu::risc5_cbo_cache_hints_t>(bits))
        return action(pool_type_tag_t<typename pool_for<kind_, fu::risc5_wrs_t, fu::risc5_cbo_cache_hints_t>::type> {});
    if (selects<fu::risc5_wrs_t, fu::risc5_cache_hints_t>(bits))
        return action(pool_type_tag_t<typename pool_for<kind_, fu::risc5_wrs_t, fu::risc5_cache_hints_t>::type> {});
    if (selects<fu::risc5_pause_t, fu::risc5_cache_hints_t>(bits))
        return action(pool_type_tag_t<typename pool_for<kind_, fu::risc5_pause_t, fu::risc5_cache_hints_t>::type> {});
#endif
    return action(pool_type_tag_t<typename pool_for<kind_, fu::standard_yield_t, fu::standard_cache_hints_t>::type> {});
}

/**
 *  @brief Dispatches to the stored pool of a known @p kind_, decoding the stored capability bits
 *      through the same `select_pool` cascade that chose them at construction.
 *  @sa `visit`, which selects the kind first. There is no bitmask overlap: the shape is the tag,
 *      and the waiter and hints bits together name the concrete type.
 */
template <fu::pool_kind_t kind_, typename visitor_type_>
auto visit_kind(visitor_type_ &&visitor, pool_variants_t &variants) {
    return select_pool<kind_>(variants.capabilities_, [&](auto tag) {
        using pool_t = typename decltype(tag)::type;
        return visitor(*reinterpret_cast<pool_t *>(variants.storage_));
    });
}

/**
 *  @brief What a failed call leaves in a `size_t` output.
 *
 *  Not part of the contract - the contract is that the output holds nothing meaningful unless the
 *  status is `fu_success_k`. This just makes a caller who ignores the status read something
 *  obviously wrong instead of a plausible zero.
 */
static constexpr std::size_t poisoned_size_k = static_cast<std::size_t>(-1);

/** @brief Lowers a C++ status onto the C vocabulary; the values agree, so this only retypes. */
inline fu_status_t lower(fu::status_t status) noexcept { return static_cast<fu_status_t>(status); }

/** @brief Runs @p visitor on the live pool and returns its result, or @p empty on empty storage. */
template <typename visitor_type_, typename result_type_>
result_type_ visit(visitor_type_ &&visitor, pool_variants_t &variants, result_type_ empty) {
    switch (variants.kind_) {
    case fu::pool_kind_t::colocated_k: return visit_kind<fu::pool_kind_t::colocated_k>(visitor, variants);
    case fu::pool_kind_t::distributed_k: return visit_kind<fu::pool_kind_t::distributed_k>(visitor, variants);
    case fu::pool_kind_t::flat_k: return visit_kind<fu::pool_kind_t::flat_k>(visitor, variants);
    case fu::pool_kind_t::unknown_k: return empty;
    }
    return empty; // An out-of-enum `kind_` is undefined behavior upstream; stay total for `-Wreturn-type`.
}

/** @brief Runs @p visitor on the live pool for its side effects; a no-op on empty storage. */
template <typename visitor_type_>
void visit(visitor_type_ &&visitor, pool_variants_t &variants) {
    switch (variants.kind_) {
    case fu::pool_kind_t::colocated_k: visit_kind<fu::pool_kind_t::colocated_k>(visitor, variants); break;
    case fu::pool_kind_t::distributed_k: visit_kind<fu::pool_kind_t::distributed_k>(visitor, variants); break;
    case fu::pool_kind_t::flat_k: visit_kind<fu::pool_kind_t::flat_k>(visitor, variants); break;
    case fu::pool_kind_t::unknown_k: break; // No pool spawned yet
    }
}

/**
 *  @brief Constructs into @p variants the pool of the requested @p kind_k, picking the best
 *      (waiter, cache-hints) pair the @p effective capabilities allow and forwarding @p args to
 *      that pool's constructor.
 *
 *  One cascade for every pool shape - `select_pool` is the sole place the C ABI turns a capability
 *  mask into a concrete `flat_pool` / `colocated_pool` / `distributed_pool` instantiation.
 */
template <fu::pool_kind_t pool_kind_, typename... args_types_>
static void construct_pool(pool_variants_t &variants, fu::capabilities_t effective, args_types_ &&...args) noexcept {
    select_pool<pool_kind_>(effective, [&](auto tag) {
        using pool_t = typename decltype(tag)::type;
        variants.construct<pool_t>(std::forward<args_types_>(args)...);
    });
}

/**
 *  @brief What a `fu_pool_t` actually points at: a pool, plus the state the C callbacks need.
 *
 *  The C ABI passes a lambda as a context pointer and a function pointer, and the unsafe dispatch
 *  APIs return before the callback runs - so both must outlive the call and live here rather than
 *  on the caller's stack.
 */
struct opaque_pool_t {
    /** @brief The one live pool - flat, colocated, or distributed - or empty before the first spawn. */
    pool_variants_t variants;
    /** @brief The capability envelope `machine_capabilities() & allowed`, fixed at creation. */
    fu::capabilities_t effective {fu::capabilities_unknown_k};
    /** @brief Context held across a non-blocking `fu_pool_unsafe_for_threads` until its join. */
    fu_lambda_context_t current_context {nullptr};
    /** @brief Callback held across a non-blocking `fu_pool_unsafe_for_threads` until its join. */
    fu_for_threads_t current_callback {nullptr};
    /** @brief The caller's pool name, kept so a re-spawn can rebuild the variant without losing it. */
    char name[FU_POOL_NAME_CAPACITY] {};

    opaque_pool_t(char const *pool_name, fu::capabilities_t pool_capabilities) noexcept : effective(pool_capabilities) {
        char const *const source = pool_name ? pool_name : "forkunion";
        size_t i = 0;
        for (; i + 1 < sizeof(name) && source[i]; ++i) name[i] = source[i];
        name[i] = '\0';
        // `variants` starts empty (kind `unknown_k`); the first spawn builds the pool the topology dictates.
    }

    /** @brief A shim to redirect unsafe callbacks to the current context. */
    void operator()(fu::thread_in_domain_t pinned) const noexcept {
        current_callback(current_context, pinned.thread, pinned.compute_domain);
    }
};

/** @brief Terminates and destroys the pool in @p variants, resetting it to the empty `unknown_k` state. */
static void destroy_variant(pool_variants_t &variants) noexcept {
    visit(
        [](auto &variant) {
            using variant_t = std::remove_reference_t<decltype(variant)>;
            variant.terminate();
            variant.~variant_t();
        },
        variants);
    variants.kind_ = fu::pool_kind_t::unknown_k;
}

/** @brief This machine's capabilities - the CPU busy-wait waiters and the memory facilities - probed once. */
static fu::capabilities_t machine_capabilities(void) {
    static fu::capabilities_t const capabilities =
        static_cast<fu::capabilities_t>(fu::cpu_capabilities() | fu::ram_capabilities());
    return capabilities;
}

using machine_topology_t = fu::machine_topology_t;

/** @brief Recovers the `machine_topology_t` behind an opaque `fu_topology_t` handle. */
static fu::machine_topology_t *upcast_topology(fu_topology_t topology) noexcept {
    return std::launder(reinterpret_cast<fu::machine_topology_t *>(topology));
}

/*  `CXX_VISIBILITY_PRESET hidden` keeps the pool templates out of the dynamic symbol table, but alone
 *  it exports nothing, so this re-opens the C ABI below. Guarded on `__GNUC__`, not `__clang__`:
 *  clang-cl defines the latter yet rejects the pragma, and takes its exports from the `.def` instead.  */
#if defined(__GNUC__)
#pragma GCC visibility push(default)
#endif

extern "C" {

#pragma region Metadata

/*  The C enum and the C++ one are spelled out separately - one for callers who have no C++, one for
 *  callers who want it `constexpr`. Nothing but these assertions keeps them from drifting apart.  */
#define fu_assert_same_bit_(c_name, cpp_name)                                           \
    static_assert(static_cast<unsigned>(c_name) == static_cast<unsigned>(fu::cpp_name), \
                  #c_name " drifted from " #cpp_name)

fu_assert_same_bit_(fu_capability_x86_pause_k, capability_x86_pause_k);
fu_assert_same_bit_(fu_capability_x86_tpause_k, capability_x86_tpause_k);
fu_assert_same_bit_(fu_capability_arm64_yield_k, capability_arm64_yield_k);
fu_assert_same_bit_(fu_capability_arm64_wfet_k, capability_arm64_wfet_k);
fu_assert_same_bit_(fu_capability_risc5_pause_k, capability_risc5_pause_k);
fu_assert_same_bit_(fu_capability_risc5_wrs_k, capability_risc5_wrs_k);
fu_assert_same_bit_(fu_capability_x86_cldemote_k, capability_x86_cldemote_k);
fu_assert_same_bit_(fu_capability_arm64_dc_cvac_k, capability_arm64_dc_cvac_k);
fu_assert_same_bit_(fu_capability_risc5_zicbom_k, capability_risc5_zicbom_k);
fu_assert_same_bit_(fu_capability_any_yield_k, capability_any_yield_k);
fu_assert_same_bit_(fu_capability_os_threads_k, capability_os_threads_k);
fu_assert_same_bit_(fu_capability_topology_k, capability_topology_k);
fu_assert_same_bit_(fu_capability_place_threads_by_affinity_k, capability_place_threads_by_affinity_k);
fu_assert_same_bit_(fu_capability_place_threads_by_core_class_k, capability_place_threads_by_core_class_k);
fu_assert_same_bit_(fu_capability_reschedule_threads_by_class_k, capability_reschedule_threads_by_class_k);
fu_assert_same_bit_(fu_capability_place_memory_on_domain_k, capability_place_memory_on_domain_k);
fu_assert_same_bit_(fu_capability_place_huge_pages_on_domain_k, capability_place_huge_pages_on_domain_k);
fu_assert_same_bit_(fu_capability_huge_transparent_pages_k, capability_huge_transparent_pages_k);
fu_assert_same_bit_(fu_capability_colocate_pools_on_domain_k, capability_colocate_pools_on_domain_k);

#undef fu_assert_same_bit_

char const *fu_status_to_string(fu_status_t status) { return fu::status_to_string(static_cast<fu::status_t>(status)); }

int fu_version_major(void) { return FORKUNION_VERSION_MAJOR; }
int fu_version_minor(void) { return FORKUNION_VERSION_MINOR; }
int fu_version_patch(void) { return FORKUNION_VERSION_PATCH; }

fu_capabilities_t fu_comptime_capabilities(void) { return static_cast<fu_capabilities_t>(fu::comptime_capabilities()); }

fu_capabilities_t fu_runtime_capabilities(void) { return static_cast<fu_capabilities_t>(machine_capabilities()); }

fu_status_t fu_name_capabilities(fu_capabilities_t capabilities, char *name_buffer, size_t name_buffer_length,
                                 size_t *written_out) {
    if (!written_out) return fu_invalid_argument_k;
    *written_out = poisoned_size_k;
    if (!name_buffer || name_buffer_length == 0) return fu_invalid_argument_k;
    char *pos = name_buffer;
    char *const end = name_buffer + name_buffer_length - 1;
    for (unsigned bit = 1; bit != 0 && pos < end; bit <<= 1) {
        if ((capabilities & static_cast<fu_capabilities_t>(bit)) == 0) continue;
        char const *const name = fu::capability_name(static_cast<fu::capabilities_t>(bit));
        if (!name) continue; // A bit we set, but do not name
        int const written = std::snprintf(pos, static_cast<size_t>(end - pos), pos == name_buffer ? "%s" : ",%s", name);
        if (written <= 0) break;
        pos += written < end - pos ? written : end - pos; // ! Clamp on truncation
    }
    *pos = '\0';
    *written_out = static_cast<size_t>(pos - name_buffer);
    return fu_success_k;
}

// Defined below with the pool allocator; declared here for `fu_topology_new`.
void *fu_aligned_malloc(std::size_t size, std::size_t alignment) noexcept;
void fu_aligned_free(void *ptr, std::size_t alignment) noexcept;

fu_status_t fu_topology_new(fu_topology_t *topology_out) {
    if (!topology_out) return fu_invalid_argument_k;
    *topology_out = nullptr;
    void *raw = fu_aligned_malloc(sizeof(fu::machine_topology_t), alignof(fu::machine_topology_t));
    if (!raw) return fu_bad_alloc_k;
    fu::machine_topology_t *topology = new (raw) fu::machine_topology_t();
    // An allocation failure and a machine that will not describe itself are different problems,
    // and the caller can now tell them apart.
    if (fu::status_t const harvested = topology->harvest(); fu::failed(harvested)) {
        topology->~machine_topology_t();
        fu_aligned_free(raw, alignof(fu::machine_topology_t));
        return lower(harvested);
    }
    *topology_out = reinterpret_cast<fu_topology_t>(topology);
    return fu_success_k;
}

void fu_topology_delete(fu_topology_t handle) {
    if (!handle) return;
    fu::machine_topology_t *topology = upcast_topology(handle);
    topology->~machine_topology_t();
    fu_aligned_free(topology, alignof(fu::machine_topology_t));
}

fu_status_t fu_logical_cores_count_in(FU_MAYBE_UNUSED_ fu_topology_t topology,
                                      FU_MAYBE_UNUSED_ size_t compute_domain_index, size_t *cores_out) {
    if (!cores_out) return fu_invalid_argument_k;
    *cores_out = poisoned_size_k;
    if (!topology) return fu_invalid_argument_k;
    if (compute_domain_index >= (*upcast_topology(topology)).compute_domains_count()) return fu_invalid_argument_k;
    *cores_out = (*upcast_topology(topology))
                     .compute_domain_at(static_cast<fu::compute_domain_index_t>(compute_domain_index))
                     .logical_cores_count;
    return fu_success_k;
}

fu_status_t fu_logical_cores_count(FU_MAYBE_UNUSED_ fu_topology_t topology, size_t *cores_out) {
    if (!cores_out) return fu_invalid_argument_k;
    *cores_out = poisoned_size_k;
    if (!topology) return fu_invalid_argument_k;
    *cores_out = (*upcast_topology(topology)).logical_cores_count();
    return fu_success_k;
}

fu_status_t fu_compute_domains_count(FU_MAYBE_UNUSED_ fu_topology_t topology, size_t *count_out) {
    if (!count_out) return fu_invalid_argument_k;
    *count_out = poisoned_size_k;
    if (!topology) return fu_invalid_argument_k;
    *count_out = (*upcast_topology(topology)).compute_domains_count();
    return fu_success_k;
}

fu_status_t fu_compute_level_in(FU_MAYBE_UNUSED_ fu_topology_t topology, FU_MAYBE_UNUSED_ size_t compute_domain_index,
                                size_t *level_out) {
    if (!level_out) return fu_invalid_argument_k;
    *level_out = poisoned_size_k;
    if (!topology) return fu_invalid_argument_k;
    if (compute_domain_index >= (*upcast_topology(topology)).compute_domains_count()) return fu_invalid_argument_k;
    *level_out = (*upcast_topology(topology))
                     .compute_domain_at(static_cast<fu::compute_domain_index_t>(compute_domain_index))
                     .compute_level;
    return fu_success_k;
}

fu_status_t fu_compute_levels_count(FU_MAYBE_UNUSED_ fu_topology_t topology, size_t *count_out) {
    if (!count_out) return fu_invalid_argument_k;
    *count_out = poisoned_size_k;
    if (!topology) return fu_invalid_argument_k;
    *count_out = (*upcast_topology(topology)).compute_levels_count();
    return fu_success_k;
}

fu_status_t fu_compute_capacity_in(FU_MAYBE_UNUSED_ fu_topology_t topology,
                                   FU_MAYBE_UNUSED_ size_t compute_domain_index, size_t *capacity_out) {
    if (!capacity_out) return fu_invalid_argument_k;
    *capacity_out = poisoned_size_k;
    if (!topology) return fu_invalid_argument_k;
    if (compute_domain_index >= (*upcast_topology(topology)).compute_domains_count()) return fu_invalid_argument_k;
    *capacity_out = (*upcast_topology(topology))
                        .compute_domain_at(static_cast<fu::compute_domain_index_t>(compute_domain_index))
                        .capacity;
    return fu_success_k;
}

fu_status_t fu_compute_cache_bytes_in(FU_MAYBE_UNUSED_ fu_topology_t topology,
                                      FU_MAYBE_UNUSED_ size_t compute_domain_index, size_t *bytes_out) {
    if (!bytes_out) return fu_invalid_argument_k;
    *bytes_out = poisoned_size_k;
    if (!topology) return fu_invalid_argument_k;
    if (compute_domain_index >= (*upcast_topology(topology)).compute_domains_count()) return fu_invalid_argument_k;
    *bytes_out = (*upcast_topology(topology))
                     .compute_domain_at(static_cast<fu::compute_domain_index_t>(compute_domain_index))
                     .cache_bytes;
    return fu_success_k;
}

fu_status_t fu_memory_domains_count(FU_MAYBE_UNUSED_ fu_topology_t topology, size_t *count_out) {
    if (!count_out) return fu_invalid_argument_k;
    *count_out = poisoned_size_k;
    if (!topology) return fu_invalid_argument_k;
    *count_out = (*upcast_topology(topology)).memory_domains_count();
    return fu_success_k;
}

fu_status_t fu_local_memory_of(FU_MAYBE_UNUSED_ fu_topology_t topology, FU_MAYBE_UNUSED_ size_t compute_domain_index,
                               size_t *memory_domain_out) {
    if (!memory_domain_out) return fu_invalid_argument_k;
    *memory_domain_out = poisoned_size_k;
    if (!topology) return fu_invalid_argument_k;
    // Never bounds-checked before, because there was no way to report the refusal - an
    // out-of-range domain silently answered memory domain 0, which is a real answer elsewhere.
    if (compute_domain_index >= (*upcast_topology(topology)).compute_domains_count()) return fu_invalid_argument_k;
    *memory_domain_out =
        (*upcast_topology(topology)).local_memory_of(static_cast<fu::compute_domain_index_t>(compute_domain_index));
    return fu_success_k;
}

fu_status_t fu_volume_ram_in(FU_MAYBE_UNUSED_ fu_topology_t topology, FU_MAYBE_UNUSED_ size_t memory_domain_index,
                             size_t *bytes_out) {
    if (!bytes_out) return fu_invalid_argument_k;
    *bytes_out = poisoned_size_k;
    if (!topology) return fu_invalid_argument_k;
    if (memory_domain_index >= (*upcast_topology(topology)).memory_domains_count()) return fu_invalid_argument_k;
    *bytes_out = (*upcast_topology(topology))
                     .memory_domain_at(static_cast<fu::memory_domain_index_t>(memory_domain_index))
                     .volume_ram;
    return fu_success_k;
}

fu_status_t fu_volume_ram(FU_MAYBE_UNUSED_ fu_topology_t topology, size_t *bytes_out) {
    if (!bytes_out) return fu_invalid_argument_k;
    *bytes_out = poisoned_size_k;
    if (!topology) return fu_invalid_argument_k;
    *bytes_out = fu::volume_ram();
    return fu_success_k;
}

fu_status_t fu_volume_huge_pages_in(FU_MAYBE_UNUSED_ fu_topology_t topology,
                                    FU_MAYBE_UNUSED_ size_t memory_domain_index, size_t *bytes_out) {
    if (!bytes_out) return fu_invalid_argument_k;
    *bytes_out = poisoned_size_k;
    if (!topology) return fu_invalid_argument_k;
    if (memory_domain_index >= (*upcast_topology(topology)).memory_domains_count()) return fu_invalid_argument_k;
    size_t total_volume = 0;
    auto const &node =
        (*upcast_topology(topology)).memory_domain_at(static_cast<fu::memory_domain_index_t>(memory_domain_index));
    for (auto const &page_size : node.page_sizes) total_volume += page_size.bytes_per_page * page_size.free_pages;
    *bytes_out = total_volume;
    return fu_success_k;
}

fu_status_t fu_volume_huge_pages(FU_MAYBE_UNUSED_ fu_topology_t topology, size_t *bytes_out) {
    if (!bytes_out) return fu_invalid_argument_k;
    *bytes_out = poisoned_size_k;
    if (!topology) return fu_invalid_argument_k;
    size_t total_volume = 0;
    for (size_t memory_domain = 0; memory_domain < (*upcast_topology(topology)).memory_domains_count();
         ++memory_domain) {
        size_t domain_volume = 0;
        if (fu_status_t const got = fu_volume_huge_pages_in(topology, memory_domain, &domain_volume);
            got != fu_success_k)
            return got;
        total_volume += domain_volume;
    }
    *bytes_out = total_volume;
    return fu_success_k;
}

fu_status_t fu_huge_pages_count_in(FU_MAYBE_UNUSED_ fu_topology_t topology, FU_MAYBE_UNUSED_ size_t memory_domain_index,
                                   size_t *pages_out) {
    if (!pages_out) return fu_invalid_argument_k;
    *pages_out = poisoned_size_k;
    if (!topology) return fu_invalid_argument_k;
    if (memory_domain_index >= (*upcast_topology(topology)).memory_domains_count()) return fu_invalid_argument_k;
    size_t total_pages = 0;
    auto const &node =
        (*upcast_topology(topology)).memory_domain_at(static_cast<fu::memory_domain_index_t>(memory_domain_index));
    for (auto const &page_size : node.page_sizes) total_pages += page_size.free_pages;
    *pages_out = total_pages;
    return fu_success_k;
}

fu_status_t fu_huge_pages_count(FU_MAYBE_UNUSED_ fu_topology_t topology, size_t *pages_out) {
    if (!pages_out) return fu_invalid_argument_k;
    *pages_out = poisoned_size_k;
    if (!topology) return fu_invalid_argument_k;
    size_t total_pages = 0;
    for (size_t memory_domain = 0; memory_domain < (*upcast_topology(topology)).memory_domains_count();
         ++memory_domain) {
        size_t domain_pages = 0;
        if (fu_status_t const got = fu_huge_pages_count_in(topology, memory_domain, &domain_pages); got != fu_success_k)
            return got;
        total_pages += domain_pages;
    }
    *pages_out = total_pages;
    return fu_success_k;
}

#pragma endregion Metadata

#pragma region Memory

fu_status_t fu_memory_domain_id_at_index(fu_topology_t topology, size_t memory_domain_index,
                                         fu_memory_domain_id_t *memory_domain_id_out) {
    if (!memory_domain_id_out) return fu_invalid_argument_k;
    // A real domain may carry -1 where the OS names none, so it cannot double as a refusal.
    *memory_domain_id_out = -1;
    if (!topology) return fu_invalid_argument_k;
    if (memory_domain_index >= (*upcast_topology(topology)).memory_domains_count()) return fu_invalid_argument_k;
    *memory_domain_id_out = (*upcast_topology(topology))
                                .memory_domain_at(static_cast<fu::memory_domain_index_t>(memory_domain_index))
                                .memory_domain_id;
    return fu_success_k;
}

fu_status_t fu_allocate_at_least_on_domain_id(fu_memory_domain_id_t memory_domain_id, size_t minimum_bytes,
                                              size_t *allocated_bytes, size_t *bytes_per_page, void **memory_out) {
    if (!memory_out || !allocated_bytes || !bytes_per_page) return fu_invalid_argument_k;
    *memory_out = nullptr;
    *allocated_bytes = poisoned_size_k;
    *bytes_per_page = poisoned_size_k;
    fu::domain_allocator_t allocator(static_cast<fu::memory_domain_id_t>(memory_domain_id));
    auto result = allocator.allocate_at_least(minimum_bytes);
    if (!result) return fu_bad_alloc_k;
    *allocated_bytes = result.count;
    *bytes_per_page = result.bytes_per_page();
    *memory_out = result.ptr;
    return fu_success_k;
}

fu_status_t fu_allocate_on_domain_id(fu_memory_domain_id_t memory_domain_id, size_t bytes, void **memory_out) {
    if (!memory_out) return fu_invalid_argument_k;
    *memory_out = nullptr;
    fu::domain_allocator_t allocator(static_cast<fu::memory_domain_id_t>(memory_domain_id));
    void *const memory = allocator.allocate(bytes);
    if (!memory) return fu_bad_alloc_k;
    *memory_out = memory;
    return fu_success_k;
}

void fu_free_on_domain_id(fu_memory_domain_id_t memory_domain_id, void *pointer, FU_MAYBE_UNUSED_ size_t bytes) {
    fu::domain_allocator_t allocator(static_cast<fu::memory_domain_id_t>(memory_domain_id));
    allocator.deallocate(reinterpret_cast<char *>(pointer), bytes);
}

fu_status_t fu_allocate_symmetric(fu_topology_t topology, size_t bytes_per_domain, size_t *stride_bytes,
                                  size_t *memory_domains_count, size_t *total_bytes, size_t *bytes_per_page,
                                  void **memory_out) {
    if (!memory_out) return fu_invalid_argument_k;
    *memory_out = nullptr;
    if (!topology) return fu_invalid_argument_k;
    fu::symmetric_memory_allocator_t allocator(*upcast_topology(topology));
    auto result = allocator.allocate_at_least(bytes_per_domain);
    if (!result) return fu_bad_alloc_k;
    if (stride_bytes) *stride_bytes = result.stride_bytes;
    if (memory_domains_count) *memory_domains_count = result.domains;
    if (total_bytes) *total_bytes = result.bytes;
    if (bytes_per_page) *bytes_per_page = result.bytes_per_page();
    *memory_out = result.ptr;
    return fu_success_k;
}

void fu_free_symmetric(void *base, size_t total_bytes) {
    fu::symmetric_memory_allocator_t::allocation_type allocation {};
    allocation.ptr = reinterpret_cast<char *>(base);
    allocation.bytes = total_bytes;
    fu::symmetric_memory_allocator_t {}.deallocate(allocation);
}

#pragma endregion Memory

#pragma region Lifetime

/**
 *  @brief Cross-platform aligned memory allocation via C++17 over-aligned `operator new`.
 *  @note Returns nullptr on failure, never throws exceptions.
 */
inline void *fu_aligned_malloc(std::size_t size, std::size_t alignment) noexcept {
    return ::operator new(size, std::align_val_t {alignment}, std::nothrow);
}

/**
 *  @brief Cross-platform aligned memory deallocation.
 *  @note Matches `fu_aligned_malloc` - must use the same alignment value.
 */
inline void fu_aligned_free(void *ptr, std::size_t alignment) noexcept {
    ::operator delete(ptr, std::align_val_t {alignment}, std::nothrow);
}

fu_status_t fu_pool_new(FU_MAYBE_UNUSED_ char const *name, fu_capabilities_t allowed, fu_pool_t *pool_out) {
    if (!pool_out) return fu_invalid_argument_k;
    *pool_out = nullptr;
    fu::capabilities_t const effective =
        static_cast<fu::capabilities_t>(machine_capabilities() & static_cast<fu::capabilities_t>(allowed));
    opaque_pool_t *opaque =
        static_cast<opaque_pool_t *>(fu_aligned_malloc(sizeof(opaque_pool_t), alignof(opaque_pool_t)));
    if (!opaque) return fu_bad_alloc_k;
    new (opaque) opaque_pool_t(name, effective);
    *pool_out = reinterpret_cast<fu_pool_t>(opaque);
    return fu_success_k;
}

/** @brief Safely cast `fu_pool_t*` to `opaque_pool_t*` avoiding alignment violation warnings. */
inline opaque_pool_t *upcast_pool(fu_pool_t pool) noexcept {
    return std::launder(reinterpret_cast<opaque_pool_t *>(pool));
}

void fu_pool_delete(fu_pool_t pool) {
    // The header documents NULL as a no-op, and `fu_topology_delete` honours that with a runtime
    // check. An `assert` vanishes under NDEBUG, so this one segfaulted where its siblings did not.
    if (!pool) return;

    opaque_pool_t *opaque = upcast_pool(pool);
    destroy_variant(opaque->variants);

    // Call the object's destructor and deallocate the memory
    opaque->~opaque_pool_t();
    fu_aligned_free(opaque, alignof(opaque_pool_t));
}

fu_status_t fu_pool_capabilities(fu_pool_t pool, fu_capabilities_t *capabilities_out) {
    if (!capabilities_out) return fu_invalid_argument_k;
    *capabilities_out = fu_capabilities_unknown_k;
    if (!pool) return fu_invalid_argument_k;
    pool_variants_t const &variants = upcast_pool(pool)->variants;
    // The only query no sentinel can serve: a portable-fallback pool reports an empty mask, and
    // `fu_capabilities_unknown_k` is that same zero.
    if (variants.kind_ == fu::pool_kind_t::unknown_k) return fu_not_spawned_k;
    *capabilities_out = static_cast<fu_capabilities_t>(variants.capabilities_);
    return fu_success_k;
}

fu_status_t fu_pool_spawn(fu_topology_t topology, fu_pool_t pool, size_t threads,
                          fu_caller_exclusivity_t c_exclusivity) {
    if (!pool || !topology) return fu_invalid_argument_k;
    if (c_exclusivity != fu_caller_inclusive_k && c_exclusivity != fu_caller_exclusive_k) return fu_invalid_argument_k;
    opaque_pool_t *opaque = upcast_pool(pool);
    auto exclusivity = c_exclusivity == fu_caller_inclusive_k ? fu::caller_inclusive_k : fu::caller_exclusive_k;

    // A whole-machine pool is distributed when the mask allows placing memory on domains, else flat.
    // Rebuild to that shape if an earlier spawn left another, then spawn: `visit_kind` fixes the shape
    // and dispatches on the waiter alone, so each branch instantiates only its own `spawn`.
    if (opaque->effective & fu::capability_place_memory_on_domain_k) {
        if (opaque->variants.kind_ != fu::pool_kind_t::distributed_k) {
            destroy_variant(opaque->variants);
            construct_pool<fu::pool_kind_t::distributed_k>(opaque->variants, opaque->effective, opaque->name);
        }
        return lower(visit_kind<fu::pool_kind_t::distributed_k>(
            [&](auto &variant) { return variant.spawn(*upcast_topology(topology), threads, exclusivity); },
            opaque->variants));
    }
    if (opaque->variants.kind_ != fu::pool_kind_t::flat_k) {
        destroy_variant(opaque->variants);
        construct_pool<fu::pool_kind_t::flat_k>(opaque->variants, opaque->effective);
    }
    return lower(visit_kind<fu::pool_kind_t::flat_k>([&](auto &variant) { return variant.spawn(threads, exclusivity); },
                                                     opaque->variants));
}

fu_status_t fu_pool_spawn_on(fu_topology_t topology, fu_pool_t pool, size_t compute_domain_index, size_t threads,
                             fu_caller_exclusivity_t c_exclusivity) {
    if (!pool || !topology) return fu_invalid_argument_k;
    if (c_exclusivity != fu_caller_inclusive_k && c_exclusivity != fu_caller_exclusive_k) return fu_invalid_argument_k;
    opaque_pool_t *opaque = upcast_pool(pool);
    auto exclusivity = c_exclusivity == fu_caller_inclusive_k ? fu::caller_inclusive_k : fu::caller_exclusive_k;

    fu::machine_topology_t const &machine = *upcast_topology(topology);
    // A refused spawn and an out-of-range domain are distinct answers here.
    if (compute_domain_index >= machine.compute_domains_count()) return fu_invalid_argument_k;

    // Pin to a single compute domain: ensure the variant is colocated, rebuilding if it is not.
    if (opaque->variants.kind_ != fu::pool_kind_t::colocated_k) {
        destroy_variant(opaque->variants);
        construct_pool<fu::pool_kind_t::colocated_k>(opaque->variants, opaque->effective, opaque->name);
    }
    return lower(visit_kind<fu::pool_kind_t::colocated_k>(
        [&](auto &variant) {
            return variant.spawn(
                machine.compute_domain_at(static_cast<fu::compute_domain_index_t>(compute_domain_index)), threads,
                exclusivity);
        },
        opaque->variants));
}

/** @brief Safely cast `fu_fabric_t` to `fu::measured_fabric_t*` avoiding alignment violation warnings. */
inline fu::measured_fabric_t *upcast_fabric(fu_fabric_t fabric) noexcept {
    return std::launder(reinterpret_cast<fu::measured_fabric_t *>(fabric));
}

fu_status_t fu_fabric_new(fu_fabric_t *fabric_out) {
    if (!fabric_out) return fu_invalid_argument_k;
    *fabric_out = nullptr;
    void *raw = fu_aligned_malloc(sizeof(fu::measured_fabric_t), alignof(fu::measured_fabric_t));
    if (!raw) return fu_bad_alloc_k;
    *fabric_out = reinterpret_cast<fu_fabric_t>(new (raw) fu::measured_fabric_t());
    return fu_success_k;
}

void fu_fabric_delete(fu_fabric_t fabric) {
    if (!fabric) return;
    fu::measured_fabric_t *upcast = upcast_fabric(fabric);
    upcast->~measured_fabric();
    fu_aligned_free(upcast, alignof(fu::measured_fabric_t));
}

fu_status_t fu_fabric_harvest(fu_topology_t topology, fu_pool_t pool, fu_fabric_t fabric) {
    if (!pool || !fabric || !topology) return fu_invalid_argument_k;
    opaque_pool_t *opaque = upcast_pool(pool);
    // Only the distributed pool spans memory domains; a flat or colocated pool has no fabric to walk.
    if (opaque->variants.kind_ != fu::pool_kind_t::distributed_k) return fu_config_mismatch_k;
    return lower(visit_kind<fu::pool_kind_t::distributed_k>(
        [&](auto &variant) { return upcast_fabric(fabric)->harvest(*upcast_topology(topology), variant); },
        opaque->variants));
}

fu_status_t fu_fabric_memory_latency(fu_fabric_t fabric, size_t compute_domain_index, size_t memory_domain_index,
                                     size_t *nanoseconds_out) {
    if (!nanoseconds_out) return fu_invalid_argument_k;
    *nanoseconds_out = poisoned_size_k;
    if (!fabric) return fu_invalid_argument_k;
    *nanoseconds_out =
        upcast_fabric(fabric)->memory_latency(static_cast<fu::compute_domain_index_t>(compute_domain_index),
                                              static_cast<fu::memory_domain_index_t>(memory_domain_index));
    return fu_success_k;
}

fu_status_t fu_fabric_memory_bandwidth(fu_fabric_t fabric, size_t compute_domain_index, size_t memory_domain_index,
                                       size_t *megabytes_per_second_out) {
    if (!megabytes_per_second_out) return fu_invalid_argument_k;
    *megabytes_per_second_out = poisoned_size_k;
    if (!fabric) return fu_invalid_argument_k;
    *megabytes_per_second_out =
        upcast_fabric(fabric)->memory_bandwidth(static_cast<fu::compute_domain_index_t>(compute_domain_index),
                                                static_cast<fu::memory_domain_index_t>(memory_domain_index));
    return fu_success_k;
}

fu_status_t fu_fabric_memory_distance(fu_fabric_t fabric, size_t compute_domain_index, size_t memory_domain_index,
                                      size_t *distance_out) {
    if (!distance_out) return fu_invalid_argument_k;
    *distance_out = poisoned_size_k;
    if (!fabric) return fu_invalid_argument_k;
    *distance_out =
        upcast_fabric(fabric)->memory_distance(static_cast<fu::compute_domain_index_t>(compute_domain_index),
                                               static_cast<fu::memory_domain_index_t>(memory_domain_index));
    return fu_success_k;
}

fu_status_t fu_fabric_memory_level_in(fu_fabric_t fabric, size_t memory_domain_index, size_t *level_out) {
    if (!level_out) return fu_invalid_argument_k;
    *level_out = poisoned_size_k;
    if (!fabric) return fu_invalid_argument_k;
    // Tier 0 is the fastest medium, so it is a real answer rather than a refusal.
    *level_out = upcast_fabric(fabric)->memory_level_in(static_cast<fu::memory_domain_index_t>(memory_domain_index));
    return fu_success_k;
}

fu_status_t fu_fabric_memory_levels_count(fu_fabric_t fabric, size_t *levels_out) {
    if (!levels_out) return fu_invalid_argument_k;
    *levels_out = poisoned_size_k;
    if (!fabric) return fu_invalid_argument_k;
    *levels_out = upcast_fabric(fabric)->memory_levels_count();
    return fu_success_k;
}

fu_status_t fu_pool_caller_exclusivity(fu_pool_t pool, fu_caller_exclusivity_t *exclusivity_out) {
    if (!exclusivity_out) return fu_invalid_argument_k;
    // `fu_caller_inclusive_k` is zero, and was also what an unspawned pool answered.
    *exclusivity_out = fu_caller_inclusive_k;
    if (!pool) return fu_invalid_argument_k;
    opaque_pool_t *opaque = upcast_pool(pool);
    if (opaque->variants.kind_ == fu::pool_kind_t::unknown_k) return fu_not_spawned_k;
    *exclusivity_out = visit(
        [](auto &variant) {
            return variant.caller_exclusivity() == fu::caller_inclusive_k ? fu_caller_inclusive_k
                                                                          : fu_caller_exclusive_k;
        },
        opaque->variants, fu_caller_inclusive_k);
    return fu_success_k;
}

fu_status_t fu_pool_compute_domains_count(fu_pool_t pool, size_t *count_out) {
    if (!count_out) return fu_invalid_argument_k;
    *count_out = poisoned_size_k;
    if (!pool) return fu_invalid_argument_k;
    opaque_pool_t *opaque = upcast_pool(pool);
    if (opaque->variants.kind_ == fu::pool_kind_t::unknown_k) return fu_not_spawned_k;
    *count_out =
        visit([](auto &variant) { return variant.compute_domains_count(); }, opaque->variants, std::size_t {0});
    return fu_success_k;
}

fu_status_t fu_pool_threads_count_in(fu_pool_t pool, size_t compute_domain_index, size_t *threads_out) {
    if (!threads_out) return fu_invalid_argument_k;
    *threads_out = poisoned_size_k;
    if (!pool) return fu_invalid_argument_k;
    opaque_pool_t *opaque = upcast_pool(pool);
    if (opaque->variants.kind_ == fu::pool_kind_t::unknown_k) return fu_not_spawned_k;
    *threads_out = visit([=](auto &variant) { return variant.threads_count(compute_domain_index); }, opaque->variants,
                         std::size_t {0});
    return fu_success_k;
}

fu_status_t fu_pool_threads_count(fu_pool_t pool, size_t *threads_out) {
    if (!threads_out) return fu_invalid_argument_k;
    *threads_out = poisoned_size_k;
    if (!pool) return fu_invalid_argument_k;
    opaque_pool_t *opaque = upcast_pool(pool);
    if (opaque->variants.kind_ == fu::pool_kind_t::unknown_k) return fu_not_spawned_k;
    *threads_out = visit([](auto &variant) { return variant.threads_count(); }, opaque->variants, std::size_t {0});
    return fu_success_k;
}

fu_status_t fu_pool_locate_thread_in(fu_pool_t pool, size_t global_thread_index, size_t compute_domain_index,
                                     size_t *local_index_out) {
    if (!local_index_out) return fu_invalid_argument_k;
    *local_index_out = poisoned_size_k;
    if (!pool) return fu_invalid_argument_k;
    opaque_pool_t *opaque = upcast_pool(pool);
    if (opaque->variants.kind_ == fu::pool_kind_t::unknown_k) return fu_not_spawned_k;
    // Local index 0 is the first thread of every domain, so it is a real answer.
    *local_index_out =
        visit([=](auto &variant) { return variant.thread_local_index(global_thread_index, compute_domain_index); },
              opaque->variants, std::size_t {0});
    return fu_success_k;
}

void fu_pool_sleep(fu_pool_t pool, size_t micros) {
    assert(pool != nullptr);
    opaque_pool_t *opaque = upcast_pool(pool);
    visit([=](auto &variant) { variant.sleep(micros); }, opaque->variants);
}

void fu_pool_terminate(fu_pool_t pool) {
    assert(pool != nullptr);
    opaque_pool_t *opaque = upcast_pool(pool);
    visit([](auto &variant) { variant.terminate(); }, opaque->variants);
}

#pragma endregion Lifetime

#pragma region Primary API

fu_status_t fu_pool_for_threads(fu_pool_t pool, fu_for_threads_t callback, fu_lambda_context_t context) {
    if (!pool || !callback) return fu_invalid_argument_k;
    opaque_pool_t *opaque = upcast_pool(pool);
    // Without this an unspawned pool runs zero callbacks and returns normally, which a caller
    // cannot tell from a completed dispatch.
    if (opaque->variants.kind_ == fu::pool_kind_t::unknown_k) return fu_not_spawned_k;
    visit(
        [&](auto &variant) {
            variant.for_threads([=](fu::thread_in_domain_t pinned) noexcept { //
                callback(context, pinned.thread, pinned.compute_domain);
            });
        },
        opaque->variants);
    return fu_success_k;
}

fu_status_t fu_pool_for_slices(fu_pool_t pool, size_t n, fu_for_range_t callback, fu_lambda_context_t context) {
    if (!pool || !callback) return fu_invalid_argument_k;
    opaque_pool_t *opaque = upcast_pool(pool);
    // Without this an unspawned pool runs zero callbacks and returns normally, which a caller
    // cannot tell from a completed dispatch.
    if (opaque->variants.kind_ == fu::pool_kind_t::unknown_k) return fu_not_spawned_k;
    visit(
        [&](auto &variant) {
            variant.for_slices(n, [=](fu::tasks_range_t range, fu::thread_in_domain_t at) noexcept { //
                callback(context, range.first, range.count, at.thread, at.compute_domain);
            });
        },
        opaque->variants);
    return fu_success_k;
}

fu_status_t fu_pool_for_n(fu_pool_t pool, size_t n, fu_for_task_t callback, fu_lambda_context_t context) {
    if (!pool || !callback) return fu_invalid_argument_k;
    opaque_pool_t *opaque = upcast_pool(pool);
    // Without this an unspawned pool runs zero callbacks and returns normally, which a caller
    // cannot tell from a completed dispatch.
    if (opaque->variants.kind_ == fu::pool_kind_t::unknown_k) return fu_not_spawned_k;
    visit(
        [&](auto &variant) {
            variant.for_n(n, [=](std::size_t task, fu::thread_in_domain_t at) noexcept { //
                callback(context, task, at.thread, at.compute_domain);
            });
        },
        opaque->variants);
    return fu_success_k;
}

fu_status_t fu_pool_for_n_dynamic(fu_pool_t pool, size_t n, fu_for_task_t callback, fu_lambda_context_t context) {
    if (!pool || !callback) return fu_invalid_argument_k;
    opaque_pool_t *opaque = upcast_pool(pool);
    // Without this an unspawned pool runs zero callbacks and returns normally, which a caller
    // cannot tell from a completed dispatch.
    if (opaque->variants.kind_ == fu::pool_kind_t::unknown_k) return fu_not_spawned_k;
    visit(
        [&](auto &variant) {
            variant.for_n_dynamic(n, [=](std::size_t task, fu::thread_in_domain_t at) noexcept { //
                callback(context, task, at.thread, at.compute_domain);
            });
        },
        opaque->variants);
    return fu_success_k;
}

#pragma endregion Primary API

#pragma region Flexible API

fu_status_t fu_pool_unsafe_for_threads(fu_pool_t pool, fu_for_threads_t callback, fu_lambda_context_t context,
                                       fu_generation_t *generation_out) {
    if (!generation_out) return fu_invalid_argument_k;
    *generation_out = 0;
    if (!pool || !callback) return fu_invalid_argument_k;
    opaque_pool_t *opaque = upcast_pool(pool);
    if (opaque->variants.kind_ == fu::pool_kind_t::unknown_k) return fu_not_spawned_k;
    opaque->current_context = context;
    opaque->current_callback = callback;
    *generation_out = visit([&](auto &variant) -> fu_generation_t { return variant.unsafe_for_threads(*opaque); },
                            opaque->variants, fu_generation_t {0});
    return fu_success_k;
}

fu_status_t fu_pool_is_complete(fu_pool_t pool, fu_generation_t generation, fu_bool_t *complete_out) {
    if (!complete_out) return fu_invalid_argument_k;
    *complete_out = 0;
    if (!pool) return fu_invalid_argument_k;
    opaque_pool_t *opaque = upcast_pool(pool);
    if (opaque->variants.kind_ == fu::pool_kind_t::unknown_k) return fu_not_spawned_k;
    *complete_out = visit(
        [generation](auto &variant) -> fu_bool_t {
            return variant.is_complete(
                       static_cast<typename std::remove_reference_t<decltype(variant)>::generation_t>(generation))
                       ? 1
                       : 0;
        },
        opaque->variants, fu_bool_t {0});
    return fu_success_k;
}

void fu_pool_unsafe_join(fu_pool_t pool, fu_generation_t generation) {
    assert(pool != nullptr);
    opaque_pool_t *opaque = upcast_pool(pool);
    if (opaque->current_callback == nullptr) return; // Idempotent: nothing is in flight
    visit(
        [generation](auto &variant) {
            variant.unsafe_join(
                static_cast<typename std::remove_reference_t<decltype(variant)>::generation_t>(generation));
        },
        opaque->variants);
    opaque->current_context = nullptr;
    opaque->current_callback = nullptr;
}

#pragma endregion Flexible API
}

#if defined(__GNUC__)
#pragma GCC visibility pop
#endif
