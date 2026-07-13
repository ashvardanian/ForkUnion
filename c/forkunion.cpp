/**
 *  @brief  Low-latency OpenMP-style NUMA-aware cross-platform fine-grained parallelism library.
 *  @file   forkunion.cpp
 *  @author Ash Vardanian
 *  @date   June 27, 2025
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

/** @brief The concrete pool type for a shape and a yield type - the reverse map from a pool's `kind_k`. */
template <fu::pool_kind_t kind_, typename yield_type_>
struct pool_for;
template <typename yield_type_>
struct pool_for<fu::pool_kind_t::flat_k, yield_type_> {
    using type = fu::flat_pool<thread_allocator_t, yield_type_>;
};
template <typename yield_type_>
struct pool_for<fu::pool_kind_t::colocated_k, yield_type_> {
    using type = fu::colocated_pool<yield_type_>;
};
template <typename yield_type_>
struct pool_for<fu::pool_kind_t::distributed_k, yield_type_> {
    using type = fu::distributed_pool<yield_type_>;
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

    using pool_traits_t = max_size_align< //
#if FU_DETECT_ARCH_X86_64_
        fu::flat_pool<thread_allocator_t, fu::x86_pause_t>,  //
        fu::flat_pool<thread_allocator_t, fu::x86_tpause_t>, //
#elif FU_DETECT_ARCH_ARM64_
        fu::flat_pool<thread_allocator_t, fu::arm64_yield_t>, //
#if FU_DETECT_INLINE_ASM_SUPPORT_ // ? `WFET` is inline-assembly only
        fu::flat_pool<thread_allocator_t, fu::arm64_wfet_t>, //
#endif
#elif FU_DETECT_INLINE_ASM_SUPPORT_ && FU_DETECT_ARCH_RISC5_
        fu::flat_pool<thread_allocator_t, fu::risc5_pause_t>, //
        fu::flat_pool<thread_allocator_t, fu::risc5_wrs_t>,   //
#endif

        fu::colocated_pool<fu::standard_yield_t>,   // ? Single-compute-domain pools
        fu::distributed_pool<fu::standard_yield_t>, // ? Whole-machine pools
#if FU_DETECT_ARCH_X86_64_
        fu::colocated_pool<fu::x86_pause_t>,    //
        fu::colocated_pool<fu::x86_tpause_t>,   //
        fu::distributed_pool<fu::x86_pause_t>,  //
        fu::distributed_pool<fu::x86_tpause_t>, //
#elif FU_DETECT_ARCH_ARM64_
        fu::colocated_pool<fu::arm64_yield_t>,   //
        fu::distributed_pool<fu::arm64_yield_t>, //
#if FU_DETECT_INLINE_ASM_SUPPORT_ // ? `WFET` is inline-assembly only
        fu::colocated_pool<fu::arm64_wfet_t>,   //
        fu::distributed_pool<fu::arm64_wfet_t>, //
#endif
#elif FU_DETECT_INLINE_ASM_SUPPORT_ && FU_DETECT_ARCH_RISC5_
        fu::colocated_pool<fu::risc5_pause_t>,   //
        fu::colocated_pool<fu::risc5_wrs_t>,     //
        fu::distributed_pool<fu::risc5_pause_t>, //
        fu::distributed_pool<fu::risc5_wrs_t>,   //
#endif

        fu::flat_pool<thread_allocator_t, fu::standard_yield_t> //
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
        new (storage_) pool_type_(std::forward<args_types_>(args)...);
        kind_ = pool_type_::kind_k;
        capabilities_ = pool_type_::micro_yield_t::capability_k;
    }
};

/**
 *  @brief Dispatches to the stored pool of a known @p kind_, decoding only the single waiter bit.
 *  @sa `visit`, which selects the kind first. There is no bitmask overlap: the shape is the tag, the
 *       waiter is one bit, and `pool_for` turns the pair back into the concrete type.
 */
template <fu::pool_kind_t kind_, typename visitor_type_>
auto visit_kind(visitor_type_ &&visitor, pool_variants_t &variants) {
#if FU_DETECT_ARCH_X86_64_
    if (variants.capabilities_ & fu::capability_x86_tpause_k)
        return visitor(*reinterpret_cast<typename pool_for<kind_, fu::x86_tpause_t>::type *>(variants.storage_));
    if (variants.capabilities_ & fu::capability_x86_pause_k)
        return visitor(*reinterpret_cast<typename pool_for<kind_, fu::x86_pause_t>::type *>(variants.storage_));
#elif FU_DETECT_ARCH_ARM64_
#if FU_DETECT_INLINE_ASM_SUPPORT_ // ? `WFET` is inline-assembly only
    if (variants.capabilities_ & fu::capability_arm64_wfet_k)
        return visitor(*reinterpret_cast<typename pool_for<kind_, fu::arm64_wfet_t>::type *>(variants.storage_));
#endif
    if (variants.capabilities_ & fu::capability_arm64_yield_k)
        return visitor(*reinterpret_cast<typename pool_for<kind_, fu::arm64_yield_t>::type *>(variants.storage_));
#elif FU_DETECT_INLINE_ASM_SUPPORT_ && FU_DETECT_ARCH_RISC5_
    if (variants.capabilities_ & fu::capability_risc5_wrs_k)
        return visitor(*reinterpret_cast<typename pool_for<kind_, fu::risc5_wrs_t>::type *>(variants.storage_));
    if (variants.capabilities_ & fu::capability_risc5_pause_k)
        return visitor(*reinterpret_cast<typename pool_for<kind_, fu::risc5_pause_t>::type *>(variants.storage_));
#endif
    return visitor(*reinterpret_cast<typename pool_for<kind_, fu::standard_yield_t>::type *>(variants.storage_));
}

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
    case fu::pool_kind_t::unknown_k: break; // ? No pool spawned yet
    }
}

/**
 *  @brief Constructs into @p variants the pool of the requested @p kind_k, picking the best waiter
 *         the @p effective capabilities allow and forwarding @p args to that pool's constructor.
 *
 *  One waiter cascade for every pool shape - the sole place the C ABI turns a capability mask into a
 *  concrete `flat_pool` / `colocated_pool` / `distributed_pool` instantiation.
 */
template <fu::pool_kind_t pool_kind_, typename... args_types_>
static void construct_pool(pool_variants_t &variants, FU_MAYBE_UNUSED_ fu::capabilities_t effective,
                           args_types_ &&...args) noexcept {
#if FU_DETECT_ARCH_X86_64_
    if (effective & fu::capability_x86_tpause_k)
        return variants.construct<typename pool_for<pool_kind_, fu::x86_tpause_t>::type>(
            std::forward<args_types_>(args)...);
    if (effective & fu::capability_x86_pause_k)
        return variants.construct<typename pool_for<pool_kind_, fu::x86_pause_t>::type>(
            std::forward<args_types_>(args)...);
#elif FU_DETECT_ARCH_ARM64_
#if FU_DETECT_INLINE_ASM_SUPPORT_ // ? `WFET` is inline-assembly only
    if (effective & fu::capability_arm64_wfet_k)
        return variants.construct<typename pool_for<pool_kind_, fu::arm64_wfet_t>::type>(
            std::forward<args_types_>(args)...);
#endif
    if (effective & fu::capability_arm64_yield_k)
        return variants.construct<typename pool_for<pool_kind_, fu::arm64_yield_t>::type>(
            std::forward<args_types_>(args)...);
#elif FU_DETECT_INLINE_ASM_SUPPORT_ && FU_DETECT_ARCH_RISC5_
    if (effective & fu::capability_risc5_wrs_k)
        return variants.construct<typename pool_for<pool_kind_, fu::risc5_wrs_t>::type>(
            std::forward<args_types_>(args)...);
    if (effective & fu::capability_risc5_pause_k)
        return variants.construct<typename pool_for<pool_kind_, fu::risc5_pause_t>::type>(
            std::forward<args_types_>(args)...);
#endif
    return variants.construct<typename pool_for<pool_kind_, fu::standard_yield_t>::type>(
        std::forward<args_types_>(args)...);
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
    char name[16] {};

    opaque_pool_t(char const *pool_name, fu::capabilities_t pool_capabilities) noexcept : effective(pool_capabilities) {
        char const *const source = pool_name ? pool_name : "forkunion";
        size_t i = 0;
        for (; i + 1 < sizeof(name) && source[i]; ++i) name[i] = source[i];
        name[i] = '\0';
        // `variants` starts empty (kind `unknown_k`); the first spawn builds the pool the topology dictates.
    }

    /** @brief A shim to redirect unsafe callbacks to the current context. */
    void operator()(fu::local_thread_t pinned) const noexcept {
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

int fu_version_major(void) { return FORKUNION_VERSION_MAJOR; }
int fu_version_minor(void) { return FORKUNION_VERSION_MINOR; }
int fu_version_patch(void) { return FORKUNION_VERSION_PATCH; }

fu_capabilities_t fu_comptime_capabilities(void) { return static_cast<fu_capabilities_t>(fu::comptime_capabilities()); }

fu_capabilities_t fu_runtime_capabilities(void) { return static_cast<fu_capabilities_t>(machine_capabilities()); }

size_t fu_name_capabilities(fu_capabilities_t capabilities, char *name_buffer, size_t name_buffer_length) {
    if (!name_buffer || name_buffer_length == 0) return 0;
    char *pos = name_buffer;
    char *const end = name_buffer + name_buffer_length - 1;
    for (unsigned bit = 1; bit != 0 && pos < end; bit <<= 1) {
        if ((capabilities & static_cast<fu_capabilities_t>(bit)) == 0) continue;
        char const *const name = fu::capability_name(static_cast<fu::capabilities_t>(bit));
        if (!name) continue; // ? A bit we set, but do not name
        int const written = std::snprintf(pos, static_cast<size_t>(end - pos), pos == name_buffer ? "%s" : ",%s", name);
        if (written <= 0) break;
        pos += written < end - pos ? written : end - pos; // ! Clamp on truncation
    }
    *pos = '\0';
    return static_cast<size_t>(pos - name_buffer);
}

// Defined below with the pool allocator; declared here for `fu_topology_new`.
void *fu_aligned_malloc(std::size_t size, std::size_t alignment) noexcept;
void fu_aligned_free(void *ptr, std::size_t alignment) noexcept;

fu_topology_t fu_topology_new(void) {
    void *raw = fu_aligned_malloc(sizeof(fu::machine_topology_t), alignof(fu::machine_topology_t));
    if (!raw) return nullptr;
    fu::machine_topology_t *topology = new (raw) fu::machine_topology_t();
    if (!topology->try_harvest()) {
        topology->~machine_topology_t();
        fu_aligned_free(raw, alignof(fu::machine_topology_t));
        return nullptr;
    }
    return reinterpret_cast<fu_topology_t>(topology);
}

void fu_topology_delete(fu_topology_t handle) {
    if (!handle) return;
    fu::machine_topology_t *topology = upcast_topology(handle);
    topology->~machine_topology_t();
    fu_aligned_free(topology, alignof(fu::machine_topology_t));
}

size_t fu_logical_cores_count_in(FU_MAYBE_UNUSED_ fu_topology_t topology,
                                 FU_MAYBE_UNUSED_ size_t compute_domain_index) {
    if (!topology) return 0;
    if (compute_domain_index >= (*upcast_topology(topology)).compute_domains_count()) return 0;
    return (*upcast_topology(topology))
        .compute_domain_at(static_cast<fu::compute_domain_index_t>(compute_domain_index))
        .logical_cores_count;
}

size_t fu_logical_cores_count(FU_MAYBE_UNUSED_ fu_topology_t topology) {
    if (!topology) return 0;
    return (*upcast_topology(topology)).logical_cores_count();
}

size_t fu_compute_domains_count(FU_MAYBE_UNUSED_ fu_topology_t topology) {
    if (!topology) return 0;
    return (*upcast_topology(topology)).compute_domains_count();
}

size_t fu_compute_level_in(FU_MAYBE_UNUSED_ fu_topology_t topology, FU_MAYBE_UNUSED_ size_t compute_domain_index) {
    if (!topology) return 0;
    if (compute_domain_index >= (*upcast_topology(topology)).compute_domains_count()) return 0;
    return (*upcast_topology(topology))
        .compute_domain_at(static_cast<fu::compute_domain_index_t>(compute_domain_index))
        .compute_level;
}

size_t fu_compute_levels_count(FU_MAYBE_UNUSED_ fu_topology_t topology) {
    if (!topology) return 0;
    return (*upcast_topology(topology)).compute_levels_count();
}

size_t fu_compute_capacity_in(FU_MAYBE_UNUSED_ fu_topology_t topology, FU_MAYBE_UNUSED_ size_t compute_domain_index) {
    if (!topology) return 0;
    if (compute_domain_index >= (*upcast_topology(topology)).compute_domains_count()) return 0;
    return (*upcast_topology(topology))
        .compute_domain_at(static_cast<fu::compute_domain_index_t>(compute_domain_index))
        .capacity;
}

size_t fu_compute_cache_bytes_in(FU_MAYBE_UNUSED_ fu_topology_t topology,
                                 FU_MAYBE_UNUSED_ size_t compute_domain_index) {
    if (!topology) return 0;
    if (compute_domain_index >= (*upcast_topology(topology)).compute_domains_count()) return 0;
    return (*upcast_topology(topology))
        .compute_domain_at(static_cast<fu::compute_domain_index_t>(compute_domain_index))
        .cache_bytes;
}

size_t fu_memory_domains_count(FU_MAYBE_UNUSED_ fu_topology_t topology) {
    if (!topology) return 0;
    return (*upcast_topology(topology)).memory_domains_count();
}

size_t fu_memory_level_in(FU_MAYBE_UNUSED_ fu_topology_t topology, FU_MAYBE_UNUSED_ size_t memory_domain_index) {
    if (!topology) return 0;
    if (memory_domain_index >= (*upcast_topology(topology)).memory_domains_count()) return 0;
    return (*upcast_topology(topology))
        .memory_domain_at(static_cast<fu::memory_domain_index_t>(memory_domain_index))
        .memory_level;
}

size_t fu_memory_levels_count(FU_MAYBE_UNUSED_ fu_topology_t topology) {
    if (!topology) return 0;
    return (*upcast_topology(topology)).memory_levels_count();
}

size_t fu_local_memory_of(FU_MAYBE_UNUSED_ fu_topology_t topology, FU_MAYBE_UNUSED_ size_t compute_domain_index) {
    if (!topology) return 0;
    return (*upcast_topology(topology)).local_memory_of(static_cast<fu::compute_domain_index_t>(compute_domain_index));
}

size_t fu_memory_distance(FU_MAYBE_UNUSED_ fu_topology_t topology, FU_MAYBE_UNUSED_ size_t compute_domain_index,
                          FU_MAYBE_UNUSED_ size_t memory_domain_index) {
    if (!topology) return 0;
    return (*upcast_topology(topology))
        .memory_distance(static_cast<fu::compute_domain_index_t>(compute_domain_index),
                         static_cast<fu::memory_domain_index_t>(memory_domain_index));
}

size_t fu_memory_bandwidth(FU_MAYBE_UNUSED_ fu_topology_t topology, FU_MAYBE_UNUSED_ size_t compute_domain_index,
                           FU_MAYBE_UNUSED_ size_t memory_domain_index) {
    if (!topology) return 0;
    return (*upcast_topology(topology))
        .memory_bandwidth(static_cast<fu::compute_domain_index_t>(compute_domain_index),
                          static_cast<fu::memory_domain_index_t>(memory_domain_index));
}

size_t fu_memory_latency(FU_MAYBE_UNUSED_ fu_topology_t topology, FU_MAYBE_UNUSED_ size_t compute_domain_index,
                         FU_MAYBE_UNUSED_ size_t memory_domain_index) {
    if (!topology) return 0;
    return (*upcast_topology(topology))
        .memory_latency(static_cast<fu::compute_domain_index_t>(compute_domain_index),
                        static_cast<fu::memory_domain_index_t>(memory_domain_index));
}

size_t fu_volume_ram_in(FU_MAYBE_UNUSED_ fu_topology_t topology, FU_MAYBE_UNUSED_ size_t memory_domain_index) {
    if (!topology) return 0;
    if (memory_domain_index >= (*upcast_topology(topology)).memory_domains_count()) return 0;
    return (*upcast_topology(topology))
        .memory_domain_at(static_cast<fu::memory_domain_index_t>(memory_domain_index))
        .volume_ram;
}

size_t fu_volume_ram(FU_MAYBE_UNUSED_ fu_topology_t topology) { return fu::volume_ram(); }

size_t fu_volume_huge_pages_in(FU_MAYBE_UNUSED_ fu_topology_t topology, FU_MAYBE_UNUSED_ size_t memory_domain_index) {
    if (!topology) return 0;
    if (memory_domain_index >= (*upcast_topology(topology)).memory_domains_count()) return 0;
    size_t total_volume = 0;
    auto const &node =
        (*upcast_topology(topology)).memory_domain_at(static_cast<fu::memory_domain_index_t>(memory_domain_index));
    for (auto const &page_size : node.page_sizes) total_volume += page_size.bytes_per_page * page_size.free_pages;
    return total_volume;
}

size_t fu_volume_huge_pages(FU_MAYBE_UNUSED_ fu_topology_t topology) {
    if (!topology) return 0;
    size_t total_volume = 0;
    for (size_t memory_domain = 0; memory_domain < (*upcast_topology(topology)).memory_domains_count(); ++memory_domain)
        total_volume += fu_volume_huge_pages_in(topology, memory_domain);
    return total_volume;
}

size_t fu_huge_pages_count_in(FU_MAYBE_UNUSED_ fu_topology_t topology, FU_MAYBE_UNUSED_ size_t memory_domain_index) {
    if (!topology) return 0;
    if (memory_domain_index >= (*upcast_topology(topology)).memory_domains_count()) return 0;
    size_t total_pages = 0;
    auto const &node =
        (*upcast_topology(topology)).memory_domain_at(static_cast<fu::memory_domain_index_t>(memory_domain_index));
    for (auto const &page_size : node.page_sizes) total_pages += page_size.free_pages;
    return total_pages;
}

size_t fu_huge_pages_count(FU_MAYBE_UNUSED_ fu_topology_t topology) {
    if (!topology) return 0;
    size_t total_pages = 0;
    for (size_t memory_domain = 0; memory_domain < (*upcast_topology(topology)).memory_domains_count(); ++memory_domain)
        total_pages += fu_huge_pages_count_in(topology, memory_domain);
    return total_pages;
}

#pragma endregion Metadata

#pragma region Memory

fu_memory_domain_id_t fu_memory_domain_id_at_index(fu_topology_t topology, size_t memory_domain_index) {
    if (!topology) return -1;
    if (memory_domain_index >= (*upcast_topology(topology)).memory_domains_count()) return -1;
    return (*upcast_topology(topology))
        .memory_domain_at(static_cast<fu::memory_domain_index_t>(memory_domain_index))
        .memory_domain_id;
}

void *fu_allocate_at_least_on_domain_id(fu_memory_domain_id_t memory_domain_id, size_t minimum_bytes,
                                        size_t *allocated_bytes, size_t *bytes_per_page) {
    fu::domain_allocator_t allocator(static_cast<fu::memory_domain_id_t>(memory_domain_id));
    auto result = allocator.allocate_at_least(minimum_bytes);
    if (!result) return nullptr;
    *allocated_bytes = result.count;
    *bytes_per_page = result.bytes_per_page();
    return result.ptr;
}

void *fu_allocate_on_domain_id(fu_memory_domain_id_t memory_domain_id, size_t bytes) {
    fu::domain_allocator_t allocator(static_cast<fu::memory_domain_id_t>(memory_domain_id));
    return allocator.allocate(bytes);
}

void fu_free_on_domain_id(fu_memory_domain_id_t memory_domain_id, void *pointer, FU_MAYBE_UNUSED_ size_t bytes) {
    fu::domain_allocator_t allocator(static_cast<fu::memory_domain_id_t>(memory_domain_id));
    allocator.deallocate(reinterpret_cast<char *>(pointer), bytes);
}

void *fu_allocate_symmetric(fu_topology_t topology, size_t bytes_per_domain, size_t *stride_bytes,
                            size_t *memory_domains_count, size_t *total_bytes, size_t *bytes_per_page) {
    if (!topology) return nullptr;
    fu::symmetric_memory_allocator_t allocator(*upcast_topology(topology));
    auto result = allocator.allocate_at_least(bytes_per_domain);
    if (!result) return nullptr;
    if (stride_bytes) *stride_bytes = result.stride_bytes;
    if (memory_domains_count) *memory_domains_count = result.domains;
    if (total_bytes) *total_bytes = result.bytes;
    if (bytes_per_page) *bytes_per_page = result.bytes_per_page();
    return result.ptr;
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

fu_pool_t fu_pool_new(FU_MAYBE_UNUSED_ char const *name, fu_capabilities_t allowed) {
    fu::capabilities_t const effective =
        static_cast<fu::capabilities_t>(machine_capabilities() & static_cast<fu::capabilities_t>(allowed));
    opaque_pool_t *opaque =
        static_cast<opaque_pool_t *>(fu_aligned_malloc(sizeof(opaque_pool_t), alignof(opaque_pool_t)));
    if (!opaque) return nullptr;
    new (opaque) opaque_pool_t(name, effective);
    return reinterpret_cast<fu_pool_t>(opaque);
}

/** @brief Safely cast `fu_pool_t*` to `opaque_pool_t*` avoiding alignment violation warnings. */
inline opaque_pool_t *upcast_pool(fu_pool_t pool) noexcept {
    return std::launder(reinterpret_cast<opaque_pool_t *>(pool));
}

void fu_pool_delete(fu_pool_t pool) {
    assert(pool != nullptr);

    opaque_pool_t *opaque = upcast_pool(pool);
    destroy_variant(opaque->variants);

    // Call the object's destructor and deallocate the memory
    opaque->~opaque_pool_t();
    fu_aligned_free(opaque, alignof(opaque_pool_t));
}

fu_capabilities_t fu_pool_capabilities(fu_pool_t pool) {
    if (!pool) return fu_capabilities_unknown_k;
    pool_variants_t const &variants = upcast_pool(pool)->variants;
    return static_cast<fu_capabilities_t>(variants.capabilities_);
}

fu_bool_t fu_pool_spawn(fu_topology_t topology, fu_pool_t pool, size_t threads, fu_caller_exclusivity_t c_exclusivity) {
    assert(pool != nullptr);
    assert(c_exclusivity == fu_caller_inclusive_k || c_exclusivity == fu_caller_exclusive_k);
    if (!topology) return 0;
    opaque_pool_t *opaque = upcast_pool(pool);
    auto exclusivity = c_exclusivity == fu_caller_inclusive_k ? fu::caller_inclusive_k : fu::caller_exclusive_k;

    // A whole-machine pool is distributed when the mask allows placing memory on domains, else flat.
    // Rebuild to that shape if an earlier spawn left another, then spawn: `visit_kind` fixes the shape
    // and dispatches on the waiter alone, so each branch instantiates only its own `try_spawn`.
    if (opaque->effective & fu::capability_place_memory_on_domain_k) {
        if (opaque->variants.kind_ != fu::pool_kind_t::distributed_k) {
            destroy_variant(opaque->variants);
            construct_pool<fu::pool_kind_t::distributed_k>(opaque->variants, opaque->effective, opaque->name);
        }
        return visit_kind<fu::pool_kind_t::distributed_k>(
            [&](auto &variant) { return variant.try_spawn(*upcast_topology(topology), threads, exclusivity); },
            opaque->variants);
    }
    if (opaque->variants.kind_ != fu::pool_kind_t::flat_k) {
        destroy_variant(opaque->variants);
        construct_pool<fu::pool_kind_t::flat_k>(opaque->variants, opaque->effective);
    }
    return visit_kind<fu::pool_kind_t::flat_k>([&](auto &variant) { return variant.try_spawn(threads, exclusivity); },
                                               opaque->variants);
}

fu_bool_t fu_pool_spawn_on(fu_topology_t topology, fu_pool_t pool, size_t compute_domain_index, size_t threads,
                           fu_caller_exclusivity_t c_exclusivity) {
    assert(pool != nullptr);
    assert(c_exclusivity == fu_caller_inclusive_k || c_exclusivity == fu_caller_exclusive_k);
    if (!topology) return 0;
    opaque_pool_t *opaque = upcast_pool(pool);
    auto exclusivity = c_exclusivity == fu_caller_inclusive_k ? fu::caller_inclusive_k : fu::caller_exclusive_k;

    fu::machine_topology_t const &machine = *upcast_topology(topology);
    if (compute_domain_index >= machine.compute_domains_count()) return 0;

    // Pin to a single compute domain: ensure the variant is colocated, rebuilding if it is not.
    if (opaque->variants.kind_ != fu::pool_kind_t::colocated_k) {
        destroy_variant(opaque->variants);
        construct_pool<fu::pool_kind_t::colocated_k>(opaque->variants, opaque->effective, opaque->name);
    }
    return visit_kind<fu::pool_kind_t::colocated_k>(
        [&](auto &variant) {
            return variant.try_spawn(
                machine.compute_domain_at(static_cast<fu::compute_domain_index_t>(compute_domain_index)), threads,
                exclusivity);
        },
        opaque->variants);
}

fu_caller_exclusivity_t fu_pool_caller_exclusivity(fu_pool_t pool) {
    assert(pool != nullptr);
    opaque_pool_t *opaque = upcast_pool(pool);
    return visit(
        [](auto &variant) {
            return variant.caller_exclusivity() == fu::caller_inclusive_k ? fu_caller_inclusive_k
                                                                          : fu_caller_exclusive_k;
        },
        opaque->variants, fu_caller_inclusive_k);
}

size_t fu_pool_compute_domains_count(fu_pool_t pool) {
    assert(pool != nullptr);
    opaque_pool_t *opaque = upcast_pool(pool);
    return visit([](auto &variant) { return variant.compute_domains_count(); }, opaque->variants, std::size_t {0});
}

size_t fu_pool_threads_count_in(fu_pool_t pool, size_t compute_domain_index) {
    assert(pool != nullptr);
    opaque_pool_t *opaque = upcast_pool(pool);
    return visit([=](auto &variant) { return variant.threads_count(compute_domain_index); }, opaque->variants,
                 std::size_t {0});
}

size_t fu_pool_threads_count(fu_pool_t pool) {
    assert(pool != nullptr);
    opaque_pool_t *opaque = upcast_pool(pool);
    return visit([](auto &variant) { return variant.threads_count(); }, opaque->variants, std::size_t {0});
}

size_t fu_pool_locate_thread_in(fu_pool_t pool, size_t global_thread_index, size_t compute_domain_index) {
    assert(pool != nullptr);
    opaque_pool_t *opaque = upcast_pool(pool);
    return visit([=](auto &variant) { return variant.thread_local_index(global_thread_index, compute_domain_index); },
                 opaque->variants, std::size_t {0});
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

void fu_pool_for_threads(fu_pool_t pool, fu_for_threads_t callback, fu_lambda_context_t context) {
    assert(pool != nullptr && callback != nullptr);
    opaque_pool_t *opaque = upcast_pool(pool);
    visit(
        [&](auto &variant) {
            variant.for_threads([=](fu::local_thread_t pinned) noexcept { //
                callback(context, pinned.thread, pinned.compute_domain);
            });
        },
        opaque->variants);
}

void fu_pool_for_slices(fu_pool_t pool, size_t n, fu_for_slices_t callback, fu_lambda_context_t context) {
    assert(pool != nullptr && callback != nullptr);
    opaque_pool_t *opaque = upcast_pool(pool);
    visit(
        [&](auto &variant) {
            variant.for_slices(n, [=](fu::local_prong_t prong, std::size_t count) noexcept { //
                callback(context, prong.task, count, prong.thread, prong.compute_domain);
            });
        },
        opaque->variants);
}

void fu_pool_for_n(fu_pool_t pool, size_t n, fu_for_prongs_t callback, fu_lambda_context_t context) {
    assert(pool != nullptr && callback != nullptr);
    opaque_pool_t *opaque = upcast_pool(pool);
    visit(
        [&](auto &variant) {
            variant.for_n(n, [=](fu::local_prong_t prong) noexcept { //
                callback(context, prong.task, prong.thread, prong.compute_domain);
            });
        },
        opaque->variants);
}

void fu_pool_for_n_dynamic(fu_pool_t pool, size_t n, fu_for_prongs_t callback, fu_lambda_context_t context) {
    assert(pool != nullptr && callback != nullptr);
    opaque_pool_t *opaque = upcast_pool(pool);
    visit(
        [&](auto &variant) {
            variant.for_n_dynamic(n, [=](fu::local_prong_t prong) noexcept { //
                callback(context, prong.task, prong.thread, prong.compute_domain);
            });
        },
        opaque->variants);
}

#pragma endregion Primary API

#pragma region Flexible API

fu_generation_t fu_pool_unsafe_for_threads(fu_pool_t pool, fu_for_threads_t callback, fu_lambda_context_t context) {
    assert(pool != nullptr && callback != nullptr);
    opaque_pool_t *opaque = upcast_pool(pool);
    opaque->current_context = context;
    opaque->current_callback = callback;
    return visit([&](auto &variant) -> fu_generation_t { return variant.unsafe_for_threads(*opaque); },
                 opaque->variants, fu_generation_t {0});
}

fu_bool_t fu_pool_is_complete(fu_pool_t pool, fu_generation_t generation) {
    assert(pool != nullptr);
    opaque_pool_t *opaque = upcast_pool(pool);
    return visit(
        [generation](auto &variant) -> fu_bool_t {
            return variant.is_complete(
                       static_cast<typename std::remove_reference_t<decltype(variant)>::generation_t>(generation))
                       ? 1
                       : 0;
        },
        opaque->variants, fu_bool_t {0});
}

void fu_pool_unsafe_join(fu_pool_t pool, fu_generation_t generation) {
    assert(pool != nullptr);
    opaque_pool_t *opaque = upcast_pool(pool);
    if (opaque->current_callback == nullptr) return; // ? Idempotent: nothing is in flight
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
