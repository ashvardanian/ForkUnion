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

/** @brief The capability bit for a busy-wait yield type, or unknown for the portable `standard_yield_t`. */
template <typename yield_type_>
constexpr fu::capabilities_t waiter_bit_of() noexcept {
#if FU_DETECT_ASM_YIELDS_
#if FU_DETECT_ARCH_X86_64_
    if constexpr (std::is_same_v<yield_type_, fu::x86_pause_t>) return fu::capability_x86_pause_k;
    else if constexpr (std::is_same_v<yield_type_, fu::x86_tpause_t>)
        return fu::capability_x86_tpause_k;
    else
#endif
#if FU_DETECT_ARCH_ARM64_
        if constexpr (std::is_same_v<yield_type_, fu::arm64_yield_t>)
        return fu::capability_arm64_yield_k;
    else if constexpr (std::is_same_v<yield_type_, fu::arm64_wfet_t>)
        return fu::capability_arm64_wfet_k;
    else
#endif
#if FU_DETECT_ARCH_RISC5_
        if constexpr (std::is_same_v<yield_type_, fu::risc5_pause_t>)
        return fu::capability_risc5_pause_k;
    else if constexpr (std::is_same_v<yield_type_, fu::risc5_wrs_t>)
        return fu::capability_risc5_wrs_k;
    else
#endif
#endif
        return fu::capabilities_unknown_k;
}

/** @brief Maps a concrete pool type to its `pool_kind_t` shape and the yield type it waits with. */
template <typename pool_type_>
struct pool_shape_of;
template <typename yield_type_>
struct pool_shape_of<fu::flat_pool<thread_allocator_t, yield_type_>> {
    static constexpr fu::pool_kind_t kind_k = fu::pool_kind_t::flat_k;
    using yield_t = yield_type_;
};
#if FU_WITH_COLOCATE_POOLS_ON_DOMAIN
template <typename yield_type_>
struct pool_shape_of<fu::colocated_pool<yield_type_>> {
    static constexpr fu::pool_kind_t kind_k = fu::pool_kind_t::colocated_k;
    using yield_t = yield_type_;
};
template <typename yield_type_>
struct pool_shape_of<fu::distributed_pool<yield_type_>> {
    static constexpr fu::pool_kind_t kind_k = fu::pool_kind_t::distributed_k;
    using yield_t = yield_type_;
};
#endif

/** @brief The concrete pool type for a shape and a yield type - the inverse of `pool_shape_of`. */
template <fu::pool_kind_t kind_, typename yield_type_>
struct pool_for;
template <typename yield_type_>
struct pool_for<fu::pool_kind_t::flat_k, yield_type_> {
    using type = fu::flat_pool<thread_allocator_t, yield_type_>;
};
#if FU_WITH_COLOCATE_POOLS_ON_DOMAIN
template <typename yield_type_>
struct pool_for<fu::pool_kind_t::colocated_k, yield_type_> {
    using type = fu::colocated_pool<yield_type_>;
};
template <typename yield_type_>
struct pool_for<fu::pool_kind_t::distributed_k, yield_type_> {
    using type = fu::distributed_pool<yield_type_>;
};
#endif

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
#if FU_DETECT_ASM_YIELDS_
#if FU_DETECT_ARCH_X86_64_
        fu::flat_pool<thread_allocator_t, fu::x86_pause_t>,  //
        fu::flat_pool<thread_allocator_t, fu::x86_tpause_t>, //
#endif
#if FU_DETECT_ARCH_ARM64_
        fu::flat_pool<thread_allocator_t, fu::arm64_yield_t>, //
        fu::flat_pool<thread_allocator_t, fu::arm64_wfet_t>,  //
#endif
#if FU_DETECT_ARCH_RISC5_
        fu::flat_pool<thread_allocator_t, fu::risc5_pause_t>, //
        fu::flat_pool<thread_allocator_t, fu::risc5_wrs_t>,   //
#endif
#endif // FU_DETECT_ASM_YIELDS_

#if FU_WITH_COLOCATE_POOLS_ON_DOMAIN
        fu::colocated_pool<fu::standard_yield_t>,   // ? Single-compute-domain pools
        fu::distributed_pool<fu::standard_yield_t>, // ? Whole-machine pools
#if FU_DETECT_ASM_YIELDS_
#if FU_DETECT_ARCH_X86_64_
        fu::colocated_pool<fu::x86_pause_t>,    //
        fu::colocated_pool<fu::x86_tpause_t>,   //
        fu::distributed_pool<fu::x86_pause_t>,  //
        fu::distributed_pool<fu::x86_tpause_t>, //
#endif
#if FU_DETECT_ARCH_ARM64_
        fu::colocated_pool<fu::arm64_yield_t>,   //
        fu::colocated_pool<fu::arm64_wfet_t>,    //
        fu::distributed_pool<fu::arm64_yield_t>, //
        fu::distributed_pool<fu::arm64_wfet_t>,  //
#endif
#if FU_DETECT_ARCH_RISC5_
        fu::colocated_pool<fu::risc5_pause_t>,   //
        fu::colocated_pool<fu::risc5_wrs_t>,     //
        fu::distributed_pool<fu::risc5_pause_t>, //
        fu::distributed_pool<fu::risc5_wrs_t>,   //
#endif
#endif // FU_DETECT_ASM_YIELDS_
#endif // FU_WITH_PLACE_MEMORY_ON_DOMAIN

        fu::flat_pool<thread_allocator_t, fu::standard_yield_t> //
        >;

    alignas(pool_traits_t::alignment_k) std::uint8_t storage_[pool_traits_t::size_k];
    /** @brief The shape of the stored pool, and the single waiter bit it uses - together they name the type. */
    fu::pool_kind_t kind_ {fu::pool_kind_t::flat_k};
    fu::capabilities_t waiter_ {fu::capabilities_unknown_k};

    pool_variants_t() = default;
    ~pool_variants_t() = default;

    template <typename pool_type_, typename... args_types_>
    pool_variants_t(std::in_place_type_t<pool_type_>, args_types_ &&...args) noexcept {
        construct<pool_type_>(std::forward<args_types_>(args)...);
    }

    template <typename pool_type_, typename... args_types_>
    void construct(args_types_ &&...args) noexcept {
        new (storage_) pool_type_(std::forward<args_types_>(args)...);
        kind_ = pool_shape_of<pool_type_>::kind_k;
        waiter_ = waiter_bit_of<typename pool_shape_of<pool_type_>::yield_t>();
    }
};

/**
 *  @brief Dispatches to the stored pool of a known @p kind_, decoding only the single waiter bit.
 *  @sa `visit`, which selects the kind first. There is no bitmask overlap: the shape is the tag, the
 *       waiter is one bit, and `pool_for` turns the pair back into the concrete type.
 */
template <fu::pool_kind_t kind_, typename visitor_type_>
auto visit_kind(visitor_type_ &&visitor, pool_variants_t &variants) {
#if FU_DETECT_ASM_YIELDS_
#if FU_DETECT_ARCH_X86_64_
    if (variants.waiter_ & fu::capability_x86_tpause_k)
        return visitor(*reinterpret_cast<typename pool_for<kind_, fu::x86_tpause_t>::type *>(variants.storage_));
    if (variants.waiter_ & fu::capability_x86_pause_k)
        return visitor(*reinterpret_cast<typename pool_for<kind_, fu::x86_pause_t>::type *>(variants.storage_));
#endif
#if FU_DETECT_ARCH_ARM64_
    if (variants.waiter_ & fu::capability_arm64_wfet_k)
        return visitor(*reinterpret_cast<typename pool_for<kind_, fu::arm64_wfet_t>::type *>(variants.storage_));
    if (variants.waiter_ & fu::capability_arm64_yield_k)
        return visitor(*reinterpret_cast<typename pool_for<kind_, fu::arm64_yield_t>::type *>(variants.storage_));
#endif
#if FU_DETECT_ARCH_RISC5_
    if (variants.waiter_ & fu::capability_risc5_wrs_k)
        return visitor(*reinterpret_cast<typename pool_for<kind_, fu::risc5_wrs_t>::type *>(variants.storage_));
    if (variants.waiter_ & fu::capability_risc5_pause_k)
        return visitor(*reinterpret_cast<typename pool_for<kind_, fu::risc5_pause_t>::type *>(variants.storage_));
#endif
#endif
    return visitor(*reinterpret_cast<typename pool_for<kind_, fu::standard_yield_t>::type *>(variants.storage_));
}

// ? Custom visit function to replace std::visit - the shape picks the arm, the waiter picks the type.
template <typename visitor_type_>
auto visit(visitor_type_ &&visitor, pool_variants_t &variants) {
#if FU_WITH_COLOCATE_POOLS_ON_DOMAIN
    switch (variants.kind_) {
    case fu::pool_kind_t::colocated_k: return visit_kind<fu::pool_kind_t::colocated_k>(visitor, variants);
    case fu::pool_kind_t::distributed_k: return visit_kind<fu::pool_kind_t::distributed_k>(visitor, variants);
    case fu::pool_kind_t::flat_k: break;
    }
#endif
    return visit_kind<fu::pool_kind_t::flat_k>(visitor, variants);
}

/**
 *  @brief What a `fu_pool_t` actually points at: a pool, plus the state the C callbacks need.
 *
 *  The C ABI passes a lambda as a context pointer and a function pointer, and the unsafe dispatch
 *  APIs return before the callback runs - so both must outlive the call and live here rather than
 *  on the caller's stack.
 */
struct opaque_pool_t {
    pool_variants_t variants;
    /** Current context for the unsafe callbacks. */
    fu_lambda_context_t current_context;
    /** Current callback for the unsafe callbacks. */
    fu_for_threads_t current_callback;
    /** Target compute domain for a spawn-on pool, else 0. */
    size_t compute_domain_index {0};
    /** The caller's pool name, kept so `fu_pool_spawn_on` can rebuild the pool without losing it. */
    char name[16] {};

    template <typename pool_type_, typename... args_types_>
    opaque_pool_t(char const *pool_name, std::in_place_type_t<pool_type_> inplace, args_types_ &&...args) noexcept
        : variants(inplace, std::forward<args_types_>(args)...), current_context(nullptr), current_callback(nullptr) {
        char const *const source = pool_name ? pool_name : "forkunion";
        size_t i = 0;
        for (; i + 1 < sizeof(name) && source[i]; ++i) name[i] = source[i];
        name[i] = '\0';
    }

    /** @brief A shim to redirect unsafe callbacks to the current context. */
    void operator()(fu::local_thread_t pinned) const noexcept {
        current_callback(current_context, pinned.thread, pinned.compute_domain);
    }
};

static fu::machine_topology_t global_topology {};
static fu::capabilities_t global_capabilities {fu::capabilities_unknown_k};
static char global_capabilities_string[128] {};

/**
 *  @brief Harvests the machine's topology and capabilities, exactly once, on whichever thread asks first.
 *  @note Never called directly; `globals_initialize` wraps it in the one-time initialization.
 *
 *  Two threads racing here would both run `try_harvest`, and the second `reset()` would free buffers
 *  the first was still filling. The C ABI has no `fu_initialize`, so any entry point may be the first,
 *  and callers routinely reach several of them from several threads at once.
 */
static bool globals_initialize_once(void) {
#if FU_WITH_COLOCATE_POOLS_ON_DOMAIN
    if (!global_topology.try_harvest()) return false;
#endif

    fu::capabilities_t cpu_caps = fu::cpu_capabilities();
    fu::capabilities_t ram_caps = fu::ram_capabilities();

    global_capabilities = static_cast<fu::capabilities_t>(cpu_caps | ram_caps);

    // Now, populate the capabilities string, comma-separated to match its compile-time twin. Walk the
    // bits and defer every name to `fu::capability_name`, so there is one source of truth for names.
    char *pos = global_capabilities_string;
    char *const end = global_capabilities_string + sizeof(global_capabilities_string) - 1;
    for (unsigned bit = 1; bit; bit <<= 1) {
        if ((global_capabilities & bit) == 0) continue;
        char const *const name = fu::capability_name(static_cast<fu::capabilities_t>(bit));
        if (!name) continue; // ? A bit we set, but do not name
        int const written =
            std::snprintf(pos, static_cast<size_t>(end - pos), pos == global_capabilities_string ? "%s" : ",%s", name);
        if (written <= 0 || written >= end - pos) break; // ! Truncated; keep what fits
        pos += written;
    }
    if (pos == global_capabilities_string) std::snprintf(pos, static_cast<size_t>(end - pos), "none");
    return true;
}

bool globals_initialize(void) {
    // A function-local static is initialized exactly once, and every other thread blocks until the
    // winner is done - which is precisely the guarantee `try_harvest` needs, and costs one relaxed
    // load per call afterwards.
    static bool const initialized = globals_initialize_once();
    return initialized;
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

char const *fu_comptime_capabilities_string(void) {
    /*  Every bit is a macro, so the answer is a string literal. Each name carries a leading comma
     *  and we skip it on the way out, which beats trimming a trailing one - and beats `std::string`,
     *  which this library does not use and does not want to start allocating from.  */
    static char const joined[] =
#if FU_WITH_OS_THREADS
        ",os_threads"
#endif
#if FU_WITH_TOPOLOGY
        ",topology"
#endif
#if FU_WITH_PLACE_THREADS_BY_AFFINITY
        ",place_threads_by_affinity"
#endif
#if FU_WITH_PLACE_THREADS_BY_CORE_CLASS
        ",place_threads_by_core_class"
#endif
#if FU_WITH_RESCHEDULE_THREADS_BY_CLASS
        ",reschedule_threads_by_class"
#endif
#if FU_WITH_PLACE_MEMORY_ON_DOMAIN
        ",place_memory_on_domain"
#endif
#if FU_WITH_PLACE_HUGE_PAGES_ON_DOMAIN
        ",place_huge_pages_on_domain"
#endif
#if FU_WITH_COLOCATE_POOLS_ON_DOMAIN
        ",colocate_pools_on_domain"
#endif
        ;
    return sizeof(joined) > 1 ? &joined[1] : "none";
}

fu_capabilities_t fu_runtime_capabilities(void) {
    if (!globals_initialize()) return fu_capabilities_unknown_k;
    return static_cast<fu_capabilities_t>(global_capabilities);
}

char const *fu_runtime_capabilities_string(void) {
    if (!globals_initialize()) return nullptr;
    return &global_capabilities_string[0];
}

char const *fu_capability_name(fu_capabilities_t capability) {
    return fu::capability_name(static_cast<fu::capabilities_t>(capability));
}

fu_capabilities_t fu_capability_named(char const *name) {
    return static_cast<fu_capabilities_t>(fu::capability_named(name));
}

size_t fu_logical_cores_count_in(FU_MAYBE_UNUSED_ size_t compute_domain_index) {
#if FU_WITH_COLOCATE_POOLS_ON_DOMAIN
    if (!globals_initialize()) return 0;
    if (compute_domain_index >= global_topology.compute_domains_count()) return 0;
    return global_topology.compute_domain_at(static_cast<fu::compute_domain_index_t>(compute_domain_index)).core_count;
#else
    return compute_domain_index == 0 ? std::thread::hardware_concurrency() : 0;
#endif
}

size_t fu_logical_cores_count(void) {
#if FU_WITH_COLOCATE_POOLS_ON_DOMAIN
    if (!globals_initialize()) return 0;
    return global_topology.threads_count();
#else
    // ! Not `hardware_concurrency`, which counts the machine's cores rather than the ones a
    // ! `taskset` or a cgroup `cpuset` left us. Sizing a pool from the former oversubscribes.
    return fu::allowed_cores_count();
#endif
}

size_t fu_compute_domains_count(void) {
#if FU_WITH_COLOCATE_POOLS_ON_DOMAIN
    if (!globals_initialize()) return 0;
    return global_topology.compute_domains_count();
#else
    return 1;
#endif
}

size_t fu_compute_level_in(FU_MAYBE_UNUSED_ size_t compute_domain_index) {
#if FU_WITH_COLOCATE_POOLS_ON_DOMAIN
    if (!globals_initialize()) return 0;
    if (compute_domain_index >= global_topology.compute_domains_count()) return 0;
    return global_topology.compute_domain_at(static_cast<fu::compute_domain_index_t>(compute_domain_index))
        .compute_level;
#else
    return 0;
#endif
}

size_t fu_compute_levels_count(void) {
#if FU_WITH_COLOCATE_POOLS_ON_DOMAIN
    if (!globals_initialize()) return 0;
    return global_topology.compute_levels_count();
#else
    return 1;
#endif
}

size_t fu_compute_capacity_in(FU_MAYBE_UNUSED_ size_t compute_domain_index) {
#if FU_WITH_COLOCATE_POOLS_ON_DOMAIN
    if (!globals_initialize()) return 0;
    if (compute_domain_index >= global_topology.compute_domains_count()) return 0;
    return global_topology.compute_domain_at(static_cast<fu::compute_domain_index_t>(compute_domain_index)).capacity;
#else
    return 0; // ? No per-core throughput without a harvested topology
#endif
}

size_t fu_compute_cache_bytes_in(FU_MAYBE_UNUSED_ size_t compute_domain_index) {
#if FU_WITH_COLOCATE_POOLS_ON_DOMAIN
    if (!globals_initialize()) return 0;
    if (compute_domain_index >= global_topology.compute_domains_count()) return 0;
    return global_topology.compute_domain_at(static_cast<fu::compute_domain_index_t>(compute_domain_index)).cache_bytes;
#else
    return 0;
#endif
}

size_t fu_memory_domains_count(void) {
#if FU_WITH_COLOCATE_POOLS_ON_DOMAIN
    if (!globals_initialize()) return 0;
    return global_topology.memory_domains_count();
#else
    return 1;
#endif
}

size_t fu_memory_level_in(FU_MAYBE_UNUSED_ size_t memory_domain_index) {
#if FU_WITH_COLOCATE_POOLS_ON_DOMAIN
    if (!globals_initialize()) return 0;
    if (memory_domain_index >= global_topology.memory_domains_count()) return 0;
    return global_topology.memory_domain_at(static_cast<fu::memory_domain_index_t>(memory_domain_index)).memory_level;
#else
    return 0;
#endif
}

size_t fu_memory_levels_count(void) {
#if FU_WITH_COLOCATE_POOLS_ON_DOMAIN
    if (!globals_initialize()) return 0;
    return global_topology.memory_levels_count();
#else
    return 1; // ? One uniform tier, mirroring `fu_compute_levels_count`
#endif
}

size_t fu_local_memory_of(FU_MAYBE_UNUSED_ size_t compute_domain_index) {
#if FU_WITH_COLOCATE_POOLS_ON_DOMAIN
    if (!globals_initialize()) return 0;
    return global_topology.local_memory_of(static_cast<fu::compute_domain_index_t>(compute_domain_index));
#else
    return 0;
#endif
}

size_t fu_memory_distance(FU_MAYBE_UNUSED_ size_t compute_domain_index, FU_MAYBE_UNUSED_ size_t memory_domain_index) {
#if FU_WITH_COLOCATE_POOLS_ON_DOMAIN
    if (!globals_initialize()) return 0;
    return global_topology.distance(static_cast<fu::compute_domain_index_t>(compute_domain_index),
                                    static_cast<fu::memory_domain_index_t>(memory_domain_index));
#else
    return compute_domain_index == 0 && memory_domain_index == 0 ? 10 : 0;
#endif
}

size_t fu_memory_bandwidth(FU_MAYBE_UNUSED_ size_t compute_domain_index, FU_MAYBE_UNUSED_ size_t memory_domain_index) {
#if FU_WITH_COLOCATE_POOLS_ON_DOMAIN
    if (!globals_initialize()) return 0;
    return global_topology.memory_bandwidth(static_cast<fu::compute_domain_index_t>(compute_domain_index),
                                            static_cast<fu::memory_domain_index_t>(memory_domain_index));
#else
    return 0;
#endif
}

size_t fu_memory_latency(FU_MAYBE_UNUSED_ size_t compute_domain_index, FU_MAYBE_UNUSED_ size_t memory_domain_index) {
#if FU_WITH_COLOCATE_POOLS_ON_DOMAIN
    if (!globals_initialize()) return 0;
    return global_topology.memory_latency(static_cast<fu::compute_domain_index_t>(compute_domain_index),
                                          static_cast<fu::memory_domain_index_t>(memory_domain_index));
#else
    return 0;
#endif
}

size_t fu_volume_ram_in(FU_MAYBE_UNUSED_ size_t memory_domain_index) {
#if FU_WITH_COLOCATE_POOLS_ON_DOMAIN
    if (!globals_initialize()) return 0;
    if (memory_domain_index >= global_topology.memory_domains_count()) return 0;
    return global_topology.memory_domain_at(static_cast<fu::memory_domain_index_t>(memory_domain_index)).memory_size;
#else
    return memory_domain_index == 0 ? fu::ram_total_bytes() : 0;
#endif
}

size_t fu_volume_ram(void) { return fu::ram_total_bytes(); }

size_t fu_volume_huge_pages_in(FU_MAYBE_UNUSED_ size_t memory_domain_index) {
#if FU_WITH_PLACE_MEMORY_ON_DOMAIN
    if (!globals_initialize()) return 0;
    if (memory_domain_index >= global_topology.memory_domains_count()) return 0;
    size_t total_volume = 0;
    auto const &node = global_topology.memory_domain_at(static_cast<fu::memory_domain_index_t>(memory_domain_index));
    for (auto const &page_size : node.page_sizes) total_volume += page_size.bytes_per_page * page_size.free_pages;
    return total_volume;
#else
    return 0;
#endif
}

size_t fu_volume_huge_pages(void) {
#if FU_WITH_PLACE_MEMORY_ON_DOMAIN
    if (!globals_initialize()) return 0;
    size_t total_volume = 0;
    for (size_t memory_domain = 0; memory_domain < global_topology.memory_domains_count(); ++memory_domain)
        total_volume += fu_volume_huge_pages_in(memory_domain);
    return total_volume;
#else
    return 0;
#endif
}

size_t fu_huge_pages_count_in(FU_MAYBE_UNUSED_ size_t memory_domain_index) {
#if FU_WITH_PLACE_MEMORY_ON_DOMAIN
    if (!globals_initialize()) return 0;
    if (memory_domain_index >= global_topology.memory_domains_count()) return 0;
    size_t total_pages = 0;
    auto const &node = global_topology.memory_domain_at(static_cast<fu::memory_domain_index_t>(memory_domain_index));
    for (auto const &page_size : node.page_sizes) total_pages += page_size.free_pages;
    return total_pages;
#else
    return 0;
#endif
}

size_t fu_huge_pages_count(void) {
#if FU_WITH_PLACE_MEMORY_ON_DOMAIN
    if (!globals_initialize()) return 0;
    size_t total_pages = 0;
    for (size_t memory_domain = 0; memory_domain < global_topology.memory_domains_count(); ++memory_domain)
        total_pages += fu_huge_pages_count_in(memory_domain);
    return total_pages;
#else
    return 0;
#endif
}

#pragma endregion Metadata

#pragma region Memory

void *fu_allocate_at_least_in(                                         //
    FU_MAYBE_UNUSED_ size_t memory_domain_index, size_t minimum_bytes, //
    size_t *allocated_bytes, size_t *bytes_per_page) {

#if FU_WITH_PLACE_MEMORY_ON_DOMAIN
    auto const &node = global_topology.memory_domain_at(static_cast<fu::memory_domain_index_t>(memory_domain_index));
    fu::memory_domain_allocator_t allocator(node.memory_domain_id);
    auto result = allocator.allocate_at_least(minimum_bytes);
    if (!result) return nullptr;
    *allocated_bytes = result.count;
    *bytes_per_page = result.bytes_per_page();
    return result.ptr;
#else
    auto result = std::malloc(minimum_bytes);
    if (!result) return nullptr;
    *allocated_bytes = minimum_bytes;
    *bytes_per_page = fu::ram_page_size();
    return result;
#endif
}

void *fu_allocate_in(FU_MAYBE_UNUSED_ size_t memory_domain_index, size_t bytes) {

#if FU_WITH_PLACE_MEMORY_ON_DOMAIN
    auto const &node = global_topology.memory_domain_at(static_cast<fu::memory_domain_index_t>(memory_domain_index));
    fu::memory_domain_allocator_t allocator(node.memory_domain_id);
    return allocator.allocate(bytes);
#else
    return std::malloc(bytes);
#endif
}

void fu_free_in(FU_MAYBE_UNUSED_ size_t memory_domain_index, void *pointer, FU_MAYBE_UNUSED_ size_t bytes) {
#if FU_WITH_PLACE_MEMORY_ON_DOMAIN
    auto const &node = global_topology.memory_domain_at(static_cast<fu::memory_domain_index_t>(memory_domain_index));
    fu::memory_domain_allocator_t allocator(node.memory_domain_id);
    allocator.deallocate(reinterpret_cast<char *>(pointer), bytes);
#else
    std::free(pointer);
#endif
}

#pragma endregion Memory

#pragma region Lifetime

/**
 *  @brief Cross-platform aligned memory allocation.
 *  @note Returns nullptr on failure, never throws exceptions.
 */
inline void *fu_aligned_malloc(std::size_t size, std::size_t alignment) noexcept {
#if FU_ON_WINDOWS
    return _aligned_malloc(size, alignment);
#elif FU_ON_POSIX
    void *ptr = nullptr;
    return (posix_memalign(&ptr, alignment, size) == 0) ? ptr : nullptr;
#else
    return ::operator new(size, std::align_val_t {alignment}, std::nothrow);
#endif
}

/**
 *  @brief Cross-platform aligned memory deallocation.
 *  @note Matches fu_aligned_malloc - must use same alignment value.
 */
inline void fu_aligned_free(void *ptr, FU_MAYBE_UNUSED_ std::size_t alignment) noexcept {
#if FU_ON_WINDOWS
    _aligned_free(ptr);
#elif FU_ON_POSIX
    std::free(ptr);
#else
    ::operator delete(ptr, std::align_val_t {alignment}, std::nothrow);
#endif
}

fu_pool_t *fu_pool_new(FU_MAYBE_UNUSED_ char const *name, fu_capabilities_t allowed) {
    if (!globals_initialize()) return nullptr;

    // Intersect the machine's capabilities with the caller's allow-mask, then run the same
    // priority cascade over the result. Clearing a waiter bit drops to the next-lower waiter;
    // clearing `numa_aware` gates out the distributed block below and forces the basic pool.
    fu::capabilities_t const effective =
        static_cast<fu::capabilities_t>(global_capabilities & static_cast<fu::capabilities_t>(allowed));

    opaque_pool_t *opaque =
        static_cast<opaque_pool_t *>(fu_aligned_malloc(sizeof(opaque_pool_t), alignof(opaque_pool_t)));
    if (!opaque) return nullptr;

    // Best case, use the NUMA-aware distributed pool
#if FU_WITH_COLOCATE_POOLS_ON_DOMAIN
    // Only take the NUMA-aware distributed pool when the mask still allows it.
    if (effective & fu::capability_place_memory_on_domain_k) {
        fu::machine_topology_t copied_topology;
        if (!copied_topology.try_assign(global_topology)) {
            fu_aligned_free(opaque, alignof(opaque_pool_t));
            return nullptr;
        }

#if FU_DETECT_ASM_YIELDS_
#if FU_DETECT_ARCH_X86_64_
        if (effective & fu::capability_x86_tpause_k) {
            new (opaque) opaque_pool_t(name, std::in_place_type<fu::distributed_pool<fu::x86_tpause_t>>, name,
                                       std::move(copied_topology));
            return reinterpret_cast<fu_pool_t *>(opaque);
        }
        if (effective & fu::capability_x86_pause_k) {
            new (opaque) opaque_pool_t(name, std::in_place_type<fu::distributed_pool<fu::x86_pause_t>>, name,
                                       std::move(copied_topology));
            return reinterpret_cast<fu_pool_t *>(opaque);
        }
#endif
#if FU_DETECT_ARCH_ARM64_
        if (effective & fu::capability_arm64_wfet_k) {
            new (opaque) opaque_pool_t(name, std::in_place_type<fu::distributed_pool<fu::arm64_wfet_t>>, name,
                                       std::move(copied_topology));
            return reinterpret_cast<fu_pool_t *>(opaque);
        }
        if (effective & fu::capability_arm64_yield_k) {
            new (opaque) opaque_pool_t(name, std::in_place_type<fu::distributed_pool<fu::arm64_yield_t>>, name,
                                       std::move(copied_topology));
            return reinterpret_cast<fu_pool_t *>(opaque);
        }
#endif
#if FU_DETECT_ARCH_RISC5_
        if (effective & fu::capability_risc5_wrs_k) {
            new (opaque) opaque_pool_t(name, std::in_place_type<fu::distributed_pool<fu::risc5_wrs_t>>, name,
                                       std::move(copied_topology));
            return reinterpret_cast<fu_pool_t *>(opaque);
        }
        if (effective & fu::capability_risc5_pause_k) {
            new (opaque) opaque_pool_t(name, std::in_place_type<fu::distributed_pool<fu::risc5_pause_t>>, name,
                                       std::move(copied_topology));
            return reinterpret_cast<fu_pool_t *>(opaque);
        }
#endif
#endif // FU_DETECT_ASM_YIELDS_
       // No specific waiter survived the mask, but NUMA did: build the distributed pool with the
       // portable waiter rather than dropping to a single-domain basic pool.
        new (opaque) opaque_pool_t(name, std::in_place_type<fu::distributed_pool<fu::standard_yield_t>>, name,
                                   std::move(copied_topology));
        return reinterpret_cast<fu_pool_t *>(opaque);
    } // effective & numa_aware
#endif // FU_WITH_COLOCATE_POOLS_ON_DOMAIN

    // Common case of using modern hardware, but not having Linux installed
#if FU_DETECT_ASM_YIELDS_
#if FU_DETECT_ARCH_X86_64_
    if (effective & fu::capability_x86_tpause_k) {
        new (opaque) opaque_pool_t(name, std::in_place_type<fu::flat_pool<thread_allocator_t, fu::x86_tpause_t>>);
        return reinterpret_cast<fu_pool_t *>(opaque);
    }
    if (effective & fu::capability_x86_pause_k) {
        new (opaque) opaque_pool_t(name, std::in_place_type<fu::flat_pool<thread_allocator_t, fu::x86_pause_t>>);
        return reinterpret_cast<fu_pool_t *>(opaque);
    }
#endif
#if FU_DETECT_ARCH_ARM64_
    if (effective & fu::capability_arm64_wfet_k) {
        new (opaque) opaque_pool_t(name, std::in_place_type<fu::flat_pool<thread_allocator_t, fu::arm64_wfet_t>>);
        return reinterpret_cast<fu_pool_t *>(opaque);
    }
    if (effective & fu::capability_arm64_yield_k) {
        new (opaque) opaque_pool_t(name, std::in_place_type<fu::flat_pool<thread_allocator_t, fu::arm64_yield_t>>);
        return reinterpret_cast<fu_pool_t *>(opaque);
    }
#endif
#if FU_DETECT_ARCH_RISC5_
    if (effective & fu::capability_risc5_wrs_k) {
        new (opaque) opaque_pool_t(name, std::in_place_type<fu::flat_pool<thread_allocator_t, fu::risc5_wrs_t>>);
        return reinterpret_cast<fu_pool_t *>(opaque);
    }
    if (effective & fu::capability_risc5_pause_k) {
        new (opaque) opaque_pool_t(name, std::in_place_type<fu::flat_pool<thread_allocator_t, fu::risc5_pause_t>>);
        return reinterpret_cast<fu_pool_t *>(opaque);
    }
#endif
#endif // FU_DETECT_ASM_YIELDS_

    // Worst case, use the standard yield pool
    new (opaque) opaque_pool_t(name, std::in_place_type<fu::flat_pool<thread_allocator_t, fu::standard_yield_t>>);
    return reinterpret_cast<fu_pool_t *>(opaque);
}

/** @brief Safely cast `fu_pool_t*` to `opaque_pool_t*` avoiding alignment violation warnings. */
inline opaque_pool_t *upcast_pool(fu_pool_t *pool) noexcept {
    return std::launder(reinterpret_cast<opaque_pool_t *>(pool));
}

void fu_pool_delete(fu_pool_t *pool) {
    assert(pool != nullptr);

    opaque_pool_t *opaque = upcast_pool(pool);
    visit([](auto &variant) { variant.terminate(); }, opaque->variants);

    // Call the object's destructor and deallocate the memory
    opaque->~opaque_pool_t();
    fu_aligned_free(opaque, alignof(opaque_pool_t));
}

fu_bool_t fu_pool_spawn(fu_pool_t *pool, size_t threads, fu_caller_exclusivity_t c_exclusivity) {
    assert(pool != nullptr);
    assert(c_exclusivity == fu_caller_inclusive_k || c_exclusivity == fu_caller_exclusive_k);
    opaque_pool_t *opaque = upcast_pool(pool);
    auto exclusivity = c_exclusivity == fu_caller_inclusive_k ? fu::caller_inclusive_k : fu::caller_exclusive_k;
    return visit(
        [&](auto &variant) -> fu_bool_t {
#if FU_WITH_COLOCATE_POOLS_ON_DOMAIN
            using variant_t = std::remove_reference_t<decltype(variant)>;
            // A compute-domain pool binds to a specific compute domain rather than a bare thread count.
            if constexpr (pool_shape_of<variant_t>::kind_k == fu::pool_kind_t::colocated_k) {
                if (opaque->compute_domain_index >= global_topology.compute_domains_count()) return 0;
                return variant.try_spawn(global_topology.compute_domain_at(
                                             static_cast<fu::compute_domain_index_t>(opaque->compute_domain_index)),
                                         threads, exclusivity);
            }
            else
#endif
                return variant.try_spawn(threads, exclusivity);
        },
        opaque->variants);
}

fu_bool_t fu_pool_spawn_on(fu_pool_t *pool, FU_MAYBE_UNUSED_ size_t compute_domain_index, size_t threads,
                           fu_caller_exclusivity_t c_exclusivity) {
    assert(pool != nullptr);
    assert(c_exclusivity == fu_caller_inclusive_k || c_exclusivity == fu_caller_exclusive_k);
    opaque_pool_t *opaque = upcast_pool(pool);
    auto exclusivity = c_exclusivity == fu_caller_inclusive_k ? fu::caller_inclusive_k : fu::caller_exclusive_k;

#if FU_WITH_COLOCATE_POOLS_ON_DOMAIN
    if (compute_domain_index >= global_topology.compute_domains_count()) return 0;

    // The current variant already records the waiter chosen at creation, so a spawn-on pool honours
    // the same one without a separate stored mask. Read it before the destructor runs below.
    fu::capabilities_t const waiter = opaque->variants.waiter_;

    // `fu_pool_new` builds a distributed pool for the whole machine; rebuild it in place as a
    // single `colocated_pool` bound to this compute domain, then spawn it there.
    visit(
        [](auto &variant) {
            using variant_t = std::remove_reference_t<decltype(variant)>;
            variant.terminate();
            variant.~variant_t();
        },
        opaque->variants);

#if FU_DETECT_ASM_YIELDS_
#if FU_DETECT_ARCH_X86_64_
    if (waiter & fu::capability_x86_tpause_k)
        opaque->variants.construct<fu::colocated_pool<fu::x86_tpause_t>>(opaque->name);
    else if (waiter & fu::capability_x86_pause_k)
        opaque->variants.construct<fu::colocated_pool<fu::x86_pause_t>>(opaque->name);
    else
#endif
#if FU_DETECT_ARCH_ARM64_
        if (waiter & fu::capability_arm64_wfet_k)
        opaque->variants.construct<fu::colocated_pool<fu::arm64_wfet_t>>(opaque->name);
    else if (waiter & fu::capability_arm64_yield_k)
        opaque->variants.construct<fu::colocated_pool<fu::arm64_yield_t>>(opaque->name);
    else
#endif
#if FU_DETECT_ARCH_RISC5_
        if (waiter & fu::capability_risc5_wrs_k)
        opaque->variants.construct<fu::colocated_pool<fu::risc5_wrs_t>>(opaque->name);
    else if (waiter & fu::capability_risc5_pause_k)
        opaque->variants.construct<fu::colocated_pool<fu::risc5_pause_t>>(opaque->name);
    else
#endif
#endif // FU_DETECT_ASM_YIELDS_
        opaque->variants.construct<fu::colocated_pool<fu::standard_yield_t>>(opaque->name);

    opaque->compute_domain_index = compute_domain_index;
    return visit(
        [&](auto &variant) -> fu_bool_t {
            using variant_t = std::remove_reference_t<decltype(variant)>;
            if constexpr (pool_shape_of<variant_t>::kind_k == fu::pool_kind_t::colocated_k)
                return variant.try_spawn(
                    global_topology.compute_domain_at(static_cast<fu::compute_domain_index_t>(compute_domain_index)),
                    threads, exclusivity);
            else
                return 0;
        },
        opaque->variants);
#else
    // Without NUMA there is a single compute domain - spawning on it is spawning everywhere.
    if (compute_domain_index != 0) return 0;
    return visit([&](auto &variant) -> fu_bool_t { return variant.try_spawn(threads, exclusivity); }, opaque->variants);
#endif
}

fu_caller_exclusivity_t fu_pool_caller_exclusivity(fu_pool_t *pool) {
    assert(pool != nullptr);
    opaque_pool_t *opaque = upcast_pool(pool);
    return visit(
        [](auto &variant) {
            return variant.caller_exclusivity() == fu::caller_inclusive_k ? fu_caller_inclusive_k
                                                                          : fu_caller_exclusive_k;
        },
        opaque->variants);
}

size_t fu_pool_compute_domains_count(fu_pool_t *pool) {
    assert(pool != nullptr);
    opaque_pool_t *opaque = upcast_pool(pool);
    return visit([](auto &variant) { return variant.compute_domains_count(); }, opaque->variants);
}

size_t fu_pool_threads_count_in(fu_pool_t *pool, size_t compute_domain_index) {
    assert(pool != nullptr);
    opaque_pool_t *opaque = upcast_pool(pool);
    return visit([=](auto &variant) { return variant.threads_count(compute_domain_index); }, opaque->variants);
}

size_t fu_pool_threads_count(fu_pool_t *pool) {
    assert(pool != nullptr);
    opaque_pool_t *opaque = upcast_pool(pool);
    return visit([](auto &variant) { return variant.threads_count(); }, opaque->variants);
}

size_t fu_pool_locate_thread_in(fu_pool_t *pool, size_t global_thread_index, size_t compute_domain_index) {
    assert(pool != nullptr);
    opaque_pool_t *opaque = upcast_pool(pool);
    return visit([=](auto &variant) { return variant.thread_local_index(global_thread_index, compute_domain_index); },
                 opaque->variants);
}

void fu_pool_sleep(fu_pool_t *pool, size_t micros) {
    assert(pool != nullptr);
    opaque_pool_t *opaque = upcast_pool(pool);
    visit([=](auto &variant) { variant.sleep(micros); }, opaque->variants);
}

void fu_pool_terminate(fu_pool_t *pool) {
    assert(pool != nullptr);
    opaque_pool_t *opaque = upcast_pool(pool);
    visit([](auto &variant) { variant.terminate(); }, opaque->variants);
}

#pragma endregion Lifetime

#pragma region Primary API

void fu_pool_for_threads(fu_pool_t *pool, fu_for_threads_t callback, fu_lambda_context_t context) {
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

void fu_pool_for_slices(fu_pool_t *pool, size_t n, fu_for_slices_t callback, fu_lambda_context_t context) {
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

void fu_pool_for_n(fu_pool_t *pool, size_t n, fu_for_prongs_t callback, fu_lambda_context_t context) {
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

void fu_pool_for_n_dynamic(fu_pool_t *pool, size_t n, fu_for_prongs_t callback, fu_lambda_context_t context) {
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

fu_generation_t fu_pool_unsafe_for_threads(fu_pool_t *pool, fu_for_threads_t callback, fu_lambda_context_t context) {
    assert(pool != nullptr && callback != nullptr);
    opaque_pool_t *opaque = upcast_pool(pool);
    opaque->current_context = context;
    opaque->current_callback = callback;
    return visit([&](auto &variant) -> fu_generation_t { return variant.unsafe_for_threads(*opaque); },
                 opaque->variants);
}

fu_bool_t fu_pool_is_complete(fu_pool_t *pool, fu_generation_t generation) {
    assert(pool != nullptr);
    opaque_pool_t *opaque = upcast_pool(pool);
    return visit(
        [generation](auto &variant) -> fu_bool_t {
            return variant.is_complete(
                       static_cast<typename std::remove_reference_t<decltype(variant)>::generation_t>(generation))
                       ? 1
                       : 0;
        },
        opaque->variants);
}

void fu_pool_unsafe_join(fu_pool_t *pool, fu_generation_t generation) {
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
