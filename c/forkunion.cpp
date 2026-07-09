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

/**
 *  @brief Custom variant implementation to avoid MSVC `std::variant` alignment issues.
 *
 *  MSVC cannot handle alignas > 64 when objects are passed by value in `std::variant`.
 *  This custom implementation uses a tagged union with manual type management.
 *  @see https://github.com/ashvardanian/ForkUnion/issues/26
 */
struct pool_variants_t {

    // ? Helper to compute max size and alignment of types
    template <typename... types_>
    struct max_size_align {
        static constexpr std::size_t size_k = std::max({sizeof(types_)...});
        static constexpr std::size_t alignment_k = std::max({alignof(types_)...});
    };

    using pool_traits_t = max_size_align< //
#if FU_WITH_ASM_YIELDS_
#if FU_DETECT_ARCH_X86_64_
        fu::basic_pool<thread_allocator_t, fu::x86_pause_t>,  //
        fu::basic_pool<thread_allocator_t, fu::x86_tpause_t>, //
#endif
#if FU_DETECT_ARCH_ARM64_
        fu::basic_pool<thread_allocator_t, fu::arm64_yield_t>, //
        fu::basic_pool<thread_allocator_t, fu::arm64_wfet_t>,  //
#endif
#if FU_DETECT_ARCH_RISC5_
        fu::basic_pool<thread_allocator_t, fu::risc5_pause_t>, //
#endif
#endif // FU_WITH_ASM_YIELDS_

#if FU_ENABLE_NUMA
        fu::linux_compute_domain_pool<fu::standard_yield_t>, // ? Single-compute-domain pools
        fu::linux_distributed_pool<fu::standard_yield_t>,    // ? Whole-machine pools
#if FU_WITH_ASM_YIELDS_
#if FU_DETECT_ARCH_X86_64_
        fu::linux_compute_domain_pool<fu::x86_pause_t>,  //
        fu::linux_compute_domain_pool<fu::x86_tpause_t>, //
        fu::linux_distributed_pool<fu::x86_pause_t>,     //
        fu::linux_distributed_pool<fu::x86_tpause_t>,    //
#endif
#if FU_DETECT_ARCH_ARM64_
        fu::linux_compute_domain_pool<fu::arm64_yield_t>, //
        fu::linux_compute_domain_pool<fu::arm64_wfet_t>,  //
        fu::linux_distributed_pool<fu::arm64_yield_t>,    //
        fu::linux_distributed_pool<fu::arm64_wfet_t>,     //
#endif
#if FU_DETECT_ARCH_RISC5_
        fu::linux_compute_domain_pool<fu::risc5_pause_t>, //
        fu::linux_distributed_pool<fu::risc5_pause_t>,    //
#endif
#endif // FU_WITH_ASM_YIELDS_
#endif // FU_ENABLE_NUMA

        fu::basic_pool<thread_allocator_t, fu::standard_yield_t> //
        >;

    alignas(pool_traits_t::alignment_k) std::uint8_t storage_[pool_traits_t::size_k];
    fu::capabilities_t capabilities_ {fu::capabilities_unknown_k}; // ? Which pool type is stored

    pool_variants_t() = default;
    ~pool_variants_t() = default;

    template <typename pool_type_, typename... args_types_>
    pool_variants_t(std::in_place_type_t<pool_type_>, args_types_ &&...args) noexcept {
        construct<pool_type_>(std::forward<args_types_>(args)...);
    }

    template <typename pool_type_, typename... args_types_>
    void construct(args_types_ &&...args) noexcept {
        new (storage_) pool_type_(std::forward<args_types_>(args)...);

        // ? Set capabilities based on pool type
        capabilities_ = fu::capabilities_unknown_k;

        if constexpr (std::is_same_v<pool_type_, fu::basic_pool<thread_allocator_t, fu::standard_yield_t>>) {
            capabilities_ = fu::capabilities_unknown_k;
        }
#if FU_WITH_ASM_YIELDS_
#if FU_DETECT_ARCH_X86_64_
        else if constexpr (std::is_same_v<pool_type_, fu::basic_pool<thread_allocator_t, fu::x86_pause_t>>) {
            capabilities_ = fu::capability_x86_pause_k;
        }
        else if constexpr (std::is_same_v<pool_type_, fu::basic_pool<thread_allocator_t, fu::x86_tpause_t>>) {
            capabilities_ = fu::capability_x86_tpause_k;
        }
#endif
#if FU_DETECT_ARCH_ARM64_
        else if constexpr (std::is_same_v<pool_type_, fu::basic_pool<thread_allocator_t, fu::arm64_yield_t>>) {
            capabilities_ = fu::capability_arm64_yield_k;
        }
        else if constexpr (std::is_same_v<pool_type_, fu::basic_pool<thread_allocator_t, fu::arm64_wfet_t>>) {
            capabilities_ = fu::capability_arm64_wfet_k;
        }
#endif
#if FU_DETECT_ARCH_RISC5_
        else if constexpr (std::is_same_v<pool_type_, fu::basic_pool<thread_allocator_t, fu::risc5_pause_t>>) {
            capabilities_ = fu::capability_risc5_pause_k;
        }
#endif
#endif
#if FU_ENABLE_NUMA
        else if constexpr (std::is_same_v<pool_type_, fu::linux_compute_domain_pool<fu::standard_yield_t>>) {
            capabilities_ = fu::capability_compute_domain_k | fu::capability_numa_aware_k;
        }
        else if constexpr (std::is_same_v<pool_type_, fu::linux_distributed_pool<fu::standard_yield_t>>) {
            capabilities_ = fu::capability_numa_aware_k;
        }
#if FU_WITH_ASM_YIELDS_
#if FU_DETECT_ARCH_X86_64_
        else if constexpr (std::is_same_v<pool_type_, fu::linux_compute_domain_pool<fu::x86_pause_t>>) {
            capabilities_ = fu::capability_x86_pause_k | fu::capability_compute_domain_k | fu::capability_numa_aware_k;
        }
        else if constexpr (std::is_same_v<pool_type_, fu::linux_compute_domain_pool<fu::x86_tpause_t>>) {
            capabilities_ = fu::capability_x86_tpause_k | fu::capability_compute_domain_k | fu::capability_numa_aware_k;
        }
        else if constexpr (std::is_same_v<pool_type_, fu::linux_distributed_pool<fu::x86_pause_t>>) {
            capabilities_ = fu::capability_x86_pause_k | fu::capability_numa_aware_k;
        }
        else if constexpr (std::is_same_v<pool_type_, fu::linux_distributed_pool<fu::x86_tpause_t>>) {
            capabilities_ = fu::capability_x86_tpause_k | fu::capability_numa_aware_k;
        }
#endif
#if FU_DETECT_ARCH_ARM64_
        else if constexpr (std::is_same_v<pool_type_, fu::linux_compute_domain_pool<fu::arm64_yield_t>>) {
            capabilities_ =
                fu::capability_arm64_yield_k | fu::capability_compute_domain_k | fu::capability_numa_aware_k;
        }
        else if constexpr (std::is_same_v<pool_type_, fu::linux_compute_domain_pool<fu::arm64_wfet_t>>) {
            capabilities_ = fu::capability_arm64_wfet_k | fu::capability_compute_domain_k | fu::capability_numa_aware_k;
        }
        else if constexpr (std::is_same_v<pool_type_, fu::linux_distributed_pool<fu::arm64_yield_t>>) {
            capabilities_ = fu::capability_arm64_yield_k | fu::capability_numa_aware_k;
        }
        else if constexpr (std::is_same_v<pool_type_, fu::linux_distributed_pool<fu::arm64_wfet_t>>) {
            capabilities_ = fu::capability_arm64_wfet_k | fu::capability_numa_aware_k;
        }
#endif
#if FU_DETECT_ARCH_RISC5_
        else if constexpr (std::is_same_v<pool_type_, fu::linux_compute_domain_pool<fu::risc5_pause_t>>) {
            capabilities_ =
                fu::capability_risc5_pause_k | fu::capability_compute_domain_k | fu::capability_numa_aware_k;
        }
        else if constexpr (std::is_same_v<pool_type_, fu::linux_distributed_pool<fu::risc5_pause_t>>) {
            capabilities_ = fu::capability_risc5_pause_k | fu::capability_numa_aware_k;
        }
#endif
#endif
#endif
    }
};

// ? Custom visit function to replace std::visit
template <typename visitor_type_>
auto visit(visitor_type_ &&visitor, pool_variants_t &variants) {
    if (!(variants.capabilities_ & fu::capability_numa_aware_k)) {
        // ? Basic pools
        if (variants.capabilities_ == fu::capabilities_unknown_k) {
            return visitor(
                *reinterpret_cast<fu::basic_pool<thread_allocator_t, fu::standard_yield_t> *>(variants.storage_));
        }
#if FU_WITH_ASM_YIELDS_
#if FU_DETECT_ARCH_X86_64_
        else if (variants.capabilities_ == fu::capability_x86_pause_k) {
            return visitor(*reinterpret_cast<fu::basic_pool<thread_allocator_t, fu::x86_pause_t> *>(variants.storage_));
        }
        else if (variants.capabilities_ == fu::capability_x86_tpause_k) {
            return visitor(
                *reinterpret_cast<fu::basic_pool<thread_allocator_t, fu::x86_tpause_t> *>(variants.storage_));
        }
#endif
#if FU_DETECT_ARCH_ARM64_
        else if (variants.capabilities_ == fu::capability_arm64_yield_k) {
            return visitor(
                *reinterpret_cast<fu::basic_pool<thread_allocator_t, fu::arm64_yield_t> *>(variants.storage_));
        }
        else if (variants.capabilities_ == fu::capability_arm64_wfet_k) {
            return visitor(
                *reinterpret_cast<fu::basic_pool<thread_allocator_t, fu::arm64_wfet_t> *>(variants.storage_));
        }
#endif
#if FU_DETECT_ARCH_RISC5_
        else if (variants.capabilities_ == fu::capability_risc5_pause_k) {
            return visitor(
                *reinterpret_cast<fu::basic_pool<thread_allocator_t, fu::risc5_pause_t> *>(variants.storage_));
        }
#endif
#endif
    }
#if FU_ENABLE_NUMA
    else {
        // ? Single-compute_domain pool pinned to one NUMA node, with its best busy-wait yield
        if (variants.capabilities_ & fu::capability_compute_domain_k) {
#if FU_WITH_ASM_YIELDS_
#if FU_DETECT_ARCH_X86_64_
            if (variants.capabilities_ & fu::capability_x86_tpause_k)
                return visitor(*reinterpret_cast<fu::linux_compute_domain_pool<fu::x86_tpause_t> *>(variants.storage_));
            if (variants.capabilities_ & fu::capability_x86_pause_k)
                return visitor(*reinterpret_cast<fu::linux_compute_domain_pool<fu::x86_pause_t> *>(variants.storage_));
#endif
#if FU_DETECT_ARCH_ARM64_
            if (variants.capabilities_ & fu::capability_arm64_wfet_k)
                return visitor(*reinterpret_cast<fu::linux_compute_domain_pool<fu::arm64_wfet_t> *>(variants.storage_));
            if (variants.capabilities_ & fu::capability_arm64_yield_k)
                return visitor(
                    *reinterpret_cast<fu::linux_compute_domain_pool<fu::arm64_yield_t> *>(variants.storage_));
#endif
#if FU_DETECT_ARCH_RISC5_
            if (variants.capabilities_ & fu::capability_risc5_pause_k)
                return visitor(
                    *reinterpret_cast<fu::linux_compute_domain_pool<fu::risc5_pause_t> *>(variants.storage_));
#endif
#endif
            return visitor(*reinterpret_cast<fu::linux_compute_domain_pool<fu::standard_yield_t> *>(variants.storage_));
        }
        // ? NUMA-aware distributed pools spanning all nodes
        else if (variants.capabilities_ == fu::capability_numa_aware_k) {
            return visitor(*reinterpret_cast<fu::linux_distributed_pool<fu::standard_yield_t> *>(variants.storage_));
        }
#if FU_WITH_ASM_YIELDS_
#if FU_DETECT_ARCH_X86_64_
        else if (variants.capabilities_ == (fu::capability_x86_pause_k | fu::capability_numa_aware_k)) {
            return visitor(*reinterpret_cast<fu::linux_distributed_pool<fu::x86_pause_t> *>(variants.storage_));
        }
        else if (variants.capabilities_ == (fu::capability_x86_tpause_k | fu::capability_numa_aware_k)) {
            return visitor(*reinterpret_cast<fu::linux_distributed_pool<fu::x86_tpause_t> *>(variants.storage_));
        }
#endif
#if FU_DETECT_ARCH_ARM64_
        else if (variants.capabilities_ == (fu::capability_arm64_yield_k | fu::capability_numa_aware_k)) {
            return visitor(*reinterpret_cast<fu::linux_distributed_pool<fu::arm64_yield_t> *>(variants.storage_));
        }
        else if (variants.capabilities_ == (fu::capability_arm64_wfet_k | fu::capability_numa_aware_k)) {
            return visitor(*reinterpret_cast<fu::linux_distributed_pool<fu::arm64_wfet_t> *>(variants.storage_));
        }
#endif
#if FU_DETECT_ARCH_RISC5_
        else if (variants.capabilities_ == (fu::capability_risc5_pause_k | fu::capability_numa_aware_k)) {
            return visitor(*reinterpret_cast<fu::linux_distributed_pool<fu::risc5_pause_t> *>(variants.storage_));
        }
#endif
#endif
    }
#endif
    // ? Default fallback
    return visitor(*reinterpret_cast<fu::basic_pool<thread_allocator_t, fu::standard_yield_t> *>(variants.storage_));
}

// ? Trait matching any yield specialization of the single-compute_domain pool
template <typename pool_type_>
struct is_compute_domain_pool : std::false_type {};
#if FU_ENABLE_NUMA
template <typename yield_type_>
struct is_compute_domain_pool<fu::linux_compute_domain_pool<yield_type_>> : std::true_type {};
#endif

struct opaque_pool_t {
    pool_variants_t variants;
    fu_lambda_context_t current_context; // ? Current context for the unsafe callbacks
    fu_for_threads_t current_callback;   // ? Current callback for the unsafe callbacks
    size_t compute_domain_index {0};     // ? Target compute domain for a spawn-on pool, else 0

    template <typename pool_type_, typename... args_types_>
    opaque_pool_t(std::in_place_type_t<pool_type_> inplace, args_types_ &&...args) noexcept
        : variants(inplace, std::forward<args_types_>(args)...), current_context(nullptr), current_callback(nullptr) {}

    /** @brief A shim to redirect unsafe callbacks to the current context. */
    void operator()(fu::local_thread_t pinned) const noexcept {
        current_callback(current_context, pinned.thread, pinned.compute_domain);
    }
};

static bool global_initialized {false};
static fu::numa_topology_t global_numa_topology {};
static fu::capabilities_t global_capabilities {fu::capabilities_unknown_k};
static char global_capabilities_string[128] {};

bool globals_initialize(void) {
    if (global_initialized) return true;

#if FU_ENABLE_NUMA
    if (!global_numa_topology.try_harvest()) return false;
#endif

    fu::capabilities_t cpu_caps = fu::cpu_capabilities();
    fu::capabilities_t ram_caps = fu::ram_capabilities();

    global_capabilities = static_cast<fu::capabilities_t>(cpu_caps | ram_caps);
    global_initialized = true;

    // Now, populate the capabilities string:
    char *pos = global_capabilities_string;
    char *end = global_capabilities_string + sizeof(global_capabilities_string) - 1;
    pos += std::snprintf(pos, end - pos, "serial");

    // Start with base capability level
    if (global_capabilities & fu::capability_numa_aware_k) pos += std::snprintf(pos, end - pos, "+numa");
    if (global_capabilities & fu::capability_huge_pages_k) pos += std::snprintf(pos, end - pos, "+hp");
    if (global_capabilities & fu::capability_huge_pages_transparent_k) pos += std::snprintf(pos, end - pos, "+thp");

    // Add CPU-specific extensions
    if (global_capabilities & fu::capability_x86_pause_k) pos += std::snprintf(pos, end - pos, "+x86_pause");
    if (global_capabilities & fu::capability_x86_tpause_k) pos += std::snprintf(pos, end - pos, "+x86_tpause");
    if (global_capabilities & fu::capability_arm64_yield_k) pos += std::snprintf(pos, end - pos, "+arm64_yield");
    if (global_capabilities & fu::capability_arm64_wfet_k) pos += std::snprintf(pos, end - pos, "+arm64_wfet");
    if (global_capabilities & fu::capability_risc5_pause_k) pos += std::snprintf(pos, end - pos, "+risc5_pause");
    return true;
}

extern "C" {

int fu_version_major(void) { return FORKUNION_VERSION_MAJOR; }
int fu_version_minor(void) { return FORKUNION_VERSION_MINOR; }
int fu_version_patch(void) { return FORKUNION_VERSION_PATCH; }
int fu_numa_enabled(void) { return FU_ENABLE_NUMA; }

#pragma region - Metadata

char const *fu_capabilities_string(void) {
    if (!globals_initialize()) return nullptr;
    return &global_capabilities_string[0];
}

size_t fu_count_logical_cores(void) {
#if FU_ENABLE_NUMA
    if (!globals_initialize()) return 0;
    return global_numa_topology.threads_count();
#else
    return std::thread::hardware_concurrency();
#endif
}

size_t fu_count_compute_domains(void) {
#if FU_ENABLE_NUMA
    if (!globals_initialize()) return 0;
    return global_numa_topology.compute_domains_count();
#else
    return 1;
#endif
}

size_t fu_count_memory_domains(void) {
#if FU_ENABLE_NUMA
    if (!globals_initialize()) return 0;
    return global_numa_topology.memory_domains_count();
#else
    return 1;
#endif
}

size_t fu_count_compute_levels(void) {
#if FU_ENABLE_NUMA
    if (!globals_initialize()) return 0;
    return global_numa_topology.compute_levels_count();
#else
    return 1;
#endif
}

size_t fu_compute_level_in(FU_MAYBE_UNUSED_ size_t compute_domain_index) {
#if FU_ENABLE_NUMA
    if (!globals_initialize()) return 0;
    if (compute_domain_index >= global_numa_topology.compute_domains_count()) return 0;
    return global_numa_topology.compute_domain_at(compute_domain_index).compute_level;
#else
    return 0;
#endif
}

size_t fu_memory_level_in(FU_MAYBE_UNUSED_ size_t memory_domain_index) {
#if FU_ENABLE_NUMA
    if (!globals_initialize()) return 0;
    if (memory_domain_index >= global_numa_topology.memory_domains_count()) return 0;
    return 0; // ? All memory is one tier until HBM/CXL tier harvesting lands
#else
    return 0;
#endif
}

size_t fu_local_memory_of(FU_MAYBE_UNUSED_ size_t compute_domain_index) {
#if FU_ENABLE_NUMA
    if (!globals_initialize()) return 0;
    return global_numa_topology.local_memory_of(compute_domain_index);
#else
    return 0;
#endif
}

size_t fu_memory_distance(FU_MAYBE_UNUSED_ size_t compute_domain_index, FU_MAYBE_UNUSED_ size_t memory_domain_index) {
#if FU_ENABLE_NUMA
    if (!globals_initialize()) return 0;
    return global_numa_topology.distance(compute_domain_index, memory_domain_index);
#else
    return compute_domain_index == 0 && memory_domain_index == 0 ? 10 : 0;
#endif
}

size_t fu_volume_ram(void) { return fu::get_ram_total_volume(); }

size_t fu_volume_ram_in(FU_MAYBE_UNUSED_ size_t memory_domain_index) {
#if FU_ENABLE_NUMA
    if (!globals_initialize()) return 0;
    if (memory_domain_index >= global_numa_topology.nodes_count()) return 0;
    return global_numa_topology.node(memory_domain_index).memory_size;
#else
    return memory_domain_index == 0 ? fu::get_ram_total_volume() : 0;
#endif
}

size_t fu_volume_huge_pages_in(FU_MAYBE_UNUSED_ size_t memory_domain_index) {
#if FU_ENABLE_NUMA
    if (!globals_initialize()) return 0;
    if (memory_domain_index >= global_numa_topology.nodes_count()) return 0;
    size_t total_volume = 0;
    auto const &node = global_numa_topology.node(memory_domain_index);
    for (auto const &page_size : node.page_sizes) total_volume += page_size.bytes_per_page * page_size.free_pages;
    return total_volume;
#else
    return 0;
#endif
}

size_t fu_volume_huge_pages(void) {
#if FU_ENABLE_NUMA
    if (!globals_initialize()) return 0;
    size_t total_volume = 0;
    for (size_t memory_domain = 0; memory_domain < global_numa_topology.nodes_count(); ++memory_domain)
        total_volume += fu_volume_huge_pages_in(memory_domain);
    return total_volume;
#else
    return 0;
#endif
}

size_t fu_count_huge_pages_in(FU_MAYBE_UNUSED_ size_t memory_domain_index) {
#if FU_ENABLE_NUMA
    if (!globals_initialize()) return 0;
    if (memory_domain_index >= global_numa_topology.nodes_count()) return 0;
    size_t total_pages = 0;
    auto const &node = global_numa_topology.node(memory_domain_index);
    for (auto const &page_size : node.page_sizes) total_pages += page_size.free_pages;
    return total_pages;
#else
    return 0;
#endif
}

size_t fu_count_huge_pages(void) {
#if FU_ENABLE_NUMA
    if (!globals_initialize()) return 0;
    size_t total_pages = 0;
    for (size_t memory_domain = 0; memory_domain < global_numa_topology.nodes_count(); ++memory_domain)
        total_pages += fu_count_huge_pages_in(memory_domain);
    return total_pages;
#else
    return 0;
#endif
}

#pragma endregion - Metadata

#pragma region - Memory

void *fu_allocate_at_least_in(                                         //
    FU_MAYBE_UNUSED_ size_t memory_domain_index, size_t minimum_bytes, //
    size_t *allocated_bytes, size_t *bytes_per_page) {

#if FU_ENABLE_NUMA
    auto const &node = global_numa_topology.node(memory_domain_index);
    fu::linux_numa_allocator_t allocator(node.node_id);
    auto result = allocator.allocate_at_least(minimum_bytes);
    if (!result) return nullptr;
    *allocated_bytes = result.count;
    *bytes_per_page = result.bytes_per_page();
    return result.ptr;
#else
    auto result = std::malloc(minimum_bytes);
    if (!result) return nullptr;
    *allocated_bytes = minimum_bytes;
    *bytes_per_page = fu::get_ram_page_size();
    return result;
#endif
}

void *fu_allocate_in(FU_MAYBE_UNUSED_ size_t memory_domain_index, size_t bytes) {

#if FU_ENABLE_NUMA
    auto const &node = global_numa_topology.node(memory_domain_index);
    fu::linux_numa_allocator_t allocator(node.node_id);
    return allocator.allocate(bytes);
#else
    return std::malloc(bytes);
#endif
}

void fu_free_in(FU_MAYBE_UNUSED_ size_t memory_domain_index, void *pointer, FU_MAYBE_UNUSED_ size_t bytes) {
#if FU_ENABLE_NUMA
    auto const &node = global_numa_topology.node(memory_domain_index);
    fu::linux_numa_allocator_t allocator(node.node_id);
    allocator.deallocate(reinterpret_cast<char *>(pointer), bytes);
#else
    std::free(pointer);
#endif
}

#pragma endregion - Memory

#pragma region - Lifetime

/**
 *  @brief Cross-platform aligned memory allocation.
 *  @note Returns nullptr on failure, never throws exceptions.
 */
inline void *fu_aligned_malloc(std::size_t size, std::size_t alignment) noexcept {
#if defined(_MSC_VER)
    return _aligned_malloc(size, alignment);
#elif defined(__unix__) || defined(__unix) || defined(unix) || defined(__APPLE__)
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
inline void fu_aligned_free(void *ptr, std::size_t alignment) noexcept {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#elif defined(__unix__) || defined(__unix) || defined(unix) || defined(__APPLE__)
    std::free(ptr);
#else
    ::operator delete(ptr, std::align_val_t {alignment}, std::nothrow);
#endif
}

fu_pool_t *fu_pool_new(FU_MAYBE_UNUSED_ char const *name) {
    if (!globals_initialize()) return nullptr;

    opaque_pool_t *opaque =
        static_cast<opaque_pool_t *>(fu_aligned_malloc(sizeof(opaque_pool_t), alignof(opaque_pool_t)));
    if (!opaque) return nullptr;

    // Best case, use the NUMA-aware distributed pool
#if FU_ENABLE_NUMA
    fu::numa_topology_t copied_topology;
    if (!copied_topology.try_assign(global_numa_topology)) {
        fu_aligned_free(opaque, alignof(opaque_pool_t));
        return nullptr;
    }

#if FU_WITH_ASM_YIELDS_
#if FU_DETECT_ARCH_X86_64_
    if (global_capabilities & fu::capability_x86_tpause_k) {
        new (opaque) opaque_pool_t(std::in_place_type<fu::linux_distributed_pool<fu::x86_tpause_t>>, name,
                                   std::move(copied_topology));
        return reinterpret_cast<fu_pool_t *>(opaque);
    }
    if (global_capabilities & fu::capability_x86_pause_k) {
        new (opaque) opaque_pool_t(std::in_place_type<fu::linux_distributed_pool<fu::x86_pause_t>>, name,
                                   std::move(copied_topology));
        return reinterpret_cast<fu_pool_t *>(opaque);
    }
#endif
#if FU_DETECT_ARCH_ARM64_
    if (global_capabilities & fu::capability_arm64_wfet_k) {
        new (opaque) opaque_pool_t(std::in_place_type<fu::linux_distributed_pool<fu::arm64_wfet_t>>, name,
                                   std::move(copied_topology));
        return reinterpret_cast<fu_pool_t *>(opaque);
    }
    if (global_capabilities & fu::capability_arm64_yield_k) {
        new (opaque) opaque_pool_t(std::in_place_type<fu::linux_distributed_pool<fu::arm64_yield_t>>, name,
                                   std::move(copied_topology));
        return reinterpret_cast<fu_pool_t *>(opaque);
    }
#endif
#if FU_DETECT_ARCH_RISC5_
    if (global_capabilities & fu::capability_risc5_pause_k) {
        new (opaque) opaque_pool_t(std::in_place_type<fu::linux_distributed_pool<fu::risc5_pause_t>>, name,
                                   std::move(copied_topology));
        return reinterpret_cast<fu_pool_t *>(opaque);
    }
#endif
#endif // FU_WITH_ASM_YIELDS_
#endif // FU_ENABLE_NUMA

    // Common case of using modern hardware, but not having Linux installed
#if FU_WITH_ASM_YIELDS_
#if FU_DETECT_ARCH_X86_64_
    if (global_capabilities & fu::capability_x86_tpause_k) {
        new (opaque) opaque_pool_t(std::in_place_type<fu::basic_pool<thread_allocator_t, fu::x86_tpause_t>>);
        return reinterpret_cast<fu_pool_t *>(opaque);
    }
    if (global_capabilities & fu::capability_x86_pause_k) {
        new (opaque) opaque_pool_t(std::in_place_type<fu::basic_pool<thread_allocator_t, fu::x86_pause_t>>);
        return reinterpret_cast<fu_pool_t *>(opaque);
    }
#endif
#if FU_DETECT_ARCH_ARM64_
    if (global_capabilities & fu::capability_arm64_wfet_k) {
        new (opaque) opaque_pool_t(std::in_place_type<fu::basic_pool<thread_allocator_t, fu::arm64_wfet_t>>);
        return reinterpret_cast<fu_pool_t *>(opaque);
    }
    if (global_capabilities & fu::capability_arm64_yield_k) {
        new (opaque) opaque_pool_t(std::in_place_type<fu::basic_pool<thread_allocator_t, fu::arm64_yield_t>>);
        return reinterpret_cast<fu_pool_t *>(opaque);
    }
#endif
#if FU_DETECT_ARCH_RISC5_
    if (global_capabilities & fu::capability_risc5_pause_k) {
        new (opaque) opaque_pool_t(std::in_place_type<fu::basic_pool<thread_allocator_t, fu::risc5_pause_t>>);
        return reinterpret_cast<fu_pool_t *>(opaque);
    }
#endif
#endif // FU_WITH_ASM_YIELDS_

    // Worst case, use the standard yield pool
    new (opaque) opaque_pool_t(std::in_place_type<fu::basic_pool<thread_allocator_t, fu::standard_yield_t>>);
    return reinterpret_cast<fu_pool_t *>(opaque);
}

inline opaque_pool_t *upcast_pool(fu_pool_t *pool) noexcept; // ? Defined below, used by `fu_pool_spawn_on`

fu_bool_t fu_pool_spawn_on(fu_pool_t *pool, FU_MAYBE_UNUSED_ size_t compute_domain_index, size_t threads,
                           fu_caller_exclusivity_t c_exclusivity) {
    assert(pool != nullptr);
    assert(c_exclusivity == fu_caller_inclusive_k || c_exclusivity == fu_caller_exclusive_k);
    opaque_pool_t *opaque = upcast_pool(pool);
    auto exclusivity = c_exclusivity == fu_caller_inclusive_k ? fu::caller_inclusive_k : fu::caller_exclusive_k;

#if FU_ENABLE_NUMA
    if (compute_domain_index >= global_numa_topology.compute_domains_count()) return 0;

    // `fu_pool_new` builds a distributed pool for the whole machine; rebuild it in place as a
    // single `linux_compute_domain_pool` bound to this compute domain, then spawn it there.
    visit(
        [](auto &variant) {
            using variant_t = std::remove_reference_t<decltype(variant)>;
            variant.terminate();
            variant.~variant_t();
        },
        opaque->variants);

#if FU_WITH_ASM_YIELDS_
#if FU_DETECT_ARCH_X86_64_
    if (global_capabilities & fu::capability_x86_tpause_k)
        opaque->variants.construct<fu::linux_compute_domain_pool<fu::x86_tpause_t>>("forkunion");
    else if (global_capabilities & fu::capability_x86_pause_k)
        opaque->variants.construct<fu::linux_compute_domain_pool<fu::x86_pause_t>>("forkunion");
    else
#endif
#if FU_DETECT_ARCH_ARM64_
        if (global_capabilities & fu::capability_arm64_wfet_k)
        opaque->variants.construct<fu::linux_compute_domain_pool<fu::arm64_wfet_t>>("forkunion");
    else if (global_capabilities & fu::capability_arm64_yield_k)
        opaque->variants.construct<fu::linux_compute_domain_pool<fu::arm64_yield_t>>("forkunion");
    else
#endif
#if FU_DETECT_ARCH_RISC5_
        if (global_capabilities & fu::capability_risc5_pause_k)
        opaque->variants.construct<fu::linux_compute_domain_pool<fu::risc5_pause_t>>("forkunion");
    else
#endif
#endif // FU_WITH_ASM_YIELDS_
        opaque->variants.construct<fu::linux_compute_domain_pool<fu::standard_yield_t>>("forkunion");

    opaque->compute_domain_index = compute_domain_index;
    return visit(
        [&](auto &variant) -> fu_bool_t {
            using variant_t = std::remove_reference_t<decltype(variant)>;
            if constexpr (is_compute_domain_pool<variant_t>::value)
                return variant.try_spawn(global_numa_topology.compute_domain_at(compute_domain_index), threads,
                                         exclusivity);
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

size_t fu_count_logical_cores_in(FU_MAYBE_UNUSED_ size_t compute_domain_index) {
#if FU_ENABLE_NUMA
    if (!globals_initialize()) return 0;
    if (compute_domain_index >= global_numa_topology.compute_domains_count()) return 0;
    return global_numa_topology.compute_domain_at(compute_domain_index).core_count;
#else
    return compute_domain_index == 0 ? std::thread::hardware_concurrency() : 0;
#endif
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
            using variant_t = std::remove_reference_t<decltype(variant)>;
#if FU_ENABLE_NUMA
            // A compute-domain pool binds to a specific compute domain rather than a bare thread count.
            if constexpr (is_compute_domain_pool<variant_t>::value) {
                if (opaque->compute_domain_index >= global_numa_topology.compute_domains_count()) return 0;
                return variant.try_spawn(global_numa_topology.compute_domain_at(opaque->compute_domain_index), threads,
                                         exclusivity);
            }
            else
#endif
                return variant.try_spawn(threads, exclusivity);
        },
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

size_t fu_pool_count_compute_domains(fu_pool_t *pool) {
    assert(pool != nullptr);
    opaque_pool_t *opaque = upcast_pool(pool);
    return visit([](auto &variant) { return variant.compute_domains_count(); }, opaque->variants);
}

size_t fu_pool_count_threads(fu_pool_t *pool) {
    assert(pool != nullptr);
    opaque_pool_t *opaque = upcast_pool(pool);
    return visit([](auto &variant) { return variant.threads_count(); }, opaque->variants);
}

size_t fu_pool_count_threads_in(fu_pool_t *pool, size_t compute_domain_index) {
    assert(pool != nullptr);
    opaque_pool_t *opaque = upcast_pool(pool);
    return visit([=](auto &variant) { return variant.threads_count(compute_domain_index); }, opaque->variants);
}

size_t fu_pool_locate_thread_in(fu_pool_t *pool, size_t global_thread_index, size_t compute_domain_index) {
    assert(pool != nullptr);
    opaque_pool_t *opaque = upcast_pool(pool);
    return visit([=](auto &variant) { return variant.thread_local_index(global_thread_index, compute_domain_index); },
                 opaque->variants);
}

#pragma endregion - Lifetime

#pragma region - Primary API

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

#pragma endregion - Primary API

#pragma region - Flexible API

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

#pragma endregion - Flexible API
}
