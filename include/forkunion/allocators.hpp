/**
 *  @file allocators.hpp
 *  @brief NUMA-aware allocators and the domain-distributed containers built on them.
 *  @note Included by `<forkunion.hpp>`; not meant to be included on its own.
 *
 *  Everything here is about @b where bytes live: the per-platform allocators that pin a block to a
 *  memory domain, and the `replicated_array` / `sharded_array` containers that spread a sequence across
 *  domains. Execution - the pools - lives in `flat.hpp` and `distributed.hpp`.
 */
#pragma once
#include "topology.hpp" // `machine_topology`, `ram_page_size`, `memory_domain_id_t`

namespace ashvardanian {
namespace forkunion {

/**
 *  @brief Tries binding the given address range to a specific NUMA @p `memory_domain_id`.
 *  @retval true if binding succeeded, false otherwise.
 */
FU_MAYBE_UNUSED_ static inline bool linux_numa_bind(void *ptr, std::size_t size_bytes,
                                                    memory_domain_id_t memory_domain_id) noexcept {
#if FU_WITH_PLACE_MEMORY_ON_DOMAIN && FU_ON_LINUX
    // ! `MPOL_F_STATIC_NODES` is a @b mode flag - it belongs OR-ed into the policy, not in the trailing
    // ! `flags` argument, which only accepts `MPOL_MF_*`. Those flags are 0: this memory is freshly mapped
    // ! and unfaulted, so there is nothing to migrate, and `MPOL_MF_MOVE` would only demand a `CAP_SYS_NICE`
    // ! that a sandbox may refuse.
    return linux_bind_range_to_domain(ptr, size_bytes, memory_domain_id, mpol_bind_k | mpol_static_nodes_k);
#else
    fu_unused_(ptr);
    fu_unused_(size_bytes);
    fu_unused_(memory_domain_id);
    return false;
#endif // FU_WITH_PLACE_MEMORY_ON_DOMAIN
}

/**
 *  @brief Tries allocating uninitialized memory and binding it to a specific NUMA @p `memory_domain_id`.
 *  @retval nullptr if allocation failed or the page size is unsupported.
 *  @retval pointer to the allocated memory on success.
 */
FU_MAYBE_UNUSED_ static inline void *linux_numa_allocate(std::size_t size_bytes, std::size_t page_size_bytes,
                                                         memory_domain_id_t memory_domain_id) noexcept {
    assert(memory_domain_id >= 0 && "NUMA node ID must be non-negative");

#if FU_WITH_PLACE_MEMORY_ON_DOMAIN && FU_ON_LINUX

    // One path, not two: `numa_alloc_onnode` was only ever the `mmap` plus `mbind` below, so the base
    // page is that ladder without `MAP_HUGETLB` rather than a fast path beside it.
    int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS;
    if (page_size_bytes != ram_page_size()) {
#if FU_WITH_PLACE_HUGE_PAGES_ON_DOMAIN
        // Huge/explicit page sizes must be exact multiples
        assert(size_bytes % page_size_bytes == 0 && "Size must be a multiple of page size");

        // Make sure the page size makes sense for Linux
        if (page_size_bytes == page_size_4k_k) { mmap_flags |= MAP_HUGETLB; }
        else if (page_size_bytes == page_size_2m_k) { mmap_flags |= MAP_HUGETLB | static_cast<int>(MAP_HUGE_2MB); }
        else if (page_size_bytes == page_size_1g_k) { mmap_flags |= MAP_HUGETLB | static_cast<int>(MAP_HUGE_1GB); }
        else { return nullptr; } // ! Unsupported page size
#else
        return nullptr; // ! Every page size but the base one needs `MAP_HUGETLB`
#endif
    }

    // ? `mmap` rounds the length up to a page, as the `munmap` in `linux_numa_free` will - so the
    // ? rounding `numa_alloc_onnode`/`numa_free` hid still matches on both sides.
    void *result_ptr = ::mmap(nullptr, size_bytes, PROT_READ | PROT_WRITE, mmap_flags, -1, 0);
    if (result_ptr == MAP_FAILED) return nullptr; // ! Allocation failed

    // Binding is best-effort. The pages are already validly mapped; a kernel that refuses `mbind` -
    // qemu-user answers ENOSYS, a seccomp sandbox EPERM - still gave us memory, just placed by the
    // default policy rather than pinned to this domain. Discarding it would fail an allocation that in
    // fact succeeded; `runtime_capabilities()` is where a caller learns placement was unavailable.
    linux_numa_bind(result_ptr, size_bytes, memory_domain_id);
    return result_ptr;

#else
    fu_unused_(size_bytes);
    fu_unused_(page_size_bytes);
    fu_unused_(memory_domain_id);
    return nullptr;
#endif // FU_WITH_PLACE_MEMORY_ON_DOMAIN
}

FU_MAYBE_UNUSED_ static inline void linux_numa_free(void *ptr, std::size_t size_bytes) noexcept {
    assert(ptr != nullptr && "Pointer must not be null");
    assert(size_bytes > 0 && "Size must be greater than zero");
#if FU_WITH_PLACE_MEMORY_ON_DOMAIN && FU_ON_LINUX
    ::munmap(ptr, size_bytes); // ? What `numa_free` was, rounding the length up just as `mmap` did
#else
    fu_unused_(ptr);
    fu_unused_(size_bytes);
#endif
}

/**
 *  @brief STL-compatible allocator bound to a single memory domain, prioritizing huge pages.
 *
 *  A light-weight, but high-latency BLOB allocator, tied to a specific NUMA node ID.
 *  Every allocation is a system call to `mmap` and subsequent `mbind`, aligned to at
 *  least 4 KB page size.
 *
 *  @section C++ 23 Functionality
 *
 *  Whenever possible, the newer `allocate_at_least` API should be used to reduce the
 *  number of reallocations.
 */
template <typename value_type_ = char>
struct linux_numa_allocator {
    using value_type = value_type_;
    using size_type = std::size_t;
    using propagate_on_container_move_assignment = std::true_type;

  private:
    /** @brief The OS's id for the memory domain this pool allocates and runs on. */
    memory_domain_id_t memory_domain_id_ {-1};
    /** RAM page size in bytes, typically 4 KB. */
    size_type default_page_size_ {0};

  public:
    memory_domain_id_t memory_domain_id() const noexcept { return memory_domain_id_; }
    size_type default_page_size() const noexcept { return default_page_size_; }

    constexpr linux_numa_allocator() noexcept = default;
    explicit constexpr linux_numa_allocator(memory_domain_id_t id, size_type paging = ram_page_size()) noexcept
        : memory_domain_id_(id), default_page_size_(paging) {}

    template <typename other_type_>
    explicit constexpr linux_numa_allocator(linux_numa_allocator<other_type_> const &o) noexcept
        : memory_domain_id_(o.memory_domain_id()), default_page_size_(o.default_page_size()) {}

    /**
     *  @brief Allocates memory for at least `size` elements of `value_type`.
     *  @param[in] size The number of elements to allocate.
     *  @param[in] page_size_bytes The size of the memory page to allocate, must be a multiple of `sizeof(value_type)`.
     *  @return allocation_result with a pointer to the allocated memory and the number of elements allocated.
     *  @retval empty object if the allocation failed or the size is not a multiple of `sizeof(value_type)`.
     */
    allocation_result<value_type *, size_type> allocate_at_least(size_type size, size_type page_size_bytes) noexcept {
        size_type const size_bytes = size * sizeof(value_type);
        size_type const aligned_size_bytes = round_up_to_multiple(size_bytes, page_size_bytes);

        // Check if the new size is actually perfectly divisible by the `sizeof(value_type)`
        if (aligned_size_bytes % sizeof(value_type)) return {}; // ! Not a size multiple
        auto result_ptr = allocate(aligned_size_bytes / sizeof(value_type), page_size_bytes);
        if (!result_ptr) return {}; // ! Allocation failed
        size_type const pages_count = (page_size_bytes == 0) ? 0 : (aligned_size_bytes / page_size_bytes);
        return {result_ptr, size, aligned_size_bytes, pages_count};
    }

    /**
     *  @brief Allocates a memory for `size` elements of `value_type`.
     *  @param[in] size The number of elements to allocate.
     *  @param[in] page_size_bytes The size of the memory page to allocate, must be a multiple of `sizeof(value_type)`.
     *  @return allocation_result with a pointer to the allocated memory and the number of elements allocated.
     *  @retval empty object if the allocation failed or the size is not a multiple of `sizeof(value_type)`.
     */
    value_type *allocate(size_type size, size_type page_size_bytes) noexcept {
        size_type const size_bytes = size * sizeof(value_type);
        void *result_ptr = linux_numa_allocate(size_bytes, page_size_bytes, memory_domain_id_);
        if (!result_ptr) return {}; // ! Allocation failed
        return static_cast<value_type *>(result_ptr);
    }

    /**
     *  @brief Allocates memory for at least `size` elements of `value_type`.
     *  @param[in] size The number of elements to allocate.
     *  @return allocation_result with a pointer to the allocated memory and the number of elements allocated.
     *  @retval empty object if the allocation failed or the size is not a multiple of `sizeof(value_type)`.
     */
    allocation_result<value_type *, size_type> allocate_at_least(size_type size) noexcept {
        // Go through all of the typical Linux page sizes,
        // finding the largest one that makes sense and doesn't fail.
        size_type const size_bytes = size * sizeof(value_type);

        // Try 1 GB Huge Pages, for buffers larger than 2 GB
        if (size_bytes >= (2u * page_size_1g_k))
            if (auto result = allocate_at_least(size, page_size_1g_k); result) return result;

        // Try 2 MB Huge Pages, for buffers larger than 4 MB
        if (size_bytes >= (2u * page_size_2m_k))
            if (auto result = allocate_at_least(size, page_size_2m_k); result) return result;

        return allocate_at_least(size, default_page_size_);
    }

    /**
     *  @brief Allocates memory for `size` elements of `value_type`.
     *  @param[in] size The number of elements to allocate.
     *  @return allocation_result with a pointer to the allocated memory and the number of elements allocated.
     *  @retval empty object if the allocation failed or the size is not a multiple of `sizeof(value_type)`.
     */
    value_type *allocate(size_type size) noexcept {
        // Go through all of the typical Linux page sizes,
        // finding the largest one that makes sense and doesn't fail.
        size_type const size_bytes = size * sizeof(value_type);

        // ! Unlike `allocate_at_least`, we can't round the request up to the page boundary here:
        // ! the matching `deallocate(p, n)` only knows `n`, so it would unmap less than we mapped.
        // ! Huge Pages are therefore only an option for exact multiples of their size.

        // Try 1 GB Huge Pages, for buffers larger than 2 GB
        if (size_bytes >= (2u * page_size_1g_k) && size_bytes % page_size_1g_k == 0)
            if (auto result = allocate(size, page_size_1g_k); result) return result;

        // Try 2 MB Huge Pages, for buffers larger than 4 MB
        if (size_bytes >= (2u * page_size_2m_k) && size_bytes % page_size_2m_k == 0)
            if (auto result = allocate(size, page_size_2m_k); result) return result;

        return allocate(size, default_page_size_);
    }

    void deallocate(value_type *p, size_type n) noexcept { linux_numa_free(p, n * sizeof(value_type)); }

    template <typename other_type_>
    bool operator==(linux_numa_allocator<other_type_> const &o) const noexcept {
        return memory_domain_id_ == o.memory_domain_id_ && default_page_size_ == o.default_page_size_;
    }

    template <typename other_type_>
    bool operator!=(linux_numa_allocator<other_type_> const &o) const noexcept {
        return memory_domain_id_ != o.memory_domain_id_ || default_page_size_ != o.default_page_size_;
    }
};

using linux_numa_allocator_t = linux_numa_allocator<>;

/**
 *  @brief Maps one range of `domains * stride_bytes` and binds slice @p d to `domain_ids[d]`.
 *  @retval nullptr if the mapping or any slice binding failed, or the page size is unsupported.
 *
 *  Unlike `linux_numa_allocate`, which places a block on a single node, this reserves @b one contiguous
 *  virtual range and `mbind`s each equal-stride slice to its own node - a symmetric layout where the MMU
 *  then serves every slice from its local memory.
 */
FU_MAYBE_UNUSED_ static inline void *linux_symmetric_allocate(machine_topology_t const &topology,
                                                              std::size_t stride_bytes,
                                                              std::size_t page_size_bytes) noexcept {
#if FU_WITH_PLACE_MEMORY_ON_DOMAIN && FU_ON_LINUX
    std::size_t const domains = topology.memory_domains_count();
    if (domains == 0 || stride_bytes == 0) return nullptr;
    std::size_t const total_bytes = domains * stride_bytes;

    int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS;
    if (page_size_bytes != ram_page_size()) {
#if FU_WITH_PLACE_HUGE_PAGES_ON_DOMAIN
        if (page_size_bytes == page_size_2m_k) { mmap_flags |= MAP_HUGETLB | static_cast<int>(MAP_HUGE_2MB); }
        else if (page_size_bytes == page_size_1g_k) { mmap_flags |= MAP_HUGETLB | static_cast<int>(MAP_HUGE_1GB); }
        else { return nullptr; } // ! Unsupported page size
#else
        return nullptr; // ! Every page size but the base one needs `MAP_HUGETLB`
#endif
    }

    void *base = ::mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE, mmap_flags, -1, 0);
    if (base == MAP_FAILED) return nullptr; // ! Mapping failed

    // A single memory domain needs no placement - the one slice is the whole mapping on the only node.
    if (domains > 1)
        for (std::size_t domain = 0; domain != domains; ++domain) {
            memory_domain_id_t const memory_domain_id =
                topology.memory_domain_at(static_cast<memory_domain_index_t>(domain)).memory_domain_id;
            void *slice = static_cast<char *>(base) + domain * stride_bytes;
            // Best-effort, as in `linux_numa_allocate`: the slice is validly mapped, and a kernel that
            // refuses `mbind` still gave us distinct memory - default placement, not a failed allocation.
            linux_numa_bind(slice, stride_bytes, memory_domain_id);
        }
    return base;
#else
    fu_unused_(topology);
    fu_unused_(stride_bytes);
    fu_unused_(page_size_bytes);
    return nullptr;
#endif // FU_WITH_PLACE_MEMORY_ON_DOMAIN
}

FU_MAYBE_UNUSED_ static inline void linux_symmetric_free(void *ptr, std::size_t total_bytes) noexcept {
#if FU_WITH_PLACE_MEMORY_ON_DOMAIN && FU_ON_LINUX
    if (ptr) ::munmap(ptr, total_bytes);
#else
    fu_unused_(ptr);
    fu_unused_(total_bytes);
#endif
}

/**
 *  @brief Allocator for a @b symmetric mapping - one range striped across every memory domain.
 *
 *  A sibling of `linux_numa_allocator`, but instead of a block on a single node it returns a
 *  `symmetric_allocation_result` carrying the base pointer @b and the per-slice `stride_bytes`, so
 *  slice `d` of element `i` lives at `ptr + d * stride_bytes + i * sizeof(value_type)`. It borrows the
 *  @ref machine_topology it stripes across; hand it the one you harvested and let it outlive the mapping.
 */
template <typename value_type_ = char>
struct linux_symmetric_allocator {
    using value_type = value_type_;
    using size_type = std::size_t;
    using allocation_type = symmetric_allocation_result<value_type, size_type>;

  private:
    /** @brief The topology whose domains the mapping stripes across; borrowed, not owned. */
    machine_topology_t const *topology_ {nullptr};
    /** @brief RAM page size in bytes, typically 4 KB. */
    size_type default_page_size_ {0};

  public:
    machine_topology_t const *topology() const noexcept { return topology_; }
    size_type default_page_size() const noexcept { return default_page_size_; }

    constexpr linux_symmetric_allocator() noexcept = default;
    explicit linux_symmetric_allocator(machine_topology_t const &topology, size_type paging = ram_page_size()) noexcept
        : topology_(&topology), default_page_size_(paging) {}

    template <typename other_type_>
    explicit constexpr linux_symmetric_allocator(linux_symmetric_allocator<other_type_> const &o) noexcept
        : topology_(o.topology()), default_page_size_(o.default_page_size()) {}

    /** @brief Allocates at least @p size elements @b per domain, page-aligning the per-slice stride. */
    allocation_type allocate_at_least(size_type size, size_type page_size_bytes) noexcept {
        if (!topology_) return {}; // ! No topology to stripe across
        size_type const stride_bytes = round_up_to_multiple(size * sizeof(value_type), page_size_bytes);
        void *base = linux_symmetric_allocate(*topology_, stride_bytes, page_size_bytes);
        if (!base) return {}; // ! Allocation failed
        size_type const domains = topology_->memory_domains_count();
        size_type const total_bytes = domains * stride_bytes;
        size_type const pages = (page_size_bytes == 0) ? 0 : (total_bytes / page_size_bytes);
        return {static_cast<value_type *>(base), size, stride_bytes, domains, total_bytes, pages};
    }

    /** @brief Allocates at least @p size elements per domain, trying the largest huge page that fits. */
    allocation_type allocate_at_least(size_type size) noexcept {
        if (!topology_) return {};
        size_type const size_bytes = size * sizeof(value_type);
        if (size_bytes >= (2u * page_size_1g_k))
            if (auto result = allocate_at_least(size, page_size_1g_k); result) return result;
        if (size_bytes >= (2u * page_size_2m_k))
            if (auto result = allocate_at_least(size, page_size_2m_k); result) return result;
        return allocate_at_least(size, default_page_size_);
    }

    void deallocate(allocation_type const &allocation) noexcept {
        linux_symmetric_free(allocation.ptr, allocation.bytes);
    }
};

using linux_symmetric_allocator_t = linux_symmetric_allocator<>;

/**
 *  @brief Restores the calling thread's memory-domain policy saved by `freebsd_domain_prefer`.
 *  @note A no-op where the save failed, so a `getdomain` the kernel refused never widens the policy.
 */
FU_MAYBE_UNUSED_ static inline void freebsd_domain_restore(FU_MAYBE_UNUSED_ void const *saved_set,
                                                           FU_MAYBE_UNUSED_ int saved_policy,
                                                           FU_MAYBE_UNUSED_ bool have_saved) noexcept {
#if FU_WITH_PLACE_MEMORY_ON_DOMAIN && FU_ON_FREEBSD
    if (!have_saved) return;
    ::cpuset_setdomain(CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, sizeof(domainset_t),
                       static_cast<domainset_t const *>(saved_set), saved_policy);
#else
    fu_unused_(saved_set);
    fu_unused_(saved_policy);
    fu_unused_(have_saved);
#endif
}

/**
 *  @brief First-touches every page of `[ptr, ptr + size_bytes)` so a PREFER policy binds them.
 *  @note Anonymous pages are already zero, so writing a zero changes nothing but the residency.
 */
FU_MAYBE_UNUSED_ static inline void freebsd_first_touch(FU_MAYBE_UNUSED_ void *ptr,
                                                        FU_MAYBE_UNUSED_ std::size_t size_bytes) noexcept {
#if FU_WITH_PLACE_MEMORY_ON_DOMAIN && FU_ON_FREEBSD
    std::size_t const stride = static_cast<std::size_t>(ram_page_size());
    if (stride == 0) return;
    for (std::size_t offset = 0; offset < size_bytes; offset += stride) static_cast<volatile char *>(ptr)[offset] = 0;
#else
    fu_unused_(ptr);
    fu_unused_(size_bytes);
#endif
}

/**
 *  @brief Points the calling thread's `domainset` at @p memory_domain_id with a PREFER policy.
 *  @retval true when the prior domainset was captured into @p saved_set / @p saved_policy for restore.
 *
 *  FreeBSD 12 removed `numa_setaffinity`, so there is no per-allocation `numa_alloc_onnode`: the memory
 *  domain is a @b thread policy. The caller sets it, first-touches the region so its pages fault on the
 *  preferred domain, then restores the saved policy - the domainset analogue of Linux's per-mapping `mbind`.
 */
FU_MAYBE_UNUSED_ static inline bool freebsd_domain_prefer(FU_MAYBE_UNUSED_ memory_domain_id_t memory_domain_id,
                                                          FU_MAYBE_UNUSED_ void *saved_set,
                                                          FU_MAYBE_UNUSED_ int *saved_policy) noexcept {
#if FU_WITH_PLACE_MEMORY_ON_DOMAIN && FU_ON_FREEBSD
    bool const have_saved = ::cpuset_getdomain(CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, sizeof(domainset_t),
                                               static_cast<domainset_t *>(saved_set), saved_policy) == 0;
    domainset_t target;
    DOMAINSET_ZERO(&target);
    DOMAINSET_SET(static_cast<int>(memory_domain_id), &target);
    ::cpuset_setdomain(CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, sizeof(target), &target, DOMAINSET_POLICY_PREFER);
    return have_saved;
#else
    fu_unused_(memory_domain_id);
    fu_unused_(saved_set);
    fu_unused_(saved_policy);
    return false;
#endif
}

/**
 *  @brief Tries allocating uninitialized memory and placing it on a specific NUMA @p `memory_domain_id`.
 *  @retval nullptr if allocation failed or the page size is unsupported.
 *  @retval pointer to the allocated memory on success.
 *  @sa `linux_numa_allocate` is the Linux counterpart; FreeBSD places by thread policy, not `mbind`.
 *
 *  Larger-than-base page sizes ask for `MAP_ALIGNED_SUPER` - an alignment hint, since FreeBSD's superpages
 *  are transparent and have no `MAP_HUGETLB`-style pool. It is best-effort: a failed super-aligned map
 *  retries with base pages so the allocator stays total, matching the Linux `nullptr`-on-unsupported contract.
 */
FU_MAYBE_UNUSED_ static inline void *freebsd_domain_allocate(std::size_t size_bytes, std::size_t page_size_bytes,
                                                             memory_domain_id_t memory_domain_id) noexcept {
    assert(memory_domain_id >= 0 && "NUMA node ID must be non-negative");
#if FU_WITH_PLACE_MEMORY_ON_DOMAIN && FU_ON_FREEBSD
    if (size_bytes == 0) return nullptr;

    int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS;
    bool huge_requested = false;
    if (page_size_bytes > static_cast<std::size_t>(ram_page_size())) {
#if FU_WITH_PLACE_HUGE_PAGES_ON_DOMAIN
        // ? Default superpage alignment; not a size-specific pool like Linux's MAP_HUGETLB
        mmap_flags |= MAP_ALIGNED_SUPER;
        huge_requested = true;
#else
        return nullptr; // ! Every page size but the base one needs the huge-page capability
#endif
    }

    domainset_t saved;
    int saved_policy = 0;
    bool const have_saved = freebsd_domain_prefer(memory_domain_id, &saved, &saved_policy);

    void *result_ptr = ::mmap(nullptr, size_bytes, PROT_READ | PROT_WRITE, mmap_flags, -1, 0);
    if (result_ptr == MAP_FAILED && huge_requested) {
        // Superpage alignment could not be satisfied; retry with base pages so the allocator stays total.
        mmap_flags &= ~MAP_ALIGNED_SUPER;
        result_ptr = ::mmap(nullptr, size_bytes, PROT_READ | PROT_WRITE, mmap_flags, -1, 0);
    }
    if (result_ptr != MAP_FAILED) freebsd_first_touch(result_ptr, size_bytes);

    freebsd_domain_restore(&saved, saved_policy, have_saved);
    return result_ptr == MAP_FAILED ? nullptr : result_ptr;
#else
    fu_unused_(size_bytes);
    fu_unused_(page_size_bytes);
    fu_unused_(memory_domain_id);
    return nullptr;
#endif
}

FU_MAYBE_UNUSED_ static inline void freebsd_domain_free(void *ptr, std::size_t size_bytes) noexcept {
#if FU_WITH_PLACE_MEMORY_ON_DOMAIN && FU_ON_FREEBSD
    if (ptr) ::munmap(ptr, size_bytes);
#else
    fu_unused_(ptr);
    fu_unused_(size_bytes);
#endif
}

/**
 *  @brief STL-compatible allocator bound to a single memory domain on FreeBSD.
 *  @sa `linux_numa_allocator` is the Linux counterpart; both satisfy the pool's allocator needs.
 *
 *  Mirrors `linux_numa_allocator`'s huge-page ladder, but every placement is a thread-policy PREFER plus
 *  a first-touch rather than a per-mapping `mbind`, and larger pages are the `MAP_ALIGNED_SUPER` hint.
 */
template <typename value_type_ = char>
struct freebsd_numa_allocator {
    using value_type = value_type_;
    using size_type = std::size_t;
    using propagate_on_container_move_assignment = std::true_type;

  private:
    memory_domain_id_t memory_domain_id_ {-1};
    size_type default_page_size_ {0};

  public:
    memory_domain_id_t memory_domain_id() const noexcept { return memory_domain_id_; }
    size_type default_page_size() const noexcept { return default_page_size_; }

    constexpr freebsd_numa_allocator() noexcept = default;
    explicit constexpr freebsd_numa_allocator(memory_domain_id_t id, size_type paging = ram_page_size()) noexcept
        : memory_domain_id_(id), default_page_size_(paging) {}

    template <typename other_type_>
    explicit constexpr freebsd_numa_allocator(freebsd_numa_allocator<other_type_> const &o) noexcept
        : memory_domain_id_(o.memory_domain_id()), default_page_size_(o.default_page_size()) {}

    allocation_result<value_type *, size_type> allocate_at_least(size_type size, size_type page_size_bytes) noexcept {
        size_type const size_bytes = size * sizeof(value_type);
        size_type const aligned_size_bytes = round_up_to_multiple(size_bytes, page_size_bytes);
        if (aligned_size_bytes % sizeof(value_type)) return {}; // ! Not a size multiple
        auto result_ptr = allocate(aligned_size_bytes / sizeof(value_type), page_size_bytes);
        if (!result_ptr) return {}; // ! Allocation failed
        size_type const pages_count = (page_size_bytes == 0) ? 0 : (aligned_size_bytes / page_size_bytes);
        return {result_ptr, size, aligned_size_bytes, pages_count};
    }

    value_type *allocate(size_type size, size_type page_size_bytes) noexcept {
        void *result_ptr = freebsd_domain_allocate(size * sizeof(value_type), page_size_bytes, memory_domain_id_);
        if (!result_ptr) return {}; // ! Allocation failed
        return static_cast<value_type *>(result_ptr);
    }

    allocation_result<value_type *, size_type> allocate_at_least(size_type size) noexcept {
        size_type const size_bytes = size * sizeof(value_type);
        if (size_bytes >= (2u * page_size_1g_k))
            if (auto result = allocate_at_least(size, page_size_1g_k); result) return result;
        if (size_bytes >= (2u * page_size_2m_k))
            if (auto result = allocate_at_least(size, page_size_2m_k); result) return result;
        return allocate_at_least(size, default_page_size_);
    }

    value_type *allocate(size_type size) noexcept {
        size_type const size_bytes = size * sizeof(value_type);
        // ! Like the Linux ladder, `deallocate(p, n)` only knows `n`, so exact-multiple huge pages only.
        if (size_bytes >= (2u * page_size_1g_k) && size_bytes % page_size_1g_k == 0)
            if (auto result = allocate(size, page_size_1g_k); result) return result;
        if (size_bytes >= (2u * page_size_2m_k) && size_bytes % page_size_2m_k == 0)
            if (auto result = allocate(size, page_size_2m_k); result) return result;
        return allocate(size, default_page_size_);
    }

    void deallocate(value_type *p, size_type n) noexcept { freebsd_domain_free(p, n * sizeof(value_type)); }

    template <typename other_type_>
    bool operator==(freebsd_numa_allocator<other_type_> const &o) const noexcept {
        return memory_domain_id_ == o.memory_domain_id() && default_page_size_ == o.default_page_size();
    }
    template <typename other_type_>
    bool operator!=(freebsd_numa_allocator<other_type_> const &o) const noexcept {
        return !(*this == o);
    }
};

using freebsd_numa_allocator_t = freebsd_numa_allocator<>;

/**
 *  @brief Maps one range of `domains * stride_bytes` and PREFER-touches slice @p d on `domain_ids[d]`.
 *  @retval nullptr if the mapping failed or the page size is unsupported.
 *  @sa `linux_symmetric_allocate` is the Linux counterpart; FreeBSD places each slice by thread policy.
 */
FU_MAYBE_UNUSED_ static inline void *freebsd_symmetric_allocate(machine_topology_t const &topology,
                                                                std::size_t stride_bytes,
                                                                std::size_t page_size_bytes) noexcept {
#if FU_WITH_PLACE_MEMORY_ON_DOMAIN && FU_ON_FREEBSD
    std::size_t const domains = topology.memory_domains_count();
    if (domains == 0 || stride_bytes == 0) return nullptr;
    std::size_t const total_bytes = domains * stride_bytes;

    int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS;
    bool huge_requested = false;
    if (page_size_bytes > static_cast<std::size_t>(ram_page_size())) {
#if FU_WITH_PLACE_HUGE_PAGES_ON_DOMAIN
        mmap_flags |= MAP_ALIGNED_SUPER;
        huge_requested = true;
#else
        return nullptr; // ! Every page size but the base one needs the huge-page capability
#endif
    }

    void *base = ::mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE, mmap_flags, -1, 0);
    if (base == MAP_FAILED && huge_requested) {
        mmap_flags &= ~MAP_ALIGNED_SUPER;
        base = ::mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE, mmap_flags, -1, 0);
    }
    if (base == MAP_FAILED) return nullptr; // ! Mapping failed

    // A single memory domain needs no placement - the one slice is the whole mapping on the only node.
    if (domains > 1) {
        domainset_t saved;
        int saved_policy = 0;
        bool const have_saved =
            ::cpuset_getdomain(CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, sizeof(saved), &saved, &saved_policy) == 0;
        for (std::size_t domain = 0; domain != domains; ++domain) {
            memory_domain_id_t const memory_domain_id =
                topology.memory_domain_at(static_cast<memory_domain_index_t>(domain)).memory_domain_id;
            domainset_t target;
            DOMAINSET_ZERO(&target);
            DOMAINSET_SET(static_cast<int>(memory_domain_id), &target);
            ::cpuset_setdomain(CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, sizeof(target), &target, DOMAINSET_POLICY_PREFER);
            freebsd_first_touch(static_cast<char *>(base) + domain * stride_bytes, stride_bytes);
        }
        freebsd_domain_restore(&saved, saved_policy, have_saved);
    }
    return base;
#else
    fu_unused_(topology);
    fu_unused_(stride_bytes);
    fu_unused_(page_size_bytes);
    return nullptr;
#endif
}

FU_MAYBE_UNUSED_ static inline void freebsd_symmetric_free(void *ptr, std::size_t total_bytes) noexcept {
#if FU_WITH_PLACE_MEMORY_ON_DOMAIN && FU_ON_FREEBSD
    if (ptr) ::munmap(ptr, total_bytes);
#else
    fu_unused_(ptr);
    fu_unused_(total_bytes);
#endif
}

/**
 *  @brief Allocator for a @b symmetric mapping on FreeBSD - one range striped across every memory domain.
 *  @sa `linux_symmetric_allocator` is the Linux counterpart; borrows the @ref machine_topology it stripes.
 */
template <typename value_type_ = char>
struct freebsd_symmetric_allocator {
    using value_type = value_type_;
    using size_type = std::size_t;
    using allocation_type = symmetric_allocation_result<value_type, size_type>;

  private:
    machine_topology_t const *topology_ {nullptr};
    size_type default_page_size_ {0};

  public:
    machine_topology_t const *topology() const noexcept { return topology_; }
    size_type default_page_size() const noexcept { return default_page_size_; }

    constexpr freebsd_symmetric_allocator() noexcept = default;
    explicit freebsd_symmetric_allocator(machine_topology_t const &topology,
                                         size_type paging = ram_page_size()) noexcept
        : topology_(&topology), default_page_size_(paging) {}

    template <typename other_type_>
    explicit constexpr freebsd_symmetric_allocator(freebsd_symmetric_allocator<other_type_> const &o) noexcept
        : topology_(o.topology()), default_page_size_(o.default_page_size()) {}

    allocation_type allocate_at_least(size_type size, size_type page_size_bytes) noexcept {
        if (!topology_) return {}; // ! No topology to stripe across
        size_type const stride_bytes = round_up_to_multiple(size * sizeof(value_type), page_size_bytes);
        void *base = freebsd_symmetric_allocate(*topology_, stride_bytes, page_size_bytes);
        if (!base) return {}; // ! Allocation failed
        size_type const domains = topology_->memory_domains_count();
        size_type const total_bytes = domains * stride_bytes;
        size_type const pages = (page_size_bytes == 0) ? 0 : (total_bytes / page_size_bytes);
        return {static_cast<value_type *>(base), size, stride_bytes, domains, total_bytes, pages};
    }

    allocation_type allocate_at_least(size_type size) noexcept {
        if (!topology_) return {};
        size_type const size_bytes = size * sizeof(value_type);
        if (size_bytes >= (2u * page_size_1g_k))
            if (auto result = allocate_at_least(size, page_size_1g_k); result) return result;
        if (size_bytes >= (2u * page_size_2m_k))
            if (auto result = allocate_at_least(size, page_size_2m_k); result) return result;
        return allocate_at_least(size, default_page_size_);
    }

    void deallocate(allocation_type const &allocation) noexcept {
        freebsd_symmetric_free(allocation.ptr, allocation.bytes);
    }
};

using freebsd_symmetric_allocator_t = freebsd_symmetric_allocator<>;

/**
 *  @brief Enables `SeLockMemoryPrivilege` for the current process, needed before large-page allocation.
 *  @retval true if the privilege is now held by the process token.
 *  @note This only @b enables a privilege the account already holds; the account must first be granted
 *      "Lock pages in memory" (Local Security Policy / `SeLockMemoryPrivilege`), typically by an admin.
 *      Call once at start-up, then construct a `windows_numa_allocator` with `large_pages = true`.
 */
FU_MAYBE_UNUSED_ static inline bool windows_enable_lock_memory_privilege() noexcept {
#if FU_ON_WINDOWS
    HANDLE token = nullptr;
    if (!::OpenProcessToken(::GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token)) return false;
    TOKEN_PRIVILEGES privileges = {};
    privileges.PrivilegeCount = 1;
    privileges.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    bool const enabled = ::LookupPrivilegeValue(nullptr, SE_LOCK_MEMORY_NAME, &privileges.Privileges[0].Luid) &&
                         ::AdjustTokenPrivileges(token, FALSE, &privileges, 0, nullptr, nullptr) &&
                         ::GetLastError() == ERROR_SUCCESS; // ! `AdjustTokenPrivileges` succeeds even when it did not
    ::CloseHandle(token);
    return enabled;
#else
    return false;
#endif
}

/**
 *  @brief Allocates uninitialized memory placed on a specific NUMA @p memory_domain_id on Windows.
 *  @param[in] large_pages Request `MEM_LARGE_PAGES`; the size is rounded up to `GetLargePageMinimum()`.
 *  @retval nullptr if allocation failed, the size is zero, or NUMA memory is unavailable.
 *
 *  `VirtualAllocExNuma` reserves and commits a range whose pages the kernel will fault in on the
 *  requested node - the Windows analogue of Linux's `mbind`, folded into the allocation call. Large
 *  pages need `SeLockMemoryPrivilege` (see `windows_enable_lock_memory_privilege`) and can still fail
 *  under memory fragmentation; a caller wanting a soft failure should retry with @p large_pages false.
 */
FU_MAYBE_UNUSED_ static inline void *windows_numa_allocate(std::size_t size_bytes, memory_domain_id_t memory_domain_id,
                                                           bool large_pages = false) noexcept {
    assert(memory_domain_id >= 0 && "NUMA node ID must be non-negative");
#if FU_WITH_PLACE_MEMORY_ON_DOMAIN && FU_ON_WINDOWS
    if (size_bytes == 0) return nullptr;
    DWORD allocation_type = MEM_RESERVE | MEM_COMMIT;
    if (large_pages) {
        SIZE_T const large_page_bytes = ::GetLargePageMinimum();
        if (large_page_bytes == 0) return nullptr; // ! Large pages unsupported on this system
        size_bytes = round_up_to_multiple(size_bytes, static_cast<std::size_t>(large_page_bytes));
        allocation_type |= MEM_LARGE_PAGES;
    }
    return ::VirtualAllocExNuma(::GetCurrentProcess(), nullptr, size_bytes, allocation_type, PAGE_READWRITE,
                                static_cast<DWORD>(memory_domain_id));
#else
    fu_unused_(size_bytes);
    fu_unused_(memory_domain_id);
    fu_unused_(large_pages);
    return nullptr;
#endif
}

FU_MAYBE_UNUSED_ static inline void windows_numa_free(void *ptr) noexcept {
#if FU_WITH_PLACE_MEMORY_ON_DOMAIN && FU_ON_WINDOWS
    if (ptr) ::VirtualFree(ptr, 0, MEM_RELEASE); // ? Size must be 0 with MEM_RELEASE
#else
    fu_unused_(ptr);
#endif
}

/**
 *  @brief Reserves one range of `domains * stride_bytes` and commits slice @p d on its own node.
 *  @retval nullptr if the reservation or any slice commit failed.
 *
 *  The Windows analogue of `linux_symmetric_allocate`: one `VirtualAlloc(MEM_RESERVE)` for the whole
 *  range, then a per-slice `VirtualAllocExNuma(MEM_COMMIT, node)`, so each slice's pages fault in on its
 *  own node. Base pages only - large pages cannot be reserved and committed in separate steps.
 */
FU_MAYBE_UNUSED_ static inline void *windows_symmetric_allocate(machine_topology_t const &topology,
                                                                std::size_t stride_bytes) noexcept {
#if FU_WITH_PLACE_MEMORY_ON_DOMAIN && FU_ON_WINDOWS
    std::size_t const domains = topology.memory_domains_count();
    if (domains == 0 || stride_bytes == 0) return nullptr;
    std::size_t const total_bytes = domains * stride_bytes;

    void *base = ::VirtualAlloc(nullptr, total_bytes, MEM_RESERVE, PAGE_READWRITE);
    if (!base) return nullptr; // ! Reservation failed

    for (std::size_t domain = 0; domain != domains; ++domain) {
        memory_domain_id_t const memory_domain_id =
            topology.memory_domain_at(static_cast<memory_domain_index_t>(domain)).memory_domain_id;
        void *slice = static_cast<char *>(base) + domain * stride_bytes;
        if (!::VirtualAllocExNuma(::GetCurrentProcess(), slice, stride_bytes, MEM_COMMIT, PAGE_READWRITE,
                                  static_cast<DWORD>(memory_domain_id))) {
            ::VirtualFree(base, 0, MEM_RELEASE); // ? A slice would not commit; release the reservation
            return nullptr;                      // ! Commit failed
        }
    }
    return base;
#else
    fu_unused_(topology);
    fu_unused_(stride_bytes);
    return nullptr;
#endif
}

FU_MAYBE_UNUSED_ static inline void windows_symmetric_free(void *ptr) noexcept {
#if FU_WITH_PLACE_MEMORY_ON_DOMAIN && FU_ON_WINDOWS
    if (ptr) ::VirtualFree(ptr, 0, MEM_RELEASE);
#else
    fu_unused_(ptr);
#endif
}

/**
 *  @brief STL-compatible allocator bound to a single memory domain on Windows, backed by `VirtualAllocExNuma`.
 *  @sa `linux_numa_allocator` is the Linux counterpart; both satisfy the pool's allocator needs.
 *
 *  Deliberately plainer than the Linux allocator: it exposes only `allocate`/`deallocate`, so
 *  `dynamic_padded_array` takes its ordinary `allocate(total)` path rather than the sized
 *  `allocate_at_least` one. Large pages are opt-in per allocator instance (`large_pages` ctor flag);
 *  they need `SeLockMemoryPrivilege` first (@sa `windows_enable_lock_memory_privilege`) and round every
 *  request up to `GetLargePageMinimum()`. The pool constructs the allocator without them, since its own
 *  state is small; a caller wanting large pages for bulk data opts in explicitly.
 */
template <typename value_type_ = char>
struct windows_numa_allocator {
    using value_type = value_type_;
    using size_type = std::size_t;
    using propagate_on_container_move_assignment = std::true_type;

  private:
    memory_domain_id_t memory_domain_id_ {-1};
    size_type default_page_size_ {0};
    bool large_pages_ {false};

  public:
    memory_domain_id_t memory_domain_id() const noexcept { return memory_domain_id_; }
    size_type default_page_size() const noexcept { return default_page_size_; }
    bool large_pages() const noexcept { return large_pages_; }

    constexpr windows_numa_allocator() noexcept = default;
    explicit windows_numa_allocator(memory_domain_id_t id, size_type paging = ram_page_size(),
                                    bool large_pages = false) noexcept
        : memory_domain_id_(id), default_page_size_(paging), large_pages_(large_pages) {}

    template <typename other_type_>
    explicit constexpr windows_numa_allocator(windows_numa_allocator<other_type_> const &o) noexcept
        : memory_domain_id_(o.memory_domain_id()), default_page_size_(o.default_page_size()),
          large_pages_(o.large_pages()) {}

    /**
     *  @brief Allocates memory for at least `size` elements of `value_type`.
     *  @param[in] size The number of elements to allocate.
     *  @return allocation_result with a pointer to the allocated memory and the number of elements allocated.
     *  @retval empty object if the allocation failed.
     *  @note Unlike `linux_numa_allocator` there is no huge-page ladder: `VirtualAllocExNuma` commits at
     *      the base page size, or the large-page size when `large_pages` is set on this allocator.
     */
    allocation_result<value_type *, size_type> allocate_at_least(size_type size) noexcept {
        size_type const page_size_bytes = default_page_size_ ? default_page_size_ : ram_page_size();
        size_type const aligned_size_bytes = round_up_to_multiple(size * sizeof(value_type), page_size_bytes);
        void *result_ptr = windows_numa_allocate(aligned_size_bytes, memory_domain_id_, large_pages_);
        if (!result_ptr) return {}; // ! Allocation failed
        size_type const pages_count = (page_size_bytes == 0) ? 0 : (aligned_size_bytes / page_size_bytes);
        return {static_cast<value_type *>(result_ptr), size, aligned_size_bytes, pages_count};
    }

    value_type *allocate(size_type size) noexcept {
        return static_cast<value_type *>(
            windows_numa_allocate(size * sizeof(value_type), memory_domain_id_, large_pages_));
    }

    void deallocate(value_type *p, size_type) noexcept { windows_numa_free(p); }

    template <typename other_type_>
    bool operator==(windows_numa_allocator<other_type_> const &o) const noexcept {
        return memory_domain_id_ == o.memory_domain_id() && default_page_size_ == o.default_page_size() &&
               large_pages_ == o.large_pages();
    }
    template <typename other_type_>
    bool operator!=(windows_numa_allocator<other_type_> const &o) const noexcept {
        return !(*this == o);
    }
};

using windows_numa_allocator_t = windows_numa_allocator<>;

/**
 *  @brief Allocator for a @b symmetric mapping on Windows - one reservation committed per node.
 *  @sa `linux_symmetric_allocator` is the Linux counterpart.
 *
 *  Reserves one range and commits each equal-stride slice on its own node via `VirtualAllocExNuma`, so
 *  slice `d` of element `i` lives at `ptr + d * stride_bytes + i * sizeof(value_type)`. Base pages only -
 *  large pages cannot be reserved then committed in slices. Borrows the @ref machine_topology it stripes
 *  across; hand it the one you harvested and let it outlive the mapping.
 */
template <typename value_type_ = char>
struct windows_symmetric_allocator {
    using value_type = value_type_;
    using size_type = std::size_t;
    using allocation_type = symmetric_allocation_result<value_type, size_type>;

  private:
    /** @brief The topology whose domains the mapping stripes across; borrowed, not owned. */
    machine_topology_t const *topology_ {nullptr};
    /** @brief RAM page size in bytes, typically 4 KB. */
    size_type default_page_size_ {0};

  public:
    machine_topology_t const *topology() const noexcept { return topology_; }
    size_type default_page_size() const noexcept { return default_page_size_; }

    constexpr windows_symmetric_allocator() noexcept = default;
    explicit windows_symmetric_allocator(machine_topology_t const &topology,
                                         size_type paging = ram_page_size()) noexcept
        : topology_(&topology), default_page_size_(paging) {}

    template <typename other_type_>
    explicit constexpr windows_symmetric_allocator(windows_symmetric_allocator<other_type_> const &o) noexcept
        : topology_(o.topology()), default_page_size_(o.default_page_size()) {}

    /** @brief Allocates at least @p size elements @b per domain, page-aligning the per-slice stride. */
    allocation_type allocate_at_least(size_type size) noexcept {
        if (!topology_) return {}; // ! No topology to stripe across
        size_type const page_size_bytes = default_page_size_ ? default_page_size_ : ram_page_size();
        size_type const stride_bytes = round_up_to_multiple(size * sizeof(value_type), page_size_bytes);
        void *base = windows_symmetric_allocate(*topology_, stride_bytes);
        if (!base) return {}; // ! Allocation failed
        size_type const domains = topology_->memory_domains_count();
        size_type const total_bytes = domains * stride_bytes;
        size_type const pages = (page_size_bytes == 0) ? 0 : (total_bytes / page_size_bytes);
        return {static_cast<value_type *>(base), size, stride_bytes, domains, total_bytes, pages};
    }

    void deallocate(allocation_type const &allocation) noexcept { windows_symmetric_free(allocation.ptr); }
};

using windows_symmetric_allocator_t = windows_symmetric_allocator<>;

/**
 *  @brief A heap-backed, cache-line-aligned stand-in for the NUMA allocators, where no domains exist.
 *
 *  Mirrors `linux_numa_allocator`'s interface so `domain_allocator_t` stays uniform: it ignores
 *  the memory domain and reports the base page size, letting the domain-aware code paths compile and
 *  run everywhere while degenerating to plain heap allocation. Every block is aligned to at least
 *  `default_alignment_k`, so the NUMA backends' page alignment and this fallback offer one alignment
 *  contract - over-aligned element types need no caller-side padding on any platform.
 */
template <typename value_type_ = char>
struct portable_aligned_allocator {
    using value_type = value_type_;
    using size_type = std::size_t;

  private:
    memory_domain_id_t memory_domain_id_ {-1};
    size_type default_page_size_ {ram_page_size()};

    static void *allocate_aligned(size_type bytes) noexcept {
        return ::operator new(bytes, std::align_val_t {default_alignment_k}, std::nothrow);
    }

  public:
    memory_domain_id_t memory_domain_id() const noexcept { return memory_domain_id_; }
    size_type default_page_size() const noexcept { return default_page_size_; }

    constexpr portable_aligned_allocator() noexcept = default;
    explicit constexpr portable_aligned_allocator(memory_domain_id_t id, size_type paging = ram_page_size()) noexcept
        : memory_domain_id_(id), default_page_size_(paging) {}
    template <typename other_type_>
    explicit constexpr portable_aligned_allocator(portable_aligned_allocator<other_type_> const &o) noexcept
        : memory_domain_id_(o.memory_domain_id()), default_page_size_(o.default_page_size()) {}

    /** @brief Allocates at least @p size elements, rounded up to the page size. */
    allocation_result<value_type *, size_type> allocate_at_least(size_type size) noexcept {
        size_type const page_size_bytes = default_page_size_ ? default_page_size_ : ram_page_size();
        size_type const aligned_size_bytes = round_up_to_multiple(size * sizeof(value_type), page_size_bytes);
        void *result_ptr = allocate_aligned(aligned_size_bytes);
        if (!result_ptr) return {}; // ! Allocation failed
        size_type const pages_count = (page_size_bytes == 0) ? 0 : (aligned_size_bytes / page_size_bytes);
        return {static_cast<value_type *>(result_ptr), size, aligned_size_bytes, pages_count};
    }

    /** @brief Allocates exactly @p size elements. */
    value_type *allocate(size_type size) noexcept {
        return static_cast<value_type *>(allocate_aligned(size * sizeof(value_type)));
    }

    void deallocate(value_type *p, size_type) noexcept { ::operator delete(p, std::align_val_t {default_alignment_k}); }
};

using portable_aligned_allocator_t = portable_aligned_allocator<>;

/**
 *  @brief A heap-backed stand-in for the symmetric allocators, where domains cannot be bound.
 *  @sa `linux_symmetric_allocator` / `windows_symmetric_allocator` are the domain-pinning siblings.
 *
 *  Mirrors their `allocate_at_least(size) -> symmetric_allocation_result` shape so the containers stay
 *  uniform, but the one contiguous block is merely `default_alignment_k`-aligned - no `mbind`, no
 *  `VirtualAllocExNuma`. The slice count still comes from the @ref machine_topology, so on a machine the
 *  platform reports as single-domain the mapping is one stride wide and the same addressing
 *  `ptr + d * stride_bytes + i * sizeof(value_type)` holds everywhere. Borrows the topology; hand it the
 *  one you harvested and let it outlive the mapping.
 */
template <typename value_type_ = char>
struct portable_symmetric_allocator {
    using value_type = value_type_;
    using size_type = std::size_t;
    using allocation_type = symmetric_allocation_result<value_type, size_type>;

  private:
    machine_topology_t const *topology_ {nullptr};
    size_type default_page_size_ {ram_page_size()};

  public:
    machine_topology_t const *topology() const noexcept { return topology_; }
    size_type default_page_size() const noexcept { return default_page_size_; }

    constexpr portable_symmetric_allocator() noexcept = default;
    explicit portable_symmetric_allocator(machine_topology_t const &topology,
                                          size_type paging = ram_page_size()) noexcept
        : topology_(&topology), default_page_size_(paging) {}
    template <typename other_type_>
    explicit constexpr portable_symmetric_allocator(portable_symmetric_allocator<other_type_> const &o) noexcept
        : topology_(o.topology()), default_page_size_(o.default_page_size()) {}

    /** @brief Allocates one `default_alignment_k`-aligned block of `domains * stride_bytes`, unbound. */
    allocation_type allocate_at_least(size_type size) noexcept {
        if (!topology_) return {}; // ! No topology to size the slice count
        size_type const page_size_bytes = default_page_size_ ? default_page_size_ : ram_page_size();
        size_type const stride_bytes = round_up_to_multiple(size * sizeof(value_type), page_size_bytes);
        size_type const domains = topology_->memory_domains_count();
        if (domains == 0 || stride_bytes == 0) return {};
        size_type const total_bytes = domains * stride_bytes;
        void *base = ::operator new(total_bytes, std::align_val_t {default_alignment_k}, std::nothrow);
        if (!base) return {}; // ! Allocation failed
        size_type const pages = (page_size_bytes == 0) ? 0 : (total_bytes / page_size_bytes);
        return {static_cast<value_type *>(base), size, stride_bytes, domains, total_bytes, pages};
    }

    void deallocate(allocation_type const &allocation) noexcept {
        ::operator delete(allocation.ptr, std::align_val_t {default_alignment_k});
    }
};

using portable_symmetric_allocator_t = portable_symmetric_allocator<>;

/**
 *  @brief The NUMA-placing allocator for this platform, or a `malloc`-backed one where there is none.
 *  @sa Selected as `colocated_pool::allocator_t` so the pool's own state lands on its node.
 */
#if FU_WITH_PLACE_MEMORY_ON_DOMAIN && FU_ON_LINUX
using domain_allocator_t = linux_numa_allocator_t;
#elif FU_WITH_PLACE_MEMORY_ON_DOMAIN && FU_ON_WINDOWS
using domain_allocator_t = windows_numa_allocator_t;
#elif FU_WITH_PLACE_MEMORY_ON_DOMAIN && FU_ON_FREEBSD
using domain_allocator_t = freebsd_numa_allocator_t;
#else
using domain_allocator_t = portable_aligned_allocator_t;
#endif

/**
 *  @brief The symmetric mapping allocator for this platform, or a heap-backed one where there is none.
 *  @sa Backs `replicated_array` / `sharded_array`; one range striped across every memory domain.
 */
#if FU_WITH_PLACE_MEMORY_ON_DOMAIN && FU_ON_LINUX
using symmetric_memory_allocator_t = linux_symmetric_allocator_t;
#elif FU_WITH_PLACE_MEMORY_ON_DOMAIN && FU_ON_WINDOWS
using symmetric_memory_allocator_t = windows_symmetric_allocator_t;
#elif FU_WITH_PLACE_MEMORY_ON_DOMAIN && FU_ON_FREEBSD
using symmetric_memory_allocator_t = freebsd_symmetric_allocator_t;
#else
using symmetric_memory_allocator_t = portable_symmetric_allocator_t;
#endif

#pragma region Distributed Arrays

/** @brief A minimal `std::span`-like non-owning view over a contiguous range - C++17 has no `std::span`. */
template <typename value_type_>
class span {
    value_type_ *data_ {nullptr};
    std::size_t size_ {0};

  public:
    using element_type = value_type_;
    using value_type = std::remove_cv_t<value_type_>;
    using size_type = std::size_t;
    using pointer = value_type_ *;
    using reference = value_type_ &;
    using iterator = value_type_ *;

    constexpr span() noexcept = default;
    constexpr span(value_type_ *data, std::size_t size) noexcept : data_(data), size_(size) {}

    /** @brief Widens a mutable span to a `const` one - the qualification conversion `std::span` allows. */
    template <typename other_type_,
              typename = std::enable_if_t<std::is_convertible<other_type_ (*)[], value_type_ (*)[]>::value>>
    constexpr span(span<other_type_> const &other) noexcept : data_(other.data()), size_(other.size()) {}

    constexpr value_type_ *data() const noexcept { return data_; }
    constexpr std::size_t size() const noexcept { return size_; }
    constexpr std::size_t size_bytes() const noexcept { return size_ * sizeof(value_type_); }
    constexpr bool empty() const noexcept { return size_ == 0; }

    constexpr value_type_ &operator[](std::size_t index) const noexcept { return data_[index]; }
    constexpr value_type_ &front() const noexcept { return data_[0]; }
    constexpr value_type_ &back() const noexcept { return data_[size_ - 1]; }

    constexpr value_type_ *begin() const noexcept { return data_; }
    constexpr value_type_ *end() const noexcept { return data_ + size_; }

    constexpr span subspan(std::size_t offset, std::size_t count) const noexcept { return {data_ + offset, count}; }
    constexpr span first(std::size_t count) const noexcept { return {data_, count}; }
    constexpr span last(std::size_t count) const noexcept { return {data_ + (size_ - count), count}; }
};

/**
 *  @brief One uninitialized replica of a sequence per memory domain - a domain-local buffer.
 *  @tparam value_type_ The element type; treated as raw storage - the container runs no constructors or destructors.
 *  @tparam allocator_type_ A symmetric mapping allocator - one range striped across the domains.
 *
 *  For read-shared data - graph adjacency, body positions. The logical length is @b `n`; the footprint is
 *  `n * memory_domains_count()`. It owns memory and hands out slices - the caller fills every replica and
 *  keeps them coherent. Backed by one symmetric mapping, so replica `d` of element `i` lives at
 *  `base() + d * stride_bytes() + i * sizeof(value_type)`.
 */
template <typename value_type_, typename allocator_type_ = symmetric_memory_allocator_t>
struct replicated_array {
    using value_type = value_type_;
    using allocator_type = allocator_type_;
    using span_type = span<value_type_>;
    using const_span_type = span<value_type_ const>;
    using symmetric_allocator_type =
        typename std::allocator_traits<allocator_type_>::template rebind_alloc<value_type_>;

  private:
    symmetric_allocation_result<value_type_> allocation_ {};
    /** @brief Logical length `n`, identical in every replica. */
    std::size_t size_ {0};

  public:
    replicated_array() noexcept = default;
    ~replicated_array() noexcept { reset(); }
    replicated_array(replicated_array &&other) noexcept : allocation_(other.allocation_), size_(other.size_) {
        other.allocation_ = {};
        other.size_ = 0;
    }
    replicated_array &operator=(replicated_array &&other) noexcept {
        if (this != &other) {
            reset();
            allocation_ = other.allocation_;
            size_ = other.size_;
            other.allocation_ = {};
            other.size_ = 0;
        }
        return *this;
    }
    replicated_array(replicated_array const &) = delete;
    replicated_array &operator=(replicated_array const &) = delete;

    /** @brief Frees the mapping. Elements are raw storage, so no destructors run. */
    void reset() noexcept {
        if (!allocation_) return;
        symmetric_allocator_type().deallocate(allocation_);
        allocation_ = {};
        size_ = 0;
    }

    /** @brief Allocates one uninitialized length-@p n replica per memory domain; the caller first-touches them. */
    bool try_resize_uninitialized(machine_topology_t const &topology, std::size_t n) noexcept {
        reset();
        if (n == 0) return true;
        symmetric_allocator_type allocator(topology);
        allocation_ = allocator.allocate_at_least(n);
        if (!allocation_) return false; // ! Allocation failed
        size_ = n;
        return true;
    }

    std::size_t size() const noexcept { return size_; }
    std::size_t memory_domains_count() const noexcept { return allocation_.domains; }
    bool empty() const noexcept { return size_ == 0; }
    value_type_ *base() const noexcept { return allocation_.ptr; }
    std::size_t stride_bytes() const noexcept { return allocation_.stride_bytes; }

    /** @brief One element of the replica on @p memory_domain. */
    value_type_ &at(memory_domain_index_t memory_domain, std::size_t local_index) noexcept {
        return allocation_.slice(static_cast<std::size_t>(memory_domain))[local_index];
    }
    value_type_ const &at(memory_domain_index_t memory_domain, std::size_t local_index) const noexcept {
        return allocation_.slice(static_cast<std::size_t>(memory_domain))[local_index];
    }

    /** @brief The whole replica living on @p memory_domain. */
    span_type on_memory_domain(memory_domain_index_t memory_domain) noexcept {
        return {allocation_.slice(static_cast<std::size_t>(memory_domain)), size_};
    }
    const_span_type on_memory_domain(memory_domain_index_t memory_domain) const noexcept {
        return {allocation_.slice(static_cast<std::size_t>(memory_domain)), size_};
    }
};

/**
 *  @brief A sequence partitioned across memory domains, each element stored once - a domain-local buffer.
 *  @tparam value_type_ The element type; treated as raw storage - the container runs no constructors or destructors.
 *  @tparam allocator_type_ A symmetric mapping allocator - one range striped across the domains.
 *
 *  For datasets too large to replicate. The logical length is @b `n`; the footprint is `n`. Each domain owns
 *  a @b contiguous logical segment of `segment() == ceil(n / memory_domains_count())` elements, so element
 *  @b `i` lives at `{i / segment(), i % segment()}` - see `location_of` - and a sequential scan of a domain's
 *  shard is sequential in memory. It owns memory and hands out slices; the caller fills them. Backed by one
 *  symmetric mapping whose uniform stride is that segment, so a shorter trailing shard leaves slots unused.
 */
template <typename value_type_, typename allocator_type_ = symmetric_memory_allocator_t>
struct sharded_array {
    using value_type = value_type_;
    using allocator_type = allocator_type_;
    using span_type = span<value_type_>;
    using const_span_type = span<value_type_ const>;
    using symmetric_allocator_type =
        typename std::allocator_traits<allocator_type_>::template rebind_alloc<value_type_>;

    /** @brief Where a logical element lives: which memory domain and its index within that shard. */
    struct location_t {
        memory_domain_index_t memory_domain {};
        std::size_t local_index {0};
    };

  private:
    symmetric_allocation_result<value_type_> allocation_ {};
    /** @brief Logical length `n`, summed across the shards. */
    std::size_t size_ {0};

  public:
    sharded_array() noexcept = default;
    ~sharded_array() noexcept { reset(); }
    sharded_array(sharded_array &&other) noexcept : allocation_(other.allocation_), size_(other.size_) {
        other.allocation_ = {};
        other.size_ = 0;
    }
    sharded_array &operator=(sharded_array &&other) noexcept {
        if (this != &other) {
            reset();
            allocation_ = other.allocation_;
            size_ = other.size_;
            other.allocation_ = {};
            other.size_ = 0;
        }
        return *this;
    }
    sharded_array(sharded_array const &) = delete;
    sharded_array &operator=(sharded_array const &) = delete;

    /** @brief Frees the mapping. Elements are raw storage, so no destructors run. */
    void reset() noexcept {
        if (!allocation_) return;
        symmetric_allocator_type().deallocate(allocation_);
        allocation_ = {};
        size_ = 0;
    }

    /** @brief Allocates uninitialized storage for @p n elements in contiguous per-domain segments. */
    bool try_resize_uninitialized(machine_topology_t const &topology, std::size_t n) noexcept {
        reset();
        if (n == 0) return true;
        std::size_t const domains = topology.memory_domains_count();
        if (domains == 0) return false;
        symmetric_allocator_type allocator(topology);
        allocation_ = allocator.allocate_at_least(div_ceil(n, domains));
        if (!allocation_) return false; // ! Allocation failed
        size_ = n;
        return true;
    }

    std::size_t size() const noexcept { return size_; }
    std::size_t memory_domains_count() const noexcept { return allocation_.domains; }
    bool empty() const noexcept { return size_ == 0; }
    value_type_ *base() const noexcept { return allocation_.ptr; }
    std::size_t stride_bytes() const noexcept { return allocation_.stride_bytes; }

    /** @brief The contiguous logical segment size per domain - `ceil(n / memory_domains_count())`. */
    std::size_t segment() const noexcept { return allocation_.count; }

    /** @brief How many elements the shard on @p memory_domain holds - a trailing shard may be shorter. */
    std::size_t length_on_memory_domain(memory_domain_index_t memory_domain) const noexcept {
        std::size_t const start = static_cast<std::size_t>(memory_domain) * allocation_.count;
        if (start >= size_) return 0;
        std::size_t const remaining = size_ - start;
        return remaining < allocation_.count ? remaining : allocation_.count;
    }

    /** @brief The memory domain and local index that store logical element @p logical_index.
     *  @note Requires a non-empty array - `segment()` is the divisor and is zero when empty. */
    location_t location_of(std::size_t logical_index) const noexcept {
        return {static_cast<memory_domain_index_t>(logical_index / allocation_.count),
                logical_index % allocation_.count};
    }
    /** @brief The logical index of the element at @p local_index on @p memory_domain - inverse of `location_of`. */
    std::size_t logical_index_of(memory_domain_index_t memory_domain, std::size_t local_index) const noexcept {
        return static_cast<std::size_t>(memory_domain) * allocation_.count + local_index;
    }

    /** @brief The single home of a logical element - freely mutable, unlike a replica. */
    value_type_ &at(memory_domain_index_t memory_domain, std::size_t local_index) noexcept {
        return allocation_.slice(static_cast<std::size_t>(memory_domain))[local_index];
    }
    value_type_ const &at(memory_domain_index_t memory_domain, std::size_t local_index) const noexcept {
        return allocation_.slice(static_cast<std::size_t>(memory_domain))[local_index];
    }

    /** @brief The whole shard living on @p memory_domain. */
    span_type on_memory_domain(memory_domain_index_t memory_domain) noexcept {
        return {allocation_.slice(static_cast<std::size_t>(memory_domain)), length_on_memory_domain(memory_domain)};
    }
    const_span_type on_memory_domain(memory_domain_index_t memory_domain) const noexcept {
        return {allocation_.slice(static_cast<std::size_t>(memory_domain)), length_on_memory_domain(memory_domain)};
    }
};

#pragma endregion Distributed Arrays

} // namespace forkunion
} // namespace ashvardanian
