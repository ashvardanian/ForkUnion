/**
 *  @file topology.hpp
 *  @brief The hardware description: memory domains, compute domains, and the topology that holds them.
 *  @note Included by `<forkunion.hpp>`; not meant to be included on its own.
 */
#pragma once
#include "capabilities.hpp"

namespace ashvardanian {
namespace forkunion {

/**
 *  @brief One page size the kernel offers, and how many pages of it exist.
 *
 *  A machine reports several: the base page every allocation uses by default, and whichever huge
 *  page sizes the hardware and kernel agree on. `available_pages` counts what was reserved, and
 *  `free_pages` what nobody has taken yet, so an allocator can tell "unsupported" from "exhausted".
 *  @sa `ram_capabilities`
 */
struct ram_page_setting_t {
    /** Huge page size in bytes, e.g. 4 KB, 2 MB, or 1 GB. */
    std::size_t bytes_per_page {0};
    /** Number of pages available for this size, 0 if not available. */
    std::size_t available_pages {0};
    /** Number of pages available and unused, 0 if not available. */
    std::size_t free_pages {0};
};

/**
 *  @brief Fetches the RAM page size in bytes.
 *  @retval The size of a memory page in bytes, typically 4096 on most systems.
 *  @note On Linux, this is the system page size, which may differ from Huge Pages sizes.
 */
FU_MAYBE_UNUSED_ static inline std::size_t get_ram_page_size() noexcept {
#if FU_ON_WINDOWS
    SYSTEM_INFO system_info;
    ::GetSystemInfo(&system_info);
    return static_cast<std::size_t>(system_info.dwPageSize);
#elif FU_WITH_NUMA_MEMORY
    return static_cast<std::size_t>(::numa_pagesize());
#elif defined(__unix__) || defined(__unix) || defined(unix) || defined(__APPLE__)
    return static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
#else
    return 4096;
#endif
}

/**
 *  @brief Fetches the total RAM amount available on the system in bytes.
 *  @retval Total system RAM in bytes, or 0 if detection fails.
 *  @note This function provides cross-platform detection of total physical memory.
 */
FU_MAYBE_UNUSED_ static inline std::size_t get_ram_total_volume() noexcept {
#if defined(__linux__)
    // On Linux, read from /proc/meminfo
    FILE *meminfo_file = ::fopen("/proc/meminfo", "r");
    if (!meminfo_file) return 0;

    char line[256];
    while (::fgets(line, sizeof(line), meminfo_file)) {
        if (::strncmp(line, "MemTotal:", 9) == 0) {
            std::size_t memory_kb = 0;
            if (::sscanf(line, "MemTotal: %zu kB", &memory_kb) == 1) {
                ::fclose(meminfo_file);
                return memory_kb * 1024; // Convert kB to bytes
            }
        }
    }
    ::fclose(meminfo_file);
    return 0;
#elif defined(__APPLE__)
    // On macOS, use sysctl
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    std::uint64_t memory_bytes = 0;
    std::size_t size = sizeof(memory_bytes);
    if (::sysctl(mib, 2, &memory_bytes, &size, nullptr, 0) == 0) return static_cast<std::size_t>(memory_bytes);
    return 0;
#elif defined(_WIN32)
    // On Windows, use GlobalMemoryStatusEx
    MEMORYSTATUSEX mem_status;
    mem_status.dwLength = sizeof(mem_status);
    if (::GlobalMemoryStatusEx(&mem_status)) return static_cast<std::size_t>(mem_status.ullTotalPhys);
    return 0;
#elif defined(__unix__) || defined(__unix) || defined(unix)
    // On other Unix systems, try sysconf
    long pages = ::sysconf(_SC_PHYS_PAGES);
    long page_size = ::sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) return static_cast<std::size_t>(pages) * static_cast<std::size_t>(page_size);
    return 0;
#else
    // Fallback: return 0 if platform is not supported
    return 0;
#endif
}

/**
 *  @brief Describes the configured & supported (by OS & CPU) memory pages sizes.
 *
 *  This class avoids HugeTLBfs in favor of a direct access to the @b `/sys` filesystem.
 *  Aside from fetching the stats, it also allows us to change settings if admin privileges
 *  are granted to running process.
 *
 *  @section Huge Pages & Transparent Huge Pages
 *
 *  Virtual Address Space (VAS) is divided into pages, typically 4 KB in size.
 *  Converting a virtual address to a physical address requires a page table lookup.
 *  Think of it as a hash table... and as everyone knows, hash table lookups and updates
 *  aren't free, so most chips have a "Translation Lookaside Buffer" @b (TLB) cache
 *  as part of the "Memory Management Unit" @b (MMU) to speed up the process.
 *
 *  To keep it fast, in Big Data applications, one would like to use larger pages,
 *  to reduce the number of distinct entries in the TLB cache. Going from 4 KB to
 *  2 MB or 1 GB "Huge Pages" @b (HPs), reduces the table size by 512 or 262K times,
 *  respectively.
 *
 *  To benefit from those, some applications rely on "Transparent Huge Pages" @b (THP),
 *  which are automatically allocated by the kernel. Such implicit behaviour isn't
 *  great for performance-oriented applications, so the `linux_numa_allocator` provides
 *  a @b `fetch_max_huge_size` API.
 *
 *  @see https://docs.kernel.org/admin-guide/mm/hugetlbpage.html
 */
template <std::size_t max_page_sizes_ = 4>
class ram_page_settings {
    static constexpr std::size_t max_page_sizes_k = max_page_sizes_;
    /** Huge page sizes in bytes; a machine offers a handful, so the storage is inline. */
    limited_array<ram_page_setting_t, max_page_sizes_k> sizes_ {};
    /** Total memory available on this NUMA node. */
    std::size_t total_memory_bytes_ {0};

  public:
    /**
     *  @brief Finds the largest Huge Pages size available for the given NUMA node.
     */
    ram_page_setting_t largest_free() const noexcept {
        if (sizes_.empty()) return {};
        ram_page_setting_t largest = sizes_[0];
        for (std::size_t i = 1; i < sizes_.size(); ++i)
            if (sizes_[i].free_pages > largest.free_pages) largest = sizes_[i];
        return largest;
    }

    /**
     *  @brief Fetches all available huge page sizes for the given NUMA node.
     *  @note Kernel support doesn't mean that pages of that size have a valid mount point.
     */
    bool try_harvest(FU_MAYBE_UNUSED_ numa_node_id_t node_id) noexcept {
        assert(node_id >= 0 && "NUMA node ID must be non-negative");

#if FU_WITH_HUGE_PAGES && FU_ON_WINDOWS
        // Windows exposes exactly one large-page size, and only when the caller holds the
        // `SeLockMemoryPrivilege`; there is no per-node pool to enumerate or reserve.
        fu_unused_(node_id);
        SIZE_T const large_page_bytes = ::GetLargePageMinimum();
        if (large_page_bytes == 0) return false; // ? Large pages unavailable on this system
        // ? Windows commits large pages on demand, with no reserved pool to report
        ram_page_setting_t only {};
        only.bytes_per_page = static_cast<std::size_t>(large_page_bytes);
        only.available_pages = 0;
        only.free_pages = 0;
        sizes_.clear();
        sizes_.try_push_back(only);
        total_memory_bytes_ = 0;
        return true;

#elif FU_WITH_HUGE_PAGES && FU_ON_LINUX

        sizes_.clear();

        // Build path to NUMA node's hugepages directory
        char hugepages_path[256];
        int path_result = std::snprintf(            //
            hugepages_path, sizeof(hugepages_path), //
            "/sys/devices/system/node/node%d/hugepages", node_id);
        if (path_result < 0 || static_cast<std::size_t>(path_result) >= sizeof(hugepages_path))
            return false; // ? Path too long

        DIR *hugepages_dir = ::opendir(hugepages_path);
        if (!hugepages_dir) return false; // ? Can't open NUMA node hugepages directory

        struct dirent *entry;
        while ((entry = ::readdir(hugepages_dir)) != nullptr && !sizes_.full()) {
            // Look for directories named "hugepages-*kB"
            if (entry->d_type != DT_DIR) continue;
            if (::strncmp(entry->d_name, "hugepages-", 10) != 0) continue;

            // Extract size from directory name (e.g., "hugepages-2048kB" -> 2048)
            char const *size_start = entry->d_name + 10; // ? Skip "hugepages-"
            char *size_end = nullptr;
            std::size_t bytes_per_page_kb = static_cast<std::size_t>(::strtoull(size_start, &size_end, 10));

            // Verify the suffix is "kB"
            if (!size_end || std::strcmp(size_end, "kB") != 0) continue;
            if (bytes_per_page_kb == 0) continue; // ? Invalid size

            std::size_t const bytes_per_page = bytes_per_page_kb * 1024;

            // Read NUMA-node-specific huge page statistics
            char nr_hugepages_path[512];
            char free_hugepages_path[512];

            path_result = std::snprintf(                      //
                nr_hugepages_path, sizeof(nr_hugepages_path), //
                "%s/%s/nr_hugepages", hugepages_path, entry->d_name);
            if (path_result < 0 || static_cast<std::size_t>(path_result) >= sizeof(nr_hugepages_path))
                continue; // ? Path too long

            path_result = std::snprintf(                          //
                free_hugepages_path, sizeof(free_hugepages_path), //
                "%s/%s/free_hugepages", hugepages_path, entry->d_name);
            if (path_result < 0 || static_cast<std::size_t>(path_result) >= sizeof(free_hugepages_path))
                continue; // ? Path too long

            // Read allocated huge pages count
            FILE *nr_file = ::fopen(nr_hugepages_path, "r");
            if (!nr_file) continue; // ? Can't read allocation count

            std::size_t allocated_pages = 0;
            std::size_t free_pages = 0;
            if (::fscanf(nr_file, "%zu", &allocated_pages) != 1) {
                ::fclose(nr_file);
                continue; // ? Failed to parse allocated count
            }
            ::fclose(nr_file);

            // Read free huge pages count
            FILE *free_file = ::fopen(free_hugepages_path, "r");
            if (free_file) {
                if (::fscanf(free_file, "%zu", &free_pages) != 1) {
                    free_pages = 0; // ? Assume none are free if parsing fails
                }
                ::fclose(free_file);
            }

            // Add to our list with NUMA node information
            ram_page_setting_t setting {};
            setting.bytes_per_page = bytes_per_page;
            setting.available_pages = allocated_pages;
            setting.free_pages = free_pages;
            sizes_.try_push_back(setting); // ? Guarded by `!sizes_.full()` above
        }
        ::closedir(hugepages_dir);

        // Read total memory for this NUMA node from meminfo
        char meminfo_path[256];
        path_result =
            std::snprintf(meminfo_path, sizeof(meminfo_path), "/sys/devices/system/node/node%d/meminfo", node_id);
        if (path_result > 0 && static_cast<std::size_t>(path_result) < sizeof(meminfo_path)) {
            FILE *meminfo_file = ::fopen(meminfo_path, "r");
            if (meminfo_file) {
                char line[256];
                while (::fgets(line, sizeof(line), meminfo_file)) {
                    if (::strncmp(line, "Node ", 5) == 0 && ::strstr(line, " MemTotal:")) {
                        // Parse line like "Node 0 MemTotal:    32768000 kB"
                        std::size_t memory_kb = 0;
                        if (::sscanf(line, "Node %*d MemTotal: %zu kB", &memory_kb) == 1) {
                            total_memory_bytes_ = memory_kb * 1024; // Convert kB to bytes
                            break;
                        }
                    }
                }
                ::fclose(meminfo_file);
            }
        }

        return true;
#else
        fu_unused_(node_id);
        return false;
#endif
    }

    std::size_t size() const noexcept { return sizes_.size(); }
    std::size_t total_memory_bytes() const noexcept { return total_memory_bytes_; }
    ram_page_setting_t const *begin() const noexcept { return sizes_.data(); }
    ram_page_setting_t const *end() const noexcept { return sizes_.data() + sizes_.size(); }
    ram_page_setting_t const &operator[](std::size_t const index) const noexcept {
        assert(index < sizes_.size() && "Index is out of bounds");
        return sizes_[index];
    }

    /**
     *  @brief Attempts to reserve huge pages of a specific size on the current NUMA node.
     *  @param[in] page_size_bytes The size of huge pages to reserve (must match an available size)
     *  @param[in] num_pages Number of pages to reserve
     *  @return true if reservation was successful, false otherwise
     *  @note Requires root privileges or appropriate capabilities
     */
    bool try_change(numa_node_id_t node_id, std::size_t page_size_bytes, std::size_t num_pages) noexcept {
        assert(node_id >= 0 && "NUMA node ID must be non-negative");

        // Find the matching page size entry
        std::size_t page_index = sizes_.size();
        for (std::size_t i = 0; i < sizes_.size(); ++i) {
            if (sizes_[i].bytes_per_page == page_size_bytes) {
                page_index = i;
                break;
            }
        }
        if (page_index >= sizes_.size()) return false; // ? Page size not found

        // Calculate the page size in kB for the directory name
        std::size_t const page_size_kb = page_size_bytes / 1024;

        // Build path to the nr_hugepages file
        char nr_hugepages_path[512];
        int const path_result = std::snprintf(                                        //
            nr_hugepages_path, sizeof(nr_hugepages_path),                             //
            "/sys/devices/system/node/node%d/hugepages/hugepages-%zukB/nr_hugepages", //
            node_id, page_size_kb);

        if (path_result < 0 || static_cast<std::size_t>(path_result) >= sizeof(nr_hugepages_path))
            return false; // ? Path too long

        // Write the new reservation count
        FILE *nr_file = ::fopen(nr_hugepages_path, "w");
        if (!nr_file) return false; // ? Can't open for writing (likely permissions issue)

        bool const update_success = (::fprintf(nr_file, "%zu", num_pages) > 0);
        ::fclose(nr_file);
        if (!update_success) return false; // ? Failed to write the number of pages

        // Refresh our internal state if write was successful
        return try_harvest(node_id);
    }
};

using ram_page_settings_t = ram_page_settings<>;

/**
 *  @brief Describes a NUMA node, containing its ID, memory size, and core IDs.
 *  @sa Views different slices of the `numa_topology` structure.
 */
template <std::size_t max_page_sizes_ = 4>
struct numa_node {
    static constexpr std::size_t max_page_sizes_k = max_page_sizes_;

    /** Unique NUMA node ID, in [0, numa_max_node()). */
    numa_node_id_t node_id {-1};
    /** Physical CPU socket ID. */
    numa_socket_id_t socket_id {-1};
    /** RAM volume in bytes. */
    std::size_t memory_size {0};
    /** Memory tier ordinal, sorted fastest-to-slowest (0 = fastest). */
    std::size_t memory_level {0};
    /** Pointer to the first core ID in the `core_ids` array. */
    numa_core_id_t const *first_core_id {nullptr};
    /** Number of items in the `core_ids` array. */
    std::size_t core_count {0};
    /** Huge page sizes available on this NUMA node. */
    ram_page_settings<max_page_sizes_k> page_sizes {};
};

using numa_node_t = numa_node<>;

/**
 *  @brief One bindable cluster of cores sharing a QoS class and locality - the compute axis.
 *  @sa `numa_node` is the memory axis; a `numa_topology` exposes both plus their affinity.
 *
 *  A compute domain is a contiguous run of same-capacity cores within a single NUMA node. It is
 *  what a pool binds to and the index a worker callback receives. Several compute domains may map
 *  to one memory domain (performance and efficiency cores sharing a memory controller), which is
 *  why compute and memory are separate axes rather than a single "colocation" cell.
 */
struct compute_domain_t {
    /** The NUMA node these cores live on. */
    numa_node_id_t node_id {-1};
    /** @brief Index of the local memory domain (this node). */
    /** Which memory domain these cores allocate from. */
    memory_domain_index_t memory_domain_index {};
    /** QoS ordinal, sorted least-to-most performant. */
    std::size_t compute_level {0};
    /**
     *  @brief Relative throughput of @b one core here; `capacity_unknown_k` when unavailable.
     *  @sa `compute_level` is a dense ordinal for grouping - never divide by it.
     *
     *  Unlike `compute_level`, this is a magnitude, so it may be summed and divided. Where the
     *  kernel publishes a per-core rating - such as the Linux scheduler's `cpu_capacity`, scaled so
     *  the fastest core present reads 1024 - it lands here. Platforms that rank cores without
     *  quantifying them leave this unknown, and callers weigh domains by `core_count` instead.
     */
    std::size_t capacity {0};
    /**
     *  @brief Bytes of the deepest cache private to this domain's cores; 0 when unknown.
     *
     *  Sizes a cache-resident chunk, which is a different question from how many chunks a domain
     *  should receive - cores of equal throughput may back onto very differently sized caches, so
     *  neither number can be derived from the other. Shared by every core in the domain.
     */
    std::size_t cache_bytes {0};
    /** Pointer to the first core ID in this domain. */
    numa_core_id_t const *first_core_id {nullptr};
    /** Number of cores in this domain. */
    std::size_t core_count {0};
};

/** Sentinel for `compute_domain_t::capacity` when the platform exposes no per-core throughput. */
static constexpr std::size_t capacity_unknown_k = 0;

#if FU_WITH_TOPOLOGY && FU_ON_LINUX
/**
 *  @brief Owns a `libnuma` CPU mask - the one harvest resource an allocator-aware array cannot hold.
 */
struct numa_cpumask_guard {
    struct bitmask *mask {nullptr};

    numa_cpumask_guard() noexcept : mask(::numa_allocate_cpumask()) {}
    numa_cpumask_guard(numa_cpumask_guard const &) = delete;
    numa_cpumask_guard &operator=(numa_cpumask_guard const &) = delete;
    ~numa_cpumask_guard() noexcept {
        if (mask) ::numa_free_cpumask(mask);
    }
};

/**
 *  @brief Clears from @p cpus every core that @p allowed does not hold.
 *  @note A node whose every core is masked away survives as a cpuless memory domain, which the rest
 *        of the harvest already models - its memory stays ours to allocate from.
 */
FU_MAYBE_UNUSED_ static inline void restrict_cpumask_to(struct bitmask *cpus, affinity_mask const &allowed) noexcept {
    for (std::size_t bit = 0; bit < cpus->size; ++bit)
        if (::numa_bitmask_isbitset(cpus, static_cast<unsigned int>(bit)) &&
            !allowed.contains(static_cast<numa_core_id_t>(bit)))
            ::numa_bitmask_clearbit(cpus, static_cast<unsigned int>(bit));
}
#endif

/**
 *  @brief Fetches the socket ID for a given CPU core.
 *  @param[in] core_id The CPU core ID to query.
 *  @retval Socket ID (>= 0) if successful.
 *  @retval -1 if failed.
 */
FU_MAYBE_UNUSED_ static inline numa_socket_id_t get_socket_id_for_core(
    FU_MAYBE_UNUSED_ numa_core_id_t core_id) noexcept {

    int socket_id = -1;

#if defined(__linux__)
    char socket_path[256];
    int path_result = std::snprintf(      //
        socket_path, sizeof(socket_path), //
        "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", core_id);
    if (path_result < 0 || static_cast<std::size_t>(path_result) >= sizeof(socket_path)) return -1; // ? Path too long

    FILE *socket_file = ::fopen(socket_path, "r");
    if (!socket_file) return -1; // ? Can't read socket info

    if (::fscanf(socket_file, "%d", &socket_id) != 1) socket_id = -1; // ? Failed to parse
    ::fclose(socket_file);
#endif

    return socket_id;
}

/**
 *  @brief Fetches the scheduler capacity of a CPU core, used to separate QoS classes.
 *  @retval A capacity value where larger means more performant, or 0 if unavailable.
 *
 *  Reads `/sys/devices/system/cpu/cpuN/cpu_capacity`, which the kernel populates from the
 *  Energy Model on ARM @b big.LITTLE/DynamIQ and from ITMT/Thread-Director on Intel @b hybrid
 *  chips (performance cores report ~1024, efficiency cores less). A return of 0 means the core
 *  is homogeneous or the kernel does not expose capacities - the whole node is then one class.
 */
FU_MAYBE_UNUSED_ static inline std::size_t get_capacity_for_core(FU_MAYBE_UNUSED_ numa_core_id_t core_id) noexcept {

    std::size_t capacity = 0;

#if defined(__linux__)
    char capacity_path[256];
    int path_result = std::snprintf(          //
        capacity_path, sizeof(capacity_path), //
        "/sys/devices/system/cpu/cpu%d/cpu_capacity", core_id);
    if (path_result < 0 || static_cast<std::size_t>(path_result) >= sizeof(capacity_path)) return 0; // ? Path too long

    FILE *capacity_file = ::fopen(capacity_path, "r");
    if (!capacity_file) return 0; // ? Homogeneous cores or unsupported kernel

    unsigned long long parsed = 0;
    if (::fscanf(capacity_file, "%llu", &parsed) == 1) capacity = static_cast<std::size_t>(parsed);
    ::fclose(capacity_file);
#endif

    return capacity;
}

/**
 *  @brief Whether @p node_id appears in a Linux range-list file such as "0", "0-3", or "0,2-4".
 *  @sa Used to map a NUMA node onto its kernel memory tier.
 */
FU_MAYBE_UNUSED_ static inline bool nodelist_contains(FU_MAYBE_UNUSED_ char const *path,
                                                      FU_MAYBE_UNUSED_ numa_node_id_t node_id) noexcept {
#if defined(__linux__)
    FILE *file = ::fopen(path, "r");
    if (!file) return false;

    char line[256];
    bool found = false;
    if (::fgets(line, sizeof(line), file)) {
        char const *cursor = line;
        while (*cursor) {
            char *next = nullptr;
            long const low = ::strtol(cursor, &next, 10);
            if (next == cursor) break; // ? No number left to parse
            long high = low;
            cursor = next;
            if (*cursor == '-') high = ::strtol(cursor + 1, &next, 10), cursor = next;
            if (node_id >= low && node_id <= high) {
                found = true;
                break;
            }
            while (*cursor == ',' || *cursor == ' ' || *cursor == '\n') ++cursor;
        }
    }
    ::fclose(file);
    return found;
#else
    return false;
#endif
}

/**
 *  @brief Fetches the kernel memory-tier ordinal for a NUMA node, used to separate memory levels.
 *  @retval A tier number where smaller means faster, or `numeric_limits<size_t>::max()` if unknown.
 *
 *  Scans `/sys/devices/virtual/memory_tiering/memory_tierN/nodelist`, which the kernel populates by
 *  abstract distance - HBM below DRAM, CXL and persistent memory above it. When the sysfs is absent
 *  (older kernels, no tiering) every node reports the sentinel and collapses to a single memory level.
 */
FU_MAYBE_UNUSED_ static inline std::size_t get_memory_tier_for_node(FU_MAYBE_UNUSED_ numa_node_id_t node_id) noexcept {
#if FU_WITH_TOPOLOGY && FU_ON_LINUX
    DIR *dir = ::opendir("/sys/devices/virtual/memory_tiering");
    if (!dir) return std::numeric_limits<std::size_t>::max();

    std::size_t tier = std::numeric_limits<std::size_t>::max();
    for (dirent *entry; (entry = ::readdir(dir)) != nullptr;) {
        unsigned parsed = 0;
        if (::sscanf(entry->d_name, "memory_tier%u", &parsed) != 1) continue;
        char path[256];
        int const written = std::snprintf(path, sizeof(path), //
                                          "/sys/devices/virtual/memory_tiering/%s/nodelist", entry->d_name);
        if (written < 0 || static_cast<std::size_t>(written) >= sizeof(path)) continue; // ? Path too long
        if (nodelist_contains(path, node_id)) {
            tier = parsed;
            break;
        }
    }
    ::closedir(dir);
    return tier;
#else
    return std::numeric_limits<std::size_t>::max();
#endif
}

/**
 *  @brief Reads one HMAT performance metric for the initiator-to-target NUMA edge, or 0 if unknown.
 *  @param metric_name A sysfs leaf: "read_bandwidth", "write_bandwidth", "read_latency", "write_latency".
 *  @retval Bandwidth in MB/s or latency in nanoseconds, or 0 when the machine exposes no HMAT table.
 *
 *  The kernel publishes per-node HMAT numbers under `access0` (nearest initiators) and `access1` (all
 *  initiators); we take the first access class that lists @p initiator among its initiators.
 */
FU_MAYBE_UNUSED_ static inline std::size_t read_hmat_metric(FU_MAYBE_UNUSED_ numa_node_id_t initiator,
                                                            FU_MAYBE_UNUSED_ numa_node_id_t target,
                                                            FU_MAYBE_UNUSED_ char const *metric_name) noexcept {
#if defined(__linux__)
    for (int access_class = 0; access_class < 2; ++access_class) {
        char initiator_path[320];
        int written = std::snprintf(initiator_path, sizeof(initiator_path),
                                    "/sys/devices/system/node/node%d/access%d/initiators/node%d", target, access_class,
                                    initiator);
        if (written < 0 || static_cast<std::size_t>(written) >= sizeof(initiator_path)) continue;
        if (::access(initiator_path, F_OK) != 0) continue; // ? Initiator is not in this access class

        char metric_path[320];
        written =
            std::snprintf(metric_path, sizeof(metric_path), "/sys/devices/system/node/node%d/access%d/initiators/%s",
                          target, access_class, metric_name);
        if (written < 0 || static_cast<std::size_t>(written) >= sizeof(metric_path)) continue;

        FILE *file = ::fopen(metric_path, "r");
        if (!file) continue;
        unsigned long long parsed = 0;
        std::size_t value = 0;
        if (::fscanf(file, "%llu", &parsed) == 1) value = static_cast<std::size_t>(parsed);
        ::fclose(file);
        if (value) return value;
    }
    return 0;
#else
    return 0;
#endif
}

#if FU_ON_WINDOWS
/**
 *  @brief One accumulator per (processor group, efficiency class): the union of that class's core masks
 *         within the group, and the largest private cache seen for it.
 *  @sa `try_harvest_windows` builds these from the processor-core and cache relationships, then reads
 *      them back per NUMA node - so it never keeps a per-processor scratch table.
 */
struct win_group_class_cell_t {
    KAFFINITY mask {0};          // ? OR of every core of this class in this group
    std::size_t cache_bytes {0}; // ? Largest L1/L2 (private) cache seen for those cores
};

/**
 *  @brief Detects whether this SDK's `NUMA_NODE_RELATIONSHIP` exposes the multi-group `GroupMasks[]`.
 *  @note Version macros are unreliable here - MinGW reports Windows 7 yet defines the member - so we
 *        probe the member itself. Older SDKs model a node as a single group, read by the fallback.
 */
template <typename numa_relationship_type_, typename = void>
struct win_numa_has_group_masks : std::false_type {};
template <typename numa_relationship_type_>
struct win_numa_has_group_masks<numa_relationship_type_,
                                std::void_t<decltype(std::declval<numa_relationship_type_ &>().GroupMasks)>>
    : std::true_type {};

/**
 *  @brief Invokes @p visitor(group, mask) for every processor group a NUMA @p node owns.
 *  @note Templated on the node type so the discarded `if constexpr` branch is dependent and only the
 *        supported member is ever compiled. A node spanning several groups (the largest servers) is
 *        thus enumerated in full on new SDKs, and read through its single group on old ones.
 */
template <typename numa_relationship_type_, typename visitor_type_>
static inline void win_numa_for_each_group(numa_relationship_type_ const &node, visitor_type_ &&visitor) noexcept {
    if constexpr (win_numa_has_group_masks<numa_relationship_type_>::value) {
        if (node.GroupCount == 0) return visitor(node.GroupMask.Group, node.GroupMask.Mask);
        for (WORD g = 0; g < node.GroupCount; ++g) visitor(node.GroupMasks[g].Group, node.GroupMasks[g].Mask);
    }
    else { visitor(node.GroupMask.Group, node.GroupMask.Mask); }
}

/**
 *  @brief Best-effort socket id for a NUMA @p node: the processor package that owns its cores.
 *  @retval @p fallback when package data is unavailable or no package matches.
 *  @note Packages and nodes are both few, so scanning the package buffer per node needs no scratch.
 */
FU_MAYBE_UNUSED_ static inline numa_socket_id_t win_socket_for_node( //
    BYTE const *package_buffer, DWORD package_len, NUMA_NODE_RELATIONSHIP const &node,
    numa_socket_id_t fallback) noexcept {
    if (!package_buffer) return fallback;
    numa_socket_id_t socket_index = 0;
    for (DWORD offset = 0; offset < package_len;) {
        auto const *record = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX const *>(package_buffer + offset);
        if (record->Relationship == RelationProcessorPackage) {
            bool intersects = false;
            PROCESSOR_RELATIONSHIP const &package = record->Processor;
            for (WORD g = 0; g < package.GroupCount; ++g) {
                GROUP_AFFINITY const package_affinity = package.GroupMask[g];
                win_numa_for_each_group(node, [&](WORD node_group, KAFFINITY node_mask) noexcept {
                    if (node_group == package_affinity.Group && (node_mask & package_affinity.Mask)) intersects = true;
                });
            }
            if (intersects) return socket_index;
            socket_index += 1;
        }
        if (record->Size == 0) break;
        offset += record->Size;
    }
    return fallback;
}
#endif // FU_ON_WINDOWS

#if defined(__APPLE__)
/**
 *  @brief Reads an unsigned integer `sysctl` by name (e.g. "hw.nperflevels"), or 0 if unavailable.
 *  @sa Used to harvest the Apple Silicon performance-level topology.
 */
FU_MAYBE_UNUSED_ static inline std::size_t sysctl_uint(char const *name) noexcept {
    unsigned long long value = 0;
    std::size_t length = sizeof(value);
    if (::sysctlbyname(name, &value, &length, nullptr, 0) != 0) return 0;
    return static_cast<std::size_t>(value);
}
#endif

/**
 *  @brief NUMA topology descriptor: describing memory pools and core counts next to them.
 *
 *  Uses dynamic memory to store the NUMA nodes and their cores. Assuming we may soon have
 *  Intel "Sierra Forest"-like CPUs with 288 cores with up to 8 sockets per node, this structure
 *  can easily grow to 10 KB.
 */
template <std::size_t max_page_sizes_ = 4, typename allocator_type_ = std::allocator<char>>
struct numa_topology {

    using allocator_t = allocator_type_;
    using cores_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<int>;
    using nodes_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<numa_node_t>;
    using domains_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<compute_domain_t>;
    using capacities_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<std::size_t>;
    static constexpr std::size_t max_page_sizes_k = max_page_sizes_;

  private:
    allocator_t allocator_ {};
    /** Memory domains, one per NUMA node. */
    dynamic_array<numa_node_t, nodes_allocator_t> nodes_;
    /** Core IDs grouped by node then QoS; the nodes and domains below slice into this. */
    dynamic_array<numa_core_id_t, cores_allocator_t> node_core_ids_;
    /** Compute domains, one per same-QoS core run within a node; sized for the worst case. */
    dynamic_array<compute_domain_t, domains_allocator_t> compute_domains_;
    /** Number of memory domains / NUMA nodes. */
    std::size_t nodes_count_ {0};
    /** Total number of cores in all nodes. */
    std::size_t cores_count_ {0};
    /** Number of compute domains actually written, never more than `compute_domains_.size()`. */
    std::size_t compute_domains_count_ {0};
    /** Number of distinct QoS classes (>= 1). */
    std::size_t compute_levels_count_ {1};
    /** Number of distinct memory tiers (>= 1). */
    std::size_t memory_levels_count_ {1};

  public:
    constexpr numa_topology() noexcept = default;

    // ! The arrays move their heap pointers, so the `first_core_id` slices the nodes and domains
    // ! hold into `node_core_ids_` survive a move untouched.
    numa_topology(numa_topology &&o) noexcept
        : allocator_(std::move(o.allocator_)), nodes_(std::move(o.nodes_)), node_core_ids_(std::move(o.node_core_ids_)),
          compute_domains_(std::move(o.compute_domains_)), nodes_count_(std::exchange(o.nodes_count_, 0)),
          cores_count_(std::exchange(o.cores_count_, 0)),
          compute_domains_count_(std::exchange(o.compute_domains_count_, 0)),
          compute_levels_count_(std::exchange(o.compute_levels_count_, 1)),
          memory_levels_count_(std::exchange(o.memory_levels_count_, 1)) {}

    numa_topology &operator=(numa_topology &&other) noexcept {
        if (this != &other) {
            allocator_ = std::move(other.allocator_);
            nodes_ = std::move(other.nodes_);
            node_core_ids_ = std::move(other.node_core_ids_);
            compute_domains_ = std::move(other.compute_domains_);
            nodes_count_ = std::exchange(other.nodes_count_, 0);
            cores_count_ = std::exchange(other.cores_count_, 0);
            compute_domains_count_ = std::exchange(other.compute_domains_count_, 0);
            compute_levels_count_ = std::exchange(other.compute_levels_count_, 1);
            memory_levels_count_ = std::exchange(other.memory_levels_count_, 1);
        }
        return *this;
    }

    numa_topology(numa_topology const &) = delete;
    numa_topology &operator=(numa_topology const &) = delete;

    ~numa_topology() noexcept { reset(); }

    void reset() noexcept {
        nodes_.reset();
        node_core_ids_.reset();
        compute_domains_.reset();
        nodes_count_ = cores_count_ = compute_domains_count_ = 0;
        compute_levels_count_ = 1;
        memory_levels_count_ = 1;
    }

    /** @brief Number of memory domains (one per NUMA node). @sa `compute_domains_count`. */
    std::size_t nodes_count() const noexcept { return nodes_count_; }
    std::size_t memory_domains_count() const noexcept { return nodes_count_; }
    std::size_t threads_count() const noexcept { return cores_count_; }

    /** @brief The memory domain at @p node_index, in [0, `memory_domains_count()`). */
    numa_node_t const &node(memory_domain_index_t const node_index) const noexcept {
        assert(node_index < nodes_count_ && "Node ID is out of bounds");
        return nodes_[node_index];
    }
    numa_node_t const &memory_domain(memory_domain_index_t const memory_domain_index) const noexcept {
        return node(memory_domain_index);
    }

    /** @brief Number of compute domains (one per same-QoS core run within a node). */
    std::size_t compute_domains_count() const noexcept { return compute_domains_count_; }
    /** @brief Number of distinct QoS classes across all compute domains (>= 1). */
    std::size_t compute_levels_count() const noexcept { return compute_levels_count_; }
    /** @brief Number of distinct memory tiers across all memory domains (>= 1). */
    std::size_t memory_levels_count() const noexcept { return memory_levels_count_; }

    /** @brief The compute domain at @p compute_domain_index, in [0, `compute_domains_count()`). */
    compute_domain_t const &compute_domain_at(compute_domain_index_t const compute_domain_index) const noexcept {
        assert(compute_domain_index < compute_domains_count_ && "Compute domain ID is out of bounds");
        return compute_domains_[compute_domain_index];
    }

    /** @brief The memory domain nearest a compute domain (its NUMA node); 0 if out of range. */
    memory_domain_index_t local_memory_of(compute_domain_index_t const compute_domain_index) const noexcept {
        if (compute_domain_index >= compute_domains_count_) return memory_domain_index_t {};
        return compute_domains_[compute_domain_index].memory_domain_index;
    }

    /** @brief Relative access distance from a compute domain to a memory domain (10 = local). */
    std::size_t distance(compute_domain_index_t const compute_domain_index,
                         memory_domain_index_t const memory_domain_index) const noexcept {
        if (compute_domain_index >= compute_domains_count_ || memory_domain_index >= nodes_count_) return 0;
#if FU_WITH_TOPOLOGY_METRICS
        numa_node_id_t const from = compute_domains_[compute_domain_index].node_id;
        numa_node_id_t const to = nodes_[memory_domain_index].node_id;
        int const numa_dist = ::numa_distance(from, to);
        return numa_dist > 0 ? static_cast<std::size_t>(numa_dist) : (from == to ? 10u : 20u);
#else
        return compute_domains_[compute_domain_index].memory_domain_index == memory_domain_index ? 10u : 20u;
#endif
    }

    /** @brief HMAT read bandwidth (MB/s) from a compute domain to a memory domain, or 0 if unknown. */
    std::size_t memory_bandwidth(compute_domain_index_t const compute_domain_index,
                                 memory_domain_index_t const memory_domain_index) const noexcept {
        if (compute_domain_index >= compute_domains_count_ || memory_domain_index >= nodes_count_) return 0;
        numa_node_id_t const from = compute_domains_[compute_domain_index].node_id;
        numa_node_id_t const to = nodes_[memory_domain_index].node_id;
        return read_hmat_metric(from, to, "read_bandwidth");
    }

    /** @brief HMAT read latency (nanoseconds) from a compute domain to a memory domain, or 0 if unknown. */
    std::size_t memory_latency(compute_domain_index_t const compute_domain_index,
                               memory_domain_index_t const memory_domain_index) const noexcept {
        if (compute_domain_index >= compute_domains_count_ || memory_domain_index >= nodes_count_) return 0;
        numa_node_id_t const from = compute_domains_[compute_domain_index].node_id;
        numa_node_id_t const to = nodes_[memory_domain_index].node_id;
        return read_hmat_metric(from, to, "read_latency");
    }

    /**
     *  @brief Harvests CPU-memory topology - Linux NUMA nodes, or Apple Silicon performance levels.
     *  @retval false if the platform lacks topology support or the harvest failed.
     *  @retval true if the harvest was successful and the topology is ready to use.
     */
    bool try_harvest() noexcept {
#if FU_WITH_TOPOLOGY && FU_ON_LINUX
        reset();

        // The cores this process may actually run on. A cgroup `cpuset` or a `taskset` narrows it,
        // and a domain's CPU list must be intersected with it - otherwise we would size the pool
        // from the machine and pin workers onto cores the kernel will never schedule us on.
        affinity_mask allowed;
        bool const allowed_known = allowed.try_capture() && allowed.count() != 0;

        if (::numa_available() < 0) return false; // ! Linux kernel lacks NUMA support
        ::numa_node_to_cpu_update();              // ? Reset the outdated stale state

        // The only resource here the arrays below cannot own for us.
        numa_cpumask_guard numa_mask_guard;
        struct bitmask *const numa_mask = numa_mask_guard.mask;
        if (!numa_mask) return false; // ! Allocation failed

        // First pass - measure
        std::size_t fetched_nodes = 0, fetched_cores = 0;
        numa_node_id_t const max_numa_node_id = ::numa_max_node();
        for (numa_node_id_t node_id = 0; node_id <= max_numa_node_id; ++node_id) {
            long long dummy;
            if (::numa_node_size64(node_id, &dummy) < 0) continue; // ! Offline node
            ::numa_bitmask_clearall(numa_mask);
            if (::numa_node_to_cpus(node_id, numa_mask) < 0) continue; // ! Invalid CPU map
            if (allowed_known) restrict_cpumask_to(numa_mask, allowed);
            // A cpuless memory domain (HBM-flat, CXL expander, GPU HBM) reports zero cores, yet is a
            // valid memory domain - and so is one whose every core was masked away from us.
            std::size_t const node_cores = static_cast<std::size_t>(::numa_bitmask_weight(numa_mask));
            fetched_nodes += 1;
            fetched_cores += node_cores;
        }
        if (fetched_nodes == 0) return false; // ! Zero nodes is not a valid state

        // Second pass - allocate. At most one compute domain per core (fully heterogeneous node).
        // A failed `try_resize` leaves its array empty, and every array frees itself on the way out.
        dynamic_array<numa_node_t, nodes_allocator_t> nodes {nodes_allocator_t {allocator_}};
        dynamic_array<numa_core_id_t, cores_allocator_t> core_ids {cores_allocator_t {allocator_}};
        dynamic_array<compute_domain_t, domains_allocator_t> domains {domains_allocator_t {allocator_}};
        if (!nodes.try_resize(fetched_nodes)) return false;
        if (!core_ids.try_resize(fetched_cores)) return false;
        if (!domains.try_resize(fetched_cores)) return false;
        numa_node_t *const nodes_ptr = nodes.data();
        numa_core_id_t *const core_ids_ptr = core_ids.data();
        compute_domain_t *const domains_ptr = domains.data();

        // A scratch table of every configured CPU's capacity, filled once below and read by the
        // per-node QoS split (which is O(n^2) in comparisons) instead of re-opening sysfs each time.
        std::size_t const configured_cores = static_cast<std::size_t>(::numa_num_configured_cpus());
        if (configured_cores == 0) return false; // ! No CPUs is not a valid state
        dynamic_array<std::size_t, capacities_allocator_t> capacities {capacities_allocator_t {allocator_}};
        if (!capacities.try_resize(configured_cores)) return false;
        std::size_t *const core_capacities = capacities.data();

        // Populate
        for (numa_node_id_t node_id = 0, core_index = 0, node_index = 0; node_id <= max_numa_node_id; ++node_id) {
            long long free_memory_size; // ? Only an out-parameter, the total size comes back as the return value
            long long const total_memory_size = ::numa_node_size64(node_id, &free_memory_size);
            if (total_memory_size < 0) continue;
            ::numa_bitmask_clearall(numa_mask);
            if (::numa_node_to_cpus(node_id, numa_mask) < 0) continue;
            if (allowed_known) restrict_cpumask_to(numa_mask, allowed);

            numa_node_t &node = nodes_ptr[node_index];
            node.node_id = node_id;
            node.memory_size = static_cast<std::size_t>(total_memory_size);
            node.first_core_id = core_ids_ptr + core_index;
            node.core_count = static_cast<std::size_t>(::numa_bitmask_weight(numa_mask));

            // Most likely, this will fill `core_ids_ptr` with `std::iota`-like values
            for (std::size_t bit_offset = 0; bit_offset < numa_mask->size; ++bit_offset)
                if (::numa_bitmask_isbitset(numa_mask, static_cast<unsigned int>(bit_offset)))
                    core_ids_ptr[core_index++] = static_cast<numa_core_id_t>(bit_offset);

            // ? Cpuless memory domains have no core to query - default the socket and skip the lookup.
            // ! Only valid once `first_core_id` points to initialized entries, hence after the loop above
            node.socket_id = node.core_count > 0 ? get_socket_id_for_core(node.first_core_id[0]) : -1;

            // Fetch Huge Page sizes for this NUMA node
            node.page_sizes.try_harvest(node_id); // ! We are not raising the failure - Huge Pages are optional
            node_index++;
        }

        // Commit. The arrays keep their heap pointers across the move, so every `first_core_id`
        // slice written above stays valid.
        nodes_ = std::move(nodes);
        node_core_ids_ = std::move(core_ids);
        nodes_count_ = fetched_nodes;
        cores_count_ = fetched_cores;

        // Let's sort all the nodes by their socket ID, then by number of cores, then by first core ID
        bubble_sort(nodes_.data(), nodes_count_, [](numa_node_t const &a, numa_node_t const &b) noexcept {
            if (a.socket_id != b.socket_id) return a.socket_id < b.socket_id;
            if (a.core_count != b.core_count) return a.core_count > b.core_count;  // ? Sort by descending core count
            numa_core_id_t const a_first = a.core_count ? a.first_core_id[0] : -1; // ? Cpuless slices are empty
            numa_core_id_t const b_first = b.core_count ? b.first_core_id[0] : -1;
            return a_first < b_first; // ? Sort by first core ID
        });

        // Cache each harvested core's scheduler capacity once, keyed by core id. The QoS split
        // below sorts and compares capacities repeatedly, so reading sysfs per comparison would be
        // O(cores^2) file opens on a large node - a single pass here amortizes that to O(cores).
        for (std::size_t i = 0; i < configured_cores; ++i) core_capacities[i] = 0;
        for (std::size_t core_index = 0; core_index < cores_count_; ++core_index) {
            numa_core_id_t const core_id = node_core_ids_[core_index];
            if (static_cast<std::size_t>(core_id) < configured_cores)
                core_capacities[static_cast<std::size_t>(core_id)] = get_capacity_for_core(core_id);
        }

        // Split each memory domain's cores into compute domains by QoS class. We sort each node's
        // cores by scheduler capacity, then cut the sorted run at every capacity change. Cores
        // within a `numa_node` slice are mutable here - `first_core_id` still bounds the slice.
        {
            std::size_t domains_written = 0;
            for (std::size_t node_index = 0; node_index < nodes_count_; ++node_index) {
                numa_node_t &node = nodes_[node_index];
                numa_core_id_t *node_cores = const_cast<numa_core_id_t *>(node.first_core_id);

                // Ascending capacity groups efficiency cores before performance cores.
                bubble_sort(node_cores, node.core_count,
                            [core_capacities](numa_core_id_t const &a, numa_core_id_t const &b) noexcept {
                                return core_capacities[static_cast<std::size_t>(a)] <
                                       core_capacities[static_cast<std::size_t>(b)];
                            });

                std::size_t run_begin = 0;
                for (std::size_t core = 1; core <= node.core_count; ++core) {
                    bool const at_end = core == node.core_count;
                    bool const capacity_changed =
                        !at_end && core_capacities[static_cast<std::size_t>(node_cores[core])] !=
                                       core_capacities[static_cast<std::size_t>(node_cores[run_begin])];
                    if (!at_end && !capacity_changed) continue;

                    compute_domain_t &domain = domains_ptr[domains_written++];
                    domain.node_id = node.node_id;
                    domain.memory_domain_index = static_cast<memory_domain_index_t>(node_index);
                    domain.compute_level = core_capacities[static_cast<std::size_t>(
                        node_cores[run_begin])]; // ? Raw capacity, ranked below
                    // ! Keep the raw magnitude too - `compute_level` is about to collapse into an ordinal.
                    domain.capacity = core_capacities[static_cast<std::size_t>(node_cores[run_begin])];
                    domain.cache_bytes = 0; // ? Not yet read from `sys/devices/system/cpu/cpu*/cache`
                    domain.first_core_id = node_cores + run_begin;
                    domain.core_count = core - run_begin;
                    run_begin = core;
                }
            }

            // Re-rank the raw capacities into dense QoS ordinals, sorted least-to-most performant.
            compute_domains_ = std::move(domains);
            compute_domains_count_ = domains_written;
            compute_levels_count_ = dense_rank(
                domains_written,
                [core_capacities, domains_ptr](std::size_t index) noexcept {
                    return core_capacities[static_cast<std::size_t>(domains_ptr[index].first_core_id[0])];
                },
                [domains_ptr](std::size_t index, std::size_t rank) noexcept {
                    domains_ptr[index].compute_level = rank;
                });
        }

        // Rank memory domains into dense tier ordinals, sorted fastest-to-slowest (lower = faster). Raw
        // tiers are snapshotted into scratch so ranking in place never corrupts a repeated tier. Absent
        // the memory-tiering sysfs, every node collapses to a single memory level.
        {
            dynamic_array<std::size_t, capacities_allocator_t> raw_tiers {capacities_allocator_t {allocator_}};
            if (raw_tiers.try_resize(nodes_count_)) {
                std::size_t *const tiers = raw_tiers.data();
                for (std::size_t i = 0; i < nodes_count_; ++i) tiers[i] = get_memory_tier_for_node(nodes_[i].node_id);
                memory_levels_count_ = dense_rank(
                    nodes_count_, [tiers](std::size_t index) noexcept { return tiers[index]; },
                    [this](std::size_t index, std::size_t rank) noexcept { nodes_[index].memory_level = rank; });
            }
        }

        return true; // ? Every scratch array above frees itself here
#endif // FU_WITH_TOPOLOGY
#if defined(__APPLE__)
        return try_harvest_apple();
#elif FU_ON_WINDOWS
        return try_harvest_windows();
#else
        return false;
#endif
    }

#if defined(__APPLE__)
    /**
     *  @brief Harvests the Apple Silicon topology from `sysctl` performance levels.
     *  @retval false if the machine reports no logical CPUs or an allocation failed.
     *
     *  Apple Silicon is one UMA memory domain shared by every core, so we build a single memory
     *  domain. The compute axis is cut twice: first by `hw.perflevelN`, then by `cpusperl2` within
     *  each level, because a performance level may span several L2 clusters that share no cache.
     *  A compute domain is cores sharing a QoS class @b and locality, so the cluster is the unit.
     *  Domains from one level all carry that level's `compute_level`; `hw.perflevel0` ranks highest.
     *
     *  Performance levels rank cores without rating them, and the parts they distinguish need not
     *  differ in clock - some pair equal scalar throughput with unequal cache. So `capacity` stays
     *  `capacity_unknown_k` and only `cache_bytes` is populated; weighting work by the level ordinal
     *  would hand a wide-cache cluster more tasks than it can necessarily retire any faster.
     *
     *  @note macOS exposes no hard pinning - `thread_policy_set(THREAD_AFFINITY_POLICY)` returns
     *        `KERN_NOT_SUPPORTED` on arm64, and QoS classes are the only placement lever. These
     *        domains are therefore descriptive: `spawn_on` reports them, the kernel still migrates.
     */
    bool try_harvest_apple() noexcept {
        std::size_t const total_cores = sysctl_uint("hw.logicalcpu");
        if (total_cores == 0) return false;
        std::size_t const memory_size = sysctl_uint("hw.memsize");
        std::size_t const levels = sysctl_uint("hw.nperflevels");

        // Count the non-empty performance levels so each becomes one compute level.
        std::size_t nonempty_levels = 0;
        for (std::size_t level = 0; level < levels; ++level) {
            char name[64];
            std::snprintf(name, sizeof(name), "hw.perflevel%zu.logicalcpu", level);
            if (sysctl_uint(name)) nonempty_levels += 1;
        }
        if (nonempty_levels == 0) nonempty_levels = 1; // ? One level covering every core

        // `compute_domains_` is sized to `cores_count_` across the class (at most one domain per core).
        dynamic_array<numa_node_t, nodes_allocator_t> nodes {nodes_allocator_t {allocator_}};
        dynamic_array<numa_core_id_t, cores_allocator_t> core_ids {cores_allocator_t {allocator_}};
        dynamic_array<compute_domain_t, domains_allocator_t> domains {domains_allocator_t {allocator_}};
        if (!nodes.try_resize(1)) return false;
        if (!core_ids.try_resize(total_cores)) return false;
        if (!domains.try_resize(total_cores)) return false;
        numa_node_t *const nodes_ptr = nodes.data();
        numa_core_id_t *const core_ids_ptr = core_ids.data();
        compute_domain_t *const domains_ptr = domains.data();
        for (std::size_t i = 0; i < total_cores; ++i) core_ids_ptr[i] = static_cast<numa_core_id_t>(i);

        numa_node_t &node = nodes_ptr[0];
        node.node_id = 0;
        node.socket_id = 0;
        node.memory_size = memory_size;
        node.memory_level = 0;
        node.first_core_id = core_ids_ptr;
        node.core_count = total_cores;

        // One compute domain per L2 cluster. Apple lists cores fastest-level first, so `hw.perflevel0`
        // takes the highest compute level; every cluster carved out of it repeats that same level.
        std::size_t core_offset = 0, domains_written = 0, levels_written = 0;
        for (std::size_t level = 0; level < levels && core_offset < total_cores; ++level) {
            char name[64];
            std::snprintf(name, sizeof(name), "hw.perflevel%zu.logicalcpu", level);
            std::size_t level_cores = sysctl_uint(name);
            if (level_cores == 0) continue;
            if (core_offset + level_cores > total_cores) level_cores = total_cores - core_offset;

            std::snprintf(name, sizeof(name), "hw.perflevel%zu.l2cachesize", level);
            std::size_t const level_cache_bytes = sysctl_uint(name);
            std::snprintf(name, sizeof(name), "hw.perflevel%zu.cpusperl2", level);
            std::size_t cores_per_cluster = sysctl_uint(name);
            // ? A level with no `cpusperl2` is one undivided cluster, not zero-sized ones.
            if (cores_per_cluster == 0 || cores_per_cluster > level_cores) cores_per_cluster = level_cores;

            std::size_t const level_rank = nonempty_levels - 1 - levels_written;
            for (std::size_t cut = 0; cut < level_cores; cut += cores_per_cluster) {
                std::size_t const cluster_cores = (level_cores - cut) < cores_per_cluster //
                                                      ? (level_cores - cut)
                                                      : cores_per_cluster;
                compute_domain_t &domain = domains_ptr[domains_written];
                domain.node_id = 0;
                domain.memory_domain_index = static_cast<memory_domain_index_t>(0);
                domain.compute_level = level_rank; // ? Sibling clusters share their level's rank
                domain.capacity = capacity_unknown_k;
                domain.cache_bytes = level_cache_bytes; // ? L2 is private to the cluster, shared within it
                domain.first_core_id = core_ids_ptr + core_offset + cut;
                domain.core_count = cluster_cores;
                domains_written += 1;
            }
            core_offset += level_cores;
            levels_written += 1;
        }

        // Fallback: no perflevel data - one compute domain over every core.
        if (domains_written == 0) {
            compute_domain_t &domain = domains_ptr[0];
            domain.node_id = 0;
            domain.memory_domain_index = static_cast<memory_domain_index_t>(0);
            domain.compute_level = 0;
            domain.capacity = capacity_unknown_k;
            domain.cache_bytes = 0;
            domain.first_core_id = core_ids_ptr;
            domain.core_count = total_cores;
            domains_written = 1;
            levels_written = 1;
        }

        // Commit. Moving an array keeps its heap pointer, so every `first_core_id` stays valid.
        nodes_ = std::move(nodes);
        node_core_ids_ = std::move(core_ids);
        compute_domains_ = std::move(domains);
        nodes_count_ = 1;
        cores_count_ = total_cores;
        compute_domains_count_ = domains_written;
        compute_levels_count_ = levels_written; // ! Several clusters may share one level - not `domains_written`
        memory_levels_count_ = 1;
        return true;
    }
#endif // defined(__APPLE__)

#if FU_ON_WINDOWS
    /**
     *  @brief Harvests the Windows topology from `GetLogicalProcessorInformationEx`.
     *  @retval false if the machine reports no NUMA node or an allocation failed.
     *
     *  Windows describes a machine in the same two axes this library already uses. A @b processor
     *  @b group holds at most 64 logical processors sharing one `KAFFINITY` mask; groups are cut along
     *  NUMA boundaries, so a NUMA node maps to one memory domain. Within a node the compute axis is cut
     *  by @b efficiency @b class - the kernel's rank of a core's performance, where a higher class is
     *  more performant - so a hybrid P/E chip yields one compute domain per class, exactly as Apple's
     *  performance levels do. Non-hybrid chips report class 0 for every core and collapse to a single
     *  compute domain per node.
     *
     *  Rather than a per-processor scratch table, the harvest accumulates one mask per (processor group,
     *  efficiency class) - a @ref `win_group_class_cell_t` - so its working set is tiny and it reads the
     *  compute domains straight out of masks, the unit Windows itself speaks in.
     *
     *  Efficiency class is an ordinal, not a magnitude - it ranks cores without rating them - so
     *  `capacity` stays `capacity_unknown_k` and callers weigh domains by `core_count`, mirroring the
     *  Apple path. `cache_bytes` is the largest private (L1/L2) cache the kernel reports for the class.
     *
     *  @note A `numa_core_id_t` here is not a flat index: it packs the (group, in-group bit) pair via
     *        `win_encode_core_id`, which `pin_thread_to_cores` decodes back into a `GROUP_AFFINITY`.
     *  @note A NUMA node spanning several processor groups (the largest servers) is enumerated in full
     *        where the SDK exposes `GroupMasks[]`; @sa `win_numa_for_each_group`.
     */
    bool try_harvest_windows() noexcept {
        // Pull one relationship class into a heap buffer the caller frees. The record layout is
        // variable-length: every entry carries its own `Size`, and iteration advances by it.
        auto query = [](LOGICAL_PROCESSOR_RELATIONSHIP relationship, DWORD &out_len) -> BYTE * {
            DWORD len = 0;
            ::GetLogicalProcessorInformationEx(relationship, nullptr, &len);
            if (len == 0) return nullptr; // ! Nothing to report, or an unexpected failure
            BYTE *buffer = static_cast<BYTE *>(std::malloc(len));
            if (!buffer) return nullptr; // ! Out of memory
            auto *typed = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buffer);
            if (!::GetLogicalProcessorInformationEx(relationship, typed, &len)) {
                std::free(buffer);
                return nullptr; // ! The machine changed between the sizing and the fill
            }
            out_len = len;
            return buffer;
        };
        using cell_allocator_t =
            typename std::allocator_traits<allocator_t>::template rebind_alloc<win_group_class_cell_t>;

        // Everything the `failed_harvest:` label frees must be declared and initialized before the
        // first `goto`, so a jump there never skips an initializer. Only the Win32 buffers still need
        // the label; the arrays below own themselves.
        cell_allocator_t cell_alloc {allocator_};
        BYTE *numa_buf = nullptr, *package_buf = nullptr;
        win_group_class_cell_t *cells = nullptr; // ? [group * class_count + class]: core mask + private cache

        // ! Declared before the first `goto`: jumping over a non-trivial destructor is ill-formed.
        // ! They free themselves at every exit, so the label below need only mind the Win32 buffers.
        dynamic_array<numa_node_t, nodes_allocator_t> nodes {nodes_allocator_t {allocator_}};
        dynamic_array<numa_core_id_t, cores_allocator_t> core_ids {cores_allocator_t {allocator_}};
        dynamic_array<compute_domain_t, domains_allocator_t> domains {domains_allocator_t {allocator_}};
        numa_node_t *nodes_ptr = nullptr;
        numa_core_id_t *core_ids_ptr = nullptr;
        compute_domain_t *domains_ptr = nullptr;
        DWORD numa_len = 0, cores_len = 0, cache_len = 0, package_len = 0;
        std::size_t class_count = 1, cell_count = 0; // ? `class_count` starts at 1: a non-hybrid machine
        std::size_t counted_nodes = 0, counted_cores = 0;
        std::size_t core_cursor = 0, node_index = 0, domain_cursor = 0, levels = 1;

        // A processor group holds at most 64 logical processors; `group_span` bounds the cell grid.
        WORD const group_span = ::GetActiveProcessorGroupCount() ? ::GetActiveProcessorGroupCount() : 1;

        // Pass 1: from the physical-core relationships, learn how many efficiency classes exist, then
        // OR each core into its (group, class) cell. Two sweeps of one buffer - measure, then fill.
        {
            BYTE *cores_buf = query(RelationProcessorCore, cores_len);
            if (!cores_buf) goto failed_harvest; // ! No processor information at all
            for (DWORD offset = 0; offset < cores_len;) {
                auto *record = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(cores_buf + offset);
                if (record->Relationship == RelationProcessorCore &&
                    static_cast<std::size_t>(record->Processor.EfficiencyClass) + 1 > class_count)
                    class_count = static_cast<std::size_t>(record->Processor.EfficiencyClass) + 1;
                if (record->Size == 0) break; // ! Malformed record; stop rather than spin
                offset += record->Size;
            }
            cell_count = static_cast<std::size_t>(group_span) * class_count;
            cells = cell_alloc.allocate(cell_count);
            if (!cells) {
                std::free(cores_buf);
                goto failed_harvest; // ! Out of memory
            }
            for (std::size_t i = 0; i < cell_count; ++i) cells[i] = win_group_class_cell_t {};
            for (DWORD offset = 0; offset < cores_len;) {
                auto *record = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(cores_buf + offset);
                if (record->Relationship == RelationProcessorCore) {
                    std::size_t const cls = record->Processor.EfficiencyClass;
                    for (WORD g = 0; g < record->Processor.GroupCount; ++g) {
                        GROUP_AFFINITY const affinity = record->Processor.GroupMask[g];
                        if (affinity.Group < group_span)
                            cells[static_cast<std::size_t>(affinity.Group) * class_count + cls].mask |= affinity.Mask;
                    }
                }
                if (record->Size == 0) break;
                offset += record->Size;
            }
            std::free(cores_buf);
        }

        // Pass 2: attribute each private (L1/L2) cache to the (group, class) cells its cores belong to.
        // L3 is shared across domains, so it would misreport a domain's private cache and is skipped.
        if (BYTE *cache_buf = query(RelationCache, cache_len)) {
            for (DWORD offset = 0; offset < cache_len;) {
                auto *record = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(cache_buf + offset);
                if (record->Relationship == RelationCache && record->Cache.Level <= 2) {
                    GROUP_AFFINITY const affinity = record->Cache.GroupMask;
                    std::size_t const size_bytes = static_cast<std::size_t>(record->Cache.CacheSize);
                    if (affinity.Group < group_span)
                        for (std::size_t c = 0; c < class_count; ++c) {
                            win_group_class_cell_t &cell =
                                cells[static_cast<std::size_t>(affinity.Group) * class_count + c];
                            if ((affinity.Mask & cell.mask) && size_bytes > cell.cache_bytes)
                                cell.cache_bytes = size_bytes;
                        }
                }
                if (record->Size == 0) break;
                offset += record->Size;
            }
            std::free(cache_buf);
        }

        // Packages are kept until the fill so each node can be tagged with its socket; null is fine.
        package_buf = query(RelationProcessorPackage, package_len);

        // Pass 3: count nodes and total cores - counting the exact bits pass 4 will emit (each node
        // group intersected with the class cells), so the allocation matches the fill and `reset()`
        // later frees the same length it was handed.
        numa_buf = query(RelationNumaNode, numa_len);
        if (!numa_buf) goto failed_harvest; // ! No NUMA information at all
        for (DWORD offset = 0; offset < numa_len;) {
            auto *record = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(numa_buf + offset);
            if (record->Relationship == RelationNumaNode) {
                counted_nodes += 1;
                win_numa_for_each_group(record->NumaNode, [&](WORD group, KAFFINITY node_mask) noexcept {
                    if (group >= group_span) return; // ? Beyond the cell grid; skip as the fill does
                    for (std::size_t c = 0; c < class_count; ++c)
                        counted_cores += popcount(static_cast<KAFFINITY>(
                            node_mask & cells[static_cast<std::size_t>(group) * class_count + c].mask));
                });
            }
            if (record->Size == 0) break;
            offset += record->Size;
        }
        if (counted_nodes == 0 || counted_cores == 0) goto failed_harvest; // ! Nothing to spawn onto

        // Allocate the committed arrays. `compute_domains_` is sized to the core count - at most one
        // domain per core. `try_resize` value-initializes, so the members the fill below does not touch
        // - the `page_sizes` inventory - hold their zeroed defaults rather than garbage.
        if (!nodes.try_resize(counted_nodes)) goto failed_harvest;    // ! Out of memory
        if (!core_ids.try_resize(counted_cores)) goto failed_harvest; // ! Out of memory
        if (!domains.try_resize(counted_cores)) goto failed_harvest;  // ! Out of memory
        nodes_ptr = nodes.data();
        core_ids_ptr = core_ids.data();
        domains_ptr = domains.data();

        // Pass 4: fill each node, emitting its cores one efficiency class at a time (most performant
        // first) so a class's cores land contiguously and become one compute domain - reading the
        // (group, class) cells rather than any per-core table.
        for (DWORD offset = 0; offset < numa_len;) {
            auto *record = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(numa_buf + offset);
            if (record->Relationship != RelationNumaNode) {
                if (record->Size == 0) break;
                offset += record->Size;
                continue;
            }
            NUMA_NODE_RELATIONSHIP const &numa_node = record->NumaNode;
            std::size_t const node_first_core = core_cursor;

            for (std::size_t class_step = 0; class_step < class_count; ++class_step) {
                std::size_t const cls = class_count - 1 - class_step; // ? Fastest (highest) class first
                std::size_t const domain_first_core = core_cursor;
                std::size_t domain_cache_bytes = 0;
                win_numa_for_each_group(numa_node, [&](WORD group, KAFFINITY node_mask) noexcept {
                    if (group >= group_span) return; // ? Beyond the cell grid; should not happen
                    win_group_class_cell_t const &cell = cells[static_cast<std::size_t>(group) * class_count + cls];
                    KAFFINITY const domain_mask = node_mask & cell.mask;
                    if (!domain_mask) return;
                    if (cell.cache_bytes > domain_cache_bytes) domain_cache_bytes = cell.cache_bytes;
                    for (unsigned bit = 0; bit < win_processors_per_group_k; ++bit)
                        if (domain_mask & (static_cast<KAFFINITY>(1) << bit))
                            core_ids_ptr[core_cursor++] = win_encode_core_id(group, bit);
                });
                if (core_cursor == domain_first_core) continue; // ? No cores of this class on this node

                compute_domain_t &domain = domains_ptr[domain_cursor++];
                domain.node_id = static_cast<numa_node_id_t>(numa_node.NodeNumber);
                domain.memory_domain_index = static_cast<memory_domain_index_t>(node_index);
                domain.compute_level = cls; // ? Raw class now; dense-ranked below
                domain.capacity = capacity_unknown_k;
                domain.cache_bytes = domain_cache_bytes;
                domain.first_core_id = core_ids_ptr + domain_first_core;
                domain.core_count = core_cursor - domain_first_core;
            }

            ULONGLONG available_bytes = 0;
            (void)::GetNumaAvailableMemoryNodeEx(static_cast<USHORT>(numa_node.NodeNumber), &available_bytes);

            numa_node_t &node = nodes_ptr[node_index];
            node.node_id = static_cast<numa_node_id_t>(numa_node.NodeNumber);
            node.socket_id = win_socket_for_node(package_buf, package_len, numa_node,
                                                 static_cast<numa_socket_id_t>(numa_node.NodeNumber));
            node.memory_size =
                static_cast<std::size_t>(available_bytes); // ? Available, not installed - Windows has no per-node total
            node.memory_level = 0;
            node.first_core_id = core_ids_ptr + node_first_core;
            node.core_count = core_cursor - node_first_core;
            node.page_sizes.try_harvest(node.node_id); // ! Optional: records the large-page size if available
            node_index += 1;
            if (record->Size == 0) break;
            offset += record->Size;
        }
        std::free(package_buf);
        package_buf = nullptr;
        std::free(numa_buf);
        numa_buf = nullptr; // ? Owned buffers released; keeps the label's blanket free safe on fall-through
        cell_alloc.deallocate(cells, cell_count);
        cells = nullptr;

        // Collapse the raw efficiency classes into a dense 0..K-1 rank, least-performant first - the
        // ordinal `compute_level` promises. A non-hybrid machine ranks every domain 0: one level.
        if (domain_cursor != 0)
            levels = dense_rank(
                domain_cursor,
                [domains_ptr](std::size_t i) noexcept {
                    return static_cast<std::size_t>(domains_ptr[i].compute_level);
                },
                [domains_ptr](std::size_t i, std::size_t rank) noexcept { domains_ptr[i].compute_level = rank; });

        // Commit. Moving an array keeps its heap pointer, so every `first_core_id` stays valid.
        nodes_ = std::move(nodes);
        node_core_ids_ = std::move(core_ids);
        compute_domains_ = std::move(domains);
        nodes_count_ = counted_nodes;
        cores_count_ = core_cursor;
        compute_domains_count_ = domain_cursor;
        compute_levels_count_ = levels;
        memory_levels_count_ = 1; // ? Windows exposes no memory-tiering ranking
        return true;

    failed_harvest: // ? Only the Win32 buffers are ours to free; the arrays unwind themselves
        if (cells) cell_alloc.deallocate(cells, cell_count);
        std::free(numa_buf);
        std::free(package_buf);
        return false;
    }
#endif // FU_ON_WINDOWS

    /**
     *  @brief Copy-assigns the topology from @p other.
     *
     *  Instead of a copy-constructor we expose an explicit operation that can
     *  FAIL - returning `false` if *any* intermediate allocation fails.
     *
     *  @param other Source topology.
     *  @retval true  Success, the current instance now owns a deep copy.
     *  @retval false Allocation failed, the current instance is unchanged.
     */
    bool try_assign(numa_topology const &other) noexcept {
        if (this == &other) return true; // ? Self-assignment is a no-op

        // Prepare scratch. Any `try_resize` that fails frees whatever the others took, on the way out.
        dynamic_array<numa_node_t, nodes_allocator_t> scratch_nodes {nodes_allocator_t {allocator_}};
        dynamic_array<numa_core_id_t, cores_allocator_t> scratch_core_ids {cores_allocator_t {allocator_}};
        dynamic_array<compute_domain_t, domains_allocator_t> scratch_domains {domains_allocator_t {allocator_}};
        if (!scratch_nodes.try_resize(other.nodes_count_)) return false;    // ! OOM
        if (!scratch_core_ids.try_resize(other.cores_count_)) return false; // ! OOM
        if (!scratch_domains.try_resize(other.cores_count_)) return false;  // ! OOM

        // Deep copy, re-basing every `first_core_id` into our own core-id block
        numa_core_id_t const *const other_cores = other.node_core_ids_.data();
        if (other.cores_count_ > 0)
            std::memcpy(scratch_core_ids.data(), other_cores, other.cores_count_ * sizeof(numa_core_id_t));
        for (std::size_t i = 0; i < other.nodes_count_; ++i) {
            scratch_nodes[i] = other.nodes_[i];
            std::ptrdiff_t const offset = other.nodes_[i].first_core_id - other_cores;
            scratch_nodes[i].first_core_id = scratch_core_ids.data() + offset;
        }
        for (std::size_t i = 0; i < other.compute_domains_count_; ++i) {
            scratch_domains[i] = other.compute_domains_[i];
            std::ptrdiff_t const offset = other.compute_domains_[i].first_core_id - other_cores;
            scratch_domains[i].first_core_id = scratch_core_ids.data() + offset;
        }

        nodes_ = std::move(scratch_nodes); // ? Assignment frees the old buffers
        node_core_ids_ = std::move(scratch_core_ids);
        compute_domains_ = std::move(scratch_domains);
        nodes_count_ = other.nodes_count_;
        cores_count_ = other.cores_count_;
        compute_domains_count_ = other.compute_domains_count_;
        compute_levels_count_ = other.compute_levels_count_;
        memory_levels_count_ = other.memory_levels_count_;
        return true;
    }
};

using numa_topology_t = numa_topology<>;

static constexpr std::size_t page_size_4k = 4ull * 1024ull;                       // 4 KB
static constexpr std::size_t page_size_2m_k = 2ull * 1024ull * 1024ull;           // 2 MB
static constexpr std::size_t page_size_1g_k = 1ull * 1024ull * 1024ull * 1024ull; // 1 GB

} // namespace forkunion
} // namespace ashvardanian
