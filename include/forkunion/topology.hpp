/**
 *  @file topology.hpp
 *  @brief The hardware description: memory domains, compute domains, and the topology that holds them.
 *  @note Included by `<forkunion.hpp>`; not meant to be included on its own.
 */
#pragma once
#include "capabilities.hpp"

namespace ashvardanian {
namespace forkunion {

struct ram_page_setting_t {
    std::size_t bytes_per_page {0};  // ? Huge page size in bytes, e.g. 4 KB, 2 MB, or 1 GB
    std::size_t available_pages {0}; // ? Number of pages available for this size, 0 if not available
    std::size_t free_pages {0};      // ? Number of pages available and unused, 0 if not available
};

/**
 *  @brief Fetches the RAM page size in bytes.
 *  @retval The size of a memory page in bytes, typically 4096 on most systems.
 *  @note On Linux, this is the system page size, which may differ from Huge Pages sizes.
 */
FU_MAYBE_UNUSED_ static inline std::size_t get_ram_page_size() noexcept {
#if FU_WITH_NUMA_MEMORY
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
    std::array<ram_page_setting_t, max_page_sizes_k> sizes_ {0}; // ? Huge page sizes in bytes
    std::size_t count_sizes_ {0};                                // ? Number of supported huge page sizes
    std::size_t total_memory_bytes_ {0};                         // ? Total memory available on this NUMA node
  public:
    /**
     *  @brief Finds the largest Huge Pages size available for the given NUMA node.
     */
    ram_page_setting_t largest_free() const noexcept {
        if (!count_sizes_) return {};
        ram_page_setting_t largest = sizes_[0];
        for (std::size_t i = 1; i < count_sizes_; ++i)
            if (sizes_[i].free_pages > largest.free_pages) largest = sizes_[i];
        return largest;
    }

    /**
     *  @brief Fetches all available huge page sizes for the given NUMA node.
     *  @note Kernel support doesn't mean that pages of that size have a valid mount point.
     */
    bool try_harvest(FU_MAYBE_UNUSED_ numa_node_id_t node_id) noexcept {
        assert(node_id >= 0 && "NUMA node ID must be non-negative");

#if FU_WITH_HUGE_PAGES

        std::size_t count_sizes = 0; // ? Number of sizes found

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
        while ((entry = ::readdir(hugepages_dir)) != nullptr && count_sizes < max_page_sizes_k) {
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
            sizes_[count_sizes].bytes_per_page = bytes_per_page;
            sizes_[count_sizes].available_pages = allocated_pages;
            sizes_[count_sizes].free_pages = free_pages;
            ++count_sizes;
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

        count_sizes_ = count_sizes;
        return true;
#else
        fu_unused_(node_id);
        return false;
#endif
    }

    std::size_t size() const noexcept { return count_sizes_; }
    std::size_t total_memory_bytes() const noexcept { return total_memory_bytes_; }
    ram_page_setting_t const *begin() const noexcept { return sizes_.data(); }
    ram_page_setting_t const *end() const noexcept { return sizes_.data() + count_sizes_; }
    ram_page_setting_t const &operator[](std::size_t const index) const noexcept {
        assert(index < count_sizes_ && "Index is out of bounds");
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
        std::size_t page_index = count_sizes_;
        for (std::size_t i = 0; i < count_sizes_; ++i) {
            if (sizes_[i].bytes_per_page == page_size_bytes) {
                page_index = i;
                break;
            }
        }
        if (page_index >= count_sizes_) return false; // ? Page size not found

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

    /** @brief Unique NUMA node ID, in [0, numa_max_node()). */
    numa_node_id_t node_id {-1};
    /** @brief Physical CPU socket ID. */
    numa_socket_id_t socket_id {-1};
    /** @brief RAM volume in bytes. */
    std::size_t memory_size {0};
    /** @brief Memory tier ordinal, sorted fastest-to-slowest (0 = fastest). */
    std::size_t memory_level {0};
    /** @brief Pointer to the first core ID in the `core_ids` array. */
    numa_core_id_t const *first_core_id {nullptr};
    /** @brief Number of items in the `core_ids` array. */
    std::size_t core_count {0};
    /** @brief Huge page sizes available on this NUMA node. */
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
    /** @brief The NUMA node these cores live on. */
    numa_node_id_t node_id {-1};
    /** @brief Index of the local memory domain (this node). */
    memory_domain_index_t memory_domain_index {}; // ? Which memory domain these cores allocate from
    /** @brief QoS ordinal, sorted least-to-most performant. */
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
    /** @brief Pointer to the first core ID in this domain. */
    numa_core_id_t const *first_core_id {nullptr};
    /** @brief Number of cores in this domain. */
    std::size_t core_count {0};
};

/** @brief Sentinel for `compute_domain_t::capacity` when the platform exposes no per-core throughput. */
static constexpr std::size_t capacity_unknown_k = 0;

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
    numa_node_t *nodes_ {nullptr};                // ? Memory domains (one per NUMA node)
    numa_core_id_t *node_core_ids_ {nullptr};     // ? Core IDs in [0, threads_count), grouped by node then QoS
    compute_domain_t *compute_domains_ {nullptr}; // ? Compute domains (same-QoS core runs within a node)
    std::size_t nodes_count_ {0};                 // ? Number of memory domains / NUMA nodes
    std::size_t cores_count_ {0};                 // ? Total number of cores in all nodes
    std::size_t compute_domains_count_ {0};       // ? Number of compute domains
    std::size_t compute_levels_count_ {1};        // ? Number of distinct QoS classes (>= 1)
    std::size_t memory_levels_count_ {1};         // ? Number of distinct memory tiers (>= 1)

  public:
    constexpr numa_topology() noexcept = default;
    numa_topology(numa_topology &&o) noexcept
        : allocator_(std::move(o.allocator_)), nodes_(o.nodes_), node_core_ids_(o.node_core_ids_),
          compute_domains_(o.compute_domains_), nodes_count_(o.nodes_count_), cores_count_(o.cores_count_),
          compute_domains_count_(o.compute_domains_count_), compute_levels_count_(o.compute_levels_count_),
          memory_levels_count_(o.memory_levels_count_) {
        o.nodes_ = nullptr;
        o.node_core_ids_ = nullptr;
        o.compute_domains_ = nullptr;
        o.nodes_count_ = 0;
        o.cores_count_ = 0;
        o.compute_domains_count_ = 0;
        o.compute_levels_count_ = 1;
        o.memory_levels_count_ = 1;
    }

    numa_topology &operator=(numa_topology &&other) noexcept {
        if (this != &other) {
            reset(); // ? Reset the current state
            allocator_ = std::move(other.allocator_);
            nodes_ = std::exchange(other.nodes_, nullptr);
            node_core_ids_ = std::exchange(other.node_core_ids_, nullptr);
            compute_domains_ = std::exchange(other.compute_domains_, nullptr);
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
        cores_allocator_t cores_alloc {allocator_};
        nodes_allocator_t nodes_alloc {allocator_};
        domains_allocator_t domains_alloc {allocator_};

        if (node_core_ids_) cores_alloc.deallocate(node_core_ids_, cores_count_);
        if (nodes_) nodes_alloc.deallocate(nodes_, nodes_count_);
        if (compute_domains_) domains_alloc.deallocate(compute_domains_, cores_count_);

        nodes_ = nullptr;
        node_core_ids_ = nullptr;
        compute_domains_ = nullptr;
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
        struct bitmask *numa_mask = nullptr;
        numa_node_t *nodes_ptr = nullptr;
        numa_core_id_t *core_ids_ptr = nullptr;
        compute_domain_t *domains_ptr = nullptr;
        std::size_t *core_capacities = nullptr; // ? Scheduler capacity keyed by core id, cached for the QoS split
        numa_node_id_t max_numa_node_id = -1;

        // Allocators must be visible to the cleanup path
        nodes_allocator_t nodes_alloc {allocator_};
        cores_allocator_t cores_alloc {allocator_};
        domains_allocator_t domains_alloc {allocator_};
        capacities_allocator_t capacities_alloc {allocator_};

        // These counters are reused in the failure handler
        std::size_t fetched_nodes = 0, fetched_cores = 0, configured_cores = 0;

        if (::numa_available() < 0) goto failed_harvest; // ! Linux kernel lacks NUMA support
        ::numa_node_to_cpu_update();                     // ? Reset the outdated stale state

        numa_mask = ::numa_allocate_cpumask();
        if (!numa_mask) goto failed_harvest; // ! Allocation failed

        // First pass - measure
        max_numa_node_id = ::numa_max_node();
        for (numa_node_id_t node_id = 0; node_id <= max_numa_node_id; ++node_id) {
            long long dummy;
            if (::numa_node_size64(node_id, &dummy) < 0) continue; // ! Offline node
            ::numa_bitmask_clearall(numa_mask);
            if (::numa_node_to_cpus(node_id, numa_mask) < 0) continue; // ! Invalid CPU map
            // A cpuless memory domain (HBM-flat, CXL expander, GPU HBM) reports zero cores, yet is a
            // valid memory domain - count the node and add its (possibly zero) cores.
            std::size_t const node_cores = static_cast<std::size_t>(::numa_bitmask_weight(numa_mask));
            fetched_nodes += 1;
            fetched_cores += node_cores;
        }
        if (fetched_nodes == 0) goto failed_harvest; // ! Zero nodes is not a valid state

        // Second pass - allocate. At most one compute domain per core (fully heterogeneous node).
        nodes_ptr = nodes_alloc.allocate(fetched_nodes);
        core_ids_ptr = cores_alloc.allocate(fetched_cores);
        domains_ptr = domains_alloc.allocate(fetched_cores);
        if (!nodes_ptr || !core_ids_ptr || !domains_ptr) goto failed_harvest; // ! Allocation failed

        // A scratch table of every configured CPU's capacity, filled once below and read by the
        // per-node QoS split (which is O(n^2) in comparisons) instead of re-opening sysfs each time.
        configured_cores = static_cast<std::size_t>(::numa_num_configured_cpus());
        if (configured_cores == 0) goto failed_harvest; // ! No CPUs is not a valid state
        core_capacities = capacities_alloc.allocate(configured_cores);
        if (!core_capacities) goto failed_harvest; // ! Allocation failed

        // Populate
        for (numa_node_id_t node_id = 0, core_index = 0, node_index = 0; node_id <= max_numa_node_id; ++node_id) {
            long long free_memory_size; // ? Only an out-parameter, the total size comes back as the return value
            long long const total_memory_size = ::numa_node_size64(node_id, &free_memory_size);
            if (total_memory_size < 0) continue;
            ::numa_bitmask_clearall(numa_mask);
            if (::numa_node_to_cpus(node_id, numa_mask) < 0) continue;

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

        // Commit
        nodes_ = nodes_ptr;
        node_core_ids_ = core_ids_ptr;
        nodes_count_ = fetched_nodes;
        cores_count_ = fetched_cores;
        ::numa_free_cpumask(numa_mask); // ? Clean up

        // Let's sort all the nodes by their socket ID, then by number of cores, then by first core ID
        bubble_sort(nodes_, nodes_count_, [](numa_node_t const &a, numa_node_t const &b) noexcept {
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
            compute_domains_ = domains_ptr;
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
        if (std::size_t *raw_tiers = capacities_alloc.allocate(nodes_count_)) {
            for (std::size_t i = 0; i < nodes_count_; ++i) raw_tiers[i] = get_memory_tier_for_node(nodes_[i].node_id);
            memory_levels_count_ = dense_rank(
                nodes_count_, [raw_tiers](std::size_t index) noexcept { return raw_tiers[index]; },
                [this, raw_tiers](std::size_t index, std::size_t rank) noexcept { nodes_[index].memory_level = rank; });
            capacities_alloc.deallocate(raw_tiers, nodes_count_);
        }

        capacities_alloc.deallocate(core_capacities, configured_cores); // ? Scratch, not part of the committed state
        return true;

    failed_harvest:
        if (nodes_ptr) nodes_alloc.deallocate(nodes_ptr, fetched_nodes);
        if (core_ids_ptr) cores_alloc.deallocate(core_ids_ptr, fetched_cores);
        if (domains_ptr) domains_alloc.deallocate(domains_ptr, fetched_cores);
        if (core_capacities) capacities_alloc.deallocate(core_capacities, configured_cores);
        if (numa_mask) ::numa_free_cpumask(numa_mask);
#endif // FU_WITH_TOPOLOGY
#if defined(__APPLE__)
        return try_harvest_apple();
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

        nodes_allocator_t nodes_alloc {allocator_};
        cores_allocator_t cores_alloc {allocator_};
        domains_allocator_t domains_alloc {allocator_};

        // `compute_domains_` is sized to `cores_count_` across the class (at most one domain per core).
        numa_node_t *nodes_ptr = nodes_alloc.allocate(1);
        numa_core_id_t *core_ids_ptr = cores_alloc.allocate(total_cores);
        compute_domain_t *domains_ptr = domains_alloc.allocate(total_cores);
        if (!nodes_ptr || !core_ids_ptr || !domains_ptr) {
            if (nodes_ptr) nodes_alloc.deallocate(nodes_ptr, 1);
            if (core_ids_ptr) cores_alloc.deallocate(core_ids_ptr, total_cores);
            if (domains_ptr) domains_alloc.deallocate(domains_ptr, total_cores);
            return false;
        }
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

        reset(); // ? Free any prior state before committing
        nodes_ = nodes_ptr;
        node_core_ids_ = core_ids_ptr;
        compute_domains_ = domains_ptr;
        nodes_count_ = 1;
        cores_count_ = total_cores;
        compute_domains_count_ = domains_written;
        compute_levels_count_ = levels_written; // ! Several clusters may share one level - not `domains_written`
        memory_levels_count_ = 1;
        return true;
    }
#endif // defined(__APPLE__)

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

        // Prepare scratch
        nodes_allocator_t nodes_alloc {allocator_};
        cores_allocator_t cores_alloc {allocator_};
        domains_allocator_t domains_alloc {allocator_};

        numa_node_t *scratch_nodes = nullptr;
        numa_core_id_t *scratch_core_ids = nullptr;
        compute_domain_t *scratch_domains = nullptr;
        if (other.nodes_count_) {
            scratch_nodes = nodes_alloc.allocate(other.nodes_count_);
            if (!scratch_nodes) return false; // ! OOM
        }
        if (other.cores_count_) {
            scratch_core_ids = cores_alloc.allocate(other.cores_count_);
            scratch_domains = domains_alloc.allocate(other.cores_count_);
            if (!scratch_core_ids || !scratch_domains) {
                if (scratch_nodes) nodes_alloc.deallocate(scratch_nodes, other.nodes_count_);
                if (scratch_core_ids) cores_alloc.deallocate(scratch_core_ids, other.cores_count_);
                if (scratch_domains) domains_alloc.deallocate(scratch_domains, other.cores_count_);
                return false; // ! OOM
            }
        }

        // Deep copy, re-basing every `first_core_id` into our own core-id block
        if (other.cores_count_ > 0)
            std::memcpy(scratch_core_ids, other.node_core_ids_, other.cores_count_ * sizeof(numa_core_id_t));
        for (std::size_t i = 0; i < other.nodes_count_; ++i) {
            scratch_nodes[i] = other.nodes_[i];
            std::ptrdiff_t const offset = other.nodes_[i].first_core_id - other.node_core_ids_;
            scratch_nodes[i].first_core_id = scratch_core_ids + offset;
        }
        for (std::size_t i = 0; i < other.compute_domains_count_; ++i) {
            scratch_domains[i] = other.compute_domains_[i];
            std::ptrdiff_t const offset = other.compute_domains_[i].first_core_id - other.node_core_ids_;
            scratch_domains[i].first_core_id = scratch_core_ids + offset;
        }

        reset(); // ? Free old buffers
        nodes_ = scratch_nodes;
        node_core_ids_ = scratch_core_ids;
        compute_domains_ = scratch_domains;
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
