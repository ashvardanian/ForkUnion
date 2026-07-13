//! Low-latency OpenMP-style NUMA-aware cross-platform fine-grained parallelism library.
//!
//! ForkUnion provides a minimalistic cross-platform thread-pool implementation for fork-join
//! parallelism, avoiding dynamic memory allocations, exceptions, system calls, and heavy
//! Compare-And-Swap instructions on the hot path.
//!
//! Unlike std.Thread.Pool (which is a task queue for async work), ForkUnion is designed for
//! data parallelism and tight parallel loops - think OpenMP's `#pragma omp parallel for`.
//!
//! Basic usage:
//! ```zig
//! const fu = @import("forkunion");
//!
//! const topo = try fu.Topology.init();
//! defer topo.deinit();
//!
//! var pool = try fu.Pool.init(topo, 4, .inclusive);
//! defer pool.deinit();
//!
//! // Execute work on each thread (like OpenMP parallel)
//! pool.forThreads(struct {
//!     fn work(thread_idx: usize, compute_domain_idx: usize) void {
//!         std.debug.print("Thread {}\n", .{thread_idx});
//!     }
//! }.work, {});
//!
//! // Distribute 1000 tasks across threads (like OpenMP parallel for)
//! var results = [_]i32{0} ** 1000;
//! pool.forN(1000, processTask, .{ .results = &results });
//! ```

const std = @import("std");
const builtin = @import("builtin");

// C ABI types
const c = struct {
    // Library metadata
    extern fn fu_version_major() c_int;
    extern fn fu_version_minor() c_int;
    extern fn fu_version_patch() c_int;
    extern fn fu_comptime_capabilities() u32;
    extern fn fu_runtime_capabilities() u32;
    extern fn fu_name_capabilities(caps: u32, buf: [*]u8, len: usize) usize;

    // Topology lifecycle
    extern fn fu_topology_new() ?*anyopaque;
    extern fn fu_topology_delete(topology: *anyopaque) void;

    // Compute topology
    extern fn fu_logical_cores_count(topology: *anyopaque) usize;
    extern fn fu_compute_domains_count(topology: *anyopaque) usize;
    extern fn fu_compute_levels_count(topology: *anyopaque) usize;
    extern fn fu_logical_cores_count_in(topology: *anyopaque, compute_domain_index: usize) usize;
    extern fn fu_compute_level_in(topology: *anyopaque, compute_domain_index: usize) usize;
    extern fn fu_compute_capacity_in(topology: *anyopaque, compute_domain_index: usize) usize;
    extern fn fu_compute_cache_bytes_in(topology: *anyopaque, compute_domain_index: usize) usize;

    // Memory topology
    extern fn fu_memory_domains_count(topology: *anyopaque) usize;
    extern fn fu_memory_levels_count(topology: *anyopaque) usize;
    extern fn fu_memory_level_in(topology: *anyopaque, memory_domain_index: usize) usize;
    extern fn fu_volume_ram(topology: *anyopaque) usize;
    extern fn fu_volume_ram_in(topology: *anyopaque, memory_domain_index: usize) usize;
    extern fn fu_volume_huge_pages(topology: *anyopaque) usize;
    extern fn fu_volume_huge_pages_in(topology: *anyopaque, memory_domain_index: usize) usize;
    extern fn fu_huge_pages_count(topology: *anyopaque) usize;
    extern fn fu_huge_pages_count_in(topology: *anyopaque, memory_domain_index: usize) usize;

    // Affinity
    extern fn fu_local_memory_of(topology: *anyopaque, compute_domain_index: usize) usize;
    extern fn fu_memory_distance(topology: *anyopaque, compute_domain_index: usize, memory_domain_index: usize) usize;
    extern fn fu_memory_bandwidth(topology: *anyopaque, compute_domain_index: usize, memory_domain_index: usize) usize;
    extern fn fu_memory_latency(topology: *anyopaque, compute_domain_index: usize, memory_domain_index: usize) usize;

    // Allocation
    extern fn fu_memory_domain_id_at_index(topology: *anyopaque, memory_domain_index: usize) i32;
    extern fn fu_allocate_on_domain_id(memory_domain_id: i32, bytes: usize) ?*anyopaque;
    extern fn fu_allocate_at_least_on_domain_id(
        memory_domain_id: i32,
        minimum_bytes: usize,
        allocated_bytes: *usize,
        bytes_per_page: *usize,
    ) ?*anyopaque;
    extern fn fu_free_on_domain_id(memory_domain_id: i32, pointer: *anyopaque, bytes: usize) void;
    extern fn fu_allocate_symmetric(
        topology: *anyopaque,
        bytes_per_domain: usize,
        stride_bytes: *usize,
        memory_domains_count: *usize,
        total_bytes: *usize,
        bytes_per_page: *usize,
    ) ?*anyopaque;
    extern fn fu_free_symmetric(base: *anyopaque, total_bytes: usize) void;

    // Pool lifecycle & introspection
    extern fn fu_pool_new(name: ?[*:0]const u8, allowed: u32) ?*anyopaque;
    extern fn fu_pool_delete(pool: *anyopaque) void;
    extern fn fu_pool_spawn(topology: *anyopaque, pool: *anyopaque, threads: usize, exclusivity: c_int) c_int;
    extern fn fu_pool_spawn_on(topology: *anyopaque, pool: *anyopaque, compute_domain_index: usize, threads: usize, exclusivity: c_int) c_int;
    extern fn fu_pool_terminate(pool: *anyopaque) void;
    extern fn fu_pool_sleep(pool: *anyopaque, micros: usize) void;
    extern fn fu_pool_caller_exclusivity(pool: *anyopaque) c_int;
    extern fn fu_pool_threads_count(pool: *anyopaque) usize;
    extern fn fu_pool_compute_domains_count(pool: *anyopaque) usize;
    extern fn fu_pool_threads_count_in(pool: *anyopaque, compute_domain_index: usize) usize;
    extern fn fu_pool_locate_thread_in(pool: *anyopaque, global_thread_index: usize, compute_domain_index: usize) usize;

    // Parallel dispatch
    extern fn fu_pool_for_threads(
        pool: *anyopaque,
        callback: *const fn (?*anyopaque, usize, usize) callconv(.c) void,
        context: ?*anyopaque,
    ) void;
    extern fn fu_pool_for_n(
        pool: *anyopaque,
        n: usize,
        callback: *const fn (?*anyopaque, usize, usize, usize) callconv(.c) void,
        context: ?*anyopaque,
    ) void;
    extern fn fu_pool_for_n_dynamic(
        pool: *anyopaque,
        n: usize,
        callback: *const fn (?*anyopaque, usize, usize, usize) callconv(.c) void,
        context: ?*anyopaque,
    ) void;
    extern fn fu_pool_for_slices(
        pool: *anyopaque,
        n: usize,
        callback: *const fn (?*anyopaque, usize, usize, usize, usize) callconv(.c) void,
        context: ?*anyopaque,
    ) void;

    // Generation tokens
    extern fn fu_pool_unsafe_for_threads(
        pool: *anyopaque,
        callback: *const fn (?*anyopaque, usize, usize) callconv(.c) void,
        context: ?*anyopaque,
    ) usize;
    extern fn fu_pool_is_complete(pool: *anyopaque, generation: usize) c_int;
    extern fn fu_pool_unsafe_join(pool: *anyopaque, generation: usize) void;
};

/// Errors that can occur during thread pool operations
pub const Error = error{
    /// Failed to create thread pool
    CreationFailed,
    /// Failed to spawn worker threads
    SpawnFailed,
    /// Platform not supported
    UnsupportedPlatform,
};

/// Defines whether the calling thread participates in task execution
pub const CallerExclusivity = enum(c_int) {
    /// Calling thread participates in workload (spawns N-1 workers)
    inclusive = 0,
    /// Calling thread only coordinates (spawns N workers)
    exclusive = 1,
};

/// A "prong" - metadata about a task's execution context
pub const Prong = struct {
    /// The logical index of the task being processed
    task_index: usize,
    /// The physical thread executing this task
    thread_index: usize,
    /// The compute domain (a same-QoS core cluster)
    compute_domain_index: usize,
};

/// Returns the library version as a struct
pub fn version() struct { major: u32, minor: u32, patch: u32 } {
    return .{
        .major = @intCast(c.fu_version_major()),
        .minor = @intCast(c.fu_version_minor()),
        .patch = @intCast(c.fu_version_patch()),
    };
}

/// Everything the library can do, whether decided when it was compiled or found on this machine.
///
/// One bit per facility, and two accessors ask two questions of the same bit. `comptimeCapabilities`
/// reports whether the code was _built_: a set `place_huge_pages_on_domain` means we compiled the
/// path that asks for them. `runtimeCapabilities` reports whether the machine _offers_ it now.
///
/// Neither implies the other. A binary that built `place_memory_on_domain` runs perfectly well on a
/// single-node box, where the runtime accessor never sets that bit; and a machine with four NUMA
/// nodes reports none of them to a build that left the topology out.
pub const Capabilities = packed struct(u32) {
    /// x86 `pause` instruction
    x86_pause: bool = false,
    /// x86-64 `tpause` instruction, with `WAITPKG` support
    x86_tpause: bool = false,
    /// Arm `yield` instruction
    arm64_yield: bool = false,
    /// AArch64 `wfet` instruction, with `FEAT_WFxT` support
    arm64_wfet: bool = false,
    /// RISC-V `pause` instruction
    risc5_pause: bool = false,
    /// RISC-V `WRS.STO` monitored wait, from the `Zawrs` extension
    risc5_wrs: bool = false,

    /// Own the raw OS thread handle instead of a `std::thread`
    os_threads: bool = false,
    /// Enumerate this machine's cores, compute domains, and memory domains
    topology: bool = false,
    /// Bind a thread to a set of cores, choosing where it runs
    place_threads_by_affinity: bool = false,
    /// Steer a thread onto a class of core at creation, choosing where it runs
    place_threads_by_core_class: bool = false,
    /// Reclass a thread's scheduler to sleep or wake it, choosing when it runs
    reschedule_threads_by_class: bool = false,
    /// Place a buffer's pages on a chosen memory domain
    place_memory_on_domain: bool = false,
    /// Place larger-than-base pages on a chosen memory domain
    place_huge_pages_on_domain: bool = false,
    /// The kernel promotes base pages to huge pages on its own
    huge_transparent_pages: bool = false,
    /// The domain-aware `colocated_pool` and `distributed_pool` are compiled in
    colocate_pools_on_domain: bool = false,

    _unused: u17 = 0,

    /// All-ones allow-mask: pass to a pool constructor to disable capability filtering.
    pub fn all() Capabilities {
        return @bitCast(@as(u32, 0xFFFF_FFFF));
    }
};

/// Which kernel facilities this build of ForkUnion was compiled to use.
pub fn comptimeCapabilities() Capabilities {
    return @bitCast(c.fu_comptime_capabilities());
}

/// The set `comptimeCapabilities` bits, comma-separated, like `"threads,topology"`.
///
/// POLISH: writes into a caller-provided buffer so the returned slice borrows `buf`; the
/// buffer must outlive the slice. Callers pass their own stack buffer.
pub fn comptimeCapabilitiesString(buf: []u8) []const u8 {
    const written = c.fu_name_capabilities(c.fu_comptime_capabilities(), buf.ptr, buf.len);
    return buf[0..written];
}

/// Which features this machine turned out to offer, probing the CPU and the memory system.
pub fn runtimeCapabilities() Capabilities {
    return @bitCast(c.fu_runtime_capabilities());
}

/// The set `runtimeCapabilities` bits, comma-separated, like `"arm64_yield,place_memory_on_domain"`.
///
/// POLISH: writes into a caller-provided buffer so the returned slice borrows `buf`; the
/// buffer must outlive the slice. Callers pass their own stack buffer.
pub fn runtimeCapabilitiesString(buf: []u8) []const u8 {
    const written = c.fu_name_capabilities(c.fu_runtime_capabilities(), buf.ptr, buf.len);
    return buf[0..written];
}

/// An explicit, owned handle to this machine's discovered compute and memory topology.
///
/// Build one with `init`, query it through the methods below, and hand it to pool constructors
/// and memory-domain allocators so they know which machine they are placing work and pages onto.
pub const Topology = struct {
    handle: *anyopaque,

    /// Discovers this machine's topology, returning an owned handle.
    pub fn init() Error!Topology {
        const h = c.fu_topology_new() orelse return Error.CreationFailed;
        return .{ .handle = h };
    }

    /// Releases the topology handle.
    pub fn deinit(self: Topology) void {
        c.fu_topology_delete(self.handle);
    }

    /// Returns the number of logical CPU cores available
    pub fn countLogicalCores(self: Topology) usize {
        return c.fu_logical_cores_count(self.handle);
    }

    /// Returns the number of memory domains available
    pub fn memoryDomainsCount(self: Topology) usize {
        return c.fu_memory_domains_count(self.handle);
    }

    /// Resolves a memory domain's dense index to the OS id the allocators take; -1 if out of range.
    pub fn memoryDomainIdAtIndex(self: Topology, memory_domain_index: usize) i32 {
        return c.fu_memory_domain_id_at_index(self.handle, memory_domain_index);
    }

    /// Returns the number of distinct thread compute_domains
    pub fn countComputeDomains(self: Topology) usize {
        return c.fu_compute_domains_count(self.handle);
    }

    /// Returns the number of logical cores backing a given compute domain (0 if out of range).
    pub fn countLogicalCoresIn(self: Topology, compute_domain_index: usize) usize {
        return c.fu_logical_cores_count_in(self.handle, compute_domain_index);
    }

    /// Returns the performance level of a compute domain (higher = more performant).
    pub fn computeLevelIn(self: Topology, compute_domain_index: usize) usize {
        return c.fu_compute_level_in(self.handle, compute_domain_index);
    }

    /// Returns the performance level of a memory domain (lower = faster: HBM < DDR < CXL).
    pub fn memoryLevelIn(self: Topology, memory_domain_index: usize) usize {
        return c.fu_memory_level_in(self.handle, memory_domain_index);
    }

    /// Returns the memory domain nearest a given compute domain (its local allocation target).
    pub fn localMemoryOf(self: Topology, compute_domain_index: usize) usize {
        return c.fu_local_memory_of(self.handle, compute_domain_index);
    }

    /// Returns the relative access distance from a compute domain to a memory domain (10 = local).
    pub fn memoryDistance(self: Topology, compute_domain_index: usize, memory_domain_index: usize) usize {
        return c.fu_memory_distance(self.handle, compute_domain_index, memory_domain_index);
    }

    /// Returns the HMAT read bandwidth (MB/s) from a compute domain to a memory domain, or 0 if unknown.
    pub fn memoryBandwidth(self: Topology, compute_domain_index: usize, memory_domain_index: usize) usize {
        return c.fu_memory_bandwidth(self.handle, compute_domain_index, memory_domain_index);
    }

    /// Returns the HMAT read latency (nanoseconds) from a compute domain to a memory domain, or 0 if unknown.
    pub fn memoryLatency(self: Topology, compute_domain_index: usize, memory_domain_index: usize) usize {
        return c.fu_memory_latency(self.handle, compute_domain_index, memory_domain_index);
    }

    /// Returns the number of distinct Quality-of-Service levels.
    ///
    /// May be smaller than `countComputeDomains`, as several domains can share one level - equally-fast
    /// cores may still be split across cache clusters, or across NUMA nodes.
    pub fn countComputeLevels(self: Topology) usize {
        return c.fu_compute_levels_count(self.handle);
    }

    /// Returns the number of distinct memory tiers, the memory-axis twin of `countComputeLevels`.
    pub fn countMemoryLevels(self: Topology) usize {
        return c.fu_memory_levels_count(self.handle);
    }

    /// Returns the relative throughput of one core in a compute domain (0 if unknown).
    ///
    /// A magnitude on the Linux `cpu_capacity` scale, where 1024 is the fastest core present. Weight
    /// work by this - `computeLevelIn` is a dense ordinal and must never be divided by. Platforms that
    /// rank cores without rating them report 0; weigh by core count instead.
    pub fn computeCapacityIn(self: Topology, compute_domain_index: usize) usize {
        return c.fu_compute_capacity_in(self.handle, compute_domain_index);
    }

    /// Returns the bytes of deepest cache private to a compute domain's cores (0 if unknown).
    ///
    /// Sizes a cache-resident chunk, a different question from how many chunks a domain deserves -
    /// domains of equal throughput may back onto very differently sized caches.
    pub fn computeCacheBytesIn(self: Topology, compute_domain_index: usize) usize {
        return c.fu_compute_cache_bytes_in(self.handle, compute_domain_index);
    }

    /// Returns the total RAM volume (bytes) across all memory domains, regardless of page size.
    pub fn volumeRam(self: Topology) usize {
        return c.fu_volume_ram(self.handle);
    }

    /// Returns the RAM volume (bytes) held by a given memory domain (0 if out of range).
    pub fn volumeRamIn(self: Topology, memory_domain_index: usize) usize {
        return c.fu_volume_ram_in(self.handle, memory_domain_index);
    }

    /// Returns the total huge-page volume (bytes) across all memory domains.
    pub fn volumeHugePages(self: Topology) usize {
        return c.fu_volume_huge_pages(self.handle);
    }

    /// Returns the huge-page volume (bytes) available in a given memory domain (0 if out of range).
    pub fn volumeHugePagesIn(self: Topology, memory_domain_index: usize) usize {
        return c.fu_volume_huge_pages_in(self.handle, memory_domain_index);
    }

    /// Returns the total number of free huge pages across all memory domains.
    pub fn countHugePages(self: Topology) usize {
        return c.fu_huge_pages_count(self.handle);
    }

    /// Returns the number of free huge pages in a given memory domain (0 if out of range).
    pub fn countHugePagesIn(self: Topology, memory_domain_index: usize) usize {
        return c.fu_huge_pages_count_in(self.handle, memory_domain_index);
    }
};

/// Result of a memory-domain allocation; carries only the OS id, so it can outlive the topology.
pub const AllocationResult = struct {
    memory_domain_id: i32,
    ptr: [*]u8,
    allocated_bytes: usize,
    bytes_per_page: usize,

    /// Returns the allocated memory as a slice
    pub fn asSlice(self: AllocationResult) []u8 {
        return self.ptr[0..self.allocated_bytes];
    }

    /// Frees the allocation
    pub fn free(self: AllocationResult) void {
        c.fu_free_on_domain_id(self.memory_domain_id, @ptrCast(self.ptr), self.allocated_bytes);
    }
};

/// Allocates memory on a memory domain with optimal page size; id from `Topology.memoryDomainIdAtIndex`.
pub fn allocateAtLeast(memory_domain_id: i32, minimum_bytes: usize) ?AllocationResult {
    var allocated_bytes: usize = undefined;
    var bytes_per_page: usize = undefined;

    const ptr = c.fu_allocate_at_least_on_domain_id(
        memory_domain_id,
        minimum_bytes,
        &allocated_bytes,
        &bytes_per_page,
    ) orelse return null;

    return .{
        .memory_domain_id = memory_domain_id,
        .ptr = @ptrCast(@alignCast(ptr)),
        .allocated_bytes = allocated_bytes,
        .bytes_per_page = bytes_per_page,
    };
}

/// Allocates exactly the requested bytes on a memory domain; id from `Topology.memoryDomainIdAtIndex`.
pub fn allocate(memory_domain_id: i32, bytes: usize) ?[*]u8 {
    const ptr = c.fu_allocate_on_domain_id(memory_domain_id, bytes) orelse return null;
    return @ptrCast(@alignCast(ptr));
}

/// Allocator bound to a single memory domain, compatible with Zig's `std.mem.Allocator` interface.
///
/// This type is the allocation API: build one for a memory domain's OS id - from
/// `Topology.memoryDomainIdAtIndex` - and it both hands out `AllocationResult`s through
/// `allocateAtLeast`/`allocate` and backs a `std.mem.Allocator` through `allocator`.
pub const DomainAllocator = struct {
    memory_domain_id: i32,

    const Self = @This();
    const Allocator = std.mem.Allocator;

    const Header = packed struct {
        base_addr: usize,
        allocated_bytes: usize,
    };

    const vtable = Allocator.VTable{
        .alloc = alloc,
        .resize = resize,
        .remap = remap,
        .free = free,
    };

    /// Binds to the memory domain named by @p memory_domain_id, or null if the id is the -1 sentinel.
    pub fn init(memory_domain_id: i32) ?Self {
        if (memory_domain_id < 0) return null;
        return .{ .memory_domain_id = memory_domain_id };
    }

    /// Allocates at least @p minimum_bytes on this domain with the optimal page size, or null on failure.
    pub fn allocateAtLeast(self: Self, minimum_bytes: usize) ?AllocationResult {
        var allocated_bytes: usize = undefined;
        var bytes_per_page: usize = undefined;
        const ptr = c.fu_allocate_at_least_on_domain_id(
            self.memory_domain_id,
            minimum_bytes,
            &allocated_bytes,
            &bytes_per_page,
        ) orelse return null;
        return .{
            .memory_domain_id = self.memory_domain_id,
            .ptr = @ptrCast(@alignCast(ptr)),
            .allocated_bytes = allocated_bytes,
            .bytes_per_page = bytes_per_page,
        };
    }

    /// Allocates exactly @p bytes on this domain, or null on failure.
    pub fn allocate(self: Self, bytes: usize) ?[*]u8 {
        const ptr = c.fu_allocate_on_domain_id(self.memory_domain_id, bytes) orelse return null;
        return @ptrCast(@alignCast(ptr));
    }

    pub fn allocator(self: *Self) Allocator {
        return .{ .ptr = self, .vtable = &vtable };
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        _ = ret_addr;
        const self: *Self = @ptrCast(@alignCast(ctx));
        const effective_len = if (len == 0) 1 else len;
        const slice = self.allocSlice(effective_len, alignment) orelse return null;
        return slice.ptr;
    }

    fn resize(
        ctx: *anyopaque,
        buf: []u8,
        alignment: std.mem.Alignment,
        new_len: usize,
        ret_addr: usize,
    ) bool {
        _ = ret_addr;
        const self: *Self = @ptrCast(@alignCast(ctx));
        return self.resizeInPlace(buf, alignment, new_len);
    }

    fn remap(
        ctx: *anyopaque,
        buf: []u8,
        alignment: std.mem.Alignment,
        new_len: usize,
        ret_addr: usize,
    ) ?[*]u8 {
        _ = ret_addr;
        const self: *Self = @ptrCast(@alignCast(ctx));
        const result = self.remapSlice(buf, alignment, new_len) orelse return null;
        return result.ptr;
    }

    fn free(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        _ = alignment;
        _ = ret_addr;
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.freeSlice(buf);
    }

    fn allocSlice(self: *Self, len: usize, alignment: std.mem.Alignment) ?[]u8 {
        const header_size = @sizeOf(Header);
        const alignment_bytes = alignment.toByteUnits();
        const with_header = std.math.add(usize, len, header_size) catch return null;
        const request_bytes = std.math.add(usize, with_header, alignment_bytes) catch return null;

        var allocated_bytes: usize = undefined;
        var bytes_per_page: usize = undefined;
        const raw_ptr = c.fu_allocate_at_least_on_domain_id(
            self.memory_domain_id,
            request_bytes,
            &allocated_bytes,
            &bytes_per_page,
        ) orelse return null;

        const base_addr = @intFromPtr(raw_ptr);
        const data_addr = alignment.forward(base_addr + header_size);
        if (data_addr + len > base_addr + allocated_bytes) {
            c.fu_free_on_domain_id(self.memory_domain_id, raw_ptr, allocated_bytes);
            return null;
        }

        const header_ptr = @as(*Header, @ptrFromInt(data_addr - header_size));
        header_ptr.* = .{
            .base_addr = base_addr,
            .allocated_bytes = allocated_bytes,
        };

        const data_ptr = @as([*]u8, @ptrFromInt(data_addr));
        return data_ptr[0..len];
    }

    fn resizeInPlace(self: *Self, buf: []u8, alignment: std.mem.Alignment, new_len: usize) bool {
        _ = self;
        _ = alignment;
        if (buf.len == 0) return false;
        if (new_len == 0) return false;
        if (new_len <= buf.len) return true;
        return false;
    }

    fn remapSlice(self: *Self, buf: []u8, alignment: std.mem.Alignment, new_len: usize) ?[]u8 {
        if (buf.len == 0) return null;
        if (new_len == 0) {
            self.freeSlice(buf);
            return buf[0..0];
        }
        if (new_len <= buf.len) return buf[0..new_len];

        const new_slice = self.allocSlice(new_len, alignment) orelse return null;
        @memcpy(new_slice[0..buf.len], buf);
        self.freeSlice(buf);
        return new_slice;
    }

    fn freeSlice(self: *Self, buf: []u8) void {
        if (buf.len == 0) return;
        const header_ptr = @as(*Header, @ptrFromInt(@intFromPtr(buf.ptr) - @sizeOf(Header)));
        const header = header_ptr.*;
        const base_ptr = @as(*anyopaque, @ptrFromInt(header.base_addr));
        c.fu_free_on_domain_id(self.memory_domain_id, base_ptr, header.allocated_bytes);
    }
};

/// One full length-`len` copy of a sequence per memory domain, so every thread reads a node-local replica.
///
/// A thin owner of one symmetric mapping the allocator stripes across the nodes - replica `d` lives at
/// `base + d * stride_bytes` and holds `len` elements. Raw uninitialized storage: the caller fills every
/// replica and keeps them coherent, mirroring the C++ `replicated_array`. `T` must be plain-old-data,
/// since the container runs no constructors or destructors.
pub fn ReplicatedArray(comptime T: type) type {
    return struct {
        const Self = @This();

        base_bytes: ?[*]u8 = null,
        stride_bytes: usize = 0,
        domains: usize = 0,
        total_bytes: usize = 0,
        len: usize = 0,

        /// Allocates one uninitialized length-`n` replica per memory domain, or null on failure.
        pub fn init(topology: Topology, n: usize) ?Self {
            if (n == 0) return Self{};
            var stride_bytes: usize = 0;
            var domains: usize = 0;
            var total_bytes: usize = 0;
            var bytes_per_page: usize = 0;
            const base_bytes = c.fu_allocate_symmetric(
                topology.handle,
                n * @sizeOf(T),
                &stride_bytes,
                &domains,
                &total_bytes,
                &bytes_per_page,
            ) orelse return null;
            return .{
                .base_bytes = @ptrCast(@alignCast(base_bytes)),
                .stride_bytes = stride_bytes,
                .domains = domains,
                .total_bytes = total_bytes,
                .len = n,
            };
        }

        /// Unmaps the whole striped mapping.
        pub fn deinit(self: *Self) void {
            if (self.base_bytes) |base_bytes| c.fu_free_symmetric(@ptrCast(base_bytes), self.total_bytes);
            self.* = .{};
        }

        /// Whether the container holds no elements.
        pub fn isEmpty(self: Self) bool {
            return self.len == 0;
        }

        /// The typed base pointer of the first replica's storage.
        pub fn base(self: Self) [*]T {
            return @ptrCast(@alignCast(self.base_bytes.?));
        }

        /// The number of per-domain replicas.
        pub fn memoryDomainsCount(self: Self) usize {
            return self.domains;
        }

        /// The page-aligned byte distance between consecutive replicas.
        pub fn strideBytes(self: Self) usize {
            return self.stride_bytes;
        }

        /// The whole replica living on `memory_domain_index`.
        pub fn onMemoryDomain(self: Self, memory_domain_index: usize) []T {
            const slice_base = self.base_bytes.? + memory_domain_index * self.stride_bytes;
            const typed: [*]T = @ptrCast(@alignCast(slice_base));
            return typed[0..self.len];
        }

        /// One element of the replica on `memory_domain_index`.
        pub fn at(self: Self, memory_domain_index: usize, local_index: usize) *T {
            return &self.onMemoryDomain(memory_domain_index)[local_index];
        }
    };
}

/// A sequence partitioned across memory domains as contiguous segments, each element stored once.
///
/// Each domain owns a contiguous logical segment of `segment` elements, so element `i` lives at
/// `{i / segment, i % segment}` - see `locationOf` - and a scan of a domain's shard is sequential in
/// memory. Backed by one symmetric mapping; the trailing shard may be short. `T` must be plain-old-data,
/// mirroring `ReplicatedArray`.
pub fn ShardedArray(comptime T: type) type {
    return struct {
        const Self = @This();

        base_bytes: ?[*]u8 = null,
        stride_bytes: usize = 0,
        domains: usize = 0,
        total_bytes: usize = 0,
        len: usize = 0,
        segment: usize = 0,

        /// Allocates uninitialized storage for `n` elements partitioned round-robin across the domains.
        pub fn init(topology: Topology, n: usize) ?Self {
            if (n == 0) return Self{};
            const domains = topology.memoryDomainsCount();
            if (domains == 0) return null;
            const segment = (n + domains - 1) / domains;
            var stride_bytes: usize = 0;
            var domains_out: usize = 0;
            var total_bytes: usize = 0;
            var bytes_per_page: usize = 0;
            const base_bytes = c.fu_allocate_symmetric(
                topology.handle,
                segment * @sizeOf(T),
                &stride_bytes,
                &domains_out,
                &total_bytes,
                &bytes_per_page,
            ) orelse return null;
            return .{
                .base_bytes = @ptrCast(@alignCast(base_bytes)),
                .stride_bytes = stride_bytes,
                .domains = domains_out,
                .total_bytes = total_bytes,
                .len = n,
                .segment = segment,
            };
        }

        /// Unmaps the whole striped mapping.
        pub fn deinit(self: *Self) void {
            if (self.base_bytes) |base_bytes| c.fu_free_symmetric(@ptrCast(base_bytes), self.total_bytes);
            self.* = .{};
        }

        /// Whether the container holds no elements.
        pub fn isEmpty(self: Self) bool {
            return self.len == 0;
        }

        /// The typed base pointer of the first shard's storage.
        pub fn base(self: Self) [*]T {
            return @ptrCast(@alignCast(self.base_bytes.?));
        }

        /// The number of shards, one per memory domain.
        pub fn memoryDomainsCount(self: Self) usize {
            return self.domains;
        }

        /// The page-aligned byte distance between consecutive shards.
        pub fn strideBytes(self: Self) usize {
            return self.stride_bytes;
        }

        /// How many elements the shard on `memory_domain_index` holds - a trailing shard may be shorter.
        pub fn lengthOnMemoryDomain(self: Self, memory_domain_index: usize) usize {
            const start = memory_domain_index * self.segment;
            if (start >= self.len) return 0;
            return @min(self.len - start, self.segment);
        }

        /// The memory domain and local index that store logical element `logical_index`.
        pub fn locationOf(self: Self, logical_index: usize) struct { memory_domain: usize, local_index: usize } {
            return .{
                .memory_domain = logical_index / self.segment,
                .local_index = logical_index % self.segment,
            };
        }

        /// The logical index of the element at `local_index` on `memory_domain_index` - inverse of `locationOf`.
        pub fn logicalIndexOf(self: Self, memory_domain_index: usize, local_index: usize) usize {
            return memory_domain_index * self.segment + local_index;
        }

        /// The whole shard living on `memory_domain_index`.
        pub fn onMemoryDomain(self: Self, memory_domain_index: usize) []T {
            const slice_base = self.base_bytes.? + memory_domain_index * self.stride_bytes;
            const typed: [*]T = @ptrCast(@alignCast(slice_base));
            return typed[0..self.lengthOnMemoryDomain(memory_domain_index)];
        }

        /// The single home of a logical element.
        pub fn at(self: Self, memory_domain_index: usize, local_index: usize) *T {
            return &self.onMemoryDomain(memory_domain_index)[local_index];
        }
    };
}

/// Thread pool for fork-join parallelism
pub const Pool = struct {
    handle: *anyopaque,

    /// Creates a new thread pool
    pub fn init(topology: Topology, thread_count: usize, exclusivity: CallerExclusivity) Error!Pool {
        return initNamed(topology, null, thread_count, exclusivity);
    }

    /// Creates a new named thread pool
    pub fn initNamed(
        topology: Topology,
        name: ?[]const u8,
        thread_count: usize,
        exclusivity: CallerExclusivity,
    ) Error!Pool {
        return initNamedWithCapabilities(topology, name, thread_count, exclusivity, Capabilities.all());
    }

    /// As `initNamed`, but constrains the pool to `allowed`: clear a waiter bit to force a
    /// lower-priority busy-wait, or clear `place_memory_on_domain` to force the flat (non-NUMA) pool.
    pub fn initNamedWithCapabilities(
        topology: Topology,
        name: ?[]const u8,
        thread_count: usize,
        exclusivity: CallerExclusivity,
        allowed: Capabilities,
    ) Error!Pool {
        // Convert name to null-terminated string if provided
        // SAFETY: C library copies name into internal buffer immediately
        var name_buf: [16:0]u8 = undefined;
        const name_z: ?[*:0]const u8 = if (name) |n|
            std.fmt.bufPrintZ(&name_buf, "{s}", .{n[0..@min(n.len, 15)]}) catch unreachable
        else
            null;

        const handle = c.fu_pool_new(name_z, @bitCast(allowed)) orelse return Error.CreationFailed;
        errdefer c.fu_pool_delete(handle);

        // C++ validates threads > 0 and returns false if invalid
        const success = c.fu_pool_spawn(topology.handle, handle, thread_count, @intFromEnum(exclusivity));
        if (success == 0) return Error.SpawnFailed;

        return .{ .handle = handle };
    }

    /// Spawns a thread pool pinned to a single compute domain (cores of one QoS + locality).
    ///
    /// The pool's threads and memory-domain-local allocations stay on `compute_domain_index`, in
    /// `0..countComputeDomains()`. Spawn one per compute domain and coordinate them from a
    /// single thread with the generation-token API. On builds without NUMA, only compute
    /// domain 0 is valid.
    pub fn spawnOn(topology: Topology, compute_domain_index: usize, thread_count: usize, exclusivity: CallerExclusivity) Error!Pool {
        return spawnOnWithCapabilities(topology, compute_domain_index, thread_count, exclusivity, Capabilities.all());
    }

    /// As `spawnOn`, but constrains the colocated pool to `allowed`.
    pub fn spawnOnWithCapabilities(
        topology: Topology,
        compute_domain_index: usize,
        thread_count: usize,
        exclusivity: CallerExclusivity,
        allowed: Capabilities,
    ) Error!Pool {
        const handle = c.fu_pool_new(null, @bitCast(allowed)) orelse return Error.CreationFailed;
        errdefer c.fu_pool_delete(handle);

        const success = c.fu_pool_spawn_on(topology.handle, handle, compute_domain_index, thread_count, @intFromEnum(exclusivity));
        if (success == 0) return Error.SpawnFailed;

        return .{ .handle = handle };
    }

    /// Destroys the thread pool
    pub fn deinit(self: Pool) void {
        c.fu_pool_delete(self.handle);
    }

    /// Returns the number of threads in the pool
    pub fn threads(self: *const Pool) usize {
        return c.fu_pool_threads_count(self.handle);
    }

    /// Returns whether the calling thread participates in the workload.
    ///
    /// Queries the pool directly rather than caching, so it stays correct across
    /// `terminate` and re-spawning with a different exclusivity.
    pub fn callerExclusivity(self: *const Pool) CallerExclusivity {
        return @enumFromInt(c.fu_pool_caller_exclusivity(self.handle));
    }

    /// Returns the number of compute_domains in the pool
    pub fn compute_domains(self: *const Pool) usize {
        return c.fu_pool_compute_domains_count(self.handle);
    }

    /// Returns the number of threads in a specific compute_domain
    pub fn countThreadsIn(self: *const Pool, compute_domain_index: usize) usize {
        return c.fu_pool_threads_count_in(self.handle, compute_domain_index);
    }

    /// Converts global thread index to local index within compute_domain
    pub fn locateThreadIn(self: *const Pool, global_thread_index: usize, compute_domain_index: usize) usize {
        return c.fu_pool_locate_thread_in(self.handle, global_thread_index, compute_domain_index);
    }

    /// Terminates all worker threads (pool can be respawned)
    pub fn terminate(self: *const Pool) void {
        c.fu_pool_terminate(self.handle);
    }

    /// Puts worker threads into power-saving sleep state
    pub fn sleep(self: *const Pool, microseconds: usize) void {
        c.fu_pool_sleep(self.handle, microseconds);
    }

    /// Executes a callback on all threads (blocking)
    ///
    /// The callback function signature must match the context type:
    /// - If context is `void`: `fn(usize, usize) void`
    /// - If context is type `T`: `fn(usize, usize, T) void`
    pub fn forThreads(
        self: *const Pool,
        comptime func: anytype,
        context: anytype,
    ) void {
        const Context = @TypeOf(context);

        // Validate function signature at compile time
        const expected_type = if (Context == void)
            fn (usize, usize) void
        else
            fn (usize, usize, Context) void;

        if (@TypeOf(func) != expected_type) {
            @compileError("Function signature must be: " ++ @typeName(expected_type));
        }

        if (Context == void) {
            const Wrapper = struct {
                fn callback(_: ?*anyopaque, thread_idx: usize, compute_domain_idx: usize) callconv(.c) void {
                    func(thread_idx, compute_domain_idx);
                }
            };
            c.fu_pool_for_threads(self.handle, Wrapper.callback, null);
        } else {
            const Wrapper = struct {
                fn callback(ctx: ?*anyopaque, thread_idx: usize, compute_domain_idx: usize) callconv(.c) void {
                    const typed_ctx: *const Context = @ptrCast(@alignCast(ctx));
                    func(thread_idx, compute_domain_idx, typed_ctx.*);
                }
            };
            c.fu_pool_for_threads(self.handle, Wrapper.callback, @ptrCast(@constCast(&context)));
        }
    }

    /// Distributes N tasks across threads with static scheduling (blocking)
    ///
    /// The callback function signature must match the context type:
    /// - If context is `void`: `fn(Prong) void`
    /// - If context is type `T`: `fn(Prong, T) void`
    pub fn forN(
        self: *const Pool,
        n: usize,
        comptime func: anytype,
        context: anytype,
    ) void {
        const Context = @TypeOf(context);

        // Validate function signature at compile time
        const expected_type = if (Context == void)
            fn (Prong) void
        else
            fn (Prong, Context) void;

        if (@TypeOf(func) != expected_type) {
            @compileError("Function signature must be: " ++ @typeName(expected_type));
        }

        if (Context == void) {
            // Stateless path - no context
            const Wrapper = struct {
                fn callback(
                    _: ?*anyopaque,
                    task_idx: usize,
                    thread_idx: usize,
                    compute_domain_idx: usize,
                ) callconv(.c) void {
                    const prong = Prong{
                        .task_index = task_idx,
                        .thread_index = thread_idx,
                        .compute_domain_index = compute_domain_idx,
                    };
                    func(prong);
                }
            };
            c.fu_pool_for_n(self.handle, n, Wrapper.callback, null);
        } else {
            // Stateful path - pass context
            const Wrapper = struct {
                fn callback(
                    ctx: ?*anyopaque,
                    task_idx: usize,
                    thread_idx: usize,
                    compute_domain_idx: usize,
                ) callconv(.c) void {
                    const prong = Prong{
                        .task_index = task_idx,
                        .thread_index = thread_idx,
                        .compute_domain_index = compute_domain_idx,
                    };
                    // SAFETY: Context pointer valid for duration of blocking call
                    const typed_ctx: *const Context = @ptrCast(@alignCast(ctx));
                    func(prong, typed_ctx.*);
                }
            };
            c.fu_pool_for_n(self.handle, n, Wrapper.callback, @ptrCast(@constCast(&context)));
        }
    }

    /// Distributes N tasks with dynamic work-stealing (blocking)
    ///
    /// The callback function signature must match the context type:
    /// - If context is `void`: `fn(Prong) void`
    /// - If context is type `T`: `fn(Prong, T) void`
    pub fn forNDynamic(
        self: *const Pool,
        n: usize,
        comptime func: anytype,
        context: anytype,
    ) void {
        const Context = @TypeOf(context);

        // Validate function signature at compile time
        const expected_type = if (Context == void)
            fn (Prong) void
        else
            fn (Prong, Context) void;

        if (@TypeOf(func) != expected_type) {
            @compileError("Function signature must be: " ++ @typeName(expected_type));
        }

        if (Context == void) {
            // Stateless path - no context
            const Wrapper = struct {
                fn callback(
                    _: ?*anyopaque,
                    task_idx: usize,
                    thread_idx: usize,
                    compute_domain_idx: usize,
                ) callconv(.c) void {
                    const prong = Prong{
                        .task_index = task_idx,
                        .thread_index = thread_idx,
                        .compute_domain_index = compute_domain_idx,
                    };
                    func(prong);
                }
            };
            c.fu_pool_for_n_dynamic(self.handle, n, Wrapper.callback, null);
        } else {
            // Stateful path - pass context
            const Wrapper = struct {
                fn callback(
                    ctx: ?*anyopaque,
                    task_idx: usize,
                    thread_idx: usize,
                    compute_domain_idx: usize,
                ) callconv(.c) void {
                    const prong = Prong{
                        .task_index = task_idx,
                        .thread_index = thread_idx,
                        .compute_domain_index = compute_domain_idx,
                    };
                    // SAFETY: Context pointer valid for duration of blocking call
                    const typed_ctx: *const Context = @ptrCast(@alignCast(ctx));
                    func(prong, typed_ctx.*);
                }
            };
            c.fu_pool_for_n_dynamic(self.handle, n, Wrapper.callback, @ptrCast(@constCast(&context)));
        }
    }

    /// Distributes N tasks as slices (blocking)
    ///
    /// The callback function signature must match the context type:
    /// - If context is `void`: `fn(Prong, usize) void`
    /// - If context is type `T`: `fn(Prong, usize, T) void`
    ///
    /// The second parameter is the slice count for this chunk.
    pub fn forSlices(
        self: *const Pool,
        n: usize,
        comptime func: anytype,
        context: anytype,
    ) void {
        const Context = @TypeOf(context);

        // Validate function signature at compile time
        const expected_type = if (Context == void)
            fn (Prong, usize) void
        else
            fn (Prong, usize, Context) void;

        if (@TypeOf(func) != expected_type) {
            @compileError("Function signature must be: " ++ @typeName(expected_type));
        }

        if (Context == void) {
            // Stateless path - no context
            const Wrapper = struct {
                fn callback(
                    _: ?*anyopaque,
                    first_idx: usize,
                    count: usize,
                    thread_idx: usize,
                    compute_domain_idx: usize,
                ) callconv(.c) void {
                    const prong = Prong{
                        .task_index = first_idx,
                        .thread_index = thread_idx,
                        .compute_domain_index = compute_domain_idx,
                    };
                    func(prong, count);
                }
            };
            c.fu_pool_for_slices(self.handle, n, Wrapper.callback, null);
        } else {
            // Stateful path - pass context
            const Wrapper = struct {
                fn callback(
                    ctx: ?*anyopaque,
                    first_idx: usize,
                    count: usize,
                    thread_idx: usize,
                    compute_domain_idx: usize,
                ) callconv(.c) void {
                    const prong = Prong{
                        .task_index = first_idx,
                        .thread_index = thread_idx,
                        .compute_domain_index = compute_domain_idx,
                    };
                    // SAFETY: Context pointer valid for duration of blocking call
                    const typed_ctx: *const Context = @ptrCast(@alignCast(ctx));
                    func(prong, count, typed_ctx.*);
                }
            };
            c.fu_pool_for_slices(self.handle, n, Wrapper.callback, @ptrCast(@constCast(&context)));
        }
    }

    /// Executes callback on all threads without blocking (unsafe).
    /// Returns an always-odd generation token to pass to `isComplete` or `unsafeJoin`.
    ///
    /// The callback function signature must match the context type:
    /// - If context is `void`: `fn(usize, usize) void`
    /// - Otherwise context must be a pointer (like `*const T`), received as-is:
    ///   `fn(usize, usize, @TypeOf(context)) void`
    ///
    /// Unlike the blocking `forThreads`, this call returns while worker threads may
    /// still be running, so the context can't be copied into this stack frame: it must
    /// be a caller-owned pointer whose pointee outlives `unsafeJoin`.
    pub fn unsafeForThreads(
        self: *const Pool,
        comptime func: anytype,
        context: anytype,
    ) usize {
        const Context = @TypeOf(context);

        if (Context == void) {
            // Validate function signature at compile time
            const expected_type = fn (usize, usize) void;
            if (@TypeOf(func) != expected_type) {
                @compileError("Function signature must be: " ++ @typeName(expected_type));
            }
            const Wrapper = struct {
                fn callback(_: ?*anyopaque, thread_index: usize, compute_domain_index: usize) callconv(.c) void {
                    func(thread_index, compute_domain_index);
                }
            };
            return c.fu_pool_unsafe_for_threads(self.handle, Wrapper.callback, null);
        } else {
            // The dispatch returns before the workers finish, so a by-value context
            // would dangle - require a caller-owned pointer instead.
            if (@typeInfo(Context) != .pointer)
                @compileError("Non-blocking dispatch requires a pointer context (like `&my_context`) " ++
                    "whose pointee outlives `unsafeJoin`; got: " ++ @typeName(Context));

            // Validate function signature at compile time
            const expected_type = fn (usize, usize, Context) void;
            if (@TypeOf(func) != expected_type) {
                @compileError("Function signature must be: " ++ @typeName(expected_type));
            }
            const Wrapper = struct {
                fn callback(erased_context: ?*anyopaque, thread_index: usize, compute_domain_index: usize) callconv(.c) void {
                    const typed_context: Context = @ptrCast(@alignCast(erased_context));
                    func(thread_index, compute_domain_index, typed_context);
                }
            };
            return c.fu_pool_unsafe_for_threads(self.handle, Wrapper.callback, @ptrCast(@constCast(context)));
        }
    }

    /// Returns true if the given generation has completed.
    ///
    /// A `true` result also guarantees visibility of every contributor's writes. On
    /// caller-inclusive pools this can only turn `true` once `unsafeJoin` contributes
    /// the calling thread's slice, so poll-then-join is reserved for exclusive pools.
    pub fn isComplete(self: *const Pool, generation: usize) bool {
        return c.fu_pool_is_complete(self.handle, generation) != 0;
    }

    /// Blocks until the given generation completes (unsafe).
    /// On caller-inclusive pools this also executes the calling thread's slice.
    /// Idempotent: joining an already-joined generation returns immediately.
    pub fn unsafeJoin(self: *const Pool, generation: usize) void {
        c.fu_pool_unsafe_join(self.handle, generation);
    }
};

test "version info" {
    std.debug.print("Running test: version info\n", .{});
    const v = version();
    try std.testing.expect(v.major >= 0);
    try std.testing.expect(v.minor >= 0);
}

test "system capabilities" {
    std.debug.print("Running test: system capabilities\n", .{});
    const comptime_caps = comptimeCapabilities();
    const runtime_caps = runtimeCapabilities();
    var comptime_buf: [256]u8 = undefined;
    var runtime_buf: [256]u8 = undefined;
    std.debug.print("  comptime: {s}\n", .{comptimeCapabilitiesString(&comptime_buf)});
    std.debug.print("  runtime:  {s}\n", .{runtimeCapabilitiesString(&runtime_buf)});
    try std.testing.expect(runtimeCapabilitiesString(&runtime_buf).len > 0);

    const topo = try Topology.init();
    defer topo.deinit();

    // Threads are the one facility every supported platform has.
    try std.testing.expect(comptime_caps.os_threads);

    // The aggregate is implied, never hand-set: pools need threads and a topology to spawn onto.
    try std.testing.expectEqual(
        comptime_caps.os_threads and comptime_caps.topology,
        comptime_caps.colocate_pools_on_domain,
    );

    // Placing pages on a node presumes we discovered the nodes.
    if (comptime_caps.place_memory_on_domain) try std.testing.expect(comptime_caps.topology);

    // Without the pools, the library can still see exactly one domain, and never more.
    if (!comptime_caps.colocate_pools_on_domain) try std.testing.expectEqual(@as(usize, 1), topo.countComputeDomains());

    // One facility, two questions of the same bit: a machine can only _offer_ page placement if this
    // build compiled the path that asks for it.
    if (runtime_caps.place_memory_on_domain) try std.testing.expect(comptime_caps.place_memory_on_domain);
}

test "system metadata" {
    std.debug.print("Running test: system metadata\n", .{});
    const topo = try Topology.init();
    defer topo.deinit();

    const cores = topo.countLogicalCores();
    try std.testing.expect(cores > 0);

    const numa = topo.memoryDomainsCount();
    try std.testing.expect(numa > 0);

    const colocs = topo.countComputeDomains();
    try std.testing.expect(colocs > 0);
}

test "pool creation and destruction" {
    std.debug.print("Running test: pool creation and destruction\n", .{});
    const topo = try Topology.init();
    defer topo.deinit();
    var pool = try Pool.init(topo, 2, .inclusive);
    defer pool.deinit();

    try std.testing.expectEqual(2, pool.threads());
}

test "caller exclusivity query" {
    std.debug.print("Running test: caller exclusivity query\n", .{});
    // The pool is the single source of truth, queried live (not cached).
    const topo = try Topology.init();
    defer topo.deinit();
    var inclusive = try Pool.init(topo, 2, .inclusive);
    defer inclusive.deinit();
    try std.testing.expectEqual(CallerExclusivity.inclusive, inclusive.callerExclusivity());

    var exclusive = try Pool.init(topo, 2, .exclusive);
    defer exclusive.deinit();
    try std.testing.expectEqual(CallerExclusivity.exclusive, exclusive.callerExclusivity());
}

test "per-compute_domain pool" {
    std.debug.print("Running test: per-compute_domain pool\n", .{});
    const topo = try Topology.init();
    defer topo.deinit();
    const compute_domains = topo.countComputeDomains();
    try std.testing.expect(compute_domains >= 1);

    // A pool pinned to compute_domain 0, sized to that compute_domain's core count.
    const cores = @max(topo.countLogicalCoresIn(0), 1);
    var pool = try Pool.spawnOn(topo, 0, cores, .exclusive);
    defer pool.deinit();

    var counter = std.atomic.Value(usize).init(0);
    const context = struct { counter_ptr: *std.atomic.Value(usize) }{ .counter_ptr = &counter };
    const generation = pool.unsafeForThreads(struct {
        fn worker(thread_index: usize, compute_domain_index: usize, ctx: *const @TypeOf(context)) void {
            _ = thread_index;
            _ = compute_domain_index;
            _ = ctx.counter_ptr.fetchAdd(1, .monotonic);
        }
    }.worker, &context);
    pool.unsafeJoin(generation);
    try std.testing.expectEqual(pool.threads(), counter.load(.acquire));
}

test "named pool creation" {
    std.debug.print("Running test: named pool creation\n", .{});
    const topo = try Topology.init();
    defer topo.deinit();
    var pool = try Pool.initNamed(topo, null, 2, .inclusive);
    defer pool.deinit();

    try std.testing.expectEqual(2, pool.threads());
}

test "for_threads execution" {
    std.debug.print("Running test: for_threads execution\n", .{});
    const topo = try Topology.init();
    defer topo.deinit();
    var pool = try Pool.init(topo, 4, .inclusive);
    defer pool.deinit();

    var visited = [_]std.atomic.Value(bool){std.atomic.Value(bool).init(false)} ** 4;

    const Context = struct {
        visited_ptr: *[4]std.atomic.Value(bool),
    };

    pool.forThreads(struct {
        fn worker(thread_idx: usize, compute_domain_idx: usize, ctx: Context) void {
            _ = compute_domain_idx;
            if (thread_idx < 4) {
                ctx.visited_ptr[thread_idx].store(true, .release);
            }
        }
    }.worker, Context{ .visited_ptr = &visited });

    // Verify all threads executed
    for (0..4) |i| {
        try std.testing.expect(visited[i].load(.acquire));
    }
}

test "for_n static scheduling" {
    std.debug.print("Running test: for_n static scheduling\n", .{});
    const topo = try Topology.init();
    defer topo.deinit();
    var pool = try Pool.init(topo, 4, .inclusive);
    defer pool.deinit();

    var visited = [_]std.atomic.Value(bool){std.atomic.Value(bool).init(false)} ** 100;

    const Context = struct {
        visited_ptr: *[100]std.atomic.Value(bool),
    };

    pool.forN(100, struct {
        fn worker(prong: Prong, ctx: Context) void {
            ctx.visited_ptr[prong.task_index].store(true, .release);
        }
    }.worker, Context{ .visited_ptr = &visited });

    // Verify all tasks executed
    for (0..100) |i| {
        try std.testing.expect(visited[i].load(.acquire));
    }
}

test "for_n_dynamic work stealing" {
    std.debug.print("Running test: for_n_dynamic work stealing\n", .{});
    const topo = try Topology.init();
    defer topo.deinit();
    var pool = try Pool.init(topo, 4, .inclusive);
    defer pool.deinit();

    var counter = std.atomic.Value(usize).init(0);

    const Context = struct {
        counter_ptr: *std.atomic.Value(usize),
    };

    pool.forNDynamic(100, struct {
        fn worker(prong: Prong, ctx: Context) void {
            _ = prong;
            _ = ctx.counter_ptr.fetchAdd(1, .monotonic);
        }
    }.worker, Context{ .counter_ptr = &counter });

    try std.testing.expectEqual(100, counter.load(.acquire));
}

test "for_slices execution" {
    std.debug.print("Running test: for_slices execution\n", .{});
    const topo = try Topology.init();
    defer topo.deinit();
    var pool = try Pool.init(topo, 4, .inclusive);
    defer pool.deinit();

    var data = [_]i32{0} ** 1000;
    var total = std.atomic.Value(usize).init(0);

    const Context = struct {
        data_ptr: *[1000]i32,
        total_ptr: *std.atomic.Value(usize),
    };

    pool.forSlices(1000, struct {
        fn worker(prong: Prong, count: usize, ctx: Context) void {
            var local_sum: usize = 0;
            for (0..count) |i| {
                const idx = prong.task_index + i;
                ctx.data_ptr[idx] = @intCast(idx);
                local_sum += 1;
            }
            _ = ctx.total_ptr.fetchAdd(local_sum, .monotonic);
        }
    }.worker, Context{ .data_ptr = &data, .total_ptr = &total });

    // Verify all elements were processed
    try std.testing.expectEqual(1000, total.load(.acquire));
    for (0..1000) |i| {
        try std.testing.expectEqual(@as(i32, @intCast(i)), data[i]);
    }
}

test "NUMA allocation" {
    std.debug.print("Running test: NUMA allocation\n", .{});
    if (!comptimeCapabilities().place_memory_on_domain) return error.SkipZigTest;

    const topo = try Topology.init();
    defer topo.deinit();
    const allocation = allocateAtLeast(topo.memoryDomainIdAtIndex(0), 1024) orelse return error.SkipZigTest;
    defer allocation.free();

    try std.testing.expect(allocation.allocated_bytes >= 1024);
    try std.testing.expectEqual(topo.memoryDomainIdAtIndex(0), allocation.memory_domain_id);

    // Write to memory to ensure it's usable
    const slice = allocation.asSlice();
    for (0..@min(1024, slice.len)) |i| {
        slice[i] = @intCast(i & 0xFF);
    }
}

test "NUMA allocator integrates with std collections" {
    std.debug.print("Running test: NUMA allocator integrates with std collections\n", .{});
    if (!comptimeCapabilities().place_memory_on_domain) return error.SkipZigTest;

    const topo = try Topology.init();
    defer topo.deinit();
    var domain_alloc = DomainAllocator.init(topo.memoryDomainIdAtIndex(0)) orelse return error.SkipZigTest;
    const allocator = domain_alloc.allocator();

    var list = try std.ArrayList(u64).initCapacity(allocator, 0);
    defer list.deinit(allocator);
    try list.appendSlice(allocator, &[_]u64{ 1, 2, 3, 4, 5 });
    try std.testing.expectEqual(5, list.items.len);
    try std.testing.expectEqual(3, list.items[2]);

    var map = std.AutoHashMap(u32, u32).init(allocator);
    defer map.deinit();
    try map.put(10, 100);
    try map.put(20, 200);
    try map.put(30, 300);
    try std.testing.expectEqual(3, map.count());
    try std.testing.expectEqual(200, map.get(20).?);

    var buf = try allocator.alloc(u8, 128);
    defer allocator.free(buf);
    @memset(buf, 0xAB);

    buf = try allocator.realloc(buf, 512);
    try std.testing.expectEqual(512, buf.len);
    try std.testing.expectEqual(0xAB, buf[0]);
}

test "for_n void context" {
    std.debug.print("Running test: for_n void context\n", .{});
    const topo = try Topology.init();
    defer topo.deinit();
    var pool = try Pool.init(topo, 4, .inclusive);
    defer pool.deinit();

    var counter = std.atomic.Value(usize).init(0);

    // Use a wrapper struct to capture the pointer via comptime closure
    const S = struct {
        var counter_ptr: *std.atomic.Value(usize) = undefined;
        fn worker(prong: Prong) void {
            _ = prong;
            _ = counter_ptr.fetchAdd(1, .monotonic);
        }
    };
    S.counter_ptr = &counter;

    pool.forN(50, S.worker, {});

    try std.testing.expectEqual(50, counter.load(.acquire));
}

test "unsafe_for_threads and join" {
    std.debug.print("Running test: unsafe_for_threads and join\n", .{});
    const topo = try Topology.init();
    defer topo.deinit();
    var pool = try Pool.init(topo, 4, .inclusive);
    defer pool.deinit();

    var counter = std.atomic.Value(usize).init(0);

    const Context = struct {
        counter_ptr: *std.atomic.Value(usize),
    };

    // The context must be a caller-owned pointer: the dispatch returns while
    // worker threads are still reading through it, until `unsafeJoin` completes.
    const context = Context{ .counter_ptr = &counter };
    const generation = pool.unsafeForThreads(struct {
        fn worker(thread_index: usize, compute_domain_index: usize, worker_context: *const Context) void {
            _ = thread_index;
            _ = compute_domain_index;
            _ = worker_context.counter_ptr.fetchAdd(1, .monotonic);
        }
    }.worker, &context);

    // Generation tokens are always odd
    try std.testing.expect(generation & 1 == 1);
    pool.unsafeJoin(generation);

    // After join, isComplete must be true
    try std.testing.expect(pool.isComplete(generation));

    // All 4 threads should have executed, the caller included
    try std.testing.expectEqual(4, counter.load(.acquire));
}

test "generation polling on exclusive pool" {
    std.debug.print("Running test: generation polling on exclusive pool\n", .{});
    const topo = try Topology.init();
    defer topo.deinit();
    var pool = try Pool.init(topo, 4, .exclusive);
    defer pool.deinit();

    var counter = std.atomic.Value(usize).init(0);

    const Context = struct {
        counter_ptr: *std.atomic.Value(usize),
    };

    const context = Context{ .counter_ptr = &counter };
    const generation = pool.unsafeForThreads(struct {
        fn worker(thread_index: usize, compute_domain_index: usize, worker_context: *const Context) void {
            _ = thread_index;
            _ = compute_domain_index;
            _ = worker_context.counter_ptr.fetchAdd(1, .monotonic);
        }
    }.worker, &context);

    // On exclusive pools the caller owes no slice, so polling alone reaches completion
    try std.testing.expect(generation & 1 == 1);
    while (!pool.isComplete(generation)) {
        std.atomic.spinLoopHint();
    }
    pool.unsafeJoin(generation);

    // All 4 worker threads should have executed
    try std.testing.expectEqual(4, counter.load(.acquire));
}

test "ReplicatedArray per-domain buffer" {
    // Each replica is an independent length-`n` buffer, so a domain-dependent fill must read back
    // exactly on its own domain - any aliasing between replicas would corrupt the pattern.
    const topo = try Topology.init();
    defer topo.deinit();

    const n: usize = 4096;
    var replicas = ReplicatedArray(u32).init(topo, n) orelse return error.OutOfMemory;
    defer replicas.deinit();
    try std.testing.expectEqual(n, replicas.len);
    try std.testing.expectEqual(topo.memoryDomainsCount(), replicas.memoryDomainsCount());

    const domains = replicas.memoryDomainsCount();
    for (0..domains) |domain| {
        const replica = replicas.onMemoryDomain(domain);
        try std.testing.expectEqual(n, replica.len);
        for (replica, 0..) |*slot, index| slot.* = @intCast(domain * n + index);
    }
    for (0..domains) |domain| {
        for (0..n) |index| {
            try std.testing.expectEqual(@as(u32, @intCast(domain * n + index)), replicas.at(domain, index).*);
        }
    }
}

test "ShardedArray segment round trip" {
    // The shards tile the logical range exactly once, and `locationOf` is the inverse of
    // `logicalIndexOf` - every element has a single home and the two mappings agree.
    const topo = try Topology.init();
    defer topo.deinit();

    const n: usize = 4096;
    var shards = ShardedArray(u32).init(topo, n) orelse return error.OutOfMemory;
    defer shards.deinit();
    const domains = shards.memoryDomainsCount();
    const segment = shards.segment;
    try std.testing.expectEqual(n, shards.len);

    var footprint: usize = 0;
    for (0..domains) |domain| footprint += shards.lengthOnMemoryDomain(domain);
    try std.testing.expectEqual(n, footprint);

    for (0..domains) |domain| {
        const shard = shards.onMemoryDomain(domain);
        for (shard, 0..) |*slot, local| slot.* = @intCast(domain * segment + local);
    }
    for (0..n) |logical| {
        const location = shards.locationOf(logical);
        try std.testing.expectEqual(logical, shards.logicalIndexOf(location.memory_domain, location.local_index));
        try std.testing.expectEqual(@as(u32, @intCast(logical)), shards.at(location.memory_domain, location.local_index).*);
    }
}
