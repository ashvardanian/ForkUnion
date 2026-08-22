//! Machine topology discovery and library capability introspection.
//!
//! Mirrors the C++ `topology` header: an owned `Topology` handle enumerating this machine's
//! compute and memory domains, the `Capabilities` bitset describing what the library was built
//! for and what the machine offers, and the library version accessors.

const std = @import("std");
const types = @import("types.zig");
const ComputeDomain = types.ComputeDomain;
const MemoryDomain = types.MemoryDomain;
const MemoryDomainId = types.MemoryDomainId;

extern fn fu_version_major() c_int;
extern fn fu_version_minor() c_int;
extern fn fu_version_patch() c_int;
extern fn fu_comptime_capabilities() u32;
extern fn fu_runtime_capabilities() u32;
extern fn fu_name_capabilities(caps: u32, buf: [*]u8, len: usize) usize;

extern fn fu_topology_new() ?*anyopaque;
extern fn fu_topology_delete(topology: *anyopaque) void;

extern fn fu_logical_cores_count(topology: *anyopaque) usize;
extern fn fu_compute_domains_count(topology: *anyopaque) usize;
extern fn fu_compute_levels_count(topology: *anyopaque) usize;
extern fn fu_logical_cores_count_in(topology: *anyopaque, compute_domain_index: usize) usize;
extern fn fu_compute_level_in(topology: *anyopaque, compute_domain_index: usize) usize;
extern fn fu_compute_capacity_in(topology: *anyopaque, compute_domain_index: usize) usize;
extern fn fu_compute_cache_bytes_in(topology: *anyopaque, compute_domain_index: usize) usize;

extern fn fu_memory_domains_count(topology: *anyopaque) usize;
extern fn fu_volume_ram(topology: *anyopaque) usize;
extern fn fu_volume_ram_in(topology: *anyopaque, memory_domain_index: usize) usize;
extern fn fu_volume_huge_pages(topology: *anyopaque) usize;
extern fn fu_volume_huge_pages_in(topology: *anyopaque, memory_domain_index: usize) usize;
extern fn fu_huge_pages_count(topology: *anyopaque) usize;
extern fn fu_huge_pages_count_in(topology: *anyopaque, memory_domain_index: usize) usize;
extern fn fu_local_memory_of(topology: *anyopaque, compute_domain_index: usize) usize;
extern fn fu_memory_domain_id_at_index(topology: *anyopaque, memory_domain_index: usize) i32;

/// Errors that can occur during thread pool operations
pub const Error = error{
    /// A handle could not be created. The C API reports no reason, so this covers an allocation
    /// failure and a platform that refused the request alike; splitting it would be a guess.
    CreationFailed,
    /// The pool could not start the requested number of threads.
    SpawnFailed,
};

/// Defines whether the calling thread participates in task execution
pub const CallerExclusivity = enum(c_int) {
    /// Calling thread participates in workload (spawns N-1 workers)
    inclusive = 0,
    /// Calling thread only coordinates (spawns N workers)
    exclusive = 1,
};

/// Returns the library version as a struct
pub fn version() struct { major: u32, minor: u32, patch: u32 } {
    return .{
        .major = @intCast(fu_version_major()),
        .minor = @intCast(fu_version_minor()),
        .patch = @intCast(fu_version_patch()),
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

    /// `CLDEMOTE` moves a just-written line toward the shared LLC and retains it. Reporting-only:
    /// the emitter is chosen at compile time by `FU_WITH_DEMOTE_CACHE_LINES`, never dispatched.
    x86_cldemote: bool = false,
    /// `DC CVAC` cleans a dirty line to the coherency point - AArch64's nearest demote. Set where
    /// EL0 execution is known-legal, i.e. Linux, which sets `SCTLR_EL1.UCI`.
    arm64_dc_cvac: bool = false,
    /// The kernel enabled user-mode Zicbom cache-block management, attested through `hwprobe` -
    /// the hook for a future runtime-dispatched `cbo.clean`; nothing emits it yet.
    risc5_zicbom: bool = false,

    _unused: u14 = 0,

    /// All-ones allow-mask: pass to a pool constructor to disable capability filtering.
    pub fn all() Capabilities {
        return @bitCast(@as(u32, 0xFFFF_FFFF));
    }
};

/// Which kernel facilities this build of ForkUnion was compiled to use.
pub fn comptimeCapabilities() Capabilities {
    return @bitCast(fu_comptime_capabilities());
}

/// The set `comptimeCapabilities` bits, comma-separated, like `"threads,topology"`.
///
/// POLISH: writes into a caller-provided buffer so the returned slice borrows `buf`; the
/// buffer must outlive the slice. Callers pass their own stack buffer.
pub fn comptimeCapabilitiesString(buf: []u8) []const u8 {
    const written = fu_name_capabilities(fu_comptime_capabilities(), buf.ptr, buf.len);
    return buf[0..written];
}

/// Which features this machine turned out to offer, probing the CPU and the memory system.
pub fn runtimeCapabilities() Capabilities {
    return @bitCast(fu_runtime_capabilities());
}

/// The set `runtimeCapabilities` bits, comma-separated, like `"arm64_yield,place_memory_on_domain"`.
///
/// POLISH: writes into a caller-provided buffer so the returned slice borrows `buf`; the
/// buffer must outlive the slice. Callers pass their own stack buffer.
pub fn runtimeCapabilitiesString(buf: []u8) []const u8 {
    const written = fu_name_capabilities(fu_runtime_capabilities(), buf.ptr, buf.len);
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
        const h = fu_topology_new() orelse return Error.CreationFailed;
        return .{ .handle = h };
    }

    /// Releases the topology handle.
    pub fn deinit(self: Topology) void {
        fu_topology_delete(self.handle);
    }

    /// Returns the number of logical CPU cores available
    pub fn countLogicalCores(self: Topology) usize {
        return fu_logical_cores_count(self.handle);
    }

    /// Returns the number of memory domains available
    pub fn countMemoryDomains(self: Topology) usize {
        return fu_memory_domains_count(self.handle);
    }

    /// Resolves a memory domain's dense index to the OS id the allocators take; `.none` if out of range.
    pub fn memoryDomainIdAtIndex(self: Topology, memory_domain: MemoryDomain) MemoryDomainId {
        return @enumFromInt(fu_memory_domain_id_at_index(self.handle, memory_domain.index()));
    }

    /// Returns the number of distinct thread compute_domains
    pub fn countComputeDomains(self: Topology) usize {
        return fu_compute_domains_count(self.handle);
    }

    /// Returns the number of logical cores backing a given compute domain (0 if out of range).
    pub fn countLogicalCoresIn(self: Topology, compute_domain: ComputeDomain) usize {
        return fu_logical_cores_count_in(self.handle, compute_domain.index());
    }

    /// Returns the performance level of a compute domain (higher = more performant).
    pub fn computeLevelIn(self: Topology, compute_domain: ComputeDomain) usize {
        return fu_compute_level_in(self.handle, compute_domain.index());
    }

    /// Returns the memory domain nearest a given compute domain (its local allocation target).
    ///
    /// Performance - tiers, latencies, bandwidths, distances - is not the topology's to declare:
    /// harvest a `Fabric` to measure it in-process.
    pub fn localMemoryOf(self: Topology, compute_domain: ComputeDomain) MemoryDomain {
        return @enumFromInt(fu_local_memory_of(self.handle, compute_domain.index()));
    }

    /// Returns the number of distinct Quality-of-Service levels.
    ///
    /// May be smaller than `countComputeDomains`, as several domains can share one level - equally-fast
    /// cores may still be split across cache clusters, or across NUMA nodes.
    pub fn countComputeLevels(self: Topology) usize {
        return fu_compute_levels_count(self.handle);
    }

    /// Returns the relative throughput of one core in a compute domain (0 if unknown).
    ///
    /// A magnitude on the Linux `cpu_capacity` scale, where 1024 is the fastest core present. Weight
    /// work by this - `computeLevelIn` is a dense ordinal and must never be divided by. Platforms that
    /// rank cores without rating them report 0; weigh by core count instead.
    pub fn computeCapacityIn(self: Topology, compute_domain: ComputeDomain) usize {
        return fu_compute_capacity_in(self.handle, compute_domain.index());
    }

    /// Returns the bytes of deepest cache private to a compute domain's cores (0 if unknown).
    ///
    /// Sizes a cache-resident chunk, a different question from how many chunks a domain deserves -
    /// domains of equal throughput may back onto very differently sized caches.
    pub fn computeCacheBytesIn(self: Topology, compute_domain: ComputeDomain) usize {
        return fu_compute_cache_bytes_in(self.handle, compute_domain.index());
    }

    /// Returns the total RAM volume (bytes) across all memory domains, regardless of page size.
    pub fn volumeRam(self: Topology) usize {
        return fu_volume_ram(self.handle);
    }

    /// Returns the RAM volume (bytes) held by a given memory domain (0 if out of range).
    pub fn volumeRamIn(self: Topology, memory_domain: MemoryDomain) usize {
        return fu_volume_ram_in(self.handle, memory_domain.index());
    }

    /// Returns the total huge-page volume (bytes) across all memory domains.
    pub fn volumeHugePages(self: Topology) usize {
        return fu_volume_huge_pages(self.handle);
    }

    /// Returns the huge-page volume (bytes) available in a given memory domain (0 if out of range).
    pub fn volumeHugePagesIn(self: Topology, memory_domain: MemoryDomain) usize {
        return fu_volume_huge_pages_in(self.handle, memory_domain.index());
    }

    /// Returns the total number of free huge pages across all memory domains.
    pub fn countHugePages(self: Topology) usize {
        return fu_huge_pages_count(self.handle);
    }

    /// Returns the number of free huge pages in a given memory domain (0 if out of range).
    pub fn countHugePagesIn(self: Topology, memory_domain: MemoryDomain) usize {
        return fu_huge_pages_count_in(self.handle, memory_domain.index());
    }
};

test "version info" {
    // The components are unsigned, so `>= 0` asserts nothing. An all-zero triple is the failure
    // worth catching - the signature of a stubbed or mislinked core - since the numbers themselves
    // live in the build manifest and pinning them here would only duplicate it.
    const v = version();
    try std.testing.expect(v.major + v.minor + v.patch > 0);
}

test "capability bits match the C ABI numbering" {
    // The packed struct's field order is the wire format, so a field inserted in the wrong place
    // silently remaps every bit above it - and a field left out drops the core's answer into
    // padding, unseen. Pin the boundaries of each group against the C header's `1 << n`.
    const bit = struct {
        fn at(position: u5) Capabilities {
            return @bitCast(@as(u32, 1) << position);
        }
    }.at;
    try std.testing.expect(bit(0).x86_pause);
    try std.testing.expect(bit(5).risc5_wrs);
    try std.testing.expect(bit(6).os_threads);
    try std.testing.expect(bit(14).colocate_pools_on_domain);
    try std.testing.expect(bit(15).x86_cldemote);
    try std.testing.expect(bit(16).arm64_dc_cvac);
    try std.testing.expect(bit(17).risc5_zicbom);

    // Every bit the core defines must have a field; none may fall through to `_unused`.
    const all_defined: Capabilities = @bitCast(@as(u32, (1 << 18) - 1));
    try std.testing.expectEqual(0, all_defined._unused);
}

test "system capabilities" {
    const comptime_caps = comptimeCapabilities();
    const runtime_caps = runtimeCapabilities();
    var runtime_buf: [256]u8 = undefined;
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

    // A machine can only _offer_ page placement if this build compiled the path that asks for it.
    if (runtime_caps.place_memory_on_domain) try std.testing.expect(comptime_caps.place_memory_on_domain);
}

test "system metadata" {
    const topo = try Topology.init();
    defer topo.deinit();

    const cores = topo.countLogicalCores();
    try std.testing.expect(cores > 0);

    const numa = topo.countMemoryDomains();
    try std.testing.expect(numa > 0);

    const colocs = topo.countComputeDomains();
    try std.testing.expect(colocs > 0);
}
