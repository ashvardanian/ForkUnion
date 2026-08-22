//! Memory-domain-aware allocation, mirroring the C++ `allocators` and `distributed` headers.
//!
//! An `AllocationResult`/`DomainAllocator` pin buffers to a single memory domain, while
//! `ReplicatedArray` and `ShardedArray` lay a sequence out across every domain - one full replica
//! per domain, or one contiguous shard per domain.

const std = @import("std");
const Topology = @import("topology.zig").Topology;
const types = @import("types.zig");
const MemoryDomain = types.MemoryDomain;
const MemoryDomainId = types.MemoryDomainId;

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

/// Result of a memory-domain allocation; carries only the OS id, so it can outlive the topology.
pub const AllocationResult = struct {
    memory_domain_id: MemoryDomainId,
    ptr: [*]u8,
    allocated_bytes: usize,
    bytes_per_page: usize,

    /// Returns the allocated memory as a slice
    pub fn asSlice(self: AllocationResult) []u8 {
        return self.ptr[0..self.allocated_bytes];
    }

    /// Frees the allocation
    pub fn free(self: AllocationResult) void {
        fu_free_on_domain_id(self.memory_domain_id.identifier(), @ptrCast(self.ptr), self.allocated_bytes);
    }
};

/// Allocates memory on a memory domain with optimal page size; id from `Topology.memoryDomainIdAtIndex`.
pub fn allocateAtLeast(memory_domain_id: MemoryDomainId, minimum_bytes: usize) ?AllocationResult {
    var allocated_bytes: usize = undefined;
    var bytes_per_page: usize = undefined;

    const ptr = fu_allocate_at_least_on_domain_id(
        memory_domain_id.identifier(),
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
pub fn allocate(memory_domain_id: MemoryDomainId, bytes: usize) ?[*]u8 {
    const ptr = fu_allocate_on_domain_id(memory_domain_id.identifier(), bytes) orelse return null;
    return @ptrCast(@alignCast(ptr));
}

/// Allocator bound to a single memory domain, compatible with Zig's `std.mem.Allocator` interface.
///
/// This type is the allocation API: build one for a memory domain's OS id - from
/// `Topology.memoryDomainIdAtIndex` - and it both hands out `AllocationResult`s through
/// `allocateAtLeast`/`allocate` and backs a `std.mem.Allocator` through `allocator`.
pub const DomainAllocator = struct {
    memory_domain_id: MemoryDomainId,

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

    /// Binds to the memory domain named by @p memory_domain_id, which must name a real domain -
    /// ask `isValid` first, since an out-of-range lookup answers `.none`.
    pub fn init(memory_domain_id: MemoryDomainId) Self {
        std.debug.assert(memory_domain_id.isValid());
        return .{ .memory_domain_id = memory_domain_id };
    }

    /// Allocates at least @p minimum_bytes on this domain with the optimal page size, or null on failure.
    pub fn allocateAtLeast(self: Self, minimum_bytes: usize) ?AllocationResult {
        var allocated_bytes: usize = undefined;
        var bytes_per_page: usize = undefined;
        const ptr = fu_allocate_at_least_on_domain_id(
            self.memory_domain_id.identifier(),
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
        const ptr = fu_allocate_on_domain_id(self.memory_domain_id.identifier(), bytes) orelse return null;
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
        const raw_ptr = fu_allocate_at_least_on_domain_id(
            self.memory_domain_id.identifier(),
            request_bytes,
            &allocated_bytes,
            &bytes_per_page,
        ) orelse return null;

        const base_addr = @intFromPtr(raw_ptr);
        const data_addr = alignment.forward(base_addr + header_size);
        if (data_addr + len > base_addr + allocated_bytes) {
            fu_free_on_domain_id(self.memory_domain_id.identifier(), raw_ptr, allocated_bytes);
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
        fu_free_on_domain_id(self.memory_domain_id.identifier(), base_ptr, header.allocated_bytes);
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

        /// Allocates one uninitialized length-`n` replica per memory domain.
        pub fn init(topology: Topology, n: usize) std.mem.Allocator.Error!Self {
            if (n == 0) return Self{};
            var stride_bytes: usize = 0;
            var domains: usize = 0;
            var total_bytes: usize = 0;
            var bytes_per_page: usize = 0;
            const base_bytes = fu_allocate_symmetric(
                topology.handle,
                n * @sizeOf(T),
                &stride_bytes,
                &domains,
                &total_bytes,
                &bytes_per_page,
            ) orelse return error.OutOfMemory;
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
            if (self.base_bytes) |base_bytes| fu_free_symmetric(@ptrCast(base_bytes), self.total_bytes);
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
        pub fn countMemoryDomains(self: Self) usize {
            return self.domains;
        }

        /// The page-aligned byte distance between consecutive replicas.
        pub fn strideBytes(self: Self) usize {
            return self.stride_bytes;
        }

        /// The whole replica living on `memory_domain`.
        pub fn onMemoryDomain(self: Self, memory_domain: MemoryDomain) []T {
            const slice_base = self.base_bytes.? + memory_domain.index() * self.stride_bytes;
            const typed: [*]T = @ptrCast(@alignCast(slice_base));
            return typed[0..self.len];
        }

        /// One element of the replica on `memory_domain`.
        pub fn at(self: Self, memory_domain: MemoryDomain, local_index: usize) *T {
            return &self.onMemoryDomain(memory_domain)[local_index];
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
        pub fn init(topology: Topology, n: usize) std.mem.Allocator.Error!Self {
            if (n == 0) return Self{};
            const domains = topology.countMemoryDomains();
            if (domains == 0) return error.OutOfMemory;
            const segment = std.math.divCeil(usize, n, domains) catch unreachable;
            var stride_bytes: usize = 0;
            var domains_out: usize = 0;
            var total_bytes: usize = 0;
            var bytes_per_page: usize = 0;
            const base_bytes = fu_allocate_symmetric(
                topology.handle,
                segment * @sizeOf(T),
                &stride_bytes,
                &domains_out,
                &total_bytes,
                &bytes_per_page,
            ) orelse return error.OutOfMemory;
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
            if (self.base_bytes) |base_bytes| fu_free_symmetric(@ptrCast(base_bytes), self.total_bytes);
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
        pub fn countMemoryDomains(self: Self) usize {
            return self.domains;
        }

        /// The page-aligned byte distance between consecutive shards.
        pub fn strideBytes(self: Self) usize {
            return self.stride_bytes;
        }

        /// How many elements the shard on `memory_domain` holds - a trailing shard may be shorter.
        pub fn lengthOnMemoryDomain(self: Self, memory_domain: MemoryDomain) usize {
            const start = memory_domain.index() * self.segment;
            if (start >= self.len) return 0;
            return @min(self.len - start, self.segment);
        }

        /// The memory domain and local index that store logical element `logical_index`.
        pub fn locationOf(self: Self, logical_index: usize) struct { memory_domain: MemoryDomain, local_index: usize } {
            return .{
                .memory_domain = MemoryDomain.at(logical_index / self.segment),
                .local_index = logical_index % self.segment,
            };
        }

        /// The logical index of the element at `local_index` on `memory_domain` - inverse of `locationOf`.
        pub fn logicalIndexOf(self: Self, memory_domain: MemoryDomain, local_index: usize) usize {
            return memory_domain.index() * self.segment + local_index;
        }

        /// The whole shard living on `memory_domain`.
        pub fn onMemoryDomain(self: Self, memory_domain: MemoryDomain) []T {
            const slice_base = self.base_bytes.? + memory_domain.index() * self.stride_bytes;
            const typed: [*]T = @ptrCast(@alignCast(slice_base));
            return typed[0..self.lengthOnMemoryDomain(memory_domain)];
        }

        /// The single home of a logical element.
        pub fn at(self: Self, memory_domain: MemoryDomain, local_index: usize) *T {
            return &self.onMemoryDomain(memory_domain)[local_index];
        }
    };
}

test "NUMA allocation" {
    if (!@import("topology.zig").comptimeCapabilities().place_memory_on_domain) return error.SkipZigTest;

    const topo = try Topology.init();
    defer topo.deinit();
    const first_domain = MemoryDomain.at(0);
    // The capability guard above already said this machine places pages on a domain, so a 1 KiB
    // request failing is a defect - skipping here would bury it.
    const allocation = allocateAtLeast(topo.memoryDomainIdAtIndex(first_domain), 1024) orelse return error.OutOfMemory;
    defer allocation.free();

    try std.testing.expect(allocation.allocated_bytes >= 1024);
    try std.testing.expectEqual(topo.memoryDomainIdAtIndex(first_domain), allocation.memory_domain_id);

    // Write to memory to ensure it's usable
    const slice = allocation.asSlice();
    for (0..@min(1024, slice.len)) |i| {
        slice[i] = @intCast(i & 0xFF);
    }
}

test "NUMA allocator integrates with std collections" {
    if (!@import("topology.zig").comptimeCapabilities().place_memory_on_domain) return error.SkipZigTest;

    const topo = try Topology.init();
    defer topo.deinit();
    const memory_domain_id = topo.memoryDomainIdAtIndex(MemoryDomain.at(0));
    if (!memory_domain_id.isValid()) return error.SkipZigTest;
    var domain_alloc = DomainAllocator.init(memory_domain_id);
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

test "ReplicatedArray per-domain buffer" {
    // Each replica is an independent length-`n` buffer, so a domain-dependent fill must read back
    // exactly on its own domain - any aliasing between replicas would corrupt the pattern.
    const topo = try Topology.init();
    defer topo.deinit();

    const n: usize = 4096;
    var replicas = try ReplicatedArray(u32).init(topo, n);
    defer replicas.deinit();
    try std.testing.expectEqual(n, replicas.len);
    try std.testing.expectEqual(topo.countMemoryDomains(), replicas.countMemoryDomains());

    const domains = replicas.countMemoryDomains();
    for (0..domains) |domain| {
        const replica = replicas.onMemoryDomain(MemoryDomain.at(domain));
        try std.testing.expectEqual(n, replica.len);
        for (replica, 0..) |*slot, index| slot.* = @intCast(domain * n + index);
    }
    for (0..domains) |domain| {
        for (0..n) |index| {
            const slot = replicas.at(MemoryDomain.at(domain), index);
            try std.testing.expectEqual(@as(u32, @intCast(domain * n + index)), slot.*);
        }
    }
}

test "ShardedArray segment round trip" {
    // The shards tile the logical range exactly once, and `locationOf` is the inverse of
    // `logicalIndexOf` - every element has a single home and the two mappings agree.
    const topo = try Topology.init();
    defer topo.deinit();

    const n: usize = 4096;
    var shards = try ShardedArray(u32).init(topo, n);
    defer shards.deinit();
    const domains = shards.countMemoryDomains();
    const segment = shards.segment;
    try std.testing.expectEqual(n, shards.len);

    var footprint: usize = 0;
    for (0..domains) |domain| footprint += shards.lengthOnMemoryDomain(MemoryDomain.at(domain));
    try std.testing.expectEqual(n, footprint);

    for (0..domains) |domain| {
        const shard = shards.onMemoryDomain(MemoryDomain.at(domain));
        for (shard, 0..) |*slot, local| slot.* = @intCast(domain * segment + local);
    }
    for (0..n) |logical| {
        const location = shards.locationOf(logical);
        try std.testing.expectEqual(logical, shards.logicalIndexOf(location.memory_domain, location.local_index));
        try std.testing.expectEqual(@as(u32, @intCast(logical)), shards.at(location.memory_domain, location.local_index).*);
    }
}
