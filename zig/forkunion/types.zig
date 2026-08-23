//! Value types mirroring the C++ `types` header, and the `Status`/`Error` vocabulary every
//! other module reports through.
//!
//! Holds the `ComputeDomain`/`MemoryDomain`/`MemoryDomainId` machine coordinates, the `TasksRange`
//! work descriptor and its `ThreadInDomain` locator, and the small parity primitives the parallel
//! layer is built from:
//! the `IndexedSplit` fair-run splitter, the `CacheAligned` padding wrapper, and the
//! `SyncConstPtr`/`SyncMutPtr` raw-pointer views that let disjoint slices cross the FFI boundary.

const std = @import("std");

extern fn fu_status_to_string(status: c_int) [*:0]const u8;

/// Why a call into the C core failed, mirroring `fu_status_t` value-for-value.
///
/// Non-exhaustive, so a status a newer core reports arrives intact rather than coerced.
pub const Status = enum(c_int) {
    success = 0,
    unknown = -1,
    bad_alloc = -2,
    capacity_exhausted = -3,
    invalid_argument = -4,
    config_mismatch = -5,
    already_spawned = -6,
    not_spawned = -7,
    thread_refused = -8,
    topology_unavailable = -9,
    permission_denied = -10,
    unsupported = -11,
    _,

    /// Static, English description; mirrors `fu_status_to_string`.
    pub fn describe(self: Status) [:0]const u8 {
        return std.mem.span(fu_status_to_string(@intFromEnum(self)));
    }
};

/// One error per status, so `@errorName` carries as much as the status itself does.
///
/// Zig error values hold no payload, so the symbol name that named which call failed cannot
/// travel with the error. Reach for `Status` where that detail matters.
pub const Error = error{
    Unknown,
    BadAlloc,
    CapacityExhausted,
    InvalidArgument,
    ConfigMismatch,
    AlreadySpawned,
    NotSpawned,
    ThreadRefused,
    TopologyUnavailable,
    PermissionDenied,
    Unsupported,
};

/// Lifts a raw `fu_status_t` into the error set; `success` is the only non-error.
pub fn check(raw: c_int) Error!void {
    return switch (@as(Status, @enumFromInt(raw))) {
        .success => {},
        .bad_alloc => Error.BadAlloc,
        .capacity_exhausted => Error.CapacityExhausted,
        .invalid_argument => Error.InvalidArgument,
        .config_mismatch => Error.ConfigMismatch,
        .already_spawned => Error.AlreadySpawned,
        .not_spawned => Error.NotSpawned,
        .thread_refused => Error.ThreadRefused,
        .topology_unavailable => Error.TopologyUnavailable,
        .permission_denied => Error.PermissionDenied,
        .unsupported => Error.Unsupported,
        else => Error.Unknown,
    };
}

/// The width the padding wrapper aligns to: two cache lines, because most x86 parts prefetch in
/// pairs, so one line of separation still leaves two accumulators sharing a prefetch unit.
///
/// The core spells this `default_alignment_k` and deliberately does not use
/// `std::hardware_destructive_interference_size`, which trips GCC ABI warnings and picks worse.
pub const default_alignment = 128;

/// A dense compute-domain index, in `0..Topology.computeDomainsCount()`.
///
/// A distinct type from `MemoryDomain` because the two index different axes of the machine and
/// the compiler is the only thing that can tell them apart at a call site.
pub const ComputeDomain = enum(usize) {
    _,

    /// Wraps a dense index, as produced by iterating `0..computeDomainsCount()`.
    pub fn at(dense_index: usize) ComputeDomain {
        return @enumFromInt(dense_index);
    }

    /// The dense index, for arithmetic and for the FFI boundary.
    pub fn index(self: ComputeDomain) usize {
        return @intFromEnum(self);
    }
};

/// A dense memory-domain index, in `0..Topology.memoryDomainsCount()`.
pub const MemoryDomain = enum(usize) {
    _,

    /// Wraps a dense index, as produced by iterating `0..memoryDomainsCount()`.
    pub fn at(dense_index: usize) MemoryDomain {
        return @enumFromInt(dense_index);
    }

    /// The dense index, for arithmetic and for the FFI boundary.
    pub fn index(self: MemoryDomain) usize {
        return @intFromEnum(self);
    }
};

/// The sparse OS id the kernel labels a NUMA node with, which the allocators key off.
///
/// Distinct from `MemoryDomain`: that one iterates, this one allocates. An `AllocationResult`
/// carries only this id, so it can outlive the `Topology` that resolved it.
pub const MemoryDomainId = enum(i32) {
    /// Names no domain - what an out-of-range lookup answers.
    none = -1,
    _,

    /// The raw OS id, for the FFI boundary.
    pub fn identifier(self: MemoryDomainId) i32 {
        return @intFromEnum(self);
    }

    /// Whether this id names a real domain rather than the `none` sentinel.
    pub fn isValid(self: MemoryDomainId) bool {
        return @intFromEnum(self) >= 0;
    }
};

/// One thread, situated in one compute domain - the "where I am" every callback receives.
///
/// Every dispatch hands its callback two things: the work, and this. A callback placing memory
/// reads `compute_domain` to find the node it runs on; one indexing per-thread scratch reads
/// `thread`.
pub const ThreadInDomain = struct {
    /// The physical thread executing the work.
    thread: usize,
    /// The compute domain the thread is pinned to - a same-QoS core cluster.
    compute_domain: ComputeDomain,
};

/// A half-open `[start, start + len)` slice of a task range, as handed out by `IndexedSplit.get`.
/// A half-open `[first, first + count)` run of task indices - the "what work" of a slice dispatch.
///
/// Zig has no first-class range value, so `first` and `count` feed the language's own `for`
/// syntax: `for (range.first..range.end()) |task|`. An idle thread receives `count == 0`, and
/// every dispatch still calls it exactly once.
pub const TasksRange = struct {
    /// The first task index in the run.
    first: usize,
    /// How many tasks the run covers; zero means an idle thread.
    count: usize,

    /// One past the last task index, so `first..end()` is the half-open span.
    pub fn end(self: TasksRange) usize {
        return self.first + self.count;
    }

    /// Whether the run covers no tasks at all.
    pub fn isEmpty(self: TasksRange) bool {
        return self.count == 0;
    }

    /// The run's slice of `data`, which must be at least `end()` long.
    pub fn of(self: TasksRange, comptime T: type, data: []T) []T {
        return data[self.first..][0..self.count];
    }
};

/// Splits a range of tasks into fair-sized runs for parallel distribution.
///
/// The first `tasks % threads` runs get `ceil(tasks / threads)` tasks; the rest get
/// `floor(tasks / threads)`. This minimizes size variance across threads. Mirrors the C++
/// `indexed_split` and Lemire's fair-run scheme.
/// See: https://lemire.me/blog/2025/05/22/dividing-an-array-into-fair-sized-chunks/
pub const IndexedSplit = struct {
    quotient: usize,
    remainder: usize,

    /// Builds a split of `tasks_count` tasks across `threads_count` threads; `threads_count` can't be zero.
    pub fn init(tasks_count: usize, threads_count: usize) IndexedSplit {
        std.debug.assert(threads_count > 0);
        return .{
            .quotient = tasks_count / threads_count,
            .remainder = tasks_count % threads_count,
        };
    }

    /// Returns the `{ first, count }` run owned by thread `thread_index`.
    pub fn get(self: IndexedSplit, thread_index: usize) TasksRange {
        const first = self.quotient * thread_index + @min(thread_index, self.remainder);
        const count = self.quotient + @as(usize, if (thread_index < self.remainder) 1 else 0);
        return .{ .first = first, .count = count };
    }
};

/// Wraps a value in cache-line-aligned, cache-line-padded storage so per-thread accumulators
/// never share a cache line and thus never false-share.
///
/// Mirrors the C++ `cache_aligned` and the Rust `CacheAligned<T>`: allocate one per thread as
/// scratch, then combine after the parallel region.
pub fn CacheAligned(comptime T: type) type {
    return struct {
        value: T align(default_alignment),
    };
}

/// A `Send + Sync`-style read-only raw-pointer view, letting one immutable buffer be read by
/// every worker across the FFI callback boundary.
///
/// The caller guarantees the pointee outlives the parallel region and is not mutated while shared.
pub fn SyncConstPtr(comptime T: type) type {
    return struct {
        const Self = @This();

        ptr: [*]const T,

        /// Wraps a raw pointer to the start of the shared buffer.
        pub fn init(ptr: [*]const T) Self {
            return .{ .ptr = ptr };
        }

        /// A reference to the element at `index`; the caller owns bounds checking.
        pub fn get(self: Self, index: usize) *const T {
            return &self.ptr[index];
        }

        /// The underlying raw pointer.
        pub fn asPtr(self: Self) [*]const T {
            return self.ptr;
        }
    };
}

/// A `Send + Sync`-style mutable raw-pointer view, letting workers write disjoint slots of one
/// buffer across the FFI callback boundary.
///
/// The caller guarantees each worker touches disjoint indices and the pointee outlives the region.
pub fn SyncMutPtr(comptime T: type) type {
    return struct {
        const Self = @This();

        ptr: [*]T,

        /// Wraps a raw pointer to the start of the shared buffer.
        pub fn init(ptr: [*]T) Self {
            return .{ .ptr = ptr };
        }

        /// A mutable pointer to the element at `index`; the caller owns disjointness and bounds.
        pub fn get(self: Self, index: usize) *T {
            return &self.ptr[index];
        }

        /// The underlying raw pointer.
        pub fn asPtr(self: Self) [*]T {
            return self.ptr;
        }
    };
}

test "IndexedSplit fair runs tile the range" {
    // The runs must cover [0, tasks) exactly once, be contiguous, and differ in size by at most one.
    const cases = [_]struct { tasks: usize, threads: usize }{
        .{ .tasks = 0, .threads = 4 },
        .{ .tasks = 1, .threads = 4 },
        .{ .tasks = 10, .threads = 4 },
        .{ .tasks = 12, .threads = 4 },
        .{ .tasks = 1000, .threads = 7 },
        .{ .tasks = 5, .threads = 1 },
    };
    for (cases) |case| {
        const split = IndexedSplit.init(case.tasks, case.threads);
        var expected_start: usize = 0;
        var min_len: usize = std.math.maxInt(usize);
        var max_len: usize = 0;
        for (0..case.threads) |thread| {
            const range = split.get(thread);
            try std.testing.expectEqual(expected_start, range.first);
            expected_start += range.count;
            min_len = @min(min_len, range.count);
            max_len = @max(max_len, range.count);
        }
        try std.testing.expectEqual(case.tasks, expected_start);
        try std.testing.expect(max_len - min_len <= 1);
    }
}

test "CacheAligned pads rather than merely aligns" {
    // `align(default_alignment)` is the wrapper's only source of alignment, so asserting the two
    // agree proves nothing. These are the properties that do not follow from the definition.
    try std.testing.expect(std.math.isPowerOfTwo(default_alignment));
    try std.testing.expect(default_alignment >= 64);

    // A one-byte payload still occupies the whole span - alignment without padding would let two
    // accumulators share one span and false-share anyway.
    try std.testing.expectEqual(default_alignment, @sizeOf(CacheAligned(u8)));
    try std.testing.expectEqual(default_alignment, @alignOf(CacheAligned(u8)));

    var slots = [_]CacheAligned(usize){.{ .value = 0 }} ** 4;
    for (&slots, 0..) |*slot, i| slot.value = i * 7;
    for (&slots, 0..) |*slot, i| try std.testing.expectEqual(i * 7, slot.value);

    // Neighbouring accumulators never land in one span, whatever the span turns out to be.
    const gap = @intFromPtr(&slots[1]) - @intFromPtr(&slots[0]);
    try std.testing.expect(gap >= default_alignment);
}

test "SyncConstPtr and SyncMutPtr index a shared buffer" {
    var data = [_]u32{ 10, 20, 30, 40, 50 };

    const reader = SyncConstPtr(u32).init(&data);
    try std.testing.expectEqual(@as(u32, 10), reader.get(0).*);
    try std.testing.expectEqual(@as(u32, 50), reader.get(4).*);
    try std.testing.expectEqual(@as([*]const u32, &data), reader.asPtr());

    const writer = SyncMutPtr(u32).init(&data);
    for (0..data.len) |i| writer.get(i).* = @intCast(i * i);
    for (0..data.len) |i| try std.testing.expectEqual(@as(u32, @intCast(i * i)), data[i]);
    try std.testing.expectEqual(@as([*]u32, &data), writer.asPtr());
}

test "domain coordinates round-trip" {
    // The dense axes wrap and unwrap without loss. That the two are distinct types needs no
    // assertion - mixing them is a compile error, which no runtime check could observe.
    try std.testing.expectEqual(@as(usize, 3), ComputeDomain.at(3).index());
    try std.testing.expectEqual(@as(usize, 0), MemoryDomain.at(0).index());
    try std.testing.expectEqual(@as(usize, std.math.maxInt(usize)), ComputeDomain.at(std.math.maxInt(usize)).index());

    // The OS id names its absence rather than leaking a bare -1.
    try std.testing.expect(!MemoryDomainId.none.isValid());
    try std.testing.expectEqual(@as(i32, -1), MemoryDomainId.none.identifier());
    const first: MemoryDomainId = @enumFromInt(0);
    try std.testing.expect(first.isValid());
    try std.testing.expectEqual(@as(i32, 0), first.identifier());
}
