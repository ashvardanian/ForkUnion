//! Pure-logic value types mirroring the C++ `types` header - no FFI.
//!
//! Holds the `Prong` execution-context descriptor plus the small parity primitives the parallel
//! layer is built from: the `IndexedSplit` fair-chunk splitter, the `CacheAligned` padding wrapper,
//! and the `SyncConstPtr`/`SyncMutPtr` raw-pointer views that let disjoint slices cross the FFI
//! callback boundary.

const std = @import("std");

/// The cache-line width the padding wrapper aligns to, matching the C++ core's default alignment.
pub const default_alignment = 64;

/// A "prong" - metadata about a task's execution context
pub const Prong = struct {
    /// The logical index of the task being processed
    task_index: usize,
    /// The physical thread executing this task
    thread_index: usize,
    /// The compute domain (a same-QoS core cluster)
    compute_domain_index: usize,
};

/// A half-open `[start, start + len)` slice of a task range, as handed out by `IndexedSplit.get`.
pub const IndexedRange = struct {
    /// The first task index in the chunk.
    start: usize,
    /// The number of tasks in the chunk.
    len: usize,
};

/// Splits a range of tasks into fair-sized chunks for parallel distribution.
///
/// The first `tasks % threads` chunks get `ceil(tasks / threads)` tasks; the rest get
/// `floor(tasks / threads)`. This minimizes size variance across threads. Mirrors the C++
/// `indexed_split` and Lemire's fair-chunk scheme.
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

    /// Returns the `{ start, len }` chunk owned by thread `thread_index`.
    pub fn get(self: IndexedSplit, thread_index: usize) IndexedRange {
        const start = self.quotient * thread_index + @min(thread_index, self.remainder);
        const len = self.quotient + @as(usize, if (thread_index < self.remainder) 1 else 0);
        return .{ .start = start, .len = len };
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

test "IndexedSplit fair chunks tile the range" {
    // The chunks must cover [0, tasks) exactly once, be contiguous, and differ in size by at most one.
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
            try std.testing.expectEqual(expected_start, range.start);
            expected_start += range.len;
            min_len = @min(min_len, range.len);
            max_len = @max(max_len, range.len);
        }
        try std.testing.expectEqual(case.tasks, expected_start);
        try std.testing.expect(max_len - min_len <= 1);
    }
}

test "CacheAligned pads to a cache line" {
    try std.testing.expectEqual(default_alignment, @alignOf(CacheAligned(u8)));
    try std.testing.expectEqual(default_alignment, @alignOf(CacheAligned(u64)));

    var slots = [_]CacheAligned(usize){.{ .value = 0 }} ** 4;
    for (&slots, 0..) |*slot, i| slot.value = i * 7;
    for (&slots, 0..) |*slot, i| try std.testing.expectEqual(i * 7, slot.value);

    // Distinct accumulators land on distinct cache lines.
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
