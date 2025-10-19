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
//! var pool = try fu.Pool.init(4, .inclusive);
//! defer pool.deinit();
//!
//! // Execute work on each thread (like OpenMP parallel)
//! pool.forThreads(struct {
//!     fn work(thread_idx: usize, colocation_idx: usize) void {
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
    extern fn fu_version_major() c_int;
    extern fn fu_version_minor() c_int;
    extern fn fu_version_patch() c_int;
    extern fn fu_enabled_numa() c_int;
    extern fn fu_capabilities_string() [*:0]const u8;

    extern fn fu_count_logical_cores() usize;
    extern fn fu_count_colocations() usize;
    extern fn fu_count_numa_nodes() usize;
    extern fn fu_count_quality_levels() usize;
    extern fn fu_volume_any_pages() usize;
    extern fn fu_volume_any_pages_in(numa_node_index: usize) usize;
    extern fn fu_volume_huge_pages_in(numa_node_index: usize) usize;

    extern fn fu_pool_new(name: ?[*:0]const u8) ?*anyopaque;
    extern fn fu_pool_delete(pool: *anyopaque) void;
    extern fn fu_pool_spawn(pool: *anyopaque, threads: usize, exclusivity: c_int) c_int;
    extern fn fu_pool_terminate(pool: *anyopaque) void;
    extern fn fu_pool_count_threads(pool: *anyopaque) usize;
    extern fn fu_pool_count_colocations(pool: *anyopaque) usize;
    extern fn fu_pool_count_threads_in(pool: *anyopaque, colocation_index: usize) usize;
    extern fn fu_pool_locate_thread_in(pool: *anyopaque, global_thread_index: usize, colocation_index: usize) usize;

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

    extern fn fu_pool_unsafe_for_threads(
        pool: *anyopaque,
        callback: *const fn (?*anyopaque, usize, usize) callconv(.c) void,
        context: ?*anyopaque,
    ) void;
    extern fn fu_pool_unsafe_join(pool: *anyopaque) void;
    extern fn fu_pool_sleep(pool: *anyopaque, micros: usize) void;

    extern fn fu_allocate_at_least(
        numa_node_index: usize,
        minimum_bytes: usize,
        allocated_bytes: *usize,
        bytes_per_page: *usize,
    ) ?*anyopaque;
    extern fn fu_allocate(numa_node_index: usize, bytes: usize) ?*anyopaque;
    extern fn fu_free(numa_node_index: usize, pointer: *anyopaque, bytes: usize) void;
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
    /// The colocation group (NUMA node + QoS level)
    colocation_index: usize,
};

/// Returns the library version as a struct
pub fn version() struct { major: u32, minor: u32, patch: u32 } {
    return .{
        .major = @intCast(c.fu_version_major()),
        .minor = @intCast(c.fu_version_minor()),
        .patch = @intCast(c.fu_version_patch()),
    };
}

/// Returns true if NUMA support was compiled into the library
pub fn numaEnabled() bool {
    return c.fu_enabled_numa() != 0;
}

/// Returns a string describing available platform capabilities
pub fn capabilitiesString() [*:0]const u8 {
    return c.fu_capabilities_string();
}

/// Returns the number of logical CPU cores available
pub fn countLogicalCores() usize {
    return c.fu_count_logical_cores();
}

/// Returns the number of NUMA nodes available
pub fn countNumaNodes() usize {
    return c.fu_count_numa_nodes();
}

/// Returns the number of distinct thread colocations
pub fn countColocations() usize {
    return c.fu_count_colocations();
}

/// Returns the number of distinct Quality-of-Service levels
pub fn countQualityLevels() usize {
    return c.fu_count_quality_levels();
}

/// Returns total volume of pages available across all NUMA nodes
pub fn volumeAnyPages() usize {
    return c.fu_volume_any_pages();
}

/// Returns volume of pages available on a specific NUMA node
pub fn volumeAnyPagesIn(numa_node_index: usize) usize {
    return c.fu_volume_any_pages_in(numa_node_index);
}

/// Returns volume of huge pages available on a specific NUMA node
pub fn volumeHugePagesIn(numa_node_index: usize) usize {
    return c.fu_volume_huge_pages_in(numa_node_index);
}

/// NUMA-aware memory allocation result
pub const NumaAllocation = struct {
    ptr: [*]u8,
    allocated_bytes: usize,
    bytes_per_page: usize,
    numa_node: usize,

    /// Returns the allocated memory as a slice
    pub fn asSlice(self: NumaAllocation) []u8 {
        return self.ptr[0..self.allocated_bytes];
    }

    /// Frees the NUMA allocation
    pub fn free(self: NumaAllocation) void {
        c.fu_free(self.numa_node, @ptrCast(self.ptr), self.allocated_bytes);
    }
};

/// Allocates memory on a specific NUMA node with optimal page size
pub fn allocateAtLeast(numa_node_index: usize, minimum_bytes: usize) ?NumaAllocation {
    var allocated_bytes: usize = undefined;
    var bytes_per_page: usize = undefined;

    const ptr = c.fu_allocate_at_least(
        numa_node_index,
        minimum_bytes,
        &allocated_bytes,
        &bytes_per_page,
    ) orelse return null;

    return .{
        .ptr = @ptrCast(@alignCast(ptr)),
        .allocated_bytes = allocated_bytes,
        .bytes_per_page = bytes_per_page,
        .numa_node = numa_node_index,
    };
}

/// Allocates exactly the requested bytes on a specific NUMA node
pub fn allocate(numa_node_index: usize, bytes: usize) ?[*]u8 {
    const ptr = c.fu_allocate(numa_node_index, bytes) orelse return null;
    return @ptrCast(@alignCast(ptr));
}

/// NUMA-aware allocator compatible with Zig's allocator interface.
pub const NumaAllocator = struct {
    node_index: usize,

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

    pub fn init(node_index: usize) Self {
        return .{ .node_index = node_index };
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
        const raw_ptr = c.fu_allocate_at_least(
            self.node_index,
            request_bytes,
            &allocated_bytes,
            &bytes_per_page,
        ) orelse return null;

        const base_addr = @intFromPtr(raw_ptr);
        const data_addr = alignment.forward(base_addr + header_size);
        if (data_addr + len > base_addr + allocated_bytes) {
            c.fu_free(self.node_index, raw_ptr, allocated_bytes);
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
        c.fu_free(self.node_index, base_ptr, header.allocated_bytes);
    }
};

/// Thread pool for fork-join parallelism
pub const Pool = struct {
    handle: *anyopaque,

    /// Creates a new thread pool
    pub fn init(thread_count: usize, exclusivity: CallerExclusivity) Error!Pool {
        return initNamed(null, thread_count, exclusivity);
    }

    /// Creates a new named thread pool
    pub fn initNamed(
        name: ?[]const u8,
        thread_count: usize,
        exclusivity: CallerExclusivity,
    ) Error!Pool {
        // Convert name to null-terminated string if provided
        // SAFETY: C library copies name into internal buffer immediately
        var name_buf: [16:0]u8 = undefined;
        const name_z: ?[*:0]const u8 = if (name) |n|
            std.fmt.bufPrintZ(&name_buf, "{s}", .{n[0..@min(n.len, 15)]}) catch unreachable
        else
            null;

        const handle = c.fu_pool_new(name_z) orelse return Error.CreationFailed;
        errdefer c.fu_pool_delete(handle);

        // C++ validates threads > 0 and returns false if invalid
        const success = c.fu_pool_spawn(handle, thread_count, @intFromEnum(exclusivity));
        if (success == 0) return Error.SpawnFailed;

        return .{ .handle = handle };
    }

    /// Destroys the thread pool
    pub fn deinit(self: Pool) void {
        c.fu_pool_delete(self.handle);
    }

    /// Returns the number of threads in the pool
    pub fn threads(self: *const Pool) usize {
        return c.fu_pool_count_threads(self.handle);
    }

    /// Returns the number of colocations in the pool
    pub fn colocations(self: *const Pool) usize {
        return c.fu_pool_count_colocations(self.handle);
    }

    /// Returns the number of threads in a specific colocation
    pub fn countThreadsIn(self: *const Pool, colocation_index: usize) usize {
        return c.fu_pool_count_threads_in(self.handle, colocation_index);
    }

    /// Converts global thread index to local index within colocation
    pub fn locateThreadIn(self: *const Pool, global_thread_index: usize, colocation_index: usize) usize {
        return c.fu_pool_locate_thread_in(self.handle, global_thread_index, colocation_index);
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
    /// Note: context parameter exists for API uniformity but is not passed to the callback
    pub fn forThreads(
        self: *const Pool,
        comptime func: fn (usize, usize) void,
        context: anytype,
    ) void {
        _ = context; // Not used - kept for API consistency with forN/forNDynamic/forSlices
        const Wrapper = struct {
            fn callback(_: ?*anyopaque, thread_idx: usize, colocation_idx: usize) callconv(.c) void {
                func(thread_idx, colocation_idx);
            }
        };

        c.fu_pool_for_threads(self.handle, Wrapper.callback, null);
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
                    colocation_idx: usize,
                ) callconv(.c) void {
                    const prong = Prong{
                        .task_index = task_idx,
                        .thread_index = thread_idx,
                        .colocation_index = colocation_idx,
                    };
                    func(prong, {});
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
                    colocation_idx: usize,
                ) callconv(.c) void {
                    const prong = Prong{
                        .task_index = task_idx,
                        .thread_index = thread_idx,
                        .colocation_index = colocation_idx,
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
                    colocation_idx: usize,
                ) callconv(.c) void {
                    const prong = Prong{
                        .task_index = task_idx,
                        .thread_index = thread_idx,
                        .colocation_index = colocation_idx,
                    };
                    func(prong, {});
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
                    colocation_idx: usize,
                ) callconv(.c) void {
                    const prong = Prong{
                        .task_index = task_idx,
                        .thread_index = thread_idx,
                        .colocation_index = colocation_idx,
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
                    colocation_idx: usize,
                ) callconv(.c) void {
                    const prong = Prong{
                        .task_index = first_idx,
                        .thread_index = thread_idx,
                        .colocation_index = colocation_idx,
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
                    colocation_idx: usize,
                ) callconv(.c) void {
                    const prong = Prong{
                        .task_index = first_idx,
                        .thread_index = thread_idx,
                        .colocation_index = colocation_idx,
                    };
                    // SAFETY: Context pointer valid for duration of blocking call
                    const typed_ctx: *const Context = @ptrCast(@alignCast(ctx));
                    func(prong, count, typed_ctx.*);
                }
            };
            c.fu_pool_for_slices(self.handle, n, Wrapper.callback, @ptrCast(@constCast(&context)));
        }
    }

    /// Executes callback on all threads without blocking (unsafe)
    pub fn unsafeForThreads(
        self: *const Pool,
        comptime func: fn (usize, usize) void,
        context: anytype,
    ) void {
        _ = context; // Not used - kept for API consistency
        const Wrapper = struct {
            fn callback(_: ?*anyopaque, thread_idx: usize, colocation_idx: usize) callconv(.c) void {
                func(thread_idx, colocation_idx);
            }
        };

        c.fu_pool_unsafe_for_threads(self.handle, Wrapper.callback, null);
    }

    /// Blocks until current parallel operation completes (unsafe)
    pub fn unsafeJoin(self: *const Pool) void {
        c.fu_pool_unsafe_join(self.handle);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "version info" {
    std.debug.print("Running test: version info\n", .{});
    const v = version();
    try std.testing.expect(v.major >= 0);
    try std.testing.expect(v.minor >= 0);
}

test "system capabilities" {
    std.debug.print("Running test: system capabilities\n", .{});
    const caps = capabilitiesString();
    try std.testing.expect(std.mem.len(caps) > 0);
}

test "system metadata" {
    std.debug.print("Running test: system metadata\n", .{});
    const cores = countLogicalCores();
    try std.testing.expect(cores > 0);

    const numa = countNumaNodes();
    try std.testing.expect(numa >= 0);

    const colocs = countColocations();
    try std.testing.expect(colocs > 0);
}

test "pool creation and destruction" {
    std.debug.print("Running test: pool creation and destruction\n", .{});
    var pool = try Pool.init(2, .inclusive);
    defer pool.deinit();

    try std.testing.expectEqual(@as(usize, 2), pool.threads());
}

test "named pool creation" {
    std.debug.print("Running test: named pool creation\n", .{});
    var pool = try Pool.initNamed(null, 2, .inclusive);
    defer pool.deinit();

    try std.testing.expectEqual(@as(usize, 2), pool.threads());
}

test "for_threads execution" {
    std.debug.print("Running test: for_threads execution\n", .{});
    var pool = try Pool.init(4, .inclusive);
    defer pool.deinit();

    const State = struct {
        var visited: [4]std.atomic.Value(bool) = [_]std.atomic.Value(bool){
            std.atomic.Value(bool).init(false),
            std.atomic.Value(bool).init(false),
            std.atomic.Value(bool).init(false),
            std.atomic.Value(bool).init(false),
        };

        fn worker(thread_idx: usize, colocation_idx: usize) void {
            _ = colocation_idx;
            if (thread_idx < 4) {
                visited[thread_idx].store(true, .release);
            }
        }
    };

    pool.forThreads(State.worker, {});

    // Verify all threads executed
    for (0..4) |i| {
        try std.testing.expect(State.visited[i].load(.acquire));
    }
}

test "for_n static scheduling" {
    std.debug.print("Running test: for_n static scheduling\n", .{});
    var pool = try Pool.init(4, .inclusive);
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
    var pool = try Pool.init(4, .inclusive);
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

    try std.testing.expectEqual(@as(usize, 100), counter.load(.acquire));
}

test "for_slices execution" {
    std.debug.print("Running test: for_slices execution\n", .{});
    var pool = try Pool.init(4, .inclusive);
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
    try std.testing.expectEqual(@as(usize, 1000), total.load(.acquire));
    for (0..1000) |i| {
        try std.testing.expectEqual(@as(i32, @intCast(i)), data[i]);
    }
}

test "NUMA allocation" {
    std.debug.print("Running test: NUMA allocation\n", .{});
    if (!numaEnabled()) return error.SkipZigTest;

    const allocation = allocateAtLeast(0, 1024) orelse return error.SkipZigTest;
    defer allocation.free();

    try std.testing.expect(allocation.allocated_bytes >= 1024);
    try std.testing.expectEqual(@as(usize, 0), allocation.numa_node);

    // Write to memory to ensure it's usable
    const slice = allocation.asSlice();
    for (0..@min(1024, slice.len)) |i| {
        slice[i] = @intCast(i & 0xFF);
    }
}

test "NUMA allocator integrates with std collections" {
    std.debug.print("Running test: NUMA allocator integrates with std collections\n", .{});
    if (!numaEnabled()) return error.SkipZigTest;

    var numa_alloc = NumaAllocator.init(0);
    const allocator = numa_alloc.allocator();

    var list = try std.ArrayList(u64).initCapacity(allocator, 0);
    defer list.deinit(allocator);
    try list.appendSlice(allocator, &[_]u64{ 1, 2, 3, 4, 5 });
    try std.testing.expectEqual(@as(usize, 5), list.items.len);
    try std.testing.expectEqual(@as(u64, 3), list.items[2]);

    var map = std.AutoHashMap(u32, u32).init(allocator);
    defer map.deinit();
    try map.put(10, 100);
    try map.put(20, 200);
    try map.put(30, 300);
    try std.testing.expectEqual(@as(usize, 3), map.count());
    try std.testing.expectEqual(@as(u32, 200), map.get(20).?);

    var buf = try allocator.alloc(u8, 128);
    defer allocator.free(buf);
    @memset(buf, 0xAB);

    buf = try allocator.realloc(buf, 512);
    try std.testing.expectEqual(@as(usize, 512), buf.len);
    try std.testing.expectEqual(@as(u8, 0xAB), buf[0]);
}
