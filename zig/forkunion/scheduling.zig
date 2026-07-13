//! The fork-join thread pool, mirroring the C++ `flat` and `distributed` scheduling layer.
//!
//! `Pool` spawns worker threads onto a `Topology`, then dispatches parallel loops - `forThreads`,
//! `forN`, `forNDynamic`, `forSlices` - and the non-blocking generation-token API on top of the C ABI.

const std = @import("std");
const topology = @import("topology.zig");
const Topology = topology.Topology;
const Error = topology.Error;
const CallerExclusivity = topology.CallerExclusivity;
const Capabilities = topology.Capabilities;
const Prong = @import("types.zig").Prong;

// Pool lifecycle & introspection
extern fn fu_pool_new(name: ?[*:0]const u8, allowed: u32) ?*anyopaque;
extern fn fu_pool_delete(pool: *anyopaque) void;
extern fn fu_pool_spawn(topology: *anyopaque, pool: *anyopaque, threads: usize, exclusivity: c_int) c_int;
extern fn fu_pool_spawn_on(topology: *anyopaque, pool: *anyopaque, compute_domain_index: usize, threads: usize, exclusivity: c_int) c_int;
extern fn fu_pool_terminate(pool: *anyopaque) void;
extern fn fu_pool_sleep(pool: *anyopaque, micros: usize) void;
extern fn fu_pool_caller_exclusivity(pool: *anyopaque) c_int;
extern fn fu_pool_capabilities(pool: *anyopaque) u32;
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

/// Thread pool for fork-join parallelism
pub const Pool = struct {
    handle: *anyopaque,

    /// Creates a new thread pool
    pub fn init(topo: Topology, thread_count: usize, exclusivity: CallerExclusivity) Error!Pool {
        return initNamed(topo, null, thread_count, exclusivity);
    }

    /// Creates a new named thread pool
    pub fn initNamed(
        topo: Topology,
        name: ?[]const u8,
        thread_count: usize,
        exclusivity: CallerExclusivity,
    ) Error!Pool {
        return initNamedWithCapabilities(topo, name, thread_count, exclusivity, Capabilities.all());
    }

    /// As `initNamed`, but constrains the pool to `allowed`: clear a waiter bit to force a
    /// lower-priority busy-wait, or clear `place_memory_on_domain` to force the flat (non-NUMA) pool.
    pub fn initNamedWithCapabilities(
        topo: Topology,
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

        const handle = fu_pool_new(name_z, @bitCast(allowed)) orelse return Error.CreationFailed;
        errdefer fu_pool_delete(handle);

        // C++ validates threads > 0 and returns false if invalid
        const success = fu_pool_spawn(topo.handle, handle, thread_count, @intFromEnum(exclusivity));
        if (success == 0) return Error.SpawnFailed;

        return .{ .handle = handle };
    }

    /// Spawns a thread pool pinned to a single compute domain (cores of one QoS + locality).
    ///
    /// The pool's threads and memory-domain-local allocations stay on `compute_domain_index`, in
    /// `0..countComputeDomains()`. Spawn one per compute domain and coordinate them from a
    /// single thread with the generation-token API. On builds without NUMA, only compute
    /// domain 0 is valid.
    pub fn spawnOn(topo: Topology, compute_domain_index: usize, thread_count: usize, exclusivity: CallerExclusivity) Error!Pool {
        return spawnOnWithCapabilities(topo, compute_domain_index, thread_count, exclusivity, Capabilities.all());
    }

    /// As `spawnOn`, but constrains the colocated pool to `allowed`.
    pub fn spawnOnWithCapabilities(
        topo: Topology,
        compute_domain_index: usize,
        thread_count: usize,
        exclusivity: CallerExclusivity,
        allowed: Capabilities,
    ) Error!Pool {
        const handle = fu_pool_new(null, @bitCast(allowed)) orelse return Error.CreationFailed;
        errdefer fu_pool_delete(handle);

        const success = fu_pool_spawn_on(topo.handle, handle, compute_domain_index, thread_count, @intFromEnum(exclusivity));
        if (success == 0) return Error.SpawnFailed;

        return .{ .handle = handle };
    }

    /// Destroys the thread pool
    pub fn deinit(self: Pool) void {
        fu_pool_delete(self.handle);
    }

    /// Returns the number of threads in the pool
    pub fn threads(self: *const Pool) usize {
        return fu_pool_threads_count(self.handle);
    }

    /// Returns whether the calling thread participates in the workload.
    ///
    /// Queries the pool directly rather than caching, so it stays correct across
    /// `terminate` and re-spawning with a different exclusivity.
    pub fn callerExclusivity(self: *const Pool) CallerExclusivity {
        return @enumFromInt(fu_pool_caller_exclusivity(self.handle));
    }

    /// Returns the capabilities this pool actually spawned with - the allow-mask intersected with
    /// what the build compiled and the machine offers, so a NUMA-less box reports the flat pool.
    pub fn capabilities(self: *const Pool) Capabilities {
        return @bitCast(fu_pool_capabilities(self.handle));
    }

    /// Returns the number of compute_domains in the pool
    pub fn compute_domains(self: *const Pool) usize {
        return fu_pool_compute_domains_count(self.handle);
    }

    /// Returns the number of threads in a specific compute_domain
    pub fn countThreadsIn(self: *const Pool, compute_domain_index: usize) usize {
        return fu_pool_threads_count_in(self.handle, compute_domain_index);
    }

    /// Converts global thread index to local index within compute_domain
    pub fn locateThreadIn(self: *const Pool, global_thread_index: usize, compute_domain_index: usize) usize {
        return fu_pool_locate_thread_in(self.handle, global_thread_index, compute_domain_index);
    }

    /// Terminates all worker threads (pool can be respawned)
    pub fn terminate(self: *const Pool) void {
        fu_pool_terminate(self.handle);
    }

    /// Puts worker threads into power-saving sleep state
    pub fn sleep(self: *const Pool, microseconds: usize) void {
        fu_pool_sleep(self.handle, microseconds);
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
            fu_pool_for_threads(self.handle, Wrapper.callback, null);
        } else {
            const Wrapper = struct {
                fn callback(ctx: ?*anyopaque, thread_idx: usize, compute_domain_idx: usize) callconv(.c) void {
                    const typed_ctx: *const Context = @ptrCast(@alignCast(ctx));
                    func(thread_idx, compute_domain_idx, typed_ctx.*);
                }
            };
            fu_pool_for_threads(self.handle, Wrapper.callback, @ptrCast(@constCast(&context)));
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
            fu_pool_for_n(self.handle, n, Wrapper.callback, null);
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
            fu_pool_for_n(self.handle, n, Wrapper.callback, @ptrCast(@constCast(&context)));
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
            fu_pool_for_n_dynamic(self.handle, n, Wrapper.callback, null);
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
            fu_pool_for_n_dynamic(self.handle, n, Wrapper.callback, @ptrCast(@constCast(&context)));
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
            fu_pool_for_slices(self.handle, n, Wrapper.callback, null);
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
            fu_pool_for_slices(self.handle, n, Wrapper.callback, @ptrCast(@constCast(&context)));
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
            return fu_pool_unsafe_for_threads(self.handle, Wrapper.callback, null);
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
            return fu_pool_unsafe_for_threads(self.handle, Wrapper.callback, @ptrCast(@constCast(context)));
        }
    }

    /// Returns true if the given generation has completed.
    ///
    /// A `true` result also guarantees visibility of every contributor's writes. On
    /// caller-inclusive pools this can only turn `true` once `unsafeJoin` contributes
    /// the calling thread's slice, so poll-then-join is reserved for exclusive pools.
    pub fn isComplete(self: *const Pool, generation: usize) bool {
        return fu_pool_is_complete(self.handle, generation) != 0;
    }

    /// Blocks until the given generation completes (unsafe).
    /// On caller-inclusive pools this also executes the calling thread's slice.
    /// Idempotent: joining an already-joined generation returns immediately.
    pub fn unsafeJoin(self: *const Pool, generation: usize) void {
        fu_pool_unsafe_join(self.handle, generation);
    }
};

test "pool creation and destruction" {
    const topo = try Topology.init();
    defer topo.deinit();
    var pool = try Pool.init(topo, 2, .inclusive);
    defer pool.deinit();

    try std.testing.expectEqual(2, pool.threads());
}

test "caller exclusivity query" {
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

test "pool capabilities reflect the build" {
    const topo = try Topology.init();
    defer topo.deinit();
    // A pool reports the effective capabilities it spawned with; the allow-mask is a hard ceiling.
    var pool = try Pool.init(topo, 2, .inclusive);
    defer pool.deinit();
    const full: u32 = @bitCast(pool.capabilities());

    // Clearing a bit in the allow-mask must clear it in the effective set - the mask can only
    // subtract. Forcing off `place_memory_on_domain` demotes a NUMA pool to the flat pool.
    var flat_mask = Capabilities.all();
    flat_mask.place_memory_on_domain = false;
    var flat_pool = try Pool.initNamedWithCapabilities(topo, null, 2, .inclusive, flat_mask);
    defer flat_pool.deinit();
    const flat: u32 = @bitCast(flat_pool.capabilities());

    try std.testing.expect(!flat_pool.capabilities().place_memory_on_domain);
    // The masked pool is a subset of the unmasked one - masking never adds a facility.
    try std.testing.expectEqual(flat, flat & full);
}

test "per-compute_domain pool" {
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
    const topo = try Topology.init();
    defer topo.deinit();
    var pool = try Pool.initNamed(topo, null, 2, .inclusive);
    defer pool.deinit();

    try std.testing.expectEqual(2, pool.threads());
}

test "for_threads execution" {
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

test "for_n void context" {
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

    // Exclusive pools complete without the caller contributing a slice, so `unsafeJoin`
    // blocks purely on the workers; afterwards the completion query must observe them done -
    // a deterministic check with no busy-wait and no timing assumptions.
    try std.testing.expect(generation & 1 == 1);
    pool.unsafeJoin(generation);
    try std.testing.expect(pool.isComplete(generation));

    // All 4 worker threads should have executed
    try std.testing.expectEqual(4, counter.load(.acquire));
}
