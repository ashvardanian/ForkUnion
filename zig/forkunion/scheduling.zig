//! The fork-join thread pool, mirroring the C++ `flat` and `distributed` scheduling layer.
//!
//! `Pool` spawns worker threads onto a `Topology`, then dispatches parallel loops - `forThreads`,
//! `forN`, `forNDynamic`, `forSlices`, `forSlicesMut` - and the non-blocking generation-token API
//! on top of the C ABI. Every dispatch takes the context before the callback and forwards it as a
//! caller-owned pointer, so the callback sees the qualifiers the caller chose.

const std = @import("std");
const topology = @import("topology.zig");
const Topology = topology.Topology;
const Error = types.Error;
const CallerExclusivity = topology.CallerExclusivity;
const Capabilities = topology.Capabilities;
const types = @import("types.zig");
const ComputeDomain = types.ComputeDomain;
const MemoryDomain = types.MemoryDomain;
const IndexedSplit = types.IndexedSplit;
const ThreadInDomain = types.ThreadInDomain;
const TasksRange = types.TasksRange;

extern fn fu_pool_new(name: ?[*:0]const u8, allowed: u32, pool_out: *?*anyopaque) c_int;
extern fn fu_pool_delete(pool: *anyopaque) void;
extern fn fu_pool_spawn(topology: *anyopaque, pool: *anyopaque, threads: usize, exclusivity: c_int) c_int;
extern fn fu_pool_spawn_on(topology: *anyopaque, pool: *anyopaque, compute_domain_index: usize, threads: usize, exclusivity: c_int) c_int;
extern fn fu_pool_terminate(pool: *anyopaque) void;
extern fn fu_pool_sleep(pool: *anyopaque, micros: usize) void;
extern fn fu_pool_caller_exclusivity(pool: *anyopaque, exclusivity_out: *c_int) c_int;
extern fn fu_pool_capabilities(pool: *anyopaque, capabilities_out: *u32) c_int;
extern fn fu_pool_threads_count(pool: *anyopaque, out: *usize) c_int;
extern fn fu_pool_compute_domains_count(pool: *anyopaque, out: *usize) c_int;
extern fn fu_pool_threads_count_in(pool: *anyopaque, compute_domain_index: usize, out: *usize) c_int;
extern fn fu_pool_locate_thread_in(pool: *anyopaque, global_thread_index: usize, compute_domain_index: usize, out: *usize) c_int;

extern fn fu_pool_for_threads(
    pool: *anyopaque,
    callback: *const fn (?*anyopaque, usize, usize) callconv(.c) void,
    context: ?*anyopaque,
) c_int;
extern fn fu_pool_for_n(
    pool: *anyopaque,
    n: usize,
    callback: *const fn (?*anyopaque, usize, usize, usize) callconv(.c) void,
    context: ?*anyopaque,
) c_int;
extern fn fu_pool_for_n_dynamic(
    pool: *anyopaque,
    n: usize,
    callback: *const fn (?*anyopaque, usize, usize, usize) callconv(.c) void,
    context: ?*anyopaque,
) c_int;
extern fn fu_pool_for_slices(
    pool: *anyopaque,
    n: usize,
    callback: *const fn (?*anyopaque, usize, usize, usize, usize) callconv(.c) void,
    context: ?*anyopaque,
) c_int;

extern fn fu_pool_unsafe_for_threads(
    pool: *anyopaque,
    callback: *const fn (?*anyopaque, usize, usize) callconv(.c) void,
    context: ?*anyopaque,
    generation_out: *usize,
) c_int;
extern fn fu_pool_is_complete(pool: *anyopaque, generation: usize, complete_out: *c_int) c_int;
extern fn fu_pool_unsafe_join(pool: *anyopaque, generation: usize) void;

extern fn fu_fabric_new(fabric_out: *?*anyopaque) c_int;
extern fn fu_fabric_delete(fabric: *anyopaque) void;
extern fn fu_fabric_harvest(topology: *anyopaque, pool: *anyopaque, fabric: *anyopaque) c_int;
extern fn fu_fabric_memory_latency(fabric: *anyopaque, compute_domain_index: usize, memory_domain_index: usize, out: *usize) c_int;
extern fn fu_fabric_memory_bandwidth(fabric: *anyopaque, compute_domain_index: usize, memory_domain_index: usize, out: *usize) c_int;
extern fn fu_fabric_memory_distance(fabric: *anyopaque, compute_domain_index: usize, memory_domain_index: usize, out: *usize) c_int;
extern fn fu_fabric_memory_level_in(fabric: *anyopaque, memory_domain_index: usize, out: *usize) c_int;
extern fn fu_fabric_memory_levels_count(fabric: *anyopaque, out: *usize) c_int;

/// Rejects a callback the trampoline cannot call, before instantiation buries the reason under a
/// generic-expansion trace. Only the shape is checked - argument coercion is left to the call
/// itself, so a `*const T` parameter still accepts a `*T` context.
inline fn checkCallable(comptime func: anytype, comptime arity: usize) void {
    const info = @typeInfo(@TypeOf(func));
    if (info != .@"fn")
        @compileError(std.fmt.comptimePrint(
            "Callback must be a `fn`, got `{s}`",
            .{@typeName(@TypeOf(func))},
        ));
    if (info.@"fn".params.len != arity)
        @compileError(std.fmt.comptimePrint(
            "Callback must take {d} parameters, `{s}` takes {d}",
            .{ arity, @typeName(@TypeOf(func)), info.@"fn".params.len },
        ));
    // Null for a generic `fn`, whose return type is not known until it is instantiated.
    if (info.@"fn".return_type) |returned| {
        if (returned != void)
            @compileError(std.fmt.comptimePrint(
                "Callback must return `void`, got `{s}`",
                .{@typeName(returned)},
            ));
    }
}

/// Rejects a context that cannot cross the FFI boundary. Every dispatch hands the C API one
/// `*anyopaque`, so a slice or a many-item pointer has no single address to pass.
inline fn checkContext(comptime Context: type) void {
    if (Context == void) return;
    const info = @typeInfo(Context);
    if (info != .pointer or info.pointer.size != .one)
        @compileError(std.fmt.comptimePrint(
            "Context must be `void` or a single-item pointer like `&my_context`, got `{s}`",
            .{@typeName(Context)},
        ));
}

/// The pointer a dispatch hands the C API. A `void` context travels as null and is never read.
inline fn contextPointer(comptime Context: type, context: Context) ?*anyopaque {
    return if (Context == void) null else @ptrCast(@constCast(context));
}

/// How a pool is spawned: width, placement, and the facilities it may use.
///
/// One options struct rather than a family of constructors, so a call site names only what it
/// changes and a new knob costs no new overload.
pub const PoolOptions = struct {
    /// Threads to run, the calling thread included when `exclusivity` is `.inclusive`.
    threads: usize,
    /// Which cores the pool may use: every compute domain, or exactly one.
    placement: Placement = .everywhere,
    /// Optional thread name, clipped by the core to what the platform's thread naming accepts.
    name: ?[]const u8 = null,
    /// Whether the calling thread runs work or only coordinates.
    exclusivity: CallerExclusivity = .inclusive,
    /// Allow-mask: clear a waiter bit to force a lower-priority busy-wait, or clear
    /// `place_memory_on_domain` to force the flat, non-NUMA pool.
    allowed: Capabilities = Capabilities.all(),

    /// Both placements are named, so "every domain" is a state rather than a missing index.
    pub const Placement = union(enum) {
        /// Spread across every compute domain the topology reports.
        everywhere,
        /// Pin threads and domain-local allocations to one compute domain. On builds without
        /// NUMA only compute domain 0 is valid.
        on_compute_domain: ComputeDomain,
    };
};

/// Thread pool for fork-join parallelism
pub const Pool = struct {
    handle: *anyopaque,

    /// Spawns a pool onto `topo`, per `options`.
    ///
    /// Spawn one pool per compute domain with `.placement = .{ .on_compute_domain = … }` and
    /// coordinate them from a single thread with the generation-token API.
    pub fn init(topo: Topology, options: PoolOptions) Error!Pool {
        // SAFETY: the C library copies the name into an internal buffer immediately, then clips it
        // to whatever the platform's thread naming accepts - this only has to null-terminate a copy.
        // `FU_POOL_NAME_CAPACITY`; the C side clips anything longer.
        var name_buf: [16]u8 = undefined;
        const name_z: ?[*:0]const u8 = if (options.name) |given|
            std.fmt.bufPrintZ(&name_buf, "{s}", .{given[0..@min(given.len, name_buf.len - 1)]}) catch unreachable
        else
            null;

        var new_handle: ?*anyopaque = null;
        try types.check(fu_pool_new(name_z, @bitCast(options.allowed), &new_handle));
        const handle = new_handle orelse return Error.BadAlloc;
        errdefer fu_pool_delete(handle);

        const exclusivity = @intFromEnum(options.exclusivity);
        const spawned = switch (options.placement) {
            .everywhere => fu_pool_spawn(topo.handle, handle, options.threads, exclusivity),
            .on_compute_domain => |domain| fu_pool_spawn_on(topo.handle, handle, domain.index(), options.threads, exclusivity),
        };
        // ? A thread limit, a zero count, and an out-of-range domain now arrive apart.
        try types.check(spawned);

        return .{ .handle = handle };
    }

    /// Destroys the thread pool
    pub fn deinit(self: Pool) void {
        fu_pool_delete(self.handle);
    }

    /// Returns the number of threads in the pool
    pub fn threadsCount(self: Pool) Error!usize {
        var answer: usize = std.math.maxInt(usize);
        try types.check(fu_pool_threads_count(self.handle, &answer));
        return answer;
    }

    /// Returns whether the calling thread participates in the workload.
    ///
    /// Queries the pool directly rather than caching, so it stays correct across
    /// `terminate` and re-spawning with a different exclusivity.
    pub fn callerExclusivity(self: Pool) Error!CallerExclusivity {
        var raw: c_int = 0;
        try types.check(fu_pool_caller_exclusivity(self.handle, &raw));
        return @enumFromInt(raw);
    }

    /// Returns the capabilities this pool actually spawned with - the allow-mask intersected with
    /// what the build compiled and the machine offers, so a NUMA-less box reports the flat pool.
    pub fn capabilities(self: Pool) Error!Capabilities {
        var bits: u32 = 0;
        try types.check(fu_pool_capabilities(self.handle, &bits));
        return @bitCast(bits);
    }

    /// Returns the number of compute domains the pool spans
    pub fn computeDomainsCount(self: Pool) Error!usize {
        var answer: usize = std.math.maxInt(usize);
        try types.check(fu_pool_compute_domains_count(self.handle, &answer));
        return answer;
    }

    /// Returns the number of threads in a specific compute domain
    pub fn threadsCountIn(self: Pool, compute_domain: ComputeDomain) Error!usize {
        var answer: usize = 0;
        try types.check(fu_pool_threads_count_in(self.handle, compute_domain.index(), &answer));
        return answer;
    }

    /// Converts a global thread index to its local index within a compute domain
    pub fn locateThreadIn(self: Pool, global_thread_index: usize, compute_domain: ComputeDomain) Error!usize {
        var answer: usize = std.math.maxInt(usize);
        try types.check(fu_pool_locate_thread_in(self.handle, global_thread_index, compute_domain.index(), &answer));
        return answer;
    }

    /// Terminates all worker threads (pool can be respawned)
    pub fn terminate(self: Pool) void {
        fu_pool_terminate(self.handle);
    }

    /// Puts worker threads into power-saving sleep state
    pub fn sleep(self: Pool, microseconds: usize) void {
        fu_pool_sleep(self.handle, microseconds);
    }

    /// Executes a callback on all threads (blocking)
    ///
    /// The context is a single-item pointer the caller owns, forwarded to the callback verbatim,
    /// so its `const`-ness carries through:
    /// - If context is `void`: `fn (usize, ComputeDomain) void`
    /// - Otherwise: `fn (@TypeOf(context), usize, ComputeDomain) void`
    pub fn forThreads(
        self: Pool,
        context: anytype,
        comptime func: anytype,
    ) Error!void {
        const Context = @TypeOf(context);
        checkContext(Context);
        checkCallable(func, if (Context == void) 2 else 3);

        const Wrapper = struct {
            fn callback(raw: ?*anyopaque, thread_index: usize, compute_domain_index: usize) callconv(.c) void {
                const compute_domain = ComputeDomain.at(compute_domain_index);
                if (Context == void)
                    func(thread_index, compute_domain)
                else
                    func(@as(Context, @ptrCast(@alignCast(raw))), thread_index, compute_domain);
            }
        };
        try types.check(fu_pool_for_threads(self.handle, Wrapper.callback, contextPointer(Context, context)));
    }

    /// Distributes N tasks across threads with static scheduling (blocking)
    ///
    /// - If context is `void`: `fn (usize, ThreadInDomain) void`
    /// - Otherwise: `fn (@TypeOf(context), usize, ThreadInDomain) void`
    pub fn forN(
        self: Pool,
        n: usize,
        context: anytype,
        comptime func: anytype,
    ) Error!void {
        const Context = @TypeOf(context);
        checkContext(Context);
        checkCallable(func, if (Context == void) 2 else 3);

        const Wrapper = struct {
            fn callback(
                raw: ?*anyopaque,
                task_index: usize,
                thread_index: usize,
                compute_domain_index: usize,
            ) callconv(.c) void {
                const at = ThreadInDomain{
                    .thread = thread_index,
                    .compute_domain = ComputeDomain.at(compute_domain_index),
                };
                if (Context == void)
                    func(task_index, at)
                else
                    func(@as(Context, @ptrCast(@alignCast(raw))), task_index, at);
            }
        };
        try types.check(fu_pool_for_n(self.handle, n, Wrapper.callback, contextPointer(Context, context)));
    }

    /// Distributes N tasks with dynamic work-stealing (blocking)
    ///
    /// - If context is `void`: `fn (usize, ThreadInDomain) void`
    /// - Otherwise: `fn (@TypeOf(context), usize, ThreadInDomain) void`
    pub fn forNDynamic(
        self: Pool,
        n: usize,
        context: anytype,
        comptime func: anytype,
    ) Error!void {
        const Context = @TypeOf(context);
        checkContext(Context);
        checkCallable(func, if (Context == void) 2 else 3);

        const Wrapper = struct {
            fn callback(
                raw: ?*anyopaque,
                task_index: usize,
                thread_index: usize,
                compute_domain_index: usize,
            ) callconv(.c) void {
                const at = ThreadInDomain{
                    .thread = thread_index,
                    .compute_domain = ComputeDomain.at(compute_domain_index),
                };
                if (Context == void)
                    func(task_index, at)
                else
                    func(@as(Context, @ptrCast(@alignCast(raw))), task_index, at);
            }
        };
        try types.check(fu_pool_for_n_dynamic(self.handle, n, Wrapper.callback, contextPointer(Context, context)));
    }

    /// Distributes N tasks as slices (blocking)
    ///
    /// The `TasksRange` names the half-open run of task indices this worker drew.
    /// - If context is `void`: `fn (TasksRange, ThreadInDomain) void`
    /// - Otherwise: `fn (@TypeOf(context), TasksRange, ThreadInDomain) void`
    pub fn forSlices(
        self: Pool,
        n: usize,
        context: anytype,
        comptime func: anytype,
    ) Error!void {
        const Context = @TypeOf(context);
        checkContext(Context);
        checkCallable(func, if (Context == void) 2 else 3);

        const Wrapper = struct {
            fn callback(
                raw: ?*anyopaque,
                first_index: usize,
                count: usize,
                thread_index: usize,
                compute_domain_index: usize,
            ) callconv(.c) void {
                const range = TasksRange{ .first = first_index, .count = count };
                const at = ThreadInDomain{
                    .thread = thread_index,
                    .compute_domain = ComputeDomain.at(compute_domain_index),
                };
                if (Context == void)
                    func(range, at)
                else
                    func(@as(Context, @ptrCast(@alignCast(raw))), range, at);
            }
        };
        try types.check(fu_pool_for_slices(self.handle, n, Wrapper.callback, contextPointer(Context, context)));
    }

    /// Splits `data` into one contiguous chunk per thread and runs `func` on each (blocking).
    ///
    /// The chunks partition `data`, so no two threads observe overlapping elements and the
    /// callback can write its own slice with no atomics and no index arithmetic.
    /// - If context is `void`: `fn ([]T, ThreadInDomain) void`
    /// - Otherwise: `fn (@TypeOf(context), []T, ThreadInDomain) void`
    pub fn forSlicesMut(
        self: Pool,
        comptime T: type,
        data: []T,
        context: anytype,
        comptime func: anytype,
    ) Error!void {
        const Context = @TypeOf(context);
        checkContext(Context);
        checkCallable(func, if (Context == void) 2 else 3);

        // Rides on `forSlices` rather than re-deriving the split: every thread is dispatched
        // exactly once, an idle one with an empty range, so the partitions cannot drift apart.
        const Scatter = struct { data: []T, context: Context };
        var scatter = Scatter{ .data = data, .context = context };
        try self.forSlices(data.len, &scatter, struct {
            fn spread(carried: *const Scatter, range: TasksRange, at: ThreadInDomain) void {
                const chunk = range.of(T, carried.data);
                if (Context == void) func(chunk, at) else func(carried.context, chunk, at);
            }
        }.spread);
    }

    /// Executes callback on all threads without blocking (unsafe).
    /// Returns an always-odd generation token to pass to `isComplete` or `unsafeJoin`.
    ///
    /// - If context is `void`: `fn (usize, ComputeDomain) void`
    /// - Otherwise: `fn (@TypeOf(context), usize, ComputeDomain) void`
    ///
    /// Unlike the blocking dispatches, this one returns while worker threads may still be running,
    /// so the pointee must outlive `unsafeJoin` rather than merely the call.
    pub fn unsafeForThreads(
        self: Pool,
        context: anytype,
        comptime func: anytype,
    ) Error!usize {
        const Context = @TypeOf(context);
        checkContext(Context);
        checkCallable(func, if (Context == void) 2 else 3);

        const Wrapper = struct {
            fn callback(raw: ?*anyopaque, thread_index: usize, compute_domain_index: usize) callconv(.c) void {
                const compute_domain = ComputeDomain.at(compute_domain_index);
                if (Context == void)
                    func(thread_index, compute_domain)
                else
                    func(@as(Context, @ptrCast(@alignCast(raw))), thread_index, compute_domain);
            }
        };
        var generation: usize = 0;
        try types.check(fu_pool_unsafe_for_threads(self.handle, Wrapper.callback, contextPointer(Context, context), &generation));
        return generation;
    }

    /// Returns true if the given generation has completed.
    ///
    /// A `true` result also guarantees visibility of every contributor's writes. On
    /// caller-inclusive pools this can only turn `true` once `unsafeJoin` contributes
    /// the calling thread's slice, so poll-then-join is reserved for exclusive pools.
    pub fn isComplete(self: Pool, generation: usize) Error!bool {
        var complete: c_int = 0;
        try types.check(fu_pool_is_complete(self.handle, generation, &complete));
        return complete != 0;
    }

    /// Blocks until the given generation completes (unsafe).
    /// On caller-inclusive pools this also executes the calling thread's slice.
    /// Idempotent: joining an already-joined generation returns immediately.
    pub fn unsafeJoin(self: Pool, generation: usize) void {
        fu_pool_unsafe_join(self.handle, generation);
    }
};

/// The measured memory fabric - what this process observed, as opposed to the structure a
/// `Topology` declares. Two query families: edge queries `(initiator, target)` describe one
/// interconnect link; medium queries `(target)` describe the memory pool itself, independent of
/// any initiator.
///
/// Completes the `harvest` pipeline: a `Topology` is harvested first and stays immutable, a
/// `Pool` spawns on it, and the fabric then harvests through that pool's pinned workers,
/// snapshotting what it needs so the topology may be freed after. Before a harvest every query
/// answers 0, and `memoryLevelsCount` answers 1.
pub const Fabric = struct {
    handle: *anyopaque,

    /// Creates an empty, unharvested fabric.
    pub fn init() Error!Fabric {
        var new_handle: ?*anyopaque = null;
        try types.check(fu_fabric_new(&new_handle));
        return .{ .handle = new_handle orelse return Error.BadAlloc };
    }

    /// Destroys the fabric and frees its observations.
    pub fn deinit(self: Fabric) void {
        fu_fabric_delete(self.handle);
    }

    /// Measures the memory fabric through the pool's pinned workers, replacing any previous
    /// harvest; the `topo` is only read.
    ///
    /// Returns `false` on allocation failure, or for a pool whose workers are not pinned per
    /// domain - flat pools and those pinned to one compute domain have no fabric to walk; it is left
    /// empty, never half-written. Not thread-safe: it dispatches on the pool and rebuilds the
    /// fabric, so call it between task batches. Expect seconds of runtime on large fabrics.
    pub fn harvest(self: Fabric, topo: Topology, pool: Pool) Error!void {
        // ? A flat pool now reports `ConfigMismatch`, not the same failure as an exhausted heap.
        return types.check(fu_fabric_harvest(topo.handle, pool.handle, self.handle));
    }

    /// Returns the measured dependent-load latency (nanoseconds) on an edge - the best recording;
    /// 0 before a harvest, for an edge no worker could reach, or an out-of-range index.
    pub fn memoryLatency(self: Fabric, compute_domain: ComputeDomain, memory_domain: MemoryDomain) Error!usize {
        var answer: usize = std.math.maxInt(usize);
        try types.check(fu_fabric_memory_latency(self.handle, compute_domain.index(), memory_domain.index(), &answer));
        return answer;
    }

    /// Returns the measured saturated read bandwidth (MB/s) on an edge, streamed by all the
    /// initiator domain's workers at once - the best recording; 0 if unreached or out of range.
    pub fn memoryBandwidth(self: Fabric, compute_domain: ComputeDomain, memory_domain: MemoryDomain) Error!usize {
        var answer: usize = std.math.maxInt(usize);
        try types.check(fu_fabric_memory_bandwidth(self.handle, compute_domain.index(), memory_domain.index(), &answer));
        return answer;
    }

    /// Returns the relative access distance on an edge (10 = local, per the SLIT convention):
    /// the measured latency ratio to the initiator's local domain, clamped so local carries the
    /// row's minimum; unwalked edges fall back to 10-local / 20-remote.
    pub fn memoryDistance(self: Fabric, compute_domain: ComputeDomain, memory_domain: MemoryDomain) Error!usize {
        var answer: usize = std.math.maxInt(usize);
        try types.check(fu_fabric_memory_distance(self.handle, compute_domain.index(), memory_domain.index(), &answer));
        return answer;
    }

    /// Returns the derived speed class of a memory domain (lower = faster: HBM < DDR < CXL),
    /// keyed by the best bandwidth any initiator sustains to it, ties split by the best latency.
    pub fn memoryLevelIn(self: Fabric, memory_domain: MemoryDomain) Error!usize {
        var answer: usize = std.math.maxInt(usize);
        try types.check(fu_fabric_memory_level_in(self.handle, memory_domain.index(), &answer));
        return answer;
    }

    /// Returns the number of distinct derived memory tiers, the memory-axis twin of
    /// `Topology.computeLevelsCount`; 1 on single-tier systems and before a harvest.
    pub fn memoryLevelsCount(self: Fabric) Error!usize {
        var answer: usize = std.math.maxInt(usize);
        try types.check(fu_fabric_memory_levels_count(self.handle, &answer));
        return answer;
    }
};

test "pool creation and destruction" {
    const topo = try Topology.init();
    defer topo.deinit();
    const pool = try Pool.init(topo, .{ .threads = 2 });
    defer pool.deinit();

    try std.testing.expectEqual(2, try pool.threadsCount());
}

test "caller exclusivity query" {
    // The pool is the single source of truth, queried live (not cached).
    const topo = try Topology.init();
    defer topo.deinit();
    const inclusive = try Pool.init(topo, .{ .threads = 2, .exclusivity = .inclusive });
    defer inclusive.deinit();
    try std.testing.expectEqual(CallerExclusivity.inclusive, inclusive.callerExclusivity());

    const exclusive = try Pool.init(topo, .{ .threads = 2, .exclusivity = .exclusive });
    defer exclusive.deinit();
    try std.testing.expectEqual(CallerExclusivity.exclusive, exclusive.callerExclusivity());
}

test "pool capabilities reflect the build" {
    const topo = try Topology.init();
    defer topo.deinit();
    const pool = try Pool.init(topo, .{ .threads = 2 });
    defer pool.deinit();
    const full: u32 = @bitCast(try pool.capabilities());

    // Clearing a bit in the allow-mask must clear it in the effective set - the mask can
    // only subtract: forcing off `place_memory_on_domain` demotes a NUMA pool to the flat pool.
    var flat_mask = Capabilities.all();
    flat_mask.place_memory_on_domain = false;
    const flat_pool = try Pool.init(topo, .{ .threads = 2, .allowed = flat_mask });
    defer flat_pool.deinit();
    const flat: u32 = @bitCast(try flat_pool.capabilities());

    try std.testing.expect(!(try flat_pool.capabilities()).place_memory_on_domain);
    // The masked pool is a subset of the unmasked one - masking never adds a facility.
    try std.testing.expectEqual(flat, flat & full);
}

test "per-compute_domain pool" {
    const topo = try Topology.init();
    defer topo.deinit();
    const compute_domains = try topo.computeDomainsCount();
    try std.testing.expect(compute_domains >= 1);

    const first = ComputeDomain.at(0);
    const cores = @max(try topo.logicalCoresCountIn(first), 1);
    const pool = try Pool.init(topo, .{
        .threads = cores,
        .placement = .{ .on_compute_domain = first },
        .exclusivity = .exclusive,
    });
    defer pool.deinit();

    var counter = std.atomic.Value(usize).init(0);
    const generation = try pool.unsafeForThreads(&counter, struct {
        fn worker(tally: *std.atomic.Value(usize), thread_index: usize, compute_domain: ComputeDomain) void {
            _ = thread_index;
            _ = compute_domain;
            _ = tally.fetchAdd(1, .monotonic);
        }
    }.worker);
    pool.unsafeJoin(generation);
    try std.testing.expectEqual(try pool.threadsCount(), counter.load(.acquire));
}

test "fabric harvest fills edges" {
    const topo = try Topology.init();
    defer topo.deinit();
    const pool = try Pool.init(topo, .{ .threads = 4 });
    defer pool.deinit();
    const fabric = try Fabric.init();
    defer fabric.deinit();

    // An unharvested fabric answers zeros and a single tier.
    try std.testing.expectEqual(0, fabric.memoryLatency(ComputeDomain.at(0), MemoryDomain.at(0)));
    try std.testing.expectEqual(1, try fabric.memoryLevelsCount());

    // A flat pool without domain placement has no fabric to walk. Skip rather than return: a bare
    // return counts as a pass, so the summary would claim coverage this run never had.
    fabric.harvest(topo, pool) catch return error.SkipZigTest;

    // Every reachable edge must carry sane observations; emulated-NUMA guests may measure
    // equal local and remote costs, so nothing stronger is asserted.
    const first = ComputeDomain.at(0);
    const local = try topo.localMemoryOf(first);
    try std.testing.expect(try fabric.memoryLatency(first, local) > 0);
    try std.testing.expect(try fabric.memoryBandwidth(first, local) > 0);
    try std.testing.expectEqual(10, try fabric.memoryDistance(first, local));
    try std.testing.expect(try fabric.memoryLevelsCount() >= 1);
}

test "named pool creation" {
    const topo = try Topology.init();
    defer topo.deinit();

    // A name longer than any platform's thread naming accepts must clip rather than fail, so
    // exercise the clip path instead of a comfortably short name.
    const long_name = "a-pool-name-far-longer-than-any-platform-thread-naming-accepts";
    const pool = try Pool.init(topo, .{ .threads = 2, .name = long_name });
    defer pool.deinit();
    try std.testing.expectEqual(2, try pool.threadsCount());
}

test "for_threads execution" {
    const topo = try Topology.init();
    defer topo.deinit();
    const pool = try Pool.init(topo, .{ .threads = 4 });
    defer pool.deinit();

    var visited = [_]std.atomic.Value(bool){std.atomic.Value(bool).init(false)} ** 4;

    try pool.forThreads(&visited, struct {
        fn worker(seen: *[4]std.atomic.Value(bool), thread_index: usize, compute_domain: ComputeDomain) void {
            _ = compute_domain;
            if (thread_index < 4) seen[thread_index].store(true, .release);
        }
    }.worker);

    for (0..4) |i| try std.testing.expect(visited[i].load(.acquire));
}

test "for_n static scheduling" {
    const topo = try Topology.init();
    defer topo.deinit();
    const pool = try Pool.init(topo, .{ .threads = 4 });
    defer pool.deinit();

    var visited = [_]std.atomic.Value(bool){std.atomic.Value(bool).init(false)} ** 100;

    try pool.forN(100, &visited, struct {
        fn worker(seen: *[100]std.atomic.Value(bool), task: usize, at: ThreadInDomain) void {
            _ = at;
            seen[task].store(true, .release);
        }
    }.worker);

    for (0..100) |i| try std.testing.expect(visited[i].load(.acquire));
}

test "for_n_dynamic work stealing" {
    const topo = try Topology.init();
    defer topo.deinit();
    const pool = try Pool.init(topo, .{ .threads = 4 });
    defer pool.deinit();

    var counter = std.atomic.Value(usize).init(0);

    try pool.forNDynamic(100, &counter, struct {
        fn worker(tally: *std.atomic.Value(usize), task: usize, at: ThreadInDomain) void {
            _ = task;
            _ = at;
            _ = tally.fetchAdd(1, .monotonic);
        }
    }.worker);

    try std.testing.expectEqual(100, counter.load(.acquire));
}

test "for_slices execution" {
    const topo = try Topology.init();
    defer topo.deinit();
    const pool = try Pool.init(topo, .{ .threads = 4 });
    defer pool.deinit();

    var data = [_]i32{0} ** 1000;
    var total = std.atomic.Value(usize).init(0);

    const Context = struct {
        data: *[1000]i32,
        total: *std.atomic.Value(usize),
    };
    var context = Context{ .data = &data, .total = &total };

    try pool.forSlices(1000, &context, struct {
        fn worker(carried: *const Context, range: TasksRange, at: ThreadInDomain) void {
            _ = at;
            for (range.first..range.end()) |task| carried.data[task] = @intCast(task);
            _ = carried.total.fetchAdd(range.count, .monotonic);
        }
    }.worker);

    try std.testing.expectEqual(1000, total.load(.acquire));
    for (0..1000) |i| try std.testing.expectEqual(@as(i32, @intCast(i)), data[i]);
}

test "for_slices_mut hands each thread a disjoint chunk" {
    // The chunks must tile `data` exactly once, so a plain non-atomic write per element is enough:
    // any overlap or gap would show up as a wrong value in the readback.
    const topo = try Topology.init();
    defer topo.deinit();
    const pool = try Pool.init(topo, .{ .threads = 4 });
    defer pool.deinit();

    var data = [_]u64{0} ** 1000;
    var base: u64 = 7;

    try pool.forSlicesMut(u64, &data, &base, struct {
        fn fill(offset: *const u64, chunk: []u64, at: ThreadInDomain) void {
            _ = at;
            for (chunk) |*slot| slot.* = offset.*;
        }
    }.fill);

    for (data) |value| try std.testing.expectEqual(@as(u64, 7), value);
}

test "for_n void context" {
    const topo = try Topology.init();
    defer topo.deinit();
    const pool = try Pool.init(topo, .{ .threads = 4 });
    defer pool.deinit();

    // A stateless kernel needs no context at all; anything it must reach travels in one. With no
    // context there is nothing to write to, so the callback can only check its own arguments -
    // that every task actually runs is covered by the pointer-context tests above.
    try pool.forN(50, {}, struct {
        fn worker(task: usize, at: ThreadInDomain) void {
            std.debug.assert(task < 50);
            std.debug.assert(at.compute_domain.index() < 1024);
        }
    }.worker);
}

test "unsafe_for_threads and join" {
    const topo = try Topology.init();
    defer topo.deinit();
    const pool = try Pool.init(topo, .{ .threads = 4 });
    defer pool.deinit();

    var counter = std.atomic.Value(usize).init(0);

    // The pointee must outlive `unsafeJoin`: the dispatch returns while workers still read it.
    const generation = try pool.unsafeForThreads(&counter, struct {
        fn worker(tally: *std.atomic.Value(usize), thread_index: usize, compute_domain: ComputeDomain) void {
            _ = thread_index;
            _ = compute_domain;
            _ = tally.fetchAdd(1, .monotonic);
        }
    }.worker);

    try std.testing.expect(generation & 1 == 1);
    pool.unsafeJoin(generation);
    try std.testing.expect(try pool.isComplete(generation));
    try std.testing.expectEqual(4, counter.load(.acquire));
}

test "generation polling on exclusive pool" {
    const topo = try Topology.init();
    defer topo.deinit();
    const pool = try Pool.init(topo, .{ .threads = 4, .exclusivity = .exclusive });
    defer pool.deinit();

    var counter = std.atomic.Value(usize).init(0);

    const generation = try pool.unsafeForThreads(&counter, struct {
        fn worker(tally: *std.atomic.Value(usize), thread_index: usize, compute_domain: ComputeDomain) void {
            _ = thread_index;
            _ = compute_domain;
            _ = tally.fetchAdd(1, .monotonic);
        }
    }.worker);

    try std.testing.expect(generation & 1 == 1);

    // What an exclusive pool is for: the caller owes no slice, so the generation completes without
    // it and can be polled first. On an inclusive pool this loop would never finish.
    var polls: usize = 0;
    while (!try pool.isComplete(generation)) : (polls += 1) {
        if (polls > 1_000_000) return error.GenerationNeverCompleted;
        std.Thread.yield() catch std.atomic.spinLoopHint();
    }

    // Completion also publishes every contributor's writes, so no join is needed to read them.
    try std.testing.expectEqual(4, counter.load(.acquire));

    // Joining an already-complete generation returns immediately and changes nothing.
    pool.unsafeJoin(generation);
    try std.testing.expect(try pool.isComplete(generation));
    try std.testing.expectEqual(4, counter.load(.acquire));
}
