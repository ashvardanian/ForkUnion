//! Demo app: Connected Components by label propagation, with ForkUnion and `std.Io.Group`.
//!
//! The N-body simulation gives every task an identical cost, so it can only measure dispatch latency.
//! Label propagation is the opposite end of fork-join usage: one parallel sweep per round, repeated
//! until no label changes - so a single pass pays the dispatch-and-join tax once per round, and the
//! graph's topology decides how many rounds there are.
//!
//! The generator strings `C` independent R-MAT communities on a ring, joined by one bridge edge per
//! neighbouring pair. The global minimum label must walk the ring, so convergence takes O(C) rounds
//! while each round stays a bandwidth-bound sweep - the fork-join frequency is the controlled axis.
//!
//! The labels are double-buffered: every round reads the immutable previous array and each vertex
//! writes only its own slot in the next - no atomics, no races, and every round is a pure function of
//! the last. Rounds-to-convergence, every intermediate label, and the final fixed point are therefore
//! identical across schedules, backends, thread counts, and languages - the C++, Rust, and this Zig
//! port all print the same component count, round count, and checksum.
//!
//! Environment variables:
//! - PROPAGATION_SCALE: each community has 2^scale vertices (default: 14)
//! - PROPAGATION_COMMUNITIES: communities strung on the ring (default: 64)
//! - PROPAGATION_EDGE_FACTOR: edges generated per vertex, before deduplication (default: 16)
//! - PROPAGATION_BACKEND: one of the backend names below (default: forkunion_static_shared)
//! - PROPAGATION_THREADS: number of threads (default: CPU count)
//! - PROPAGATION_SECONDS: wall-clock budget per run, reporting the sustained rate (default: 10)
//! - PROPAGATION_ITERATIONS: run an exact pass count instead, when set
//! - PROPAGATION_CHECK: also converge serially, and fail unless labels and rounds agree exactly
//!
//! The ForkUnion backends are the four cells of forkunion_{static,dynamic}_{shared,replicated};
//! the baseline is std_io_group, the standard library's own fork-join answer. Cells run bare, with no pinning
//! environment; the residual spread on SMT machines is preemption - one delayed hyperthread stalls
//! every barrier of a pass - which the fixed window amortizes. The _replicated backends are a
//! deliberate non-win on this workload: the hot traffic is the shared label array every round must
//! see fresh, so replicating the read-only CSR pays nothing here, unlike N-body's replicated bodies.
//! Build and run from the scripts/ directory:
//!
//! ```sh
//! cd scripts
//! zig build -Doptimize=ReleaseFast
//! PROPAGATION_BACKEND=forkunion_static_shared ./zig-out/bin/forkunion_propagation
//! ```

const std = @import("std");
const fu = @import("forkunion");

/// Reads a process environment variable, or null if unset. Borrows libc's storage - no free needed.
fn envVar(name: [*:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |value| std.mem.span(value) else null;
}

/// Reads a string environment variable, or `fallback` if unset.
fn envString(name: [*:0]const u8, fallback: []const u8) []const u8 {
    return envVar(name) orelse fallback;
}

/// Parses an unsigned environment variable, falling back silently on absence or a bad value.
fn envUsize(name: [*:0]const u8, fallback: usize) usize {
    if (envVar(name)) |text| return std.fmt.parseInt(usize, text, 10) catch fallback;
    return fallback;
}

/// Parses a fractional environment variable, falling back silently on absence or a bad value.
fn envF64(name: [*:0]const u8, fallback: f64) f64 {
    if (envVar(name)) |text| return std.fmt.parseFloat(f64, text) catch fallback;
    return fallback;
}

/// Whether an environment variable is present at all.
fn envFlag(name: [*:0]const u8) bool {
    return envVar(name) != null;
}

/// Reads the monotonic clock in nanoseconds, for timing the pass loop.
fn monotonicNanos() u64 {
    var timespec: std.posix.timespec = undefined;
    _ = std.posix.system.clock_gettime(.MONOTONIC, &timespec);
    return @as(u64, @intCast(timespec.sec)) * std.time.ns_per_s + @as(u64, @intCast(timespec.nsec));
}

/// Writes one preformatted result line to STDOUT; diagnostics stay on STDERR via `std.debug.print`.
fn writeStdout(text: []const u8) void {
    _ = std.c.write(std.Io.File.stdout().handle, text.ptr, text.len);
}

// Graph generation
//
// A counter-based SplitMix64 instead of a stateful generator: each draw is a pure function of its
// counter, so iterations are order-free, the fill parallelizes without sharding generator state, and
// the graph is bit-identical at any thread count - and across the C++, Rust, and Zig ports.

/// A component name: the smallest vertex index reachable so far.
const Label = u32;
/// Per-thread change tally, cache-aligned so two threads never share a line.
const Counter = fu.CacheAligned(u64);

const Edge = struct {
    row: u32,
    column: u32,

    fn lessThan(_: void, a: Edge, b: Edge) bool {
        return if (a.row != b.row) a.row < b.row else a.column < b.column;
    }
    fn equals(a: Edge, b: Edge) bool {
        return a.row == b.row and a.column == b.column;
    }
};

/// Sorts past every valid edge; marks dropped self-loops, trimmed together with the duplicates.
const sentinel_edge = Edge{ .row = std.math.maxInt(u32), .column = std.math.maxInt(u32) };

/// The SplitMix64 avalanche behind every random draw - a pure function of the `counter`.
inline fn splitMix(counter: u64) u64 {
    var x = (counter +% 1) *% 0x9E37_79B9_7F4A_7C15;
    x = (x ^ (x >> 30)) *% 0xBF58_476D_1CE4_E5B9;
    x = (x ^ (x >> 27)) *% 0x94D0_49BB_1331_11EB;
    return x ^ (x >> 31);
}

/// One quadrant choice in `[0, 100)` - same draw and counter scheme as every sibling benchmark.
inline fn randomPercent(counter: u64) u32 {
    return @intCast(splitMix(counter) % 100);
}

/// One bridge endpoint in `[0, bound)`, from the same avalanche.
inline fn randomIndex(counter: u64, bound: u32) u32 {
    return @intCast(splitMix(counter) % bound);
}

/// A read-only CSR as two slices - the interface every kernel takes.
const CsrView = struct {
    row_offsets: []const u64,
    column_indices: []const u32,

    fn vertices(self: CsrView) u32 {
        return @intCast(self.row_offsets.len - 1);
    }
};

/// The two CSR arrays built once on the host; `page_allocator` backs them so `retouch` sees virgin pages.
const CsrHost = struct {
    row_offsets: []u64,
    column_indices: []u32,

    fn view(self: *const CsrHost) CsrView {
        return .{ .row_offsets = self.row_offsets, .column_indices = self.column_indices };
    }
};

/// Everything the parallel R-MAT fill reads or writes; one entry per raw edge.
const FillContext = struct {
    slots: [*]Edge,
    scale: usize,
    raw_local: usize,
};

/// Fills raw edge `e`'s pair of slots `2e, 2e+1` from its quadrant walk; self-loops stay sentinels.
fn fillEdge(context: *const FillContext, task: usize, at: fu.ThreadInDomain) void {
    _ = at;
    const e = task;
    var row: u32 = 0;
    var column: u32 = 0;
    var bit: usize = context.scale;
    while (bit > 0) {
        bit -= 1;
        const r = randomPercent(@as(u64, @intCast(e)) * 64 + @as(u64, @intCast(bit))); // a=57 b=19 c=19 d=5
        const step = @as(u32, 1) << @intCast(bit);
        if (r < 57) continue; // Stay in the dense quadrant
        if (r < 76) {
            column |= step;
        } else if (r < 95) {
            row |= step;
        } else {
            row |= step;
            column |= step;
        }
    }
    if (row != column) {
        const base: u32 = @intCast((e / context.raw_local) << @intCast(context.scale)); // This community's range
        context.slots[e * 2] = .{ .row = base + row, .column = base + column }; // Symmetrize
        context.slots[e * 2 + 1] = .{ .row = base + column, .column = base + row };
    }
}

/// Generates the necklace: `communities` independent R-MAT graphs of `2^scale` vertices, joined in a
/// ring by one bridge per neighbouring pair, and scatters it all into a CSR.
///
/// Community `c` owns global edge indices `[c * raw_local, (c+1) * raw_local)` and the vertex range
/// `[c << scale, (c+1) << scale)`; the quadrant walk uses the same `e * 64 + bit` counters as the
/// single-graph generators. Bridge draws live in their own counter range above all edge draws.
fn generateNecklace(pool: fu.Pool, scale: usize, communities: usize, edge_factor: usize) !CsrHost {
    const allocator = std.heap.page_allocator;
    const community_vertices = @as(usize, 1) << @intCast(scale);
    const vertices = communities * community_vertices;
    const raw_local = community_vertices * edge_factor;
    const raw_edges = communities * raw_local;
    const bridges: usize = if (communities > 1) communities else 0;

    // The pair of slots `2e, 2e+1` belongs to edge `e`; self-loops stay sentinels.
    var edges = try allocator.alloc(Edge, raw_edges * 2 + bridges * 2);
    defer allocator.free(edges);
    @memset(edges, sentinel_edge);
    const fill = FillContext{ .slots = edges.ptr, .scale = scale, .raw_local = raw_local };
    try pool.forN(raw_edges, &fill, fillEdge);

    // Bridges: endpoints in each community's first 64 vertices - R-MAT's quadrant bias piles the
    // hubs at low indices, so a low endpoint is essentially guaranteed well-connected.
    const hub_core: u32 = @intCast(@min(community_vertices, 64));
    const bridge_base = @as(u64, @intCast(raw_edges)) * 64;
    for (0..bridges) |j| {
        const u = (@as(u32, @intCast(j << @intCast(scale)))) + randomIndex(bridge_base + 2 * @as(u64, @intCast(j)), hub_core);
        const v = (@as(u32, @intCast(((j + 1) % communities) << @intCast(scale)))) +
            randomIndex(bridge_base + 2 * @as(u64, @intCast(j)) + 1, hub_core);
        edges[raw_edges * 2 + j * 2] = .{ .row = u, .column = v };
        edges[raw_edges * 2 + j * 2 + 1] = .{ .row = v, .column = u };
    }

    std.mem.sort(Edge, edges, {}, Edge.lessThan);

    // Dedup in place; every sentinel sorts to the tail and at most one survives, trimmed with the rest.
    var unique_count: usize = 0;
    for (edges, 0..) |edge, i| {
        if (i > 0 and Edge.equals(edge, edges[unique_count - 1])) continue;
        edges[unique_count] = edge;
        unique_count += 1;
    }
    while (unique_count > 0 and Edge.equals(edges[unique_count - 1], sentinel_edge)) unique_count -= 1;
    const live = edges[0..unique_count];

    // CSR: count the degrees into `row_offsets`, then prefix-sum them into row starts.
    var row_offsets = try allocator.alloc(u64, vertices + 1);
    @memset(row_offsets, 0);
    for (live) |edge| row_offsets[edge.row + 1] += 1;
    for (0..vertices) |v| row_offsets[v + 1] += row_offsets[v];
    var column_indices = try allocator.alloc(u32, live.len);
    var cursor = try allocator.alloc(u64, vertices);
    defer allocator.free(cursor);
    @memcpy(cursor, row_offsets[0..vertices]);
    for (live) |edge| {
        column_indices[cursor[edge.row]] = edge.column;
        cursor[edge.row] += 1;
    }
    return .{ .row_offsets = row_offsets, .column_indices = column_indices };
}

// Deterministic first touch
//
// Generation first-touches pages on whichever cores ran the fill, so every process rolls a different
// page placement and throughput swings run to run. Copying into virgin pages from the static split of
// PINNED threads makes placement a pure function of the topology - identical for every backend,
// process, and language.

/// Everything one retouch copy reads or writes; `forSlices` hands each thread one contiguous range.
fn RetouchContext(comptime T: type) type {
    return struct {
        source: [*]const T,
        destination: [*]T,
    };
}

/// Rewrites `values` into fresh pages, first-touched by the pinned pool's static split.
fn retouchDeterministically(comptime T: type, pool: fu.Pool, values: *[]T) !void {
    const allocator = std.heap.page_allocator;
    const placed = try allocator.alloc(T, values.len); // ? Pages stay unfaulted until the copy below
    const Context = RetouchContext(T);
    const context = Context{ .source = values.ptr, .destination = placed.ptr };
    try pool.forSlices(values.len, &context, struct {
        fn copy(carried: *const Context, range: fu.TasksRange, at: fu.ThreadInDomain) void {
            _ = at;
            const first = range.first;
            @memcpy(carried.destination[first..][0..range.count], carried.source[first..][0..range.count]);
        }
    }.copy);
    allocator.free(values.*);
    values.* = placed;
}

// Label-propagation kernels

/// The smallest label visible from `v`: its own, or the smallest among its neighbours'.
inline fn minLabelOf(row_offsets: [*]const u64, column_indices: [*]const u32, old_labels: [*]const Label, v: usize) Label {
    var best = old_labels[v];
    const begin = row_offsets[v];
    const end = row_offsets[v + 1];
    var position = begin;
    while (position < end) : (position += 1) {
        const candidate = old_labels[column_indices[position]];
        if (candidate < best) best = candidate;
    }
    return best;
}

// Compile-time axes - Zig takes real enums as comptime parameters.
const Schedule = enum { static, dynamic };
const Placement = enum { shared, replicated };

/// Everything one round of any ForkUnion backend reads or writes; the comptime placement elides the rest.
const WorkContext = struct {
    row_offsets: [*]const u64,
    column_indices: [*]const u32,
    old_labels: [*]const Label,
    new_labels: [*]Label,
    counters: [*]Counter,
    /// Each compute domain's nearest memory domain, probed once so the kernel never calls the FFI.
    local_memory: []const fu.MemoryDomain,
    replicas_offsets: ?*fu.ReplicatedArray(u64),
    replicas_columns: ?*fu.ReplicatedArray(u32),
};

/// One vertex's update: read the immutable previous labels, write only your own slot.
fn labelKernel(comptime placement: Placement) fn (*const WorkContext, usize, fu.ThreadInDomain) void {
    return struct {
        fn update(work: *const WorkContext, task: usize, at: fu.ThreadInDomain) void {
            const v = task;
            var row_offsets = work.row_offsets;
            var column_indices = work.column_indices;
            if (placement == .replicated) {
                const memory_domain = work.local_memory[at.compute_domain.index()];
                row_offsets = work.replicas_offsets.?.onMemoryDomain(memory_domain).ptr;
                column_indices = work.replicas_columns.?.onMemoryDomain(memory_domain).ptr;
            }
            const next = minLabelOf(row_offsets, column_indices, work.old_labels, v);
            work.new_labels[v] = next;
            work.counters[at.thread].value += @intFromBool(next != work.old_labels[v]);
        }
    }.update;
}

/// Dispatches the round on the chosen schedule: pre-divided static, or work-stolen dynamic.
fn forNScheduled(comptime schedule: Schedule, pool: fu.Pool, n: usize, comptime kernel: anytype, work: *const WorkContext) !void {
    if (schedule == .static) try pool.forN(n, work, kernel) else try pool.forNDynamic(n, work, kernel);
}

/// One convergence pass on the ForkUnion pool; every round is one fork-join dispatch.
fn runForkUnion(
    comptime schedule: Schedule,
    comptime placement: Placement,
    pool: fu.Pool,
    local_memory: []const fu.MemoryDomain,
    graph: CsrView,
    labels_a: []Label,
    labels_b: []Label,
    counters: []Counter,
    replicas_offsets: ?*fu.ReplicatedArray(u64),
    replicas_columns: ?*fu.ReplicatedArray(u32),
) !usize {
    const vertices = graph.vertices();
    for (labels_a, 0..) |*label, v| label.* = @intCast(v);

    var rounds: usize = 0;
    var old_labels = labels_a;
    var new_labels = labels_b;
    while (true) {
        for (counters) |*counter| counter.value = 0;
        const work = WorkContext{
            .row_offsets = graph.row_offsets.ptr,
            .column_indices = graph.column_indices.ptr,
            .old_labels = old_labels.ptr,
            .new_labels = new_labels.ptr,
            .counters = counters.ptr,
            .local_memory = local_memory,
            .replicas_offsets = replicas_offsets,
            .replicas_columns = replicas_columns,
        };
        try forNScheduled(schedule, pool, vertices, labelKernel(placement), &work);
        rounds += 1;
        var changes: u64 = 0;
        for (counters) |*counter| changes += counter.value;
        std.mem.swap([]Label, &old_labels, &new_labels);
        if (changes == 0) break;
    }
    return rounds;
}

// std.Io.Group backend (static work division)
// The standard library's fork-join answer since 0.16 removed `std.Thread.Pool`: one `Io.Group` per
// round, one task per vertex slice, awaited before the labels swap.

fn runStdIoGroup(allocator: std.mem.Allocator, io: std.Io, graph: CsrView, labels_a: []Label, labels_b: []Label, n_threads: usize) !usize {
    const vertices: usize = graph.vertices();
    const changes_per_thread = try allocator.alloc(Counter, n_threads);
    defer allocator.free(changes_per_thread);
    const chunk = std.math.divCeil(usize, vertices, n_threads) catch unreachable;

    for (labels_a, 0..) |*label, v| label.* = @intCast(v);
    var rounds: usize = 0;
    var old_labels = labels_a;
    var new_labels = labels_b;
    while (true) {
        for (changes_per_thread) |*counter| counter.value = 0;
        var group: std.Io.Group = .init;
        defer group.cancel(io);
        for (0..n_threads) |thread_id| {
            const start = thread_id * chunk;
            if (start >= vertices) break;
            const end = @min(start + chunk, vertices);
            group.async(io, struct {
                fn sweep(g: CsrView, old: []const Label, new: []Label, tally: *Counter, range_start: usize, range_end: usize) void {
                    var changed: u64 = 0;
                    for (range_start..range_end) |v| {
                        const next = minLabelOf(g.row_offsets.ptr, g.column_indices.ptr, old.ptr, v);
                        new[v] = next;
                        changed += @intFromBool(next != old[v]);
                    }
                    tally.value = changed;
                }
            }.sweep, .{ graph, old_labels, new_labels, &changes_per_thread[thread_id], start, end });
        }
        try group.await(io);
        rounds += 1;
        var changes: u64 = 0;
        for (changes_per_thread) |*counter| changes += counter.value;
        std.mem.swap([]Label, &old_labels, &new_labels);
        if (changes == 0) break;
    }
    return rounds;
}

/// Converges serially from `labels[v] = v`, returning the rounds taken - the reference for the CHECK.
fn convergeSerially(graph: CsrView, labels_a: []Label, labels_b: []Label) usize {
    const vertices: usize = graph.vertices();
    for (labels_a, 0..) |*label, v| label.* = @intCast(v);
    var rounds: usize = 0;
    var old_labels = labels_a;
    var new_labels = labels_b;
    while (true) {
        var changed: u64 = 0;
        for (0..vertices) |v| {
            const next = minLabelOf(graph.row_offsets.ptr, graph.column_indices.ptr, old_labels.ptr, v);
            new_labels[v] = next;
            changed += @intFromBool(next != old_labels[v]);
        }
        rounds += 1;
        std.mem.swap([]Label, &old_labels, &new_labels);
        if (changed == 0) break;
    }
    return rounds;
}

// Harness

const Backend = enum {
    forkunion_static_shared,
    forkunion_dynamic_shared,
    forkunion_static_replicated,
    forkunion_dynamic_replicated,
    std_io_group,
};

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const scale = envUsize("PROPAGATION_SCALE", 14);
    const communities = envUsize("PROPAGATION_COMMUNITIES", 64);
    const edge_factor = envUsize("PROPAGATION_EDGE_FACTOR", 16);
    const backend_name = envString("PROPAGATION_BACKEND", "forkunion_static_shared");
    const budget_seconds = envF64("PROPAGATION_SECONDS", 10); // The primary knob: a fixed window
    const n_iters = envUsize("PROPAGATION_ITERATIONS", 0); // Overrides with an exact count when set
    const check = envFlag("PROPAGATION_CHECK");

    const backend = std.meta.stringToEnum(Backend, backend_name) orelse {
        std.debug.print("Unsupported backend: '{s}'\n", .{backend_name});
        inline for (@typeInfo(Backend).@"enum".fields) |field| std.debug.print("  {s}\n", .{field.name});
        return error.UnsupportedBackend;
    };

    const topology = try fu.Topology.init();
    defer topology.deinit();
    var n_threads = envUsize("PROPAGATION_THREADS", 0);
    if (n_threads == 0) n_threads = try topology.logicalCoresCount();

    // One pinned pool spawns for EVERY backend - first to give the graph and label pages their
    // deterministic first touch, then to serve the ForkUnion backends; std_io_group ignores it.
    const pool = try fu.Pool.init(topology, .{ .threads = n_threads, .name = "fu-propagate" });
    defer pool.deinit();

    // The standard library's executor, sized to match: past `async_limit` an `Io.async` runs the
    // task inline on the caller, so N-1 workers plus the caller mirrors an inclusive pool of N.
    var threaded: std.Io.Threaded = .init(allocator, .{ .async_limit = .limited(n_threads - 1) });
    defer threaded.deinit();
    const io = threaded.io();

    var host = try generateNecklace(pool, scale, communities, edge_factor);
    const vertices: usize = @as(usize, host.row_offsets.len - 1);
    var labels_a = try allocator.alloc(Label, vertices);
    var labels_b = try allocator.alloc(Label, vertices);
    @memset(labels_a, 0);
    @memset(labels_b, 0);
    try retouchDeterministically(u64, pool, &host.row_offsets);
    try retouchDeterministically(u32, pool, &host.column_indices);
    try retouchDeterministically(Label, pool, &labels_a);
    try retouchDeterministically(Label, pool, &labels_b);
    const graph = host.view();

    var line_buffer: [256]u8 = undefined;
    const header = std.fmt.bufPrint(&line_buffer, "vertices {}, directed edges {}, communities {}\n", .{
        vertices, graph.column_indices.len, communities,
    }) catch return;
    writeStdout(header);

    // Per-node CSR replicas for the `_replicated` cells; only the immutable CSR replicates - the
    // label buffers stay shared by nature, since every round must see every neighbour's last label.
    var replicas_offsets: ?fu.ReplicatedArray(u64) = null;
    var replicas_columns: ?fu.ReplicatedArray(u32) = null;
    defer if (replicas_offsets) |*r| r.deinit();
    defer if (replicas_columns) |*r| r.deinit();
    if (backend == .forkunion_static_replicated or backend == .forkunion_dynamic_replicated) {
        replicas_offsets = try fu.ReplicatedArray(u64).init(topology, graph.row_offsets.len);
        replicas_columns = try fu.ReplicatedArray(u32).init(topology, graph.column_indices.len);
        for (0..try topology.memoryDomainsCount()) |domain| {
            const memory_domain = fu.MemoryDomain.at(domain);
            @memcpy(replicas_offsets.?.onMemoryDomain(memory_domain), graph.row_offsets);
            @memcpy(replicas_columns.?.onMemoryDomain(memory_domain), graph.column_indices);
        }
    }

    const counters = try allocator.alloc(Counter, n_threads);
    defer allocator.free(counters);

    // Probed once here, so the per-vertex kernel indexes a slice instead of crossing the FFI.
    const local_memory = try allocator.alloc(fu.MemoryDomain, try topology.computeDomainsCount());
    defer allocator.free(local_memory);
    for (local_memory, 0..) |*slot, compute_domain|
        slot.* = try topology.localMemoryOf(fu.ComputeDomain.at(compute_domain));

    const runOnce = struct {
        fn call(b: Backend, p: fu.Pool, alloc: std.mem.Allocator, standard_io: std.Io, locals: []const fu.MemoryDomain, g: CsrView, a: []Label, bb: []Label, c: []Counter, ro: ?*fu.ReplicatedArray(u64), rc: ?*fu.ReplicatedArray(u32), threads: usize) !usize {
            return switch (b) {
                .forkunion_static_shared => try runForkUnion(.static, .shared, p, locals, g, a, bb, c, ro, rc),
                .forkunion_dynamic_shared => try runForkUnion(.dynamic, .shared, p, locals, g, a, bb, c, ro, rc),
                .forkunion_static_replicated => try runForkUnion(.static, .replicated, p, locals, g, a, bb, c, ro, rc),
                .forkunion_dynamic_replicated => try runForkUnion(.dynamic, .replicated, p, locals, g, a, bb, c, ro, rc),
                .std_io_group => try runStdIoGroup(alloc, standard_io, g, a, bb, threads),
            };
        }
    }.call;

    const ro_ptr: ?*fu.ReplicatedArray(u64) = if (replicas_offsets) |*r| r else null;
    const rc_ptr: ?*fu.ReplicatedArray(u32) = if (replicas_columns) |*r| r else null;

    // One untimed warmup pass: page-faults and cache warming would otherwise bias the first timed
    // pass, and by a different amount for each backend.
    var rounds = try runOnce(backend, pool, allocator, io, local_memory, graph, labels_a, labels_b, counters, ro_ptr, rc_ptr, n_threads);

    // A fixed time budget beats a fixed pass count: every backend runs the same wall-clock window -
    // long enough to amortize scheduling noise - and reports the rate it sustained, with no
    // per-backend pass-count guessing. PROPAGATION_ITERATIONS forces an exact count instead.
    const budget_ns: u64 = @intFromFloat(budget_seconds * std.time.ns_per_s);
    const started = monotonicNanos();
    var passes: usize = 0;
    if (n_iters > 0) {
        for (0..n_iters) |_| rounds = try runOnce(backend, pool, allocator, io, local_memory, graph, labels_a, labels_b, counters, ro_ptr, rc_ptr, n_threads);
        passes = n_iters;
    } else {
        while (true) {
            rounds = try runOnce(backend, pool, allocator, io, local_memory, graph, labels_a, labels_b, counters, ro_ptr, rc_ptr, n_threads);
            passes += 1;
            if (monotonicNanos() - started >= budget_ns) break;
        }
    }
    const seconds = @as(f64, @floatFromInt(monotonicNanos() - started)) / std.time.ns_per_s / @as(f64, @floatFromInt(passes));

    // The fixed point sits in both buffers - the terminal round changed nothing - so read either.
    var components: u64 = 0;
    var checksum: u64 = 0;
    for (labels_a, 0..) |label, v| {
        components += @intFromBool(label == @as(Label, @intCast(v)));
        checksum +%= label;
    }
    // MTEPS - millions of directed edges scanned per second; `rounds * edges` is the exact scan
    // count, identical in every cell by the double-buffered determinism.
    const mteps = @as(f64, @floatFromInt(rounds)) * @as(f64, @floatFromInt(graph.column_indices.len)) / seconds / 1e6;
    const line = std.fmt.bufPrint(&line_buffer, "{s}: {} components, {} rounds, checksum {}, {d:.2} s/pass, {d:.1} MTEPS\n", .{
        backend_name, components, rounds, checksum, seconds, mteps,
    }) catch return;
    writeStdout(line);

    if (check) {
        const serial_a = try allocator.alloc(Label, vertices);
        defer allocator.free(serial_a);
        const serial_b = try allocator.alloc(Label, vertices);
        defer allocator.free(serial_b);
        const serial_rounds = convergeSerially(graph, serial_a, serial_b);
        if (serial_rounds != rounds or !std.mem.eql(Label, serial_a, labels_a)) {
            std.debug.print("MISMATCH: serial converged in {} rounds\n", .{serial_rounds});
            return error.SerialMismatch;
        }
        writeStdout("check: matches the serial labels and rounds\n");
    }
}
