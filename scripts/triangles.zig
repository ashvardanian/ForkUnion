//! Triangle counting on a power-law graph, with ForkUnion and raw std threads.
//!
//! The N-body simulation gives every task an identical cost, so it can only measure dispatch latency.
//! Triangle counting gives them wildly different costs: the work at a vertex grows with its degree and
//! with the degrees of its neighbours, and an R-MAT graph draws those degrees from a power law. A
//! handful of hub vertices carry most of the arithmetic, and R-MAT clusters them at low indices, so a
//! static split hands one thread nearly all of them.
//!
//! `forkunion_vertex_centric_static` and `forkunion_vertex_centric_dynamic` put one task per vertex;
//! `forkunion_vertex_centric_static` is hopeless because the slice holding the hubs decides the makespan.
//! `forkunion_edge_centric` flattens the adjacency into a tape of one item per `u < v` edge, weighs each
//! item by `degree(u) + degree(v)`, prefix-sums the weights, and cuts the cost axis into equal parts - a
//! static dispatch that beats a stealing one. `forkunion_edge_centric_replicated` gives every memory domain
//! its own read-only replica of the CSR, so no thread reaches across the interconnect for a neighbour list.
//!
//! Environment variables:
//! - TRIANGLES_SCALE: the graph has `2^scale` vertices (default 18)
//! - TRIANGLES_EDGE_FACTOR: edges generated per vertex, before deduplication (default 16)
//! - TRIANGLES_BACKEND: one of the backend names below (default forkunion_vertex_centric_static)
//! - TRIANGLES_THREADS: number of threads (default all hardware threads)
//! - TRIANGLES_ITERATIONS: repeat the count this many times, reporting the per-pass time (default 1)
//! - TRIANGLES_CHECK: also count serially, and fail unless the totals agree
//!
//! The backends include: forkunion_vertex_centric_static, forkunion_vertex_centric_dynamic, forkunion_edge_centric,
//! forkunion_edge_centric_replicated, std_threads. Build and run from the scripts/ directory:
//! ```sh
//! cd scripts
//! zig build -Doptimize=ReleaseFast
//! time TRIANGLES_SCALE=20 TRIANGLES_BACKEND=forkunion_edge_centric ./zig-out/bin/triangles
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

/// Whether an environment variable is present at all.
fn envFlag(name: [*:0]const u8) bool {
    return envVar(name) != null;
}

/// Reads the monotonic clock in nanoseconds, for timing one counting pass.
fn monotonicNanos() u64 {
    var timespec: std.posix.timespec = undefined;
    _ = std.posix.system.clock_gettime(.MONOTONIC, &timespec);
    return @as(u64, @intCast(timespec.sec)) * std.time.ns_per_s + @as(u64, @intCast(timespec.nsec));
}

/// Writes one preformatted line to STDOUT; diagnostics stay on STDERR via `std.debug.print`.
fn writeStdout(text: []const u8) void {
    _ = std.c.write(std.Io.File.stdout().handle, text.ptr, text.len);
}

/// Per-thread tally, cache-aligned so two threads never share a line.
const Counter = struct {
    value: u64 align(64) = 0,
};

/// A read-only CSR-plus-tape as five slices - the interface every kernel takes.
///
/// `row_offsets` and `column_indices` are the usual CSR pair, each adjacency sorted ascending.
/// `above_offsets[u]` is where the neighbours greater than `u` begin within `u`'s row, `tape_offsets[u]`
/// is the prefix sum of tape items before `u`, and `work_offsets` is the prefix sum of each item's cost.
/// The view is agnostic to where the bytes live - the host build, or a replica.
const CsrView = struct {
    row_offsets: []const u64,
    column_indices: []const u32,
    tape_offsets: []const u64,
    above_offsets: []const u32,
    work_offsets: []const u64,

    fn vertices(self: CsrView) usize {
        return self.row_offsets.len - 1;
    }

    fn edges(self: CsrView) usize {
        return self.column_indices.len;
    }

    fn degree(self: CsrView, v: usize) u32 {
        return @intCast(self.row_offsets[v + 1] - self.row_offsets[v]);
    }

    /// Number of `u < v` pairs; the length of the work tape.
    fn tapeLength(self: CsrView) u64 {
        return self.tape_offsets[self.tape_offsets.len - 1];
    }

    /// Total cost of the tape, in units of "one adjacency element compared".
    fn totalWork(self: CsrView) u64 {
        return self.work_offsets[self.work_offsets.len - 1];
    }

    /// The first tape item whose cumulative cost reaches `work` - the merge-path search.
    fn itemAtWork(self: CsrView, work: u64) u64 {
        var low: usize = 0;
        var high: usize = self.work_offsets.len;
        while (low < high) {
            const mid = low + (high - low) / 2;
            if (self.work_offsets[mid] < work) low = mid + 1 else high = mid;
        }
        return low;
    }

    /// The vertex owning tape item `item`, by binary search into the prefix sum.
    fn ownerOf(self: CsrView, item: u64) u32 {
        var low: usize = 0;
        var high: usize = self.tape_offsets.len;
        while (low < high) {
            const mid = low + (high - low) / 2;
            if (self.tape_offsets[mid] <= item) low = mid + 1 else high = mid;
        }
        return @intCast(low - 1);
    }
};

/// The five CSR-plus-tape arrays built once on the host.
const CsrHost = struct {
    row_offsets: []u64,
    column_indices: []u32,
    tape_offsets: []u64,
    above_offsets: []u32,
    work_offsets: []u64,
    allocator: std.mem.Allocator,

    fn deinit(self: *CsrHost) void {
        self.allocator.free(self.row_offsets);
        self.allocator.free(self.column_indices);
        self.allocator.free(self.tape_offsets);
        self.allocator.free(self.above_offsets);
        self.allocator.free(self.work_offsets);
    }

    fn view(self: CsrHost) CsrView {
        return .{
            .row_offsets = self.row_offsets,
            .column_indices = self.column_indices,
            .tape_offsets = self.tape_offsets,
            .above_offsets = self.above_offsets,
            .work_offsets = self.work_offsets,
        };
    }
};

/// The upper bound of `key` in the sorted slice - the first index whose element is greater than `key`.
fn upperBound(slice: []const u32, key: u32) usize {
    var low: usize = 0;
    var high: usize = slice.len;
    while (low < high) {
        const mid = low + (high - low) / 2;
        if (slice[mid] <= key) low = mid + 1 else high = mid;
    }
    return low;
}

/// Generates a Kronecker/R-MAT graph, as specified by Graph500, and builds its tape.
///
/// Recursing into the `a` quadrant with probability 57% is what makes the degrees power-law, and what
/// keeps the hubs near index zero, where a static split will trip over them.
fn generateRmat(allocator: std.mem.Allocator, scale: usize, edge_factor: usize) !CsrHost {
    const vertices = @as(usize, 1) << @as(u6, @intCast(scale));
    const raw_edges = vertices * edge_factor;

    // Build the COO edge list, packing each `(row, column)` into one u64 so a plain sort is lexicographic.
    var edges = try allocator.alloc(u64, raw_edges * 2);
    defer allocator.free(edges);
    var edge_count: usize = 0;

    var generator = std.Random.DefaultPrng.init(0x1234_5678_9abc_def0);
    const random = generator.random();
    for (0..raw_edges) |_| {
        var row: u32 = 0;
        var column: u32 = 0;
        var bit: usize = scale;
        while (bit > 0) {
            bit -= 1;
            const r = random.uintLessThan(u8, 100); // a=57 b=19 c=19 d=5, integer and portable
            const step = @as(u32, 1) << @as(u5, @intCast(bit));
            if (r < 57) {
                continue; // Stay in the dense quadrant
            } else if (r < 76) {
                column |= step;
            } else if (r < 95) {
                row |= step;
            } else {
                row |= step;
                column |= step;
            }
        }
        if (row == column) continue; // Drop self-loops
        edges[edge_count] = (@as(u64, row) << 32) | column; // Symmetrize
        edge_count += 1;
        edges[edge_count] = (@as(u64, column) << 32) | row;
        edge_count += 1;
    }

    std.mem.sort(u64, edges[0..edge_count], {}, comptime std.sort.asc(u64));
    var unique: usize = 0;
    for (edges[0..edge_count], 0..) |edge, i| {
        if (i == 0 or edge != edges[unique - 1]) {
            edges[unique] = edge;
            unique += 1;
        }
    }
    edge_count = unique;

    // CSR: count the degrees into `row_offsets`, then prefix-sum them into row starts.
    const row_offsets = try allocator.alloc(u64, vertices + 1);
    @memset(row_offsets, 0);
    for (edges[0..edge_count]) |edge| {
        const row: usize = @intCast(edge >> 32);
        row_offsets[row + 1] += 1;
    }
    for (0..vertices) |v| row_offsets[v + 1] += row_offsets[v];

    const column_indices = try allocator.alloc(u32, edge_count);
    const cursor = try allocator.alloc(u64, vertices);
    defer allocator.free(cursor);
    for (0..vertices) |v| cursor[v] = row_offsets[v];
    for (edges[0..edge_count]) |edge| {
        const row: usize = @intCast(edge >> 32);
        column_indices[@intCast(cursor[row])] = @truncate(edge);
        cursor[row] += 1;
    }

    // Tape: one item per `u < v` pair, in vertex order.
    const above_offsets = try allocator.alloc(u32, vertices);
    const tape_offsets = try allocator.alloc(u64, vertices + 1);
    tape_offsets[0] = 0;
    for (0..vertices) |u| {
        const start: usize = @intCast(row_offsets[u]);
        const end: usize = @intCast(row_offsets[u + 1]);
        const row = column_indices[start..end];
        const above = upperBound(row, @intCast(u)); // First neighbour greater than `u`
        above_offsets[u] = @intCast(above);
        tape_offsets[u + 1] = tape_offsets[u] + @as(u64, row.len - above);
    }

    // Weigh each tape item by the two adjacencies it intersects, so balancing this weight spreads the
    // hubs that balancing the bare item count would bunch into one slice.
    const tape_length: usize = @intCast(tape_offsets[vertices]);
    const work_offsets = try allocator.alloc(u64, tape_length + 1);
    work_offsets[0] = 0;
    for (0..vertices) |u| {
        const u_degree = row_offsets[u + 1] - row_offsets[u];
        var position: usize = @intCast(row_offsets[u] + above_offsets[u]);
        var item: usize = @intCast(tape_offsets[u]);
        const item_end: usize = @intCast(tape_offsets[u + 1]);
        while (item < item_end) : ({
            item += 1;
            position += 1;
        }) {
            const v: usize = column_indices[position];
            const v_degree = row_offsets[v + 1] - row_offsets[v];
            work_offsets[item + 1] = work_offsets[item] + u_degree + v_degree;
        }
    }

    return .{
        .row_offsets = row_offsets,
        .column_indices = column_indices,
        .tape_offsets = tape_offsets,
        .above_offsets = above_offsets,
        .work_offsets = work_offsets,
        .allocator = allocator,
    };
}

/// Counts triangles `u < v < w` for the single edge at tape item `item`, whose owner is `owner`.
///
/// Both adjacencies are sorted, so the candidates for `w` are the suffix of `N(u)` past `v` and the
/// suffix of `N(v)` past `v`. Intersecting them counts each triangle exactly once, and needs no atomics.
inline fn countTrianglesOnItem(graph: *const CsrView, item: u64, owner: u32) u64 {
    const columns = graph.column_indices;
    const owner_index: usize = owner;
    const owner_end: usize = @intCast(graph.row_offsets[owner_index + 1]);
    const position: usize = @intCast(graph.row_offsets[owner_index] + @as(u64, graph.above_offsets[owner_index]) + (item - graph.tape_offsets[owner_index]));

    const other: usize = columns[position];
    var a: usize = position + 1;
    const a_end: usize = owner_end;
    var b: usize = @intCast(graph.row_offsets[other] + @as(u64, graph.above_offsets[other]));
    const b_end: usize = @intCast(graph.row_offsets[other + 1]);

    var triangles: u64 = 0;
    while (a < a_end and b < b_end) {
        const column_a = columns[a];
        const column_b = columns[b];
        if (column_a < column_b) {
            a += 1;
        } else if (column_b < column_a) {
            b += 1;
        } else {
            triangles += 1;
            a += 1;
            b += 1;
        }
    }
    return triangles;
}

/// Counts every triangle whose smallest vertex is `u`. Cost grows with `degree(u)` squared.
inline fn countTrianglesAtVertex(graph: *const CsrView, u: u32) u64 {
    const u_index: usize = u;
    var triangles: u64 = 0;
    var item: u64 = graph.tape_offsets[u_index];
    const item_end: u64 = graph.tape_offsets[u_index + 1];
    while (item < item_end) : (item += 1) triangles += countTrianglesOnItem(graph, item, u);
    return triangles;
}

/// Walks a contiguous run of the tape, resolving the owning vertex only when it changes.
///
/// One binary search per slice, then a forward walk - the CPU spelling of a merge-path.
inline fn countTrianglesOnSlice(graph: *const CsrView, first: u64, count: u64) u64 {
    if (count == 0) return 0;
    var triangles: u64 = 0;
    var owner = graph.ownerOf(first);
    var item = first;
    while (item < first + count) : (item += 1) {
        while (item >= graph.tape_offsets[@as(usize, owner) + 1]) owner += 1; // Amortized O(1) per item
        triangles += countTrianglesOnItem(graph, item, owner);
    }
    return triangles;
}

/// One read-only replica of the CSR per memory domain, so no adjacency is ever remote.
const ReplicatedCsr = struct {
    row_offsets: fu.ReplicatedArray(u64),
    column_indices: fu.ReplicatedArray(u32),
    tape_offsets: fu.ReplicatedArray(u64),
    above_offsets: fu.ReplicatedArray(u32),
    work_offsets: fu.ReplicatedArray(u64),

    fn init(topology: fu.Topology, host: CsrHost) ?ReplicatedCsr {
        return .{
            .row_offsets = replicate(u64, topology, host.row_offsets) orelse return null,
            .column_indices = replicate(u32, topology, host.column_indices) orelse return null,
            .tape_offsets = replicate(u64, topology, host.tape_offsets) orelse return null,
            .above_offsets = replicate(u32, topology, host.above_offsets) orelse return null,
            .work_offsets = replicate(u64, topology, host.work_offsets) orelse return null,
        };
    }

    fn deinit(self: *ReplicatedCsr) void {
        self.row_offsets.deinit();
        self.column_indices.deinit();
        self.tape_offsets.deinit();
        self.above_offsets.deinit();
        self.work_offsets.deinit();
    }

    fn onMemoryDomain(self: ReplicatedCsr, memory_domain: usize) CsrView {
        return .{
            .row_offsets = self.row_offsets.onMemoryDomain(memory_domain),
            .column_indices = self.column_indices.onMemoryDomain(memory_domain),
            .tape_offsets = self.tape_offsets.onMemoryDomain(memory_domain),
            .above_offsets = self.above_offsets.onMemoryDomain(memory_domain),
            .work_offsets = self.work_offsets.onMemoryDomain(memory_domain),
        };
    }
};

/// Copies one host array into every per-domain replica, each slice landing on its own node.
fn replicate(comptime T: type, topology: fu.Topology, host: []const T) ?fu.ReplicatedArray(T) {
    var replica = fu.ReplicatedArray(T).init(topology, host.len) orelse return null;
    for (0..replica.memoryDomainsCount()) |domain| @memcpy(replica.onMemoryDomain(domain), host);
    return replica;
}

/// Counts a contiguous vertex stripe per raw std thread, storing each stripe's tally in its own counter.
fn stdThreadsCount(allocator: std.mem.Allocator, graph_view: CsrView, counters: []Counter, n_threads: usize) !void {
    const vertices = graph_view.vertices();
    const threads = try allocator.alloc(std.Thread, n_threads);
    defer allocator.free(threads);
    const chunk = (vertices + n_threads - 1) / n_threads;

    var spawned: usize = 0;
    for (0..n_threads) |thread_id| {
        const start = thread_id * chunk;
        if (start >= vertices) break;
        const end = @min(start + chunk, vertices);
        threads[spawned] = try std.Thread.spawn(.{}, struct {
            fn work(graph: CsrView, counter: *Counter, range_start: usize, range_end: usize) void {
                var sum: u64 = 0;
                for (range_start..range_end) |v| sum += countTrianglesAtVertex(&graph, @intCast(v));
                counter.value += sum;
            }
        }.work, .{ graph_view, &counters[thread_id], start, end });
        spawned += 1;
    }
    for (threads[0..spawned]) |t| t.join();
}

// Registry

/// Which execution engine a backend runs on, so `main` builds exactly the resource it needs.
const Engine = enum { forkunion, forkunion_replicated, std_threads };

/// Everything a backend reads or writes for one counting pass; `main` owns the lifetimes.
const Context = struct {
    graph: CsrView,
    replicas: ?*const ReplicatedCsr,
    topology: fu.Topology,
    pool: ?*fu.Pool,
    counters: []Counter,
    allocator: std.mem.Allocator,
    n_threads: usize,
};

/// The dispatch table - a name, its counting pass, and the engine it runs on.
const Backend = struct {
    name: []const u8,
    run: *const fn (*Context) void,
    engine: Engine,
};

fn runVertexCentricStatic(context: *Context) void {
    const WorkContext = struct { graph: CsrView, counters: [*]Counter };
    context.pool.?.forN(context.graph.vertices(), struct {
        fn calc(prong: fu.Prong, c: WorkContext) void {
            // Each thread owns one counter; tasks on one thread run in order, so the add is race-free.
            c.counters[prong.thread_index].value += countTrianglesAtVertex(&c.graph, @intCast(prong.task_index));
        }
    }.calc, WorkContext{ .graph = context.graph, .counters = context.counters.ptr });
}

fn runVertexCentricDynamic(context: *Context) void {
    const WorkContext = struct { graph: CsrView, counters: [*]Counter };
    context.pool.?.forNDynamic(context.graph.vertices(), struct {
        fn calc(prong: fu.Prong, c: WorkContext) void {
            c.counters[prong.thread_index].value += countTrianglesAtVertex(&c.graph, @intCast(prong.task_index));
        }
    }.calc, WorkContext{ .graph = context.graph, .counters = context.counters.ptr });
}

fn runEdgeCentric(context: *Context) void {
    const WorkContext = struct { graph: CsrView, counters: [*]Counter, threads: u64 };
    context.pool.?.forThreads(struct {
        fn work(thread_index: usize, compute_domain_index: usize, c: WorkContext) void {
            _ = compute_domain_index;
            const total = c.graph.totalWork();
            const thread: u64 = @intCast(thread_index);
            const first = c.graph.itemAtWork(total * thread / c.threads);
            const last = c.graph.itemAtWork(total * (thread + 1) / c.threads);
            c.counters[thread_index].value += countTrianglesOnSlice(&c.graph, first, last - first);
        }
    }.work, WorkContext{ .graph = context.graph, .counters = context.counters.ptr, .threads = @intCast(context.n_threads) });
}

fn runEdgeCentricReplicated(context: *Context) void {
    const WorkContext = struct {
        pool: *const fu.Pool,
        topology: fu.Topology,
        replicas: *const ReplicatedCsr,
        counters: [*]Counter,
    };
    context.pool.?.forThreads(struct {
        fn work(thread_index: usize, compute_domain_index: usize, c: WorkContext) void {
            const memory_domain = c.topology.localMemoryOf(compute_domain_index);
            const replica = c.replicas.onMemoryDomain(memory_domain);

            const domains: u64 = @intCast(c.pool.compute_domains());
            const threads_here: u64 = @intCast(c.pool.countThreadsIn(compute_domain_index));
            const local_index: u64 = @intCast(c.pool.locateThreadIn(thread_index, compute_domain_index));
            const domain: u64 = @intCast(compute_domain_index);
            const total = replica.totalWork();
            const domain_low = total * domain / domains;
            const domain_high = total * (domain + 1) / domains;
            const domain_work = domain_high - domain_low;
            const first = replica.itemAtWork(domain_low + domain_work * local_index / threads_here);
            const last = replica.itemAtWork(domain_low + domain_work * (local_index + 1) / threads_here);
            c.counters[thread_index].value += countTrianglesOnSlice(&replica, first, last - first);
        }
    }.work, WorkContext{ .pool = context.pool.?, .topology = context.topology, .replicas = context.replicas.?, .counters = context.counters.ptr });
}

fn runStdThreads(context: *Context) void {
    stdThreadsCount(context.allocator, context.graph, context.counters, context.n_threads) catch |err|
        std.debug.panic("std_threads backend failed: {}", .{err});
}

const backends = [_]Backend{
    .{ .name = "forkunion_vertex_centric_static", .run = runVertexCentricStatic, .engine = .forkunion },
    .{ .name = "forkunion_vertex_centric_dynamic", .run = runVertexCentricDynamic, .engine = .forkunion },
    .{ .name = "forkunion_edge_centric", .run = runEdgeCentric, .engine = .forkunion },
    .{ .name = "forkunion_edge_centric_replicated", .run = runEdgeCentricReplicated, .engine = .forkunion_replicated },
    .{ .name = "std_threads", .run = runStdThreads, .engine = .std_threads },
};

pub fn main() !void {
    var general_purpose_allocator = std.heap.DebugAllocator(.{}){};
    defer _ = general_purpose_allocator.deinit();
    const allocator = general_purpose_allocator.allocator();

    const topology = try fu.Topology.init();
    defer topology.deinit();

    const scale = envUsize("TRIANGLES_SCALE", 18);
    const edge_factor = envUsize("TRIANGLES_EDGE_FACTOR", 16);
    const backend = envString("TRIANGLES_BACKEND", "forkunion_vertex_centric_static");
    const check = envFlag("TRIANGLES_CHECK");
    var n_threads = envUsize("TRIANGLES_THREADS", 0);
    var iterations = envUsize("TRIANGLES_ITERATIONS", 1);
    if (n_threads == 0) n_threads = topology.countLogicalCores();
    if (iterations == 0) iterations = 1;

    var host = try generateRmat(allocator, scale, edge_factor);
    defer host.deinit();
    const graph = host.view();
    const vertices = graph.vertices();
    var max_degree: u32 = 0;
    for (0..vertices) |v| max_degree = @max(max_degree, graph.degree(v));
    var banner_buffer: [256]u8 = undefined;
    const banner = std.fmt.bufPrint(&banner_buffer, "vertices {}, directed edges {}, tape {}, max degree {} (mean {d:.1})\n", .{
        vertices,
        graph.edges(),
        graph.tapeLength(),
        max_degree,
        @as(f64, @floatFromInt(graph.edges())) / @as(f64, @floatFromInt(vertices)),
    }) catch return;
    writeStdout(banner);

    const selected = blk: {
        for (&backends) |*entry| {
            if (std.mem.eql(u8, entry.name, backend)) break :blk entry;
        }
        std.debug.print("Unknown backend: {s}\n", .{backend});
        std.debug.print("Available backends:", .{});
        for (&backends) |*entry| std.debug.print(" {s}", .{entry.name});
        std.debug.print("\n", .{});
        return error.UnknownBackend;
    };

    const counters = try allocator.alloc(Counter, n_threads);
    defer allocator.free(counters);

    // Build only the engine resources the selected backend needs.
    var pool: ?fu.Pool = null;
    defer if (pool) |*p| p.deinit();
    var replicas: ?ReplicatedCsr = null;
    defer if (replicas) |*r| r.deinit();

    switch (selected.engine) {
        .forkunion, .forkunion_replicated => {
            pool = try fu.Pool.init(topology, n_threads, .inclusive);
            if (selected.engine == .forkunion_replicated) {
                replicas = ReplicatedCsr.init(topology, host) orelse return error.OutOfMemory;
            }
        },
        .std_threads => {},
    }

    var context = Context{
        .graph = graph,
        .replicas = if (replicas) |*r| r else null,
        .topology = topology,
        .pool = if (pool) |*p| p else null,
        .counters = counters,
        .allocator = allocator,
        .n_threads = n_threads,
    };

    // Run the pass `iterations` times, timing it; it leaves its per-thread tallies in `counters`, which
    // are zeroed before each run and summed after the last.
    const started = monotonicNanos();
    var it: usize = 0;
    while (it < iterations) : (it += 1) {
        for (counters) |*counter| counter.value = 0;
        selected.run(&context);
    }
    const elapsed_ns = monotonicNanos() - started;
    const seconds = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_s / @as(f64, @floatFromInt(iterations));
    var triangles: u64 = 0;
    for (counters) |counter| triangles += counter.value;
    var line_buffer: [256]u8 = undefined;
    const line = std.fmt.bufPrint(&line_buffer, "{s}: {} triangles in {d:.3} s\n", .{ backend, triangles, seconds }) catch return;
    writeStdout(line);

    if (check) {
        var serial: u64 = 0;
        for (0..vertices) |v| serial += countTrianglesAtVertex(&graph, @intCast(v));
        if (serial != triangles) {
            std.debug.print("MISMATCH: serial counted {}\n", .{serial});
            return error.Mismatch;
        }
        writeStdout("check: matches the serial count\n");
    }
}
