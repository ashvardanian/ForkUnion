//! N-Body simulation benchmark comparing different parallelism libraries.
//!
//! Compares the synchronization overhead of different thread-pool implementations:
//! - forkunion_static_shared: static work division (N tasks pre-divided into thread slices)
//! - forkunion_dynamic_shared: dynamic work-stealing (ForkUnion's work-stealing scheduler)
//! - forkunion_static_replicated: static, with body positions replicated into each domain's local memory
//! - forkunion_dynamic_replicated: work-stealing, over the same per-domain replicas
//! - std_threads: static work division (one raw std.Thread per slice, joined per pass)
//! - libxev: dynamic lock-free queue (Mitchell Hashimoto's lock-free thread pool)
//!
//! On a machine with one memory domain the replicas collapse to one, and the portable allocator backs
//! them, so the `replicated_*` backends compile and run everywhere.
//!
//! Environment variables:
//! - NBODY_COUNT: number of bodies (default: number of threads)
//! - NBODY_SECONDS: wall-clock budget per run, reporting the sustained rate (default: 10)
//! - NBODY_ITERATIONS: run an exact iteration count instead, when set
//! - NBODY_BACKEND: one of the backend names above (default: forkunion_static_shared)
//! - NBODY_THREADS: number of threads (default: CPU count)
//!
//! Build and run from the scripts/ directory:
//! ```sh
//! cd scripts
//! zig build -Doptimize=ReleaseFast
//! NBODY_COUNT=512 NBODY_BACKEND=forkunion_static_shared ./zig-out/bin/forkunion_nbody
//! NBODY_COUNT=512 NBODY_BACKEND=forkunion_static_replicated ./zig-out/bin/forkunion_nbody
//! NBODY_COUNT=512 NBODY_BACKEND=libxev ./zig-out/bin/forkunion_nbody
//! ```

const std = @import("std");
const fu = @import("forkunion");
const xev = @import("xev");

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

/// Reads the monotonic clock in nanoseconds, for timing the iteration loop.
fn monotonicNanos() u64 {
    var timespec: std.posix.timespec = undefined;
    _ = std.posix.system.clock_gettime(.MONOTONIC, &timespec);
    return @as(u64, @intCast(timespec.sec)) * std.time.ns_per_s + @as(u64, @intCast(timespec.nsec));
}

/// Writes one preformatted result line to STDOUT; diagnostics stay on STDERR via `std.debug.print`.
fn writeStdout(text: []const u8) void {
    _ = std.c.write(std.Io.File.stdout().handle, text.ptr, text.len);
}

// Physical constants
const G: f32 = 6.674e-11;
const DT: f32 = 0.01;
const SOFTEN: f32 = 1.0e-9;

const Vector3 = struct {
    x: f32 = 0,
    y: f32 = 0,
    z: f32 = 0,

    fn addAssign(self: *Vector3, other: Vector3) void {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
};

const Body = struct {
    position: Vector3 = .{},
    velocity: Vector3 = .{},
    mass: f32 = 0,
};

/// The SplitMix64 avalanche behind every random draw - a pure function of the `counter`.
///
/// Deliberately not `std.Random.DefaultPrng`: the standard generators differ across languages - and
/// the C++ distributions even across standard libraries - so no two harnesses would simulate the
/// same system. Each draw is a pure function of its counter instead, and the bodies are
/// bit-identical across the C++, Rust, and Zig ports of this hash.
inline fn splitMix(counter: u64) u64 {
    var x = (counter +% 1) *% 0x9E37_79B9_7F4A_7C15;
    x = (x ^ (x >> 30)) *% 0xBF58_476D_1CE4_E5B9;
    x = (x ^ (x >> 27)) *% 0x94D0_49BB_1331_11EB;
    return x ^ (x >> 31);
}

/// One draw in `[0, 1)`: the top 24 bits scaled by 2^-24 - both steps exact in `f32`.
inline fn randomUnit(counter: u64) f32 {
    return @as(f32, @floatFromInt(splitMix(counter) >> 40)) * (1.0 / 16777216.0);
}

/// Fast reciprocal square root (Quake-style with one Newton iteration)
inline fn fastRsqrt(x: f32) f32 {
    const i = 0x5f3759df -% (@as(u32, @bitCast(x)) >> 1);
    var y = @as(f32, @bitCast(i));
    const x2 = 0.5 * x;
    y *= 1.5 - x2 * y * y;
    return y;
}

inline fn gravitationalForce(bi: *const Body, bj: *const Body) Vector3 {
    const dx = bj.position.x - bi.position.x;
    const dy = bj.position.y - bi.position.y;
    const dz = bj.position.z - bi.position.z;
    const l2_squared = dx * dx + dy * dy + dz * dz + SOFTEN;
    const l2_reciprocal = fastRsqrt(l2_squared);
    const l2_cube_reciprocal = l2_reciprocal * l2_reciprocal * l2_reciprocal;
    const mag = G * bi.mass * bj.mass * l2_cube_reciprocal;
    return .{
        .x = mag * dx,
        .y = mag * dy,
        .z = mag * dz,
    };
}

/// How many independent accumulator chains the force sweep keeps. Reassociating a float reduction
/// is exactly what fast-math permits and strict IEEE forbids - so the reassociation is written out
/// by hand instead: the same eight lanes in the C++, Rust, and Zig kernels, reduced in the same
/// fixed order. Every compiler then faces the same strict-IEEE optimization problem with the same
/// freedom, and the language columns compare schedulers rather than compiler flag sets.
const force_lanes = 8;

/// Net gravitational force on `bi` over `bodies`, in eight explicit lanes.
inline fn netForce(bi: *const Body, bodies: []const Body) Vector3 {
    var fx = [_]f32{0} ** force_lanes;
    var fy = [_]f32{0} ** force_lanes;
    var fz = [_]f32{0} ** force_lanes;
    const blocked = bodies.len - bodies.len % force_lanes;
    var j: usize = 0;
    while (j < blocked) : (j += force_lanes) {
        for (0..force_lanes) |lane| {
            const f = gravitationalForce(bi, &bodies[j + lane]);
            fx[lane] += f.x;
            fy[lane] += f.y;
            fz[lane] += f.z;
        }
    }
    for (blocked..bodies.len) |tail| {
        const f = gravitationalForce(bi, &bodies[tail]);
        fx[tail - blocked] += f.x;
        fy[tail - blocked] += f.y;
        fz[tail - blocked] += f.z;
    }
    // The one reduction shape every language shares; changing it changes the bits.
    return .{
        .x = ((fx[0] + fx[1]) + (fx[2] + fx[3])) + ((fx[4] + fx[5]) + (fx[6] + fx[7])),
        .y = ((fy[0] + fy[1]) + (fy[2] + fy[3])) + ((fy[4] + fy[5]) + (fy[6] + fy[7])),
        .z = ((fz[0] + fz[1]) + (fz[2] + fz[3])) + ((fz[4] + fz[5]) + (fz[6] + fz[7])),
    };
}

inline fn applyForce(b: *Body, f: *const Vector3) void {
    b.velocity.x += f.x / b.mass * DT;
    b.velocity.y += f.y / b.mass * DT;
    b.velocity.z += f.z / b.mass * DT;

    b.position.x += b.velocity.x * DT;
    b.position.y += b.velocity.y * DT;
    b.position.z += b.velocity.z * DT;
    // ? Wrap into the unit box to keep every distance - and so every force - inside the normal
    // ? `f32` range forever: no overflows into NaN, and no denormals for x86 to stall on.
    b.position.x -= @floor(b.position.x);
    b.position.y -= @floor(b.position.y);
    b.position.z -= @floor(b.position.z);
}

// Compile-time axes - Zig takes real enums as comptime parameters. nbody is all-to-all, so there is
// no decomposition axis - only schedule and placement.
const Schedule = enum { static, dynamic };
const Placement = enum { shared, replicated };

// ForkUnion kernels

/// Everything either pass of a ForkUnion backend reads or writes; the comptime placement elides the rest.
const WorkContext = struct {
    bodies_ptr: [*]Body,
    forces_ptr: [*]Vector3,
    topology: fu.Topology,
    replicas: ?*fu.ReplicatedArray(Body),
    n: usize,
};

/// The all-to-all sweep: every body reads every other, from the shared array or its node-local replica.
fn forceKernel(comptime placement: Placement) fn (fu.Prong, WorkContext) void {
    return struct {
        fn calc(prong: fu.Prong, work: WorkContext) void {
            if (placement == .replicated) {
                const local = work.replicas.?.onMemoryDomain(work.topology.localMemoryOf(prong.compute_domain_index));
                work.forces_ptr[prong.task_index] = netForce(&local[prong.task_index], local[0..work.n]);
            } else {
                const bi = &work.bodies_ptr[prong.task_index];
                work.forces_ptr[prong.task_index] = netForce(bi, work.bodies_ptr[0..work.n]);
            }
        }
    }.calc;
}

/// The second pass every ForkUnion backend shares: integrate each body by its accumulated force.
fn applyKernel(prong: fu.Prong, work: WorkContext) void {
    applyForce(&work.bodies_ptr[prong.task_index], &work.forces_ptr[prong.task_index]);
}

/// Dispatches `kernel` over `n` tasks on the chosen schedule: pre-divided static, or work-stolen dynamic.
fn forNScheduled(comptime schedule: Schedule, pool: *const fu.Pool, n: usize, comptime kernel: anytype, work: WorkContext) void {
    if (schedule == .static) pool.forN(n, kernel, work) else pool.forNDynamic(n, kernel, work);
}

// ForkUnion replicated kernels
//
// Replicate the body positions into each memory domain's local storage once per iteration, so the
// quadratic all-to-all reads stay node-local. The all-to-all cannot be sharded - every body reads
// every other - so the only locality left to win is the read side.

/// Copies canonical `bodies` into every per-domain replica, each written by the cores local to its node
/// so the pages first-touch there. Every compute domain sharing a memory domain cooperates on that
/// node's one replica, partitioned across all its threads so no element is copied twice.
fn refreshReplicas(pool: *const fu.Pool, topology: fu.Topology, bodies: []const Body, replicas: *fu.ReplicatedArray(Body)) void {
    const RefreshContext = struct {
        pool: *const fu.Pool,
        topology: fu.Topology,
        bodies: [*]const Body,
        replicas: *fu.ReplicatedArray(Body),
        n: usize,
    };
    pool.forThreads(struct {
        fn refresh(thread_index: usize, compute_domain_index: usize, context: RefreshContext) void {
            const memory_domain = context.topology.localMemoryOf(compute_domain_index);

            // Rank this thread among every thread on its memory domain, and count them, so the node's
            // whole team splits [0, n) without overlap even when several compute domains share the node.
            var threads_on_memory_domain: usize = 0;
            var local_index: usize = 0;
            for (0..context.pool.compute_domains()) |other| {
                if (context.topology.localMemoryOf(other) != memory_domain) continue;
                if (other < compute_domain_index) local_index += context.pool.countThreadsIn(other);
                threads_on_memory_domain += context.pool.countThreadsIn(other);
            }
            local_index += context.pool.locateThreadIn(thread_index, compute_domain_index);
            if (threads_on_memory_domain == 0) return;

            const chunk = (context.n + threads_on_memory_domain - 1) / threads_on_memory_domain;
            const start = local_index * chunk;
            if (start >= context.n) return;
            const end = @min(start + chunk, context.n);
            const replica = context.replicas.onMemoryDomain(memory_domain);
            @memcpy(replica[start..end], context.bodies[start..end]);
        }
    }.refresh, RefreshContext{ .pool = pool, .topology = topology, .bodies = bodies.ptr, .replicas = replicas, .n = bodies.len });
}

/// One simulation step, specialized over the schedule and placement axes; the four ForkUnion backends
/// are its instantiations. The all-to-all sweep reads either the shared array or each thread's node-local
/// replica; the apply pass then integrates the canonical bodies.
fn iterationForkUnion(
    comptime schedule: Schedule,
    comptime placement: Placement,
    pool: *const fu.Pool,
    topology: fu.Topology,
    bodies: []Body,
    forces: []Vector3,
    replicas: ?*fu.ReplicatedArray(Body),
) void {
    const n = bodies.len;
    if (placement == .replicated) refreshReplicas(pool, topology, bodies, replicas.?);

    const work = WorkContext{
        .bodies_ptr = bodies.ptr,
        .forces_ptr = forces.ptr,
        .topology = topology,
        .replicas = replicas,
        .n = n,
    };
    forNScheduled(schedule, pool, n, forceKernel(placement), work);
    pool.forN(n, applyKernel, work);
}

// std.Thread backend (static work division)
// Divides N tasks into equal slices, one raw std.Thread per slice, joined at the end of each pass.

fn iterationStdThreads(allocator: std.mem.Allocator, bodies: []Body, forces: []Vector3, n_threads: usize) !void {
    const n = bodies.len;
    const threads = try allocator.alloc(std.Thread, n_threads);
    defer allocator.free(threads);
    const chunk = (n + n_threads - 1) / n_threads;

    {
        var spawned: usize = 0;
        for (0..n_threads) |thread_id| {
            const start = thread_id * chunk;
            if (start >= n) break;
            const end = @min(start + chunk, n);
            threads[spawned] = try std.Thread.spawn(.{}, struct {
                fn calc(bodies_slice: []const Body, forces_slice: []Vector3, range_start: usize, range_end: usize) void {
                    for (range_start..range_end) |i| {
                        forces_slice[i] = netForce(&bodies_slice[i], bodies_slice);
                    }
                }
            }.calc, .{ bodies, forces, start, end });
            spawned += 1;
        }
        for (threads[0..spawned]) |t| t.join();
    }

    {
        var spawned: usize = 0;
        for (0..n_threads) |thread_id| {
            const start = thread_id * chunk;
            if (start >= n) break;
            const end = @min(start + chunk, n);
            threads[spawned] = try std.Thread.spawn(.{}, struct {
                fn apply(bodies_slice: []Body, forces_slice: []const Vector3, range_start: usize, range_end: usize) void {
                    for (range_start..range_end) |i| applyForce(&bodies_slice[i], &forces_slice[i]);
                }
            }.apply, .{ bodies, forces, start, end });
            spawned += 1;
        }
        for (threads[0..spawned]) |t| t.join();
    }
}

// libxev thread-pool backend (lock-free queue - dynamic)
// Creates one task per body, batches them, and relies on the framework's lock-free queue for dynamic
// work distribution across workers.

fn iterationLibxev(pool: *xev.ThreadPool, bodies: []Body, forces: []Vector3, allocator: std.mem.Allocator) !void {
    const n = bodies.len;

    const CalcContext = struct {
        task: xev.ThreadPool.Task,
        bodies: []const Body,
        forces: []Vector3,
        idx: usize,
        done: *std.atomic.Value(usize),

        fn run(task_ptr: *xev.ThreadPool.Task) void {
            const context: *@This() = @fieldParentPtr("task", task_ptr);
            const bi = &context.bodies[context.idx];
            var acc = Vector3{};
            for (context.bodies) |*bj| acc.addAssign(gravitationalForce(bi, bj));
            context.forces[context.idx] = acc;
            _ = context.done.fetchAdd(1, .monotonic);
        }
    };

    var calc_contexts = try allocator.alloc(CalcContext, n);
    defer allocator.free(calc_contexts);
    var calc_done = std.atomic.Value(usize).init(0);
    for (0..n) |i| {
        calc_contexts[i] = .{
            .task = .{ .callback = CalcContext.run },
            .bodies = bodies,
            .forces = forces,
            .idx = i,
            .done = &calc_done,
        };
    }
    var calc_batch = xev.ThreadPool.Batch{};
    for (calc_contexts) |*context| calc_batch.push(xev.ThreadPool.Batch.from(&context.task));
    pool.schedule(calc_batch);
    while (calc_done.load(.acquire) < n) std.atomic.spinLoopHint();

    const ApplyContext = struct {
        task: xev.ThreadPool.Task,
        bodies: []Body,
        forces: []const Vector3,
        idx: usize,
        done: *std.atomic.Value(usize),

        fn run(task_ptr: *xev.ThreadPool.Task) void {
            const context: *@This() = @fieldParentPtr("task", task_ptr);
            applyForce(&context.bodies[context.idx], &context.forces[context.idx]);
            _ = context.done.fetchAdd(1, .monotonic);
        }
    };

    var apply_contexts = try allocator.alloc(ApplyContext, n);
    defer allocator.free(apply_contexts);
    var apply_done = std.atomic.Value(usize).init(0);
    for (0..n) |i| {
        apply_contexts[i] = .{
            .task = .{ .callback = ApplyContext.run },
            .bodies = bodies,
            .forces = forces,
            .idx = i,
            .done = &apply_done,
        };
    }
    var apply_batch = xev.ThreadPool.Batch{};
    for (apply_contexts) |*context| apply_batch.push(xev.ThreadPool.Batch.from(&context.task));
    pool.schedule(apply_batch);
    while (apply_done.load(.acquire) < n) std.atomic.spinLoopHint();
}

// Registry

/// Which execution engine a backend runs on, so `main` builds exactly the resource it needs.
const Engine = enum { forkunion, forkunion_replicated, std_threads, libxev };

/// Everything a backend reads or writes for one simulation step; `main` owns the lifetimes and hands
/// each backend only the execution engine it asked for.
const Context = struct {
    bodies: []Body,
    forces: []Vector3,
    topology: fu.Topology,
    pool: ?*fu.Pool,
    replicas: ?*fu.ReplicatedArray(Body),
    xev_pool: ?*xev.ThreadPool,
    allocator: std.mem.Allocator,
    n_threads: usize,
};

/// The dispatch table - a name, its per-step function, and the engine it runs on.
const Backend = struct {
    name: []const u8,
    run: *const fn (*Context) void,
    engine: Engine,
};

fn runForkUnion(comptime schedule: Schedule, comptime placement: Placement) fn (*Context) void {
    return struct {
        fn call(context: *Context) void {
            iterationForkUnion(schedule, placement, context.pool.?, context.topology, context.bodies, context.forces, context.replicas);
        }
    }.call;
}

fn runStdThreads(context: *Context) void {
    iterationStdThreads(context.allocator, context.bodies, context.forces, context.n_threads) catch |e|
        std.debug.panic("std_threads backend failed: {}", .{e});
}

fn runLibxev(context: *Context) void {
    iterationLibxev(context.xev_pool.?, context.bodies, context.forces, context.allocator) catch |e|
        std.debug.panic("libxev backend failed: {}", .{e});
}

const backends = [_]Backend{
    .{ .name = "forkunion_static_shared", .run = runForkUnion(.static, .shared), .engine = .forkunion },
    .{ .name = "forkunion_dynamic_shared", .run = runForkUnion(.dynamic, .shared), .engine = .forkunion },
    .{ .name = "forkunion_static_replicated", .run = runForkUnion(.static, .replicated), .engine = .forkunion_replicated },
    .{ .name = "forkunion_dynamic_replicated", .run = runForkUnion(.dynamic, .replicated), .engine = .forkunion_replicated },
    .{ .name = "std_threads", .run = runStdThreads, .engine = .std_threads },
    .{ .name = "libxev", .run = runLibxev, .engine = .libxev },
};

pub fn main() !void {
    var general_purpose_allocator = std.heap.DebugAllocator(.{}){};
    defer _ = general_purpose_allocator.deinit();
    const allocator = general_purpose_allocator.allocator();

    // The machine topology is probed once and threaded through every spawn and query below.
    const topology = try fu.Topology.init();
    defer topology.deinit();

    var n_threads = envUsize("NBODY_THREADS", 0);
    if (n_threads == 0) n_threads = topology.countLogicalCores();

    const budget_seconds = envF64("NBODY_SECONDS", 10); // The primary knob: a fixed window
    const n_iters = envUsize("NBODY_ITERATIONS", 0); // Overrides with an exact count when set

    var n_bodies = envUsize("NBODY_COUNT", 0);
    if (n_bodies == 0) n_bodies = n_threads;

    const backend = envString("NBODY_BACKEND", "forkunion_static_shared");

    const bodies = try allocator.alloc(Body, n_bodies);
    defer allocator.free(bodies);
    const forces = try allocator.alloc(Vector3, n_bodies);
    defer allocator.free(forces);

    // Seven counter-based draws per body: three position coordinates, three velocity components, and
    // one mass in [1e10, 1e15) - so every language starts from bit-identical bodies.
    for (bodies, 0..) |*body, i| {
        const counter = @as(u64, i) * 7;
        body.position = .{ .x = randomUnit(counter), .y = randomUnit(counter + 1), .z = randomUnit(counter + 2) };
        body.velocity = .{ .x = randomUnit(counter + 3), .y = randomUnit(counter + 4), .z = randomUnit(counter + 5) };
        // ? Round each literal to `f32` before subtracting: Zig's comptime floats are exact, and the
        // ? folded span would otherwise differ from the C++ and Rust builds by an ULP.
        const mass_span: f32 = @as(f32, 1.0e15) - @as(f32, 1.0e10);
        body.mass = @as(f32, 1.0e10) + randomUnit(counter + 6) * mass_span;
    }

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

    // Build only the engine resources the selected backend needs.
    var pool: ?fu.Pool = null;
    defer if (pool) |*p| p.deinit();
    var replicas: ?fu.ReplicatedArray(Body) = null;
    defer if (replicas) |*r| r.deinit();
    var xev_pool: ?xev.ThreadPool = null;
    defer if (xev_pool) |*x| {
        x.shutdown();
        x.deinit();
    };

    switch (selected.engine) {
        .forkunion, .forkunion_replicated => {
            pool = try fu.Pool.init(topology, n_threads, .inclusive);
            if (selected.engine == .forkunion_replicated) {
                replicas = fu.ReplicatedArray(Body).init(topology, n_bodies) orelse return error.OutOfMemory;
            }
        },
        .std_threads => {},
        .libxev => xev_pool = xev.ThreadPool.init(.{ .max_threads = @intCast(n_threads) }),
    }

    var context = Context{
        .bodies = bodies,
        .forces = forces,
        .topology = topology,
        .pool = if (pool) |*p| p else null,
        .replicas = if (replicas) |*r| r else null,
        .xev_pool = if (xev_pool) |*x| x else null,
        .allocator = allocator,
        .n_threads = n_threads,
    };
    // A fixed time budget beats a fixed iteration count: every backend runs the same wall-clock
    // window - long enough to amortize scheduling noise - and reports the rate it sustained, with
    // no per-backend iteration guessing. NBODY_ITERATIONS forces an exact count instead.
    const budget_ns: u64 = @intFromFloat(budget_seconds * std.time.ns_per_s);
    const started = monotonicNanos();
    var passes: usize = 0;
    if (n_iters > 0) {
        for (0..n_iters) |_| selected.run(&context);
        passes = n_iters;
    } else {
        while (true) {
            selected.run(&context);
            passes += 1;
            if (monotonicNanos() - started >= budget_ns) break;
        }
    }
    const elapsed_ns = monotonicNanos() - started;
    const total_seconds = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_s;
    const us_per_iter = total_seconds / @as(f64, @floatFromInt(passes)) * std.time.us_per_s;

    var line_buffer: [256]u8 = undefined;
    const line = std.fmt.bufPrint(&line_buffer, "{s}: {} bodies, {} iters, {d:.2} us/iter ({d:.2} s total)\n", .{ backend, n_bodies, passes, us_per_iter, total_seconds }) catch return;
    writeStdout(line);
}
