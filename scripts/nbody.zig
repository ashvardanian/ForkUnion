//! N-Body simulation benchmark comparing different parallelism libraries.
//!
//! Compares the synchronization overhead of different thread-pool implementations:
//! - forkunion_static: static work division (N tasks pre-divided into thread slices)
//! - forkunion_dynamic: dynamic work-stealing (ForkUnion's work-stealing scheduler)
//! - forkunion_replicated_static: static, with body positions replicated into each domain's local memory
//! - forkunion_replicated_dynamic: work-stealing, over the same per-domain replicas
//! - std_threads: static work division (one raw std.Thread per slice, joined per pass)
//! - libxev: dynamic lock-free queue (Mitchell Hashimoto's lock-free thread pool)
//!
//! On a machine with one memory domain the replicas collapse to one, and the portable allocator backs
//! them, so the `replicated_*` backends compile and run everywhere.
//!
//! Environment variables:
//! - NBODY_COUNT: number of bodies (default: number of threads)
//! - NBODY_ITERATIONS: number of iterations (default: 1000)
//! - NBODY_BACKEND: one of the backend names above (default: forkunion_static)
//! - NBODY_THREADS: number of threads (default: CPU count)
//!
//! Build and run from the scripts/ directory:
//! ```sh
//! cd scripts
//! zig build -Doptimize=ReleaseFast
//! time NBODY_COUNT=128 NBODY_ITERATIONS=1000000 NBODY_BACKEND=forkunion_static ./zig-out/bin/nbody
//! time NBODY_COUNT=128 NBODY_ITERATIONS=1000000 NBODY_BACKEND=forkunion_replicated_static ./zig-out/bin/nbody
//! time NBODY_COUNT=128 NBODY_ITERATIONS=1000000 NBODY_BACKEND=libxev ./zig-out/bin/nbody
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

inline fn applyForce(b: *Body, f: *const Vector3) void {
    b.velocity.x += f.x / b.mass * DT;
    b.velocity.y += f.y / b.mass * DT;
    b.velocity.z += f.z / b.mass * DT;

    b.position.x += b.velocity.x * DT;
    b.position.y += b.velocity.y * DT;
    b.position.z += b.velocity.z * DT;
}

// ForkUnion kernels

/// The second pass every ForkUnion backend shares: integrate each body by its accumulated force.
fn applyPass(pool: *const fu.Pool, bodies: []Body, forces: []Vector3) void {
    const ApplyContext = struct {
        bodies_ptr: [*]Body,
        forces_ptr: [*]const Vector3,
    };
    pool.forN(bodies.len, struct {
        fn apply(prong: fu.Prong, context: ApplyContext) void {
            applyForce(&context.bodies_ptr[prong.task_index], &context.forces_ptr[prong.task_index]);
        }
    }.apply, ApplyContext{ .bodies_ptr = bodies.ptr, .forces_ptr = forces.ptr });
}

fn iterationForkUnionStatic(pool: *const fu.Pool, bodies: []Body, forces: []Vector3) void {
    const n = bodies.len;
    const CalcContext = struct {
        bodies_ptr: [*]const Body,
        forces_ptr: [*]Vector3,
        n: usize,
    };
    pool.forN(n, struct {
        fn calc(prong: fu.Prong, context: CalcContext) void {
            const bi = &context.bodies_ptr[prong.task_index];
            var acc = Vector3{};
            for (0..context.n) |j| acc.addAssign(gravitationalForce(bi, &context.bodies_ptr[j]));
            context.forces_ptr[prong.task_index] = acc;
        }
    }.calc, CalcContext{ .bodies_ptr = bodies.ptr, .forces_ptr = forces.ptr, .n = n });
    applyPass(pool, bodies, forces);
}

fn iterationForkUnionDynamic(pool: *const fu.Pool, bodies: []Body, forces: []Vector3) void {
    const n = bodies.len;
    const CalcContext = struct {
        bodies_ptr: [*]const Body,
        forces_ptr: [*]Vector3,
        n: usize,
    };
    pool.forNDynamic(n, struct {
        fn calc(prong: fu.Prong, context: CalcContext) void {
            const bi = &context.bodies_ptr[prong.task_index];
            var acc = Vector3{};
            for (0..context.n) |j| acc.addAssign(gravitationalForce(bi, &context.bodies_ptr[j]));
            context.forces_ptr[prong.task_index] = acc;
        }
    }.calc, CalcContext{ .bodies_ptr = bodies.ptr, .forces_ptr = forces.ptr, .n = n });
    applyPass(pool, bodies, forces);
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

fn iterationForkUnionReplicatedStatic(pool: *const fu.Pool, topology: fu.Topology, bodies: []Body, forces: []Vector3, replicas: *fu.ReplicatedArray(Body)) void {
    refreshReplicas(pool, topology, bodies, replicas);
    const n = bodies.len;
    const CalcContext = struct {
        topology: fu.Topology,
        replicas: *fu.ReplicatedArray(Body),
        forces_ptr: [*]Vector3,
        n: usize,
    };
    pool.forN(n, struct {
        fn calc(prong: fu.Prong, context: CalcContext) void {
            const local_bodies = context.replicas.onMemoryDomain(context.topology.localMemoryOf(prong.compute_domain_index));
            const bi = &local_bodies[prong.task_index];
            var acc = Vector3{};
            for (0..context.n) |j| acc.addAssign(gravitationalForce(bi, &local_bodies[j]));
            context.forces_ptr[prong.task_index] = acc;
        }
    }.calc, CalcContext{ .topology = topology, .replicas = replicas, .forces_ptr = forces.ptr, .n = n });
    applyPass(pool, bodies, forces);
}

fn iterationForkUnionReplicatedDynamic(pool: *const fu.Pool, topology: fu.Topology, bodies: []Body, forces: []Vector3, replicas: *fu.ReplicatedArray(Body)) void {
    refreshReplicas(pool, topology, bodies, replicas);
    const n = bodies.len;
    const CalcContext = struct {
        topology: fu.Topology,
        replicas: *fu.ReplicatedArray(Body),
        forces_ptr: [*]Vector3,
        n: usize,
    };
    pool.forNDynamic(n, struct {
        fn calc(prong: fu.Prong, context: CalcContext) void {
            const local_bodies = context.replicas.onMemoryDomain(context.topology.localMemoryOf(prong.compute_domain_index));
            const bi = &local_bodies[prong.task_index];
            var acc = Vector3{};
            for (0..context.n) |j| acc.addAssign(gravitationalForce(bi, &local_bodies[j]));
            context.forces_ptr[prong.task_index] = acc;
        }
    }.calc, CalcContext{ .topology = topology, .replicas = replicas, .forces_ptr = forces.ptr, .n = n });
    applyPass(pool, bodies, forces);
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
                        const bi = &bodies_slice[i];
                        var acc = Vector3{};
                        for (bodies_slice) |*bj| acc.addAssign(gravitationalForce(bi, bj));
                        forces_slice[i] = acc;
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

fn runForkUnionStatic(context: *Context) void {
    iterationForkUnionStatic(context.pool.?, context.bodies, context.forces);
}

fn runForkUnionDynamic(context: *Context) void {
    iterationForkUnionDynamic(context.pool.?, context.bodies, context.forces);
}

fn runForkUnionReplicatedStatic(context: *Context) void {
    iterationForkUnionReplicatedStatic(context.pool.?, context.topology, context.bodies, context.forces, context.replicas.?);
}

fn runForkUnionReplicatedDynamic(context: *Context) void {
    iterationForkUnionReplicatedDynamic(context.pool.?, context.topology, context.bodies, context.forces, context.replicas.?);
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
    .{ .name = "forkunion_static", .run = runForkUnionStatic, .engine = .forkunion },
    .{ .name = "forkunion_dynamic", .run = runForkUnionDynamic, .engine = .forkunion },
    .{ .name = "forkunion_replicated_static", .run = runForkUnionReplicatedStatic, .engine = .forkunion_replicated },
    .{ .name = "forkunion_replicated_dynamic", .run = runForkUnionReplicatedDynamic, .engine = .forkunion_replicated },
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

    const n_iters = envUsize("NBODY_ITERATIONS", 1000);

    var n_bodies = envUsize("NBODY_COUNT", 0);
    if (n_bodies == 0) n_bodies = n_threads;

    const backend = envString("NBODY_BACKEND", "forkunion_static");

    const bodies = try allocator.alloc(Body, n_bodies);
    defer allocator.free(bodies);
    const forces = try allocator.alloc(Vector3, n_bodies);
    defer allocator.free(forces);

    // Initialize bodies from a fixed seed - the benchmark only needs a spread of positions, not entropy.
    var generator = std.Random.DefaultPrng.init(0x1234_5678_9abc_def0);
    const random = generator.random();
    for (bodies) |*body| {
        body.position = .{ .x = random.float(f32), .y = random.float(f32), .z = random.float(f32) };
        body.velocity = .{ .x = random.float(f32), .y = random.float(f32), .z = random.float(f32) };
        body.mass = random.float(f32) * 9.0e24 + 1.0e20; // [1e20, 1e25)
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
    const started = monotonicNanos();
    for (0..n_iters) |_| selected.run(&context);
    const elapsed_ns = monotonicNanos() - started;
    const seconds = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_s / @as(f64, @floatFromInt(n_iters));

    var line_buffer: [256]u8 = undefined;
    const line = std.fmt.bufPrint(&line_buffer, "{s}: {} bodies, {} iters in {d:.3} s\n", .{ backend, n_bodies, n_iters, seconds }) catch return;
    writeStdout(line);
}
