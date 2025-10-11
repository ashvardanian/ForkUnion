//! N-Body simulation benchmark comparing different parallelism libraries
//!
//! Compares synchronization overhead of different thread pool implementations:
//! - fork_union_static: Static work division (N tasks pre-divided into thread slices)
//! - fork_union_dynamic: Dynamic work-stealing (Fork Union's work-stealing scheduler)
//! - std: Static work division (std.Thread.Pool with manual slicing)
//! - spice: Dynamic work-stealing (Spice's fork/join work-stealing)
//! - libxev: Dynamic lock-free queue (Mitchell Hashimoto's lock-free thread pool)
//!
//! Environment variables:
//! - NBODY_COUNT: number of bodies (default: number of threads)
//! - NBODY_ITERATIONS: number of iterations (default: 1000)
//! - NBODY_BACKEND: fork_union_static, fork_union_dynamic, std, spice, libxev
//! - NBODY_THREADS: number of threads (default: CPU count)
//!
//! Build and run from scripts/ directory:
//! ```sh
//! cd scripts
//! zig build -Doptimize=ReleaseFast
//! time NBODY_COUNT=128 NBODY_ITERATIONS=1000000 NBODY_BACKEND=fork_union_static ./zig-out/bin/nbody_zig
//! time NBODY_COUNT=128 NBODY_ITERATIONS=1000000 NBODY_BACKEND=fork_union_dynamic ./zig-out/bin/nbody_zig
//! time NBODY_COUNT=128 NBODY_ITERATIONS=1000000 NBODY_BACKEND=spice ./zig-out/bin/nbody_zig
//! time NBODY_COUNT=128 NBODY_ITERATIONS=1000000 NBODY_BACKEND=libxev ./zig-out/bin/nbody_zig
//! time NBODY_COUNT=128 NBODY_ITERATIONS=1000000 NBODY_BACKEND=std ./zig-out/bin/nbody_zig
//! ```

const std = @import("std");
const fu = @import("fork_union");
const spice = @import("spice");
const xev = @import("xev");

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
    const i = 0x5f3759df - (@as(u32, @bitCast(x)) >> 1);
    var y = @as(f32, @bitCast(i));
    const x2 = 0.5 * x;
    y *= 1.5 - x2 * y * y;
    return y;
}

inline fn gravitationalForce(bi: *const Body, bj: *const Body) Vector3 {
    const dx = bj.position.x - bi.position.x;
    const dy = bj.position.y - bi.position.y;
    const dz = bj.position.z - bi.position.z;
    const l2 = dx * dx + dy * dy + dz * dz + SOFTEN;
    const inv = fastRsqrt(l2);
    const inv3 = inv * inv * inv;
    const mag = G * bi.mass * bj.mass * inv3;
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

// ============================================================================
// Fork Union Kernels
// ============================================================================

fn iterationForkUnionStatic(pool: *fu.Pool, bodies: []Body, forces: []Vector3) void {
    const n = bodies.len;

    // First pass: calculate forces
    const CalcContext = struct {
        bodies_ptr: [*]const Body,
        forces_ptr: [*]Vector3,
        n: usize,
    };

    pool.forN(n, struct {
        fn calc(prong: fu.Prong, ctx: CalcContext) void {
            const bi = &ctx.bodies_ptr[prong.task_index];
            var acc = Vector3{};

            for (0..ctx.n) |j| {
                acc.addAssign(gravitationalForce(bi, &ctx.bodies_ptr[j]));
            }
            ctx.forces_ptr[prong.task_index] = acc;
        }
    }.calc, CalcContext{
        .bodies_ptr = bodies.ptr,
        .forces_ptr = forces.ptr,
        .n = n,
    });

    // Second pass: apply forces
    const ApplyContext = struct {
        bodies_ptr: [*]Body,
        forces_ptr: [*]const Vector3,
    };

    pool.forN(n, struct {
        fn apply(prong: fu.Prong, ctx: ApplyContext) void {
            applyForce(&ctx.bodies_ptr[prong.task_index], &ctx.forces_ptr[prong.task_index]);
        }
    }.apply, ApplyContext{
        .bodies_ptr = bodies.ptr,
        .forces_ptr = forces.ptr,
    });
}

fn iterationForkUnionDynamic(pool: *fu.Pool, bodies: []Body, forces: []Vector3) void {
    const n = bodies.len;

    // First pass: calculate forces
    const CalcContext = struct {
        bodies_ptr: [*]const Body,
        forces_ptr: [*]Vector3,
        n: usize,
    };

    pool.forNDynamic(n, struct {
        fn calc(prong: fu.Prong, ctx: CalcContext) void {
            const bi = &ctx.bodies_ptr[prong.task_index];
            var acc = Vector3{};

            for (0..ctx.n) |j| {
                acc.addAssign(gravitationalForce(bi, &ctx.bodies_ptr[j]));
            }
            ctx.forces_ptr[prong.task_index] = acc;
        }
    }.calc, CalcContext{
        .bodies_ptr = bodies.ptr,
        .forces_ptr = forces.ptr,
        .n = n,
    });

    // Second pass: apply forces
    const ApplyContext = struct {
        bodies_ptr: [*]Body,
        forces_ptr: [*]const Vector3,
    };

    pool.forNDynamic(n, struct {
        fn apply(prong: fu.Prong, ctx: ApplyContext) void {
            applyForce(&ctx.bodies_ptr[prong.task_index], &ctx.forces_ptr[prong.task_index]);
        }
    }.apply, ApplyContext{
        .bodies_ptr = bodies.ptr,
        .forces_ptr = forces.ptr,
    });
}

// ============================================================================
// std.Thread.Pool Backend (Static Work Division)
// Divides N tasks into equal slices per thread for static work distribution.
// ============================================================================

fn iterationStdPool(pool: *std.Thread.Pool, bodies: []Body, forces: []Vector3, n_threads: usize) !void {
    const n = bodies.len;

    // First pass: calculate forces
    {
        var wg: std.Thread.WaitGroup = .{};
        const chunk_size = (n + n_threads - 1) / n_threads;

        for (0..n_threads) |thread_id| {
            const start = thread_id * chunk_size;
            if (start >= n) break;
            const end = @min(start + chunk_size, n);

            pool.spawnWg(&wg, struct {
                fn calc(bodies_slice: []const Body, forces_slice: []Vector3, range_start: usize, range_end: usize) void {
                    for (range_start..range_end) |i| {
                        const bi = &bodies_slice[i];
                        var acc = Vector3{};
                        for (bodies_slice) |*bj| {
                            acc.addAssign(gravitationalForce(bi, bj));
                        }
                        forces_slice[i] = acc;
                    }
                }
            }.calc, .{ bodies, forces, start, end });
        }
        pool.waitAndWork(&wg);
    }

    // Second pass: apply forces
    {
        var wg: std.Thread.WaitGroup = .{};
        const chunk_size = (n + n_threads - 1) / n_threads;

        for (0..n_threads) |thread_id| {
            const start = thread_id * chunk_size;
            if (start >= n) break;
            const end = @min(start + chunk_size, n);

            pool.spawnWg(&wg, struct {
                fn apply(bodies_slice: []Body, forces_slice: []const Vector3, range_start: usize, range_end: usize) void {
                    for (range_start..range_end) |i| {
                        applyForce(&bodies_slice[i], &forces_slice[i]);
                    }
                }
            }.apply, .{ bodies, forces, start, end });
        }
        pool.waitAndWork(&wg);
    }
}

// ============================================================================
// Spice Backend (Work-Stealing - Dynamic)
// Uses Spice's fork/join work-stealing scheduler. Creates N futures and
// relies on the framework to dynamically distribute them across workers.
// ============================================================================

fn iterationSpice(pool: *spice.ThreadPool, bodies: []Body, forces: []Vector3, allocator: std.mem.Allocator) !void {
    const n = bodies.len;

    // First pass: calculate forces for all bodies
    const CalcArgs = struct {
        bodies: []const Body,
        forces: []Vector3,
        idx: usize,
    };

    const calc_futures = try allocator.alloc(spice.Future(CalcArgs, void), n);
    defer allocator.free(calc_futures);

    const CalcRunArgs = struct {
        bodies: []const Body,
        forces: []Vector3,
        futures: []spice.Future(CalcArgs, void),
    };

    const calc_run_args = CalcRunArgs{ .bodies = bodies, .forces = forces, .futures = calc_futures };

    _ = pool.call(void, struct {
        fn run(task: *spice.Task, args: CalcRunArgs) void {
            for (args.futures, 0..) |*fut, i| {
                fut.* = spice.Future(CalcArgs, void).init();
                fut.fork(task, struct {
                    fn calc(t: *spice.Task, calc_args: CalcArgs) void {
                        _ = t;
                        const bi = &calc_args.bodies[calc_args.idx];
                        var acc = Vector3{};
                        for (calc_args.bodies) |*bj| {
                            acc.addAssign(gravitationalForce(bi, bj));
                        }
                        calc_args.forces[calc_args.idx] = acc;
                    }
                }.calc, CalcArgs{ .bodies = args.bodies, .forces = args.forces, .idx = i });
            }

            // Join all futures
            for (args.futures) |*fut| {
                _ = fut.join(task);
            }
        }
    }.run, calc_run_args);

    // Second pass: apply forces to all bodies
    const ApplyArgs = struct {
        bodies: []Body,
        forces: []const Vector3,
        idx: usize,
    };

    const apply_futures = try allocator.alloc(spice.Future(ApplyArgs, void), n);
    defer allocator.free(apply_futures);

    const ApplyRunArgs = struct {
        bodies: []Body,
        forces: []const Vector3,
        futures: []spice.Future(ApplyArgs, void),
    };

    const apply_run_args = ApplyRunArgs{ .bodies = bodies, .forces = forces, .futures = apply_futures };

    _ = pool.call(void, struct {
        fn run(task: *spice.Task, args: ApplyRunArgs) void {
            for (args.futures, 0..) |*fut, i| {
                fut.* = spice.Future(ApplyArgs, void).init();
                fut.fork(task, struct {
                    fn apply(t: *spice.Task, apply_args: ApplyArgs) void {
                        _ = t;
                        applyForce(&apply_args.bodies[apply_args.idx], &apply_args.forces[apply_args.idx]);
                    }
                }.apply, ApplyArgs{ .bodies = args.bodies, .forces = args.forces, .idx = i });
            }

            // Join all futures
            for (args.futures) |*fut| {
                _ = fut.join(task);
            }
        }
    }.run, apply_run_args);
}

// ============================================================================
// libxev ThreadPool Backend (Lock-Free Queue - Dynamic)
// Uses libxev's lock-free thread pool with batch task scheduling. Creates N
// tasks, batches them, and relies on the framework's lock-free queue for
// dynamic work distribution across workers.
// ============================================================================

fn iterationLibxev(pool: *xev.ThreadPool, bodies: []Body, forces: []Vector3, allocator: std.mem.Allocator) !void {
    const n = bodies.len;

    // Task context for force calculation
    const CalcContext = struct {
        task: xev.ThreadPool.Task,
        bodies: []const Body,
        forces: []Vector3,
        idx: usize,
        done: *std.atomic.Value(usize),

        fn run(task_ptr: *xev.ThreadPool.Task) void {
            const ctx: *@This() = @fieldParentPtr("task", task_ptr);
            const bi = &ctx.bodies[ctx.idx];
            var acc = Vector3{};
            for (ctx.bodies) |*bj| {
                acc.addAssign(gravitationalForce(bi, bj));
            }
            ctx.forces[ctx.idx] = acc;
            _ = ctx.done.fetchAdd(1, .monotonic);
        }
    };

    // Allocate contexts for force calculation
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

    // Schedule all force calculation tasks
    var calc_batch = xev.ThreadPool.Batch{};
    for (calc_contexts) |*ctx| {
        calc_batch.push(xev.ThreadPool.Batch.from(&ctx.task));
    }
    pool.schedule(calc_batch);

    // Wait for completion
    while (calc_done.load(.acquire) < n) {
        std.atomic.spinLoopHint();
    }

    // Task context for applying forces
    const ApplyContext = struct {
        task: xev.ThreadPool.Task,
        bodies: []Body,
        forces: []const Vector3,
        idx: usize,
        done: *std.atomic.Value(usize),

        fn run(task_ptr: *xev.ThreadPool.Task) void {
            const ctx: *@This() = @fieldParentPtr("task", task_ptr);
            applyForce(&ctx.bodies[ctx.idx], &ctx.forces[ctx.idx]);
            _ = ctx.done.fetchAdd(1, .monotonic);
        }
    };

    // Allocate contexts for applying forces
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

    // Schedule all apply force tasks
    var apply_batch = xev.ThreadPool.Batch{};
    for (apply_contexts) |*ctx| {
        apply_batch.push(xev.ThreadPool.Batch.from(&ctx.task));
    }
    pool.schedule(apply_batch);

    // Wait for completion
    while (apply_done.load(.acquire) < n) {
        std.atomic.spinLoopHint();
    }
}

// ============================================================================
// Main
// ============================================================================

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse environment variables
    const n_threads = if (std.process.getEnvVarOwned(allocator, "NBODY_THREADS")) |str|
        blk: {
            defer allocator.free(str);
            break :blk try std.fmt.parseInt(usize, str, 10);
        }
    else |_|
        fu.countLogicalCores();

    const n_iters = if (std.process.getEnvVarOwned(allocator, "NBODY_ITERATIONS")) |str|
        blk: {
            defer allocator.free(str);
            break :blk try std.fmt.parseInt(usize, str, 10);
        }
    else |_|
        1000;

    const n_bodies = if (std.process.getEnvVarOwned(allocator, "NBODY_COUNT")) |str|
        blk: {
            defer allocator.free(str);
            break :blk try std.fmt.parseInt(usize, str, 10);
        }
    else |_|
        n_threads;

    const backend = if (std.process.getEnvVarOwned(allocator, "NBODY_BACKEND")) |str|
        str
    else |_|
        "fork_union_static";
    defer if (!std.mem.eql(u8, backend, "fork_union_static")) allocator.free(backend);

    // Allocate bodies and forces
    const bodies = try allocator.alloc(Body, n_bodies);
    defer allocator.free(bodies);
    const forces = try allocator.alloc(Vector3, n_bodies);
    defer allocator.free(forces);

    // Initialize bodies
    var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
    const random = prng.random();
    for (bodies) |*body| {
        body.position = .{
            .x = random.float(f32),
            .y = random.float(f32),
            .z = random.float(f32),
        };
        body.velocity = .{
            .x = random.float(f32),
            .y = random.float(f32),
            .z = random.float(f32),
        };
        body.mass = random.float(f32) * 9.0e24 + 1.0e20; // [1e20, 1e25)
    }

    // Run the chosen backend
    if (std.mem.eql(u8, backend, "fork_union_static")) {
        var pool = try fu.Pool.init(n_threads, .inclusive);
        defer pool.deinit();

        for (0..n_iters) |_| {
            iterationForkUnionStatic(&pool, bodies, forces);
        }
    } else if (std.mem.eql(u8, backend, "fork_union_dynamic")) {
        var pool = try fu.Pool.init(n_threads, .inclusive);
        defer pool.deinit();

        for (0..n_iters) |_| {
            iterationForkUnionDynamic(&pool, bodies, forces);
        }
    } else if (std.mem.eql(u8, backend, "std")) {
        var pool: std.Thread.Pool = undefined;
        try pool.init(.{ .allocator = allocator, .n_jobs = @intCast(n_threads) });
        defer pool.deinit();

        for (0..n_iters) |_| {
            try iterationStdPool(&pool, bodies, forces, n_threads);
        }
    } else if (std.mem.eql(u8, backend, "spice")) {
        var pool = spice.ThreadPool.init(allocator);
        pool.start(.{ .background_worker_count = n_threads - 1 });
        defer pool.deinit();

        for (0..n_iters) |_| {
            try iterationSpice(&pool, bodies, forces, allocator);
        }
    } else if (std.mem.eql(u8, backend, "libxev")) {
        var pool = xev.ThreadPool.init(.{ .max_threads = @intCast(n_threads) });
        defer {
            pool.shutdown();
            pool.deinit();
        }

        for (0..n_iters) |_| {
            try iterationLibxev(&pool, bodies, forces, allocator);
        }
    } else {
        std.debug.print("Unknown backend: {s}\n", .{backend});
        std.debug.print("Available backends: fork_union_static, fork_union_dynamic, std, spice, libxev\n", .{});
        return error.UnknownBackend;
    }
}
