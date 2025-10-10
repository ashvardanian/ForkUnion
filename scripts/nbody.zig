//! N-Body simulation benchmark comparing Fork Union vs std.Thread.Pool
//!
//! Environment variables:
//! - NBODY_COUNT: number of bodies (default: number of threads)
//! - NBODY_ITERATIONS: number of iterations (default: 1000)
//! - NBODY_BACKEND: fork_union_static, fork_union_dynamic, std_pool (default: fork_union_static)
//! - NBODY_THREADS: number of threads (default: CPU count)
//!
//! Build and run:
//! ```sh
//! zig build nbody -Doptimize=ReleaseFast
//! time NBODY_COUNT=128 NBODY_ITERATIONS=1000000 NBODY_BACKEND=fork_union_static \
//!     ./zig-out/bin/nbody_zig
//! ```

const std = @import("std");
const fu = @import("fork_union");

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
// std.Thread.Pool Kernel (for comparison)
// ============================================================================

fn iterationStdPool(pool: *std.Thread.Pool, bodies: []Body, forces: []Vector3) !void {
    const n = bodies.len;
    var wg: std.Thread.WaitGroup = .{};

    // First pass: calculate forces
    for (0..n) |i| {
        pool.spawnWg(&wg, struct {
            fn calc(bodies_slice: []const Body, forces_slice: []Vector3, idx: usize) void {
                const bi = &bodies_slice[idx];
                var acc = Vector3{};
                for (bodies_slice) |*bj| {
                    acc.addAssign(gravitationalForce(bi, bj));
                }
                forces_slice[idx] = acc;
            }
        }.calc, .{ bodies, forces, i });
    }
    pool.waitAndWork(&wg);

    // Second pass: apply forces
    for (0..n) |i| {
        pool.spawnWg(&wg, struct {
            fn apply(bodies_slice: []Body, forces_slice: []const Vector3, idx: usize) void {
                applyForce(&bodies_slice[idx], &forces_slice[idx]);
            }
        }.apply, .{ bodies, forces, i });
    }
    pool.waitAndWork(&wg);
}

// ============================================================================
// Main
// ============================================================================

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse environment variables
    const n_bodies_opt = try std.process.getEnvVarOwned(allocator, "NBODY_COUNT");
    defer if (n_bodies_opt.len > 0) allocator.free(n_bodies_opt);
    const n_iters_str = try std.process.getEnvVarOwned(allocator, "NBODY_ITERATIONS");
    defer if (n_iters_str.len > 0) allocator.free(n_iters_str);
    const backend_str = try std.process.getEnvVarOwned(allocator, "NBODY_BACKEND");
    defer if (backend_str.len > 0) allocator.free(backend_str);
    const n_threads_str = try std.process.getEnvVarOwned(allocator, "NBODY_THREADS");
    defer if (n_threads_str.len > 0) allocator.free(n_threads_str);

    const n_threads = if (n_threads_str.len > 0)
        try std.fmt.parseInt(usize, n_threads_str, 10)
    else
        fu.countLogicalCores();

    const n_iters = if (n_iters_str.len > 0)
        try std.fmt.parseInt(usize, n_iters_str, 10)
    else
        1000;

    const n_bodies = if (n_bodies_opt.len > 0)
        try std.fmt.parseInt(usize, n_bodies_opt, 10)
    else
        n_threads;

    const backend = if (backend_str.len > 0) backend_str else "fork_union_static";

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
        var pool = try fu.Pool.init(allocator, n_threads, .inclusive);
        defer pool.deinit();

        for (0..n_iters) |_| {
            iterationForkUnionStatic(&pool, bodies, forces);
        }
    } else if (std.mem.eql(u8, backend, "fork_union_dynamic")) {
        var pool = try fu.Pool.init(allocator, n_threads, .inclusive);
        defer pool.deinit();

        for (0..n_iters) |_| {
            iterationForkUnionDynamic(&pool, bodies, forces);
        }
    } else if (std.mem.eql(u8, backend, "std_pool")) {
        var pool: std.Thread.Pool = undefined;
        try pool.init(.{ .allocator = allocator, .n_jobs = @intCast(n_threads) });
        defer pool.deinit();

        for (0..n_iters) |_| {
            try iterationStdPool(&pool, bodies, forces);
        }
    } else {
        std.debug.print("Unknown backend: {s}\n", .{backend});
        return error.UnknownBackend;
    }
}
