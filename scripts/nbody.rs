//! Demo app: N-Body simulation with ForkUnion, Rayon, and Tokio.
//!
//! To control the script, several environment variables are used:
//!
//! - `NBODY_COUNT` - number of bodies in the simulation (default: number of threads).
//! - `NBODY_ITERATIONS` - number of iterations to run the simulation (default: 1000).
//! - `NBODY_BACKEND` - backend to use for the simulation (default: `forkunion_static`).
//! - `NBODY_THREADS` - number of threads to use for the simulation (default: number of hardware threads).
//!
//! The backends include: `forkunion_static`, `forkunion_dynamic`, `forkunion_iter_static`,
//! `forkunion_iter_dynamic`, `forkunion_replicated_static`, `forkunion_replicated_dynamic`,
//! `rayon_static`, `rayon_dynamic`, and `tokio`. The `forkunion_iter_*` backends are Rust-only - they
//! drive the same sweep through the parallel-iterator adapters. The `forkunion_replicated_*` backends
//! replicate the body positions into each memory domain's local storage - on a machine with one domain
//! the replicas collapse to one, so they run everywhere. To compile and run:
//!
//! ```sh
//! cargo run --example nbody --release
//! ```
//!
//! The default profiling scheme is 1M iterations for 128 particles on each backend. First build the
//! release binary, then benchmark each backend separately:
//!
//! ```sh
//! # Build once
//! cargo build --example nbody --release
//!
//! # Linux benchmarks (use nproc for CPU count)
//! time NBODY_COUNT=128 NBODY_THREADS=$(nproc) NBODY_ITERATIONS=1000000 \
//!     NBODY_BACKEND=rayon_static target/release/examples/nbody
//! time NBODY_COUNT=128 NBODY_THREADS=$(nproc) NBODY_ITERATIONS=1000000 \
//!     NBODY_BACKEND=forkunion_static target/release/examples/nbody
//! time NBODY_COUNT=128 NBODY_THREADS=$(nproc) NBODY_ITERATIONS=1000000 \
//!     NBODY_BACKEND=forkunion_replicated_static target/release/examples/nbody
//! time NBODY_COUNT=128 NBODY_THREADS=$(nproc) NBODY_ITERATIONS=1000000 \
//!     NBODY_BACKEND=tokio target/release/examples/nbody
//! ```
use rand::{Rng, SeedableRng};
use std::env;
use std::error::Error;
use std::time::Instant;

use forkunion as fu;
use fu::ParallelIteratorExt;
use rayon::{prelude::*, ThreadPool as RayonPool, ThreadPoolBuilder};
use tokio::runtime::Runtime as TokioRuntime;
use tokio::task::JoinSet;

/// Physical constants.
const G_CONST: f32 = 6.674e-11;
const DT_CONST: f32 = 0.01;
const SOFTEN_CONST: f32 = 1.0e-9;

/// Simple 3-vector used everywhere.
#[derive(Clone, Copy, Default)]
struct Vector3 {
    x: f32,
    y: f32,
    z: f32,
}

impl std::ops::AddAssign for Vector3 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

#[derive(Copy, Clone)]
struct Body {
    position: Vector3,
    velocity: Vector3,
    mass: f32,
}

/// Fast reciprocal square-root (one Newton step of the classic Quake hack).
#[inline]
fn fast_rsqrt(x: f32) -> f32 {
    let i = 0x5f37_59dfu32.wrapping_sub(x.to_bits() >> 1);
    let mut y = f32::from_bits(i);
    let x2 = 0.5 * x;
    y *= 1.5 - x2 * y * y;
    y
}

#[inline]
fn gravitational_force(bi: &Body, bj: &Body) -> Vector3 {
    let dx = bj.position.x - bi.position.x;
    let dy = bj.position.y - bi.position.y;
    let dz = bj.position.z - bi.position.z;
    let l2_squared = dx * dx + dy * dy + dz * dz + SOFTEN_CONST;
    let l2_reciprocal = fast_rsqrt(l2_squared);
    let l2_cube_reciprocal = l2_reciprocal * l2_reciprocal * l2_reciprocal;
    let mag = G_CONST * bi.mass * bj.mass * l2_cube_reciprocal;
    Vector3 {
        x: mag * dx,
        y: mag * dy,
        z: mag * dz,
    }
}

#[inline]
fn apply_force(b: &mut Body, f: &Vector3) {
    b.velocity.x += f.x / b.mass * DT_CONST;
    b.velocity.y += f.y / b.mass * DT_CONST;
    b.velocity.z += f.z / b.mass * DT_CONST;

    b.position.x += b.velocity.x * DT_CONST;
    b.position.y += b.velocity.y * DT_CONST;
    b.position.z += b.velocity.z * DT_CONST;
}

/// Return the number of logical CPUs visible to this process.
#[inline]
fn hardware_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Parses an unsigned environment variable, or `fallback` when unset or unparseable.
fn env_usize(name: &str, fallback: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
}

/// Reads an environment variable as a string, or `fallback` when unset.
fn env_string(name: &str, fallback: &str) -> String {
    env::var(name).unwrap_or_else(|_| fallback.into())
}

/// Whether an environment variable is present at all - part of the shared env-helper trio, kept for
/// parity with the other demos even though this one reads no boolean knobs.
#[allow(dead_code)]
fn env_flag(name: &str) -> bool {
    env::var(name).is_ok()
}

/// Everything a backend reads or writes for one simulation step; the harness owns the lifetimes and
/// hands each backend only the execution engine it asked for.
struct Ctx<'a> {
    bodies: &'a mut [Body],
    forces: &'a mut [Vector3],
    topology: Option<&'a fu::Topology>,
    pool: Option<&'a mut fu::ThreadPool>,
    replicas: Option<&'a fu::ReplicatedArray<Body>>,
    rayon: Option<&'a RayonPool>,
    tokio: Option<&'a TokioRuntime>,
}

/// All-to-all forces over the shared array, one task per body, split statically.
fn iteration_fu_static(pool: &mut fu::ThreadPool, bodies: &mut [Body], forces: &mut [Vector3]) {
    let n = bodies.len();
    {
        let bodies_ref = &*bodies;
        fu::for_each_prong_mut(pool, forces, move |force, prong| {
            let bi = &bodies_ref[prong.task_index];
            let mut accumulator = Vector3::default();
            for bj in bodies_ref.iter().take(n) {
                accumulator += gravitational_force(bi, bj);
            }
            *force = accumulator;
        });
    }
    {
        let forces_ref = &*forces;
        fu::for_each_prong_mut(pool, bodies, move |body, prong| {
            apply_force(body, &forces_ref[prong.task_index]);
        });
    }
}

/// The `forkunion_static` step, work-stolen instead of split statically.
fn iteration_fu_dynamic(pool: &mut fu::ThreadPool, bodies: &mut [Body], forces: &mut [Vector3]) {
    let n = bodies.len();
    {
        let bodies_ref = &*bodies;
        fu::for_each_prong_mut_dynamic(pool, forces, move |force, prong| {
            let bi = &bodies_ref[prong.task_index];
            let mut accumulator = Vector3::default();
            for bj in bodies_ref.iter().take(n) {
                accumulator += gravitational_force(bi, bj);
            }
            *force = accumulator;
        });
    }
    {
        let forces_ref = &*forces;
        fu::for_each_prong_mut_dynamic(pool, bodies, move |body, prong| {
            apply_force(body, &forces_ref[prong.task_index]);
        });
    }
}

/// The same all-to-all sweep, driven through the Rayon-style parallel-iterator adapters.
fn iteration_fu_iter_static(
    pool: &mut fu::ThreadPool,
    bodies: &mut [Body],
    forces: &mut [Vector3],
) {
    let n = bodies.len();
    {
        let bodies_ref = &*bodies;
        fu::IntoParallelIterator::into_par_iter(&mut forces[..])
            .with_pool(pool)
            .for_each_with_prong(|force, prong| {
                let bi = &bodies_ref[prong.task_index];
                let mut accumulator = Vector3::default();
                for bj in bodies_ref.iter().take(n) {
                    accumulator += gravitational_force(bi, bj);
                }
                *force = accumulator;
            });
    }
    {
        let forces_ref = &*forces;
        fu::IntoParallelIterator::into_par_iter(&mut bodies[..])
            .with_pool(pool)
            .for_each_with_prong(|body, prong| {
                apply_force(body, &forces_ref[prong.task_index]);
            });
    }
}

/// The parallel-iterator sweep, work-stolen instead of split statically.
fn iteration_fu_iter_dynamic(
    pool: &mut fu::ThreadPool,
    bodies: &mut [Body],
    forces: &mut [Vector3],
) {
    let n = bodies.len();
    {
        let bodies_ref = &*bodies;
        fu::IntoParallelIterator::into_par_iter(&mut forces[..])
            .with_schedule(pool, fu::DynamicScheduler)
            .for_each_with_prong(|force, prong| {
                let bi = &bodies_ref[prong.task_index];
                let mut accumulator = Vector3::default();
                for bj in bodies_ref.iter().take(n) {
                    accumulator += gravitational_force(bi, bj);
                }
                *force = accumulator;
            });
    }
    {
        let forces_ref = &*forces;
        fu::IntoParallelIterator::into_par_iter(&mut bodies[..])
            .with_schedule(pool, fu::DynamicScheduler)
            .for_each_with_prong(|body, prong| {
                apply_force(body, &forces_ref[prong.task_index]);
            });
    }
}

/// Copies canonical `bodies` into every per-domain replica, each written by the cores local to its
/// node so the pages first-touch there. Every compute domain sharing a memory domain cooperates on
/// that node's one replica, partitioned across all its threads so no element is copied twice.
fn refresh_replicas(
    topology: &fu::Topology,
    pool: &mut fu::ThreadPool,
    replicas: &fu::ReplicatedArray<Body>,
    bodies: &[Body],
) {
    let n = bodies.len();
    let source = fu::SyncConstPtr::new(bodies.as_ptr());
    pool.scope(|scope| {
        scope.broadcast(|thread_index, compute_domain_index| {
            let memory_domain = topology.local_memory_of(fu::ComputeDomain(compute_domain_index));

            // Rank this thread among every thread on its memory domain, and count them, so the node's
            // whole team splits [0, n) without overlap even when several compute domains share the node.
            let mut threads_on_memory_domain = 0usize;
            let mut local_index_on_memory_domain = 0usize;
            for other in 0..scope.compute_domains_count() {
                if topology.local_memory_of(fu::ComputeDomain(other)) != memory_domain {
                    continue;
                }
                if other < compute_domain_index {
                    local_index_on_memory_domain += scope.threads_count_in(other);
                }
                threads_on_memory_domain += scope.threads_count_in(other);
            }
            local_index_on_memory_domain +=
                scope.locate_thread_in(thread_index, compute_domain_index);

            let range = fu::IndexedSplit::new(n, threads_on_memory_domain)
                .get(local_index_on_memory_domain);
            if range.is_empty() {
                return;
            }
            // SAFETY: within a memory domain the split hands each thread a disjoint, in-bounds range,
            // and each node writes only its own replica, so no two threads alias. `bodies` is read
            // only, and both it and `replicas` outlive the join.
            let replica = replicas.replica_ptr(memory_domain);
            unsafe {
                core::ptr::copy_nonoverlapping(
                    source.get(range.start) as *const Body,
                    replica.add(range.start),
                    range.len(),
                );
            }
        });
    });
}

/// Borrows every replica as a read-only, `Sync` base pointer, indexed by compute domain, so the hot
/// loop never calls back into the topology to bridge compute to memory.
fn replicated_domain_ptrs(
    topology: &fu::Topology,
    replicas: &fu::ReplicatedArray<Body>,
) -> Vec<fu::SyncConstPtr<Body>> {
    (0..topology.compute_domains_count())
        .map(|compute_domain| {
            let memory_domain = topology.local_memory_of(fu::ComputeDomain(compute_domain));
            fu::SyncConstPtr::new(replicas.replica_ptr(memory_domain) as *const Body)
        })
        .collect()
}

/// The all-to-all interaction cannot be sharded - every body reads every other body - so the only
/// locality left to win is the read side: replicate the positions once per iteration, then keep the
/// quadratic inner loop entirely inside the caller's own memory domain.
fn iteration_fu_replicated_static(
    topology: &fu::Topology,
    pool: &mut fu::ThreadPool,
    bodies: &mut [Body],
    forces: &mut [Vector3],
    replicas: &fu::ReplicatedArray<Body>,
) {
    let n = bodies.len();
    refresh_replicas(topology, pool, replicas, bodies);
    {
        let locals = replicated_domain_ptrs(topology, replicas);
        let locals = &locals[..];
        fu::for_each_prong_mut(pool, forces, move |force, prong| {
            // SAFETY: every replica holds `n` initialized bodies and is only read here; the refresh
            // above joined, and this dispatch joins before `replicas` is touched again.
            let local = unsafe {
                core::slice::from_raw_parts(locals[prong.compute_domain_index].as_ptr(), n)
            };
            let bi = &local[prong.task_index];
            let mut accumulator = Vector3::default();
            for bj in local {
                accumulator += gravitational_force(bi, bj);
            }
            *force = accumulator;
        });
    }
    {
        let forces_ref = &*forces;
        fu::for_each_prong_mut(pool, bodies, move |body, prong| {
            apply_force(body, &forces_ref[prong.task_index]);
        });
    }
}

/// The replicated step, work-stolen instead of split statically.
fn iteration_fu_replicated_dynamic(
    topology: &fu::Topology,
    pool: &mut fu::ThreadPool,
    bodies: &mut [Body],
    forces: &mut [Vector3],
    replicas: &fu::ReplicatedArray<Body>,
) {
    let n = bodies.len();
    refresh_replicas(topology, pool, replicas, bodies);
    {
        let locals = replicated_domain_ptrs(topology, replicas);
        let locals = &locals[..];
        fu::for_each_prong_mut_dynamic(pool, forces, move |force, prong| {
            // SAFETY: as in the static variant above.
            let local = unsafe {
                core::slice::from_raw_parts(locals[prong.compute_domain_index].as_ptr(), n)
            };
            let bi = &local[prong.task_index];
            let mut accumulator = Vector3::default();
            for bj in local {
                accumulator += gravitational_force(bi, bj);
            }
            *force = accumulator;
        });
    }
    {
        let forces_ref = &*forces;
        fu::for_each_prong_mut_dynamic(pool, bodies, move |body, prong| {
            apply_force(body, &forces_ref[prong.task_index]);
        });
    }
}

/// One contiguous stripe per worker, no stealing - Rayon's `par_chunks_mut`.
fn iteration_rayon_static(pool: &RayonPool, bodies: &mut [Body], forces: &mut [Vector3]) {
    let n = bodies.len();
    let workers = pool.current_num_threads();
    let stride = n.div_ceil(workers);

    pool.install(|| {
        forces
            .par_chunks_mut(stride)
            .enumerate()
            .for_each(|(chunk_index, force_chunk)| {
                let start = chunk_index * stride;
                for (local, force) in force_chunk.iter_mut().enumerate() {
                    let i = start + local;
                    let mut accumulator = Vector3::default();
                    for j in 0..n {
                        accumulator += gravitational_force(&bodies[i], &bodies[j]);
                    }
                    *force = accumulator;
                }
            });
    });

    pool.install(|| {
        bodies
            .par_chunks_mut(stride)
            .zip(forces.par_chunks(stride))
            .for_each(|(body_chunk, force_chunk)| {
                for (b, f) in body_chunk.iter_mut().zip(force_chunk.iter()) {
                    apply_force(b, f);
                }
            });
    });
}

/// One body per work item, work-stolen - Rayon's `par_iter_mut` with a unit grain.
fn iteration_rayon_dynamic(pool: &RayonPool, bodies: &mut [Body], forces: &mut [Vector3]) {
    let n = bodies.len();

    pool.install(|| {
        forces
            .par_iter_mut()
            .with_max_len(1)
            .enumerate()
            .for_each(|(i, force)| {
                let mut accumulator = Vector3::default();
                for j in 0..n {
                    accumulator += gravitational_force(&bodies[i], &bodies[j]);
                }
                *force = accumulator;
            });
    });

    pool.install(|| {
        bodies
            .par_iter_mut()
            .with_max_len(1)
            .zip(forces.par_iter())
            .for_each(|(b, f)| apply_force(b, f));
    });
}

/// One `spawn_blocking` task per body over a shared read-only snapshot, joined into `forces`.
async fn iteration_tokio_blocking(
    set: &mut JoinSet<(usize, Vector3)>,
    bodies: &mut [Body],
    forces: &mut [Vector3],
) {
    debug_assert!(set.is_empty());
    let n = bodies.len();
    let bodies_ptr = fu::SyncConstPtr::new(bodies.as_ptr());

    for i in 0..n {
        let ptr = bodies_ptr;
        set.spawn_blocking(move || unsafe {
            let bi = ptr.get(i);
            let mut accumulator = Vector3::default();
            for j in 0..n {
                accumulator += gravitational_force(bi, ptr.get(j));
            }
            (i, accumulator)
        });
    }

    while let Some(result) = set.join_next().await {
        let (idx, accumulator) = result.expect("task panicked");
        forces[idx] = accumulator;
    }

    for (b, f) in bodies.iter_mut().zip(forces.iter()) {
        apply_force(b, f);
    }
}

// The registry entries - each unwraps only the engine its backend was set up with.

fn run_forkunion_static(c: &mut Ctx) {
    let pool = c.pool.as_deref_mut().expect("ForkUnion pool");
    iteration_fu_static(pool, c.bodies, c.forces);
}

fn run_forkunion_dynamic(c: &mut Ctx) {
    let pool = c.pool.as_deref_mut().expect("ForkUnion pool");
    iteration_fu_dynamic(pool, c.bodies, c.forces);
}

fn run_forkunion_iter_static(c: &mut Ctx) {
    let pool = c.pool.as_deref_mut().expect("ForkUnion pool");
    iteration_fu_iter_static(pool, c.bodies, c.forces);
}

fn run_forkunion_iter_dynamic(c: &mut Ctx) {
    let pool = c.pool.as_deref_mut().expect("ForkUnion pool");
    iteration_fu_iter_dynamic(pool, c.bodies, c.forces);
}

fn run_forkunion_replicated_static(c: &mut Ctx) {
    let pool = c.pool.as_deref_mut().expect("ForkUnion pool");
    let replicas = c.replicas.expect("replicas");
    let topology = c.topology.expect("topology");
    iteration_fu_replicated_static(topology, pool, c.bodies, c.forces, replicas);
}

fn run_forkunion_replicated_dynamic(c: &mut Ctx) {
    let pool = c.pool.as_deref_mut().expect("ForkUnion pool");
    let replicas = c.replicas.expect("replicas");
    let topology = c.topology.expect("topology");
    iteration_fu_replicated_dynamic(topology, pool, c.bodies, c.forces, replicas);
}

fn run_rayon_static(c: &mut Ctx) {
    let pool = c.rayon.expect("Rayon pool");
    iteration_rayon_static(pool, c.bodies, c.forces);
}

fn run_rayon_dynamic(c: &mut Ctx) {
    let pool = c.rayon.expect("Rayon pool");
    iteration_rayon_dynamic(pool, c.bodies, c.forces);
}

fn run_tokio(c: &mut Ctx) {
    let runtime = c.tokio.expect("Tokio runtime");
    let bodies = &mut *c.bodies;
    let forces = &mut *c.forces;
    runtime.block_on(async {
        let mut set = JoinSet::new();
        iteration_tokio_blocking(&mut set, bodies, forces).await;
    });
}

/// Which execution engine a backend runs on, so `main` builds exactly the resource it needs.
#[derive(Copy, Clone, PartialEq)]
enum Engine {
    ForkUnion,
    ForkUnionReplicated,
    Rayon,
    Tokio,
}

/// The dispatch table - a name, its per-step function, and the engine it runs on.
struct Backend {
    name: &'static str,
    run: fn(&mut Ctx),
    engine: Engine,
}

const BACKENDS: &[Backend] = &[
    Backend {
        name: "forkunion_static",
        run: run_forkunion_static,
        engine: Engine::ForkUnion,
    },
    Backend {
        name: "forkunion_dynamic",
        run: run_forkunion_dynamic,
        engine: Engine::ForkUnion,
    },
    Backend {
        name: "forkunion_iter_static",
        run: run_forkunion_iter_static,
        engine: Engine::ForkUnion,
    },
    Backend {
        name: "forkunion_iter_dynamic",
        run: run_forkunion_iter_dynamic,
        engine: Engine::ForkUnion,
    },
    Backend {
        name: "forkunion_replicated_static",
        run: run_forkunion_replicated_static,
        engine: Engine::ForkUnionReplicated,
    },
    Backend {
        name: "forkunion_replicated_dynamic",
        run: run_forkunion_replicated_dynamic,
        engine: Engine::ForkUnionReplicated,
    },
    Backend {
        name: "rayon_static",
        run: run_rayon_static,
        engine: Engine::Rayon,
    },
    Backend {
        name: "rayon_dynamic",
        run: run_rayon_dynamic,
        engine: Engine::Rayon,
    },
    Backend {
        name: "tokio",
        run: run_tokio,
        engine: Engine::Tokio,
    },
];

fn main() -> Result<(), Box<dyn Error>> {
    // Every knob this script understands, read once, up front.
    let count = env_usize("NBODY_COUNT", 0);
    let iterations = env_usize("NBODY_ITERATIONS", 1_000);
    let backend = env_string("NBODY_BACKEND", "forkunion_static");
    let mut threads = env_usize("NBODY_THREADS", 0);
    if threads == 0 {
        threads = hardware_threads();
    }
    let bodies_n = if count == 0 { threads } else { count };

    // Allocate and initialize bodies.
    let mut bodies = vec![
        Body {
            position: Vector3::default(),
            velocity: Vector3::default(),
            mass: 0.0
        };
        bodies_n
    ];
    let mut forces = vec![Vector3::default(); bodies_n];

    // A fixed seed - the benchmark only needs a spread of positions, not entropy, and every backend
    // must start from the same bodies to be comparable.
    let mut generator = rand::rngs::StdRng::seed_from_u64(0x1234_5678_9abc_def0);
    bodies.iter_mut().for_each(|b| {
        b.position = Vector3 {
            x: generator.random(),
            y: generator.random(),
            z: generator.random(),
        };
        b.velocity = Vector3 {
            x: generator.random(),
            y: generator.random(),
            z: generator.random(),
        };
        b.mass = generator.random_range(1.0e20..1.0e25);
    });

    let selected = match BACKENDS.iter().find(|b| b.name == backend) {
        Some(entry) => entry,
        None => {
            eprintln!("Unsupported backend: '{backend}'");
            eprint!("Available backends:");
            for entry in BACKENDS {
                eprint!(" {}", entry.name);
            }
            eprintln!();
            return Err(format!("Unsupported backend: '{backend}'").into());
        }
    };

    // Build only the engine resources the selected backend needs; only the ForkUnion engines probe
    // the topology, so the Rayon and Tokio backends never touch it.
    let mut topology = None;
    let mut fu_pool = None;
    let mut replicas = None;
    let mut rayon_pool = None;
    let mut tokio_runtime = None;
    match selected.engine {
        Engine::ForkUnion | Engine::ForkUnionReplicated => {
            let probed = fu::Topology::new().expect("Failed to detect hardware topology");
            fu_pool = Some(
                fu::ThreadPool::try_spawn(&probed, threads)
                    .unwrap_or_else(|e| panic!("Failed to start Fork-Union pool: {e}")),
            );
            if selected.engine == Engine::ForkUnionReplicated {
                replicas = Some(
                    fu::ReplicatedArray::<Body>::try_new(&probed, bodies_n)
                        .expect("Failed to allocate per-domain body replicas"),
                );
            }
            topology = Some(probed);
        }
        Engine::Rayon => {
            rayon_pool = Some(ThreadPoolBuilder::new().num_threads(threads).build()?);
        }
        Engine::Tokio => {
            tokio_runtime = Some(
                tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(threads)
                    .max_blocking_threads(threads)
                    .enable_all()
                    .build()?,
            );
        }
    }

    let mut context = Ctx {
        bodies: &mut bodies,
        forces: &mut forces,
        topology: topology.as_ref(),
        pool: fu_pool.as_mut(),
        replicas: replicas.as_ref(),
        rayon: rayon_pool.as_ref(),
        tokio: tokio_runtime.as_ref(),
    };

    let started = Instant::now();
    for _ in 0..iterations {
        (selected.run)(&mut context);
    }
    let seconds = started.elapsed().as_secs_f64() / iterations as f64;
    println!("{backend}: {bodies_n} bodies, {iterations} iters in {seconds:.3} s");
    Ok(())
}
