//! Demo app: N-Body simulation with ForkUnion, Rayon, and Tokio.
//!
//! To control the script, several environment variables are used:
//!
//! - `NBODY_COUNT` - number of bodies in the simulation (default: number of threads).
//! - `NBODY_ITERATIONS` - number of iterations to run the simulation (default: 1000).
//! - `NBODY_BACKEND` - backend to use for the simulation (default: `forkunion_static_shared`).
//! - `NBODY_THREADS` - number of threads to use for the simulation (default: number of hardware threads).
//!
//! The ForkUnion backends are the four cells of `forkunion_{static,dynamic}_{shared,replicated}`, plus
//! Rust-only `forkunion_iter_{static,dynamic}_shared` that drive the same sweep through the
//! parallel-iterator adapters; the baselines are `rayon_static`, `rayon_dynamic`, and `tokio`. The
//! `_replicated` backends replicate the body positions into each memory domain's local storage - on a
//! machine with one domain the replicas collapse to one, so they run everywhere. To compile and run:
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
//!     NBODY_BACKEND=forkunion_static_shared target/release/examples/nbody
//! time NBODY_COUNT=128 NBODY_THREADS=$(nproc) NBODY_ITERATIONS=1000000 \
//!     NBODY_BACKEND=forkunion_static_replicated target/release/examples/nbody
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

// Compile-time axes as marker types - stable Rust cannot take a custom enum as a const-generic param.
// nbody is all-to-all, so there is no decomposition axis - only schedule and placement.
trait Schedule {
    const STATIC_SCHEDULE: bool;
}
struct Static;
struct Dynamic;
impl Schedule for Static {
    const STATIC_SCHEDULE: bool = true;
}
impl Schedule for Dynamic {
    const STATIC_SCHEDULE: bool = false;
}
trait Placement {
    const REPLICATED: bool;
}
struct Shared;
struct Replicated;
impl Placement for Shared {
    const REPLICATED: bool = false;
}
impl Placement for Replicated {
    const REPLICATED: bool = true;
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

/// The read-only inputs one simulation step hands to every task, small and `Copy` so it moves into a
/// task closure for free. The pointers stand in for the `bodies` and `forces` slices, and - only when the
/// placement is replicated - `topology` and `replicas` bridge a compute domain to its node-local copy.
#[derive(Copy, Clone)]
struct WorkCtx<'a> {
    bodies_ptr: fu::SyncConstPtr<Body>,
    forces_ptr: fu::SyncConstPtr<Vector3>,
    topology: Option<&'a fu::Topology>,
    replicas: Option<&'a fu::ReplicatedArray<Body>>,
    n: usize,
}

/// The `n` bodies a thread on `compute_domain` reads: the shared canonical array, or its node-local
/// replica when the placement is replicated.
#[inline]
fn bodies_at<'a, P: Placement>(work: WorkCtx<'a>, compute_domain: usize) -> &'a [Body] {
    let base = if P::REPLICATED {
        let memory_domain = work
            .topology
            .expect("topology")
            .local_memory_of(fu::ComputeDomain(compute_domain));
        work.replicas.expect("replicas").replica_ptr(memory_domain) as *const Body
    } else {
        work.bodies_ptr.as_ptr()
    };
    // SAFETY: both the canonical array and every replica hold `n` initialized bodies, read-only for the
    // duration of the force pass, which joins before the apply pass or the next refresh mutates them.
    unsafe { core::slice::from_raw_parts(base, work.n) }
}

/// The all-to-all force on the body owning this task, summed over the array it reads.
#[inline]
fn force_kernel<P: Placement>(work: WorkCtx, prong: fu::Prong) -> Vector3 {
    let local = bodies_at::<P>(work, prong.compute_domain_index);
    let bi = &local[prong.task_index];
    let mut accumulator = Vector3::default();
    for bj in local {
        accumulator += gravitational_force(bi, bj);
    }
    accumulator
}

/// Integrates one canonical body by the force computed for it - identical for both placements.
#[inline]
fn apply_kernel(work: WorkCtx, body: &mut Body, prong: fu::Prong) {
    // SAFETY: `forces` holds `n` initialized elements, read-only while the apply pass mutates `bodies`.
    let force = unsafe { work.forces_ptr.get(prong.task_index) };
    apply_force(body, force);
}

/// Sweeps a mutating pass over `data`, split statically or work-stolen per the schedule axis.
#[inline]
fn for_each<S: Schedule, T: Send + Sync, F: Fn(&mut T, fu::Prong) + Sync + Send>(
    pool: &mut fu::ThreadPool,
    data: &mut [T],
    body: F,
) {
    if S::STATIC_SCHEDULE {
        fu::for_each_prong_mut(pool, data, body);
    } else {
        fu::for_each_prong_mut_dynamic(pool, data, body);
    }
}

/// One simulation step, specialized over the schedule and placement axes; the four ForkUnion backends
/// are its instantiations. The all-to-all sweep cannot be sharded - every body reads every other - so the
/// only locality to win is the read side: replicate the positions once per step, then keep the quadratic
/// loop node-local.
fn iteration_forkunion<S: Schedule, P: Placement>(
    topology: &fu::Topology,
    pool: &mut fu::ThreadPool,
    bodies: &mut [Body],
    forces: &mut [Vector3],
    replicas: Option<&fu::ReplicatedArray<Body>>,
) {
    let n = bodies.len();
    if P::REPLICATED {
        refresh_replicas(topology, pool, replicas.expect("replicas"), bodies);
    }

    let work = WorkCtx {
        bodies_ptr: fu::SyncConstPtr::new(bodies.as_ptr()),
        forces_ptr: fu::SyncConstPtr::new(forces.as_ptr()),
        topology: if P::REPLICATED { Some(topology) } else { None },
        replicas: if P::REPLICATED { replicas } else { None },
        n,
    };

    // Force pass: all-to-all, reading the shared array or each thread's node-local replica.
    for_each::<S, _, _>(pool, forces, move |force, prong| {
        *force = force_kernel::<P>(work, prong);
    });

    // Apply pass: integrate each canonical body by its force - identical for both placements.
    for_each::<S, _, _>(pool, bodies, move |body, prong| {
        apply_kernel(work, body, prong);
    });
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
                    let bi = &bodies[start + local];
                    let mut accumulator = Vector3::default();
                    // Iterator form elides the per-element bounds check, like the ForkUnion backends above.
                    for bj in bodies.iter().take(n) {
                        accumulator += gravitational_force(bi, bj);
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
                let bi = &bodies[i];
                let mut accumulator = Vector3::default();
                // Iterator form elides the per-element bounds check, like the ForkUnion backends above.
                for bj in bodies.iter().take(n) {
                    accumulator += gravitational_force(bi, bj);
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

fn run_forkunion<S: Schedule, P: Placement>(c: &mut Ctx) {
    let pool = c.pool.as_deref_mut().expect("ForkUnion pool");
    let topology = c.topology.expect("topology");
    iteration_forkunion::<S, P>(topology, pool, c.bodies, c.forces, c.replicas);
}

fn run_forkunion_iter_static(c: &mut Ctx) {
    let pool = c.pool.as_deref_mut().expect("ForkUnion pool");
    iteration_fu_iter_static(pool, c.bodies, c.forces);
}

fn run_forkunion_iter_dynamic(c: &mut Ctx) {
    let pool = c.pool.as_deref_mut().expect("ForkUnion pool");
    iteration_fu_iter_dynamic(pool, c.bodies, c.forces);
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
        name: "forkunion_static_shared",
        run: run_forkunion::<Static, Shared>,
        engine: Engine::ForkUnion,
    },
    Backend {
        name: "forkunion_dynamic_shared",
        run: run_forkunion::<Dynamic, Shared>,
        engine: Engine::ForkUnion,
    },
    Backend {
        name: "forkunion_static_replicated",
        run: run_forkunion::<Static, Replicated>,
        engine: Engine::ForkUnionReplicated,
    },
    Backend {
        name: "forkunion_dynamic_replicated",
        run: run_forkunion::<Dynamic, Replicated>,
        engine: Engine::ForkUnionReplicated,
    },
    Backend {
        name: "forkunion_iter_static_shared",
        run: run_forkunion_iter_static,
        engine: Engine::ForkUnion,
    },
    Backend {
        name: "forkunion_iter_dynamic_shared",
        run: run_forkunion_iter_dynamic,
        engine: Engine::ForkUnion,
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
    let backend = env_string("NBODY_BACKEND", "forkunion_static_shared");
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
    let total_seconds = started.elapsed().as_secs_f64();
    let us_per_iter = total_seconds / iterations as f64 * 1e6;
    // Per-iteration latency is the comparable unit - one `for_each` dispatch over the bodies.
    println!("{backend}: {bodies_n} bodies, {iterations} iters, {us_per_iter:.2} us/iter ({total_seconds:.2} s total)");
    Ok(())
}
