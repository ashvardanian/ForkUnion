//! Demo app: N-Body simulation with ForkUnion, Rayon, and Tokio.
//!
//! The environment variables it reads are listed in `bench/harness.rs`. The ForkUnion backends are
//! the four cells of `forkunion_{static,dynamic}_{shared,replicated}`, plus Rust-only
//! `forkunion_iter_{static,dynamic}_shared` that drive the same sweep through the parallel-iterator
//! adapters; the baselines are `rayon_static`, `rayon_dynamic`, and `tokio`. The `_replicated`
//! backends replicate the body positions into each memory domain's local storage - on a machine
//! with one domain the replicas collapse to one, so they run everywhere. Build and run:
//!
//! ```sh
//! cargo run --release --features benchmarks --bin forkunion_nbody
//! ```
//!
//! Each backend runs a fixed wall-clock window - 10 seconds by default - and reports the dispatch
//! rate it sustained. Contended-atomic paths amplify any background noise, and short dynamic runs
//! swing ~±30%, so the window sizes the iteration count to the machine instead of guessing it per
//! backend. Published comparisons run under `numactl --interleave=all`; the sibling C++ build also
//! adds `-ffast-math`, which Rust cannot express globally. First build the release binary (plain
//! `cargo build` grants neither native-CPU flag), then benchmark each backend separately:
//!
//! ```sh
//! # Build once
//! RUSTFLAGS="-C target-cpu=native" CXXFLAGS="-O3 -march=native" cargo build --release --features benchmarks
//!
//! # Benchmark each backend
//! FORKUNION_NBODY_COUNT=512 FORKUNION_BACKEND=rayon_static target/release/forkunion_nbody
//! FORKUNION_NBODY_COUNT=512 FORKUNION_BACKEND=forkunion_static_shared target/release/forkunion_nbody
//! FORKUNION_NBODY_COUNT=512 FORKUNION_BACKEND=forkunion_static_replicated target/release/forkunion_nbody
//! FORKUNION_NBODY_COUNT=512 FORKUNION_BACKEND=tokio target/release/forkunion_nbody
//! ```
//!
//! File: bench/nbody.rs
//! Author: Ash Vardanian
use std::error::Error;

use rayon::{prelude::*, ThreadPool as RayonPool, ThreadPoolBuilder};
use tokio::runtime::Runtime as TokioRuntime;
use tokio::task::JoinSet;

use forkunion as fu;
use forkunion::ParallelIteratorExt;

mod harness;
use harness::{env_parsed, Loop, Settings};

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

/// The SplitMix64 avalanche behind every random draw - a pure function of the `counter`.
///
/// Deliberately not `StdRng`: the standard generators differ across languages - and the C++
/// distributions even across standard libraries - so no two harnesses would simulate the same
/// system. Each draw is a pure function of its counter instead, and the bodies are bit-identical
/// across the C++, Rust, Zig, and Mojo ports of this hash.
#[inline]
fn split_mix(counter: u64) -> u64 {
    let mut x = counter.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Draw `counter` of stream `key` in `[0, 1)`: the top 24 bits scaled by 2^-24, exact in `f32`.
#[inline]
fn random_unit(key: u64, counter: u64) -> f32 {
    (split_mix(key.wrapping_add(counter)) >> 40) as f32 * (1.0 / 16777216.0)
}

/// Fast reciprocal square-root, one Newton step of the classic Quake hack.
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

/// How many independent accumulator chains the force sweep keeps. Reassociating a float reduction
/// is exactly what `-ffast-math` permits and strict IEEE forbids - so the reassociation is written
/// out by hand instead: the same eight lanes in the C++, Rust, and Zig kernels, reduced in the same
/// fixed order. Every compiler then faces the same strict-IEEE optimization problem with the same
/// freedom, and the language columns compare schedulers rather than compiler flag sets.
const FORCE_LANES: usize = 8;

/// Net gravitational force on `bi` over `bodies`, in eight explicit lanes.
#[inline]
fn net_force(bi: &Body, bodies: &[Body]) -> Vector3 {
    let mut fx = [0.0f32; FORCE_LANES];
    let mut fy = [0.0f32; FORCE_LANES];
    let mut fz = [0.0f32; FORCE_LANES];
    let mut chunks = bodies.chunks_exact(FORCE_LANES);
    for chunk in &mut chunks {
        for (lane, bj) in chunk.iter().enumerate() {
            let f = gravitational_force(bi, bj);
            fx[lane] += f.x;
            fy[lane] += f.y;
            fz[lane] += f.z;
        }
    }
    for (lane, bj) in chunks.remainder().iter().enumerate() {
        let f = gravitational_force(bi, bj);
        fx[lane] += f.x;
        fy[lane] += f.y;
        fz[lane] += f.z;
    }
    // The one reduction shape every language shares; changing it changes the bits.
    Vector3 {
        x: ((fx[0] + fx[1]) + (fx[2] + fx[3])) + ((fx[4] + fx[5]) + (fx[6] + fx[7])),
        y: ((fy[0] + fy[1]) + (fy[2] + fy[3])) + ((fy[4] + fy[5]) + (fy[6] + fy[7])),
        z: ((fz[0] + fz[1]) + (fz[2] + fz[3])) + ((fz[4] + fz[5]) + (fz[6] + fz[7])),
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
    // ? Wraps into the unit box to keep every distance - and every force - inside the normal `f32`
    // ? range forever: no overflows into NaN, and no denormals for x86 to stall on.
    b.position.x -= b.position.x.floor();
    b.position.y -= b.position.y.floor();
    b.position.z -= b.position.z.floor();
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

// Compile-time axes as marker types - stable Rust forbids a custom enum as a const-generic param.
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
fn iteration_fu_iter_static(pool: &mut fu::ThreadPool, bodies: &mut [Body], forces: &mut [Vector3]) {
    let n = bodies.len();
    {
        let bodies_ref = &*bodies;
        fu::IntoParallelIterator::into_par_iter(&mut forces[..])
            .with_pool(pool)
            .for_each(|force, task, _at| {
                let bi = &bodies_ref[task];
                *force = net_force(bi, &bodies_ref[..n]);
            })
            .expect("force sweep");
    }
    {
        let forces_ref = &*forces;
        fu::IntoParallelIterator::into_par_iter(&mut bodies[..])
            .with_pool(pool)
            .for_each(|body, task, _at| {
                apply_force(body, &forces_ref[task]);
            })
            .expect("apply sweep");
    }
}

/// The parallel-iterator sweep, work-stolen instead of split statically.
fn iteration_fu_iter_dynamic(pool: &mut fu::ThreadPool, bodies: &mut [Body], forces: &mut [Vector3]) {
    let n = bodies.len();
    {
        let bodies_ref = &*bodies;
        fu::IntoParallelIterator::into_par_iter(&mut forces[..])
            .with_schedule(pool, fu::DynamicScheduler)
            .for_each(|force, task, _at| {
                let bi = &bodies_ref[task];
                *force = net_force(bi, &bodies_ref[..n]);
            })
            .expect("force sweep");
    }
    {
        let forces_ref = &*forces;
        fu::IntoParallelIterator::into_par_iter(&mut bodies[..])
            .with_schedule(pool, fu::DynamicScheduler)
            .for_each(|body, task, _at| {
                apply_force(body, &forces_ref[task]);
            })
            .expect("apply sweep");
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
        let view = scope.view();
        scope.broadcast(|thread_index, compute_domain_index| {
            let memory_domain = topology
                .local_memory_of(fu::ComputeDomain(compute_domain_index))
                .expect("in-range domain");

            // Rank this thread among every thread on its memory domain, and count them, so the
            // node's whole team splits [0, n) without overlap even when several compute domains
            // share the node.
            let mut threads_on_memory_domain = 0usize;
            let mut local_index_on_memory_domain = 0usize;
            for other in 0..view.compute_domains_count() {
                if topology
                    .local_memory_of(fu::ComputeDomain(other))
                    .expect("in-range domain")
                    != memory_domain
                {
                    continue;
                }
                if other < compute_domain_index {
                    local_index_on_memory_domain += view.threads_count_in(other);
                }
                threads_on_memory_domain += view.threads_count_in(other);
            }
            local_index_on_memory_domain += view.locate_thread_in(thread_index, compute_domain_index);

            let range = fu::IndexedSplit::new(n, threads_on_memory_domain).get(local_index_on_memory_domain);
            if range.is_empty() {
                return;
            }
            // SAFETY: within a memory domain the split hands each thread a disjoint, in-bounds
            // range, and each node writes only its own replica, so no two threads alias. `bodies`
            // is read only, and both it and `replicas` outlive the join.
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

/// The read-only inputs one simulation step hands to every task, small and `Copy` so it moves into
/// a task closure for free. The pointers stand in for the `bodies` and `forces` slices, and - only
/// when the placement is replicated - `topology` and `replicas` bridge a compute domain to its
/// node-local copy.
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
            .local_memory_of(fu::ComputeDomain(compute_domain))
            .expect("in-range domain");
        work.replicas.expect("replicas").replica_ptr(memory_domain) as *const Body
    } else {
        work.bodies_ptr.as_ptr()
    };
    // SAFETY: both the canonical array and every replica hold `n` initialized bodies, read-only for
    // the duration of the force pass, which joins before the apply pass or the next refresh mutates
    // them once more.
    unsafe { core::slice::from_raw_parts(base, work.n) }
}

/// The all-to-all force on the body owning this task, summed over the array it reads.
#[inline]
fn force_kernel<P: Placement>(work: WorkCtx, task: usize, at: fu::ThreadInDomain) -> Vector3 {
    let local = bodies_at::<P>(work, at.compute_domain);
    let bi = &local[task];
    net_force(bi, local)
}

/// Integrates one canonical body by the force computed for it - identical for both placements.
#[inline]
fn apply_kernel(work: WorkCtx, body: &mut Body, task: usize, _at: fu::ThreadInDomain) {
    // SAFETY: `forces` holds `n` initialized elements, read-only while the apply pass mutates
    // `bodies`.
    let force = unsafe { work.forces_ptr.get(task) };
    apply_force(body, force);
}

/// Sweeps a mutating pass over `data`, split statically or work-stolen per the schedule axis.
#[inline]
fn for_each<S: Schedule, T: Send + Sync, F: Fn(&mut T, usize, fu::ThreadInDomain) + Sync + Send>(
    pool: &mut fu::ThreadPool,
    data: &mut [T],
    body: F,
) {
    if S::STATIC_SCHEDULE {
        fu::for_each_task_mut(pool, data, body);
    } else {
        fu::for_each_task_mut_dynamic(pool, data, body);
    }
}

/// One simulation step, specialized over the schedule and placement axes; the four ForkUnion
/// backends are its instantiations. The all-to-all sweep cannot be sharded - every body reads every
/// other - so the only locality to win is the read side: replicate the positions once per step,
/// then keep the quadratic loop node-local.
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
    for_each::<S, _, _>(pool, forces, move |force, task, at| {
        *force = force_kernel::<P>(work, task, at);
    });

    // Apply pass: integrate each canonical body by its force - identical for both placements.
    for_each::<S, _, _>(pool, bodies, move |body, task, at| {
        apply_kernel(work, body, task, at);
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
                    *force = net_force(bi, &bodies[..n]);
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
                *force = net_force(bi, &bodies[..n]);
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
async fn iteration_tokio_blocking(set: &mut JoinSet<(usize, Vector3)>, bodies: &mut [Body], forces: &mut [Vector3]) {
    debug_assert!(set.is_empty());
    let n = bodies.len();
    let bodies_ptr = fu::SyncConstPtr::new(bodies.as_ptr());

    for i in 0..n {
        let ptr = bodies_ptr;
        set.spawn_blocking(move || unsafe {
            let bi = ptr.get(i);
            let all = core::slice::from_raw_parts(ptr.get(0) as *const Body, n);
            (i, net_force(bi, all))
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

/// The backend named `name`, or `None` when `BACKENDS` has none by that name.
fn parse_backend(name: &str) -> Option<&'static Backend> {
    BACKENDS.iter().find(|backend| backend.name == name)
}

fn main() -> Result<(), Box<dyn Error>> {
    let probed = fu::Topology::new().expect("Failed to detect hardware topology");
    let settings = Settings::read(probed.logical_cores_count()?);
    let backend_names: Vec<&str> = BACKENDS.iter().map(|backend| backend.name).collect();
    let expected = format!("one of {}", backend_names.join(", "));
    let selected = env_parsed("FORKUNION_BACKEND", &BACKENDS[0], parse_backend, &expected);
    settings.print();
    let (bodies_n, threads) = (settings.bodies, settings.threads);

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

    // Seven counter-based draws per body: three position coordinates, three velocity components,
    // and one mass in [1e10, 1e15) - so every language starts from bit-identical bodies.
    let key = split_mix(settings.seed as u64);
    bodies.iter_mut().enumerate().for_each(|(i, b)| {
        let counter = i as u64 * 7;
        b.position = Vector3 {
            x: random_unit(key, counter),
            y: random_unit(key, counter + 1),
            z: random_unit(key, counter + 2),
        };
        b.velocity = Vector3 {
            x: random_unit(key, counter + 3),
            y: random_unit(key, counter + 4),
            z: random_unit(key, counter + 5),
        };
        b.mass = 1.0e10 + random_unit(key, counter + 6) * (1.0e15 - 1.0e10);
    });

    // Build only the engine resources the selected backend needs.
    let mut topology = None;
    let mut fu_pool = None;
    let mut replicas = None;
    let mut rayon_pool = None;
    let mut tokio_runtime = None;
    match selected.engine {
        Engine::ForkUnion | Engine::ForkUnionReplicated => {
            fu_pool = Some(
                fu::ThreadPool::spawn(&probed, threads)
                    .unwrap_or_else(|e| panic!("Failed to start Fork-Union pool: {e}")),
            );
            if selected.engine == Engine::ForkUnionReplicated {
                replicas = Some(
                    fu::ReplicatedArray::<Body>::new_in(&probed, bodies_n)
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

    // A fixed time budget beats a fixed iteration count: every backend runs the same wall-clock
    // window - long enough to amortize scheduling noise - and reports the rate it sustained, with
    // no per-backend iteration guessing. One call is one step: two dispatches over `n` bodies.
    let mut timed = Loop::new(settings.warmup, settings.time_limit);
    for _ in &mut timed {
        (selected.run)(&mut context);
    }
    timed.rate("bodies", bodies_n as f64);
    timed.row(selected.name).print();
    Ok(())
}
