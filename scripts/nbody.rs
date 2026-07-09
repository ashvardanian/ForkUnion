//! Demo app: N-Body simulation with ForkUnion and Rayon.
//!
//! To control the script, several environment variables are used:
//!
//! - `NBODY_COUNT` - number of bodies in the simulation (default: number of threads).
//! - `NBODY_ITERATIONS` - number of iterations to run the simulation (default: 1000).
//! - `NBODY_BACKEND` - backend to use for the simulation (default: `forkunion_static`).
//! - `NBODY_THREADS` - number of threads to use for the simulation (default: number of hardware threads).
//!
//! The backends include: `forkunion_static`, `forkunion_dynamic`, `forkunion_iter_static`,
//! `forkunion_iter_dynamic`, `rayon_static`, `rayon_dynamic`, and `tokio`. With the `numa`
//! feature on Linux, `forkunion_numa_static` and `forkunion_numa_dynamic` are also available,
//! replicating the body positions into each compute domain's local memory. To compile and run:
//!
//! ```sh
//! cargo run --example nbody --release
//! ```
//!
//! The default profiling scheme is to 1M iterations for 128 particles on each backend.
//! First build the release binary, then benchmark each backend separately:
//!
//! ```sh
//! # Build once
//! cargo build --example nbody --release
//!
//! # Linux benchmarks (use nproc for CPU count)
//! time NBODY_COUNT=128 NBODY_THREADS=$(nproc) NBODY_ITERATIONS=1000000 \
//!     NBODY_BACKEND=rayon_static target/release/examples/nbody
//! time NBODY_COUNT=128 NBODY_THREADS=$(nproc) NBODY_ITERATIONS=1000000 \
//!     NBODY_BACKEND=rayon_dynamic target/release/examples/nbody
//! time NBODY_COUNT=128 NBODY_THREADS=$(nproc) NBODY_ITERATIONS=1000000 \
//!     NBODY_BACKEND=forkunion_static target/release/examples/nbody
//! time NBODY_COUNT=128 NBODY_THREADS=$(nproc) NBODY_ITERATIONS=1000000 \
//!     NBODY_BACKEND=forkunion_dynamic target/release/examples/nbody
//! time NBODY_COUNT=128 NBODY_THREADS=$(nproc) NBODY_ITERATIONS=1000000 \
//!     NBODY_BACKEND=forkunion_iter_static target/release/examples/nbody
//! time NBODY_COUNT=128 NBODY_THREADS=$(nproc) NBODY_ITERATIONS=1000000 \
//!     NBODY_BACKEND=forkunion_iter_dynamic target/release/examples/nbody
//! time NBODY_COUNT=128 NBODY_THREADS=$(nproc) NBODY_ITERATIONS=1000000 \
//!     NBODY_BACKEND=tokio target/release/examples/nbody
//!
//! # macOS benchmarks (use sysctl -n hw.logicalcpu for CPU count)
//! time NBODY_COUNT=128 NBODY_THREADS=$(sysctl -n hw.logicalcpu) NBODY_ITERATIONS=1000000 \
//!     NBODY_BACKEND=forkunion_iter_static target/release/examples/nbody
//! ```
use rand::{rng, Rng};
use std::env;
use std::error::Error;

use forkunion as fu;
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
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
    let l2 = dx * dx + dy * dy + dz * dz + SOFTEN_CONST;
    let inv = fast_rsqrt(l2);
    let inv3 = inv * inv * inv;
    let mag = G_CONST * bi.mass * bj.mass * inv3;
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
fn hw_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn iteration_fu_static(pool: &mut fu::ThreadPool, bodies: &mut [Body], forces: &mut [Vector3]) {
    let n = bodies.len();

    // First pass: calculate forces (need read access to bodies)
    {
        let bodies_ref = &*bodies; // Convert &mut [Body] to &[Body]
        fu::for_each_prong_mut(pool, forces, move |force, prong| {
            let bi = &bodies_ref[prong.task_index];
            let mut acc = Vector3::default();

            for bj in bodies_ref.iter().take(n) {
                acc += gravitational_force(bi, bj);
            }
            *force = acc;
        });
    } // bodies_ref goes out of scope here

    // Second pass: apply forces (need read access to forces)
    {
        let forces_ref = &*forces; // Convert &mut [Vector3] to &[Vector3]
        fu::for_each_prong_mut(pool, bodies, move |body, prong| {
            apply_force(body, &forces_ref[prong.task_index]);
        });
    }
}

fn iteration_fu_dynamic(pool: &mut fu::ThreadPool, bodies: &mut [Body], forces: &mut [Vector3]) {
    let n = bodies.len();

    // First pass: calculate forces (need read access to bodies)
    {
        let bodies_ref = &*bodies; // Convert &mut [Body] to &[Body]
        fu::for_each_prong_mut_dynamic(pool, forces, move |force, prong| {
            let bi = &bodies_ref[prong.task_index];
            let mut acc = Vector3::default();

            for bj in bodies_ref.iter().take(n) {
                acc += gravitational_force(bi, bj);
            }
            *force = acc;
        });
    } // bodies_ref goes out of scope here

    // Second pass: apply forces (need read access to forces)
    {
        let forces_ref = &*forces; // Convert &mut [Vector3] to &[Vector3]
        fu::for_each_prong_mut_dynamic(pool, bodies, move |body, prong| {
            apply_force(body, &forces_ref[prong.task_index]);
        });
    }
}

fn iteration_fu_iter_static(
    pool: &mut fu::ThreadPool,
    bodies: &mut [Body],
    forces: &mut [Vector3],
) {
    use fu::ParallelIteratorExt;
    let n = bodies.len();

    // First pass: calculate forces
    {
        let bodies_ref = &*bodies;
        fu::IntoParallelIterator::into_par_iter(&mut forces[..])
            .with_pool(pool)
            .for_each_with_prong(|force, prong| {
                let bi = &bodies_ref[prong.task_index];
                let mut acc = Vector3::default();
                for bj in bodies_ref.iter().take(n) {
                    acc += gravitational_force(bi, bj);
                }
                *force = acc;
            });
    }

    // Second pass: apply forces
    {
        let forces_ref = &*forces;
        fu::IntoParallelIterator::into_par_iter(&mut bodies[..])
            .with_pool(pool)
            .for_each_with_prong(|body, prong| {
                apply_force(body, &forces_ref[prong.task_index]);
            });
    }
}

fn iteration_fu_iter_dynamic(
    pool: &mut fu::ThreadPool,
    bodies: &mut [Body],
    forces: &mut [Vector3],
) {
    use fu::ParallelIteratorExt;
    let n = bodies.len();

    // First pass: calculate forces
    {
        let bodies_ref = &*bodies;
        fu::IntoParallelIterator::into_par_iter(&mut forces[..])
            .with_schedule(pool, fu::DynamicScheduler)
            .for_each_with_prong(|force, prong| {
                let bi = &bodies_ref[prong.task_index];
                let mut acc = Vector3::default();
                for bj in bodies_ref.iter().take(n) {
                    acc += gravitational_force(bi, bj);
                }
                *force = acc;
            });
    }

    // Second pass: apply forces
    {
        let forces_ref = &*forces;
        fu::IntoParallelIterator::into_par_iter(&mut bodies[..])
            .with_schedule(pool, fu::DynamicScheduler)
            .for_each_with_prong(|body, prong| {
                apply_force(body, &forces_ref[prong.task_index]);
            });
    }
}

/// One replica of the body positions per compute domain, each pinned to that domain's
/// local memory. Mirrors `make_buffers_for_forkunion_numa` in `nbody.cpp`.
#[cfg(feature = "numa")]
fn make_numa_replicas(bodies: &[Body]) -> Vec<fu::PinnedVec<Body>> {
    (0..fu::count_compute_domains())
        .map(|compute_domain| {
            let memory_domain = fu::local_memory_of(compute_domain);
            let allocator = fu::PinnedAllocator::new(memory_domain)
                .unwrap_or_else(|| panic!("No allocator for memory domain {memory_domain}"));
            let mut replica = fu::PinnedVec::with_capacity_in(allocator, bodies.len())
                .unwrap_or_else(|| panic!("Failed to pin {} bodies", bodies.len()));
            for body in bodies {
                replica.push(*body).expect("Capacity reserved above");
            }
            replica
        })
        .collect()
}

/// Mirrors `bodies` into every domain-local replica in a single broadcast.
///
/// Each thread copies only the slice it will later read, into only the replica of the domain it
/// runs on, so each page is written by a core that owns it. One barrier serves all replicas.
#[cfg(feature = "numa")]
fn refresh_numa_replicas(
    pool: &mut fu::ThreadPool,
    bodies: &[Body],
    replicas: &mut [fu::PinnedVec<Body>],
) {
    let n = bodies.len();
    let source = fu::SyncConstPtr::new(bodies.as_ptr());
    let targets: Vec<fu::SafePtr<Body>> = replicas
        .iter_mut()
        .map(|replica| fu::SafePtr::new(replica.as_mut_slice().as_mut_ptr()))
        .collect();
    let targets = &targets[..];

    pool.scope(|scope| {
        scope.broadcast(|thread_index, compute_domain_index| {
            let threads_here = scope.count_threads_in(compute_domain_index);
            let local_index = scope.locate_thread_in(thread_index, compute_domain_index);
            let range = fu::IndexedSplit::new(n, threads_here).get(local_index);
            if range.is_empty() {
                return;
            }
            // SAFETY: within a domain the split hands each thread a disjoint, in-bounds range, and
            // each domain writes only its own replica, so no two threads alias. `bodies` is read
            // only, and both it and `replicas` outlive the join.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    source.get(range.start) as *const Body,
                    targets[compute_domain_index].get_mut_at(range.start) as *mut Body,
                    range.len(),
                );
            }
        });
    });
}

/// Borrows every replica as a read-only, `Sync` base pointer, indexed by compute domain.
#[cfg(feature = "numa")]
fn numa_replica_ptrs(replicas: &[fu::PinnedVec<Body>]) -> Vec<fu::SyncConstPtr<Body>> {
    replicas
        .iter()
        .map(|replica| fu::SyncConstPtr::new(replica.as_slice().as_ptr()))
        .collect()
}

/// The all-to-all interaction cannot be sharded - every body reads every other body - so the only
/// locality left to win is the read side: replicate the positions once per iteration, then keep the
/// quadratic inner loop entirely inside the caller's own memory domain.
#[cfg(feature = "numa")]
fn iteration_fu_numa_static(
    pool: &mut fu::ThreadPool,
    bodies: &mut [Body],
    forces: &mut [Vector3],
    replicas: &mut [fu::PinnedVec<Body>],
) {
    let n = bodies.len();
    refresh_numa_replicas(pool, bodies, replicas);

    {
        let locals = numa_replica_ptrs(replicas);
        let locals = &locals[..];
        fu::for_each_prong_mut(pool, forces, move |force, prong| {
            // SAFETY: every replica holds `n` initialized bodies and is only read here; the
            // refresh above joined, and this dispatch joins before `replicas` is touched again.
            let local = unsafe {
                core::slice::from_raw_parts(locals[prong.compute_domain_index].as_ptr(), n)
            };
            let bi = &local[prong.task_index];
            let mut acc = Vector3::default();
            for bj in local {
                acc += gravitational_force(bi, bj);
            }
            *force = acc;
        });
    }

    {
        let forces_ref = &*forces;
        fu::for_each_prong_mut(pool, bodies, move |body, prong| {
            apply_force(body, &forces_ref[prong.task_index]);
        });
    }
}

/// Same as [`iteration_fu_numa_static`], differing only in the work-stealing schedule.
#[cfg(feature = "numa")]
fn iteration_fu_numa_dynamic(
    pool: &mut fu::ThreadPool,
    bodies: &mut [Body],
    forces: &mut [Vector3],
    replicas: &mut [fu::PinnedVec<Body>],
) {
    let n = bodies.len();
    refresh_numa_replicas(pool, bodies, replicas);

    {
        let locals = numa_replica_ptrs(replicas);
        let locals = &locals[..];
        fu::for_each_prong_mut_dynamic(pool, forces, move |force, prong| {
            // SAFETY: as in the static variant above.
            let local = unsafe {
                core::slice::from_raw_parts(locals[prong.compute_domain_index].as_ptr(), n)
            };
            let bi = &local[prong.task_index];
            let mut acc = Vector3::default();
            for bj in local {
                acc += gravitational_force(bi, bj);
            }
            *force = acc;
        });
    }

    {
        let forces_ref = &*forces;
        fu::for_each_prong_mut_dynamic(pool, bodies, move |body, prong| {
            apply_force(body, &forces_ref[prong.task_index]);
        });
    }
}

fn iteration_rayon_dynamic(pool: &ThreadPool, bodies: &mut [Body], forces: &mut [Vector3]) {
    let n = bodies.len();

    pool.install(|| {
        forces
            .par_iter_mut()
            .with_max_len(1)
            .enumerate()
            .for_each(|(i, force)| {
                let mut acc = Vector3::default();
                for j in 0..n {
                    acc += gravitational_force(&bodies[i], &bodies[j]);
                }
                *force = acc;
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

// "Static" scheduling: one *contiguous* stripe per thread, no stealing.
fn iteration_rayon_static(pool: &ThreadPool, bodies: &mut [Body], forces: &mut [Vector3]) {
    let n = bodies.len();
    let workers = rayon::current_num_threads();
    let stride = n.div_ceil(workers);

    pool.install(|| {
        forces
            .par_chunks_mut(stride)
            .enumerate()
            .for_each(|(chunk_idx, f_chunk)| {
                let start = chunk_idx * stride;

                for (local, force) in f_chunk.iter_mut().enumerate() {
                    let i = start + local;
                    let mut acc = Vector3::default();
                    for j in 0..n {
                        acc += gravitational_force(&bodies[i], &bodies[j]);
                    }
                    *force = acc;
                }
            });
    });

    pool.install(|| {
        bodies
            .par_chunks_mut(stride)
            .zip(forces.par_chunks(stride))
            .for_each(|(b_chunk, f_chunk)| {
                for (b, f) in b_chunk.iter_mut().zip(f_chunk.iter()) {
                    apply_force(b, f);
                }
            });
    });
}

async fn iteration_tokio_blocking(
    set: &mut JoinSet<(usize, Vector3)>,
    bodies: &mut [Body],
    forces: &mut [Vector3],
) {
    debug_assert!(set.is_empty());

    let n = bodies.len();
    let bodies_ptr = fu::SyncConstPtr::new(bodies.as_ptr()); // Send + Sync

    for i in 0..n {
        let ptr = bodies_ptr; // capture by value (Copy)
        set.spawn_blocking(move || unsafe {
            let bi = ptr.get(i); // &Body, immutable
            let mut acc = Vector3::default();
            for j in 0..n {
                acc += gravitational_force(bi, ptr.get(j));
            }
            (i, acc)
        });
    }

    while let Some(res) = set.join_next().await {
        let (idx, acc) = res.expect("task panicked");
        forces[idx] = acc;
    }

    // This part is sequential
    for (b, f) in bodies.iter_mut().zip(forces.iter()) {
        apply_force(b, f);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Every knob this script understands, read once, up front.
    let n = env::var("NBODY_COUNT").ok().and_then(|v| v.parse().ok());
    let iters = env::var("NBODY_ITERATIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_000);
    let backend = env::var("NBODY_BACKEND").unwrap_or_else(|_| "forkunion_static".into());
    let threads = env::var("NBODY_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(hw_threads);

    let bodies_n = n.unwrap_or(threads);

    // Allocate & initialize bodies
    let mut bodies = vec![
        Body {
            position: Vector3::default(),
            velocity: Vector3::default(),
            mass: 0.0
        };
        bodies_n
    ];
    let mut forces = vec![Vector3::default(); bodies_n];

    let mut generator = rng();
    bodies.iter_mut().for_each(|b| {
        // positions & velocities in [0, 1)
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

        // mass in [1 e20, 1 e25)
        b.mass = generator.random_range(1.0e20..1.0e25);
    });

    // Run the chosen backend
    match backend.as_str() {
        "forkunion_static" => {
            let mut pool = fu::ThreadPool::try_spawn(threads)
                .unwrap_or_else(|e| panic!("Failed to start Fork-Union pool: {e}"));
            for _ in 0..iters {
                iteration_fu_static(&mut pool, &mut bodies, &mut forces);
            }
        }
        "forkunion_dynamic" => {
            let mut pool = fu::ThreadPool::try_spawn(threads)
                .unwrap_or_else(|e| panic!("Failed to start Fork-Union pool: {e}"));
            for _ in 0..iters {
                iteration_fu_dynamic(&mut pool, &mut bodies, &mut forces);
            }
        }
        "forkunion_iter_static" => {
            let mut pool = fu::ThreadPool::try_spawn(threads)
                .unwrap_or_else(|e| panic!("Failed to start Fork-Union pool: {e}"));
            for _ in 0..iters {
                iteration_fu_iter_static(&mut pool, &mut bodies, &mut forces);
            }
        }
        "forkunion_iter_dynamic" => {
            let mut pool = fu::ThreadPool::try_spawn(threads)
                .unwrap_or_else(|e| panic!("Failed to start Fork-Union pool: {e}"));
            for _ in 0..iters {
                iteration_fu_iter_dynamic(&mut pool, &mut bodies, &mut forces);
            }
        }
        #[cfg(feature = "numa")]
        "forkunion_numa_static" => {
            let mut pool = fu::ThreadPool::try_spawn(threads)
                .unwrap_or_else(|e| panic!("Failed to start Fork-Union pool: {e}"));
            let mut replicas = make_numa_replicas(&bodies);
            for _ in 0..iters {
                iteration_fu_numa_static(&mut pool, &mut bodies, &mut forces, &mut replicas);
            }
        }
        #[cfg(feature = "numa")]
        "forkunion_numa_dynamic" => {
            let mut pool = fu::ThreadPool::try_spawn(threads)
                .unwrap_or_else(|e| panic!("Failed to start Fork-Union pool: {e}"));
            let mut replicas = make_numa_replicas(&bodies);
            for _ in 0..iters {
                iteration_fu_numa_dynamic(&mut pool, &mut bodies, &mut forces, &mut replicas);
            }
        }
        "rayon_static" => {
            let pool = ThreadPoolBuilder::new().num_threads(threads).build()?;
            for _ in 0..iters {
                iteration_rayon_static(&pool, &mut bodies, &mut forces);
            }
        }
        "rayon_dynamic" => {
            let pool = ThreadPoolBuilder::new().num_threads(threads).build()?;
            for _ in 0..iters {
                iteration_rayon_dynamic(&pool, &mut bodies, &mut forces);
            }
        }
        "tokio" => {
            let pool = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(threads)
                .max_blocking_threads(threads) // 1-to-1 with CPU cores
                .enable_all()
                .build()?;

            pool.block_on(async {
                let mut set = tokio::task::JoinSet::<(usize, Vector3)>::new();
                for _ in 0..iters {
                    iteration_tokio_blocking(&mut set, &mut bodies, &mut forces).await;
                    debug_assert!(set.is_empty());
                }
            });
        }

        _ => panic!("Unsupported backend: '{backend}'"),
    }

    Ok(())
}
