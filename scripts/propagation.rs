//! Demo app: Connected Components by label propagation, with ForkUnion and Rayon.
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
//! identical across schedules, backends, thread counts, and languages.
//!
//! To control the script, several environment variables are used:
//!
//! - `PROPAGATION_SCALE` - each community has `2^scale` vertices - default 14.
//! - `PROPAGATION_COMMUNITIES` - communities strung on the ring - default 64.
//! - `PROPAGATION_EDGE_FACTOR` - edges generated per vertex, before deduplication - default 16.
//! - `PROPAGATION_BACKEND` - backend to use - default `forkunion_static_shared`.
//! - `PROPAGATION_THREADS` - number of threads to use - default all hardware threads.
//! - `PROPAGATION_SECONDS` - wall-clock budget per run, reporting the sustained rate - default 10.
//! - `PROPAGATION_ITERATIONS` - run an exact pass count instead, when set.
//! - `PROPAGATION_CHECK` - also converge serially, and fail unless labels and rounds agree exactly.
//!
//! The ForkUnion backends are the four cells of `forkunion_{static,dynamic}_{shared,replicated}`; the
//! baselines are `rayon_static` and `rayon_dynamic`. To compile and run:
//!
//! ```sh
//! RUSTFLAGS="-C target-cpu=native" CXXFLAGS="-O3 -march=native" cargo build --release --features benchmarks
//! PROPAGATION_BACKEND=forkunion_static_shared target/release/forkunion_propagation
//! ```
use std::env;
use std::error::Error;
use std::time::Instant;

use forkunion as fu;
use fu::SyncMutPtr;
use rayon::prelude::*;
use rayon::{ThreadPool as RayonPool, ThreadPoolBuilder};

/// A component name: the smallest vertex index reachable so far.
type Label = u32;
/// Per-thread change tally, cache-aligned so two threads never share a line.
type Counter = fu::CacheAligned<u64>;

/// A read-only CSR as two slices - the interface every kernel takes.
#[derive(Copy, Clone)]
struct CsrView<'a> {
    row_offsets: &'a [u64],
    column_indices: &'a [u32],
}

impl CsrView<'_> {
    fn vertices(&self) -> u32 {
        (self.row_offsets.len() - 1) as u32
    }

    fn edges(&self) -> usize {
        self.column_indices.len()
    }
}

/// The two CSR arrays built once on the host.
struct CsrHost {
    row_offsets: Vec<u64>,
    column_indices: Vec<u32>,
}

impl CsrHost {
    fn view(&self) -> CsrView<'_> {
        CsrView {
            row_offsets: &self.row_offsets,
            column_indices: &self.column_indices,
        }
    }
}

/// Sorts past every valid edge; marks dropped self-loops, trimmed together with the duplicates.
const SENTINEL_EDGE: (u32, u32) = (u32::MAX, u32::MAX);

/// The SplitMix64 avalanche behind every random draw - a pure function of the `counter`.
///
/// A counter-based generator instead of a stateful one: each draw is a pure function of its counter,
/// so iterations are order-free, the fill parallelizes without sharding generator state, and the graph
/// is bit-identical at any thread count - and across the C++, Rust, and Zig ports of this hash.
#[inline]
fn split_mix(counter: u64) -> u64 {
    let mut x = counter.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// One quadrant choice in `[0, 100)` - same draw and counter scheme as every sibling benchmark.
#[inline]
fn random_percent(counter: u64) -> u32 {
    (split_mix(counter) % 100) as u32
}

/// One bridge endpoint in `[0, bound)`, from the same avalanche.
#[inline]
fn random_index(counter: u64, bound: u32) -> u32 {
    (split_mix(counter) % bound as u64) as u32
}

/// Generates the necklace: `communities` independent R-MAT graphs of `2^scale` vertices, joined in a
/// ring by one bridge per neighbouring pair, and scatters it all into a CSR.
///
/// Community `c` owns global edge indices `[c * raw_local, (c+1) * raw_local)` and the vertex range
/// `[c << scale, (c+1) << scale)`; the quadrant walk uses the same `e * 64 + bit` counters as the
/// single-graph generators. Bridge draws live in their own counter range above all edge draws.
fn generate_necklace(scale: usize, communities: usize, edge_factor: usize) -> CsrHost {
    let community_vertices = 1usize << scale;
    let vertices = communities * community_vertices;
    let raw_local = community_vertices * edge_factor;
    let raw_edges = communities * raw_local;
    let bridges = if communities > 1 { communities } else { 0 };

    // The pair of slots `2e, 2e+1` belongs to edge `e`; self-loops stay sentinels.
    let mut edges: Vec<(u32, u32)> = vec![SENTINEL_EDGE; raw_edges * 2 + bridges * 2];
    let (rmat_slots, bridge_slots) = edges.split_at_mut(raw_edges * 2);
    rmat_slots
        .par_chunks_mut(2)
        .enumerate()
        .for_each(|(e, slots)| {
            let mut row = 0u32;
            let mut column = 0u32;
            for bit in (0..scale).rev() {
                let r = random_percent((e * 64 + bit) as u64); // a=57 b=19 c=19 d=5, integer and portable
                let step = 1u32 << bit;
                if r < 57 {
                    continue; // Stay in the dense quadrant
                } else if r < 76 {
                    column |= step;
                } else if r < 95 {
                    row |= step;
                } else {
                    row |= step;
                    column |= step;
                }
            }
            if row != column {
                let base = ((e / raw_local) << scale) as u32; // This community's vertex range
                slots[0] = (base + row, base + column); // Symmetrize; self-loops stay sentinels
                slots[1] = (base + column, base + row);
            }
        });

    // Bridges: endpoints in each community's first 64 vertices - R-MAT's quadrant bias piles the
    // hubs at low indices, so a low endpoint is essentially guaranteed well-connected.
    let hub_core = community_vertices.min(64) as u32;
    let bridge_base = raw_edges as u64 * 64;
    for j in 0..bridges {
        let u = ((j << scale) as u32) + random_index(bridge_base + 2 * j as u64, hub_core);
        let v = ((((j + 1) % communities) << scale) as u32)
            + random_index(bridge_base + 2 * j as u64 + 1, hub_core);
        bridge_slots[j * 2] = (u, v);
        bridge_slots[j * 2 + 1] = (v, u);
    }

    edges.par_sort_unstable(); // Rayon is already linked; the keys form one multiset either way
    edges.dedup();
    // At most one sentinel survives `dedup`, at the very end - trim it with the duplicates.
    while edges.last() == Some(&SENTINEL_EDGE) {
        edges.pop();
    }

    // CSR: count the degrees into `row_offsets`, then prefix-sum them into row starts.
    let mut row_offsets = vec![0u64; vertices + 1];
    for &(row, _) in &edges {
        row_offsets[row as usize + 1] += 1;
    }
    for v in 0..vertices {
        row_offsets[v + 1] += row_offsets[v];
    }
    let mut column_indices = vec![0u32; edges.len()];
    let mut cursor: Vec<u64> = row_offsets[..vertices].to_vec();
    for &(row, column) in &edges {
        column_indices[cursor[row as usize] as usize] = column;
        cursor[row as usize] += 1;
    }

    CsrHost {
        row_offsets,
        column_indices,
    }
}

/// The smallest label visible from `v`: its own, or the smallest among its neighbours'.
#[inline]
fn min_label_of(graph: &CsrView, old_labels: &[Label], v: u32) -> Label {
    // SAFETY: every column index is a valid vertex by CSR construction; unchecked indexing keeps
    // this loop identical to the C++ kernel it is benchmarked against.
    let mut best = old_labels[v as usize];
    let begin = graph.row_offsets[v as usize] as usize;
    let end = graph.row_offsets[v as usize + 1] as usize;
    for position in begin..end {
        let neighbor = unsafe { *graph.column_indices.get_unchecked(position) };
        let candidate = unsafe { *old_labels.get_unchecked(neighbor as usize) };
        if candidate < best {
            best = candidate;
        }
    }
    best
}

/// Converges serially from `labels[v] = v`, returning the rounds taken - the reference for `PROPAGATION_CHECK`.
fn converge_serially(graph: &CsrView, labels_a: &mut [Label], labels_b: &mut [Label]) -> usize {
    let vertices = graph.vertices() as usize;
    for (v, label) in labels_a.iter_mut().enumerate() {
        *label = v as Label;
    }
    let mut rounds = 0usize;
    let (mut old_labels, mut new_labels) = (labels_a, labels_b);
    loop {
        let mut changed = false;
        for v in 0..vertices {
            let next = min_label_of(graph, old_labels, v as u32);
            new_labels[v] = next;
            changed |= next != old_labels[v];
        }
        rounds += 1;
        std::mem::swap(&mut old_labels, &mut new_labels);
        if !changed {
            break;
        }
    }
    rounds
}

/// Which execution engine a backend runs on, so `main` builds exactly the resource it needs.
#[derive(Copy, Clone, PartialEq)]
enum Engine {
    ForkUnion,
    ForkUnionReplicated, // ? Also builds the per-node CSR replicas
    Rayon,
}

/// One read-only replica of the CSR per memory domain, so no adjacency is ever remote. Only the
/// immutable CSR replicates; the two label buffers stay shared by nature - every round reads
/// remote labels through the bridges and writes its own slot.
struct ReplicatedCsr {
    row_offsets: fu::ReplicatedArray<u64>,
    column_indices: fu::ReplicatedArray<u32>,
}

/// Fills every memory domain's replica of `host`, each range copied by a thread on the owning node
/// so the pages first-touch there - the same node-team partitioning as `nbody.rs`'s `refresh_replicas`.
/// Rewrites `values` into fresh pages, first-touched by the pinned pool's static split.
///
/// Generation first-touches pages on whichever cores the OS handed the unpinned worker threads, so
/// every process rolls a different page placement and throughput swings ~2x run to run. Copying into
/// virgin pages from the static split of *pinned* threads makes placement a pure function of the
/// topology - identical for every backend, process, and language.
fn retouch_deterministically<T: Copy + Sync>(pool: &mut fu::ThreadPool, values: &mut Vec<T>) {
    let n = values.len();
    let threads = pool.threads_count();
    let mut placed: Vec<T> = Vec::with_capacity(n); // ? Pages stay unfaulted until the copy below
    let source = fu::SyncConstPtr::new(values.as_ptr());
    let destination = fu::SyncMutPtr::new(placed.as_mut_ptr());
    pool.scope(|scope| {
        scope.broadcast(|thread_index, _| {
            let range = fu::IndexedSplit::new(n, threads).get(thread_index);
            if range.is_empty() {
                return;
            }
            // SAFETY: the split hands each thread a disjoint, in-bounds range; `placed` outlives the
            // join, and every element is written here before `set_len` exposes it.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    source.get(range.start),
                    destination.get(range.start) as *mut T,
                    range.len(),
                );
            }
        });
    });
    // SAFETY: all `n` elements were just written by the broadcast above.
    unsafe { placed.set_len(n) };
    *values = placed;
}

fn replicate_into<T: Copy + Sync>(
    topology: &fu::Topology,
    pool: &mut fu::ThreadPool,
    replicas: &fu::ReplicatedArray<T>,
    host: &[T],
) {
    let n = host.len();
    let source = fu::SyncConstPtr::new(host.as_ptr());
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
            // and each node writes only its own replica, so no two threads alias. `host` is read
            // only, and both it and `replicas` outlive the join.
            let replica = replicas.replica_ptr(memory_domain);
            unsafe {
                core::ptr::copy_nonoverlapping(
                    source.get(range.start),
                    replica.add(range.start),
                    range.len(),
                );
            }
        });
    });
}

/// Everything a backend reads or writes for one convergence pass; the harness owns the lifetimes.
struct Ctx<'a> {
    graph: CsrView<'a>,
    pool: Option<&'a mut fu::ThreadPool>,
    rayon: Option<&'a RayonPool>,
    topology: Option<&'a fu::Topology>,
    replicas: Option<&'a ReplicatedCsr>,
    counters: &'a mut [Counter],
    labels_a: &'a mut [Label],
    labels_b: &'a mut [Label],
    rounds: usize,
}

// Compile-time schedule axis as marker types - stable Rust cannot take a custom enum as a const-generic.
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

// Compile-time placement axis: one shared CSR vs one read-only CSR replica per memory domain.
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

/// The CSR a thread on `compute_domain` scans: the shared host arrays, or its node-local replica.
#[inline]
fn csr_at<'a, P: Placement>(
    shared: CsrView<'a>,
    topology: Option<&'a fu::Topology>,
    replicas: Option<&'a ReplicatedCsr>,
    compute_domain: usize,
) -> CsrView<'a> {
    if !P::REPLICATED {
        return shared;
    }
    let memory_domain = topology
        .expect("topology")
        .local_memory_of(fu::ComputeDomain(compute_domain));
    let replicas = replicas.expect("replicas");
    // SAFETY: every replica holds a full, initialized copy of both arrays, read-only for the whole
    // convergence pass, and `replicas` outlives the join.
    unsafe {
        CsrView {
            row_offsets: core::slice::from_raw_parts(
                replicas.row_offsets.replica_ptr(memory_domain),
                shared.row_offsets.len(),
            ),
            column_indices: core::slice::from_raw_parts(
                replicas.column_indices.replica_ptr(memory_domain),
                shared.column_indices.len(),
            ),
        }
    }
}

/// One convergence pass on the ForkUnion pool; every round is one fork-join dispatch.
fn run_forkunion<S: Schedule, P: Placement>(c: &mut Ctx) {
    let graph = c.graph;
    let vertices = graph.vertices() as usize;
    let topology = c.topology;
    let replicas = c.replicas;
    let pool = c.pool.as_deref_mut().expect("ForkUnion pool");
    for (v, label) in c.labels_a.iter_mut().enumerate() {
        *label = v as Label;
    }

    let mut rounds = 0usize;
    let (mut old_ref, mut new_ref) = (&mut *c.labels_a, &mut *c.labels_b);
    loop {
        for counter in c.counters.iter_mut() {
            counter.0 = 0;
        }
        let counters = SyncMutPtr::new(c.counters.as_mut_ptr());
        let old_labels: &[Label] = old_ref;
        let new_labels = SyncMutPtr::new(new_ref.as_mut_ptr());
        let body = move |prong: fu::Prong| {
            let v = prong.task_index as u32;
            let local = csr_at::<P>(graph, topology, replicas, prong.compute_domain_index);
            let next = min_label_of(&local, old_labels, v);
            // SAFETY: each vertex writes only its own slot; each thread owns a unique counter.
            unsafe {
                *new_labels.get(v as usize) = next;
                (*counters.get(prong.thread_index)).0 += (next != old_labels[v as usize]) as u64;
            }
        };
        if S::STATIC_SCHEDULE {
            fu::for_n(pool, vertices, body);
        } else {
            fu::for_n_dynamic(pool, vertices, body);
        }
        rounds += 1;
        let changes: u64 = c.counters.iter().map(|counter| counter.0).sum();
        std::mem::swap(&mut old_ref, &mut new_ref);
        if changes == 0 {
            break;
        }
    }
    c.rounds = rounds;
}

fn run_forkunion_static_shared(c: &mut Ctx) {
    run_forkunion::<Static, Shared>(c);
}
fn run_forkunion_dynamic_shared(c: &mut Ctx) {
    run_forkunion::<Dynamic, Shared>(c);
}
fn run_forkunion_static_replicated(c: &mut Ctx) {
    run_forkunion::<Static, Replicated>(c);
}
fn run_forkunion_dynamic_replicated(c: &mut Ctx) {
    run_forkunion::<Dynamic, Replicated>(c);
}

/// One convergence pass on Rayon; a per-worker contiguous stripe (static) or unit grain (dynamic).
fn run_rayon(c: &mut Ctx, dynamic: bool) {
    let graph = c.graph;
    let vertices = graph.vertices() as usize;
    let pool = c.rayon.expect("Rayon pool");
    for (v, label) in c.labels_a.iter_mut().enumerate() {
        *label = v as Label;
    }

    let mut rounds = 0usize;
    let (mut old_ref, mut new_ref) = (&mut *c.labels_a, &mut *c.labels_b);
    loop {
        let old_labels: &[Label] = old_ref;
        let changes: u64 = pool.install(|| {
            if dynamic {
                new_ref
                    .par_iter_mut()
                    .with_max_len(1)
                    .enumerate()
                    .map(|(v, slot)| {
                        let next = min_label_of(&graph, old_labels, v as u32);
                        *slot = next;
                        (next != old_labels[v]) as u64
                    })
                    .sum()
            } else {
                new_ref
                    .par_iter_mut()
                    .enumerate()
                    .map(|(v, slot)| {
                        let next = min_label_of(&graph, old_labels, v as u32);
                        *slot = next;
                        (next != old_labels[v]) as u64
                    })
                    .sum()
            }
        });
        rounds += 1;
        std::mem::swap(&mut old_ref, &mut new_ref);
        if changes == 0 {
            break;
        }
    }
    let _ = vertices;
    c.rounds = rounds;
}

fn run_rayon_static(c: &mut Ctx) {
    run_rayon(c, false);
}
fn run_rayon_dynamic(c: &mut Ctx) {
    run_rayon(c, true);
}

/// The dispatch table - a name, its convergence pass, and the engine it runs on.
struct Backend {
    name: &'static str,
    run: fn(&mut Ctx),
    engine: Engine,
}

const BACKENDS: &[Backend] = &[
    Backend {
        name: "forkunion_static_shared",
        run: run_forkunion_static_shared,
        engine: Engine::ForkUnion,
    },
    Backend {
        name: "forkunion_dynamic_shared",
        run: run_forkunion_dynamic_shared,
        engine: Engine::ForkUnion,
    },
    Backend {
        name: "forkunion_static_replicated",
        run: run_forkunion_static_replicated,
        engine: Engine::ForkUnionReplicated,
    },
    Backend {
        name: "forkunion_dynamic_replicated",
        run: run_forkunion_dynamic_replicated,
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
];

/// Parses a fractional environment variable, or `fallback` when unset or unparseable.
fn env_f64(name: &str, fallback: f64) -> f64 {
    env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
}

/// Parses an unsigned environment variable, or `fallback` when unset or unparseable.
fn env_usize(name: &str, fallback: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
}

fn main() -> Result<(), Box<dyn Error>> {
    let scale = env_usize("PROPAGATION_SCALE", 14);
    let communities = env_usize("PROPAGATION_COMMUNITIES", 64);
    let edge_factor = env_usize("PROPAGATION_EDGE_FACTOR", 16);
    let backend =
        env::var("PROPAGATION_BACKEND").unwrap_or_else(|_| "forkunion_static_shared".into());
    let mut threads = env_usize("PROPAGATION_THREADS", 0);
    let budget_seconds = env_f64("PROPAGATION_SECONDS", 10.0); // ? The primary knob: a fixed window
    let iterations = env_usize("PROPAGATION_ITERATIONS", 0); // ? Overrides with an exact count when set
    let check = env::var("PROPAGATION_CHECK").is_ok();
    if threads == 0 {
        threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
    }
    assert!(
        (communities << scale) <= (1usize << 32),
        "PROPAGATION_COMMUNITIES << PROPAGATION_SCALE must fit 32-bit vertex indices"
    );

    let mut host = generate_necklace(scale, communities, edge_factor);
    let vertices = host.row_offsets.len() - 1;

    // One pinned pool spawns for EVERY backend - first to give the graph and label pages their
    // deterministic first touch, then to serve the ForkUnion backends; Rayon drops it below.
    let probed = fu::Topology::new().expect("Failed to detect hardware topology");
    let mut touch_pool = fu::ThreadPool::try_spawn(&probed, threads)?;
    let mut labels_a = vec![0 as Label; vertices];
    let mut labels_b = vec![0 as Label; vertices];
    retouch_deterministically(&mut touch_pool, &mut host.row_offsets);
    retouch_deterministically(&mut touch_pool, &mut host.column_indices);
    retouch_deterministically(&mut touch_pool, &mut labels_a);
    retouch_deterministically(&mut touch_pool, &mut labels_b);

    let graph = host.view();
    println!(
        "vertices {}, directed edges {}, communities {}",
        vertices,
        graph.edges(),
        communities
    );

    let selected = match BACKENDS.iter().find(|b| b.name == backend) {
        Some(b) => b,
        None => {
            eprintln!("Unsupported backend: '{backend}'");
            eprint!("Available backends:");
            for b in BACKENDS {
                eprint!(" {}", b.name);
            }
            eprintln!();
            return Err("unsupported backend".into());
        }
    };

    let mut fu_pool = None;
    let mut rayon_pool = None;
    let mut topology = None;
    let mut replicas = None;
    match selected.engine {
        Engine::ForkUnion | Engine::ForkUnionReplicated => {
            let mut pool = touch_pool; // ? The touch pool doubles as the benchmark pool
            if selected.engine == Engine::ForkUnionReplicated {
                let csr = ReplicatedCsr {
                    row_offsets: fu::ReplicatedArray::try_new(&probed, graph.row_offsets.len())
                        .expect("Failed to allocate per-domain row-offset replicas"),
                    column_indices: fu::ReplicatedArray::try_new(
                        &probed,
                        graph.column_indices.len(),
                    )
                    .expect("Failed to allocate per-domain column-index replicas"),
                };
                replicate_into(&probed, &mut pool, &csr.row_offsets, graph.row_offsets);
                replicate_into(
                    &probed,
                    &mut pool,
                    &csr.column_indices,
                    graph.column_indices,
                );
                replicas = Some(csr);
            }
            fu_pool = Some(pool);
            topology = Some(probed);
        }
        Engine::Rayon => {
            drop(touch_pool); // ? Frees the cores before Rayon spawns its own workers
            rayon_pool = Some(ThreadPoolBuilder::new().num_threads(threads).build()?);
        }
    }

    let mut counters: Vec<Counter> = (0..threads).map(|_| fu::CacheAligned(0)).collect();

    let mut context = Ctx {
        graph,
        pool: fu_pool.as_mut(),
        rayon: rayon_pool.as_ref(),
        topology: topology.as_ref(),
        replicas: replicas.as_ref(),
        counters: &mut counters,
        labels_a: &mut labels_a,
        labels_b: &mut labels_b,
        rounds: 0,
    };

    // One untimed warmup pass: page-faults and cache warming would otherwise bias the first timed
    // pass, and by a different amount for each backend.
    (selected.run)(&mut context);

    // A fixed time budget beats a fixed pass count: every backend runs the same wall-clock window -
    // long enough to amortize scheduling noise - and reports the rate it sustained, with no
    // per-backend pass-count guessing. `PROPAGATION_ITERATIONS` forces an exact count instead.
    let started = Instant::now();
    let mut passes = 0usize;
    if iterations > 0 {
        for _ in 0..iterations {
            (selected.run)(&mut context);
            passes += 1;
        }
    } else {
        while {
            (selected.run)(&mut context);
            passes += 1;
            started.elapsed().as_secs_f64() < budget_seconds
        } {}
    }
    let seconds = started.elapsed().as_secs_f64() / passes as f64;
    let rounds = context.rounds;

    // The fixed point sits in both buffers - the terminal round changed nothing - so read either.
    let mut components = 0u64;
    let mut checksum = 0u64;
    for (v, &label) in labels_a.iter().enumerate() {
        components += (label == v as Label) as u64;
        checksum = checksum.wrapping_add(label as u64);
    }
    // MTEPS - millions of directed edges scanned per second; `rounds * edges` is the exact scan
    // count, identical in every cell by the double-buffered determinism.
    let mteps = rounds as f64 * graph.edges() as f64 / seconds / 1e6;
    println!(
        "{backend}: {components} components, {rounds} rounds, checksum {checksum}, {seconds:.2} s/pass, {mteps:.1} MTEPS"
    );

    if check {
        let mut serial_a = vec![0 as Label; vertices];
        let mut serial_b = vec![0 as Label; vertices];
        let serial_rounds = converge_serially(&graph, &mut serial_a, &mut serial_b);
        if serial_rounds != rounds || serial_a != labels_a {
            eprintln!("MISMATCH: serial converged in {serial_rounds} rounds");
            return Err("serial mismatch".into());
        }
        println!("check: matches the serial labels and rounds");
    }
    Ok(())
}
