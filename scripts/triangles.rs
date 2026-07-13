//! Demo app: triangle counting on a power-law graph, with ForkUnion and Rayon.
//!
//! The N-body simulation gives every task an identical cost, so it can only measure dispatch latency.
//! Triangle counting gives them wildly different costs: the work at a vertex grows with its degree and
//! with the degrees of its neighbours, and an R-MAT graph draws those degrees from a power law. A
//! handful of hub vertices carry most of the arithmetic.
//!
//! Crucially, the hubs are adjacent. R-MAT keeps recursing into the same quadrant, so the high-degree
//! vertices cluster at low indices, and a static split hands one thread nearly all of them.
//!
//! The obvious decomposition is one task per vertex, which is what `vertex_centric_static` and `vertex_centric_dynamic`
//! do. `vertex_centric_static` is hopeless: the slice holding the hubs decides the makespan. A GPU flattens the
//! adjacency into a tape of equally-sized work items - one per `u < v` edge - and hands each thread a
//! contiguous run. Balancing the item count still sinks the slice that owns the hubs, so `edge_centric`
//! weighs each item by `degree(u) + degree(v)`, prefix-sums the weights, and cuts the cost axis into
//! equal parts - a static dispatch that beats a stealing one.
//!
//! The CSR is read-only once built, so `edge_centric_replicated` gives every memory domain its own
//! replica - a `ReplicatedArray` per CSR array - and no thread ever reaches across the interconnect for
//! a neighbour list.
//!
//! To control the script, several environment variables are used:
//!
//! - `TRIANGLES_SCALE` - the graph has `2^scale` vertices - default 18.
//! - `TRIANGLES_EDGE_FACTOR` - edges generated per vertex, before deduplication - default 16.
//! - `TRIANGLES_BACKEND` - backend to use - default `forkunion_vertex_centric_static`.
//! - `TRIANGLES_THREADS` - number of threads to use - default all hardware threads.
//! - `TRIANGLES_ITERATIONS` - repeat the count this many times, reporting the per-pass time - default 1.
//! - `TRIANGLES_CHECK` - also count serially, and fail unless the totals agree.
//!
//! The backends include: `forkunion_vertex_centric_static`, `forkunion_vertex_centric_dynamic`, `forkunion_edge_centric`,
//! `forkunion_edge_centric_replicated`, `rayon_static`, and `rayon_dynamic`. To compile and run:
//!
//! ```sh
//! cargo build --example triangles --release
//! time TRIANGLES_SCALE=20 TRIANGLES_BACKEND=forkunion_edge_centric target/release/examples/triangles
//! ```
use std::env;
use std::error::Error;
use std::time::Instant;

use forkunion as fu;
use fu::SyncMutPtr;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rayon::{ThreadPool as RayonPool, ThreadPoolBuilder};

/// A within-row neighbour count, bounded by the vertex count.
type Degree = u32;
/// A tape-item cost `degree(u) + degree(v)`, and the prefix sum over them.
type Work = u64;
/// Per-thread tally, cache-aligned so two threads never share a line.
type Counter = fu::CacheAligned<u64>;

/// A read-only CSR-plus-tape as five slices - the interface every kernel takes.
///
/// `row_offsets` and `column_indices` are the usual CSR pair, each adjacency sorted ascending. The rest
/// turns a tape index back into an edge: `above_offsets[u]` is where the neighbours greater than `u`
/// begin within `u`'s row, `tape_offsets[u]` is the prefix sum of tape items before `u`, and
/// `work_offsets` is the prefix sum of each item's cost. The view is agnostic to where the bytes live -
/// the host build, or a replica.
#[derive(Copy, Clone)]
struct CsrView<'a> {
    row_offsets: &'a [u64],
    column_indices: &'a [u32],
    tape_offsets: &'a [u64],
    above_offsets: &'a [Degree],
    work_offsets: &'a [Work],
}

impl CsrView<'_> {
    fn vertices(&self) -> u32 {
        (self.row_offsets.len() - 1) as u32
    }

    fn edges(&self) -> usize {
        self.column_indices.len()
    }

    fn degree(&self, v: u32) -> Degree {
        (self.row_offsets[v as usize + 1] - self.row_offsets[v as usize]) as Degree
    }

    /// Number of `u < v` pairs; the length of the work tape.
    fn tape_length(&self) -> u64 {
        *self.tape_offsets.last().unwrap()
    }

    /// Total cost of the tape, in units of "one adjacency element compared".
    fn total_work(&self) -> Work {
        *self.work_offsets.last().unwrap()
    }

    /// The first tape item whose cumulative cost reaches `work` - the merge-path search.
    fn item_at_work(&self, work: Work) -> u64 {
        self.work_offsets.partition_point(|&w| w < work) as u64
    }

    /// The vertex owning tape item `item`, by binary search into the prefix sum.
    fn owner_of(&self, item: u64) -> u32 {
        (self.tape_offsets.partition_point(|&t| t <= item) - 1) as u32
    }
}

/// The five CSR-plus-tape arrays built once on the host.
struct CsrHost {
    row_offsets: Vec<u64>,
    column_indices: Vec<u32>,
    tape_offsets: Vec<u64>,
    above_offsets: Vec<Degree>,
    work_offsets: Vec<Work>,
}

impl CsrHost {
    fn view(&self) -> CsrView<'_> {
        CsrView {
            row_offsets: &self.row_offsets,
            column_indices: &self.column_indices,
            tape_offsets: &self.tape_offsets,
            above_offsets: &self.above_offsets,
            work_offsets: &self.work_offsets,
        }
    }
}

/// Generates a Kronecker/R-MAT graph, as specified by Graph500, and builds its tape.
///
/// Recursing into the `a` quadrant with probability 57% is what makes the degrees power-law, and what
/// keeps the hubs near index zero, where a static split will trip over them.
fn generate_rmat(scale: usize, edge_factor: usize) -> CsrHost {
    let vertices = 1usize << scale;
    let raw_edges = vertices * edge_factor;

    // Build the COO edge list and dedupe it.
    let mut rng = StdRng::seed_from_u64(0x1234_5678_9ABC_DEF0);
    let mut edges: Vec<(u32, u32)> = Vec::with_capacity(raw_edges * 2);
    for _ in 0..raw_edges {
        let mut row = 0u32;
        let mut column = 0u32;
        for bit in (0..scale).rev() {
            let r = rng.random_range(0..100u32); // a=57 b=19 c=19 d=5, integer and portable
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
        if row == column {
            continue; // Drop self-loops
        }
        edges.push((row, column)); // Symmetrize
        edges.push((column, row));
    }
    edges.sort_unstable();
    edges.dedup();

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

    // Tape: one item per `u < v` pair, in vertex order.
    let mut above_offsets = vec![0 as Degree; vertices];
    let mut tape_offsets = vec![0u64; vertices + 1];
    for u in 0..vertices {
        let start = row_offsets[u] as usize;
        let end = row_offsets[u + 1] as usize;
        let row = &column_indices[start..end];
        let above = row.partition_point(|&w| w <= u as u32); // First neighbour greater than `u`
        above_offsets[u] = above as Degree;
        tape_offsets[u + 1] = tape_offsets[u] + (row.len() - above) as u64;
    }

    // Weigh each tape item by the two adjacencies it intersects, so balancing this weight spreads the
    // hubs that balancing the bare item count would bunch into one slice.
    let tape_length = tape_offsets[vertices];
    let mut work_offsets = vec![0 as Work; tape_length as usize + 1];
    for u in 0..vertices {
        let u_degree = row_offsets[u + 1] - row_offsets[u];
        let mut position = row_offsets[u] as usize + above_offsets[u] as usize;
        for item in tape_offsets[u]..tape_offsets[u + 1] {
            let v = column_indices[position] as usize;
            let v_degree = row_offsets[v + 1] - row_offsets[v];
            work_offsets[item as usize + 1] = work_offsets[item as usize] + u_degree + v_degree;
            position += 1;
        }
    }

    CsrHost {
        row_offsets,
        column_indices,
        tape_offsets,
        above_offsets,
        work_offsets,
    }
}

/// Counts triangles `u < v < w` for the single edge at tape item `item`, whose owner is `owner`.
///
/// Both adjacencies are sorted, so the candidates for `w` are the suffix of `N(u)` past `v` and the
/// suffix of `N(v)` past `v`. Intersecting them counts each triangle exactly once, and needs no atomics.
#[inline]
fn count_triangles_on_item(graph: &CsrView, item: u64, owner: u32) -> u64 {
    let columns = graph.column_indices;
    let owner = owner as usize;
    let owner_end = graph.row_offsets[owner + 1] as usize;
    let position = graph.row_offsets[owner] as usize
        + graph.above_offsets[owner] as usize
        + (item - graph.tape_offsets[owner]) as usize;

    let other = columns[position] as usize;
    let mut a = position + 1;
    let a_end = owner_end;
    let mut b = graph.row_offsets[other] as usize + graph.above_offsets[other] as usize;
    let b_end = graph.row_offsets[other + 1] as usize;

    let mut triangles = 0u64;
    while a < a_end && b < b_end {
        let column_a = columns[a];
        let column_b = columns[b];
        if column_a < column_b {
            a += 1;
        } else if column_b < column_a {
            b += 1;
        } else {
            triangles += 1;
            a += 1;
            b += 1;
        }
    }
    triangles
}

/// Counts every triangle whose smallest vertex is `u`. Cost grows with `degree(u)` squared.
#[inline]
fn count_triangles_at_vertex(graph: &CsrView, u: u32) -> u64 {
    let mut triangles = 0u64;
    for item in graph.tape_offsets[u as usize]..graph.tape_offsets[u as usize + 1] {
        triangles += count_triangles_on_item(graph, item, u);
    }
    triangles
}

/// Walks a contiguous run of the tape, resolving the owning vertex only when it changes.
///
/// One binary search per slice, then a forward walk - the CPU spelling of a merge-path.
#[inline]
fn count_triangles_on_slice(graph: &CsrView, first: u64, count: u64) -> u64 {
    if count == 0 {
        return 0;
    }
    let mut triangles = 0u64;
    let mut owner = graph.owner_of(first);
    for item in first..first + count {
        while item >= graph.tape_offsets[owner as usize + 1] {
            owner += 1; // Amortized O(1) per item
        }
        triangles += count_triangles_on_item(graph, item, owner);
    }
    triangles
}

/// One read-only replica of the CSR per memory domain, so no adjacency is ever remote.
///
/// Each CSR array is its own `ReplicatedArray` - a symmetric mapping the allocator stripes across the
/// nodes - so the one-time fill can be a plain serial copy and still land each slice on its own node.
/// `on_memory_domain` assembles the five node-local slices back into a `CsrView`.
struct ReplicatedCsr {
    row_offsets: fu::ReplicatedArray<u64>,
    column_indices: fu::ReplicatedArray<u32>,
    tape_offsets: fu::ReplicatedArray<u64>,
    above_offsets: fu::ReplicatedArray<Degree>,
    work_offsets: fu::ReplicatedArray<Work>,
}

impl ReplicatedCsr {
    fn try_build(topology: &fu::Topology, host: &CsrHost) -> Option<Self> {
        Some(Self {
            row_offsets: replicate(topology, &host.row_offsets)?,
            column_indices: replicate(topology, &host.column_indices)?,
            tape_offsets: replicate(topology, &host.tape_offsets)?,
            above_offsets: replicate(topology, &host.above_offsets)?,
            work_offsets: replicate(topology, &host.work_offsets)?,
        })
    }

    fn on_memory_domain(&self, memory_domain: fu::MemoryDomain) -> CsrView<'_> {
        CsrView {
            row_offsets: self.row_offsets.on_memory_domain(memory_domain),
            column_indices: self.column_indices.on_memory_domain(memory_domain),
            tape_offsets: self.tape_offsets.on_memory_domain(memory_domain),
            above_offsets: self.above_offsets.on_memory_domain(memory_domain),
            work_offsets: self.work_offsets.on_memory_domain(memory_domain),
        }
    }
}

/// Copies one host array into every per-domain replica, each slice landing on its own node.
fn replicate<T: Copy>(topology: &fu::Topology, host: &[T]) -> Option<fu::ReplicatedArray<T>> {
    let mut replica = fu::ReplicatedArray::<T>::try_new(topology, host.len())?;
    for domain in 0..replica.memory_domains_count() {
        replica
            .on_memory_domain_mut(fu::MemoryDomain(domain))
            .copy_from_slice(host);
    }
    Some(replica)
}

/// Everything a backend reads or writes for one counting pass; the harness owns the lifetimes and hands
/// each backend only the execution engine it asked for.
struct Ctx<'a> {
    graph: CsrView<'a>,
    replicas: Option<&'a ReplicatedCsr>,
    topology: Option<&'a fu::Topology>,
    pool: Option<&'a mut fu::ThreadPool>,
    rayon: Option<&'a RayonPool>,
    counters: &'a mut [Counter],
    threads: usize,
}

/// One task per vertex, split statically - the naive baseline the hubs sink.
fn run_vertex_centric_static(c: &mut Ctx) {
    let graph = c.graph;
    let counters = SyncMutPtr::new(c.counters.as_mut_ptr());
    let pool = c.pool.as_deref_mut().expect("ForkUnion pool");
    fu::for_n(pool, graph.vertices() as usize, move |prong| {
        let triangles = count_triangles_at_vertex(&graph, prong.task_index as u32);
        // SAFETY: `prong.thread_index` is unique per thread and tasks on one thread run in order.
        unsafe {
            (*counters.get(prong.thread_index)).0 += triangles;
        }
    });
}

/// One task per vertex, work-stolen - stealing recovers the balance a static split loses.
fn run_vertex_centric_dynamic(c: &mut Ctx) {
    let graph = c.graph;
    let counters = SyncMutPtr::new(c.counters.as_mut_ptr());
    let pool = c.pool.as_deref_mut().expect("ForkUnion pool");
    fu::for_n_dynamic(pool, graph.vertices() as usize, move |prong| {
        let triangles = count_triangles_at_vertex(&graph, prong.task_index as u32);
        // SAFETY: as in the static variant above.
        unsafe {
            (*counters.get(prong.thread_index)).0 += triangles;
        }
    });
}

/// Cut the cost axis into equal slices, merge-path each cut onto the item axis - atomics-free.
fn run_edge_centric(c: &mut Ctx) {
    let graph = c.graph;
    let threads = c.threads;
    let total = graph.total_work() as usize;
    let counters = SyncMutPtr::new(c.counters.as_mut_ptr());
    let pool = c.pool.as_deref_mut().expect("ForkUnion pool");
    pool.broadcast(|thread_index, _compute_domain_index| {
        let cost = fu::IndexedSplit::new(total, threads).get(thread_index);
        let first = graph.item_at_work(cost.start as u64);
        let last = graph.item_at_work(cost.end as u64);
        let triangles = count_triangles_on_slice(&graph, first, last - first);
        // SAFETY: `thread_index` is the unique global index, so each thread owns one counter.
        unsafe {
            (*counters.get(thread_index)).0 += triangles;
        }
    });
}

/// `edge_centric` with the cost axis cut per compute domain, each thread reading its node's replica.
fn run_edge_centric_replicated(c: &mut Ctx) {
    let replicas = c.replicas.expect("replicas");
    let topology = c.topology.expect("topology");
    let counters = SyncMutPtr::new(c.counters.as_mut_ptr());
    let pool = c.pool.as_deref_mut().expect("ForkUnion pool");
    pool.scope(|scope| {
        scope.broadcast(|thread_index, compute_domain_index| {
            let memory_domain = topology.local_memory_of(fu::ComputeDomain(compute_domain_index));
            let replica = replicas.on_memory_domain(memory_domain);

            let domains = scope.compute_domains_count();
            let threads_here = scope.threads_count_in(compute_domain_index);
            let local_index = scope.locate_thread_in(thread_index, compute_domain_index);
            let total = replica.total_work() as usize;

            // Two nested balanced splits of the cost axis: first between compute domains, then between
            // the threads of this domain - reusing the same fair-chunk splitter as the count-axis loops.
            let domain_cost = fu::IndexedSplit::new(total, domains).get(compute_domain_index);
            let thread_cost =
                fu::IndexedSplit::new(domain_cost.len(), threads_here).get(local_index);
            let first = replica.item_at_work((domain_cost.start + thread_cost.start) as u64);
            let last = replica
                .item_at_work((domain_cost.start + thread_cost.start + thread_cost.len()) as u64);
            let triangles = count_triangles_on_slice(&replica, first, last - first);
            // SAFETY: `thread_index` is the unique global index, so each thread owns one counter.
            unsafe {
                (*counters.get(thread_index)).0 += triangles;
            }
        });
    });
}

/// One contiguous vertex stripe per worker, no stealing - the Rayon baseline the hubs sink.
fn run_rayon_static(c: &mut Ctx) {
    let graph = c.graph;
    let pool = c.rayon.expect("Rayon pool");
    let vertices = graph.vertices() as usize;
    let workers = pool.current_num_threads();
    let stride = vertices.div_ceil(workers);
    let sum: u64 = pool.install(|| {
        (0..workers)
            .into_par_iter()
            .map(|worker| {
                let start = worker * stride;
                let end = ((worker + 1) * stride).min(vertices);
                (start..end)
                    .map(|v| count_triangles_at_vertex(&graph, v as u32))
                    .sum::<u64>()
            })
            .sum()
    });
    c.counters[0].0 = sum;
}

/// One vertex per work item, work-stolen - Rayon's `into_par_iter` with a unit grain.
fn run_rayon_dynamic(c: &mut Ctx) {
    let graph = c.graph;
    let pool = c.rayon.expect("Rayon pool");
    let vertices = graph.vertices() as usize;
    let sum: u64 = pool.install(|| {
        (0..vertices)
            .into_par_iter()
            .with_max_len(1)
            .map(|v| count_triangles_at_vertex(&graph, v as u32))
            .sum()
    });
    c.counters[0].0 = sum;
}

/// Which execution engine a backend runs on, so `main` builds exactly the resource it needs.
#[derive(Copy, Clone, PartialEq)]
enum Engine {
    ForkUnion,
    ForkUnionReplicated,
    Rayon,
}

/// The dispatch table - a name, its counting pass, and the engine it runs on.
struct Backend {
    name: &'static str,
    run: fn(&mut Ctx),
    engine: Engine,
}

const BACKENDS: &[Backend] = &[
    Backend {
        name: "forkunion_vertex_centric_static",
        run: run_vertex_centric_static,
        engine: Engine::ForkUnion,
    },
    Backend {
        name: "forkunion_vertex_centric_dynamic",
        run: run_vertex_centric_dynamic,
        engine: Engine::ForkUnion,
    },
    Backend {
        name: "forkunion_edge_centric",
        run: run_edge_centric,
        engine: Engine::ForkUnion,
    },
    Backend {
        name: "forkunion_edge_centric_replicated",
        run: run_edge_centric_replicated,
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

fn env_usize(name: &str, fallback: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
}

fn env_string(name: &str, fallback: &str) -> String {
    env::var(name).unwrap_or_else(|_| fallback.into())
}

fn env_flag(name: &str) -> bool {
    env::var(name).is_ok()
}

fn hardware_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn main() -> Result<(), Box<dyn Error>> {
    let scale = env_usize("TRIANGLES_SCALE", 18);
    let edge_factor = env_usize("TRIANGLES_EDGE_FACTOR", 16);
    let backend = env_string("TRIANGLES_BACKEND", "forkunion_vertex_centric_static");
    let check = env_flag("TRIANGLES_CHECK");
    let mut threads = env_usize("TRIANGLES_THREADS", 0);
    let mut iterations = env_usize("TRIANGLES_ITERATIONS", 1);
    if threads == 0 {
        threads = hardware_threads();
    }
    if iterations == 0 {
        iterations = 1;
    }

    let host = generate_rmat(scale, edge_factor);
    let graph = host.view();
    let vertices = graph.vertices();
    let max_degree = (0..vertices).map(|v| graph.degree(v)).max().unwrap_or(0);
    println!(
        "vertices {}, directed edges {}, tape {}, max degree {} (mean {:.1})",
        vertices,
        graph.edges(),
        graph.tape_length(),
        max_degree,
        graph.edges() as f64 / vertices as f64
    );

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
    // the topology, so the Rayon backends never touch it.
    let mut topology = None;
    let mut fu_pool = None;
    let mut replicas = None;
    let mut rayon_pool = None;
    match selected.engine {
        Engine::ForkUnion | Engine::ForkUnionReplicated => {
            let probed = fu::Topology::new().expect("Failed to detect hardware topology");
            fu_pool = Some(
                fu::ThreadPool::try_spawn(&probed, threads)
                    .unwrap_or_else(|e| panic!("Failed to spawn the thread pool: {e}")),
            );
            if selected.engine == Engine::ForkUnionReplicated {
                replicas = Some(
                    ReplicatedCsr::try_build(&probed, &host)
                        .expect("Failed to replicate the graph across memory domains"),
                );
            }
            topology = Some(probed);
        }
        Engine::Rayon => {
            rayon_pool = Some(
                ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .expect("Failed to build the Rayon pool"),
            );
        }
    }

    let mut counters: Vec<Counter> = (0..threads).map(|_| fu::CacheAligned(0u64)).collect();
    let mut context = Ctx {
        graph,
        replicas: replicas.as_ref(),
        topology: topology.as_ref(),
        pool: fu_pool.as_mut(),
        rayon: rayon_pool.as_ref(),
        counters: &mut counters,
        threads,
    };

    // Run the pass `iterations` times, timing it; it leaves its per-thread tallies in `counters`, which
    // are zeroed before each run and summed after the last.
    let started = Instant::now();
    for _ in 0..iterations {
        for counter in context.counters.iter_mut() {
            counter.0 = 0;
        }
        (selected.run)(&mut context);
    }
    let seconds = started.elapsed().as_secs_f64() / iterations as f64;
    let triangles: u64 = context.counters.iter().map(|counter| counter.0).sum();
    println!("{backend}: {triangles} triangles in {seconds:.3} s");

    if check {
        let serial: u64 = (0..vertices)
            .map(|v| count_triangles_at_vertex(&graph, v))
            .sum();
        if serial != triangles {
            eprintln!("MISMATCH: serial counted {serial}");
            return Err(format!("MISMATCH: serial counted {serial}").into());
        }
        println!("check: matches the serial count");
    }
    Ok(())
}
