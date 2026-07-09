//! Compute-domain-aware vector search using ForkUnion and PinnedVec with SimSIMD.
//!
//! This example demonstrates how to perform efficient similarity search across
//! multiple compute domains using the PinnedVec container, ForkUnion's distributed
//! thread pool capabilities, and SimSIMD for optimized distance calculations.
//!
//! To run this example:
//!
//! ```sh
//! cargo run --example search --release
//! ```
//!
//! Environment variables:
//! - `SEARCH_SCOPE` - percentage of available memory to use for vectors (default: 25)
//! - `SEARCH_THREADS` - number of threads to use (default: available parallelism)

use rand::{rng, Rng};
use std::env;
use std::time::Instant;

use forkunion as fu;
use simsimd::{bf16, Distance, SpatialSimilarity};

/// Embedding dimensions - fixed at compile time for better performance
const EMBEDDING_DIMENSIONS: usize = 768;

/// Default percentage of memory to use for vector storage
const DEFAULT_MEMORY_SCOPE_PERCENT: usize = 25;

/// A fixed-size vector type for compile-time optimizations
type Embedding = [bf16; EMBEDDING_DIMENSIONS];

/// Result of a search operation - stored on stack to avoid heap allocations
#[derive(Debug, Clone, Copy)]
struct SearchResult {
    best_similarity: Distance,
    best_index: usize,
    compute_domain_index: usize,
}

impl SearchResult {
    fn new(compute_domain_index: usize) -> Self {
        Self {
            best_similarity: Distance::NEG_INFINITY,
            best_index: 0,
            compute_domain_index,
        }
    }

    fn update_if_better(&mut self, similarity: Distance, index: usize) {
        if similarity > self.best_similarity {
            self.best_similarity = similarity;
            self.best_index = index;
        }
    }
}

/// Compute-domain-distributed vector storage using RoundRobinVec with fixed-size vectors
type DistributedEmbeddings = fu::RoundRobinVec<Embedding>;

/// Creates a new compute-domain-distributed vector storage based on memory scope percentage
fn create_distributed_embeddings(
    pool: &mut fu::ThreadPool,
    memory_scope_percent: usize,
) -> Option<DistributedEmbeddings> {
    let compute_domains_count = fu::count_compute_domains();
    println!("Initializing storage across {compute_domains_count} compute_domains");

    // Calculate total capacity based on total system memory and scope percentage
    let total_memory = fu::volume_ram();
    let target_memory = (total_memory * memory_scope_percent) / 100;
    let vector_size = core::mem::size_of::<Embedding>();
    let total_vectors = if vector_size > 0 {
        target_memory / vector_size
    } else {
        0
    };

    if total_vectors == 0 {
        println!("Warning: Not enough memory for vectors");
        return None;
    }

    println!(
        "Target memory: {:.2} MB ({:.1}% of {:.2} MB total)",
        target_memory as f64 / 1024.0 / 1024.0,
        memory_scope_percent,
        total_memory as f64 / 1024.0 / 1024.0
    );
    println!("Total vectors to create: {total_vectors}");

    // Create RoundRobinVec and resize to target capacity
    let mut distributed_vec = DistributedEmbeddings::new()?;

    // Initialize with zero vectors first, then fill with random data
    let zero_embedding = [bf16::from_f32(0.0); EMBEDDING_DIMENSIONS];
    distributed_vec
        .resize(total_vectors, zero_embedding, pool)
        .ok()?;

    // Fill with random data using parallel fill_with
    distributed_vec.fill_with(
        || {
            let mut rng = rng();
            let mut embedding = [bf16::from_f32(0.0); EMBEDDING_DIMENSIONS];
            for (_dim, item) in embedding.iter_mut().enumerate().take(EMBEDDING_DIMENSIONS) {
                *item = bf16::from_f32(rng.random_range(-1.0..1.0));
            }
            embedding
        },
        pool,
    );

    println!(
        "Successfully created {} vectors across {} compute_domains",
        distributed_vec.len(),
        compute_domains_count
    );
    Some(distributed_vec)
}

/// Performs compute-domain-local search, each thread scanning its own compute domain's vectors
fn numa_aware_search(
    storage: &DistributedEmbeddings,
    query: &Embedding,
    pool: &mut fu::ThreadPool,
) -> SearchResult {
    let compute_domains_count = storage.compute_domains_count();

    // Use SpinMutex for the global best result
    let best_result = fu::SpinMutex::new(SearchResult::new(0));

    // A scope lets each worker borrow `storage`/`query`/`best_result` and query the pool directly,
    // with no `SafePtr` smuggling - every dispatch joins before the scope returns.
    pool.scope(|scope| {
        scope.broadcast(|thread_index, compute_domain_index| {
            // Each thread works on its own compute domain
            if compute_domain_index < compute_domains_count {
                let mut local_result = SearchResult::new(compute_domain_index);

                // Get the vectors held in this compute domain's local memory
                if let Some(node_vectors) = storage.get_compute_domain(compute_domain_index) {
                    let vectors_count = node_vectors.len();
                    let threads_in_compute_domain = scope.count_threads_in(compute_domain_index);
                    let thread_local_index =
                        scope.locate_thread_in(thread_index, compute_domain_index);

                    // Split vectors among threads in this compute domain
                    let split = fu::IndexedSplit::new(vectors_count, threads_in_compute_domain);
                    let range = split.get(thread_local_index);

                    // Search vectors assigned to this thread
                    for local_vector_idx in range {
                        if let Some(vector) = node_vectors.get(local_vector_idx) {
                            let similarity = bf16::dot(query, vector).unwrap();
                            let global_index = storage
                                .local_to_global_index(compute_domain_index, local_vector_idx);
                            local_result.update_if_better(similarity, global_index);
                        }
                    }
                }

                // Merge into the global best result
                let mut best = best_result.lock();
                if local_result.best_similarity > best.best_similarity {
                    *best = local_result;
                }
            }
        });
    });

    let result = best_result.lock();
    *result
}

/// Performs unbalanced search where threads sweep every compute domain (for comparison)
fn worst_case_search(
    storage: &DistributedEmbeddings,
    query: &Embedding,
    pool: &mut fu::ThreadPool,
) -> SearchResult {
    let compute_domains_count = storage.compute_domains_count();

    // Use SpinMutex for the global best result
    let best_result = fu::SpinMutex::new(SearchResult::new(0));

    // The same scope pattern, but each thread deliberately sweeps every compute domain to model the
    // cross-domain (non-local) access pattern for comparison.
    pool.scope(|scope| {
        scope.broadcast(|thread_index, compute_domain_index| {
            let mut local_result = SearchResult::new(compute_domain_index);
            let total_threads = scope.threads();

            for compute_domain_index in 0..compute_domains_count {
                if let Some(node_vectors) = storage.get_compute_domain(compute_domain_index) {
                    let vectors_in_node = node_vectors.len();

                    let split = fu::IndexedSplit::new(vectors_in_node, total_threads);
                    let range = split.get(thread_index);

                    // Search vectors assigned to this thread, regardless of locality
                    for local_vector_idx in range {
                        if let Some(vector) = node_vectors.get(local_vector_idx) {
                            let similarity = bf16::dot(query, vector).unwrap();
                            let global_index = storage
                                .local_to_global_index(compute_domain_index, local_vector_idx);
                            local_result.update_if_better(similarity, global_index);
                        }
                    }
                }
            }

            // Merge into the global best result
            let mut best = best_result.lock();
            if local_result.best_similarity > best.best_similarity {
                *best = local_result;
            }
        });
    });

    let result = best_result.lock();
    *result
}

/// Benchmark search performance
fn benchmark_search<F>(
    name: &str,
    storage: &DistributedEmbeddings,
    queries: &[Embedding],
    pool: &mut fu::ThreadPool,
    search_fn: F,
) where
    F: Fn(&DistributedEmbeddings, &Embedding, &mut fu::ThreadPool) -> SearchResult,
{
    println!("\n=== {name} ===");

    let start = Instant::now();
    let mut total_similarity: Distance = 0.0;

    for (i, query) in queries.iter().enumerate() {
        let result = search_fn(storage, query, pool);
        total_similarity += result.best_similarity;

        if i < 5 {
            // Print first few results
            println!(
                "Query {}: best similarity {:.6} at index {} (compute_domain {})",
                i, result.best_similarity, result.best_index, result.compute_domain_index
            );
        }
    }

    let duration = start.elapsed();
    let avg_similarity = total_similarity / queries.len() as Distance;

    println!(
        "Completed {} queries in {:.2}ms",
        queries.len(),
        duration.as_secs_f64() * 1000.0
    );
    println!(
        "Average time per query: {:.2}μs",
        duration.as_secs_f64() * 1_000_000.0 / queries.len() as f64
    );
    println!("Average similarity: {avg_similarity:.6}");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Compute-Domain-Aware Embedding Search");

    // Parse environment variables
    let memory_scope_percent = env::var("SEARCH_SCOPE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MEMORY_SCOPE_PERCENT);

    let threads = env::var("SEARCH_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        });

    // Print system information
    println!("System Information:");
    println!("  Logical cores: {}", fu::count_logical_cores());
    println!("  Memory domains: {}", fu::count_memory_domains());
    println!("  Thread compute_domains: {}", fu::count_compute_domains());
    println!("  NUMA enabled: {}", fu::numa_enabled());
    println!("Configuration:");
    println!("  Embedding dimensions: {EMBEDDING_DIMENSIONS}");
    println!("  Memory scope: {memory_scope_percent}%");
    println!("  Thread pool size: {threads}");

    // Create thread pool
    let mut pool = fu::ThreadPool::try_spawn(threads)?;

    // Initialize compute-domain-distributed vector storage
    println!();
    println!("📚 Initializing vector storage...");
    let storage = create_distributed_embeddings(&mut pool, memory_scope_percent)
        .ok_or("Failed to initialize vector storage")?;
    println!(
        "Thread pool initialized with {} threads across {} compute_domains",
        pool.threads(),
        pool.compute_domains()
    );

    // Generate random queries with fixed-size vectors
    let query_count = 100; // Fixed number of queries for consistent benchmarking
    println!();
    println!("🎯 Generating {query_count} random queries...");
    let mut rng = rng();
    let mut queries = Vec::with_capacity(query_count);

    for _ in 0..query_count {
        let mut query = [bf16::from_f32(0.0); EMBEDDING_DIMENSIONS];
        for (_dim, item) in query.iter_mut().enumerate().take(EMBEDDING_DIMENSIONS) {
            *item = bf16::from_f32(rng.random_range(-1.0..1.0));
        }
        queries.push(query);
    }

    // Benchmark different search strategies
    benchmark_search(
        "Compute-Domain-Aware Search",
        &storage,
        &queries,
        &mut pool,
        numa_aware_search,
    );

    benchmark_search(
        "Worst Case Search",
        &storage,
        &queries,
        &mut pool,
        worst_case_search,
    );

    println!();
    println!("✅ Search benchmarking completed!");
    Ok(())
}
