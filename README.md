# Fork Union 🍴

Fork Union is arguably the lowest-latency OpenMP-style NUMA-aware minimalistic scoped thread-pool designed for 'Fork-Join' parallelism in C++, C, Rust, and Zig, avoiding × [mutexes & system calls](#locks-and-mutexes), × [dynamic memory allocations](#memory-allocations), × [CAS-primitives](#atomics-and-cas), and × [false-sharing](#alignment--false-sharing) of CPU cache-lines on the hot path 🍴

## Motivation

Most "thread-pools" are not, in fact, thread-pools, but rather "task-queues" that are designed to synchronize a concurrent dynamically growing list of heap-allocated globally accessible shared objects.
In C++ terms, think of it as a `std::queue<std::function<void()>>` protected by a `std::mutex`, where each thread waits for the next task to be available and then executes it on some random core chosen by the OS scheduler.
All of that is slow... and true across C++, C, and Rust projects.
Short of [OpenMP](https://en.wikipedia.org/wiki/OpenMP), practically every other solution has high dispatch latency and noticeable memory overhead.
OpenMP, however, is not ideal for fine-grained parallelism and is less portable than the C++ and Rust standard libraries.

[![`fork_union` banner](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/fork_union.jpg?raw=true)](https://github.com/ashvardanian/fork_union)

This is where __`fork_union`__ comes in.
It's a C++ 17 library with C 99, Rust, and Zig bindings ([previously Rust implementation was standalone in v1](#why-not-reimplement-it-in-rust)).
It supports pinning threads to specific [NUMA](https://en.wikipedia.org/wiki/Non-uniform_memory_access) nodes or individual CPU cores, making it much easier to ensure data locality and halving the latency of individual loads in Big Data applications.

## Basic Usage

__`Fork Union`__ is dead-simple to use!
There is no nested parallelism, exception handling, or "future promises"; they are banned.
The thread pool itself has a few core operations:

- `try_spawn` to initialize worker threads, and 
- `for_threads` to launch a blocking callback on all threads.

Higher-level APIs for index-addressable tasks are also available:

- `for_n` - for individual evenly-sized tasks,
- `for_n_dynamic` - for individual unevenly-sized tasks,
- `for_slices` - for slices of evenly-sized tasks.

For additional flow control and tuning, following helpers are available:

- `sleep(microseconds)` - for longer naps,
- `terminate` - to kill the threads before the destructor is called,
- `unsafe_for_threads` - to broadcast a callback without blocking,
- `unsafe_join` - to block until the completion of the current broadcast.

On Linux, in C++, given the maturity and flexibility of the HPC ecosystem, it provides [NUMA extensions](#non-uniform-memory-access-numa).
That includes the `linux_colocated_pool` analog of the `basic_pool` and the `linux_numa_allocator` for allocating memory on a specific NUMA node.
Those are out-of-the-box compatible with the higher-level APIs.
Most interestingly, for Big Data applications, a higher-level `distributed_pool` class will address and balance the work across all NUMA nodes.

### Intro in Rust

To integrate into your Rust project, add the following lines to Cargo.toml:

```toml
[dependencies]
fork_union = "2.3.0"                                    # default
fork_union = { version = "2.3.0", features = ["numa"] } # with NUMA support on Linux
```

Or for the preview development version:

```toml
[dependencies]
fork_union = { git = "https://github.com/ashvardanian/fork_union.git", branch = "main-dev" }
```

A minimal example may look like this:

```rust
use fork_union as fu;
let mut pool = fu::spawn(2);
pool.for_threads(|thread_index, colocation_index| {
    println!("Hello from thread # {} on colocation # {}", thread_index + 1, colocation_index + 1);
});
```

Higher-level APIs distribute index-addressable tasks across the threads in the pool:

```rust
pool.for_n(100, |prong| {
    println!("Running task {} on thread # {}",
        prong.task_index + 1, prong.thread_index + 1);
});
pool.for_slices(100, |prong, count| {
    println!("Running slice [{}, {}) on thread # {}",
        prong.task_index, prong.task_index + count, prong.thread_index + 1);
});
pool.for_n_dynamic(100, |prong| {
    println!("Running task {} on thread # {}",
        prong.task_index + 1, prong.thread_index + 1);
});
```

A more realistic example with named threads and error handling may look like this:

```rust
use std::error::Error;
use fork_union as fu;

fn heavy_math(_: usize) {}

fn main() -> Result<(), Box<dyn Error>> {
    let mut pool = fu::ThreadPool::try_spawn(4)?;
    let mut pool = fu::ThreadPool::try_named_spawn("heavy-math", 4)?;
    pool.for_n_dynamic(400, |prong| {
        heavy_math(prong.task_index);
    });
    Ok(())
}
```

For advanced usage, refer to the [NUMA section below](#non-uniform-memory-access-numa).
For convenience Rayon-style parallel iterators pull the `prelude` module and [check out related examples](#rayon-style-parallel-iterators).

### Intro in C++

To integrate into your C++ project, either just copy the `include/fork_union.hpp` file into your project, add a Git submodule, or CMake.
For a Git submodule, run:

```bash
git submodule add https://github.com/ashvardanian/fork_union.git extern/fork_union
```

Alternatively, using CMake:

```cmake
FetchContent_Declare(
    fork_union
    GIT_REPOSITORY https://github.com/ashvardanian/fork_union
    GIT_TAG v2.3.0
)
FetchContent_MakeAvailable(fork_union)
target_link_libraries(your_target PRIVATE fork_union::fork_union)
```

Then, include the header in your C++ code:

```cpp
#include <fork_union.hpp>   // `basic_pool_t`
#include <cstdio>           // `stderr`
#include <cstdlib>          // `EXIT_SUCCESS`

namespace fu = ashvardanian::fork_union;

int main() {
    alignas(fu::default_alignment_k) fu::basic_pool_t pool;
    if (!pool.try_spawn(std::thread::hardware_concurrency())) {
        std::fprintf(stderr, "Failed to fork the threads\n");
        return EXIT_FAILURE;
    }

    // Dispatch a callback to each thread in the pool
    pool.for_threads([&](std::size_t thread_index) noexcept {
        std::printf("Hello from thread # %zu (of %zu)\n", thread_index + 1, pool.threads_count());
    });

    // Execute 1000 tasks in parallel, expecting them to have comparable runtimes
    // and mostly co-locating subsequent tasks on the same thread. Analogous to:
    //
    //      #pragma omp parallel for schedule(static)
    //      for (int i = 0; i < 1000; ++i) { ... }
    //
    // You can also think about it as a shortcut for the `for_slices` + `for`.
    pool.for_n(1000, [](std::size_t task_index) noexcept {
        std::printf("Running task %zu of 1000\n", task_index + 1);
    });
    pool.for_slices(1000, [](std::size_t first_index, std::size_t count) noexcept {
        std::printf("Running slice [%zu, %zu)\n", first_index, first_index + count);
    });

    // Like `for_n`, but each thread greedily steals tasks, without waiting for  
    // the others or expecting individual tasks to have same runtimes. Analogous to:
    //
    //      #pragma omp parallel for schedule(dynamic, 1)
    //      for (int i = 0; i < 3; ++i) { ... }
    pool.for_n_dynamic(3, [](std::size_t task_index) noexcept {
        std::printf("Running dynamic task %zu of 3\n", task_index + 1);
    });
    return EXIT_SUCCESS;
}
```

For advanced usage, refer to the [NUMA section below](#non-uniform-memory-access-numa).
NUMA detection on Linux defaults to AUTO. Override with `-D FORK_UNION_ENABLE_NUMA=ON` or `OFF`.

### Intro in Zig

To integrate into your Zig project, add Fork Union to your `build.zig.zon`:

```zig
.dependencies = .{
    .fork_union = .{
        .url = "https://github.com/ashvardanian/fork_union/archive/refs/tags/v2.3.0.tar.gz",
        .hash = "12200000000000000000000000000000000000000000000000000000000000000000",
    },
},
```

Then import and use in your code:

```zig
const std = @import("std");
const fu = @import("fork_union");

pub fn main() !void {
    var pool = try fu.Pool.init(allocator, 4, .inclusive);
    defer pool.deinit();

    // Execute work on each thread (OpenMP-style parallel)
    pool.forThreads(struct {
        fn work(thread_idx: usize, colocation_idx: usize) void {
            std.debug.print("Thread {}\n", .{thread_idx});
        }
    }.work, {});

    // Distribute 1000 tasks across threads (OpenMP-style parallel for)
    var results = [_]i32{0} ** 1000;
    pool.forN(1000, struct {
        fn process(prong: fu.Prong, ctx: Context) void {
            ctx.results[prong.task_index] = @intCast(prong.task_index * 2);
        }
    }.process, .{ .results = &results });
}
```

Unlike `std.Thread.Pool` (task queue for async work), Fork Union is designed for **data parallelism**
and **tight parallel loops** — think OpenMP's `#pragma omp parallel for` with zero allocations on the hot path.

## Alternatives & Differences

Many other thread-pool implementations are more feature-rich but have different limitations and design goals.

- Modern C++: [`taskflow/taskflow`](https://github.com/taskflow/taskflow), [`progschj/ThreadPool`](https://github.com/progschj/ThreadPool), [`bshoshany/thread-pool`](https://github.com/bshoshany/thread-pool)
- Traditional C++: [`vit-vit/CTPL`](https://github.com/vit-vit/CTPL), [`mtrebi/thread-pool`](https://github.com/mtrebi/thread-pool)
- Rust: [`tokio-rs/tokio`](https://github.com/tokio-rs/tokio), [`rayon-rs/rayon`](https://github.com/rayon-rs/rayon), [`smol-rs/smol`](https://github.com/smol-rs/smol)
- Zig: [`std.Thread.Pool`](https://ziglang.org/documentation/master/std/#std.Thread.Pool)

Those are not designed for the same OpenMP-like use cases as __`fork_union`__.
Instead, they primarily focus on task queuing, which requires significantly more work.

### Locks and Mutexes

Unlike the `std::atomic`, the `std::mutex` is a system call, and it can be expensive to acquire and release.
Its implementations generally have 2 executable paths:

- the fast path, where the mutex is not contended, where it first tries to grab the mutex via a compare-and-swap operation, and if it succeeds, it returns immediately.
- the slow path, where the mutex is contended, and it has to go through the kernel to block the thread until the mutex is available.

On Linux, the latter translates to ["futex"](https://en.wikipedia.org/wiki/Futex) ["system calls"](https://en.wikipedia.org/wiki/System_call), which is expensive.

### Memory Allocations

C++ has rich functionality for concurrent applications, like `std::future`, `std::packaged_task`, `std::function`, `std::queue`, `std::condition_variable`, and so on.
Most of those, I believe, are unusable in Big-Data applications, where you always operate in memory-constrained environments:

- The idea of raising a `std::bad_alloc` exception when there is no memory left and just hoping that someone up the call stack will catch it is not a great design idea for any Systems Engineering.
- The threat of having to synchronize ~200 physical CPU cores across 2-8 sockets and potentially dozens of [NUMA](https://en.wikipedia.org/wiki/Non-uniform_memory_access) nodes around a shared global memory allocator practically means you can't have predictable performance.

As we focus on a simpler ~~concurrency~~ parallelism model, we can avoid the complexity of allocating shared states, wrapping callbacks into some heap-allocated "tasks", and other boilerplate code.
Less work - more performance.

### Atomics and [CAS](https://en.wikipedia.org/wiki/Compare-and-swap)

Once you get to the lowest-level primitives on concurrency, you end up with the `std::atomic` and a small set of hardware-supported atomic instructions.
Hardware implements it differently:

- x86 is built around the "Total Store Order" (TSO) [memory consistency model](https://en.wikipedia.org/wiki/Memory_ordering) and provides `LOCK` variants of the `ADD` and `CMPXCHG`, which act as full-blown "fences" - no loads or stores can be reordered across it.
- Arm, on the other hand, has a "weak" memory model and provides a set of atomic instructions that are not fences, that match the C++ concurrency model, offering `acquire`, `release`, and `acq_rel` variants of each atomic instruction—such as `LDADD`, `STADD`, and `CAS` - which allow precise control over visibility and order, especially with the introduction of "Large System Extension" (LSE) instructions in Armv8.1.

In practice, a locked atomic on x86 requires the cache line in the Exclusive state in the requester's L1 cache.
This would incur a coherence transaction (Read-for-Ownership) if some other core had the line.
Both Intel and AMD handle this similarly.

It makes [Arm and Power much more suitable for lock-free programming](https://arangodb.com/2021/02/cpp-memory-model-migrating-from-x86-to-arm/) and concurrent data structures, but some observations hold for both platforms.
Most importantly, "Compare and Swap" (CAS) is a costly operation and should be avoided whenever possible.

On x86, for example, the `LOCK ADD` [can easily take 50 CPU cycles](https://travisdowns.github.io/blog/2020/07/06/concurrency-costs), being 50x slower than a regular `ADD` instruction, but still easily 5-10x faster than a `LOCK CMPXCHG` instruction.
Once contention rises, the gap naturally widens and is further amplified by the increased "failure" rate of the CAS operation, particularly when the value being compared has already changed.
That's why, for the "dynamic" mode, we resort to using an additional atomic variable as opposed to more typical CAS-based implementations.

### Alignment & False Sharing

The thread-pool needs several atomic variables to synchronize the state.
If those variables are located on the same cache line, they will be "falsely shared" between threads.
This means that when one thread updates one of the variables, it will invalidate the cache line in all other threads, causing them to reload it from memory.
This is a common problem, and the C++ standard recommends addressing it with `alignas(std::hardware_destructive_interference_size)` for your hot variables.

There are, however, caveats.
The `std::hardware_destructive_interference_size` is [generally 64 bytes on x86](https://stackoverflow.com/a/39887282), matching the size of a single cache line.
But in reality, on most x86 machines, [depending on the BIOS "spatial prefetcher" settings](https://www.techarp.com/bios-guide/cpu-adjacent-sector-prefetch/), will [fetch 2 cache lines at a time starting with Sandy Bridge](https://stackoverflow.com/a/72127222).
Because of these rules, padding hot variables to 128 bytes is a conservative but often sensible defensive measure adopted by Folly's `cacheline_align` and Java's `jdk.internal.vm.annotation.Contended`. ￼

## Pro Tips

### Non-Uniform Memory Access (NUMA)

Handling NUMA isn't trivial and is only supported on Linux with the help of the [`libnuma` library](https://github.com/numactl/numactl).
It provides the `mbind` interface to pin specific memory regions to particular NUMA nodes, as well as helper functions to query the system topology, which are exposed via the `fork_union::numa_topology` template.

Let's say you are working on a Big Data application, like brute-forcing Vector Search using the [SimSIMD](https://github.com/ashvardanian/simsimd) library on a 2 dual-socket CPU system, similar to [USearch](https://github.com/unum-cloud/usearch/pulls).
The first part of that program may be responsible for sharding the incoming stream of data between distinct memory regions.
That part, in our simple example will be single-threaded:

```cpp
#include <vector> // `std::vector`
#include <span> // `std::span`
#include <fork_union.hpp> // `linux_numa_allocator`, `numa_topology_t`, `linux_distributed_pool_t`
#include <simsimd/simsimd.h> // `simsimd_f32_cos`, `simsimd_distance_t`

namespace fu = ashvardanian::fork_union;
using floats_alloc_t = fu::linux_numa_allocator<float>;

constexpr std::size_t dimensions = 768; /// Matches most BERT-like models
static std::vector<float, floats_alloc_t> first_half(floats_alloc_t(0));
static std::vector<float, floats_alloc_t> second_half(floats_alloc_t(1));
static fu::numa_topology_t numa_topology;
static fu::linux_distributed_pool_t distributed_pool;

/// Dynamically shards incoming vectors across 2 nodes in a round-robin fashion.
void append(std::span<float, dimensions> vector) {
    bool put_in_second = first_half.size() > second_half.size();
    if (put_in_second) second_half.insert(second_half.end(), vector.begin(), vector.end());
    else first_half.insert(first_half.end(), vector.begin(), vector.end());
}
```

The concurrent part would involve spawning threads adjacent to every memory pool to find the best `search_result_t`.
The primary `search` function, in ideal world would look like this:

1. Each thread finds the best match within its "slice" of a NUMA node, tracking the best distance and index in a local CPU register.
2. All threads in each NUMA node atomically synchronize using a NUMA-local instance of `search_result_t`.
3. The main thread collects aggregates of partial results from all NUMA nodes.

That is, however, overly complicated to implement.
Such tree-like hierarchical reductions are optimal in a theoretical sense. Still, assuming the relative cost of spin-locking once at the end of a thread scope and the complexity of organizing the code, the more straightforward path is better.
A minimal example would look like this:

```cpp
/// On each NUMA node we'll synchronize the threads
struct search_result_t {
    simsimd_distance_t best_distance {std::numeric_limits<simsimd_distance_t>::max()};
    std::size_t best_index {0};
};

inline search_result_t pick_best(search_result_t const& a, search_result_t const& b) noexcept {
    return a.best_distance < b.best_distance ? a : b;
}

/// Uses all CPU threads to search for the closest vector to the @p query.
search_result_t search(std::span<float, dimensions> query) {

    bool const need_to_spawn_threads = distributed_pool.threads_count() == 0;
    if (need_to_spawn_threads) {
        assert(numa_topology.try_harvest() && "Failed to harvest NUMA topology");
        assert(numa_topology.nodes_count() == 2 && "Expected exactly 2 NUMA nodes");
        assert(distributed_pool.try_spawn(numa_topology, sizeof(search_result_t)) && "Failed to spawn NUMA pools");
    }

    search_result_t result;
    fu::spin_mutex_t result_update; // ? Lighter `std::mutex` alternative w/out system calls

    std::size_t const total_vectors =
        (first_half.size() + second_half.size()) / dimensions;

    auto slices = distributed_pool.for_slices(total_vectors,
        [&](fu::colocated_prong<> first, std::size_t count) noexcept {

        bool const in_second = first.colocation != 0;
        auto const &shard = in_second ? second_half : first_half;
        std::size_t const shard_base = in_second ? first_half.size() / dimensions : 0;
        std::size_t const local_begin = first.task - shard_base;

        search_result_t thread_local_result;
        for (std::size_t i = 0; i < count; ++i) {
            std::size_t const local_index = local_begin + i;
            std::size_t const global_index = shard_base + local_index;

            simsimd_distance_t distance;
            simsimd_f32_cos(query.data(), shard.data() + local_index * dimensions, dimensions, &distance);
            thread_local_result = pick_best(thread_local_result, {distance, global_index});
        }

        // ! Still synchronizing over a shared mutex for brevity.
        std::lock_guard<fu::spin_mutex_t> lock(result_update);
        result = pick_best(result, thread_local_result);
    });
    slices.join();
    return result;
}
```

In a dream world, we would call `distributed_pool.for_n`, but there is no clean way to make the scheduling processes aware of the data distribution in an arbitrary application, so that's left to the user.
The `for_slices` helper provides colocated metadata (`fu::colocated_prong`) that lets you pick the right shard of data based on the NUMA node, while keeping scheduling inside the distributed pool.
For more flexibility around building higher-level low-latency systems, there are unsafe APIs expecting you to manually "join" the broadcasted calls, like `unsafe_for_threads` and `unsafe_join`.

### Efficient Busy Waiting

Here's what "busy waiting" looks like in C++:

```cpp
while (!has_work_to_do())
    std::this_thread::yield();
```

On Linux, the `std::this_thread::yield()` translates into a `sched_yield` system call, which means context switching to the kernel and back.
Instead, you can replace the `standard_yield_t` wrapper with a platform-specific micro-wait instruction, which is much cheaper.
Those instructions, like [`WFET` on Arm](https://developer.arm.com/documentation/ddi0602/2025-03/Base-Instructions/WFET--Wait-for-event-with-timeout-), generally hint the CPU to transition to a low-power state.

| Wrapper         | ISA          | Instruction | Privileges |
| --------------- | ------------ | ----------- | ---------- |
| `x86_pause_t`   | x86          | `PAUSE`     | R3         |
| `x86_tpause_t`  | x86+WAITPKG  | `TPAUSE`    | R3         |
| `arm64_yield_t` | AArch64      | `YIELD`     | EL0        |
| `arm64_wfet_t`  | AArch64+WFXT | `WFET`      | EL0        |
| `risc5_pause_t` | RISC-V       | `PAUSE`     | U          |

No kernel calls.
No futexes.
Works in tight loops.

### Rayon-style Parallel Iterators

For Rayon-style ergonomics, use the parallel iterator API with the `prelude`.
Unlike Rayon, Fork Union's parallel iterators don't depend on the global state and allow explicit control over the thread pool and scheduling strategy.
For statically shaped workloads, the default static scheduling is more efficient: 

```rust
use fork_union as fu;
use fork_union::prelude::*;

let mut pool = fu::spawn(4);
let mut data: Vec<usize> = (0..1000).collect();

(&data[..])
    .into_par_iter()
    .with_pool(&mut pool)
    .for_each(|value| {
        println!("Value: {}", value);
    });
```

For dynamic work-stealing, use `with_schedule` with `DynamicScheduler`:

```rust
(&mut data[..])
    .into_par_iter()
    .with_schedule(&mut pool, DynamicScheduler)
    .for_each(|value| {
        *value *= 2;
    });
```

This easily composes with other iterator adaptors, like `map`, `filter`, and `zip`:

```rust
(&data[..])
    .into_par_iter()
    .filter(|&x| x % 2 == 0)
    .map(|x| x * x)
    .with_pool(&mut pool)
    .for_each(|value| {
        println!("Squared even: {}", value);
    });
```

Moreover, each thread can maintain its own scratch space to avoid contention during reductions.
Cache-line alignment via `CacheAligned` prevents false sharing:

```rust
// Cache-line aligned wrapper to prevent false sharing
let mut scratch: Vec<CacheAligned<usize>> =
    (0..pool.threads()).map(|_| CacheAligned(0)).collect();

(&data[..])
    .into_par_iter()
    .with_pool(&mut pool)
    .fold_with_scratch(scratch.as_mut_slice(), |acc, value, _prong| {
        acc.0 += *value;
    });
let total: usize = scratch.iter().map(|a| a.0).sum();
```

## Performance

One of the most common parallel workloads is the N-body simulation ¹.
Implementations are available in both C++ and Rust in `scripts/nbody.cpp` and `scripts/nbody.rs`, respectively.
Both are lightweight and involve little logic outside of number-crunching, so both can be easily profiled with `time` and introspected with `perf` Linux tools.
Additional NUMA-aware Search examples are available in `scripts/search.rs`.

---

C++ benchmarking results for $N=128$ bodies and $I=1e6$ iterations:

| Machine        | OpenMP (D) | OpenMP (S) | Fork Union (D) | Fork Union (S) |
| :------------- | ---------: | ---------: | -------------: | -------------: |
| 16x Intel SPR  |      18.9s |      12.4s |          16.8s |           8.7s |
| 12x Apple M2   | 1m:34.8s ² | 1m:25.9s ² |          31.5s |          20.3s |
| 96x Graviton 4 |      32.2s |      20.8s |          39.8s |          26.0s |

Rust benchmarking results for $N=128$ bodies and $I=1e6$ iterations:

| Machine        |  Rayon (D) |  Rayon (S) | Fork Union (D) | Fork Union (S) |
| :------------- | ---------: | ---------: | -------------: | -------------: |
| 16x Intel SPR  |    🔄 45.4s |    🔄 32.1s | 18.1s, 🔄 22.4s | 12.4s, 🔄 12.9s |
| 12x Apple M2   | 🔄 1m:47.8s | 🔄 1m:07.1s | 24.5s, 🔄 26.8s | 11.0s, 🔄 11.8s |
| 96x Graviton 4 | 🔄 2m:13.9s | 🔄 1m:35.6s |          18.9s |          10.1s |

> ¹ Another common workload is "Parallel Reductions" covered in a separate [repository](https://github.com/ashvardanian/ParallelReductionsBenchmark).
> ² When a combination of performance and efficiency cores is used, dynamic stealing may be more efficient than static slicing. It's also fair to say, that OpenMP is not optimized for AppleClang.
> 🔄 Rotation emoji stands for iterators, the default way to use Rayon and the opt-in slower, but more convenient variant for Fork Union.

You can rerun those benchmarks with the following commands:

```bash
cmake -B build_release -D CMAKE_BUILD_TYPE=Release
cmake --build build_release --config Release
time NBODY_COUNT=128 NBODY_ITERATIONS=1000000 NBODY_BACKEND=fork_union_static build_release/fork_union_nbody
time NBODY_COUNT=128 NBODY_ITERATIONS=1000000 NBODY_BACKEND=fork_union_dynamic build_release/fork_union_nbody
```

> Consult the header of `scripts/nbody.cpp` and `scripts/nbody.rs` for additional benchmarking options.

## Safety & Logic

There are only 3 core atomic variables in this thread-pool, and 1 for dynamically-stealing tasks.
Let's call every invocation of a `for_*` API - a "fork", and every exit from it a "join".

| Variable           | Users Perspective            | Internal Usage                        |
| :----------------- | :--------------------------- | :------------------------------------ |
| `stop`             | Stop the entire thread-pool  | Tells workers when to exit the loop   |
| `fork_generation`  | "Forks" called since init    | Tells workers to wake up on new forks |
| `threads_to_sync`  | Threads not joined this fork | Tells main thread when workers finish |
| `dynamic_progress` | Progress within this fork    | Tells workers which jobs to take      |

### Why don't we need atomics for "total_threads"?

The only way to change the number of threads is to `terminate` the entire thread-pool and then `try_spawn` it again.
Either of those operations can only be called from one thread at a time and never coincide with any running tasks.
That's ensured by the `stop`.

### Why don't we need atomics for a "job pointer"?

A new task can only be submitted from one thread that updates the number of parts for each new fork.
During that update, the workers are asleep, spinning on old values of `fork_generation` and `stop`.
They only wake up and access the new value once `fork_generation` increments, ensuring safety.

### How do we deal with overflows and `SIZE_MAX`-sized tasks?

The library entirely avoids saturating multiplication and only uses one saturating addition in "release" builds.
To test the consistency of arithmetic, the C++ template class can be instantiated with a custom `index_t`, such as `std::uint8_t` or `std::uint16_t`.
In the former case, no more than 255 threads can operate, and no more than 255 tasks can be addressed, allowing us to easily test every weird corner case of [0:255] threads competing for [0:255] tasks.

### Why not reimplement it in Rust?

The original Rust implementation was a standalone library, but in essence, Rust doesn't feel designed for parallelism, concurrency, and expert Systems Engineering.
It enforces stringent safety rules, which is excellent for building trustworthy software, but realistically, it makes lock-free concurrent programming with minimal memory allocations too complicated.
Now, the Rust library is a wrapper over the C binding of the C++ core implementation.

## Testing and Benchmarking

To run the C++ tests, use CMake:

```bash
cmake -B build_release -D CMAKE_BUILD_TYPE=Release -D BUILD_TESTING=ON
cmake --build build_release --config Release -j
ctest --test-dir build_release                  # run all tests
build_release/fork_union_nbody                  # run the benchmarks
```

For C++ debug builds, consider using the VS Code debugger presets or the following commands:

```bash
cmake -B build_debug -D CMAKE_BUILD_TYPE=Debug -D BUILD_TESTING=ON
cmake --build build_debug --config Debug        # build with Debug symbols
build_debug/fork_union_test_cpp20               # run a single test executable
```

To run static analysis:

```bash
sudo apt install cppcheck clang-tidy
cmake --build build_debug --target cppcheck     # detects bugs & undefined behavior
cmake --build build_debug --target clang-tidy   # suggest code improvements
```

To include NUMA, Huge Pages, and other optimizations on Linux, make sure to install dependencies:

```bash
sudo apt-get -y install libnuma-dev libnuma1                # NUMA
sudo apt-get -y install libhugetlbfs-dev libhugetlbfs-bin   # Huge Pages
sudo ln -s /usr/bin/ld.hugetlbfs /usr/share/libhugetlbfs/ld # Huge Pages linker
```

To build with an alternative compiler, like LLVM Clang, use the following command:

```bash
sudo apt-get install libomp-15-dev clang++-15 # OpenMP version must match Clang
cmake -B build_debug -D CMAKE_BUILD_TYPE=Debug -D CMAKE_CXX_COMPILER=clang++-15
cmake --build build_debug --config Debug
build_debug/fork_union_test_cpp20
```

Or on macOS with Apple Clang:

```bash
brew install llvm@20
cmake -B build_debug -D CMAKE_BUILD_TYPE=Debug -D CMAKE_CXX_COMPILER=$(brew --prefix llvm@20)/bin/clang++
cmake --build build_debug --config Debug
build_debug/fork_union_test_cpp20
```

For Rust, use the following commands:

```bash
rustup toolchain install                # for Alloc API
cargo build --features numa             # for NUMA support on Linux
cargo test --release                    # to run the tests fast
cargo test --features numa --release    # for NUMA tests on Linux
```

Rust provides a lot of tooling for concurrency testing, like Miri and Loom.
Most of it, however, is not applicable in this case, as the core logic is implemented in C++.

To automatically detect the Minimum Supported Rust Version (MSRV):

```sh
cargo +stable install cargo-msrv
cargo msrv find --ignore-lockfile
```

---

For Zig, use the following commands:

```bash
zig build test                          # run tests
zig build nbody -Doptimize=ReleaseFast  # build benchmark
zig build -Dnuma=true                   # enable NUMA support (Linux)

# Run benchmark
time NBODY_COUNT=128 NBODY_ITERATIONS=1000000 NBODY_BACKEND=fork_union_static \
    ./zig-out/bin/nbody_zig
```

## License

Licensed under the Apache License, Version 2.0. See `LICENSE` for details.
