# ForkUnion

[![`ForkUnion` banner](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/ForkUnion.jpg?raw=true)](https://github.com/ashvardanian/ForkUnion)

__ForkUnion__ is a NUMA-aware fork-join thread-pool for C++, C, Rust, and Zig — built for the tightest `#pragma omp parallel for`-style loops, not task queues. 🍴
It already powers NUMA-sharded vector search with [USearch](https://github.com/unum-cloud/usearch), LLM KV-cache and attention kernels with [NumKong](https://github.com/ashvardanian/NumKong), and unbalanced bioinformatics workloads — thousands of combinatorial tasks per core — with [StringZilla](https://github.com/ashvardanian/StringZilla).

On the hot path it makes __zero__ [heap allocations](#memory-allocations), __zero__ [system calls](#locks-and-mutexes), __zero__ [CAS operations](#atomics-and-cas), and suffers no [false-sharing](#alignment--false-sharing) of cache-lines.
So dispatch latency stays flat into the hundreds of cores, precisely where task-queue runtimes like Rayon collapse and even OpenMP begins to slip.
It is also unique in parking idle workers on a __hardware address monitor__ — x86 `UMONITOR`/`UMWAIT`, Arm `WFET`, RISC-V `Zawrs` — so a worker light-sleeps on the exact cache-line it is waiting for and the silicon wakes it the instant another core writes there, with no hot spinning and no kernel futex.
That bet compounds as sockets multiply: Intel Xeon Platinum and NVIDIA Vera pack up to 8 sockets per node, where shared CAS thrashes the interconnect and private `fetch_add` cursors keep synchronization off it.

> __One parallel-for dispatch:__ ForkUnion is often __~3–6× faster than OpenMP__ and __~12–16× faster than Rayon and Taskflow__ — the fork-join tax, paid once per loop, cut by an order of magnitude.
> [Full tables ↓](#performance)

It is exhaustively tested for boundary-condition scheduling with miniaturized `uint8_t` indices, even runs on your big-endian 32-bit IBM mainframe, and ships with `no_std` and Miri coverage.
The core is a C++ 17 library; the C 99, Rust, and Zig APIs bind it, and all four can pin threads to [NUMA](https://en.wikipedia.org/wiki/Non-uniform_memory_access) nodes or individual cores and allocate node-local memory.
Despite being far more deeply tied to hardware and the OS than most alternatives, ForkUnion runs on __six operating systems__ — Linux, FreeBSD, Windows, macOS, Android, and iOS — including asymmetric compute and memory topologies.
Topology harvesting and thread placement work on all six; NUMA-local memory placement is implemented on Linux, FreeBSD, and Windows, and elsewhere allocation falls back to a single memory domain.

## Basic Usage

__`ForkUnion`__ is dead-simple to use!
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
- `unsafe_for_threads` - to broadcast a callback without blocking, returning a generation token,
- `unsafe_join` - to block until the completion of a broadcasted generation,
- `is_complete` - to poll a generation token for completion without blocking.

Every dispatch is identified by an always-odd `generation` token.
On `caller_exclusive_k` pools you can dispatch, overlap your own work, poll `is_complete`, and join - the classic poll-then-join pattern.
On `caller_inclusive_k` pools the calling thread owes one slice of the work, which only runs inside `unsafe_join`, so completion can't be reached by polling alone.
The same rule shapes the RAII guard returned by `for_threads` and the `for_n` family: on exclusive pools the work starts at the guard's construction, while on inclusive pools it runs at `join` or destruction.

On Linux, in C++, given the maturity and flexibility of the HPC ecosystem, it provides [NUMA extensions](#non-uniform-memory-access-numa).
That includes the `colocated_pool` analog of the `flat_pool` and the `linux_numa_allocator` for allocating memory in a specific memory domain.
Those are out-of-the-box compatible with the higher-level APIs.
Most interestingly, for Big Data applications, a higher-level `distributed_pool` class will address and balance the work across all compute domains.

### Intro in Rust

To integrate into your Rust project, add the following lines to Cargo.toml:

```toml
[dependencies]
forkunion = "3.0.2"                                          # detect what the platform offers
forkunion = { version = "3.0.2", features = ["portable"] }   # STL thread pool only
forkunion = { version = "3.0.2", features = ["place-memory-on-domain"] } # require NUMA-aware allocations
```

Or for the preview development version:

```toml
[dependencies]
forkunion = { git = "https://github.com/ashvardanian/ForkUnion.git", branch = "main-dev" }
```

A minimal example may look like this:

```rust
use forkunion as fu;
let topology = fu::Topology::new().expect("Failed to detect hardware topology");
let mut pool = fu::spawn(&topology, 2);
pool.for_threads(&|thread_index, compute_domain_index| {
    println!("Hello from thread # {} on compute domain # {}", thread_index + 1, compute_domain_index + 1);
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
use forkunion as fu;

fn heavy_math(_: usize) {}

fn main() -> Result<(), Box<dyn Error>> {
    let topology = fu::Topology::new()?;
    let mut pool = fu::ThreadPool::try_named_spawn(&topology, "heavy-math", 4)?;
    pool.for_n_dynamic(400, |prong| {
        heavy_math(prong.task_index);
    });
    Ok(())
}
```

The `Topology` handle reports the [hardware topology](#hardware-topology), to size and place work:

```rust
let topology = fu::Topology::new()?;
for domain in 0..topology.compute_domains_count() {
    let domain = fu::ComputeDomain(domain);
    println!("domain {}: {} cores, level {}, allocate on memory domain {}",
        domain.get(), topology.logical_cores_count_in(domain),
        topology.compute_level_in(domain), topology.local_memory_of(domain).get());
}
```

For advanced usage, refer to the [NUMA section below](#non-uniform-memory-access-numa).
For convenience Rayon-style parallel iterators pull the `prelude` module and [check out related examples](#rayon-style-parallel-iterators).

### Intro in C++

To integrate into your C++ project, either copy the `include/` directory into your project, add a Git submodule, or CMake.
The `forkunion.hpp` umbrella is a list of `#include`s; the implementation lives beside it in `include/forkunion/`, one header per concern, so a platform backend can be read or replaced on its own.
For a Git submodule, run:

```bash
git submodule add https://github.com/ashvardanian/ForkUnion.git extern/forkunion
```

Alternatively, using CMake:

```cmake
FetchContent_Declare(
    forkunion
    GIT_REPOSITORY https://github.com/ashvardanian/ForkUnion
    GIT_TAG v3.0.2
)
FetchContent_MakeAvailable(forkunion)
target_link_libraries(your_target PRIVATE forkunion::header)
```

Then, include the header in your C++ code:

```cpp
#include <forkunion.hpp>    // `flat_pool_t`
#include <cstdio>           // `stderr`
#include <cstdlib>          // `EXIT_SUCCESS`

namespace fu = ashvardanian::forkunion;

int main() {
    alignas(fu::default_alignment_k) fu::flat_pool_t pool;
    if (!pool.try_spawn(fu::allowed_cores_count())) {
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
Every kernel and ISA facility the library uses is detected by default.
CMake pins each with an `AUTO`/`ON`/`OFF` tri-state like `-D FORKUNION_WITH_PLACE_MEMORY_ON_DOMAIN=ON` or `-D FORKUNION_WITH_PLACE_THREADS_BY_AFFINITY=OFF`.
Finer preprocessor gates - like the cache-line hints `FU_WITH_DEMOTE_CACHE_LINES` and `FU_WITH_PROMOTE_CACHE_LINES`, which push a just-written line toward the shared last-level cache via x86 `CLDEMOTE`, Arm `DC CVAC`, or RISC-V `Zicbom` so a consumer core finds it faster - accept the same overrides as compile definitions.
Call `fu_comptime_capabilities()` to see what survived the build, `fu_runtime_capabilities()` to see what the machine underneath actually offers, and `fu_name_capabilities()` to render either mask as text - every binding exposes the same trio, Rust spelling it `runtime_capabilities()` and Zig `runtimeCapabilities`, alongside a version accessor: `fu_version_major`/`_minor`/`_patch` in C, `version()` in Rust and Zig, and the `FORKUNION_VERSION_*` macros in C++.

### Intro in Zig

To integrate into your Zig project, let `zig fetch` pin the dependency and its content hash into `build.zig.zon`:

```bash
zig fetch --save=forkunion https://github.com/ashvardanian/ForkUnion/archive/refs/tags/v3.0.2.tar.gz
```

Then import and use in your code:

```zig
const std = @import("std");
const fu = @import("forkunion");

const Context = struct { results: []i32 };

pub fn main() !void {
    const topology = try fu.Topology.init();
    defer topology.deinit();
    var pool = try fu.Pool.init(topology, 4, .inclusive);
    defer pool.deinit();

    // Execute work on each thread (OpenMP-style parallel)
    pool.forThreads(struct {
        fn work(thread_idx: usize, compute_domain_idx: usize) void {
            _ = compute_domain_idx;
            std.debug.print("Thread {}\n", .{thread_idx});
        }
    }.work, {});

    // Distribute 1000 tasks across threads (OpenMP-style parallel for)
    var results = [_]i32{0} ** 1000;
    pool.forN(1000, struct {
        fn process(prong: fu.Prong, ctx: Context) void {
            ctx.results[prong.task_index] = @intCast(prong.task_index * 2);
        }
    }.process, Context{ .results = results[0..] });
}
```

The `Topology` handle reports the [hardware topology](#hardware-topology), to size and place work:

```zig
const topology = try fu.Topology.init();
defer topology.deinit();
var domain: usize = 0;
while (domain < topology.countComputeDomains()) : (domain += 1) {
    std.debug.print("domain {d}: {d} cores, level {d}, allocate on memory domain {d}\n", .{
        domain, topology.countLogicalCoresIn(domain), topology.computeLevelIn(domain), topology.localMemoryOf(domain),
    });
}
```

Unlike `std.Thread.Pool` task queue for async work, ForkUnion is designed for __data parallelism__
and __tight parallel loops__ — think OpenMP's `#pragma omp parallel for` with zero allocations on the hot path.

### Intro in C

ForkUnion provides a pure C99 API via `forkunion.h`, wrapping the C++ implementation in pre-compiled libraries: `libforkunion_static.a` or `libforkunion_shared.so`.
Installing also drops a `forkunion.pc`, so `pkg-config --cflags --libs forkunion` works outside CMake.
The C API uses opaque `fu_pool_t` handles and function pointers for callbacks, making it compatible with any C99+ compiler.

To integrate using CMake:

```cmake
FetchContent_Declare(
    forkunion
    GIT_REPOSITORY https://github.com/ashvardanian/ForkUnion
    GIT_TAG v3.0.2
)
FetchContent_MakeAvailable(forkunion)
target_link_libraries(your_target PRIVATE forkunion::static)
```

A minimal C example:

```c
#include <stdio.h>      // printf
#include <forkunion.h>  // fu_pool_t, fu_pool_new, fu_pool_spawn

void hello_callback(void *context, size_t thread, size_t compute_domain) {
    (void)context;
    printf("Hello from thread %zu (compute_domain %zu)\n", thread, compute_domain);
}

int main(void) {
    fu_topology_t topology = fu_topology_new();
    fu_pool_t pool = fu_pool_new("my_pool", fu_capabilities_all_k);
    if (!topology || !pool ||
        !fu_pool_spawn(topology, pool, fu_logical_cores_count(topology), fu_caller_inclusive_k))
        return 1;

    fu_pool_for_threads(pool, hello_callback, NULL);
    fu_pool_delete(pool);
    fu_topology_delete(topology);
    return 0;
}
```

The `fu_`-prefixed functions report the [hardware topology](#hardware-topology), to size and place work:

```c
fu_topology_t topology = fu_topology_new();
for (size_t domain = 0; domain < fu_compute_domains_count(topology); ++domain) {
    printf("domain %zu: %zu cores, level %zu, allocate on memory domain %zu\n",
           domain, fu_logical_cores_count_in(topology, domain), fu_compute_level_in(topology, domain),
           fu_local_memory_of(topology, domain));
}
fu_topology_delete(topology);
```

For parallel tasks with context:

```c
struct task_context {
    int *data;
    size_t size;
};

void process_task(void *ctx, size_t task, size_t thread, size_t compute_domain) {
    (void)thread; (void)compute_domain;
    struct task_context *context = (struct task_context *)ctx;
    context->data[task] = task * 2;
}

int main(void) {
    fu_topology_t topology = fu_topology_new();
    fu_pool_t pool = fu_pool_new("tasks", fu_capabilities_all_k);
    fu_pool_spawn(topology, pool, 4, fu_caller_inclusive_k);

    int data[100] = {0};
    struct task_context ctx = { .data = data, .size = 100 };
    fu_pool_for_n(pool, 100, process_task, &ctx);        // static scheduling
    fu_pool_for_n_dynamic(pool, 100, process_task, &ctx); // dynamic scheduling

    fu_pool_delete(pool);
    fu_topology_delete(topology);
    return 0;
}
```

#### GCC Nested Functions Extension

GCC supports [nested functions](https://gcc.gnu.org/onlinedocs/gcc/Nested-Functions.html) that can capture variables from the enclosing scope:

```c
#include <stdio.h>
#include <stdatomic.h>
#include <forkunion.h>

int main(void) {
    fu_topology_t topology = fu_topology_new();
    fu_pool_t pool = fu_pool_new("gcc_nested", fu_capabilities_all_k);
    fu_pool_spawn(topology, pool, 4, fu_caller_inclusive_k);

    atomic_size_t counter = 0;

    // GCC nested function - captures 'counter' from enclosing scope
    void nested_callback(void *ctx, size_t task, size_t thread, size_t compute_domain) {
        (void)ctx; (void)thread; (void)compute_domain;
        atomic_fetch_add(&counter, 1);
    }

    fu_pool_for_n(pool, 100, nested_callback, NULL);
    printf("Completed %zu tasks\n", (size_t)atomic_load(&counter));

    fu_pool_delete(pool);
    fu_topology_delete(topology);
    return 0;
}
```

Compile: `gcc -std=c11 test.c -lforkunion_static -lstdc++ -lpthread`
The archive is C++ behind a C facade, so a C consumer names the C++ runtime itself — `-lc++` under Apple's toolchain.

#### Clang Blocks Extension

Clang provides [blocks](https://clang.llvm.org/docs/BlockLanguageSpec.html) with `^{}` syntax:

```c
#include <stdio.h>
#include <stdatomic.h>
#include <Block.h>
#include <forkunion.h>

typedef void (^task_block_t)(void *, size_t, size_t, size_t);

struct block_wrapper { task_block_t block; };

void block_wrapper_fn(void *ctx, size_t task, size_t thread, size_t compute_domain) {
    ((struct block_wrapper *)ctx)->block(NULL, task, thread, compute_domain);
}

int main(void) {
    fu_topology_t topology = fu_topology_new();
    fu_pool_t pool = fu_pool_new("clang_blocks", fu_capabilities_all_k);
    fu_pool_spawn(topology, pool, 4, fu_caller_inclusive_k);

    __block atomic_size_t counter = 0;

    task_block_t my_block = ^(void *c, size_t task, size_t t, size_t col) {
        (void)c; (void)t; (void)col;
        atomic_fetch_add(&counter, 1);
    };

    task_block_t heap_block = Block_copy(my_block);
    struct block_wrapper wrapper = { .block = heap_block };

    fu_pool_for_n(pool, 100, block_wrapper_fn, &wrapper);

    Block_release(heap_block);
    printf("Completed %zu tasks\n", (size_t)atomic_load(&counter));

    fu_pool_delete(pool);
    fu_topology_delete(topology);
    return 0;
}
```

Compile: `clang -std=c11 -fblocks test.c -lforkunion_static -lc++ -lpthread -lBlocksRuntime`

## Alternatives & Differences

Most "thread-pools" are not thread-pools at all, but task-queues - think a `std::queue<std::function<void()>>` behind a `std::mutex`, executing each heap-allocated task on whatever core the OS scheduler picks.
That shape is what most alternatives are built around, and it is the wrong shape for a tight `parallel for`:

- Modern C++: [`taskflow/taskflow`](https://github.com/taskflow/taskflow), [`progschj/ThreadPool`](https://github.com/progschj/ThreadPool), [`bshoshany/thread-pool`](https://github.com/bshoshany/thread-pool)
- Traditional C++: [`vit-vit/CTPL`](https://github.com/vit-vit/CTPL), [`mtrebi/thread-pool`](https://github.com/mtrebi/thread-pool)
- Rust: [`tokio-rs/tokio`](https://github.com/tokio-rs/tokio), [`rayon-rs/rayon`](https://github.com/rayon-rs/rayon), [`smol-rs/smol`](https://github.com/smol-rs/smol)
- Zig: [`std.Thread.Pool`](https://ziglang.org/documentation/master/std/#std.Thread.Pool)

__Reach for ForkUnion__ when you have data-parallel loops, bulk-synchronous phases, or NUMA-sharded scans, and you want them to dispatch in nanoseconds with neither an allocator nor the kernel on the hot path.
__Reach for something else__ when you need task graphs, async I/O, futures and promises, work that outlives its scope, or nested parallelism — Taskflow, Tokio, and oneTBB are built for exactly that, and ForkUnion deliberately bans it.

The trade-offs line up like this, verified against each project's own source and documentation:

| Property                        | ForkUnion               | OpenMP             | Rayon                      | Taskflow              | oneTBB                |
| :------------------------------ | :---------------------- | :----------------- | :------------------------- | :-------------------- | :-------------------- |
| Hot-path heap allocations       | __None__                | ≈ None ¹           | Per spawned task ²         | Per task node         | Per task, pooled      |
| Syscalls to dispatch            | __None__                | Spin, then futex   | Spin/yield, then futex     | Spin, then futex      | Spin, then futex      |
| Dynamic-schedule primitive      | __`fetch_add` cursors__ | CAS retry loop ³   | Chase-Lev CAS deque ⁴      | Chase-Lev CAS deque ⁴ | Chase-Lev CAS deque ⁴ |
| NUMA pinning & local allocators | __Yes__                 | Yes ⁵              | No                         | No ⁶                  | Partial ⁷             |
| Hardware timed-wait             | __Yes__                 | x86 only, opt-in ⁸ | No                         | No                    | No                    |
| Languages                       | C++ · C · Rust · Zig    | C · C++ · Fortran  | Rust                       | C++                   | C++                   |
| Design center                   | Fork-join loops         | Loops + tasks      | Stealing tasks + iterators | Task-graph DAG        | Stealing tasks        |

> ¹ OpenMP reuses a persistent thread team, so a repeated `parallel for` over a fixed-size team does not re-create it; the per-region worksharing descriptor is reused but not provably allocation-free on every path.
> ² Rayon keeps `join` and parallel-iterator jobs on the stack, but `scope`/`spawn` heap-allocate one job per task, and the Chase-Lev deque grows on the heap under deep nesting.
> ³ libgomp and libomp hand out `schedule(dynamic)` iterations with an atomic fetch-add plus a CAS retry loop.
> ⁴ Rayon, Taskflow, and oneTBB all steal through a Chase-Lev deque whose steal path is a CAS.
> ⁵ `OMP_PROC_BIND` / `OMP_PLACES` pin threads, and the OpenMP 5 allocator and memory-space API places memory.
> ⁶ Taskflow exposes only a manual `WorkerInterface` hook for affinity, nothing automatic.
> ⁷ oneTBB pins threads to a NUMA node via `task_arena` constraints, but does not place memory locally for you.
> ⁸ Only LLVM/Intel `libomp` issues x86 `UMONITOR` / `UMWAIT`, gated behind `KMP_USER_LEVEL_MWAIT` and off by default; there is no Arm `WFET` or RISC-V `Zawrs` path in any OpenMP runtime.

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
This would incur a Read-for-Ownership coherence transaction if some other core had the line.
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
Because of these rules, padding hot variables to 128 bytes is a conservative but often sensible defensive measure adopted by Folly's `cacheline_align` and Java's `jdk.internal.vm.annotation.Contended`.

## Pro Tips

### Hardware Topology

A machine is described along two independent axes.
A __compute domain__ is a bindable group of cores sharing a Quality-of-Service class _and_ locality — it is what a pool spawns onto, and the index a worker callback receives.
A __memory domain__ is a bank of memory with its own capacity and access cost — it is what the allocator targets, may be _cpuless_ like a CXL expander, and may serve several compute domains at once.
The axes stay separate because they don't line up: performance and efficiency cores often share one memory controller, and equally-fast cores are often split across cache clusters or sockets.

Each axis carries a __level__, a dense ordinal grouping domains of like performance.
Levels can be fewer than domains, since several domains may share one, and the two axes count in opposite directions: compute levels grow with performance, memory levels grow with _distance_, placing HBM below DDR and CXL above.
Two objects split the answers by provenance, not by axis: a `Topology` holds what the platform __declares__ — domains, cores, QoS classes, volumes, on both axes — while a `Fabric` holds what ForkUnion __observes__: per-edge latencies, bandwidths, and distances, and the per-medium tiers derived from them.
The pipeline is `try_harvest` all the way down: a `Topology` harvests the declared structure from the OS and stays immutable, a pool spawns on it, and a `Fabric` then harvests the observed performance from the silicon through that pool's pinned workers, pointer-chasing and streaming every reachable edge.
A memory level is a property of the medium, independent of the querying core: it keys on the best bandwidth any initiator sustains to the pool, ties split by the best latency, so a 3 TB/s HBM pool outranks DDR even at equal latency.

|                   | Compute axis            | Memory axis            |
| ----------------- | ----------------------- | ---------------------- |
| Count domains     | `compute_domains_count` | `memory_domains_count` |
| Count levels      | `compute_levels_count`  | `memory_levels_count`  |
| Level of a domain | `compute_level_in`      | `memory_level_in`      |
| Faster means      | __higher__              | __lower__              |

Names are spelled here as in Rust; C prefixes them with `fu_`, and Zig spells them in camelCase.
Domain counts and compute levels answer from the `Topology`; memory levels and the per-edge magnitudes answer from a harvested `Fabric`, which C prefixes with `fu_fabric_`.

Beyond the level ordinals, two magnitudes describe a compute domain, both best-effort.
`compute_capacity_in` forwards the kernel's own rating of a core, read from `/sys/devices/system/cpu/cpuN/cpu_capacity`, normalized so the fastest core on the machine reads 1024.
That file is published by the kernel's `arch_topology` driver, so it is dependable on `arm64` and absent elsewhere — expect 0 on x86, on Windows, and on Apple.
`compute_cache_bytes_in` reports the deepest cache private to a domain's cores, enumerated exactly rather than measured — from the per-core cache hierarchy in `sysfs` on Linux, CPUID leaves elsewhere on x86, `sysctl` on Apple, and the cache relationships on Windows.
Domains of identical throughput can sit behind very differently sized caches, so neither value follows from the other.
Treat both as hints, present only where the hardware volunteers them, and never as a number a program requires.

The memory axis carries its own magnitudes, all answered by the harvested `Fabric`.
`memory_distance` gives the relative cost of reaching a memory domain from a compute domain — 10 for local, higher for remote, on ACPI SLIT's scale but from measured ratios — while `memory_bandwidth` and `memory_latency` give the observed saturated read bandwidth and dependent-load latency per edge.
Back on the `Topology`, `volume_ram` and `volume_ram_in` report installed RAM, total or per domain, while `huge_pages_count` and `volume_huge_pages` report the free huge pages a domain can still back an allocation with, as a count or in bytes.
Like the compute magnitudes, these are best-effort: fabric queries read 0 before a harvest or on edges no pinned worker could reach.

Finally, `local_memory_of` bridges the axes, naming the memory domain a given compute domain should allocate from.
Where no topology is harvested, every query degrades to a single compute domain and a single memory domain rather than failing, so these loops need no conditional compilation.

### Non-Uniform Memory Access (NUMA)

Placing memory on a chosen domain is implemented on Linux via the `mbind` syscall, on FreeBSD via `domainset` policies, and on Windows via `VirtualAllocExNuma`, and degrades to a single domain elsewhere.
The portable `domain_allocator_t` alias picks the right backend per platform - `linux_numa_allocator_t`, `freebsd_numa_allocator_t`, `windows_numa_allocator_t`, or a plain aligned fallback - each an STL-compatible allocator that reads the machine through the `forkunion::machine_topology` template.

Let's say you are working on a Big Data application, like brute-forcing Vector Search using the [NumKong](https://github.com/ashvardanian/NumKong) library on a 2 dual-socket CPU system, similar to [USearch](https://github.com/unum-cloud/usearch/pulls).
The first part of that program may be responsible for sharding the incoming stream of data between distinct memory regions.
That part, in our simple example will be single-threaded:

```cpp
#include <vector> // `std::vector`
#include <span> // `std::span`
#include <forkunion.hpp> // `linux_numa_allocator`, `machine_topology_t`, `distributed_pool_t`
#include <numkong/spatial.h> // `nk_angular_f32`, `nk_f64_t`

namespace fu = ashvardanian::forkunion;
using floats_alloc_t = fu::linux_numa_allocator<float>;

constexpr std::size_t dimensions = 768; /// Matches most BERT-like models
static std::vector<float, floats_alloc_t> first_half(floats_alloc_t(0));
static std::vector<float, floats_alloc_t> second_half(floats_alloc_t(1));
static fu::machine_topology_t machine_topology;
static fu::distributed_pool_t distributed_pool;

/// Dynamically shards incoming vectors across 2 nodes in a round-robin fashion.
void append(std::span<float, dimensions> vector) {
    bool put_in_second = first_half.size() > second_half.size();
    if (put_in_second) second_half.insert(second_half.end(), vector.begin(), vector.end());
    else first_half.insert(first_half.end(), vector.begin(), vector.end());
}
```

The concurrent part would involve spawning threads adjacent to every memory pool to find the best `search_result_t`.
The primary `search` function, in ideal world would look like this:

1. Each thread finds the best match within its "slice" of a compute domain, tracking the best distance and index in a local CPU register.
2. All threads in each compute domain atomically synchronize using a domain-local instance of `search_result_t`.
3. The main thread collects aggregates of partial results from all compute domains.

That is, however, overly complicated to implement.
Such tree-like hierarchical reductions are optimal in a theoretical sense. Still, assuming the relative cost of spin-locking once at the end of a thread scope and the complexity of organizing the code, the more straightforward path is better.
A minimal example would look like this:

```cpp
/// On each compute domain we'll synchronize the threads
struct search_result_t {
    nk_f64_t best_distance {std::numeric_limits<nk_f64_t>::max()};
    std::size_t best_index {0};
};

inline search_result_t pick_best(search_result_t const& a, search_result_t const& b) noexcept {
    return a.best_distance < b.best_distance ? a : b;
}

/// Uses all CPU threads to search for the closest vector to the @p query.
search_result_t search(std::span<float, dimensions> query) {

    bool const need_to_spawn_threads = distributed_pool.threads_count() == 0;
    if (need_to_spawn_threads) {
        assert(machine_topology.try_harvest() && "Failed to harvest NUMA topology");
        assert(machine_topology.memory_domains_count() == 2 && "Expected exactly 2 NUMA nodes");
        assert(distributed_pool.try_spawn(machine_topology) && "Failed to spawn NUMA pools");
    }

    search_result_t result;
    fu::spin_mutex_t result_update; // ? Lighter `std::mutex` alternative w/out system calls

    std::size_t const total_vectors =
        (first_half.size() + second_half.size()) / dimensions;

    auto slices = distributed_pool.for_slices(total_vectors,
        [&](fu::local_prong<> first, std::size_t count) noexcept {

        bool const in_second = first.compute_domain != 0;
        auto const &shard = in_second ? second_half : first_half;
        std::size_t const shard_base = in_second ? first_half.size() / dimensions : 0;
        std::size_t const local_begin = first.task - shard_base;

        search_result_t thread_local_result;
        for (std::size_t i = 0; i < count; ++i) {
            std::size_t const local_index = local_begin + i;
            std::size_t const global_index = shard_base + local_index;

            nk_f64_t distance;
            nk_angular_f32(query.data(), shard.data() + local_index * dimensions, dimensions, &distance);
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
The `for_slices` helper provides `fu::local_prong` compute-domain metadata that lets you pick the right shard of data based on the compute domain, while keeping scheduling inside the distributed pool.
For more flexibility around building higher-level low-latency systems, there are unsafe APIs expecting you to manually "join" the broadcasted calls: `unsafe_for_threads` returns an always-odd generation token, `is_complete` polls it without blocking, and `unsafe_join` blocks until that generation completes.

The manual two-vector sharding above is what the __symmetric allocators__ automate: `symmetric_memory_allocator_t` - `fu_allocate_symmetric` in C - maps one virtual range striped a slice per memory domain, and Rust and Zig wrap it as `ShardedArray<T>`, one shard per domain, and `ReplicatedArray<T>`, a full copy per domain for read-mostly data.
To place and coordinate pools by hand, `try_spawn_on` - `fu_pool_spawn_on` in C - pins a pool to a single compute domain, while `locate_thread_in` and `threads_count_in` map a global thread index to its domain and count the workers living there.

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
| `risc5_wrs_t`   | RISC-V+Zawrs | `WRS.NTO`   | U          |

No kernel calls.
No futexes.
Works in tight loops.

The `TPAUSE`, `WFET`, and `WRS.NTO` wrappers go a step further than a spin hint: they are _timed_ light-sleep waits, so the pool parks a worker in a low-power state with a per-loop upper bound rather than burning the core.
The waiter is also thread-aware — `micro_yield(thread_id)` lets each worker back off on its own schedule.
These same wrappers back `spin_mutex_t`, or `SpinMutex` in Rust, a syscall-free `std::mutex` alternative that spins on a yield hint instead of trapping into a futex - it is the lock used in the NUMA example above.

### Rayon-style Parallel Iterators

For Rayon-style ergonomics, use the parallel iterator API with the `prelude`.
Unlike Rayon, ForkUnion's parallel iterators don't depend on the global state and allow explicit control over the thread pool and scheduling strategy.
For statically shaped workloads, the default static scheduling is more efficient:

```rust
use forkunion as fu;
use forkunion::prelude::*;

let topology = fu::Topology::new().expect("Failed to detect hardware topology");
let mut pool = fu::spawn(&topology, 4);
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

For parallel reductions, ForkUnion provides Rayon-like convenience methods with automatic NUMA-aware cache-aligned scratch allocation:

```rust
let data: Vec<u64> = (0..1_000_000).map(|i| i as u64).collect();

// Sum all elements with automatic scratch allocation
let total: u64 = (&data[..])
    .into_par_iter()
    .with_pool(&mut pool)
    .sum();

// Count elements matching a predicate
let evens = (&data[..])
    .into_par_iter()
    .filter(|&x| x % 2 == 0)
    .with_pool(&mut pool)
    .count();

// Custom reduction (product)
let product = (&data[..])
    .into_par_iter()
    .with_pool(&mut pool)
    .reduce(
        || 1u64,                        // initial value
        |acc, value, _| *acc *= *value, // fold function
        |a, b| a * b                    // combine function
    );
```

For manual control over scratch allocation, use `reduce_with_scratch`:

```rust
// Cache-line aligned wrapper to prevent false sharing
let mut scratch: Vec<CacheAligned<u64>> =
    (0..pool.threads_count()).map(|_| CacheAligned(0)).collect();

let total = (&data[..])
    .into_par_iter()
    .with_pool(&mut pool)
    .reduce_with_scratch(
        scratch.as_mut_slice(),
        |acc, value, _| acc.0 += *value,  // fold
        |a, b| a.0 += b.0                 // combine in-place
    );
```

Beyond reductions, the iterators offer short-circuiting searches - `find_first` and `find_last` for the deterministic lowest- or highest-index match, `find_any` for the first match with cooperative cancellation, and `any` / `all` for boolean predicates that stop the moment the answer is known.
Fallible bodies get `try_for_each` and `try_fold_with_scratch`, which propagate the first error and signal the other workers to stop.
This Rayon-style layer is __Rust-only__; C, C++, and Zig expose the pool primitives `for_threads`, `for_n`, `for_n_dynamic`, and `for_slices` directly.

## Performance

Two benchmarks measure two different things, each against the same runtimes — __ForkUnion__, [__OpenMP__](https://www.openmp.org), [__Rayon__](https://github.com/rayon-rs/rayon), and [__Taskflow__](https://github.com/taskflow/taskflow) — on deliberately equal footing ¹.
N-body stresses the __dispatch path__: every task costs the same, so what is left over is scheduling latency.
Connected Components by label propagation stresses the __fork-join frequency__: one bandwidth-bound sweep per round until no label changes, so every round re-pays the dispatch-and-join tax.
Implementations live in `scripts/nbody.{cpp,rs,zig}` and `scripts/propagation.{cpp,rs,zig}`, and every binary generates a __bit-identical graph__ from the same counter-based SplitMix64 generator, so cells compare exactly across languages.

### N-Body — Dispatch Latency

Microseconds per iteration — µs ↓, lower is better — at `N=512` bodies on every logical core, as `static / dynamic`, one fixed 30-second window per cell.
This is where fork-join runtimes genuinely differ; ➕ marks C++, 🦀 marks Rust.

| Machine                    |  ➕ ForkUnion |  🦀 ForkUnion |  ➕ OpenMP |   🦀 Rayon | ➕ Taskflow |
| :------------------------- | -----------: | -----------: | --------: | --------: | ---------: |
| 18× Apple M5 Pro, macOS    |  __24 / 26__ |      28 / 32 | 115 / 137 | 222 / 339 |    83 / 94 |
| 24× Intel RPL, Windows     |    148 / 143 | __112 / 91__ | 360 / 627 | 151 / 151 | 812 / 1026 |
| 128× Intel SPR, Linux      |      54 / 86 |  __40 / 67__ | 115 / 226 | 483 / 739 |  666 / 694 |
| 192× AWS Graviton 5, Linux | 47 / __136__ | __42__ / 137 | 266 / 216 | 490 / 580 |  602 / 639 |

### Connected Components — Fork-Join Frequency

Billions of traversed edges per second — GTEPS ↑, higher is better — on a ~27M-edge "necklace" of R-MAT communities, on every logical core, as `static / dynamic`, one fixed 30-second window per cell.
Every pass converges in exactly 84 rounds - 84 fork-join dispatches - bit-identical in every cell and language; ➕ marks C++, 🦀 marks Rust.

| Machine                    |     ➕ ForkUnion |      🦀 ForkUnion |       ➕ OpenMP |     🦀 Rayon | ➕ Taskflow |
| :------------------------- | --------------: | ---------------: | -------------: | ----------: | ---------: |
| 18× Apple M5 Pro, macOS    | __24.7 / 20.5__ |      19.4 / 20.2 |    23.0 / 18.3 | 23.8 / 14.6 | 21.4 / 1.3 |
| 24× Intel RPL, Windows     |  10.0 / __7.2__ |        5.2 / 5.5 | __11.4__ / 0.1 |   9.4 / 5.2 |  6.6 / 0.7 |
| 128× Intel SPR, Linux      |     46.5 / 20.9 |  __65.2 / 31.4__ |     28.2 / 0.4 |   8.5 / 6.5 | 25.1 / 0.5 |
| 192× AWS Graviton 5, Linux |    192.6 / 77.6 | __258.7 / 87.3__ |     53.9 / 0.4 |   8.4 / 6.2 | 53.0 / 0.4 |

What the spread means - all on the 128× SPR, same binaries, same graph:

- __The static column is the fork-join tax, compounded 84 times per pass__: every runtime sweeps the identical bandwidth-bound round, so the entire spread is scheduling - dispatch latency, barrier cost, and thread placement - never arithmetic.
- __The dynamic column is the claim architecture, ~88M one-vertex claims per pass__: ForkUnion claims from private `fetch_add` cursors, Rayon steals halves through its CAS deques, and OpenMP's and Taskflow's shared queues collapse - a 128-thread contention effect for OpenMP, whose queue survives the M5 Pro's 18 threads, while Taskflow's collapse reproduces even there.
- __The two ForkUnion columns are the same pool underneath__: the scheduler, the graph, and the labels are bit-identical, so the gap between them - Rust ahead on SPR, behind on the M5 Pro - is compiler codegen of the compute loops, not the claims.
- __Which of those two columns leads is a toolchain detail__: the compilers vectorize the identical kernel slightly differently - on the 24× i9, LLVM widens the fast-inverse-square-root to 256-bit lanes where MinGW-GCC keeps its integer lanes at 128-bit, so Rust pulls ahead - and Rust can, in rare cases, reach more of a host's instruction set through runtime dispatch.

> ¹ Parity, deliberately enforced: identical `-O3` + `target-cpu=native` on all sides, no LTO anywhere, unchecked hot loops in Rust, identical kernels, one warmup pass, and the same one-vertex dynamic grain in every runtime.
> The finer per-benchmark protocol - scheduling equivalents, page-placement controls, and what each knob defaults to - lives in the `scripts/nbody.*` and `scripts/propagation.*` headers.

You can rerun these benchmarks with the following commands:

```bash
cmake -B build_release -D CMAKE_BUILD_TYPE=Release
cmake --build build_release --config Release
# Rust examples: plain `cargo build` grants neither flag, so parity needs both set explicitly
RUSTFLAGS="-C target-cpu=native" CXXFLAGS="-O3 -march=native" cargo build --release --features benchmarks
# N-body: microseconds per dispatch, sustained over a fixed window (NBODY_SECONDS, default 10 s)
NBODY_COUNT=512 NBODY_BACKEND=forkunion_static_shared build_release/forkunion_nbody
NBODY_COUNT=512 NBODY_BACKEND=taskflow_dynamic build_release/forkunion_nbody
# Connected components: traversal throughput on a necklace graph; PROPAGATION_COMMUNITIES controls the rounds
PROPAGATION_BACKEND=forkunion_static_shared build_release/forkunion_propagation
PROPAGATION_BACKEND=rayon_dynamic target/release/forkunion_propagation
```

## Safety & Logic

There are only 3 core atomic variables in this thread-pool, plus 1 private claim cursor per worker for dynamically-stealing tasks.
Let's call every invocation of a `for_*` API - a "fork", and every exit from it a "join".

| Variable          | Users Perspective             | Internal Usage                        |
| :---------------- | :---------------------------- | :------------------------------------ |
| `stop`            | Stop the entire thread-pool   | Tells workers when to exit the loop   |
| `fork_generation` | "Forks" called since init     | Tells workers to wake up on new forks |
| `threads_to_sync` | Threads not joined this fork  | Tells main thread when workers finish |
| `claim.next`      | One worker's stealable cursor | Next task in that worker's slice      |

The next task for `for_n_dynamic` calls is drained by neighbors from `claim.next` in coprime order once their own slice runs dry, overshooting by at most one.

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

Toolchain floors, never caps.
The header needs __C++17__, and a C++20+ consumer keeps its own standard — gaining concepts, the `atomic_ref` waiter, and `std::popcount` — while the pre-compiled libraries build at C++20 regardless.
The C ABI needs __C99__, and the build tooling __CMake 3.21__, __Rust 1.84__, and __Zig 0.16__.

To run the C++ tests, use CMake:

```bash
cmake -B build_release -D CMAKE_BUILD_TYPE=Release -D BUILD_TESTING=ON
cmake --build build_release --config Release -j
ctest --test-dir build_release                  # run all tests
build_release/forkunion_nbody                  # run the benchmarks
```

For C++ debug builds, consider using the VS Code debugger presets or the following commands:

```bash
cmake -B build_debug -D CMAKE_BUILD_TYPE=Debug -D BUILD_TESTING=ON
cmake --build build_debug --config Debug        # build with Debug symbols
build_debug/forkunion_test_cpp20               # run a single test executable
```

To run static analysis:

```bash
sudo apt install cppcheck clang-tidy
cmake --build build_debug --target cppcheck     # detects bugs & undefined behavior
cmake --build build_debug --target clang-tidy   # suggest code improvements
```

No packages are needed for NUMA or Huge Pages, on any distribution.
The topology is read from `/sys/devices/system/node`, pages are placed with the `mbind` syscall, and huge pages are requested with `mmap(MAP_HUGETLB)` — all kernel interfaces, so a build picks up whatever the machine it runs on offers rather than whatever the machine it was compiled on had installed.
Neither `libnuma` nor `libhugetlbfs` is linked; `ld.hugetlbfs` remaps a program's own text segment, which is a different feature entirely.

Huge pages do have to be _reserved_ before they can be placed, which is a runtime knob rather than a build dependency:

```bash
cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages          # what exists now
echo 1024 | sudo tee /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages
```

To build with an alternative compiler, like LLVM Clang, use the following command:

```bash
sudo apt-get install libomp-15-dev clang++-15 # OpenMP version must match Clang
cmake -B build_debug -D CMAKE_BUILD_TYPE=Debug -D CMAKE_CXX_COMPILER=clang++-15
cmake --build build_debug --config Debug
build_debug/forkunion_test_cpp20
```

Or on macOS with Apple Clang:

```bash
brew install llvm@20
cmake -B build_debug -D CMAKE_BUILD_TYPE=Debug -D CMAKE_CXX_COMPILER=$(brew --prefix llvm@20)/bin/clang++
cmake --build build_debug --config Debug
build_debug/forkunion_test_cpp20
```

For Rust, use the following commands to verify no_std compatibility:

```bash
rustup toolchain install
cargo build --lib --no-default-features --release
cargo build --lib --no-default-features --features place-memory-on-domain --release
cargo doc --lib --no-default-features --no-deps
```

Verify the tests pass:

```bash
cargo test --lib --release
cargo test --doc --release
cargo test --lib --features place-memory-on-domain --release
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
zig build test --summary all               # run tests
zig build -Dplace-memory-on-domain=true    # require NUMA-aware allocations (Linux)
zig build -Dportable=true                  # STL thread pool only

# Run benchmarks from the `scripts` directory
cd scripts
zig build -Doptimize=ReleaseFast
NBODY_COUNT=512 NBODY_BACKEND=forkunion_static_shared ./zig-out/bin/forkunion_nbody
PROPAGATION_BACKEND=forkunion_static_shared ./zig-out/bin/forkunion_propagation
```

Check the `scripts/nbody.zig` and `scripts/propagation.zig` headers for additional benchmarking options.

## Citation

If ForkUnion helps your research or product, please cite it:

```bibtex
@software{Vardanian_ForkUnion,
  author = {Vardanian, Ash},
  title = {{ForkUnion: Low-latency NUMA-aware fork-join thread-pool with zero allocations, syscalls, CAS, or false-sharing on the hot path for C, C++, Rust, and Zig}},
  doi = {10.5281/zenodo.21472185},
  url = {https://github.com/ashvardanian/ForkUnion},
  license = {Apache-2.0}
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is provided at the repository root.

## License

Licensed under the Apache License, Version 2.0. See `LICENSE` for details.
