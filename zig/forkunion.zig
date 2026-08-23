//! Low-latency OpenMP-style NUMA-aware cross-platform fine-grained parallelism library.
//!
//! ForkUnion provides a minimalistic cross-platform thread-pool implementation for fork-join
//! parallelism, avoiding dynamic memory allocations, exceptions, system calls, and heavy
//! Compare-And-Swap instructions on the hot path.
//!
//! Zig 0.16 removed std.Thread.Pool, and its replacement std.Io.Group allocates per task and has
//! no notion of topology. ForkUnion is designed for data parallelism and tight parallel loops -
//! think OpenMP's `#pragma omp parallel for` - with a fixed pool and no allocation on the hot path.
//!
//! Basic usage:
//! ```zig
//! const fu = @import("forkunion");
//!
//! const topo = try fu.Topology.init();
//! defer topo.deinit();
//!
//! const pool = try fu.Pool.init(topo, .{ .threads = 4 });
//! defer pool.deinit();
//!
//! // Execute work on each thread (like OpenMP parallel)
//! pool.forThreads({}, struct {
//!     fn work(thread_index: usize, compute_domain: fu.ComputeDomain) void {
//!         std.debug.print("Thread {}\n", .{thread_index});
//!     }
//! }.work);
//!
//! // Distribute 1000 tasks across threads (like OpenMP parallel for)
//! var results = [_]i32{0} ** 1000;
//! pool.forN(1000, &results, processTask);
//! ```
//!
//! Every dispatch takes the context before the callback and forwards it as a caller-owned pointer,
//! so the callback sees the qualifiers the caller chose. Pass `{}` for a kernel that needs none.
//!
//! This root re-exports every public symbol from the modules that mirror the C++ core -
//! `topology`, `types`, `allocators`, `scheduling` - so `@import("forkunion").X` resolves exactly
//! as when the binding lived in one file. Each module carries its own FFI declarations at the top
//! and its tests at the bottom.

const topology = @import("forkunion/topology.zig");
const types = @import("forkunion/types.zig");
const allocators = @import("forkunion/allocators.zig");
const scheduling = @import("forkunion/scheduling.zig");

// topology
pub const CallerExclusivity = topology.CallerExclusivity;
pub const Capabilities = topology.Capabilities;
pub const Topology = topology.Topology;
pub const version = topology.version;
pub const comptimeCapabilities = topology.comptimeCapabilities;
pub const comptimeCapabilitiesString = topology.comptimeCapabilitiesString;
pub const runtimeCapabilities = topology.runtimeCapabilities;
pub const runtimeCapabilitiesString = topology.runtimeCapabilitiesString;

// types
pub const Status = types.Status;
pub const Error = types.Error;
pub const ComputeDomain = types.ComputeDomain;
pub const MemoryDomain = types.MemoryDomain;
pub const MemoryDomainId = types.MemoryDomainId;
pub const Prong = types.Prong;
pub const default_alignment = types.default_alignment;
pub const IndexedSplit = types.IndexedSplit;
pub const IndexedRange = types.IndexedRange;
pub const CacheAligned = types.CacheAligned;
pub const SyncConstPtr = types.SyncConstPtr;
pub const SyncMutPtr = types.SyncMutPtr;

// allocators
pub const AllocationResult = allocators.AllocationResult;
pub const allocate = allocators.allocate;
pub const allocateAtLeast = allocators.allocateAtLeast;
pub const DomainAllocator = allocators.DomainAllocator;
pub const ReplicatedArray = allocators.ReplicatedArray;
pub const ShardedArray = allocators.ShardedArray;

// scheduling
pub const Pool = scheduling.Pool;
pub const PoolOptions = scheduling.PoolOptions;
pub const Fabric = scheduling.Fabric;

test { // pull each module's tests into `zig build test`
    _ = @import("forkunion/topology.zig");
    _ = @import("forkunion/types.zig");
    _ = @import("forkunion/allocators.zig");
    _ = @import("forkunion/scheduling.zig");
}
