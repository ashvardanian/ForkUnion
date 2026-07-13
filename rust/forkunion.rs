//! Low-latency OpenMP-style NUMA-aware cross-platform fine-grained parallelism library.
//!
//! ForkUnion provides a minimalistic cross-platform thread-pool implementation and Parallel Algorithms,
//! avoiding dynamic memory allocations, exceptions, system calls, and heavy Compare-And-Swap instructions.
//! The library leverages the "weak memory model" to allow Arm and IBM Power CPUs to aggressively optimize
//! execution at runtime. It also aggressively tests against overflows on smaller index types, and is safe
//! to use even with the maximal `usize` values.
//!
//! This Rust wrapper provides a safe interface around the precompiled C library, maintaining zero-allocation
//! principles while leveraging NUMA-aware optimizations and CPU-specific busy-waiting instructions.
//!
//! The wrapper mirrors the C++ core's header layout: [`topology`] for the hardware view, [`types`] for
//! portable building blocks, [`allocators`] for NUMA-aware allocation, [`scheduling`] for the thread pool
//! and its dispatch primitives, and [`parallel`] for the Rayon-style parallel iterators. Every public
//! symbol is re-exported here, so `use forkunion::*` resolves exactly as it did before the split.

#![no_std]

#[cfg(feature = "std")]
extern crate std;

#[path = "forkunion/allocators.rs"]
mod allocators;
#[path = "forkunion/parallel.rs"]
mod parallel;
#[path = "forkunion/scheduling.rs"]
mod scheduling;
#[path = "forkunion/topology.rs"]
mod topology;
#[path = "forkunion/types.rs"]
mod types;

pub use allocators::*;
pub use parallel::*;
pub use scheduling::*;
pub use topology::*;
pub use types::*;
