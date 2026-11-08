//! Low-latency OpenMP-style NUMA-aware cross-platform fine-grained parallelism library.
//!
//! ForkUnion provides a minimalistic cross-platform thread-pool implementation and Parallel
//! Algorithms, avoiding dynamic memory allocations, exceptions, system calls, and Compare-And-Swap
//! loops on the hot path.
//! The library leverages the "weak memory model" to allow Arm and IBM Power CPUs to aggressively
//! optimize execution at runtime. It also aggressively tests against overflows on smaller index
//! types, and is safe to use even with the maximal `usize` values.
//!
//! This Rust wrapper safely wraps the precompiled C library, keeping zero-allocation principles
//! while leveraging NUMA-aware optimizations and CPU-specific busy-waiting instructions.
//!
//! The wrapper mirrors the C++ core's header layout: `topology` for the hardware view, `types`
//! for portable building blocks, `allocators` for NUMA-aware allocation, `scheduling` for the
//! thread pool, its dispatch primitives, and the measured memory fabric, and `parallel` for the
//! Rayon-style parallel iterators. Every public symbol is re-exported here, so `use forkunion::*`
//! resolves exactly as it did before the split.
//!
//! File: rust/forkunion.rs
//! Author: Ash Vardanian
//! Date: June 29, 2025

#![no_std]
#![warn(missing_docs)]

extern crate alloc;
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
