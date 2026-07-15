/**
 *  @brief  Low-latency OpenMP-style NUMA-aware cross-platform fine-grained parallelism library.
 *  @file   forkunion.hpp
 *  @author Ash Vardanian
 *  @date   May 2, 2025
 *
 *  ForkUnion provides a minimalistic cross-platform thread-pool implementation and Parallel Algorithms,
 *  avoiding dynamic memory allocations, exceptions, system calls, and heavy Compare-And-Swap instructions.
 *  The library leverages the "weak memory model" to allow Arm and IBM Power CPUs to aggressively optimize
 *  execution at runtime. It also aggressively tests against overflows on smaller index types, and is safe
 *  to use even with the maximal `std::size_t` values.
 *
 *  @code{.cpp}
 *  #include <cstdio> // `std::printf`
 *  #include <cstdlib> // `EXIT_FAILURE`, `EXIT_SUCCESS`
 *  #include <forkunion.hpp> // `fu::flat_pool_t`
 *
 *  using fu = ashvardanian::forkunion;
 *  int main(int argc, char *argv[]) {
 *
 *      fu::flat_pool_t pool;
 *      if (!pool.try_spawn(fu::allowed_cores_count()))
 *          return EXIT_FAILURE;
 *
 *      pool.for_n(argc, [=](auto prong) noexcept {
 *          auto [task_index, thread_index, compute_domain_index] = prong;
 *          std::printf(
 *              "Printing argument # %zu (of %zu) from thread # %zu at compute_domain # %zu: %s\n",
 *              task_index, argc, thread_index, compute_domain_index, argv[task_index]);
 *      });
 *      return EXIT_SUCCESS;
 *  }
 *  @endcode
 *
 *  The next layer of logic is for basic index-addressable tasks. It includes basic parallel loops:
 *
 *  - `for_n` - for iterating over a range of similar duration tasks, addressable by an index.
 *  - `for_n_dynamic` - for unevenly distributed tasks, where each task may take a different time.
 *  - `for_slices` - for iterating over a range of similar duration tasks, addressable by a slice.
 *
 *  On Linux, when NUMA and PThreads are available, the library can also leverage @b NUMA-aware
 *  memory allocations and pin threads to specific physical cores to increase memory locality.
 *  It should reduce memory access latency by around 35% on average, compared to remote accesses.
 *  @sa `machine_topology_t`, `colocated_pool_t`, `distributed_pool_t`.
 *
 *  On heterogeneous chips, cores with a different @b "Quality-of-Service" (QoS) may be combined.
 *  A typical example is laptop/desktop chips, having 1 NUMA node, but 3 tiers of CPU cores:
 *  performance, efficiency, and power-saving cores. Each group will have vastly different speed,
 *  so considering them equal in tasks scheduling is a bad idea... and separating them automatically
 *  isn't feasible either. It's up to the user to isolate those groups into individual pools.
 *  @sa `core_quality_t`
 *
 *  On x86, Arm, and RISC-V (internally referred to as RISC5) architectures, depending on the CPU
 *  features available, the library also exposes cheaper @b "busy-waiting" mechanisms, such as
 *  `tpause`, `wfet`, & `yield` instructions.
 *  @sa `arm64_yield_t`, `arm64_wfet_t`, `x86_pause_t`, `x86_tpause_t`, `risc5_pause_t`.
 *
 *  The library uses modern C++ features and requires @b C++17 or newer.
 *  Using C++20 will enable additional compile-time checks (concepts) where available.
 */
#pragma once

/*  Kept in the umbrella, not a sub-header: this is where a reader looks for the version,
    and where the release workflows rewrite it.  */
#define FORKUNION_VERSION_MAJOR 3
#define FORKUNION_VERSION_MINOR 0
#define FORKUNION_VERSION_PATCH 0

/*
 *  One header per concern. Include order is dependency order; none is meant to be included alone.
 */
#include "forkunion/types.hpp"        // Vocabulary: prongs, buffers, index splitting, claim cursors
#include "forkunion/capabilities.hpp" // Yields, `cpu_capabilities`, `ram_capabilities`
#include "forkunion/topology.hpp"     // `memory_domain`, `compute_domain_t`, `machine_topology`, the harvests
#include "forkunion/allocators.hpp"   // NUMA allocators, `replicated_array`, `sharded_array`
#include "forkunion/flat.hpp"         // `flat_pool`, the portable STL thread-pool
#include "forkunion/distributed.hpp"  // `colocated_pool`, `distributed_pool`, NUMA memory
#include "forkunion/logging.hpp"      // Human-readable topology dumps
