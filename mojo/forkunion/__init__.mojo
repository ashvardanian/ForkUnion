"""Low-latency OpenMP-style NUMA-aware cross-platform fine-grained parallelism library.

ForkUnion is a minimalistic thread pool for fork-join parallelism, avoiding dynamic memory
allocations, exceptions, system calls, and heavy compare-and-swap instructions on the hot path.
Unlike `max.algorithm.parallelize`, which takes every hardware thread, the pool's width is an
argument, which is what makes it usable on a machine you are sharing.

The library is reached by `dlopen`, so a consumer builds with no linker flags at all:

```mojo
from forkunion import Library, Pool, SyncMutPointer, ThreadInDomain, Topology

@fieldwise_init
struct Squares(ImplicitlyCopyable, TrivialRegisterPassable): # the output array
    var values: SyncMutPointer[Int64]

def square(task: Int, at: ThreadInDomain, mut squares: Squares):
    squares.values.at(task) = Int64(task * task)

def main() raises:
    var library = Library()
    var topology = Topology(library)
    var pool = Pool(topology, threads=4)
    var values = List[Int64](length=64, fill=0)
    var squares = Squares(SyncMutPointer(values.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()))
    pool.for_n[square](64, squares)
    print(t"values[7] = {values[7]}") # 49
```

The callback is a parameter rather than a value, because a closure that captured anything could
not be handed to C; the state it needs travels in the scratch instead. The scratch's type and
origin are inferred from the argument, so a call site names only the work.

Every fallible entry point raises `Error` and nothing else, because Mojo allows one error
type per function and never widens a typed `raises`. A function raises if and only if it can
fail, so a refusal is never folded into an `Optional` a caller has to interpret.

This package root re-exports every public symbol from the modules that mirror the C++ core, so
`from forkunion import X` resolves whatever module X lives in.
"""

# library
from .library import Library, Symbols

# types
from .types import (
    DEFAULT_ALIGNMENT,
    bytes_for_elements,
    CacheAligned,
    CallerExclusivity,
    Capabilities,
    ComputeDomain,
    Error,
    ErrorKind,
    TasksRange,
    IndexedSplit,
    MemoryDomain,
    MemoryDomainId,
    ThreadInDomain,
    SyncConstPointer,
    SyncMutPointer,
)

# topology
from .topology import (
    Topology,
    comptime_capabilities,
    comptime_capabilities_string,
    name_capabilities,
    runtime_capabilities,
    runtime_capabilities_string,
    version,
)

# scheduling
from .scheduling import BroadcastJoin, Fabric, Pool

# allocators
from .allocators import (
    AllocationResult,
    DomainAllocator,
    ReplicatedArray,
    ShardLocation,
    ShardedArray,
    SymmetricAllocation,
    default_domain_allocator,
    local_domain_allocator,
)
