"""Low-latency OpenMP-style NUMA-aware cross-platform fine-grained parallelism library.

ForkUnion is a minimalistic thread pool for fork-join parallelism, avoiding dynamic memory
allocations, exceptions, system calls, and heavy compare-and-swap instructions on the hot path.
Unlike `max.algorithm.parallelize`, which takes every hardware thread, the pool's width is an
argument, which is what makes it usable on a machine you are sharing.

The library is reached by `dlopen`, so a consumer builds with no linker flags at all:

```mojo
from forkunion import Library, Pool, Prong, Topology

@fieldwise_init
struct Scratch(ImplicitlyCopyable, TrivialRegisterPassable):
    var out: Pointer[Int64, MutUntrackedOrigin]

def square(prong: Prong, mut scratch: Scratch):
    scratch.out[unsafe_offset=prong.task_index] = Int64(prong.task_index * prong.task_index)

def main() raises:
    var library = Library()
    var topology = Topology(library)
    var pool = Pool(topology, threads=4)
    var values = List[Int64](length=64, fill=0)
    var scratch = Scratch(values.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]())
    pool.for_n[square](64, scratch)
```

The callback is a parameter rather than a value, because a closure that captured anything could
not be handed to C; the state it needs travels in the scratch instead. The scratch's type and
origin are inferred from the argument, so a call site names only the work.

Every fallible entry point raises `ForkUnionError` and nothing else, because Mojo allows one error
type per function and never widens a typed `raises`. Where the C API reports absence rather than
failure - a domain this machine lacks, a page the kernel refused - the answer is `Optional`.

This package root re-exports every public symbol from the modules that mirror the C++ core, so
`from forkunion import X` resolves whatever module X lives in.
"""

# errors
from .errors import ErrorKind, ForkUnionError

# library
from .library import Library, Symbols

# types
from .types import (
    DEFAULT_ALIGNMENT,
    CacheAligned,
    CallerExclusivity,
    Capabilities,
    ComputeDomain,
    IndexedRange,
    IndexedSplit,
    MemoryDomain,
    MemoryDomainId,
    Prong,
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
