"""The thread pool, its dispatch primitives, and the measured memory fabric.

The callback is a parameter rather than a value, which is what lets the trampoline below become a
plain C function pointer: it names only parameters, and parameters monomorphise. A `@parameter`
closure that captured anything would make the trampoline `capturing`, which cannot be handed to C
at all, so the state a callback needs travels in an explicit scratch instead.

The scratch travels by reference. A blocking dispatch does not return until every task is done,
so the reference is alive for the whole call; taking it here rather than asking the caller for a
pointer is what stops a caller from handing over storage that dies first.
"""

from std.ffi import c_int, c_size_t

from std.memory import stack_allocation

from forkunion.library import Library
from forkunion.topology import Topology
from forkunion.types import (
    CallerExclusivity,
    Capabilities,
    ComputeDomain,
    Context,
    CString,
    Error,
    ErrorKind,
    Handle,
    MemoryDomain,
    OutHandle,
    OutSize,
    TasksRange,
    ThreadInDomain,
)


# region Signatures

comptime PoolNew = def(CString, c_int, OutHandle) thin abi("C") -> c_int
comptime PoolDelete = def(Handle) thin abi("C") -> None
comptime PoolCapabilities = def(Handle, Pointer[Int32, MutAnyOrigin]) thin abi("C") -> c_int
comptime PoolSpawn = def(Handle, Handle, c_size_t, c_int) thin abi("C") -> c_int
comptime PoolSpawnOn = def(Handle, Handle, c_size_t, c_size_t, c_int) thin abi("C") -> c_int
comptime PoolExclusivity = def(Handle, Pointer[Int32, MutAnyOrigin]) thin abi("C") -> c_int
comptime PoolCount = def(Handle, OutSize) thin abi("C") -> c_int
comptime PoolCountIn = def(Handle, c_size_t, OutSize) thin abi("C") -> c_int
comptime PoolLocateThreadIn = def(Handle, c_size_t, c_size_t, OutSize) thin abi("C") -> c_int
comptime PoolSleep = def(Handle, c_size_t) thin abi("C") -> None
comptime PoolTerminate = def(Handle) thin abi("C") -> None
comptime FabricNew = def(OutHandle) thin abi("C") -> c_int
comptime FabricDelete = def(Handle) thin abi("C") -> None
comptime FabricHarvest = def(Handle, Handle, Handle) thin abi("C") -> c_int
comptime FabricEdge = def(Handle, c_size_t, c_size_t, OutSize) thin abi("C") -> c_int
comptime FabricLevelIn = def(Handle, c_size_t, OutSize) thin abi("C") -> c_int
comptime FabricLevelsCount = def(Handle, OutSize) thin abi("C") -> c_int
comptime ForThreads = def(Context, c_size_t, c_size_t) thin abi("C") -> None
comptime ForProngs = def(Context, c_size_t, c_size_t, c_size_t) thin abi("C") -> None
comptime ForSlices = def(Context, c_size_t, c_size_t, c_size_t, c_size_t) thin abi("C") -> None
comptime PoolForThreads = def(Handle, ForThreads, Context) thin abi("C") -> c_int
comptime PoolForN = def(Handle, c_size_t, ForProngs, Context) thin abi("C") -> c_int
comptime PoolForSlices = def(Handle, c_size_t, ForSlices, Context) thin abi("C") -> c_int
comptime PoolUnsafeForThreads = def(Handle, ForThreads, Context, OutSize) thin abi("C") -> c_int
comptime PoolIsComplete = def(Handle, c_size_t, Pointer[Int32, MutAnyOrigin]) thin abi("C") -> c_int
comptime PoolUnsafeJoin = def(Handle, c_size_t) thin abi("C") -> None

# endregion Signatures


@always_inline
def _erase[Scratch: AnyType, //](mut scratch: Scratch) -> Context:
    """The scratch as the type-punned context the C API carries through a dispatch."""
    return Pointer(to=scratch).unsafe_bitcast[NoneType]().unsafe_origin_cast[MutUntrackedOrigin]()


struct Pool:
    """A spawned pool. Released when the last reference to it goes out of use.

    Every query re-asks the C core rather than caching, so the answers stay correct across a
    `terminate` followed by a fresh spawn with a different width or exclusivity.
    """

    var library: Library
    var handle: Handle

    # region Lifetime

    def __init__(
        out self,
        topology: Topology,
        threads: Int,
        exclusivity: CallerExclusivity = CallerExclusivity.INCLUSIVE,
        name: StaticString = "",
        allowed: Capabilities = Capabilities.ALL,
    ) raises Error:
        """Spawns `threads` workers across the whole machine."""
        self.library = topology.library
        self.handle = Self._create(self.library, name, allowed)
        var spawned = self.library.symbols().pool_spawn(
            topology.handle, self.handle, c_size_t(threads), exclusivity.identifier
        )
        Self._check_spawn(spawned, "fu_pool_spawn")

    @staticmethod
    def on(
        topology: Topology,
        domain: ComputeDomain,
        threads: Int,
        exclusivity: CallerExclusivity = CallerExclusivity.INCLUSIVE,
        name: StaticString = "",
        allowed: Capabilities = Capabilities.ALL,
    ) raises Error -> Self:
        """Spawns `threads` workers pinned to one compute domain."""
        var library = topology.library
        var handle = Self._create(library, name, allowed)
        var spawned = library.symbols().pool_spawn_on(
            topology.handle,
            handle,
            c_size_t(domain.index),
            c_size_t(threads),
            exclusivity.identifier,
        )
        Self._check_spawn(spawned, "fu_pool_spawn_on")
        return Self(library=library, handle=handle)

    def __init__(out self, *, library: Library, handle: Handle):
        self.library = library
        self.handle = handle

    @staticmethod
    def _create(library: Library, name: StaticString, allowed: Capabilities) raises Error -> Handle:
        """Allocates the pool object; placement is decided later, at the spawn."""
        # Stack storage the C side writes through; see `OutSize` for why the origin is `Any`.
        var out = stack_allocation[1, Int]()
        out[unsafe_offset=0] = 0
        var status = library.symbols().pool_new(
            name.unsafe_ptr().unsafe_bitcast[Int8]().unsafe_origin_cast[ImmUntrackedOrigin](),
            c_int(Int(allowed.bits)),
            out.unsafe_origin_cast[MutAnyOrigin](),
        )
        if status != 0:
            raise Error(ErrorKind.of(status), "fu_pool_new")
        return Handle(unsafe_from_address=out[unsafe_offset=0])

    @staticmethod
    def _check_spawn(spawned: c_int, symbol: StaticString) raises Error:
        """The core now names its own reason, so nothing is guessed here."""
        if spawned != 0:
            raise Error(ErrorKind.of(spawned), symbol)

    def __deinit__(deinit self):
        self.library.symbols().pool_terminate(self.handle)
        self.library.symbols().pool_delete(self.handle)

    # endregion Lifetime

    # region Queries

    @always_inline
    def _count(self, symbol: def(Handle, OutSize) thin abi("C") -> c_int, detail: StaticString) raises Error -> Int:
        """Runs a `(handle) -> status` query; an unspawned pool is a reported failure."""
        var out = stack_allocation[1, c_size_t]()
        out[unsafe_offset=0] = c_size_t(0)
        var status = symbol(self.handle, out.unsafe_origin_cast[MutAnyOrigin]())
        if status != 0:
            raise Error(ErrorKind.of(status), detail)
        return Int(out[unsafe_offset=0])

    def threads_count(self) raises Error -> Int:
        """Workers in the pool, including the caller on an inclusive pool."""
        return self._count(self.library.symbols().pool_threads_count, "fu_pool_threads_count")

    def threads_count_in(self, domain: ComputeDomain) raises Error -> Int:
        """Workers the pool placed in one compute domain."""
        var out = stack_allocation[1, c_size_t]()
        out[unsafe_offset=0] = c_size_t(0)
        var status = self.library.symbols().pool_threads_count_in(
            self.handle, c_size_t(domain.index), out.unsafe_origin_cast[MutAnyOrigin]()
        )
        if status != 0:
            raise Error(ErrorKind.of(status), "fu_pool_threads_count_in")
        return Int(out[unsafe_offset=0])

    def compute_domains_count(self) raises Error -> Int:
        """Compute domains this pool spans, which may be fewer than the machine's."""
        return self._count(self.library.symbols().pool_compute_domains_count, "fu_pool_compute_domains_count")

    def locate_thread_in(self, global_thread_index: Int, domain: ComputeDomain) raises Error -> Int:
        """A global thread index expressed as a local one inside a compute domain."""
        # Stack storage the C side writes through; see `OutSize` for why the origin is `Any`.
        var out = stack_allocation[1, c_size_t]()
        out[unsafe_offset=0] = c_size_t(0)
        var status = self.library.symbols().pool_locate_thread_in(
            self.handle,
            c_size_t(global_thread_index),
            c_size_t(domain.index),
            out.unsafe_origin_cast[MutAnyOrigin](),
        )
        if status != 0:
            raise Error(ErrorKind.of(status), "fu_pool_locate_thread_in")
        return Int(out[unsafe_offset=0])

    def caller_exclusivity(self) raises Error -> CallerExclusivity:
        """Queried live, so it stays correct across a terminate and a re-spawn."""
        # Stack storage the C side writes through; see `OutSize` for why the origin is `Any`.
        var out = stack_allocation[1, Int32]()
        out[unsafe_offset=0] = Int32(0)
        var status = self.library.symbols().pool_caller_exclusivity(self.handle, out.unsafe_origin_cast[MutAnyOrigin]())
        if status != 0:
            raise Error(ErrorKind.of(status), "fu_pool_caller_exclusivity")
        return CallerExclusivity(c_int(Int(out[unsafe_offset=0])))

    def capabilities(self) raises Error -> Capabilities:
        """The requested allow-mask narrowed by what the machine offers; masking never adds."""
        # Stack storage the C side writes through; see `OutSize` for why the origin is `Any`.
        var out = stack_allocation[1, Int32]()
        out[unsafe_offset=0] = Int32(0)
        var status = self.library.symbols().pool_capabilities(self.handle, out.unsafe_origin_cast[MutAnyOrigin]())
        if status != 0:
            raise Error(ErrorKind.of(status), "fu_pool_capabilities")
        return Capabilities(UInt32(Int(out[unsafe_offset=0])))

    def sleep(self, microseconds: Int):
        """Parks the workers; the next dispatch wakes them."""
        self.library.symbols().pool_sleep(self.handle, c_size_t(microseconds))

    def terminate(self):
        """Stops the workers but keeps the handle, so the pool can be spawned again."""
        self.library.symbols().pool_terminate(self.handle)

    # endregion Queries

    # region Blocking Dispatch

    def for_threads[
        Scratch: AnyType,
        scratch_origin: MutOrigin,
        //,
        work: def(Int, Int, mut Scratch) thin -> None,
    ](self, ref[scratch_origin] scratch: Scratch) raises Error:
        """One call per worker, which is where per-thread scratch is set up or torn down.

        The callback takes the thread and compute-domain indices; the C API supplies no task index
        here, so none is invented.
        """

        def trampoline(carried: Context, thread: c_size_t, domain: c_size_t) abi("C"):
            work(Int(thread), Int(domain), carried.unsafe_bitcast[Scratch]()[])

        var status = self.library.symbols().pool_for_threads(self.handle, trampoline, _erase(scratch))
        if status != 0:
            raise Error(ErrorKind.of(status), "fu_pool_for_threads")

    def for_n[
        Scratch: AnyType,
        scratch_origin: MutOrigin,
        //,
        work: def(Int, ThreadInDomain, mut Scratch) thin -> None,
    ](self, n: Int, ref[scratch_origin] scratch: Scratch) raises Error:
        """Splits `n` tasks into equal contiguous chunks and blocks until every one has finished."""

        def trampoline(carried: Context, task: c_size_t, thread: c_size_t, domain: c_size_t) abi("C"):
            work(Int(task), ThreadInDomain(Int(thread), Int(domain)), carried.unsafe_bitcast[Scratch]()[])

        var status = self.library.symbols().pool_for_n(self.handle, c_size_t(n), trampoline, _erase(scratch))
        if status != 0:
            raise Error(ErrorKind.of(status), "fu_pool_for_n")

    def for_n_dynamic[
        Scratch: AnyType,
        scratch_origin: MutOrigin,
        //,
        work: def(Int, ThreadInDomain, mut Scratch) thin -> None,
    ](self, n: Int, ref[scratch_origin] scratch: Scratch) raises Error:
        """The same, but tasks are claimed as threads free up, for work of uneven cost."""

        def trampoline(carried: Context, task: c_size_t, thread: c_size_t, domain: c_size_t) abi("C"):
            work(Int(task), ThreadInDomain(Int(thread), Int(domain)), carried.unsafe_bitcast[Scratch]()[])

        var status = self.library.symbols().pool_for_n_dynamic(self.handle, c_size_t(n), trampoline, _erase(scratch))
        if status != 0:
            raise Error(ErrorKind.of(status), "fu_pool_for_n_dynamic")

    def for_slices[
        Scratch: AnyType,
        scratch_origin: MutOrigin,
        //,
        work: def(TasksRange, ThreadInDomain, mut Scratch) thin -> None,
    ](self, n: Int, ref[scratch_origin] scratch: Scratch) raises Error:
        """One contiguous run per worker, for vectorized or per-slice-setup work.

        Every worker is called exactly once; an idle one receives a range with `count == 0`.
        """

        def trampoline(carried: Context, first: c_size_t, count: c_size_t, thread: c_size_t, domain: c_size_t) abi("C"):
            work(
                TasksRange(Int(first), Int(count)),
                ThreadInDomain(Int(thread), Int(domain)),
                carried.unsafe_bitcast[Scratch]()[],
            )

        var status = self.library.symbols().pool_for_slices(self.handle, c_size_t(n), trampoline, _erase(scratch))
        if status != 0:
            raise Error(ErrorKind.of(status), "fu_pool_for_slices")

    # endregion Blocking Dispatch

    # region Non-Blocking Dispatch

    def unsafe_for_threads[
        Scratch: AnyType,
        pool_origin: ImmOrigin,
        scratch_origin: MutOrigin,
        //,
        work: def(Int, Int, mut Scratch) thin -> None,
    ](ref[pool_origin] self, ref[scratch_origin] scratch: Scratch) raises Error -> BroadcastJoin[
        pool_origin, scratch_origin
    ]:
        """Dispatches on every worker without blocking, returning a guard that joins on exit.

        Use it as a context manager. On an exclusive pool the work starts here and the generation
        can be polled; on an inclusive pool the caller owes a slice, so the work runs at the join.
        """

        def trampoline(carried: Context, thread: c_size_t, domain: c_size_t) abi("C"):
            work(Int(thread), Int(domain), carried.unsafe_bitcast[Scratch]()[])

        # Stack storage the C side writes through; see `OutSize` for why the origin is `Any`.
        var out = stack_allocation[1, c_size_t]()
        out[unsafe_offset=0] = c_size_t(0)
        var status = self.library.symbols().pool_unsafe_for_threads(
            self.handle, trampoline, _erase(scratch), out.unsafe_origin_cast[MutAnyOrigin]()
        )
        if status != 0:
            raise Error(ErrorKind.of(status), "fu_pool_unsafe_for_threads")
        return BroadcastJoin[pool_origin, scratch_origin](Pointer(to=self), Int(out[unsafe_offset=0]))

    # endregion Non-Blocking Dispatch


@fieldwise_init
struct BroadcastJoin[pool_origin: ImmOrigin, scratch_origin: MutOrigin](ImplicitlyCopyable):
    """One in-flight generation, joined when the `with` block ends.

    Both origins are load-bearing. Mojo destroys a value after its last named use, so without the
    pool borrow the pool would be torn down while its workers were still running - and a
    `terminate` mid-flight trips an assertion inside the core. The scratch borrow keeps the
    storage the workers are writing through alive until the join, which a raw context pointer
    cannot express.
    """

    var pool: Pointer[Pool, Self.pool_origin]
    var generation: Int

    def __enter__(self) -> Self:
        """Hands back the guard itself, so the block can poll as well as read the generation."""
        return self

    def __exit__(mut self):
        self.pool[].library.symbols().pool_unsafe_join(self.pool[].handle, c_size_t(self.generation))

    def is_complete(self) raises Error -> Bool:
        """A non-blocking poll; meaningful on exclusive pools, where the caller owes no slice."""
        # Stack storage the C side writes through; see `OutSize` for why the origin is `Any`.
        var out = stack_allocation[1, Int32]()
        out[unsafe_offset=0] = Int32(0)
        var status = (
            self.pool[]
            .library.symbols()
            .pool_is_complete(self.pool[].handle, c_size_t(self.generation), out.unsafe_origin_cast[MutAnyOrigin]())
        )
        if status != 0:
            raise Error(ErrorKind.of(status), "fu_pool_is_complete")
        return out[unsafe_offset=0] != 0


struct Fabric:
    """What ForkUnion measured about this machine's memory, as opposed to what it declares.

    A `Topology` holds what the platform reports; a `Fabric` holds per-edge latencies, bandwidths,
    and distances observed in-process, plus the per-medium tiers derived from them.
    """

    var library: Library
    var handle: Handle

    def __init__(out self, library: Library) raises Error:
        self.library = library
        # Stack storage the C side writes through; see `OutSize` for why the origin is `Any`.
        var out = stack_allocation[1, Int]()
        out[unsafe_offset=0] = 0
        var status = library.symbols().fabric_new(out.unsafe_origin_cast[MutAnyOrigin]())
        if status != 0:
            raise Error(ErrorKind.of(status), "fu_fabric_new")
        self.handle = Handle(unsafe_from_address=out[unsafe_offset=0])

    def __deinit__(deinit self):
        self.library.symbols().fabric_delete(self.handle)

    def harvest(mut self, topology: Topology, pool: Pool) raises Error:
        """Measures every reachable edge, using the pool to drive the probes. Takes seconds."""
        var status = self.library.symbols().fabric_harvest(topology.handle, pool.handle, self.handle)
        if status != 0:
            raise Error(ErrorKind.of(status), "fu_fabric_harvest")

    @always_inline
    def _edge(
        self,
        symbol: def(Handle, c_size_t, c_size_t, OutSize) thin abi("C") -> c_int,
        compute: ComputeDomain,
        memory: MemoryDomain,
    ) raises Error -> Int:
        var out = stack_allocation[1, c_size_t]()
        out[unsafe_offset=0] = c_size_t(0)
        var status = symbol(
            self.handle,
            c_size_t(compute.index),
            c_size_t(memory.index),
            out.unsafe_origin_cast[MutAnyOrigin](),
        )
        if status != 0:
            raise Error(ErrorKind.of(status), "a fabric edge query was refused")
        return Int(out[unsafe_offset=0])

    def memory_latency(self, compute: ComputeDomain, memory: MemoryDomain) raises Error -> Int:
        """Observed access latency from one compute domain to one memory domain."""
        return self._edge(self.library.symbols().fabric_memory_latency, compute, memory)

    def memory_bandwidth(self, compute: ComputeDomain, memory: MemoryDomain) raises Error -> Int:
        """Observed bandwidth from one compute domain to one memory domain."""
        return self._edge(self.library.symbols().fabric_memory_bandwidth, compute, memory)

    def memory_distance(self, compute: ComputeDomain, memory: MemoryDomain) raises Error -> Int:
        """The SLIT-style distance, where a domain's distance to its own memory is 10."""
        return self._edge(self.library.symbols().fabric_memory_distance, compute, memory)

    def memory_level_in(self, memory: MemoryDomain) raises Error -> Int:
        """Which performance tier a memory domain landed in, counted from the fastest."""
        var out = stack_allocation[1, c_size_t]()
        out[unsafe_offset=0] = c_size_t(0)
        var status = self.library.symbols().fabric_memory_level_in(
            self.handle, c_size_t(memory.index), out.unsafe_origin_cast[MutAnyOrigin]()
        )
        if status != 0:
            raise Error(ErrorKind.of(status), "fu_fabric_memory_level_in")
        return Int(out[unsafe_offset=0])

    def memory_levels_count(self) raises Error -> Int:
        """Distinct memory tiers this machine turned out to have."""
        var out = stack_allocation[1, c_size_t]()
        out[unsafe_offset=0] = c_size_t(0)
        var status = self.library.symbols().fabric_memory_levels_count(
            self.handle, out.unsafe_origin_cast[MutAnyOrigin]()
        )
        if status != 0:
            raise Error(ErrorKind.of(status), "fu_fabric_memory_levels_count")
        return Int(out[unsafe_offset=0])
