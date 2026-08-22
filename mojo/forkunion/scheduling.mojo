"""The thread pool, its dispatch primitives, and the measured memory fabric.

The callback is a parameter rather than a value, which is what lets the trampoline below become a
plain C function pointer: it names only parameters, and parameters monomorphise. A `@parameter`
closure that captured anything would make the trampoline `capturing`, which cannot be handed to C
at all, so the state a callback needs travels in an explicit scratch instead.

The scratch travels by reference. A blocking dispatch does not return until every prong is done,
so the reference is alive for the whole call; taking it here rather than asking the caller for a
pointer is what stops a caller from handing over storage that dies first.
"""

from std.ffi import c_int, c_size_t

from forkunion.errors import ErrorKind, ForkUnionError
from forkunion.library import Context, Handle, Library
from forkunion.topology import Topology
from forkunion.types import CallerExclusivity, Capabilities, ComputeDomain, MemoryDomain, Prong


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
    ) raises ForkUnionError:
        """Spawns `threads` workers across the whole machine."""
        self.library = topology.library
        self.handle = Self._create(self.library, name, allowed)
        var spawned = self.library.symbols().pool_spawn(
            topology.handle, self.handle, c_size_t(threads), exclusivity.identifier
        )
        Self._check_spawn(threads, spawned, "fu_pool_spawn")

    @staticmethod
    def on(
        topology: Topology,
        domain: ComputeDomain,
        threads: Int,
        exclusivity: CallerExclusivity = CallerExclusivity.INCLUSIVE,
        name: StaticString = "",
        allowed: Capabilities = Capabilities.ALL,
    ) raises ForkUnionError -> Self:
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
        Self._check_spawn(threads, spawned, "fu_pool_spawn_on")
        return Self(library=library, handle=handle)

    def __init__(out self, *, library: Library, handle: Handle):
        self.library = library
        self.handle = handle

    @staticmethod
    def _create(library: Library, name: StaticString, allowed: Capabilities) raises ForkUnionError -> Handle:
        """Allocates the pool object; placement is decided later, at the spawn."""
        var address = library.symbols().pool_new(
            name.unsafe_ptr().unsafe_bitcast[Int8]().unsafe_origin_cast[ImmUntrackedOrigin](),
            c_int(Int(allowed.bits)),
        )
        if address == 0:
            raise ForkUnionError(ErrorKind.CREATION_FAILED, "fu_pool_new")
        return Handle(unsafe_from_address=address)

    @staticmethod
    def _check_spawn(threads: Int, spawned: c_int, symbol: StaticString) raises ForkUnionError:
        """Zero threads is not a pool, and is rejected here rather than blamed on the core."""
        if threads == 0:
            raise ForkUnionError(ErrorKind.INVALID_PARAMETER, "a pool needs at least one thread")
        if spawned == 0:
            raise ForkUnionError(ErrorKind.SPAWN_FAILED, symbol)

    def __deinit__(deinit self):
        self.library.symbols().pool_terminate(self.handle)
        self.library.symbols().pool_delete(self.handle)

    # endregion Lifetime

    # region Queries

    def threads_count(self) -> Int:
        """Workers in the pool, including the caller on an inclusive pool."""
        return Int(self.library.symbols().pool_threads_count(self.handle))

    def threads_count_in(self, domain: ComputeDomain) -> Int:
        """Workers the pool placed in one compute domain."""
        return Int(self.library.symbols().pool_threads_count_in(self.handle, c_size_t(domain.index)))

    def compute_domains_count(self) -> Int:
        """Compute domains this pool spans, which may be fewer than the machine's."""
        return Int(self.library.symbols().pool_compute_domains_count(self.handle))

    def locate_thread_in(self, global_thread_index: Int, domain: ComputeDomain) -> Int:
        """A global thread index expressed as a local one inside a compute domain."""
        return Int(
            self.library.symbols().pool_locate_thread_in(
                self.handle, c_size_t(global_thread_index), c_size_t(domain.index)
            )
        )

    def caller_exclusivity(self) -> CallerExclusivity:
        """Queried live, so it stays correct across a terminate and a re-spawn."""
        return CallerExclusivity(self.library.symbols().pool_caller_exclusivity(self.handle))

    def capabilities(self) -> Capabilities:
        """The requested allow-mask narrowed by what the machine offers; masking never adds."""
        return Capabilities(UInt32(Int(self.library.symbols().pool_capabilities(self.handle))))

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
    ](self, ref[scratch_origin] scratch: Scratch):
        """One call per worker, which is where per-thread scratch is set up or torn down.

        The callback takes the thread and compute-domain indices; the C API supplies no task index
        here, so none is invented.
        """

        def trampoline(carried: Context, thread: c_size_t, domain: c_size_t) abi("C"):
            work(Int(thread), Int(domain), carried.unsafe_bitcast[Scratch]()[])

        self.library.symbols().pool_for_threads(self.handle, trampoline, _erase(scratch))

    def for_n[
        Scratch: AnyType,
        scratch_origin: MutOrigin,
        //,
        work: def(Prong, mut Scratch) thin -> None,
    ](self, n: Int, ref[scratch_origin] scratch: Scratch):
        """Splits `n` prongs into equal contiguous chunks and blocks until every one has finished."""

        def trampoline(carried: Context, task: c_size_t, thread: c_size_t, domain: c_size_t) abi("C"):
            work(Prong(Int(task), Int(thread), Int(domain)), carried.unsafe_bitcast[Scratch]()[])

        self.library.symbols().pool_for_n(self.handle, c_size_t(n), trampoline, _erase(scratch))

    def for_n_dynamic[
        Scratch: AnyType,
        scratch_origin: MutOrigin,
        //,
        work: def(Prong, mut Scratch) thin -> None,
    ](self, n: Int, ref[scratch_origin] scratch: Scratch):
        """The same, but prongs are claimed as threads free up, for work of uneven cost."""

        def trampoline(carried: Context, task: c_size_t, thread: c_size_t, domain: c_size_t) abi("C"):
            work(Prong(Int(task), Int(thread), Int(domain)), carried.unsafe_bitcast[Scratch]()[])

        self.library.symbols().pool_for_n_dynamic(self.handle, c_size_t(n), trampoline, _erase(scratch))

    def for_slices[
        Scratch: AnyType,
        scratch_origin: MutOrigin,
        //,
        work: def(Prong, Int, mut Scratch) thin -> None,
    ](self, n: Int, ref[scratch_origin] scratch: Scratch):
        """One contiguous run per worker, for vectorized or per-slice-setup work.

        The prong's `task_index` is the run's first index and the second argument is its length;
        an idle worker receives a length of zero.
        """

        def trampoline(carried: Context, first: c_size_t, count: c_size_t, thread: c_size_t, domain: c_size_t) abi("C"):
            work(
                Prong(Int(first), Int(thread), Int(domain)),
                Int(count),
                carried.unsafe_bitcast[Scratch]()[],
            )

        self.library.symbols().pool_for_slices(self.handle, c_size_t(n), trampoline, _erase(scratch))

    # endregion Blocking Dispatch

    # region Non-Blocking Dispatch

    def unsafe_for_threads[
        Scratch: AnyType,
        pool_origin: ImmOrigin,
        scratch_origin: MutOrigin,
        //,
        work: def(Int, Int, mut Scratch) thin -> None,
    ](ref[pool_origin] self, ref[scratch_origin] scratch: Scratch) -> BroadcastJoin[pool_origin, scratch_origin]:
        """Dispatches on every worker without blocking, returning a guard that joins on exit.

        Use it as a context manager. On an exclusive pool the work starts here and the generation
        can be polled; on an inclusive pool the caller owes a slice, so the work runs at the join.
        """

        def trampoline(carried: Context, thread: c_size_t, domain: c_size_t) abi("C"):
            work(Int(thread), Int(domain), carried.unsafe_bitcast[Scratch]()[])

        var generation = Int(self.library.symbols().pool_unsafe_for_threads(self.handle, trampoline, _erase(scratch)))
        return BroadcastJoin[pool_origin, scratch_origin](Pointer(to=self), generation)

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

    def is_complete(self) -> Bool:
        """A non-blocking poll; meaningful on exclusive pools, where the caller owes no slice."""
        return self.pool[].library.symbols().pool_is_complete(self.pool[].handle, c_size_t(self.generation)) != 0


struct Fabric:
    """What ForkUnion measured about this machine's memory, as opposed to what it declares.

    A `Topology` holds what the platform reports; a `Fabric` holds per-edge latencies, bandwidths,
    and distances observed in-process, plus the per-medium tiers derived from them.
    """

    var library: Library
    var handle: Handle

    def __init__(out self, library: Library) raises ForkUnionError:
        self.library = library
        var address = library.symbols().fabric_new()
        if address == 0:
            raise ForkUnionError(ErrorKind.CREATION_FAILED, "fu_fabric_new")
        self.handle = Handle(unsafe_from_address=address)

    def __deinit__(deinit self):
        self.library.symbols().fabric_delete(self.handle)

    def try_harvest(mut self, topology: Topology, pool: Pool) -> Bool:
        """Measures every reachable edge, using the pool to drive the probes. Takes seconds."""
        return self.library.symbols().fabric_harvest(topology.handle, pool.handle, self.handle) != 0

    @always_inline
    def _edge(
        self,
        symbol: def(Handle, c_size_t, c_size_t) thin abi("C") -> c_size_t,
        compute: ComputeDomain,
        memory: MemoryDomain,
    ) -> Int:
        return Int(symbol(self.handle, c_size_t(compute.index), c_size_t(memory.index)))

    def memory_latency(self, compute: ComputeDomain, memory: MemoryDomain) -> Int:
        """Observed access latency from one compute domain to one memory domain."""
        return self._edge(self.library.symbols().fabric_memory_latency, compute, memory)

    def memory_bandwidth(self, compute: ComputeDomain, memory: MemoryDomain) -> Int:
        """Observed bandwidth from one compute domain to one memory domain."""
        return self._edge(self.library.symbols().fabric_memory_bandwidth, compute, memory)

    def memory_distance(self, compute: ComputeDomain, memory: MemoryDomain) -> Int:
        """The SLIT-style distance, where a domain's distance to its own memory is 10."""
        return self._edge(self.library.symbols().fabric_memory_distance, compute, memory)

    def memory_level_in(self, memory: MemoryDomain) -> Int:
        """Which performance tier a memory domain landed in, counted from the fastest."""
        return Int(self.library.symbols().fabric_memory_level_in(self.handle, c_size_t(memory.index)))

    def memory_levels_count(self) -> Int:
        """Distinct memory tiers this machine turned out to have."""
        return Int(self.library.symbols().fabric_memory_levels_count(self.handle))
