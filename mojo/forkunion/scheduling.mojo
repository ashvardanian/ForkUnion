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
"""`fu_pool_new`: allocates a pool from a name and an allow-mask, writing the handle out."""
comptime PoolDelete = def(Handle) thin abi("C") -> None
"""`fu_pool_delete`: frees a pool whose workers have already stopped."""
comptime PoolCapabilities = def(Handle, Pointer[Int32, MutAnyOrigin]) thin abi("C") -> c_int
"""`fu_pool_capabilities`: reads back the allow-mask the spawn actually settled on."""
comptime PoolSpawn = def(Handle, Handle, c_size_t, c_int) thin abi("C") -> c_int
"""`fu_pool_spawn`: starts workers across every compute domain the topology describes."""
comptime PoolSpawnOn = def(Handle, Handle, c_size_t, c_size_t, c_int) thin abi("C") -> c_int
"""`fu_pool_spawn_on`: starts workers pinned to the one compute domain named by index."""
comptime PoolExclusivity = def(Handle, Pointer[Int32, MutAnyOrigin]) thin abi("C") -> c_int
"""`fu_pool_caller_exclusivity`: reads whether the dispatching thread also runs a slice."""
comptime PoolCount = def(Handle, OutSize) thin abi("C") -> c_int
"""`fu_pool_threads_count` and `fu_pool_compute_domains_count`: a whole-pool count."""
comptime PoolCountIn = def(Handle, c_size_t, OutSize) thin abi("C") -> c_int
"""`fu_pool_threads_count_in`: the same count narrowed to one compute-domain index."""
comptime PoolLocateThreadIn = def(Handle, c_size_t, c_size_t, OutSize) thin abi("C") -> c_int
"""`fu_pool_locate_thread_in`: rebases a global thread index inside one compute domain."""
comptime PoolSleep = def(Handle, c_size_t) thin abi("C") -> None
"""`fu_pool_sleep`: parks idle workers, re-checking for work on the given microsecond period."""
comptime PoolTerminate = def(Handle) thin abi("C") -> None
"""`fu_pool_terminate`: stops the workers while leaving the handle spawnable again."""
comptime FabricNew = def(OutHandle) thin abi("C") -> c_int
"""`fu_fabric_new`: allocates an empty fabric, which holds nothing until a harvest."""
comptime FabricDelete = def(Handle) thin abi("C") -> None
"""`fu_fabric_delete`: frees a fabric and the observations it snapshotted."""
comptime FabricHarvest = def(Handle, Handle, Handle) thin abi("C") -> c_int
"""`fu_fabric_harvest`: probes every reachable edge through a pool, replacing what the fabric held."""
comptime FabricEdge = def(Handle, c_size_t, c_size_t, OutSize) thin abi("C") -> c_int
"""`fu_fabric_memory_latency`, `fu_fabric_memory_bandwidth`, and `fu_fabric_memory_distance`: one edge."""
comptime FabricLevelIn = def(Handle, c_size_t, OutSize) thin abi("C") -> c_int
"""`fu_fabric_memory_level_in`: the performance tier one memory domain was sorted into."""
comptime FabricLevelsCount = def(Handle, OutSize) thin abi("C") -> c_int
"""`fu_fabric_memory_levels_count`: how many distinct tiers the harvest distinguished."""
comptime ForThreads = def(Context, c_size_t, c_size_t) thin abi("C") -> None
"""`fu_for_threads_t`: the callback of a per-worker dispatch, which carries no task index."""
comptime ForProngs = def(Context, c_size_t, c_size_t, c_size_t) thin abi("C") -> None
"""`fu_for_task_t`: the callback of an indexed dispatch, invoked once per task."""
comptime ForSlices = def(Context, c_size_t, c_size_t, c_size_t, c_size_t) thin abi("C") -> None
"""`fu_for_range_t`: the callback of a slice dispatch, given a first-and-count run of tasks."""
comptime PoolForThreads = def(Handle, ForThreads, Context) thin abi("C") -> c_int
"""`fu_pool_for_threads`: one blocking call per worker."""
comptime PoolForN = def(Handle, c_size_t, ForProngs, Context) thin abi("C") -> c_int
"""`fu_pool_for_n` and `fu_pool_for_n_dynamic`: `n` tasks, split up front or claimed on demand."""
comptime PoolForSlices = def(Handle, c_size_t, ForSlices, Context) thin abi("C") -> c_int
"""`fu_pool_for_slices`: `n` tasks as one contiguous run per worker."""
comptime PoolUnsafeForThreads = def(Handle, ForThreads, Context, OutSize) thin abi("C") -> c_int
"""`fu_pool_unsafe_for_threads`: the non-blocking per-worker dispatch, writing a generation out."""
comptime PoolIsComplete = def(Handle, c_size_t, Pointer[Int32, MutAnyOrigin]) thin abi("C") -> c_int
"""`fu_pool_is_complete`: polls whether one generation has finished."""
comptime PoolUnsafeJoin = def(Handle, c_size_t) thin abi("C") -> None
"""`fu_pool_unsafe_join`: blocks until one generation completes, running the caller's slice first."""

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
    """The loaded core, held so its symbol table outlives every call made through the handle."""
    var handle: Handle
    """The opaque `fu_pool_t` every call below hands back to the core."""

    # region Lifetime

    def __init__(
        out self,
        topology: Topology,
        threads: Int,
        exclusivity: CallerExclusivity = CallerExclusivity.INCLUSIVE,
        name: StaticString = "",
        allowed: Capabilities = Capabilities.ALL,
    ) raises Error:
        """Spawns `threads` workers across the whole machine.

        Args:
            topology: The machine description; read only during this call, and freeable afterwards.
            threads: Worker count above zero, counting the caller itself on an inclusive pool.
            exclusivity: Whether the dispatching thread runs a slice or only coordinates.
            name: What the workers answer to in a debugger; an empty name leaves them unnamed.
            allowed: The spin hints and placement facilities the spawn may pick from, before masking.

        Raises:
            `Error` if the pool cannot be allocated, is already spawned, or the OS refuses the threads.
        """
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
        """Spawns `threads` workers pinned to one compute domain.

        Args:
            topology: The machine description; read only during this call, and freeable afterwards.
            domain: The one domain the workers and their domain-local allocations stay on.
            threads: Worker count above zero, counting the caller itself on an inclusive pool.
            exclusivity: Whether the dispatching thread runs a slice or only coordinates.
            name: What the workers answer to in a debugger; an empty name leaves them unnamed.
            allowed: The spin hints and placement facilities the spawn may pick from, before masking.

        Returns:
            A pool whose workers never leave `domain`, so one per domain can be driven side by side.

        Raises:
            `Error` if the domain is out of range, the pool is already spawned, or the OS refuses the threads.
        """
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
        """Adopts a handle that is already spawned, which is how `on` hands one back.

        Args:
            library: The loaded core the handle was created through, and is deleted through.
            handle: An `fu_pool_t` whose lifetime this pool takes over.
        """
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
        """Stops the workers before freeing the pool, so nothing is still running at the delete."""
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
        """Workers in the pool, including the caller on an inclusive pool.

        Returns:
            The width the last spawn settled on, so a re-spawn at a different width changes it.

        Raises:
            `Error` if the pool was never spawned.
        """
        return self._count(self.library.symbols().pool_threads_count, "fu_pool_threads_count")

    def threads_count_in(self, domain: ComputeDomain) raises Error -> Int:
        """Workers the pool placed in one compute domain.

        Args:
            domain: A dense compute-domain index, below `compute_domains_count` and not bounds-checked.

        Returns:
            That domain's share alone, which need not be an equal split across the domains.

        Raises:
            `Error` if the pool was never spawned.
        """
        var out = stack_allocation[1, c_size_t]()
        out[unsafe_offset=0] = c_size_t(0)
        var status = self.library.symbols().pool_threads_count_in(
            self.handle, c_size_t(domain.index), out.unsafe_origin_cast[MutAnyOrigin]()
        )
        if status != 0:
            raise Error(ErrorKind.of(status), "fu_pool_threads_count_in")
        return Int(out[unsafe_offset=0])

    def compute_domains_count(self) raises Error -> Int:
        """Compute domains this pool spans, which may be fewer than the machine's.

        Returns:
            One on a flat pool or a domain-pinned one, more once NUMA nodes or QoS tiers are crossed.

        Raises:
            `Error` if the pool was never spawned.
        """
        return self._count(self.library.symbols().pool_compute_domains_count, "fu_pool_compute_domains_count")

    def locate_thread_in(self, global_thread_index: Int, domain: ComputeDomain) raises Error -> Int:
        """A global thread index expressed as a local one inside a compute domain.

        Args:
            global_thread_index: A thread index as a dispatch reports it, spanning the whole pool.
            domain: The domain to rebase into, below `compute_domains_count` and not bounds-checked.

        Returns:
            That thread's position among the domain's own workers, where zero is the domain's first.

        Raises:
            `Error` if the pool was never spawned.
        """
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
        """Queried live, so it stays correct across a terminate and a re-spawn.

        Returns:
            `INCLUSIVE` when the dispatching thread owes a slice, `EXCLUSIVE` when it only coordinates.

        Raises:
            `Error` if the pool was never spawned.
        """
        # Stack storage the C side writes through; see `OutSize` for why the origin is `Any`.
        var out = stack_allocation[1, Int32]()
        out[unsafe_offset=0] = Int32(0)
        var status = self.library.symbols().pool_caller_exclusivity(self.handle, out.unsafe_origin_cast[MutAnyOrigin]())
        if status != 0:
            raise Error(ErrorKind.of(status), "fu_pool_caller_exclusivity")
        return CallerExclusivity(c_int(Int(out[unsafe_offset=0])))

    def capabilities(self) raises Error -> Capabilities:
        """The requested allow-mask narrowed by what the machine offers; masking never adds.

        Returns:
            The bits that survived, which the waiter and the placement strategy were chosen from.

        Raises:
            `Error` if the pool was never spawned.
        """
        # Stack storage the C side writes through; see `OutSize` for why the origin is `Any`.
        var out = stack_allocation[1, Int32]()
        out[unsafe_offset=0] = Int32(0)
        var status = self.library.symbols().pool_capabilities(self.handle, out.unsafe_origin_cast[MutAnyOrigin]())
        if status != 0:
            raise Error(ErrorKind.of(status), "fu_pool_capabilities")
        return Capabilities(UInt32(Int(out[unsafe_offset=0])))

    def sleep(self, microseconds: Int):
        """Parks the workers; the next dispatch wakes them.

        Args:
            microseconds: How long a parked worker waits before re-checking for work, above zero.
        """
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

        Parameters:
            Scratch: The scratch type, inferred from the argument.
            scratch_origin: The scratch's origin, inferred with it.
            work: The callback, taking a pool-wide thread index, a compute-domain index, and the scratch.

        Args:
            scratch: The state the callback reads and writes, borrowed for the whole call.

        Raises:
            `Error` if the pool was never spawned, or the core refuses the dispatch.
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
        """Splits `n` tasks into equal contiguous chunks and blocks until every one has finished.

        Parameters:
            Scratch: The scratch type, inferred from the argument.
            scratch_origin: The scratch's origin, inferred with it.
            work: The callback, taking a task index, a `ThreadInDomain`, and the scratch.

        Args:
            n: How many tasks to split across the pool, one callback invocation each.
            scratch: The state the callback reads and writes, borrowed for the whole call.

        Raises:
            `Error` if the pool was never spawned, or the core refuses the dispatch.
        """

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
        """The same, but tasks are claimed as threads free up, for work of uneven cost.

        Parameters:
            Scratch: The scratch type, inferred from the argument.
            scratch_origin: The scratch's origin, inferred with it.
            work: The callback, taking a task index, a `ThreadInDomain`, and the scratch.

        Args:
            n: How many tasks to hand out, no worker taking a second before it finishes its first.
            scratch: The state the callback reads and writes, borrowed for the whole call.

        Raises:
            `Error` if the pool was never spawned, or the core refuses the dispatch.
        """

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

        Parameters:
            Scratch: The scratch type, inferred from the argument.
            scratch_origin: The scratch's origin, inferred with it.
            work: The callback, taking a `TasksRange`, a `ThreadInDomain`, and the scratch.

        Args:
            n: How many tasks to divide into one run per worker, however few workers there are.
            scratch: The state the callback reads and writes, borrowed for the whole call.

        Raises:
            `Error` if the pool was never spawned, or the core refuses the dispatch.
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

        Parameters:
            Scratch: The scratch type, inferred from the argument.
            pool_origin: This pool's borrow, inferred from `self` and carried on into the guard.
            scratch_origin: The scratch's origin, inferred with the scratch and carried on too.
            work: The callback, taking a pool-wide thread index, a compute-domain index, and the scratch.

        Args:
            scratch: The state the callback reads and writes, borrowed until the join rather than the return.

        Returns:
            A guard naming the in-flight generation, which joins when its `with` block ends.

        Raises:
            `Error` if the pool was never spawned, or the core refuses the dispatch.
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

    Both origins are load-bearing, because Mojo destroys a value after its last named use and the
    dispatch outlives that point.

    Parameters:
        pool_origin: The pool's borrow, without which it would be torn down while its workers were
            still running - and a `terminate` mid-flight trips an assertion inside the core.
        scratch_origin: The scratch's borrow, keeping the storage the workers write through alive
            until the join, which a raw context pointer cannot express.
    """

    var pool: Pointer[Pool, Self.pool_origin]
    """The dispatching pool, reached at the join and at every poll."""
    var generation: Int
    """The token the core stamped on this dispatch, which the poll and the join both name."""

    def __enter__(self) -> Self:
        """Hands back the guard itself, so the block can poll as well as read the generation.

        Returns:
            This same guard, so `is_complete` stays reachable from inside the block.
        """
        return self

    def __exit__(mut self):
        """Joins the generation, running the caller's own slice first on an inclusive pool."""
        self.pool[].library.symbols().pool_unsafe_join(self.pool[].handle, c_size_t(self.generation))

    def is_complete(self) raises Error -> Bool:
        """A non-blocking poll; meaningful on exclusive pools, where the caller owes no slice.

        Returns:
            True once every worker has left this generation; on an inclusive pool, never before the join.

        Raises:
            `Error` if the pool was never spawned.
        """
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
    """The loaded core, held so its symbol table outlives every call made through the handle."""
    var handle: Handle
    """The opaque `fu_fabric_t`, which answers zeroes until a harvest has filled it."""

    def __init__(out self, library: Library) raises Error:
        """Allocates an empty fabric; every query answers zero until `harvest` has run.

        Args:
            library: The loaded core the fabric is allocated through and later queried against.

        Raises:
            `Error` if the allocation is refused.
        """
        self.library = library
        # Stack storage the C side writes through; see `OutSize` for why the origin is `Any`.
        var out = stack_allocation[1, Int]()
        out[unsafe_offset=0] = 0
        var status = library.symbols().fabric_new(out.unsafe_origin_cast[MutAnyOrigin]())
        if status != 0:
            raise Error(ErrorKind.of(status), "fu_fabric_new")
        self.handle = Handle(unsafe_from_address=out[unsafe_offset=0])

    def __deinit__(deinit self):
        """Frees the fabric along with the observations it snapshotted."""
        self.library.symbols().fabric_delete(self.handle)

    def harvest(mut self, topology: Topology, pool: Pool) raises Error:
        """Measures every reachable edge, using the pool to drive the probes. Takes seconds.

        Args:
            topology: The machine description; snapshotted here, so it may be freed once this returns.
            pool: A pool spawned across the whole machine, whose pinned workers run the probes.

        Raises:
            `Error` if the pool spans no memory domains - flat, pinned to one domain, or terminated.
        """
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
        """Observed access latency from one compute domain to one memory domain.

        Args:
            compute: The initiating domain, whose worker pointer-chases the edge.
            memory: The target domain, whose pages the probe first-touched.

        Returns:
            Dependent-load latency in nanoseconds, best of the recordings; zero for an unwalked edge.

        Raises:
            `Error` if the core refuses the query; an unharvested edge answers zero rather than failing.
        """
        return self._edge(self.library.symbols().fabric_memory_latency, compute, memory)

    def memory_bandwidth(self, compute: ComputeDomain, memory: MemoryDomain) raises Error -> Int:
        """Observed bandwidth from one compute domain to one memory domain.

        Args:
            compute: The initiating domain, whose whole worker set streams the edge at once.
            memory: The target domain, whose pages the probe first-touched.

        Returns:
            Saturated read bandwidth in megabytes per second; zero for an unwalked edge.

        Raises:
            `Error` if the core refuses the query; an unharvested edge answers zero rather than failing.
        """
        return self._edge(self.library.symbols().fabric_memory_bandwidth, compute, memory)

    def memory_distance(self, compute: ComputeDomain, memory: MemoryDomain) raises Error -> Int:
        """The SLIT-style distance, where a domain's distance to its own memory is 10.

        Args:
            compute: The initiating domain, whose local memory is clamped to its row's minimum.
            memory: The target domain, whose pages the probe first-touched.

        Returns:
            The measured latency ratio to the initiator's local memory, larger meaning farther.

        Raises:
            `Error` if the core refuses the query; an unwalked edge falls back to 10 local, 20 remote.
        """
        return self._edge(self.library.symbols().fabric_memory_distance, compute, memory)

    def memory_level_in(self, memory: MemoryDomain) raises Error -> Int:
        """Which performance tier a memory domain landed in, counted from the fastest.

        Args:
            memory: The domain to classify, independently of any initiator.

        Returns:
            A tier ordinal with zero the fastest, keyed by best sustained bandwidth and split by latency.

        Raises:
            `Error` if the core refuses the query; an unharvested fabric answers zero rather than failing.
        """
        var out = stack_allocation[1, c_size_t]()
        out[unsafe_offset=0] = c_size_t(0)
        var status = self.library.symbols().fabric_memory_level_in(
            self.handle, c_size_t(memory.index), out.unsafe_origin_cast[MutAnyOrigin]()
        )
        if status != 0:
            raise Error(ErrorKind.of(status), "fu_fabric_memory_level_in")
        return Int(out[unsafe_offset=0])

    def memory_levels_count(self) raises Error -> Int:
        """Distinct memory tiers this machine turned out to have.

        Returns:
            One on a uniform machine, more once HBM, DDR, or CXL are mixed; domains may share a tier.

        Raises:
            `Error` if the core refuses the query; an unharvested fabric answers a single tier.
        """
        var out = stack_allocation[1, c_size_t]()
        out[unsafe_offset=0] = c_size_t(0)
        var status = self.library.symbols().fabric_memory_levels_count(
            self.handle, out.unsafe_origin_cast[MutAnyOrigin]()
        )
        if status != 0:
            raise Error(ErrorKind.of(status), "fu_fabric_memory_levels_count")
        return Int(out[unsafe_offset=0])
