"""NUMA-aware allocation: pages placed on a chosen memory domain rather than wherever they land.

An allocator here is named by the operating system's own memory-domain id, not by a dense index,
and holds nothing else. That is what lets it - and every buffer it hands out - outlive the
topology handle that found the id. Ask the topology for a compute domain's local memory, resolve
that index to an id, and keep the id.
"""

from std.ffi import c_int, c_size_t
from std.math import ceildiv
from std.memory import stack_allocation
from std.sys import size_of

from forkunion.library import Library
from forkunion.topology import Topology
from forkunion.types import (
    bytes_for_elements,
    ComputeDomain,
    Error,
    ErrorKind,
    Handle,
    MemoryDomain,
    MemoryDomainId,
    OutAddress,
    OutSize,
)

# region Signatures

comptime AllocateOnDomain = def(c_int, c_size_t, OutAddress) thin abi("C") -> c_int
"""`fu_allocate_on_domain_id`: places exactly the bytes asked for, writing the address out."""

comptime AllocateAtLeastOnDomain = def(c_int, c_size_t, OutSize, OutSize, OutAddress) thin abi("C") -> c_int
"""`fu_allocate_at_least_on_domain_id`: rounds up onto the largest suitable page, reporting both sizes."""

comptime FreeOnDomain = def(c_int, Pointer[Int8, MutUntrackedOrigin], c_size_t) thin abi("C") -> None
"""`fu_free_on_domain_id`: releases pages, and needs the byte count the placement reported back."""

comptime AllocateSymmetric = def(Handle, c_size_t, OutSize, OutSize, OutSize, OutSize, OutAddress) thin abi(
    "C"
) -> c_int
"""`fu_allocate_symmetric`: one mapping striped over every domain, reporting stride, slices, total, and page."""

comptime FreeSymmetric = def(Pointer[Int8, MutUntrackedOrigin], c_size_t) thin abi("C") -> None
"""`fu_free_symmetric`: unmaps the whole range at once, so it takes the total rather than a stride."""

# endregion Signatures


comptime Bytes = Pointer[Int8, MutUntrackedOrigin]
"""The raw storage a placement allocator hands back."""


struct AllocationResult:
    """Pages placed on one memory domain, released when the last reference goes out of use."""

    var library: Library
    """The loader the release symbol comes from, kept alive for exactly as long as the pages."""
    var memory_domain_id: MemoryDomainId
    """The domain the pages sit on, replayed on release because the C free takes it too."""
    var pointer: Bytes
    """The base of the placed pages, aligned to at least a cache line."""
    var allocated_bytes: Int
    """What was actually placed, which the release has to repeat exactly."""
    var bytes_per_page: Int
    """The page size behind the pages, or zero where the entry point reported none."""

    def __init__(
        out self,
        library: Library,
        memory_domain_id: MemoryDomainId,
        address: Int,
        allocated_bytes: Int,
        bytes_per_page: Int,
    ):
        """Takes ownership of pages a placement call has already returned.

        Args:
            library: The loader to release through, which the result then keeps alive.
            memory_domain_id: The domain the pages were placed on, needed again to free them.
            address: The address the C side wrote out, not yet a typed pointer.
            allocated_bytes: What the placement really returned, which may exceed the request.
            bytes_per_page: The page size behind the pages, or zero where the call reported none.
        """
        self.library = library
        self.memory_domain_id = memory_domain_id
        self.pointer = Bytes(unsafe_from_address=address)
        self.allocated_bytes = allocated_bytes
        self.bytes_per_page = bytes_per_page

    def __deinit__(deinit self):
        """Hands the pages back to the domain they came from, at the byte count they were placed at."""
        self.library.symbols().free_on_domain_id(
            self.memory_domain_id.identifier,
            self.pointer,
            c_size_t(self.allocated_bytes),
        )

    def as_pointer[dtype: DType](mut self) -> Pointer[Scalar[dtype], origin_of(self)]:
        """The same storage, seen as the element type the caller means to put in it.

        The returned pointer carries this allocation's origin, so the compiler keeps the pages
        alive for as long as the pointer is reachable. Without that, Mojo would release the
        allocation after its last named use - which is this call - and the pointer would dangle
        before the caller ever wrote through it.

        Parameters:
            dtype: The element type to read the untyped pages as; nothing is converted.

        Returns:
            A pointer borrowing this allocation, so the pages cannot be released beneath it.
        """
        return self.pointer.unsafe_bitcast[Scalar[dtype]]().unsafe_origin_cast[origin_of(self)]()

    def count_of[dtype: DType](self) -> Int:
        """How many elements of `dtype` the placed pages hold.

        Parameters:
            dtype: The element type whose width divides the placed bytes.

        Returns:
            The count the pages fit, above what was asked for whenever page rounding rounded up.
        """
        return self.allocated_bytes // size_of[Scalar[dtype]]()


@fieldwise_init
struct DomainAllocator(Equatable, ImplicitlyCopyable):
    """Places pages on one memory domain, named by the operating system's id for it.

    Holds the id and a loader reference and nothing else, so it is cheap to copy and free of the
    topology that produced the id.
    """

    var library: Library
    """The loader the placement and release symbols come from."""
    var memory_domain_id: MemoryDomainId
    """The operating system's own id for the domain, never a dense index into the topology."""

    @staticmethod
    def at(library: Library, memory_domain_id: MemoryDomainId) raises Error -> Self:
        """An allocator for a domain the machine actually has.

        Args:
            library: The loader every allocation from here is placed and released through.
            memory_domain_id: An id from `Topology.memory_domain_id_at_index`, never a dense index.

        Returns:
            An allocator that outlives the topology handle the id was read from.

        Raises:
            When the id is negative, which is how a domain the OS names none of comes back.
        """
        if Int(memory_domain_id.identifier) < 0:
            raise Error(ErrorKind.INVALID_ARGUMENT, "the memory domain id names no domain")
        return Self(library, memory_domain_id)

    def __eq__(self, other: Self) -> Bool:
        """Two allocators match when they name the same domain; the loader reference is not compared.

        Args:
            other: The allocator whose domain id is compared with this one's.

        Returns:
            True when both place onto the same operating-system domain id.
        """
        return self.memory_domain_id == other.memory_domain_id

    def allocate(self, bytes: Int) raises Error -> AllocationResult:
        """Exactly `bytes` on this domain.

        The page size is not reported by this entry point, so the result carries zero for it.

        Args:
            bytes: The exact size to place, with no page rounding added on top of it.

        Returns:
            Pages bound to this allocator's domain, released when the result goes out of use.

        Raises:
            When the request is zero bytes, or the domain cannot back the placement.
        """
        if bytes == 0:
            raise Error(ErrorKind.INVALID_ARGUMENT, "an allocation needs bytes")
        # Stack storage the C side writes through; see `OutSize` for why the origin is `Any`.
        var out = stack_allocation[1, Int]()
        out[unsafe_offset=0] = 0
        var status = self.library.symbols().allocate_on_domain_id(
            self.memory_domain_id.identifier, c_size_t(bytes), out.unsafe_origin_cast[MutAnyOrigin]()
        )
        if status != 0:
            raise Error(ErrorKind.of(status), "fu_allocate_on_domain_id")
        return AllocationResult(self.library, self.memory_domain_id, out[unsafe_offset=0], bytes, 0)

    def allocate_at_least(self, minimum_bytes: Int) raises Error -> AllocationResult:
        """At least `minimum_bytes`, using the largest suitable page size, reporting both.

        Args:
            minimum_bytes: The floor on the request; huge-page rounding may hand back more.

        Returns:
            Pages whose real size and page size both come back on the result.

        Raises:
            When the request is zero bytes, or the domain cannot back the placement.
        """
        if minimum_bytes == 0:
            raise Error(ErrorKind.INVALID_ARGUMENT, "an allocation needs bytes")
        # Stack storage the C side writes through; see `OutSize` for why the origin is `Any`.
        var out = stack_allocation[2, c_size_t]()
        out[unsafe_offset=0] = c_size_t(0)
        out[unsafe_offset=1] = c_size_t(0)
        var address = stack_allocation[1, Int]()
        address[unsafe_offset=0] = 0
        var status = self.library.symbols().allocate_at_least_on_domain_id(
            self.memory_domain_id.identifier,
            c_size_t(minimum_bytes),
            out.unsafe_origin_cast[MutAnyOrigin](),
            Pointer(to=out[unsafe_offset=1]).unsafe_origin_cast[MutAnyOrigin](),
            address.unsafe_origin_cast[MutAnyOrigin](),
        )
        if status != 0:
            raise Error(ErrorKind.of(status), "fu_allocate_at_least_on_domain_id")
        return AllocationResult(
            self.library,
            self.memory_domain_id,
            address[unsafe_offset=0],
            Int(out[unsafe_offset=0]),
            Int(out[unsafe_offset=1]),
        )

    def allocate_for[dtype: DType](self, count: Int) raises Error -> AllocationResult:
        """Room for `count` elements of `dtype`.

        Parameters:
            dtype: The element type whose width scales the request.

        Args:
            count: How many elements to make room for, not a byte count.

        Returns:
            Pages sized to the product, released with the result.

        Raises:
            When the count times the element width would wrap, or the domain refuses the placement.
        """
        return self.allocate(bytes_for_elements(count, size_of[Scalar[dtype]]()))

    def allocate_for_at_least[dtype: DType](self, minimum_count: Int) raises Error -> AllocationResult:
        """Room for at least `minimum_count` elements of `dtype`, on the largest suitable page.

        Parameters:
            dtype: The element type whose width scales the request.

        Args:
            minimum_count: The floor on elements; read `count_of` back for what the pages really fit.

        Returns:
            Pages sized to at least the product, with the page size they used reported.

        Raises:
            When the count times the element width would wrap, or the domain refuses the placement.
        """
        return self.allocate_at_least(bytes_for_elements(minimum_count, size_of[Scalar[dtype]]()))


def default_domain_allocator(topology: Topology) raises Error -> DomainAllocator:
    """An allocator for the machine's first memory domain, which every machine has.

    Args:
        topology: Consulted only to resolve index zero to an id, and not held afterwards.

    Returns:
        An allocator that outlives the topology it was read from.

    Raises:
        When the machine cannot be described well enough to name even its first domain.
    """
    return DomainAllocator.at(topology.library, topology.memory_domain_id_at_index(MemoryDomain(0)))


def local_domain_allocator(topology: Topology, domain: ComputeDomain) raises Error -> DomainAllocator:
    """An allocator for the memory domain nearest a compute domain - "run here, allocate here".

    This is the pairing the index-versus-id distinction exists to protect: `local_memory_of`
    answers a dense index, and only `memory_domain_id_at_index` turns it into an allocator's id.

    Args:
        topology: Consulted for both halves of that resolution, and not held afterwards.
        domain: A dense compute-domain index, counted from zero - where the work will run.

    Returns:
        An allocator for the memory nearest that compute domain.

    Raises:
        When the machine has no such compute domain, or names no memory domain beside it.
    """
    return DomainAllocator.at(
        topology.library,
        topology.memory_domain_id_at_index(topology.local_memory_of(domain)),
    )


struct SymmetricAllocation:
    """One mapping striped across every memory domain at a uniform stride.

    Backs both the replicated and the sharded views: they differ only in what the per-domain run
    means, not in how the storage is laid out.
    """

    var library: Library
    """The loader the unmap goes through, kept alive until it does."""
    var base: Bytes
    """The start of the whole mapping; the run for domain `d` begins `d` strides in."""
    var stride_bytes: Int
    """The page-aligned distance between runs, which rounding may lift above what was asked."""
    var memory_domains_count: Int
    """How many runs the mapping got, which a machine with no NUMA API collapses to one."""
    var total_bytes: Int
    """The whole mapping's size, which the unmap needs rather than one run's."""
    var bytes_per_page: Int
    """The page size the mapping used, which is what drove the stride rounding."""

    def __init__(out self, topology: Topology, bytes_per_domain: Int) raises Error:
        """Maps one run per memory domain and binds each run to the domain it belongs to.

        Args:
            topology: Passed straight to the C mapping, which reads the domain count off it.
            bytes_per_domain: The usable floor per run; the stride is page-rounded up from it.

        Raises:
            When the request is zero bytes, or the mapping cannot be placed.
        """
        if bytes_per_domain == 0:
            raise Error(ErrorKind.INVALID_ARGUMENT, "a symmetric mapping needs bytes")
        # Stack storage the C side writes through; see `OutSize` for why the origin is `Any`.
        var out = stack_allocation[4, c_size_t]()
        for slot in range(4):
            out[unsafe_offset=slot] = c_size_t(0)
        var address = stack_allocation[1, Int]()
        address[unsafe_offset=0] = 0
        var status = topology.library.symbols().allocate_symmetric(
            topology.handle,
            c_size_t(bytes_per_domain),
            out.unsafe_origin_cast[MutAnyOrigin](),
            Pointer(to=out[unsafe_offset=1]).unsafe_origin_cast[MutAnyOrigin](),
            Pointer(to=out[unsafe_offset=2]).unsafe_origin_cast[MutAnyOrigin](),
            Pointer(to=out[unsafe_offset=3]).unsafe_origin_cast[MutAnyOrigin](),
            address.unsafe_origin_cast[MutAnyOrigin](),
        )
        if status != 0:
            raise Error(ErrorKind.of(status), "fu_allocate_symmetric")
        self.library = topology.library
        self.base = Bytes(unsafe_from_address=address[unsafe_offset=0])
        self.stride_bytes = Int(out[unsafe_offset=0])
        self.memory_domains_count = Int(out[unsafe_offset=1])
        self.total_bytes = Int(out[unsafe_offset=2])
        self.bytes_per_page = Int(out[unsafe_offset=3])

    def __deinit__(deinit self):
        """Unmaps the whole range in one call, which is why the total and not a stride is kept."""
        self.library.symbols().free_symmetric(self.base, c_size_t(self.total_bytes))

    def on_memory_domain[dtype: DType](mut self, domain: MemoryDomain) -> Pointer[Scalar[dtype], origin_of(self)]:
        """The run of storage living on one memory domain, seen as the caller's element type.

        Parameters:
            dtype: The element type to read the run as; nothing is converted.

        Args:
            domain: A dense memory-domain index, counted from zero, not an operating-system id.

        Returns:
            A pointer into the mapping, borrowing it so the run cannot be unmapped beneath it.
        """
        var offset = domain.index * self.stride_bytes
        return (
            Pointer(to=self.base[unsafe_offset=offset])
            .unsafe_bitcast[Scalar[dtype]]()
            .unsafe_origin_cast[origin_of(self)]()
        )


struct ReplicatedArray[dtype: DType]:
    """The same `length` elements held once per memory domain, so every reader is local.

    Write each replica, then let every compute domain read the copy nearest to it.

    Parameters:
        dtype: The element type every replica holds.
    """

    var storage: SymmetricAllocation
    """The single mapping behind every replica, one run per domain."""
    var length: Int
    """Elements in one replica, which is the same for all of them."""

    def __init__(out self, *, var storage: SymmetricAllocation, length: Int):
        """Adopts a mapping whose runs are already sized for one replica each.

        Args:
            storage: The mapping to take over, one run per domain.
            length: Elements per replica, which the caller already sized the runs for.
        """
        self.storage = storage^
        self.length = length

    @staticmethod
    def new(topology: Topology, length: Int) raises Error -> Self:
        """A replica of `length` elements on every memory domain.

        Args:
            topology: Says how many domains there are to replicate onto.
            length: Elements in each copy, not the total held across the domains.

        Returns:
            An array whose replicas start uninitialized; write each one before reading it.

        Raises:
            When the length times the element width would wrap, or the mapping cannot be placed.
        """
        return Self(
            storage=SymmetricAllocation(topology, bytes_for_elements(length, size_of[Scalar[Self.dtype]]())),
            length=length,
        )

    def memory_domains_count(self) -> Int:
        """How many replicas the mapping produced.

        Returns:
            The run count the mapping settled on, which is one where the machine has no NUMA API.
        """
        return self.storage.memory_domains_count

    def replica(mut self, domain: MemoryDomain) -> Pointer[Scalar[Self.dtype], origin_of(self)]:
        """One domain's copy, all `length` elements of it.

        Args:
            domain: A dense memory-domain index, counted from zero, not an operating-system id.

        Returns:
            A pointer to that copy's first element, borrowing the array so it outlives the call.
        """
        return self.storage.on_memory_domain[Self.dtype](domain).unsafe_origin_cast[origin_of(self)]()

    def at(mut self, domain: MemoryDomain, index: Int) -> ref[origin_of(self)] Scalar[Self.dtype]:
        """One element of one domain's copy.

        Args:
            domain: Which copy to reach into; each holds the same `length` elements.
            index: Position within that copy, below `length` and unchecked.

        Returns:
            A reference into that copy, so a write through it leaves the other replicas stale.
        """
        return self.replica(domain)[unsafe_offset=index]


@fieldwise_init
struct ShardLocation(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Where a logical index lives: which memory domain, and where inside that domain's shard."""

    var memory_domain: MemoryDomain
    """Which shard holds the element, as a dense index."""
    var local_index: Int
    """Position inside that shard, which is not the logical index it came from."""


struct ShardedArray[dtype: DType]:
    """`length` elements split across the memory domains, each domain owning one contiguous run.

    Parameters:
        dtype: The element type every shard holds.
    """

    var storage: SymmetricAllocation
    """The single mapping behind every shard, one run per domain."""
    var length: Int
    """Elements across all the shards together, not per shard."""
    var segment: Int
    """Elements one shard has room for, by ceiling division, so only the last one can be short."""

    def __init__(out self, *, var storage: SymmetricAllocation, length: Int, segment: Int):
        """Adopts a mapping whose runs are already sized for one shard each.

        Args:
            storage: The mapping to take over, one run per domain.
            length: Elements across all the shards together.
            segment: Elements each shard has room for, which is what fixes where an index lands.
        """
        self.storage = storage^
        self.length = length
        self.segment = segment

    @staticmethod
    def new(topology: Topology, length: Int) raises Error -> Self:
        """`length` elements striped across every memory domain.

        Args:
            topology: Says how many domains the elements are split over.
            length: Elements in total; ceiling division settles the per-domain run.

        Returns:
            An array whose shards start uninitialized; write one before reading it.

        Raises:
            When the per-shard count times the element width would wrap, or the mapping fails.
        """
        var domains = topology.memory_domains_count()
        var segment = ceildiv(length, domains) if domains > 0 else length
        return Self(
            storage=SymmetricAllocation(topology, bytes_for_elements(segment, size_of[Scalar[Self.dtype]]())),
            length=length,
            segment=segment,
        )

    def memory_domains_count(self) -> Int:
        """How many shards the mapping produced.

        Returns:
            The run count the mapping settled on, which is one where the machine has no NUMA API.
        """
        return self.storage.memory_domains_count

    def shard(mut self, domain: MemoryDomain) -> Pointer[Scalar[Self.dtype], origin_of(self)]:
        """One domain's run of elements.

        Args:
            domain: A dense memory-domain index, counted from zero, not an operating-system id.

        Returns:
            A pointer to that run's first element, borrowing the array so it outlives the call.
        """
        return self.storage.on_memory_domain[Self.dtype](domain).unsafe_origin_cast[origin_of(self)]()

    def length_on_memory_domain(self, domain: MemoryDomain) -> Int:
        """How many elements one domain's shard actually holds; the last one may be short.

        Args:
            domain: Which shard to measure, as a dense index.

        Returns:
            Elements really present, down to zero for a domain the split ran out of work before.
        """
        var start = domain.index * self.segment
        if start >= self.length:
            return 0
        return min(self.segment, self.length - start)

    def location_of(self, logical_index: Int) -> ShardLocation:
        """Which domain holds a logical index, and where inside that domain's shard.

        Requires a non-empty array - `segment` is the divisor and is zero when empty.

        Args:
            logical_index: A position in the whole array, not one within a single shard.

        Returns:
            The domain and the offset inside it, which `logical_index_of` turns back again.
        """
        return ShardLocation(
            MemoryDomain(logical_index // self.segment),
            logical_index % self.segment,
        )

    def logical_index_of(self, domain: MemoryDomain, local_index: Int) -> Int:
        """The inverse: a domain and a local offset back to a logical index.

        Args:
            domain: The shard the offset is measured inside, as a dense index.
            local_index: Position within that shard, below `segment`.

        Returns:
            The position in the whole array, whether or not `length` reaches that far.
        """
        return domain.index * self.segment + local_index
