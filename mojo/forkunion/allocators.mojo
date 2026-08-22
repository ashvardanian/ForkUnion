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

from forkunion.errors import ErrorKind, ForkUnionError
from forkunion.library import Library, OutSize
from forkunion.topology import Topology
from forkunion.types import ComputeDomain, MemoryDomain, MemoryDomainId

comptime Bytes = Pointer[Int8, MutUntrackedOrigin]
"""The raw storage a placement allocator hands back."""


struct AllocationResult:
    """Pages placed on one memory domain, released when the last reference goes out of use."""

    var library: Library
    var memory_domain_id: MemoryDomainId
    var pointer: Bytes
    var allocated_bytes: Int
    var bytes_per_page: Int

    def __init__(
        out self,
        library: Library,
        memory_domain_id: MemoryDomainId,
        address: Int,
        allocated_bytes: Int,
        bytes_per_page: Int,
    ):
        self.library = library
        self.memory_domain_id = memory_domain_id
        self.pointer = Bytes(unsafe_from_address=address)
        self.allocated_bytes = allocated_bytes
        self.bytes_per_page = bytes_per_page

    def __deinit__(deinit self):
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
        """
        return self.pointer.unsafe_bitcast[Scalar[dtype]]().unsafe_origin_cast[origin_of(self)]()

    def count_of[dtype: DType](self) -> Int:
        """How many elements of `dtype` the placed pages hold."""
        return self.allocated_bytes // size_of[Scalar[dtype]]()


@fieldwise_init
struct DomainAllocator(Equatable, ImplicitlyCopyable):
    """Places pages on one memory domain, named by the operating system's id for it.

    Holds the id and a loader reference and nothing else, so it is cheap to copy and free of the
    topology that produced the id.
    """

    var library: Library
    var memory_domain_id: MemoryDomainId

    @staticmethod
    def at(library: Library, memory_domain_id: Optional[MemoryDomainId]) -> Optional[Self]:
        """An allocator for a domain the machine actually has, or `None`."""
        if not memory_domain_id:
            return None
        return Self(library, memory_domain_id.value())

    def __eq__(self, other: Self) -> Bool:
        return self.memory_domain_id == other.memory_domain_id

    def allocate(self, bytes: Int) -> Optional[AllocationResult]:
        """Exactly `bytes` on this domain, or `None` where the build or the domain cannot.

        The page size is not reported by this entry point, so the result carries zero for it.
        """
        if bytes == 0:
            return None
        var address = self.library.symbols().allocate_on_domain_id(self.memory_domain_id.identifier, c_size_t(bytes))
        if address == 0:
            return None
        return AllocationResult(self.library, self.memory_domain_id, address, bytes, 0)

    def allocate_at_least(self, minimum_bytes: Int) -> Optional[AllocationResult]:
        """At least `minimum_bytes`, using the largest suitable page size, reporting both."""
        if minimum_bytes == 0:
            return None
        # Stack storage the C side writes through; see `OutSize` for why the origin is `Any`.
        var out = stack_allocation[2, c_size_t]()
        out[unsafe_offset=0] = c_size_t(0)
        out[unsafe_offset=1] = c_size_t(0)
        var address = self.library.symbols().allocate_at_least_on_domain_id(
            self.memory_domain_id.identifier,
            c_size_t(minimum_bytes),
            out.unsafe_origin_cast[MutAnyOrigin](),
            Pointer(to=out[unsafe_offset=1]).unsafe_origin_cast[MutAnyOrigin](),
        )
        if address == 0:
            return None
        return AllocationResult(
            self.library,
            self.memory_domain_id,
            address,
            Int(out[unsafe_offset=0]),
            Int(out[unsafe_offset=1]),
        )

    def allocate_for[dtype: DType](self, count: Int) -> Optional[AllocationResult]:
        """Room for `count` elements of `dtype`."""
        return self.allocate(count * size_of[Scalar[dtype]]())

    def allocate_for_at_least[dtype: DType](self, minimum_count: Int) -> Optional[AllocationResult]:
        """Room for at least `minimum_count` elements of `dtype`, on the largest suitable page."""
        return self.allocate_at_least(minimum_count * size_of[Scalar[dtype]]())


def default_domain_allocator(topology: Topology) -> Optional[DomainAllocator]:
    """An allocator for the machine's first memory domain, which every machine has."""
    return DomainAllocator.at(topology.library, topology.memory_domain_id_at_index(MemoryDomain(0)))


def local_domain_allocator(topology: Topology, domain: ComputeDomain) -> Optional[DomainAllocator]:
    """An allocator for the memory domain nearest a compute domain - "run here, allocate here".

    This is the pairing the index-versus-id distinction exists to protect: `local_memory_of`
    answers a dense index, and only `memory_domain_id_at_index` turns it into an allocator's id.
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
    var base: Bytes
    var stride_bytes: Int
    var memory_domains_count: Int
    var total_bytes: Int
    var bytes_per_page: Int

    def __init__(out self, topology: Topology, bytes_per_domain: Int) raises ForkUnionError:
        if bytes_per_domain == 0:
            raise ForkUnionError(ErrorKind.INVALID_PARAMETER, "a symmetric mapping needs bytes")
        # Stack storage the C side writes through; see `OutSize` for why the origin is `Any`.
        var out = stack_allocation[4, c_size_t]()
        for slot in range(4):
            out[unsafe_offset=slot] = c_size_t(0)
        var address = topology.library.symbols().allocate_symmetric(
            topology.handle,
            c_size_t(bytes_per_domain),
            out.unsafe_origin_cast[MutAnyOrigin](),
            Pointer(to=out[unsafe_offset=1]).unsafe_origin_cast[MutAnyOrigin](),
            Pointer(to=out[unsafe_offset=2]).unsafe_origin_cast[MutAnyOrigin](),
            Pointer(to=out[unsafe_offset=3]).unsafe_origin_cast[MutAnyOrigin](),
        )
        if address == 0:
            raise ForkUnionError(ErrorKind.CREATION_FAILED, "fu_allocate_symmetric")
        self.library = topology.library
        self.base = Bytes(unsafe_from_address=address)
        self.stride_bytes = Int(out[unsafe_offset=0])
        self.memory_domains_count = Int(out[unsafe_offset=1])
        self.total_bytes = Int(out[unsafe_offset=2])
        self.bytes_per_page = Int(out[unsafe_offset=3])

    def __deinit__(deinit self):
        self.library.symbols().free_symmetric(self.base, c_size_t(self.total_bytes))

    def on_memory_domain[dtype: DType](mut self, domain: MemoryDomain) -> Pointer[Scalar[dtype], origin_of(self)]:
        """The run of storage living on one memory domain, seen as the caller's element type."""
        var offset = domain.index * self.stride_bytes
        return (
            Pointer(to=self.base[unsafe_offset=offset])
            .unsafe_bitcast[Scalar[dtype]]()
            .unsafe_origin_cast[origin_of(self)]()
        )


struct ReplicatedArray[dtype: DType]:
    """The same `length` elements held once per memory domain, so every reader is local.

    Write each replica, then let every compute domain read the copy nearest to it. Construction
    answers `Optional`, on the same tier as the allocators: a mapping this machine cannot provide
    is an absence the caller chooses how to handle, not a failure.
    """

    var storage: SymmetricAllocation
    var length: Int

    def __init__(out self, *, var storage: SymmetricAllocation, length: Int):
        self.storage = storage^
        self.length = length

    @staticmethod
    def try_new(topology: Topology, length: Int) -> Optional[Self]:
        """A replica of `length` elements on every memory domain, or `None`."""
        if length > Int.MAX // size_of[Scalar[Self.dtype]]():
            return None
        try:
            return Self(
                storage=SymmetricAllocation(topology, length * size_of[Scalar[Self.dtype]]()),
                length=length,
            )
        except:
            return None

    def memory_domains_count(self) -> Int:
        """How many replicas the mapping produced."""
        return self.storage.memory_domains_count

    def replica(mut self, domain: MemoryDomain) -> Pointer[Scalar[Self.dtype], origin_of(self)]:
        """One domain's copy, all `length` elements of it."""
        return self.storage.on_memory_domain[Self.dtype](domain).unsafe_origin_cast[origin_of(self)]()

    def at(mut self, domain: MemoryDomain, index: Int) -> ref[origin_of(self)] Scalar[Self.dtype]:
        """One element of one domain's copy."""
        return self.replica(domain)[unsafe_offset=index]


@fieldwise_init
struct ShardLocation(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Where a logical index lives: which memory domain, and where inside that domain's shard."""

    var memory_domain: MemoryDomain
    var local_index: Int


struct ShardedArray[dtype: DType]:
    """`length` elements split across the memory domains, each domain owning one contiguous run.

    Construction answers `Optional`, for the same reason `ReplicatedArray` does.
    """

    var storage: SymmetricAllocation
    var length: Int
    var segment: Int

    def __init__(out self, *, var storage: SymmetricAllocation, length: Int, segment: Int):
        self.storage = storage^
        self.length = length
        self.segment = segment

    @staticmethod
    def try_new(topology: Topology, length: Int) -> Optional[Self]:
        """`length` elements striped across every memory domain, or `None`."""
        var domains = topology.memory_domains_count()
        var segment = ceildiv(length, domains) if domains > 0 else length
        if segment > Int.MAX // size_of[Scalar[Self.dtype]]():
            return None
        try:
            return Self(
                storage=SymmetricAllocation(topology, segment * size_of[Scalar[Self.dtype]]()),
                length=length,
                segment=segment,
            )
        except:
            return None

    def memory_domains_count(self) -> Int:
        """How many shards the mapping produced."""
        return self.storage.memory_domains_count

    def shard(mut self, domain: MemoryDomain) -> Pointer[Scalar[Self.dtype], origin_of(self)]:
        """One domain's run of elements."""
        return self.storage.on_memory_domain[Self.dtype](domain).unsafe_origin_cast[origin_of(self)]()

    def length_on_memory_domain(self, domain: MemoryDomain) -> Int:
        """How many elements one domain's shard actually holds; the last one may be short."""
        var start = domain.index * self.segment
        if start >= self.length:
            return 0
        return min(self.segment, self.length - start)

    def location_of(self, logical_index: Int) -> ShardLocation:
        """Which domain holds a logical index, and where inside that domain's shard.

        Requires a non-empty array - `segment` is the divisor and is zero when empty.
        """
        return ShardLocation(
            MemoryDomain(logical_index // self.segment),
            logical_index % self.segment,
        )

    def logical_index_of(self, domain: MemoryDomain, local_index: Int) -> Int:
        """The inverse: a domain and a local offset back to a logical index."""
        return domain.index * self.segment + local_index
