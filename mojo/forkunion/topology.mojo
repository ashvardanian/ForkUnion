"""This machine as ForkUnion sees it: cores, compute domains, memory domains, and capabilities.

A topology is read once and then only queried, so it is cheap to keep for the life of a program.
Pools never retain it - the C API reads it during the spawn and not after - so it may be released
as soon as the last spawn returns.
"""

from std.ffi import c_int, c_size_t

from forkunion.errors import ErrorKind, ForkUnionError
from forkunion.library import Handle, Library
from forkunion.types import (
    Capabilities,
    ComputeDomain,
    MemoryDomain,
    MemoryDomainId,
)

comptime _NAME_BUFFER_BYTES = 256
"""Enough for every capability name the C side can emit; the list is truncated to fit."""


struct Topology:
    """The hardware view. Released when the last reference to it goes out of use."""

    var library: Library
    var handle: Handle

    def __init__(out self, library: Library) raises ForkUnionError:
        self.library = library
        # The C API answers NULL when the harvest fails, and Mojo's `Pointer` is non-null by
        # design, so the address is checked before it becomes one.
        var address = library.symbols().topology_new()
        if address == 0:
            raise ForkUnionError(ErrorKind.CREATION_FAILED, "fu_topology_new")
        self.handle = Handle(unsafe_from_address=address)

    def __deinit__(deinit self):
        self.library.symbols().topology_delete(self.handle)

    @always_inline
    def _count(self, symbol: def(Handle) thin abi("C") -> c_size_t) -> Int:
        return Int(symbol(self.handle))

    @always_inline
    def _count_in(
        self,
        symbol: def(Handle, c_size_t) thin abi("C") -> c_size_t,
        index: Int,
    ) -> Int:
        return Int(symbol(self.handle, c_size_t(index)))

    # region Compute

    def logical_cores_count(self) -> Int:
        """Every hardware thread this machine exposes."""
        return self._count(self.library.symbols().logical_cores_count)

    def logical_cores_count_in(self, domain: ComputeDomain) -> Int:
        """The hardware threads belonging to one compute domain."""
        return self._count_in(self.library.symbols().logical_cores_count_in, domain.index)

    def compute_domains_count(self) -> Int:
        """Same-QoS core clusters, each local to one memory domain."""
        return self._count(self.library.symbols().compute_domains_count)

    def compute_levels_count(self) -> Int:
        """Distinct core classes; may be fewer than the domains, since domains can share a level."""
        return self._count(self.library.symbols().compute_levels_count)

    def compute_level_in(self, domain: ComputeDomain) -> Int:
        """Which core class a compute domain belongs to, counted from the fastest."""
        return self._count_in(self.library.symbols().compute_level_in, domain.index)

    def compute_capacity_in(self, domain: ComputeDomain) -> Int:
        """A relative throughput weight for one compute domain."""
        return self._count_in(self.library.symbols().compute_capacity_in, domain.index)

    def compute_cache_bytes_in(self, domain: ComputeDomain) -> Int:
        """The last-level cache one compute domain can reach."""
        return self._count_in(self.library.symbols().compute_cache_bytes_in, domain.index)

    # endregion Compute

    # region Memory

    def memory_domains_count(self) -> Int:
        """Banks of memory with their own capacity and access cost."""
        return self._count(self.library.symbols().memory_domains_count)

    def local_memory_of(self, domain: ComputeDomain) -> MemoryDomain:
        """The memory domain nearest a compute domain, as a dense index.

        This is an index, not an id. Pass it through `memory_domain_id_at_index` before handing it
        to an allocator, which keys off the operating system's own numbering.
        """
        return MemoryDomain(self._count_in(self.library.symbols().local_memory_of, domain.index))

    def memory_domain_id_at_index(self, domain: MemoryDomain) -> Optional[MemoryDomainId]:
        """Resolves a dense memory-domain index to the OS id the allocators take.

        Answers `None` where the C API reports -1, which is an index this machine does not have.
        """
        var identifier = self.library.symbols().memory_domain_id_at_index(self.handle, c_size_t(domain.index))
        if identifier < 0:
            return None
        return MemoryDomainId(identifier)

    def volume_ram(self) -> Int:
        """Bytes of RAM installed, regardless of page size."""
        return self._count(self.library.symbols().volume_ram)

    def volume_ram_in(self, domain: MemoryDomain) -> Int:
        """Bytes of RAM in one memory domain."""
        return self._count_in(self.library.symbols().volume_ram_in, domain.index)

    def volume_huge_pages(self) -> Int:
        """Bytes backed by free huge pages across every memory domain."""
        return self._count(self.library.symbols().volume_huge_pages)

    def volume_huge_pages_in(self, domain: MemoryDomain) -> Int:
        """Bytes backed by free huge pages in one memory domain."""
        return self._count_in(self.library.symbols().volume_huge_pages_in, domain.index)

    def huge_pages_count(self) -> Int:
        """Free huge pages of any size across every memory domain."""
        return self._count(self.library.symbols().huge_pages_count)

    def huge_pages_count_in(self, domain: MemoryDomain) -> Int:
        """Free huge pages of any size in one memory domain."""
        return self._count_in(self.library.symbols().huge_pages_count_in, domain.index)

    # endregion Memory


def version(library: Library) -> Tuple[Int, Int, Int]:
    """The loaded library's own version."""
    return library.version()


def comptime_capabilities(library: Library) -> Capabilities:
    """What this build of the library was compiled to be able to do."""
    return Capabilities(UInt32(Int(library.symbols().comptime_capabilities())))


def runtime_capabilities(library: Library) -> Capabilities:
    """What this machine actually offers, which is not implied by the above in either direction."""
    return Capabilities(UInt32(Int(library.symbols().runtime_capabilities())))


def name_capabilities(library: Library, capabilities: Capabilities) -> Optional[String]:
    """Renders a mask as a comma-separated name list such as "arm64_yield,arm64_wfet".

    Answers `None` where the C side wrote nothing, which is how it reports that it could not
    format the mask at all.
    """
    var buffer = List[UInt8](length=_NAME_BUFFER_BYTES, fill=0)
    var written = Int(
        library.symbols().name_capabilities(
            c_int(Int(capabilities.bits)),
            buffer.unsafe_ptr().unsafe_bitcast[Int8]().unsafe_origin_cast[MutAnyOrigin](),
            c_size_t(_NAME_BUFFER_BYTES),
        )
    )
    if written == 0:
        return None
    # The C side null-terminates and truncates to fit, so clamp before trusting the count, and
    # stop at the terminator in case it wrote one early.
    var length = min(written, _NAME_BUFFER_BYTES)
    for index in range(length):
        if buffer[index] == 0:
            length = index
            break
    return String(
        StringSlice(unsafe_from_utf8=Span[UInt8, origin_of(buffer)](unsafe_ptr=buffer.unsafe_ptr(), length=length))
    )


def comptime_capabilities_string(library: Library) -> Optional[String]:
    """The compiled-in capabilities, named."""
    return name_capabilities(library, comptime_capabilities(library))


def runtime_capabilities_string(library: Library) -> Optional[String]:
    """The capabilities this machine offers, named."""
    return name_capabilities(library, runtime_capabilities(library))
