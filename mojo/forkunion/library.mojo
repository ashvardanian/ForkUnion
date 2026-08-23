"""The loaded shared object and the one place a raw symbol address is ever touched.

ForkUnion is reached by `dlopen` rather than by linking, which is what lets a consumer build
without a single linker flag. Mojo's `@extern` cannot serve here: it resolves at link time, so a
symbol from a library that is only opened at runtime is an undefined reference under both the
JIT and `lld`, and using it would put `-Xlinker -lforkunion` in every consumer's build.

Every entry point is resolved once, when the library loads, into a typed `thin abi("C")` function
pointer. `OwnedDLHandle.get_function` would run `dlsym` on every call and parameterises only on
the return type, leaving arguments unchecked; a dispatch on the hot path cannot afford either.
"""

from std.ffi import OwnedDLHandle, __fn_type_is_cabi, c_int, c_size_t
from std.memory import ArcPointer
from std.sys.info import CompilationTarget

from forkunion.allocators import (
    AllocateAtLeastOnDomain,
    AllocateOnDomain,
    AllocateSymmetric,
    FreeOnDomain,
    FreeSymmetric,
)
from forkunion.scheduling import (
    FabricDelete,
    FabricEdge,
    FabricHarvest,
    FabricLevelIn,
    FabricLevelsCount,
    FabricNew,
    PoolCapabilities,
    PoolCount,
    PoolCountIn,
    PoolDelete,
    PoolExclusivity,
    PoolForN,
    PoolForSlices,
    PoolForThreads,
    PoolIsComplete,
    PoolLocateThreadIn,
    PoolNew,
    PoolSleep,
    PoolSpawn,
    PoolSpawnOn,
    PoolTerminate,
    PoolUnsafeForThreads,
    PoolUnsafeJoin,
)
from forkunion.topology import (
    Capabilities32,
    MemoryDomainIdAt,
    NameCapabilities,
    TopologyCount,
    TopologyCountIn,
    TopologyDelete,
    TopologyNew,
)
from forkunion.types import Error, ErrorKind

comptime LIBRARY_FILE = StaticString("libforkunion.dylib" if CompilationTarget.is_macos() else "libforkunion.so")
"""The loader already searches the environment's library directory, because `mojo build` bakes it
into the binary's runpath, so an unqualified name is enough for a library installed there."""

comptime BINDING_MAJOR = 3
"""The major version this binding was written against; a mismatch is refused at load."""

# region Signatures

comptime VersionPart = def() thin abi("C") -> c_int

# endregion Signatures


@always_inline
def _resolve[
    Signature: TrivialRegisterPassable
](ref handle: OwnedDLHandle, name: StaticString) raises Error -> Signature:
    """One `dlsym`, checked once, typed from here on.

    `get_symbol` answers `Optional` where `get_function` aborts the process, which is what lets a
    stale library name the symbol it is missing instead of dying without a message.
    """
    comptime assert __fn_type_is_cabi[Signature](), 'the signature must carry abi("C")'
    var address = handle.get_symbol[NoneType](name)
    if not address:
        raise Error(ErrorKind.SYMBOL_MISSING, name)
    return Pointer(to=address.value()).unsafe_bitcast[Signature]()[]


struct Symbols:
    """Every entry point the C ABI exports, resolved once when the library loads.

    Answers are never cached here, only addresses. A pool re-queries its own exclusivity and
    capabilities on every call, because `fu_pool_terminate` followed by a fresh spawn can change
    both, and the C test battery checks exactly that.
    """

    var version_major: VersionPart
    var version_minor: VersionPart
    var version_patch: VersionPart

    var comptime_capabilities: Capabilities32
    var runtime_capabilities: Capabilities32
    var name_capabilities: NameCapabilities

    var topology_new: TopologyNew
    var topology_delete: TopologyDelete
    var logical_cores_count: TopologyCount
    var logical_cores_count_in: TopologyCountIn
    var compute_domains_count: TopologyCount
    var compute_levels_count: TopologyCount
    var compute_level_in: TopologyCountIn
    var compute_capacity_in: TopologyCountIn
    var compute_cache_bytes_in: TopologyCountIn
    var memory_domains_count: TopologyCount
    var memory_domain_id_at_index: MemoryDomainIdAt
    var local_memory_of: TopologyCountIn
    var volume_ram: TopologyCount
    var volume_ram_in: TopologyCountIn
    var volume_huge_pages: TopologyCount
    var volume_huge_pages_in: TopologyCountIn
    var huge_pages_count: TopologyCount
    var huge_pages_count_in: TopologyCountIn

    var allocate_on_domain_id: AllocateOnDomain
    var allocate_at_least_on_domain_id: AllocateAtLeastOnDomain
    var free_on_domain_id: FreeOnDomain
    var allocate_symmetric: AllocateSymmetric
    var free_symmetric: FreeSymmetric

    var pool_new: PoolNew
    var pool_delete: PoolDelete
    var pool_capabilities: PoolCapabilities
    var pool_spawn: PoolSpawn
    var pool_spawn_on: PoolSpawnOn
    var pool_caller_exclusivity: PoolExclusivity
    var pool_compute_domains_count: PoolCount
    var pool_threads_count: PoolCount
    var pool_threads_count_in: PoolCountIn
    var pool_locate_thread_in: PoolLocateThreadIn
    var pool_sleep: PoolSleep
    var pool_terminate: PoolTerminate

    var fabric_new: FabricNew
    var fabric_delete: FabricDelete
    var fabric_harvest: FabricHarvest
    var fabric_memory_latency: FabricEdge
    var fabric_memory_bandwidth: FabricEdge
    var fabric_memory_distance: FabricEdge
    var fabric_memory_level_in: FabricLevelIn
    var fabric_memory_levels_count: FabricLevelsCount

    var pool_for_threads: PoolForThreads
    var pool_for_n: PoolForN
    var pool_for_n_dynamic: PoolForN
    var pool_for_slices: PoolForSlices
    var pool_unsafe_for_threads: PoolUnsafeForThreads
    var pool_is_complete: PoolIsComplete
    var pool_unsafe_join: PoolUnsafeJoin

    def __init__(out self, ref handle: OwnedDLHandle) raises Error:
        self.version_major = _resolve[VersionPart](handle, "fu_version_major")
        self.version_minor = _resolve[VersionPart](handle, "fu_version_minor")
        self.version_patch = _resolve[VersionPart](handle, "fu_version_patch")

        self.comptime_capabilities = _resolve[Capabilities32](handle, "fu_comptime_capabilities")
        self.runtime_capabilities = _resolve[Capabilities32](handle, "fu_runtime_capabilities")
        self.name_capabilities = _resolve[NameCapabilities](handle, "fu_name_capabilities")

        self.topology_new = _resolve[TopologyNew](handle, "fu_topology_new")
        self.topology_delete = _resolve[TopologyDelete](handle, "fu_topology_delete")
        self.logical_cores_count = _resolve[TopologyCount](handle, "fu_logical_cores_count")
        self.logical_cores_count_in = _resolve[TopologyCountIn](handle, "fu_logical_cores_count_in")
        self.compute_domains_count = _resolve[TopologyCount](handle, "fu_compute_domains_count")
        self.compute_levels_count = _resolve[TopologyCount](handle, "fu_compute_levels_count")
        self.compute_level_in = _resolve[TopologyCountIn](handle, "fu_compute_level_in")
        self.compute_capacity_in = _resolve[TopologyCountIn](handle, "fu_compute_capacity_in")
        self.compute_cache_bytes_in = _resolve[TopologyCountIn](handle, "fu_compute_cache_bytes_in")
        self.memory_domains_count = _resolve[TopologyCount](handle, "fu_memory_domains_count")
        self.memory_domain_id_at_index = _resolve[MemoryDomainIdAt](handle, "fu_memory_domain_id_at_index")
        self.local_memory_of = _resolve[TopologyCountIn](handle, "fu_local_memory_of")
        self.volume_ram = _resolve[TopologyCount](handle, "fu_volume_ram")
        self.volume_ram_in = _resolve[TopologyCountIn](handle, "fu_volume_ram_in")
        self.volume_huge_pages = _resolve[TopologyCount](handle, "fu_volume_huge_pages")
        self.volume_huge_pages_in = _resolve[TopologyCountIn](handle, "fu_volume_huge_pages_in")
        self.huge_pages_count = _resolve[TopologyCount](handle, "fu_huge_pages_count")
        self.huge_pages_count_in = _resolve[TopologyCountIn](handle, "fu_huge_pages_count_in")

        self.allocate_on_domain_id = _resolve[AllocateOnDomain](handle, "fu_allocate_on_domain_id")
        self.allocate_at_least_on_domain_id = _resolve[AllocateAtLeastOnDomain](
            handle, "fu_allocate_at_least_on_domain_id"
        )
        self.free_on_domain_id = _resolve[FreeOnDomain](handle, "fu_free_on_domain_id")
        self.allocate_symmetric = _resolve[AllocateSymmetric](handle, "fu_allocate_symmetric")
        self.free_symmetric = _resolve[FreeSymmetric](handle, "fu_free_symmetric")

        self.pool_new = _resolve[PoolNew](handle, "fu_pool_new")
        self.pool_delete = _resolve[PoolDelete](handle, "fu_pool_delete")
        self.pool_capabilities = _resolve[PoolCapabilities](handle, "fu_pool_capabilities")
        self.pool_spawn = _resolve[PoolSpawn](handle, "fu_pool_spawn")
        self.pool_spawn_on = _resolve[PoolSpawnOn](handle, "fu_pool_spawn_on")
        self.pool_caller_exclusivity = _resolve[PoolExclusivity](handle, "fu_pool_caller_exclusivity")
        self.pool_compute_domains_count = _resolve[PoolCount](handle, "fu_pool_compute_domains_count")
        self.pool_threads_count = _resolve[PoolCount](handle, "fu_pool_threads_count")
        self.pool_threads_count_in = _resolve[PoolCountIn](handle, "fu_pool_threads_count_in")
        self.pool_locate_thread_in = _resolve[PoolLocateThreadIn](handle, "fu_pool_locate_thread_in")
        self.pool_sleep = _resolve[PoolSleep](handle, "fu_pool_sleep")
        self.pool_terminate = _resolve[PoolTerminate](handle, "fu_pool_terminate")

        self.fabric_new = _resolve[FabricNew](handle, "fu_fabric_new")
        self.fabric_delete = _resolve[FabricDelete](handle, "fu_fabric_delete")
        self.fabric_harvest = _resolve[FabricHarvest](handle, "fu_fabric_harvest")
        self.fabric_memory_latency = _resolve[FabricEdge](handle, "fu_fabric_memory_latency")
        self.fabric_memory_bandwidth = _resolve[FabricEdge](handle, "fu_fabric_memory_bandwidth")
        self.fabric_memory_distance = _resolve[FabricEdge](handle, "fu_fabric_memory_distance")
        self.fabric_memory_level_in = _resolve[FabricLevelIn](handle, "fu_fabric_memory_level_in")
        self.fabric_memory_levels_count = _resolve[FabricLevelsCount](handle, "fu_fabric_memory_levels_count")

        self.pool_for_threads = _resolve[PoolForThreads](handle, "fu_pool_for_threads")
        self.pool_for_n = _resolve[PoolForN](handle, "fu_pool_for_n")
        self.pool_for_n_dynamic = _resolve[PoolForN](handle, "fu_pool_for_n_dynamic")
        self.pool_for_slices = _resolve[PoolForSlices](handle, "fu_pool_for_slices")
        self.pool_unsafe_for_threads = _resolve[PoolUnsafeForThreads](handle, "fu_pool_unsafe_for_threads")
        self.pool_is_complete = _resolve[PoolIsComplete](handle, "fu_pool_is_complete")
        self.pool_unsafe_join = _resolve[PoolUnsafeJoin](handle, "fu_pool_unsafe_join")


struct _Loaded:
    """The handle and its symbols, kept together so neither outlives the other."""

    var handle: OwnedDLHandle
    var symbols: Symbols

    def __init__(out self) raises Error:
        try:
            self.handle = OwnedDLHandle(String(LIBRARY_FILE))
        except:
            raise Error(ErrorKind.LIBRARY_MISSING, LIBRARY_FILE)
        self.symbols = Symbols(self.handle)
        if Int(self.symbols.version_major()) != BINDING_MAJOR:
            raise Error(
                ErrorKind.LIBRARY_MISSING,
                "major version differs from the binding",
            )


struct Library(ImplicitlyCopyable):
    """A reference to the loaded core, cheap to copy and shared by everything it creates.

    Every handle carries one, which is what keeps the shared object mapped for as long as anything
    can still call into it. It is a loader reference, not a topology reference - the C API is
    explicit that a pool never retains the topology it spawned from.
    """

    var shared: ArcPointer[_Loaded]

    def __init__(out self) raises Error:
        self.shared = ArcPointer(_Loaded())

    @always_inline
    def symbols(self) -> ref[origin_of(self.shared[].symbols)] Symbols:
        return self.shared[].symbols

    def version(self) -> Tuple[Int, Int, Int]:
        """The loaded library's own version, which is also the cheapest check that loading worked."""
        ref symbols = self.symbols()
        return (
            Int(symbols.version_major()),
            Int(symbols.version_minor()),
            Int(symbols.version_patch()),
        )
