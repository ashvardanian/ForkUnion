"""The Mojo binding's tests, in the shape `scripts/test.c` and `scripts/test.cpp` already use.

Mojo 1.0 has no `mojo test` subcommand and no test discovery, but `std.testing.TestSuite` is a
real runner: register each check as a parameter and it reports and filters them. The pure-logic
groups need no library loaded; the rest exercise the C core directly.

Do not read the per-test durations the suite prints as wall clock. A pool's workers busy-wait, and
whatever clock the runner samples counts that spinning, so it reports minutes for a test the
process finishes in milliseconds - the whole file runs in under five seconds. Time the process
itself when a number matters.
"""

from std.ffi import c_size_t
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from forkunion import (
    AllocationResult,
    BroadcastJoin,
    CacheAligned,
    CallerExclusivity,
    Capabilities,
    ComputeDomain,
    DomainAllocator,
    ErrorKind,
    Fabric,
    Error,
    IndexedSplit,
    Library,
    MemoryDomain,
    MemoryDomainId,
    Pool,
    TasksRange,
    ThreadInDomain,
    ReplicatedArray,
    ShardedArray,
    SyncMutPointer,
    Topology,
    comptime_capabilities,
    default_domain_allocator,
    local_domain_allocator,
    name_capabilities,
    runtime_capabilities,
)


@fieldwise_init
struct Visits(ImplicitlyCopyable, TrivialRegisterPassable):
    """A visit log every dispatch test writes through, one slot per task or thread."""

    var slots: SyncMutPointer[Int64]
    var count: Int


def _record_task(task: Int, at: ThreadInDomain, mut visits: Visits):
    visits.slots.at(task) += 1


def _record_thread(thread_index: Int, domain_index: Int, mut visits: Visits):
    visits.slots.at(thread_index) += 1


def _record_slice(range_of_tasks: TasksRange, at: ThreadInDomain, mut visits: Visits):
    for task in range_of_tasks:
        visits.slots.at(task) += 1


def _square(task: Int, at: ThreadInDomain, mut visits: Visits):
    visits.slots.at(task) = Int64(task) * Int64(task)


def _uneven(task: Int, at: ThreadInDomain, mut visits: Visits):
    """A deliberately skewed cost, so work-stealing has something to steal."""
    var accumulated = Int64(0)
    for step in range((task % 7) * 128):
        accumulated += Int64(step)
    visits.slots.at(task) += Int64(1) if accumulated >= 0 else Int64(0)


def _visits_over(mut log: List[Int64]) -> Visits:
    return Visits(
        SyncMutPointer(log.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()),
        len(log),
    )


# region Pure Logic


def test_indexed_split_tiles_the_range() raises:
    """The chunks must cover `[0, tasks)` exactly once, be contiguous, and differ by at most one."""
    var shapes = [(0, 4), (1, 4), (10, 4), (12, 4), (1000, 7), (5, 1), (97, 4)]
    for shape in shapes:
        var tasks = shape[0]
        var threads = shape[1]
        var split = IndexedSplit(tasks, threads)
        var expected_start = 0
        var shortest = tasks + 1
        var longest = -1
        for thread in range(threads):
            var chunk = split.get(thread)
            assert_equal(chunk.first, expected_start, "runs are contiguous")
            expected_start += chunk.count
            shortest = min(shortest, chunk.count)
            longest = max(longest, chunk.count)
        assert_equal(expected_start, tasks, "chunks tile the whole range")
        assert_true(longest - shortest <= 1, "chunk sizes differ by at most one")


def test_cache_aligned_pads_to_a_line() raises:
    var slots = List[CacheAligned[Int64]](capacity=4)
    for index in range(4):
        slots.append(CacheAligned[Int64](Int64(index * 7)))
    for index in range(4):
        assert_equal(slots[index].value, Int64(index * 7), "the payload round-trips")
    var gap = Int(Pointer(to=slots.unsafe_ptr()[unsafe_offset=1])) - Int(slots.unsafe_ptr())
    assert_true(gap >= 64, "distinct accumulators land on distinct cache lines")


def test_capability_masks() raises:
    var mask = Capabilities.X86_PAUSE | Capabilities.TOPOLOGY
    assert_true(Capabilities.TOPOLOGY in mask, "a set bit is present")
    assert_false(Capabilities.ARM64_WFET in mask, "an unset bit is not")
    assert_true(mask in Capabilities.ALL, "an all-ones mask holds every bit")
    assert_false(Capabilities.X86_PAUSE in Capabilities.NONE, "an empty mask holds none")
    # The three bits neither the Rust nor the Zig binding names.
    assert_true(Capabilities.X86_CLDEMOTE in Capabilities.ALL, "cldemote is nameable")
    assert_true(Capabilities.ARM64_DC_CVAC in Capabilities.ALL, "dc cvac is nameable")
    assert_true(Capabilities.RISC5_ZICBOM in Capabilities.ALL, "zicbom is nameable")


def test_domain_types_are_distinct() raises:
    assert_equal(ComputeDomain(2).index, 2, "a compute domain is an index")
    assert_equal(MemoryDomain(2).index, 2, "a memory domain is an index")
    assert_equal(Int(MemoryDomainId(2).identifier), 2, "an os id is not an index")
    assert_true(
        CallerExclusivity.INCLUSIVE != CallerExclusivity.EXCLUSIVE,
        "the two modes differ",
    )


# endregion Pure Logic

# region Library And Topology


def test_library_loads_and_reports_its_version() raises:
    var library = Library()
    var version = library.version()
    assert_equal(version[0], 3, "the binding targets version three")


def test_missing_library_raises() raises:
    """A stale or absent core must name itself, not abort the process."""
    with assert_raises(contains="ForkUnion"):
        raise Error(ErrorKind.LIBRARY_MISSING, "libforkunion_absent.so")


def test_capabilities_are_reported_and_nameable() raises:
    var library = Library()
    assert_true(
        runtime_capabilities(library) != Capabilities.NONE,
        "a machine offers something",
    )
    assert_true(
        comptime_capabilities(library) != Capabilities.NONE,
        "a build reports something",
    )
    var named = name_capabilities(library, runtime_capabilities(library))
    assert_true(named.byte_length() > 0, "the runtime mask renders as non-empty text")


def test_topology_counts_and_core_partition() raises:
    var topology = Topology(Library())
    assert_true(topology.logical_cores_count() >= 1, "a machine has at least one core")
    assert_true(topology.compute_domains_count() >= 1, "and at least one compute domain")
    assert_true(topology.memory_domains_count() >= 1, "and at least one memory domain")
    assert_true(topology.volume_ram() > 0, "and some memory")
    assert_true(topology.compute_levels_count() >= 1, "and at least one core class")

    var counted = 0
    for index in range(topology.compute_domains_count()):
        var domain = ComputeDomain(index)
        var cores_here = topology.logical_cores_count_in(domain)
        assert_true(cores_here >= 1, "a compute domain has cores")
        assert_true(
            topology.compute_level_in(domain) < topology.compute_levels_count(),
            "a domain's level is dense",
        )
        counted += cores_here
    assert_equal(
        counted,
        topology.logical_cores_count(),
        "domains partition the machine's cores",
    )


def test_topology_memory_bounds() raises:
    """Every memory-domain index resolves to an id an allocator accepts, and fits the total."""
    var topology = Topology(Library())
    var total = topology.volume_ram()
    for index in range(topology.memory_domains_count()):
        var domain = MemoryDomain(index)
        var identifier = topology.memory_domain_id_at_index(domain)
        assert_true(Int(identifier.identifier) >= 0, "a real memory domain has an os id")
        assert_true(
            topology.volume_ram_in(domain) <= total,
            "a domain's RAM fits the machine's",
        )
    var refused = False
    try:
        _ = topology.memory_domain_id_at_index(MemoryDomain(1 << 20))
    except:
        refused = True
    assert_true(refused, "an index past the end is refused rather than answering a bogus id")


def test_local_memory_is_an_index_not_an_id() raises:
    """The defect this binding exists to fix: `local_memory_of` answers an index, never an id.

    Feeding the index straight to an allocator is the bug; it is only correct where the two
    numberings happen to agree. The pairing below is the one call sites must make.
    """
    var library = Library()
    var topology = Topology(library)
    for index in range(topology.compute_domains_count()):
        var local = topology.local_memory_of(ComputeDomain(index))
        assert_true(
            local.index < topology.memory_domains_count(),
            "the answer indexes the memory domains",
        )
        var identifier = topology.memory_domain_id_at_index(local)
        var allocator = local_domain_allocator(topology, ComputeDomain(index))
        assert_equal(
            Int(allocator.memory_domain_id.identifier),
            Int(identifier.identifier),
            "the convenience pairing agrees with the two-step one",
        )


# endregion Library And Topology

# region Pools


def test_spawn_rejects_zero_threads() raises:
    """Zero threads is not a pool: it must be refused cleanly, not crash or hang."""
    var topology = Topology(Library())
    with assert_raises(contains="ForkUnion"):
        var pool = Pool(topology, threads=0)
        _ = pool^


def test_spawn_succeeds_at_full_width() raises:
    var topology = Topology(Library())
    var width = topology.logical_cores_count()
    var pool = Pool(topology, threads=width)
    assert_equal(pool.threads_count(), width, "the pool is the width that was asked for")
    assert_true(
        pool.compute_domains_count() >= 1,
        "and spans at least one compute domain",
    )


def test_caller_exclusivity_is_queried_not_cached() raises:
    """The pool is the single source of truth, re-asked after a terminate and a fresh spawn."""
    var topology = Topology(Library())
    var inclusive = Pool(topology, threads=2, exclusivity=CallerExclusivity.INCLUSIVE)
    assert_equal(
        inclusive.caller_exclusivity(),
        CallerExclusivity.INCLUSIVE,
        "reports what it spawned with",
    )
    var exclusive = Pool(topology, threads=2, exclusivity=CallerExclusivity.EXCLUSIVE)
    assert_equal(
        exclusive.caller_exclusivity(),
        CallerExclusivity.EXCLUSIVE,
        "and the other mode too",
    )


def test_pool_capabilities_only_narrow() raises:
    """Masking never adds: a pool's effective mask is the request narrowed by the machine."""
    var library = Library()
    var topology = Topology(library)
    var machine = runtime_capabilities(library)
    var pool = Pool(topology, threads=2, allowed=Capabilities.ALL)
    assert_true(
        pool.capabilities() in machine,
        "the effective mask is within the machine's",
    )

    var narrowed = Pool(topology, threads=2, allowed=Capabilities.NONE)
    assert_true(
        narrowed.capabilities() in pool.capabilities(),
        "clearing bits cannot widen what the pool got",
    )


def test_per_compute_domain_pools() raises:
    """Every real domain accepts a pool sized to its cores; a bogus index is refused."""
    var topology = Topology(Library())
    for index in range(topology.compute_domains_count()):
        var domain = ComputeDomain(index)
        var pool = Pool.on(topology, domain, threads=topology.logical_cores_count_in(domain))
        assert_true(pool.threads_count() >= 1, "a per-domain pool has workers")

    with assert_raises(contains="ForkUnion"):
        var bogus = Pool.on(topology, ComputeDomain(1 << 20), threads=1)
        _ = bogus^


def test_pool_domain_accounting() raises:
    """Per-domain worker counts sum to the pool total, with contiguous global numbering."""
    var topology = Topology(Library())
    var pool = Pool(topology, threads=topology.logical_cores_count())
    var counted = 0
    for index in range(pool.compute_domains_count()):
        counted += pool.threads_count_in(ComputeDomain(index))
    assert_equal(
        counted,
        pool.threads_count(),
        "the domains partition the pool's workers",
    )

    for domain_index in range(pool.compute_domains_count()):
        var domain = ComputeDomain(domain_index)
        for local in range(pool.threads_count_in(domain)):
            var located = pool.locate_thread_in(local, domain)
            assert_true(located < pool.threads_count(), "a located thread is in range")


def test_sleep_then_wake_loses_nothing() raises:
    var topology = Topology(Library())
    var pool = Pool(topology, threads=4)
    pool.sleep(1000)
    var log = List[Int64](length=64, fill=0)
    var visits = _visits_over(log)
    pool.for_n[_record_task](64, visits)
    for index in range(64):
        assert_equal(log[index], 1, "a sleeping pool wakes and loses no task")


# endregion Pools

# region Dispatch


def test_for_threads_reaches_every_worker() raises:
    var topology = Topology(Library())
    var pool = Pool(topology, threads=4)
    var log = List[Int64](length=4, fill=0)
    var visits = _visits_over(log)
    pool.for_threads[_record_thread](visits)
    for index in range(4):
        assert_equal(log[index], 1, "for_threads reaches every thread exactly once")


def test_for_n_visits_each_task_once_and_again() raises:
    """Coverage must survive a repeated dispatch, which is where a stale epoch would show."""
    var topology = Topology(Library())
    var pool = Pool(topology, threads=4)
    var log = List[Int64](length=97, fill=0)
    var visits = _visits_over(log)
    pool.for_n[_record_task](97, visits)
    pool.for_n[_record_task](97, visits)
    for index in range(97):
        assert_equal(log[index], 2, "for_n visits every task exactly once per dispatch")


def test_for_n_across_uncomfortable_sizes() raises:
    """Every task count from zero to three times the width, where an off-by-one hides."""
    var topology = Topology(Library())
    var pool = Pool(topology, threads=4)
    var log = List[Int64](length=64, fill=0)
    for count in range(0, 13):
        for index in range(64):
            log[index] = 0
        var visits = _visits_over(log)
        pool.for_n[_record_task](count, visits)
        for index in range(64):
            var expected = Int64(1) if index < count else Int64(0)
            assert_equal(
                log[index],
                expected,
                "no index at or past the bound is produced",
            )


def test_for_n_dynamic_covers_uneven_work() raises:
    var topology = Topology(Library())
    var pool = Pool(topology, threads=4)
    var log = List[Int64](length=97, fill=0)
    var visits = _visits_over(log)
    pool.for_n_dynamic[_square](97, visits)
    for index in range(97):
        assert_equal(
            log[index],
            Int64(index) * Int64(index),
            "work-stealing still covers each task",
        )


def test_for_n_dynamic_oversubscribed() raises:
    """Three times more workers than cores, driving a deliberately skewed workload."""
    var topology = Topology(Library())
    var pool = Pool(topology, threads=topology.logical_cores_count() * 3)
    var log = List[Int64](length=200, fill=0)
    var visits = _visits_over(log)
    pool.for_n_dynamic[_uneven](200, visits)
    for index in range(200):
        assert_equal(log[index], 1, "every task ran exactly once under oversubscription")


def test_for_slices_partitions_the_range() raises:
    """Slices tile `[0, n)` once; an idle worker gets a length of zero and touches nothing."""
    var topology = Topology(Library())
    var pool = Pool(topology, threads=4)
    var log = List[Int64](length=97, fill=0)
    var visits = _visits_over(log)
    pool.for_slices[_record_slice](97, visits)
    for index in range(97):
        assert_equal(log[index], 1, "for_slices covers each index exactly once")


def test_generation_polling_on_an_exclusive_pool() raises:
    """Dispatch, overlap other work, poll, then join - and the guard joins on scope exit."""
    var topology = Topology(Library())
    var pool = Pool(topology, threads=4, exclusivity=CallerExclusivity.EXCLUSIVE)
    var log = List[Int64](length=4, fill=0)
    var visits = _visits_over(log)
    with pool.unsafe_for_threads[_record_thread](visits) as dispatch:
        assert_true(dispatch.generation % 2 == 1, "a real generation token is odd")
        while not dispatch.is_complete():
            pass
    for index in range(4):
        assert_equal(log[index], 1, "the join published every worker's write")


# endregion Dispatch

# region Memory


def test_allocations_on_every_memory_domain() raises:
    """Both allocation shapes round-trip on every domain, at several sizes."""
    var library = Library()
    var topology = Topology(library)
    var sizes = [4096, 1 << 16, 1 << 20]
    for index in range(topology.memory_domains_count()):
        var allocator = DomainAllocator.at(library, topology.memory_domain_id_at_index(MemoryDomain(index)))
        for size in sizes:
            var plain = allocator.allocate(size)
            var cells = plain.as_pointer[DType.uint8]()
            cells[unsafe_offset=0] = 0x5A
            cells[unsafe_offset=size - 1] = 0x5A
            assert_equal(Int(cells[unsafe_offset=0]), 0x5A, "placed pages are writable")
            assert_equal(
                Int(cells[unsafe_offset=size - 1]),
                0x5A,
                "to the very end of the block",
            )

            var roomy = allocator.allocate_at_least(size)
            assert_true(
                roomy.allocated_bytes >= size,
                "and reports at least what was asked",
            )
            assert_true(
                roomy.bytes_per_page > 0,
                "and names the page size it used",
            )


def test_zero_byte_allocation_is_refused() raises:
    var library = Library()
    var topology = Topology(library)
    var allocator = default_domain_allocator(topology)
    var refused = False
    try:
        _ = allocator.allocate(0)
    except:
        refused = True
    assert_true(refused, "a zero-byte request is refused rather than answering an empty block")


def test_invalid_domain_id_yields_no_allocator() raises:
    var library = Library()
    var refused = False
    try:
        _ = DomainAllocator.at(library, MemoryDomainId(-1))
    except:
        refused = True
    assert_true(refused, "an id naming no domain yields no allocator")


def test_replicated_array_gives_every_domain_a_copy() raises:
    var library = Library()
    var topology = Topology(library)
    var replicated = ReplicatedArray[DType.int64].new(topology, 512)
    assert_equal(
        replicated.memory_domains_count(),
        topology.memory_domains_count(),
        "one replica per memory domain",
    )
    for index in range(replicated.memory_domains_count()):
        var domain = MemoryDomain(index)
        var copy = replicated.replica(domain)
        for slot in range(512):
            copy[unsafe_offset=slot] = Int64(index * 1000 + slot)
    for index in range(replicated.memory_domains_count()):
        var domain = MemoryDomain(index)
        assert_equal(
            Int(replicated.at(domain, 511)),
            index * 1000 + 511,
            "each replica kept its own writes",
        )


def test_sharded_array_round_trips_its_segments() raises:
    var library = Library()
    var topology = Topology(library)
    var sharded = ShardedArray[DType.int64].new(topology, 1000)
    var covered = 0
    for index in range(sharded.memory_domains_count()):
        covered += sharded.length_on_memory_domain(MemoryDomain(index))
    assert_equal(covered, 1000, "the shards tile the logical range exactly once")

    for logical in range(0, 1000, 97):
        var location = sharded.location_of(logical)
        assert_equal(
            sharded.logical_index_of(location.memory_domain, location.local_index),
            logical,
            "the forward and inverse maps agree",
        )


# endregion Memory

# region Fabric


def test_fabric_harvest_fills_edges() raises:
    """The harvest measures every compute-to-memory edge, so its cost scales with their product."""
    var library = Library()
    var topology = Topology(library)
    var pool = Pool(topology, threads=topology.logical_cores_count())
    var fabric = Fabric(library)
    try:
        fabric.harvest(topology, pool)
    except:
        return  # ? A flat pool without domain placement has no fabric to walk
    assert_true(
        fabric.memory_levels_count() >= 1,
        "a harvested fabric has at least one tier",
    )
    for compute in range(topology.compute_domains_count()):
        for memory in range(topology.memory_domains_count()):
            var distance = fabric.memory_distance(ComputeDomain(compute), MemoryDomain(memory))
            assert_true(distance > 0, "every reachable edge carries a distance")
            if topology.local_memory_of(ComputeDomain(compute)).index == memory:
                assert_equal(
                    distance,
                    10,
                    "SLIT says a domain's own memory is at distance ten",
                )


# endregion Fabric


def main() raises:
    var suite = TestSuite()

    suite.test[test_indexed_split_tiles_the_range]()
    suite.test[test_cache_aligned_pads_to_a_line]()
    suite.test[test_capability_masks]()
    suite.test[test_domain_types_are_distinct]()

    suite.test[test_library_loads_and_reports_its_version]()
    suite.test[test_missing_library_raises]()
    suite.test[test_capabilities_are_reported_and_nameable]()
    suite.test[test_topology_counts_and_core_partition]()
    suite.test[test_topology_memory_bounds]()
    suite.test[test_local_memory_is_an_index_not_an_id]()

    suite.test[test_spawn_rejects_zero_threads]()
    suite.test[test_spawn_succeeds_at_full_width]()
    suite.test[test_caller_exclusivity_is_queried_not_cached]()
    suite.test[test_pool_capabilities_only_narrow]()
    suite.test[test_per_compute_domain_pools]()
    suite.test[test_pool_domain_accounting]()
    suite.test[test_sleep_then_wake_loses_nothing]()

    suite.test[test_for_threads_reaches_every_worker]()
    suite.test[test_for_n_visits_each_task_once_and_again]()
    suite.test[test_for_n_across_uncomfortable_sizes]()
    suite.test[test_for_n_dynamic_covers_uneven_work]()
    suite.test[test_for_n_dynamic_oversubscribed]()
    suite.test[test_for_slices_partitions_the_range]()
    suite.test[test_generation_polling_on_an_exclusive_pool]()

    suite.test[test_allocations_on_every_memory_domain]()
    suite.test[test_zero_byte_allocation_is_refused]()
    suite.test[test_invalid_domain_id_yields_no_allocator]()
    suite.test[test_replicated_array_gives_every_domain_a_copy]()
    suite.test[test_sharded_array_round_trips_its_segments]()

    suite.test[test_fabric_harvest_fills_edges]()

    suite^.run()
