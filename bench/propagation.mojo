"""Demo app: Connected Components by label propagation, with ForkUnion.

The N-body simulation gives every task an identical cost, so it can only measure dispatch latency.
Label propagation is the opposite end of fork-join usage: one parallel sweep per round, repeated
until no label changes - so a single pass pays the dispatch-and-join tax once per round, and the
graph's topology decides how many rounds there are.

The generator strings `C` independent R-MAT communities on a ring, joined by one bridge edge per
neighbouring pair. The global minimum label must walk the ring, so convergence takes O(C) rounds
while each round stays a bandwidth-bound sweep - the fork-join frequency is the controlled axis.

The labels are double-buffered: every round reads the immutable previous array and each vertex
writes only its own slot in the next - no atomics, no races, and every round is a pure function of
the last. Rounds-to-convergence and the final fixed point are therefore identical across schedules,
thread counts, and languages - the C++, Rust, Zig, and this port print the same component count,
round count, and checksum.

The environment variables it reads are listed in `bench/harness.mojo`. The backends are the four
`forkunion_{static,dynamic}_{shared,replicated}` cells and the `max_parallelize` baseline.
"""

from max.algorithm import parallelize

from std.sys import exit, stderr

from forkunion import (
    CacheAligned,
    ComputeDomain,
    Library,
    MemoryDomain,
    Pool,
    ThreadInDomain,
    ReplicatedArray,
    Topology,
)

from harness import Loop, Settings, exit_unparsed, print_row, print_settings

comptime SENTINEL = UInt32(0xFFFF_FFFF)
"""Sorts past every valid edge; marks a dropped self-loop, trimmed with the duplicates."""


@always_inline
def split_mix(counter: UInt64) -> UInt64:
    """The SplitMix64 avalanche behind every random draw - a pure function of the counter."""
    var x = (counter + 1) * 0x9E37_79B9_7F4A_7C15
    x = (x ^ (x >> 30)) * 0xBF58_476D_1CE4_E5B9
    x = (x ^ (x >> 27)) * 0x94D0_49BB_1331_11EB
    return x ^ (x >> 31)


@always_inline
def random_percent(key: UInt64, counter: UInt64) -> UInt32:
    """One quadrant choice in `[0, 100)` - the same draw scheme as every sibling benchmark."""
    return UInt32(Int(split_mix(key + counter) % 100))


@always_inline
def random_index(key: UInt64, counter: UInt64, bound: UInt32) -> UInt32:
    """One bridge endpoint in `[0, bound)`, from the same avalanche."""
    return UInt32(Int(split_mix(key + counter) % UInt64(Int(bound))))


@fieldwise_init
struct FillScratch(ImplicitlyCopyable, TrivialRegisterPassable):
    """Everything the parallel R-MAT fill reads or writes; one pair of slots per raw edge."""

    var rows: Pointer[UInt32, MutUntrackedOrigin]
    var columns: Pointer[UInt32, MutUntrackedOrigin]
    var key: UInt64
    var scale: Int
    var raw_local: Int


def fill_edge(task: Int, at: ThreadInDomain, mut scratch: FillScratch):
    """Fills raw edge `e`'s pair of slots from its quadrant walk; self-loops stay sentinels."""
    var edge = task
    var row = UInt32(0)
    var column = UInt32(0)
    var bit = scratch.scale
    while bit > 0:
        bit -= 1
        # a=57 b=19 c=19 d=5, the same quadrant weights the sibling generators use.
        var draw = random_percent(scratch.key, UInt64(edge) * 64 + UInt64(bit))
        var step = UInt32(1) << UInt32(bit)
        if draw < 57:
            continue
        elif draw < 76:
            column |= step
        elif draw < 95:
            row |= step
        else:
            row |= step
            column |= step
    if row != column:
        var base = UInt32((edge // scratch.raw_local) << scratch.scale)
        scratch.rows[unsafe_offset=edge * 2] = base + row
        scratch.columns[unsafe_offset=edge * 2] = base + column
        scratch.rows[unsafe_offset=edge * 2 + 1] = base + column
        scratch.columns[unsafe_offset=edge * 2 + 1] = base + row


@fieldwise_init
struct Graph(ImplicitlyCopyable, TrivialRegisterPassable):
    """A read-only CSR as two raw runs - the interface every kernel takes."""

    var row_offsets: Pointer[UInt64, MutUntrackedOrigin]
    var column_indices: Pointer[UInt32, MutUntrackedOrigin]
    var vertices: Int
    var directed_edges: Int


@fieldwise_init
struct Round(ImplicitlyCopyable, TrivialRegisterPassable):
    """Everything one round reads or writes; the label buffers swap between rounds.

    The replica fields are read only by the replicated kernels; the shared ones ignore them, the
    way the sibling ports let a compile-time placement elide the branch.
    """

    var graph: Graph
    var old_labels: Pointer[UInt32, MutUntrackedOrigin]
    var new_labels: Pointer[UInt32, MutUntrackedOrigin]
    var counters: Pointer[CacheAligned[UInt64], MutUntrackedOrigin]
    var replica_offsets: Pointer[Pointer[UInt64, MutUntrackedOrigin], MutUntrackedOrigin]
    """Each memory domain's copy of `row_offsets`, at wherever the page-rounded stride put it."""
    var replica_columns: Pointer[Pointer[UInt32, MutUntrackedOrigin], MutUntrackedOrigin]
    """Each memory domain's copy of `column_indices`, likewise."""
    var local_memory: Pointer[Int32, MutUntrackedOrigin]


@always_inline
def min_label_of(graph: Graph, labels: Pointer[UInt32, MutUntrackedOrigin], vertex: Int) -> UInt32:
    """The smallest label among a vertex and its neighbours, read from the previous round."""
    var best = labels[unsafe_offset=vertex]
    var position = Int(graph.row_offsets[unsafe_offset=vertex])
    var end = Int(graph.row_offsets[unsafe_offset=vertex + 1])
    while position < end:
        var candidate = labels[unsafe_offset=Int(graph.column_indices[unsafe_offset=position])]
        if candidate < best:
            best = candidate
        position += 1
    return best


def label_vertex(task: Int, at: ThreadInDomain, mut round: Round):
    """One vertex's update: read the immutable previous labels, write only your own slot."""
    var vertex = task
    var next = min_label_of(round.graph, round.old_labels, vertex)
    round.new_labels[unsafe_offset=vertex] = next
    if next != round.old_labels[unsafe_offset=vertex]:
        round.counters[unsafe_offset=at.thread].value += 1


def label_vertex_replicated(task: Int, at: ThreadInDomain, mut round: Round):
    """The same update, reading the CSR replica that lives on this thread's own memory domain.

    Only the immutable CSR replicates; the label buffers stay shared by nature, since every round
    must see every neighbour's last label.
    """
    var vertex = task
    var domain = Int(round.local_memory[unsafe_offset=at.compute_domain])
    var local = Graph(
        round.replica_offsets[unsafe_offset=domain],
        round.replica_columns[unsafe_offset=domain],
        round.graph.vertices,
        round.graph.directed_edges,
    )
    var next = min_label_of(local, round.old_labels, vertex)
    round.new_labels[unsafe_offset=vertex] = next
    if next != round.old_labels[unsafe_offset=vertex]:
        round.counters[unsafe_offset=at.thread].value += 1


def generate_necklace(
    mut pool: Pool,
    key: UInt64,
    scale: Int,
    communities: Int,
    edge_factor: Int,
    mut row_offsets: List[UInt64],
    mut column_indices: List[UInt32],
) raises -> Tuple[Int, Int]:
    """The necklace: independent R-MAT communities joined in a ring, scattered into a CSR.

    Community `c` owns global edge indices `[c * raw_local, (c+1) * raw_local)` and the vertex range
    `[c << scale, (c+1) << scale)`; the quadrant walk uses the same `e * 64 + bit` counters as the
    sibling generators, and bridge draws live in their own counter range above all edge draws.
    """
    var community_vertices = 1 << scale
    var vertices = communities * community_vertices
    var raw_local = community_vertices * edge_factor
    var raw_edges = communities * raw_local
    var bridges = communities if communities > 1 else 0
    var slots = raw_edges * 2 + bridges * 2

    var rows = List[UInt32](length=slots, fill=SENTINEL)
    var columns = List[UInt32](length=slots, fill=SENTINEL)
    var scratch = FillScratch(
        rows.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        columns.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        key,
        scale,
        raw_local,
    )
    pool.for_n[fill_edge](raw_edges, scratch)

    # Bridges: endpoints in each community's first 64 vertices - R-MAT's quadrant bias piles the
    # hubs at low indices, so a low endpoint is essentially guaranteed well-connected.
    var hub_core = UInt32(min(community_vertices, 64))
    var bridge_base = UInt64(raw_edges) * 64
    for bridge in range(bridges):
        var left = UInt32(bridge << scale) + random_index(key, bridge_base + 2 * UInt64(bridge), hub_core)
        var right = UInt32(((bridge + 1) % communities) << scale) + random_index(
            key, bridge_base + 2 * UInt64(bridge) + 1, hub_core
        )
        rows[raw_edges * 2 + bridge * 2] = left
        columns[raw_edges * 2 + bridge * 2] = right
        rows[raw_edges * 2 + bridge * 2 + 1] = right
        columns[raw_edges * 2 + bridge * 2 + 1] = left

    # Sort by (row, column) through a permutation, then dedup and drop the sentinel tail.
    var order = List[Int](length=slots, fill=0)
    for index in range(slots):
        order[index] = index

    @parameter
    def before(left: Int, right: Int) -> Bool:
        if rows[left] != rows[right]:
            return rows[left] < rows[right]
        return columns[left] < columns[right]

    sort[before](order)

    var live_rows = List[UInt32](capacity=slots)
    var live_columns = List[UInt32](capacity=slots)
    for position in range(slots):
        var index = order[position]
        if rows[index] == SENTINEL:
            continue
        if (
            len(live_rows) > 0
            and rows[index] == live_rows[len(live_rows) - 1]
            and columns[index] == live_columns[len(live_columns) - 1]
        ):
            continue
        live_rows.append(rows[index])
        live_columns.append(columns[index])

    # CSR: count the degrees into `row_offsets`, then prefix-sum them into row starts.
    row_offsets = List[UInt64](length=vertices + 1, fill=0)
    for index in range(len(live_rows)):
        row_offsets[Int(live_rows[index]) + 1] += 1
    for vertex in range(vertices):
        row_offsets[vertex + 1] += row_offsets[vertex]
    column_indices = List[UInt32](length=len(live_columns), fill=0)
    var cursor = List[UInt64](length=vertices, fill=0)
    for vertex in range(vertices):
        cursor[vertex] = row_offsets[vertex]
    for index in range(len(live_rows)):
        var row = Int(live_rows[index])
        column_indices[Int(cursor[row])] = live_columns[index]
        cursor[row] += 1

    return (vertices, len(live_columns))


def converge_with_parallelize(
    template: Round,
    vertices: Int,
    threads: Int,
    mut labels_a: List[UInt32],
    mut labels_b: List[UInt32],
) -> Int:
    """The baseline: the same rounds, dispatched by `max.algorithm.parallelize`.

    It takes a capturing closure, so the change tally can be an ordinary local rather than the
    per-thread cache-aligned slots ForkUnion's non-capturing callbacks need.
    """
    for vertex in range(vertices):
        labels_a[vertex] = UInt32(vertex)
    var old_labels = labels_a.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var new_labels = labels_b.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var graph = template.graph

    var rounds = 0
    while True:

        @parameter
        def sweep(vertex: Int):
            new_labels[unsafe_offset=vertex] = min_label_of(graph, old_labels, vertex)

        parallelize[sweep](vertices, threads)
        rounds += 1
        var changes = 0
        for vertex in range(vertices):
            if new_labels[unsafe_offset=vertex] != old_labels[unsafe_offset=vertex]:
                changes += 1
        var swap = old_labels
        old_labels = new_labels
        new_labels = swap
        if changes == 0:
            break
    return rounds


def converge_on_pool[
    work: def(Int, ThreadInDomain, mut Round) thin -> None
](
    mut pool: Pool,
    template: Round,
    vertices: Int,
    mut labels_a: List[UInt32],
    mut labels_b: List[UInt32],
    mut counters: List[CacheAligned[UInt64]],
    dynamic: Bool,
) raises -> Int:
    """One convergence pass; every round is one fork-join dispatch."""
    for vertex in range(vertices):
        labels_a[vertex] = UInt32(vertex)
    var old_labels = labels_a.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var new_labels = labels_b.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var counter_base = counters.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()

    var rounds = 0
    while True:
        for index in range(len(counters)):
            counters[index].value = 0
        # Build it whole: mutating a copy of a register-passable scratch and then dispatching loses
        # the writes, because the pointer the workers get need not see them.
        var round = Round(
            template.graph,
            old_labels,
            new_labels,
            counter_base,
            template.replica_offsets,
            template.replica_columns,
            template.local_memory,
        )
        if dynamic:
            pool.for_n_dynamic[work](vertices, round)
        else:
            pool.for_n[work](vertices, round)
        rounds += 1
        var changes = UInt64(0)
        for index in range(len(counters)):
            changes += counters[index].value
        var swap = old_labels
        old_labels = new_labels
        new_labels = swap
        if changes == 0:
            break
    return rounds


def converge_serially(graph: Graph, mut labels_a: List[UInt32], mut labels_b: List[UInt32]) -> Int:
    """The same fixed point, one thread, as the oracle the parallel pass must match exactly."""
    for vertex in range(graph.vertices):
        labels_a[vertex] = UInt32(vertex)
    var old_labels = labels_a.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var new_labels = labels_b.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var rounds = 0
    while True:
        var changed = UInt64(0)
        for vertex in range(graph.vertices):
            var next = min_label_of(graph, old_labels, vertex)
            new_labels[unsafe_offset=vertex] = next
            if next != old_labels[unsafe_offset=vertex]:
                changed += 1
        rounds += 1
        var swap = old_labels
        old_labels = new_labels
        new_labels = swap
        if changed == 0:
            break
    return rounds


comptime BACKEND_GRAMMAR = (
    "one of forkunion_static_shared, forkunion_dynamic_shared, forkunion_static_replicated,"
    " forkunion_dynamic_replicated, max_parallelize"
)


def parse_backend(name: String) -> Optional[String]:
    """The backend named `name`, or nothing when this binary has none by that name."""
    if (
        name == "forkunion_static_shared"
        or name == "forkunion_dynamic_shared"
        or name == "forkunion_static_replicated"
        or name == "forkunion_dynamic_replicated"
        or name == "max_parallelize"
    ):
        return name
    return None


def main() raises:
    var library = Library()
    var topology = Topology(library)
    var settings = Settings(topology.logical_cores_count())
    if not parse_backend(settings.backend):
        exit_unparsed("FORKUNION_BACKEND", settings.backend, BACKEND_GRAMMAR)
    print_settings(settings)
    var scale = settings.scale
    var communities = settings.communities
    var threads = settings.threads
    var backend = settings.backend
    var dynamic = "dynamic" in backend
    var replicated = "replicated" in backend
    var baseline = backend == "max_parallelize"
    if scale > 32 or communities > (1 << 32) >> scale:
        print(
            "FORKUNION_PROPAGATION_COMMUNITIES << FORKUNION_PROPAGATION_SCALE must fit 32-bit vertex indices",
            file=stderr,
        )
        exit(1)

    var pool = Pool(topology, threads=threads, name="propagation")

    var row_offsets = List[UInt64]()
    var column_indices = List[UInt32]()
    var built = generate_necklace(
        pool, split_mix(UInt64(settings.seed)), scale, communities, settings.edge_factor, row_offsets, column_indices
    )
    var vertices = built[0]
    var directed_edges = built[1]
    var graph = Graph(
        row_offsets.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        column_indices.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        vertices,
        directed_edges,
    )
    print(t"vertices {vertices}, directed edges {directed_edges}, communities {communities}")

    var labels_a = List[UInt32](length=vertices, fill=0)
    var labels_b = List[UInt32](length=vertices, fill=0)
    var counters = List[CacheAligned[UInt64]](capacity=threads)
    for _ in range(threads):
        counters.append(CacheAligned[UInt64](UInt64(0)))

    # Per-domain CSR replicas for the `_replicated` cells; only the immutable CSR replicates - the
    # label buffers stay shared by nature, since every round must see every neighbour's last label.
    var replica_offsets = Optional[ReplicatedArray[DType.uint64]](None)
    var replica_columns = Optional[ReplicatedArray[DType.uint32]](None)
    var offsets_bases = List[Pointer[UInt64, MutUntrackedOrigin]]()
    var columns_bases = List[Pointer[UInt32, MutUntrackedOrigin]]()
    var local_memory = List[Int32](length=max(topology.compute_domains_count(), 1), fill=0)
    for index in range(topology.compute_domains_count()):
        local_memory[index] = Int32(topology.local_memory_of(ComputeDomain(index)).index)
    if replicated:
        try:
            replica_offsets = ReplicatedArray[DType.uint64].new(topology, vertices + 1)
            replica_columns = ReplicatedArray[DType.uint32].new(topology, directed_edges)
        except:
            print("Failed to replicate the graph across memory domains", file=stderr)
            exit(1)
        for domain in range(replica_offsets.value().memory_domains_count()):
            var offsets_replica = replica_offsets.value().replica(MemoryDomain(domain))
            for index in range(vertices + 1):
                offsets_replica[unsafe_offset=index] = row_offsets[index]
            offsets_bases.append(offsets_replica.unsafe_origin_cast[MutUntrackedOrigin]())
            var columns_replica = replica_columns.value().replica(MemoryDomain(domain))
            for index in range(directed_edges):
                columns_replica[unsafe_offset=index] = column_indices[index]
            columns_bases.append(columns_replica.unsafe_origin_cast[MutUntrackedOrigin]())

    # The label and counter pointers are refreshed every round; these are the same buffers.
    var template = Round(
        graph,
        labels_a.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        labels_b.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        counters.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        offsets_bases.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        columns_bases.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        local_memory.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
    )

    @parameter
    def one_pass() raises -> Int:
        if baseline:
            return converge_with_parallelize(template, vertices, threads, labels_a, labels_b)
        if replicated:
            return converge_on_pool[label_vertex_replicated](
                pool, template, vertices, labels_a, labels_b, counters, dynamic
            )
        return converge_on_pool[label_vertex](pool, template, vertices, labels_a, labels_b, counters, dynamic)

    # A fixed time budget beats a fixed pass count: every backend runs the same wall-clock window -
    # long enough to amortize scheduling noise - and reports the rate it sustained, with no
    # per-backend pass-count guessing. The warm-up absorbs page faults and cache warming, which
    # would otherwise bias the first timed pass, and by a different amount for each backend.
    var rounds = 0
    var loop = Loop(settings.warmup_ns, settings.time_limit_ns)
    while loop.next():
        rounds = one_pass()

    # The fixed point sits in both buffers - the terminal round changed nothing - so read either.
    var components = 0
    var checksum = UInt64(0)
    for vertex in range(vertices):
        if labels_a[vertex] == UInt32(vertex):
            components += 1
        checksum += UInt64(Int(labels_a[vertex]))

    # Directed edges scanned per call; `rounds * edges` is the exact scan count, identical in every
    # cell by the double-buffered determinism, so the "edges" rate is the MTEPS figure.
    loop.rate("edges", Float64(rounds) * Float64(directed_edges))
    loop.counter("rounds", Float64(rounds))
    print_row(loop.row(backend))
    print(t"{components} components, {rounds} rounds, checksum {checksum}")

    if settings.check:
        var serial_a = List[UInt32](length=vertices, fill=0)
        var serial_b = List[UInt32](length=vertices, fill=0)
        var serial_rounds = converge_serially(graph, serial_a, serial_b)
        if serial_rounds != rounds:
            print(t"MISMATCH: serial converged in {serial_rounds} rounds", file=stderr)
            exit(1)
        for vertex in range(vertices):
            if serial_a[vertex] != labels_a[vertex]:
                print(t"MISMATCH: serial labels differ at vertex {vertex}", file=stderr)
                exit(1)
        print("check: matches the serial labels and rounds")

    # `graph` holds raw pointers into these two, and Mojo releases a value after its last named use
    # - which would otherwise be the `Graph` construction above, long before the last read.
    _ = row_offsets^
    _ = column_indices^
    _ = local_memory^
    _ = offsets_bases^
    _ = columns_bases^
    _ = replica_offsets^
    _ = replica_columns^
