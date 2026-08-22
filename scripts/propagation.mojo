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

Environment variables, matching the sibling scripts:

- `PROPAGATION_SCALE` - each community has 2^scale vertices, default 14.
- `PROPAGATION_COMMUNITIES` - communities strung on the ring, default 64.
- `PROPAGATION_EDGE_FACTOR` - edges generated per vertex, before deduplication, default 16.
- `PROPAGATION_BACKEND` - one of the four `forkunion_{static,dynamic}_{shared,replicated}` cells, or
  the `max_parallelize` baseline; default `forkunion_static_shared`.
- `PROPAGATION_THREADS` - number of threads, default the logical core count.
- `PROPAGATION_SECONDS` - wall-clock budget per run, default 10.
- `PROPAGATION_ITERATIONS` - run an exact pass count instead, when set.
- `PROPAGATION_CHECK` - also converge serially, and fail unless labels and rounds agree exactly.
"""

from max.algorithm import parallelize

from std.os import getenv
from std.sys import num_logical_cores
from std.time import perf_counter_ns

from forkunion import (
    CacheAligned,
    ComputeDomain,
    Library,
    MemoryDomain,
    Pool,
    Prong,
    ReplicatedArray,
    Topology,
)

comptime SENTINEL = UInt32(0xFFFF_FFFF)
"""Sorts past every valid edge; marks a dropped self-loop, trimmed with the duplicates."""


def fixed(value: Float64, decimals: Int) -> String:
    """Renders a non-negative value with exactly `decimals` places, since t-strings take no spec."""
    var scale = 1
    for _ in range(decimals):
        scale *= 10
    var scaled = Int(value * Float64(scale) + 0.5)
    var whole = String(scaled // scale)
    var fraction = String(scaled % scale)
    while fraction.byte_length() < decimals:
        fraction = String("0") + fraction
    return whole + "." + fraction


def parse(name: StaticString, fallback: Int) -> Int:
    var text = getenv(name)
    if text.byte_length() == 0:
        return fallback
    try:
        return Int(text)
    except:
        return fallback


@always_inline
def split_mix(counter: UInt64) -> UInt64:
    """The SplitMix64 avalanche behind every random draw - a pure function of the counter."""
    var x = (counter + 1) * 0x9E37_79B9_7F4A_7C15
    x = (x ^ (x >> 30)) * 0xBF58_476D_1CE4_E5B9
    x = (x ^ (x >> 27)) * 0x94D0_49BB_1331_11EB
    return x ^ (x >> 31)


@always_inline
def random_percent(counter: UInt64) -> UInt32:
    """One quadrant choice in `[0, 100)` - the same draw scheme as every sibling benchmark."""
    return UInt32(Int(split_mix(counter) % 100))


@always_inline
def random_index(counter: UInt64, bound: UInt32) -> UInt32:
    """One bridge endpoint in `[0, bound)`, from the same avalanche."""
    return UInt32(Int(split_mix(counter) % UInt64(Int(bound))))


@fieldwise_init
struct FillScratch(ImplicitlyCopyable, TrivialRegisterPassable):
    """Everything the parallel R-MAT fill reads or writes; one pair of slots per raw edge."""

    var rows: Pointer[UInt32, MutUntrackedOrigin]
    var columns: Pointer[UInt32, MutUntrackedOrigin]
    var scale: Int
    var raw_local: Int


def fill_edge(prong: Prong, mut scratch: FillScratch):
    """Fills raw edge `e`'s pair of slots from its quadrant walk; self-loops stay sentinels."""
    var edge = prong.task_index
    var row = UInt32(0)
    var column = UInt32(0)
    var bit = scratch.scale
    while bit > 0:
        bit -= 1
        # a=57 b=19 c=19 d=5, the same quadrant weights the sibling generators use.
        var draw = random_percent(UInt64(edge) * 64 + UInt64(bit))
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
    var replica_offsets: Pointer[UInt64, MutUntrackedOrigin]
    var replica_columns: Pointer[UInt32, MutUntrackedOrigin]
    var replica_offsets_stride: Int
    var replica_columns_stride: Int
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


def label_vertex(prong: Prong, mut round: Round):
    """One vertex's update: read the immutable previous labels, write only your own slot."""
    var vertex = prong.task_index
    var next = min_label_of(round.graph, round.old_labels, vertex)
    round.new_labels[unsafe_offset=vertex] = next
    if next != round.old_labels[unsafe_offset=vertex]:
        round.counters[unsafe_offset=prong.thread_index].value += 1


def label_vertex_replicated(prong: Prong, mut round: Round):
    """The same update, reading the CSR replica that lives on this thread's own memory domain.

    Only the immutable CSR replicates; the label buffers stay shared by nature, since every round
    must see every neighbour's last label.
    """
    var vertex = prong.task_index
    var domain = Int(round.local_memory[unsafe_offset=prong.compute_domain_index])
    var local = Graph(
        Pointer(to=round.replica_offsets[unsafe_offset=domain * round.replica_offsets_stride]),
        Pointer(to=round.replica_columns[unsafe_offset=domain * round.replica_columns_stride]),
        round.graph.vertices,
        round.graph.directed_edges,
    )
    var next = min_label_of(local, round.old_labels, vertex)
    round.new_labels[unsafe_offset=vertex] = next
    if next != round.old_labels[unsafe_offset=vertex]:
        round.counters[unsafe_offset=prong.thread_index].value += 1


def generate_necklace(
    mut pool: Pool,
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
        scale,
        raw_local,
    )
    pool.for_n[fill_edge](raw_edges, scratch)

    # Bridges: endpoints in each community's first 64 vertices - R-MAT's quadrant bias piles the
    # hubs at low indices, so a low endpoint is essentially guaranteed well-connected.
    var hub_core = UInt32(min(community_vertices, 64))
    var bridge_base = UInt64(raw_edges) * 64
    for bridge in range(bridges):
        var left = UInt32(bridge << scale) + random_index(bridge_base + 2 * UInt64(bridge), hub_core)
        var right = UInt32(((bridge + 1) % communities) << scale) + random_index(
            bridge_base + 2 * UInt64(bridge) + 1, hub_core
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
    work: def(Prong, mut Round) thin -> None
](
    mut pool: Pool,
    template: Round,
    vertices: Int,
    mut labels_a: List[UInt32],
    mut labels_b: List[UInt32],
    mut counters: List[CacheAligned[UInt64]],
    dynamic: Bool,
) -> Int:
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
        # Build it whole: mutating a copy of a register-passable scratch and then dispatching
        # loses the writes, because the pointer the workers get need not see them.
        var round = Round(
            template.graph,
            old_labels,
            new_labels,
            counter_base,
            template.replica_offsets,
            template.replica_columns,
            template.replica_offsets_stride,
            template.replica_columns_stride,
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


def main() raises:
    var scale = parse("PROPAGATION_SCALE", 14)
    var communities = parse("PROPAGATION_COMMUNITIES", 64)
    var edge_factor = parse("PROPAGATION_EDGE_FACTOR", 16)
    var threads = parse("PROPAGATION_THREADS", num_logical_cores())
    var budget = parse("PROPAGATION_SECONDS", 10) * 1_000_000_000
    var iterations = parse("PROPAGATION_ITERATIONS", 0)
    var backend = getenv("PROPAGATION_BACKEND")
    if backend.byte_length() == 0:
        backend = String("forkunion_static_shared")
    var known = (
        backend == "forkunion_static_shared"
        or backend == "forkunion_dynamic_shared"
        or backend == "forkunion_static_replicated"
        or backend == "forkunion_dynamic_replicated"
        or backend == "max_parallelize"
    )
    if not known:
        print("Unsupported backend: '", backend, "'", sep="")
        print("  forkunion_static_shared")
        print("  forkunion_dynamic_shared")
        print("  forkunion_static_replicated")
        print("  forkunion_dynamic_replicated")
        print("  max_parallelize")
        return
    var dynamic = "dynamic" in backend
    var replicated = "replicated" in backend
    var baseline = backend == "max_parallelize"
    var check = getenv("PROPAGATION_CHECK").byte_length() > 0

    var library = Library()
    var topology = Topology(library)
    var pool = Pool(topology, threads=threads, name="propagation")

    var row_offsets = List[UInt64]()
    var column_indices = List[UInt32]()
    var built = generate_necklace(pool, scale, communities, edge_factor, row_offsets, column_indices)
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
    var domains = topology.memory_domains_count()
    var replica_offsets = ReplicatedArray[DType.uint64].try_new(topology, vertices + 1)
    var replica_columns = ReplicatedArray[DType.uint32].try_new(topology, directed_edges)
    var local_memory = List[Int32](length=max(topology.compute_domains_count(), 1), fill=0)
    for index in range(topology.compute_domains_count()):
        local_memory[index] = Int32(topology.local_memory_of(ComputeDomain(index)).index)
    if replicated:
        if not replica_offsets or not replica_columns:
            print("Unsupported backend: '", backend, "' - no symmetric mapping on this machine", sep="")
            return
        for domain in range(domains):
            var offsets_replica = replica_offsets.value().replica(MemoryDomain(domain))
            for index in range(vertices + 1):
                offsets_replica[unsafe_offset=index] = row_offsets[index]
            var columns_replica = replica_columns.value().replica(MemoryDomain(domain))
            for index in range(directed_edges):
                columns_replica[unsafe_offset=index] = column_indices[index]

    # The label and counter pointers are refreshed every round; these are the same buffers.
    var template = Round(
        graph,
        labels_a.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        labels_b.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        counters.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        replica_offsets.value()
        .replica(MemoryDomain(0))
        .unsafe_origin_cast[MutUntrackedOrigin]() if replicated else graph.row_offsets,
        replica_columns.value()
        .replica(MemoryDomain(0))
        .unsafe_origin_cast[MutUntrackedOrigin]() if replicated else graph.column_indices,
        vertices + 1,
        directed_edges,
        local_memory.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
    )

    var rounds = 0
    var passes = 0
    var started = perf_counter_ns()
    if iterations > 0:
        for _ in range(iterations):
            if baseline:
                rounds = converge_with_parallelize(template, vertices, threads, labels_a, labels_b)
            elif replicated:
                rounds = converge_on_pool[label_vertex_replicated](
                    pool, template, vertices, labels_a, labels_b, counters, dynamic
                )
            else:
                rounds = converge_on_pool[label_vertex](pool, template, vertices, labels_a, labels_b, counters, dynamic)
        passes = iterations
    else:
        while True:
            if baseline:
                rounds = converge_with_parallelize(template, vertices, threads, labels_a, labels_b)
            elif replicated:
                rounds = converge_on_pool[label_vertex_replicated](
                    pool, template, vertices, labels_a, labels_b, counters, dynamic
                )
            else:
                rounds = converge_on_pool[label_vertex](pool, template, vertices, labels_a, labels_b, counters, dynamic)
            passes += 1
            if Int(perf_counter_ns() - started) >= budget:
                break
    var seconds = Float64(Int(perf_counter_ns() - started)) / 1e9 / Float64(passes)

    # The fixed point sits in both buffers - the terminal round changed nothing - so read either.
    var components = 0
    var checksum = UInt64(0)
    for vertex in range(vertices):
        if labels_a[vertex] == UInt32(vertex):
            components += 1
        checksum += UInt64(Int(labels_a[vertex]))

    # MTEPS - millions of directed edges scanned per second; `rounds * edges` is the exact scan
    # count, identical in every cell by the double-buffered determinism.
    var mteps = Float64(rounds) * Float64(directed_edges) / seconds / 1e6
    var pace = fixed(seconds, 2)
    var rate = fixed(mteps, 1)
    print(t"{backend}: {components} components, {rounds} rounds, checksum {checksum}, {pace} s/pass, {rate} MTEPS")

    if check:
        var serial_a = List[UInt32](length=vertices, fill=0)
        var serial_b = List[UInt32](length=vertices, fill=0)
        var serial_rounds = converge_serially(graph, serial_a, serial_b)
        if serial_rounds != rounds:
            raise Error(t"MISMATCH: serial converged in {serial_rounds} rounds")
        for vertex in range(vertices):
            if serial_a[vertex] != labels_a[vertex]:
                raise Error(t"MISMATCH: serial labels differ at vertex {vertex}")
        print("check: matches the serial labels and rounds")

    # `graph` holds raw pointers into these two, and Mojo releases a value after its last named
    # use - which would otherwise be the `Graph` construction above, long before the last read.
    _ = row_offsets^
    _ = column_indices^
    _ = local_memory^
    _ = replica_offsets^
    _ = replica_columns^
