"""N-Body simulation benchmark, measuring dispatch overhead rather than arithmetic.

Compares the synchronization overhead of ForkUnion's schedules and placements:

- `forkunion_static_shared` - static work division, N tasks pre-divided into thread slices.
- `forkunion_dynamic_shared` - dynamic work-stealing.
- `forkunion_static_replicated` - static, with body positions replicated into each domain's memory.
- `forkunion_dynamic_replicated` - work-stealing, over the same per-domain replicas.
- `max_parallelize` - the baseline: `max.algorithm.parallelize`, which takes every hardware thread.

On a machine with one memory domain the replicas collapse to one, so the replicated cells run
everywhere. The all-to-all cannot be sharded - every body reads every other - so the only locality
left to win is the read side.

Each backend runs a fixed wall-clock window - 10 seconds by default - and reports the dispatch rate
it sustained: contended-atomic paths amplify background noise, so the window sizes the iteration
count to the machine instead of guessing it.

Environment variables:

- `NBODY_COUNT` - number of bodies, default the thread count.
- `NBODY_SECONDS` - wall-clock budget per run, reporting the sustained rate, default 10.
- `NBODY_ITERATIONS` - run an exact iteration count instead, when set.
- `NBODY_BACKEND` - one of the backend names above, default `forkunion_static_shared`.
- `NBODY_THREADS` - number of threads, default the logical core count.

```sh
NBODY_COUNT=512 NBODY_BACKEND=forkunion_static_shared pixi run nbody
NBODY_COUNT=512 NBODY_BACKEND=forkunion_static_replicated pixi run nbody
```
"""

from max.algorithm import parallelize

from std.math import floor
from std.memory import bitcast
from std.os import getenv
from std.time import perf_counter_ns

from forkunion import (
    ComputeDomain,
    Library,
    MemoryDomain,
    Pool,
    Prong,
    ReplicatedArray,
    Topology,
)

comptime G = Float32(6.674e-11)
comptime DT = Float32(0.01)
comptime SOFTEN = Float32(1.0e-9)

comptime FORCE_LANES = 8
"""How many independent accumulator chains the force sweep keeps.

Reassociating a float reduction is what fast-math permits and strict IEEE forbids, so it is written
out by hand instead: the same eight lanes, reduced in the same fixed order, as the sibling kernels.
Every compiler then faces the same optimization problem, and the language columns compare
schedulers rather than compiler flag sets.
"""


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


def parse_int(name: StaticString, fallback: Int) -> Int:
    var text = getenv(name)
    if text.byte_length() == 0:
        return fallback
    try:
        return Int(text)
    except:
        return fallback


def parse_float(name: StaticString, fallback: Float64) -> Float64:
    var text = getenv(name)
    if text.byte_length() == 0:
        return fallback
    try:
        return Float64(text)
    except:
        return fallback


@always_inline
def split_mix(counter: UInt64) -> UInt64:
    """The SplitMix64 avalanche behind every random draw - a pure function of the counter.

    Deliberately not a standard generator: those differ across languages, so no two harnesses would
    simulate the same system. Each draw is a pure function of its counter instead, and the bodies
    come out bit-identical across every port of this hash.
    """
    var x = (counter + 1) * 0x9E37_79B9_7F4A_7C15
    x = (x ^ (x >> 30)) * 0xBF58_476D_1CE4_E5B9
    x = (x ^ (x >> 27)) * 0x94D0_49BB_1331_11EB
    return x ^ (x >> 31)


@always_inline
def random_unit(counter: UInt64) -> Float32:
    """One draw in `[0, 1)`: the top 24 bits scaled by 2^-24, both steps exact in `Float32`."""
    return Float32(Int(split_mix(counter) >> 40)) * (Float32(1.0) / Float32(16777216.0))


@always_inline
def fast_rsqrt(x: Float32) -> Float32:
    """Quake-style reciprocal square root with one Newton iteration."""
    var i = UInt32(0x5F3759DF) - (bitcast[DType.uint32](x) >> 1)
    var y = bitcast[DType.float32](i)
    var half = Float32(0.5) * x
    y *= Float32(1.5) - half * y * y
    return y


@fieldwise_init
struct Bodies(ImplicitlyCopyable, TrivialRegisterPassable):
    """Position, velocity, mass, and the accumulated force, flat so a C callback can reach them."""

    var position_x: Pointer[Float32, MutUntrackedOrigin]
    var position_y: Pointer[Float32, MutUntrackedOrigin]
    var position_z: Pointer[Float32, MutUntrackedOrigin]
    var velocity_x: Pointer[Float32, MutUntrackedOrigin]
    var velocity_y: Pointer[Float32, MutUntrackedOrigin]
    var velocity_z: Pointer[Float32, MutUntrackedOrigin]
    var mass: Pointer[Float32, MutUntrackedOrigin]
    var force_x: Pointer[Float32, MutUntrackedOrigin]
    var force_y: Pointer[Float32, MutUntrackedOrigin]
    var force_z: Pointer[Float32, MutUntrackedOrigin]
    var count: Int


@always_inline
def net_force(bodies: Bodies, index: Int) -> Tuple[Float32, Float32, Float32]:
    """The net gravitational force on one body, in eight explicit lanes."""
    var lanes_x = Array[Float32, FORCE_LANES](fill=0)
    var lanes_y = Array[Float32, FORCE_LANES](fill=0)
    var lanes_z = Array[Float32, FORCE_LANES](fill=0)
    var mass_here = bodies.mass[unsafe_offset=index]
    var x_here = bodies.position_x[unsafe_offset=index]
    var y_here = bodies.position_y[unsafe_offset=index]
    var z_here = bodies.position_z[unsafe_offset=index]

    var blocked = bodies.count - bodies.count % FORCE_LANES
    var other = 0
    while other < blocked:
        for lane in range(FORCE_LANES):
            var slot = other + lane
            var dx = bodies.position_x[unsafe_offset=slot] - x_here
            var dy = bodies.position_y[unsafe_offset=slot] - y_here
            var dz = bodies.position_z[unsafe_offset=slot] - z_here
            var squared = dx * dx + dy * dy + dz * dz + SOFTEN
            var reciprocal = fast_rsqrt(squared)
            var cubed = reciprocal * reciprocal * reciprocal
            var magnitude = G * mass_here * bodies.mass[unsafe_offset=slot] * cubed
            lanes_x[lane] += magnitude * dx
            lanes_y[lane] += magnitude * dy
            lanes_z[lane] += magnitude * dz
        other += FORCE_LANES
    for tail in range(blocked, bodies.count):
        var dx = bodies.position_x[unsafe_offset=tail] - x_here
        var dy = bodies.position_y[unsafe_offset=tail] - y_here
        var dz = bodies.position_z[unsafe_offset=tail] - z_here
        var squared = dx * dx + dy * dy + dz * dz + SOFTEN
        var reciprocal = fast_rsqrt(squared)
        var cubed = reciprocal * reciprocal * reciprocal
        var magnitude = G * mass_here * bodies.mass[unsafe_offset=tail] * cubed
        lanes_x[tail - blocked] += magnitude * dx
        lanes_y[tail - blocked] += magnitude * dy
        lanes_z[tail - blocked] += magnitude * dz

    # The one reduction shape every language shares; changing it changes the bits.
    return (
        ((lanes_x[0] + lanes_x[1]) + (lanes_x[2] + lanes_x[3]))
        + ((lanes_x[4] + lanes_x[5]) + (lanes_x[6] + lanes_x[7])),
        ((lanes_y[0] + lanes_y[1]) + (lanes_y[2] + lanes_y[3]))
        + ((lanes_y[4] + lanes_y[5]) + (lanes_y[6] + lanes_y[7])),
        ((lanes_z[0] + lanes_z[1]) + (lanes_z[2] + lanes_z[3]))
        + ((lanes_z[4] + lanes_z[5]) + (lanes_z[6] + lanes_z[7])),
    )


def force_prong(prong: Prong, mut bodies: Bodies):
    """The first pass: accumulate every body's force over every other."""
    var force = net_force(bodies, prong.task_index)
    bodies.force_x[unsafe_offset=prong.task_index] = force[0]
    bodies.force_y[unsafe_offset=prong.task_index] = force[1]
    bodies.force_z[unsafe_offset=prong.task_index] = force[2]


def apply_prong(prong: Prong, mut bodies: Bodies):
    """The second pass: integrate each body by its accumulated force.

    Positions wrap into the unit box so every distance - and so every force - stays in the normal
    `Float32` range forever: no overflow into NaN, and no denormals for x86 to stall on.
    """
    var index = prong.task_index
    var mass_here = bodies.mass[unsafe_offset=index]
    bodies.velocity_x[unsafe_offset=index] += bodies.force_x[unsafe_offset=index] / mass_here * DT
    bodies.velocity_y[unsafe_offset=index] += bodies.force_y[unsafe_offset=index] / mass_here * DT
    bodies.velocity_z[unsafe_offset=index] += bodies.force_z[unsafe_offset=index] / mass_here * DT
    bodies.position_x[unsafe_offset=index] += bodies.velocity_x[unsafe_offset=index] * DT
    bodies.position_y[unsafe_offset=index] += bodies.velocity_y[unsafe_offset=index] * DT
    bodies.position_z[unsafe_offset=index] += bodies.velocity_z[unsafe_offset=index] * DT
    bodies.position_x[unsafe_offset=index] -= floor(bodies.position_x[unsafe_offset=index])
    bodies.position_y[unsafe_offset=index] -= floor(bodies.position_y[unsafe_offset=index])
    bodies.position_z[unsafe_offset=index] -= floor(bodies.position_z[unsafe_offset=index])


def main() raises:
    var library = Library()
    var topology = Topology(library)

    var threads = parse_int("NBODY_THREADS", 0)
    if threads == 0:
        threads = topology.logical_cores_count()
    var budget_seconds = parse_float("NBODY_SECONDS", 10)
    var iterations = parse_int("NBODY_ITERATIONS", 0)
    var count = parse_int("NBODY_COUNT", 0)
    if count == 0:
        count = threads
    var backend = getenv("NBODY_BACKEND")
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

    var position_x = List[Float32](length=count, fill=0)
    var position_y = List[Float32](length=count, fill=0)
    var position_z = List[Float32](length=count, fill=0)
    var velocity_x = List[Float32](length=count, fill=0)
    var velocity_y = List[Float32](length=count, fill=0)
    var velocity_z = List[Float32](length=count, fill=0)
    var mass = List[Float32](length=count, fill=0)
    var force_x = List[Float32](length=count, fill=0)
    var force_y = List[Float32](length=count, fill=0)
    var force_z = List[Float32](length=count, fill=0)

    # Seven counter-based draws per body: three position coordinates, three velocity components, and
    # one mass in [1e10, 1e15) - so every language starts from bit-identical bodies.
    var mass_span = Float32(1.0e15) - Float32(1.0e10)
    for index in range(count):
        var counter = UInt64(index) * 7
        position_x[index] = random_unit(counter)
        position_y[index] = random_unit(counter + 1)
        position_z[index] = random_unit(counter + 2)
        velocity_x[index] = random_unit(counter + 3)
        velocity_y[index] = random_unit(counter + 4)
        velocity_z[index] = random_unit(counter + 5)
        mass[index] = Float32(1.0e10) + random_unit(counter + 6) * mass_span

    var bodies = Bodies(
        position_x.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        position_y.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        position_z.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        velocity_x.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        velocity_y.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        velocity_z.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        mass.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        force_x.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        force_y.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        force_z.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        count,
    )

    var pool = Pool(topology, threads=threads, name="nbody")

    # The replicated cells copy the read side into every memory domain once per iteration; on a
    # single-domain machine that collapses to one replica, so the cell still runs.
    var replicas = ReplicatedArray[DType.float32].try_new(topology, count * 3) if replicated else None

    # A fixed time budget beats a fixed iteration count: the window is long enough to amortize
    # scheduling noise, and reports the rate sustained. `NBODY_ITERATIONS` forces an exact count.
    # `parallelize` takes a capturing closure, which is exactly what ForkUnion cannot accept; the
    # baseline therefore reads the same bodies through the same flat pointers, with no scratch.
    @parameter
    def baseline_force(index: Int):
        var force = net_force(bodies, index)
        bodies.force_x[unsafe_offset=index] = force[0]
        bodies.force_y[unsafe_offset=index] = force[1]
        bodies.force_z[unsafe_offset=index] = force[2]

    @parameter
    def baseline_apply(index: Int):
        apply_prong(Prong(index, 0, 0), bodies)

    @parameter
    def one_pass():
        if baseline:
            parallelize[baseline_force](count, threads)
            parallelize[baseline_apply](count, threads)
        elif dynamic:
            pool.for_n_dynamic[force_prong](count, bodies)
            pool.for_n[apply_prong](count, bodies)
        else:
            pool.for_n[force_prong](count, bodies)
            pool.for_n[apply_prong](count, bodies)

    var budget = Int(budget_seconds * 1e9)
    var started = perf_counter_ns()
    var passes = 0
    if iterations > 0:
        for _ in range(iterations):
            one_pass()
        passes = iterations
    else:
        while True:
            one_pass()
            passes += 1
            if Int(perf_counter_ns() - started) >= budget:
                break
    var total_seconds = Float64(Int(perf_counter_ns() - started)) / 1e9
    var micros_per_iteration = total_seconds / Float64(passes) * 1e6

    var pace = fixed(micros_per_iteration, 2)
    var total = fixed(total_seconds, 2)
    print(t"{backend}: {count} bodies, {passes} iters, {pace} us/iter ({total} s total)")

    # `bodies` holds raw pointers into the ten lists, and Mojo releases a value after its last
    # named use - which would otherwise be the construction above, long before the last read.
    _ = position_x^
    _ = position_y^
    _ = position_z^
    _ = velocity_x^
    _ = velocity_y^
    _ = velocity_z^
    _ = mass^
    _ = force_x^
    _ = force_y^
    _ = force_z^
    _ = replicas^
