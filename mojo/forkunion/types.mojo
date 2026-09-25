"""The portable building blocks: what a dispatch hands a callback, and how a machine is indexed.

The domain wrappers exist because the C API spells three different things `size_t` or `int`: a
dense compute-domain index, a dense memory-domain index, and the operating system's own memory
domain id. Handing a compute-domain index to an allocator that wants a memory-domain id compiles
in C, works on every machine where the two happen to agree, and misplaces pages where they do not.

The second half is pure logic with no FFI - the fair-chunk splitter, the cache-line padding
wrapper, and the raw-pointer views that let disjoint slices cross a C callback boundary. Mojo
cannot hand a capturing closure to C, so these are what a caller reaches for most.

`Error` lives here too, and is the only error type the binding raises: Mojo 1.0 allows at most one
per function and never widens a typed `raises` into a plain one, so a second would force every
caller to catch and convert. A refusal is a failure, not an absence - a memory domain the machine
does not have and a zero-byte allocation both raise rather than answering `Optional`.
"""

from std.ffi import c_int, c_size_t
from std.sys import size_of

# region Pointers

comptime Handle = Pointer[NoneType, MutUntrackedOrigin]
"""An opaque `fu_topology_t`, `fu_pool_t`, or `fu_fabric_t`."""

comptime Context = Pointer[NoneType, MutUntrackedOrigin]
"""The type-punned callback context the C API carries through a dispatch."""

comptime CString = Pointer[Int8, ImmUntrackedOrigin]
"""A borrowed null-terminated C string."""

comptime OutSize = Pointer[c_size_t, MutAnyOrigin]
"""A `size_t *` the C API writes through.

`MutAnyOrigin` rather than `MutUntrackedOrigin` is load-bearing: it is what tells the compiler the
callee may write anywhere. With a narrower origin it folds a read of the slot back to whatever the
caller last stored there, and every out-parameter reads back as zero.
"""

comptime OutHandle = Pointer[Int, MutAnyOrigin]
"""A handle-sized slot the C API writes through, with the same aliasing caveat as `OutSize`."""

comptime OutBytes = Pointer[Int8, MutAnyOrigin]
"""A `char *` buffer the C API fills, with the same aliasing caveat as `OutSize`."""

comptime OutAddress = Pointer[Int, MutAnyOrigin]
"""A `void **` slot the C API writes an allocation into, with the same aliasing caveat as `OutSize`."""

# endregion Pointers

# region Errors


@fieldwise_init
struct ErrorKind(Equatable, ImplicitlyCopyable, TrivialRegisterPassable, Writable):
    """Why a call into the C core failed."""

    var code: Int32
    """The raw `fu_status_t` value, kept even when this build does not name it."""
    comptime SUCCESS = Self(0)
    """The call completed."""
    comptime UNKNOWN = Self(-1)
    """No reason was reported, or one this build does not name."""
    comptime BAD_ALLOC = Self(-2)
    """An allocation or mapping failed; a smaller request may succeed."""
    comptime CAPACITY_EXHAUSTED = Self(-3)
    """A fixed ceiling was reached, so a smaller request will not help either."""
    comptime INVALID_ARGUMENT = Self(-4)
    """An argument was malformed, out of range, or would overflow a byte count."""
    comptime CONFIG_MISMATCH = Self(-5)
    """The handles or the pool kind cannot serve this call together."""
    comptime ALREADY_SPAWNED = Self(-6)
    """The pool is already spawned; terminate it first."""
    comptime NOT_SPAWNED = Self(-7)
    """The pool was never spawned."""
    comptime THREAD_REFUSED = Self(-8)
    """The OS declined to create a thread - a resource limit, or permissions."""
    comptime TOPOLOGY_UNAVAILABLE = Self(-9)
    """The machine could not be described."""
    comptime PERMISSION_DENIED = Self(-10)
    """A privileged operation was declined."""
    comptime UNSUPPORTED = Self(-11)
    """This build or this machine has no such facility."""
    comptime LIBRARY_MISSING = Self(-12)
    """Nothing to `dlopen`, or what loaded is a different major version."""
    comptime SYMBOL_MISSING = Self(-13)
    """The library loaded but does not export a symbol the binding needs."""

    def write_to(self, mut writer: Some[Writer]):
        """Spells the failure out in prose, so a code this build does not name still reads.

        Args:
            writer: The sink the reason is appended to.
        """
        if self == Self.LIBRARY_MISSING:
            writer.write("the core is not on the loader path")
        elif self == Self.SYMBOL_MISSING:
            writer.write("the loaded core is missing a symbol")
        elif self == Self.BAD_ALLOC:
            writer.write("an allocation failed")
        elif self == Self.CAPACITY_EXHAUSTED:
            writer.write("a fixed capacity was exhausted")
        elif self == Self.INVALID_ARGUMENT:
            writer.write("an argument was rejected")
        elif self == Self.CONFIG_MISMATCH:
            writer.write("the handles cannot serve this call together")
        elif self == Self.ALREADY_SPAWNED:
            writer.write("the pool is already spawned")
        elif self == Self.NOT_SPAWNED:
            writer.write("the pool was never spawned")
        elif self == Self.THREAD_REFUSED:
            writer.write("the OS declined to create a thread")
        elif self == Self.TOPOLOGY_UNAVAILABLE:
            writer.write("the machine could not be described")
        elif self == Self.PERMISSION_DENIED:
            writer.write("a privileged operation was declined")
        elif self == Self.UNSUPPORTED:
            writer.write("this build has no such facility")
        else:
            writer.write("the core reported no reason")

    @staticmethod
    def of(status: c_int) -> Self:
        """Lifts a raw `fu_status_t`, keeping an unnamed one rather than guessing.

        Args:
            status: Whatever the C call returned, named by this build or not.

        Returns:
            The matching kind, or one carrying the unrecognized code verbatim.
        """
        return Self(Int32(status))


@fieldwise_init
struct Error(Copyable, ImplicitlyCopyable, Writable):
    """What went wrong reaching the C core, and which symbol or argument it was."""

    var kind: ErrorKind
    """Which class of failure the C core reported."""
    var detail: StaticString
    """The symbol or argument it was about, for a reader rather than for a branch."""

    def write_to(self, mut writer: Some[Writer]):
        """Writes the reason and the detail together, prefixed so the source is unambiguous.

        Args:
            writer: The sink the message is appended to.
        """
        writer.write("ForkUnion: ", self.kind, " [", self.detail, "]")


# endregion Errors

comptime DEFAULT_ALIGNMENT = 128
"""Two cache lines, because most x86 parts prefetch in pairs, matching the C++ `default_alignment_k`."""


def bytes_for_elements(count: Int, element_bytes: Int) raises Error -> Int:
    """Bytes occupied by `count` elements of `element_bytes` each, refusing a product that wraps.

    Args:
        count: How many elements the buffer holds.
        element_bytes: The width of one element.

    Returns:
        The exact product, because the overflowing case raises rather than truncating.

    Raises:
        With `INVALID_ARGUMENT`, when the product would exceed `Int.MAX`.
    """
    if element_bytes != 0 and count > Int.MAX // element_bytes:
        raise Error(ErrorKind.INVALID_ARGUMENT, "the element count times the element size would wrap")
    return count * element_bytes


# region Dispatch


@fieldwise_init
struct ThreadInDomain(ImplicitlyCopyable, TrivialRegisterPassable):
    """One thread, situated in one compute domain - the "where I am" every callback receives.

    Every dispatch hands its callback two things: the work, and this. A callback placing memory
    reads `compute_domain` to find the node it runs on; one indexing per-thread scratch reads
    `thread`.
    """

    var thread: Int
    """The physical thread executing the work."""
    var compute_domain: Int
    """The compute domain - a same-QoS core cluster - the thread runs on."""


@fieldwise_init
struct CallerExclusivity(Equatable, ImplicitlyCopyable, TrivialRegisterPassable, Writable):
    """Whether the thread that dispatches also executes, or only coordinates."""

    var identifier: c_int
    """The raw `fu_caller_exclusivity_t` value the C API exchanges."""
    comptime INCLUSIVE = Self(0)
    """The calling thread owes one slice of the work, which runs inside the join."""
    comptime EXCLUSIVE = Self(1)
    """The calling thread only coordinates, so a dispatch can be polled before it is joined."""

    def write_to(self, mut writer: Some[Writer]):
        """Names which of the two policies this is.

        Args:
            writer: The sink the name is appended to.
        """
        writer.write("inclusive" if self == Self.INCLUSIVE else "exclusive")


# endregion Dispatch

# region Machine


@fieldwise_init
struct ComputeDomain(Equatable, ImplicitlyCopyable, TrivialRegisterPassable, Writable):
    """A dense index into this machine's compute domains, counted from zero."""

    var index: Int
    """The zero-based position among this machine's compute domains."""

    def write_to(self, mut writer: Some[Writer]):
        """Writes the index with the kind of domain spelled out, so a log line cannot be misread.

        Args:
            writer: The sink the label is appended to.
        """
        writer.write("compute domain ", self.index)


@fieldwise_init
struct MemoryDomain(Equatable, ImplicitlyCopyable, TrivialRegisterPassable, Writable):
    """A dense index into this machine's memory domains, counted from zero."""

    var index: Int
    """The zero-based position among this machine's memory domains."""

    def write_to(self, mut writer: Some[Writer]):
        """Writes the index labeled as a dense one, never bare, so it cannot pass for an OS id.

        Args:
            writer: The sink the label is appended to.
        """
        writer.write("memory domain ", self.index)


@fieldwise_init
struct MemoryDomainId(Equatable, ImplicitlyCopyable, TrivialRegisterPassable, Writable):
    """The operating system's own id for a memory domain, which is not an index.

    The allocators key off this rather than off the topology, so an allocation can outlive the
    handle that found it. Reach it through `Topology.memory_domain_id_at_index`, never by reusing
    a dense index.
    """

    var identifier: c_int
    """The id the operating system uses, which need not match any dense index."""

    def write_to(self, mut writer: Some[Writer]):
        """Writes it as the NUMA node the operating system itself calls it.

        Args:
            writer: The sink the label is appended to.
        """
        writer.write("NUMA node ", self.identifier)


@fieldwise_init
struct Capabilities(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """The bit set of spin hints and placement facilities a build or a machine offers.

    One bit per facility, and two accessors ask two questions of the same bit. `comptime_capabilities`
    reports whether the code was built with a path; `runtime_capabilities` reports whether the machine
    offers it now. Neither implies the other.
    """

    var bits: UInt32
    """One bit per facility, laid out as the C `fu_capability_t` enumerators are."""

    comptime NONE = Self(0)
    """Nothing detected, or nothing compiled in."""
    comptime X86_PAUSE = Self(1 << 0)
    """The `PAUSE` spin hint, on every x86 since the Pentium 4."""
    comptime X86_TPAUSE = Self(1 << 1)
    """`TPAUSE` sleeps the core until a deadline. Needs the `WAITPKG` feature."""
    comptime ARM64_YIELD = Self(1 << 2)
    """The `YIELD` hint, on every AArch64."""
    comptime ARM64_WFET = Self(1 << 3)
    """`WFET` sleeps the core until a deadline or an event. Needs `FEAT_WFxT`."""
    comptime RISC5_PAUSE = Self(1 << 4)
    """The `PAUSE` spin hint, from the `Zihintpause` extension."""
    comptime RISC5_WRS = Self(1 << 5)
    """`WRS.STO` sleeps the hart until a reservation breaks. Needs the `Zawrs` extension."""
    comptime OS_THREADS = Self(1 << 6)
    """Own the raw OS thread handle instead of a `std::thread`."""
    comptime TOPOLOGY = Self(1 << 7)
    """Enumerate this machine's cores, compute domains, and memory domains."""
    comptime PLACE_THREADS_BY_AFFINITY = Self(1 << 8)
    """Bind a thread to a set of cores, choosing where it runs."""
    comptime PLACE_THREADS_BY_CORE_CLASS = Self(1 << 9)
    """Steer a thread onto a class of core at creation."""
    comptime RESCHEDULE_THREADS_BY_CLASS = Self(1 << 10)
    """Reclass a thread's scheduler to sleep or wake it."""
    comptime PLACE_MEMORY_ON_DOMAIN = Self(1 << 11)
    """Place a buffer's pages on a chosen memory domain."""
    comptime PLACE_HUGE_PAGES_ON_DOMAIN = Self(1 << 12)
    """Place larger-than-base pages on a chosen memory domain."""
    comptime HUGE_TRANSPARENT_PAGES = Self(1 << 13)
    """The kernel promotes base pages to huge pages on its own."""
    comptime COLOCATE_POOLS_ON_DOMAIN = Self(1 << 14)
    """The domain-aware colocated and distributed pools are compiled."""
    comptime X86_CLDEMOTE = Self(1 << 15)
    """`CLDEMOTE` moves a just-written line toward the shared LLC. Reporting only."""
    comptime ARM64_DC_CVAC = Self(1 << 16)
    """`DC CVAC` cleans a dirty line to the coherency point."""
    comptime RISC5_ZICBOM = Self(1 << 17)
    """User-mode Zicbom cache-block management, attested through `hwprobe`."""

    comptime ANY_YIELD = Self(0x3F)
    """Every busy-wait waiter bit, to enumerate the ones a machine offers."""
    comptime ALL = Self(0xFFFF_FFFF)
    """All-ones allow-mask: hand to a pool to disable capability filtering."""

    def __contains__(self, other: Self) -> Bool:
        """Whether every bit of `other` is set here.

        Args:
            other: The facilities being asked about, one bit each.

        Returns:
            True only when all of them are present, so a multi-bit mask asks for all of them.
        """
        return (self.bits & other.bits) == other.bits

    def __or__(self, other: Self) -> Self:
        """The union, to build an allow-mask out of the named facilities.

        Args:
            other: The facilities to add.

        Returns:
            A set holding every bit either side had.
        """
        return Self(self.bits | other.bits)

    def __and__(self, other: Self) -> Self:
        """The intersection, to keep only what a build and a machine both offer.

        Args:
            other: The facilities to keep.

        Returns:
            A set holding only the bits both sides had.
        """
        return Self(self.bits & other.bits)


# endregion Machine

# region Pure Logic


comptime _TasksRangeIterator = type_of(range(0, 1))
"""What `range(first, last)` returns, which is not otherwise nameable in a signature."""


@fieldwise_init
struct TasksRange(Equatable, ImplicitlyCopyable, SizedRaising, TrivialRegisterPassable):
    """A half-open `[first, first + count)` run of task indices - the "what work" of a slice dispatch.

    Iterable, so a callback reads `for task in range` rather than rebuilding the bounds. An idle
    thread receives a range with `count == 0`, which every dispatch still calls exactly once.
    """

    var first: Int
    """The first task index in the run."""
    var count: Int
    """How many tasks the run covers; zero means an idle thread."""

    def __len__(self) -> Int:
        """How many tasks the run covers.

        Returns:
            The count, which is zero on the thread a dispatch left idle.
        """
        return self.count

    def __iter__(self) -> _TasksRangeIterator:
        """Walks the task indices, so a callback loops over the run rather than rebuilding it.

        Returns:
            A `range` over the half-open span, empty for an idle thread.
        """
        return range(self.first, self.first + self.count)

    def end(self) -> Int:
        """One past the last task index, so `first..end()` is the half-open span.

        Returns:
            The exclusive upper bound, which equals `first` when the run is empty.
        """
        return self.first + self.count

    def is_empty(self) -> Bool:
        """Whether the run covers no tasks at all.

        Returns:
            True on the idle thread every dispatch still calls exactly once.
        """
        return self.count == 0


struct IndexedSplit(ImplicitlyCopyable, TrivialRegisterPassable):
    """Splits a range of tasks into fair-sized runs, minimizing the spread across threads.

    The first `tasks % threads` runs get one extra task; the rest get the floor. Mirrors the C++
    `indexed_split`. See https://lemire.me/blog/2025/05/22/dividing-an-array-into-fair-sized-chunks/
    """

    var quotient: Int
    """The floor of tasks over threads, which every run is at least this long."""
    var remainder: Int
    """How many of the leading runs take one task more than the floor."""

    def __init__(out self, tasks_count: Int, threads_count: Int):
        """Precomputes the split once, leaving `get` free of division.

        Args:
            tasks_count: How many tasks are shared out; zero is allowed and leaves every run empty.
            threads_count: How many runs to cut, which must be positive.
        """
        debug_assert(threads_count > 0, "a split needs at least one thread")
        self.quotient = tasks_count // threads_count
        self.remainder = tasks_count % threads_count

    def get(self, thread_index: Int) -> TasksRange:
        """The run owned by `thread_index`.

        Args:
            thread_index: Which of the runs to describe, counted from zero.

        Returns:
            That thread's span, one task longer for the first `remainder` of them.
        """
        var first = self.quotient * thread_index + min(thread_index, self.remainder)
        var count = self.quotient + (1 if thread_index < self.remainder else 0)
        return TasksRange(first, count)


struct CacheAligned[T: ImplicitlyCopyable & Deinitable](Copyable):
    """Pads a value out to a full `DEFAULT_ALIGNMENT` so per-thread slots never false-share.

    Allocate one per thread as scratch, then combine after the parallel region. The padding is
    what makes an array of these stride by a whole line, which is the property that matters.

    Parameters:
        T: The payload, which must be no wider than `DEFAULT_ALIGNMENT`.
    """

    var value: Self.T
    """The payload, which a worker owns for the whole parallel region."""
    var _padding: Array[UInt8, DEFAULT_ALIGNMENT - size_of[Self.T]()]
    """Fills the rest of the line, so the next slot starts on the next one."""

    def __init__(out self, var value: Self.T):
        """Takes ownership of the payload and zeroes the line behind it.

        Args:
            value: The payload to move into the slot.
        """
        self.value = value^
        self._padding = Array[UInt8, DEFAULT_ALIGNMENT - size_of[Self.T]()](fill=0)


@fieldwise_init
struct SyncConstPointer[T: AnyType](ImplicitlyCopyable, TrivialRegisterPassable):
    """A read-only view letting one immutable buffer be read by every worker across a C callback.

    The caller guarantees the pointee outlives the parallel region and is not mutated while shared.

    Parameters:
        T: The element type, read unchanged by every worker for the whole region.
    """

    var pointer: Pointer[Self.T, ImmUntrackedOrigin]
    """The first element, its origin erased so the view survives the C callback boundary."""

    def at(self, index: Int) -> ref[ImmUntrackedOrigin] Self.T:
        """The element at `index`; the caller owns bounds checking.

        Args:
            index: An offset in elements, not in bytes.

        Returns:
            A read-only reference into the caller's buffer, live as long as that buffer is.
        """
        return self.pointer[unsafe_offset=index]


@fieldwise_init
struct SyncMutPointer[T: AnyType](ImplicitlyCopyable, TrivialRegisterPassable):
    """A writable view letting workers write disjoint slots of one buffer across a C callback.

    The caller guarantees each worker touches disjoint indices and the pointee outlives the region.

    Parameters:
        T: The element type, whose slots the workers write one apiece.
    """

    var pointer: Pointer[Self.T, MutUntrackedOrigin]
    """The first element, its origin erased so the view survives the C callback boundary."""

    def at(self, index: Int) -> ref[MutUntrackedOrigin] Self.T:
        """A reference to the element at `index`; the caller owns disjointness and bounds.

        Args:
            index: An offset in elements, not in bytes.

        Returns:
            A writable reference into the caller's buffer, unsynchronized against every other one.
        """
        return self.pointer[unsafe_offset=index]


# endregion Pure Logic
