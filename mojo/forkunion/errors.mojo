"""The one error type the binding raises, so every fallible entry point declares the same one.

Mojo 1.0 allows at most one error type per function and never widens a typed `raises` into a
plain one, so a second error type here would force every caller to catch and convert. The
`detail` names the symbol or the argument, which is what turns a failure into a diagnosis.

Absence is not failure: a memory domain the machine does not have, a zero-byte allocation, and
a page the kernel refused all answer `Optional` rather than raising.
"""


@fieldwise_init
struct ErrorKind(Equatable, ImplicitlyCopyable, TrivialRegisterPassable, Writable):
    """Why a call into the C core failed."""

    var code: Int32
    comptime LIBRARY_MISSING = Self(0)
    """Nothing to `dlopen`, or what loaded is a different major version."""
    comptime SYMBOL_MISSING = Self(1)
    """The library loaded but does not export a symbol the binding needs."""
    comptime CREATION_FAILED = Self(2)
    """A `fu_*_new` answered NULL, which the C API documents as an allocation failure."""
    comptime SPAWN_FAILED = Self(3)
    """`fu_pool_spawn` refused the requested width or placement."""
    comptime INVALID_PARAMETER = Self(4)
    """Rejected here rather than forwarded, because C would answer the same for other reasons."""

    def write_to(self, mut writer: Some[Writer]):
        if self == Self.LIBRARY_MISSING:
            writer.write("the core is not on the loader path")
        elif self == Self.SYMBOL_MISSING:
            writer.write("the loaded core is missing a symbol")
        elif self == Self.CREATION_FAILED:
            writer.write("the core could not create the handle")
        elif self == Self.SPAWN_FAILED:
            writer.write("the core could not spawn the pool")
        else:
            writer.write("the argument was rejected")


@fieldwise_init
struct ForkUnionError(Copyable, ImplicitlyCopyable, Writable):
    """What went wrong reaching the C core, and which symbol or argument it was."""

    var kind: ErrorKind
    var detail: StaticString

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ForkUnion: ", self.kind, " [", self.detail, "]")
