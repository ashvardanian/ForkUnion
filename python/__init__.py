"""NUMA-aware fork-join thread pool, packaged so other extensions can build against it.

Two ways to consume this from a native extension:

1. Link `libforkunion.so.3` shipped in this wheel, using `get_include()` and `get_library_dir()`.
2. Import the `PyCapsule` and call through its vtable, taking the *shared* pool with no linking at all.

The second is what keeps one process to one pool. See `capsule.h`.
"""

import os
import sys

__version__ = "3.0.0"

_HERE = os.path.dirname(os.path.abspath(__file__))

# Windows has no RPATH, so a consumer's `.pyd` cannot find `forkunion.dll` on its own. Adding the directory here means
# `import forkunion` before the consumer's own extension is enough - which is why consumers must import us first.
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(_HERE)

from . import _core  # noqa: E402  (must follow `add_dll_directory` on Windows)

# `PyCapsule_Import("forkunion._C_API_v3")` imports this package and reads this attribute off it, so the capsule has to
# live here rather than on the private `_core` submodule. See `capsule.h`.
_C_API_v3 = _core._C_API_v3


def get_include() -> str:
    """Header directory for `Extension(include_dirs=[...])`."""
    return os.path.join(_HERE, "include")


def get_library_dir() -> str:
    """Directory holding `libforkunion.so.3`, for `Extension(library_dirs=[...])`."""
    return _HERE


def get_libraries() -> list:
    """Libraries to link, for `Extension(libraries=[...])`. Call `create_library_symlinks()` first."""
    return ["forkunion"]


def create_library_symlinks() -> None:
    """Create the `libforkunion.so` alias that `-lforkunion` needs, once, at a consumer's build time.

    A wheel is a ZIP and ZIP cannot store symlinks, so shipping both `libforkunion.so.3` and its unversioned alias
    would mean shipping the same 2 MB library twice. Only the versioned name is packaged - it is what `SONAME` records
    and therefore what the loader looks for - and the alias `-l` wants is made here instead.

    Safe to call repeatedly; a no-op on Windows, where the import library already carries the linkable name.
    """
    if sys.platform == "win32":
        return
    versioned = "libforkunion.3.dylib" if sys.platform == "darwin" else "libforkunion.so.3"
    alias = "libforkunion.dylib" if sys.platform == "darwin" else "libforkunion.so"
    target, link = os.path.join(_HERE, versioned), os.path.join(_HERE, alias)
    if not os.path.exists(target) or os.path.exists(link):
        return
    try:
        os.symlink(versioned, link)
    except OSError:
        # A read-only or symlink-hostile site-packages; copying is wasteful but keeps the build working.
        import shutil

        shutil.copy2(target, link)


def get_runtime_library_dirs() -> list:
    """RPATH entries so a consumer's extension resolves `libforkunion.so.3` at import.

    Assumes the consumer's package sits beside this one in `site-packages`; some conda and venv layouts do not.
    """
    if sys.platform == "win32":
        return []  # ? No RPATH; `add_dll_directory` above covers it
    origin = "@loader_path" if sys.platform == "darwin" else "$ORIGIN"
    return [f"{origin}/../forkunion"]


def threads() -> int:
    """Threads in the shared pool. Zero before the first dispatch, and in a `fork()`-ed child."""
    return _core.threads()


def spawn() -> int:
    """Spawn the shared pool now rather than on first dispatch. Returns the thread count."""
    return _core.spawn()


def logical_cores() -> int:
    """Logical cores visible to this process, honoring the CPU affinity mask."""
    return _core.logical_cores()


def compute_domains() -> int:
    """Compute domains the machine reports."""
    return _core.compute_domains()


def memory_domains() -> int:
    """Memory domains - NUMA nodes - the machine reports."""
    return _core.memory_domains()


def capabilities() -> str:
    """Capabilities detected on the running host, as opposed to the build host."""
    return _core.capabilities()


__all__ = [
    "get_include",
    "get_library_dir",
    "get_libraries",
    "get_runtime_library_dirs",
    "threads",
    "spawn",
    "logical_cores",
    "compute_domains",
    "memory_domains",
    "capabilities",
]
