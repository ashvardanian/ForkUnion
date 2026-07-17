"""Route OpenMP-compiled libraries onto the ForkUnion pool, without recompiling them.

Import this **before** anything that pulls in libgomp::

    import forkunion.openmp   # first
    import torch, sklearn, xgboost

Order is the whole trick. The shim has to enter the global symbol scope before libgomp arrives as some library's
``DT_NEEDED``, because the first definition in that scope wins. Import it too late and the interposition silently does
nothing -- measured: 4 ``GOMP_parallel`` calls intercepted when imported first, 0 when imported after ``torch``, with
torch working fine either way. A silent no-op is the worst outcome, so this module raises instead.

For notebooks, or when something already imported libgomp, use the order-independent launcher::

    python -m forkunion.omp your_script.py
"""

import ctypes
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

__all__ = ["shim_path", "loaded_openmp_runtimes", "is_active", "activate"]


def shim_path() -> str:
    """Path to the interposing shim shipped in this wheel."""
    name = {"darwin": "libforkunion_gomp.dylib", "win32": "forkunion_gomp.dll"}.get(sys.platform, "libforkunion_gomp.so")
    return os.path.join(_HERE, name)


def loaded_openmp_runtimes() -> list:
    """OpenMP runtimes already mapped into this process.

    Matches the vendored, hash-renamed spellings too -- `auditwheel` rewrites the `SONAME`, so scikit-learn and torch
    can carry `libgomp-e985bcbb.so.1.0.0` and a plain `libgomp.so.1` into the same interpreter.
    """
    if not os.path.exists("/proc/self/maps"):
        return []  # ? Not Linux; the caller gets no guard, only the launcher
    found = set()
    with open("/proc/self/maps") as maps:
        for line in maps:
            match = re.search(r"/([^/\s]*lib(?:gomp|iomp5|omp)[^/\s]*\.(?:so|dylib)[^\s]*)", line)
            if match and "forkunion" not in match.group(1):
                found.add(match.group(1))
    return sorted(found)


def is_active() -> bool:
    """True if the shim is already in this process, e.g. preloaded by ``python -m forkunion.omp``."""
    if not os.path.exists("/proc/self/maps"):
        return False
    with open("/proc/self/maps") as maps:
        return any("forkunion_gomp" in line for line in maps)


def activate(force: bool = False) -> str:
    """Load the shim into the global symbol scope. Returns the path loaded.

    Raises if an OpenMP runtime is already mapped, because interposing after the fact does nothing at all and failing
    loudly beats a silent no-op. Pass ``force=True`` to load anyway -- useful only to inspect the shim itself.
    """
    # Under the launcher the shim is already preloaded and libgomp being mapped means nothing - the shim won the
    # lookup regardless. Checking this first keeps `import forkunion.openmp` harmless inside `python -m forkunion.omp`.
    if is_active():
        return shim_path()

    already = loaded_openmp_runtimes()
    if already and not force:
        raise RuntimeError(
            f"{', '.join(already)} is already loaded, so forkunion.openmp cannot interpose it.\n"
            "Import forkunion.openmp before torch / sklearn / xgboost, "
            "or run `python -m forkunion.omp your_script.py` instead."
        )
    path = shim_path()
    if not os.path.exists(path):
        raise RuntimeError(f"The OpenMP shim was not built into this wheel: {path}")
    # `RTLD_GLOBAL` is what puts these symbols in front of the libgomp some library loads later.
    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    return path


activate()
