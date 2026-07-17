"""Order-independent launcher: ``python -m forkunion.omp your_script.py``.

Re-executes the interpreter with the shim in ``LD_PRELOAD``, so it wins the symbol lookup no matter what any library
imports or when. ``forkunion.openmp`` is the nicer spelling, but it only works if nothing has loaded libgomp yet -- in
a notebook, or under a launcher that imports torch for you, that cannot be guaranteed.

    python -m forkunion.omp train.py --epochs 3
    python -m forkunion.omp -m pytest tests/
"""

import os
import sys

from .openmp import shim_path

_PRELOAD_VAR = "DYLD_INSERT_LIBRARIES" if sys.platform == "darwin" else "LD_PRELOAD"
_GUARD_VAR = "FORKUNION_OMP_ACTIVE"


def main(argv: list) -> int:
    if not argv:
        print(__doc__.strip(), file=sys.stderr)
        return 2

    if sys.platform == "win32":
        print("Windows has no LD_PRELOAD; import forkunion.openmp first instead.", file=sys.stderr)
        return 1

    path = shim_path()
    if not os.path.exists(path):
        print(f"The OpenMP shim was not built into this wheel: {path}", file=sys.stderr)
        return 1

    # Without a guard the re-executed child re-enters this module and re-executes forever.
    if os.environ.get(_GUARD_VAR) == "1":
        print("forkunion.omp re-entered itself; refusing to fork-bomb.", file=sys.stderr)
        return 1

    environment = dict(os.environ)
    existing = environment.get(_PRELOAD_VAR, "")
    environment[_PRELOAD_VAR] = f"{path}{os.pathsep if sys.platform == 'darwin' else ':'}{existing}".rstrip(":")
    environment[_GUARD_VAR] = "1"

    os.execve(sys.executable, [sys.executable] + argv, environment)
    return 0  # ? `execve` does not return


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
