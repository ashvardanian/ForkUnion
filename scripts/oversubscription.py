"""Counts the thread pools a scientific Python process ends up with, before and after the ForkUnion shim.

Run both halves and compare::

    python scripts/oversubscription.py                       # stock: every library brings its own pool
    python -m forkunion.omp scripts/oversubscription.py      # one pool for all of them

Reports pools, OpenMP runtimes, live OS threads, and wall clock. The thread count is the headline: nothing in the
ecosystem unifies these pools today, and `threadpoolctl` can only shrink each one separately.
"""

import os
import re
import sys
import time


def openmp_runtimes() -> list:
    """OpenMP runtimes mapped into this process, including `auditwheel`'s hash-renamed copies."""
    if not os.path.exists("/proc/self/maps"):
        return []
    found = set()
    with open("/proc/self/maps") as maps:
        for line in maps:
            match = re.search(r"/([^/\s]*lib(?:gomp|iomp5|omp)[^/\s]*\.so[^\s]*)", line)
            if match and "forkunion" not in match.group(1):
                found.add(match.group(1))
    return sorted(found)


def os_threads() -> int:
    return len(os.listdir("/proc/self/task")) if os.path.isdir("/proc/self/task") else -1


def shim_active() -> bool:
    if not os.path.exists("/proc/self/maps"):
        return False
    with open("/proc/self/maps") as maps:
        return any("forkunion_gomp" in line for line in maps)


def main() -> int:
    import numpy as np

    np.random.seed(0)
    samples, features = 20000, 20
    x = np.random.rand(samples, features)
    y = (x[:, 0] > 0.5).astype(int)

    started = time.perf_counter()

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=8, n_init=1, random_state=0).fit(x)

    import xgboost

    booster = xgboost.XGBClassifier(n_estimators=20, tree_method="hist", random_state=0).fit(x, y)

    import lightgbm

    forest = lightgbm.LGBMClassifier(n_estimators=20, verbose=-1, random_state=0).fit(x, y)

    elapsed = time.perf_counter() - started

    pools = "unavailable"
    try:
        from threadpoolctl import threadpool_info

        info = threadpool_info()
        pools = f"{len(info)} declaring {sum(p.get('num_threads', 0) for p in info)} threads"
    except ImportError:
        pass

    print(f"  shim active         : {shim_active()}")
    print(f"  OpenMP runtimes     : {len(openmp_runtimes())} {openmp_runtimes()}")
    print(f"  threadpoolctl pools : {pools}")
    print(f"  live OS threads     : {os_threads()}  (cores: {os.cpu_count()})")
    print(f"  wall clock          : {elapsed:.2f} s")

    # Results must not move: a scheduler swap that changes what a program computes is not a speedup.
    print(f"  kmeans inertia      : {kmeans.inertia_:.4f}")
    print(f"  xgboost mean pred   : {booster.predict_proba(x)[:, 1].mean():.8f}")
    print(f"  lightgbm mean pred  : {forest.predict_proba(x)[:, 1].mean():.8f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
