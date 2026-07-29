"""Benchmarks against hyperloglog 0.1.8 and equivalent NumPy kernels."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import math
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import hyperloglog as mojo_hll  # noqa: E402
from hyperloglog.hll import get_rho_vec  # noqa: E402


def load_upstream():
    distribution = importlib.metadata.distribution("hyperloglog")
    init_path = distribution.locate_file("hyperloglog/__init__.py")
    spec = importlib.util.spec_from_file_location(
        "upstream_hyperloglog",
        init_path,
        submodule_search_locations=[str(init_path.parent)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


upstream = load_upstream()


def timeit(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def numpy_add_hashes(hashes, p=14):
    registers = np.zeros(1 << p, dtype=np.int8)
    indices = hashes & np.uint64((1 << p) - 1)
    words = hashes >> np.uint64(p)
    ranks = get_rho_vec(words, 64 - p)
    unique, inverse = np.unique(indices, return_inverse=True)
    order = np.argsort(inverse)
    starts = np.searchsorted(inverse[order], np.arange(len(unique)))
    grouped = np.maximum.reduceat(ranks[order], starts)
    registers[unique] = np.maximum(registers[unique], grouped)
    return registers


def benchmark():
    rows = []

    values = range(200_000)
    rows.append(
        (
            "`add_bulk` 200k integers",
            lambda: mojo_hll.HyperLogLog(0.01).add_bulk(values),
            lambda: upstream.HyperLogLog(0.01).add_bulk(values),
            "upstream",
        )
    )

    rng = np.random.default_rng(42)
    hashes = rng.integers(0, np.iinfo(np.uint64).max, 2_000_000, dtype=np.uint64)
    rows.append(
        (
            "`add_hashes` 2M uint64",
            lambda: mojo_hll.HyperLogLog(0.01).add_hashes(hashes),
            lambda: numpy_add_hashes(hashes),
            "NumPy equivalent",
        )
    )

    ours = mojo_hll.HyperLogLog(0.005)
    theirs = upstream.HyperLogLog(0.005)
    ours.add_hashes(hashes)
    theirs.M[:] = ours.M

    def ours_card():
        for _ in range(1000):
            ours.card()

    def upstream_card():
        for _ in range(1000):
            theirs.card()

    rows.append(
        (
            "`card` p=16, 1000 calls",
            ours_card,
            upstream_card,
            "upstream",
        )
    )

    ours_left = mojo_hll.HyperLogLog(0.005)
    ours_right = mojo_hll.HyperLogLog(0.005)
    upstream_left = upstream.HyperLogLog(0.005)
    upstream_right = upstream.HyperLogLog(0.005)
    ours_left.M[:] = rng.integers(0, 20, ours_left.m, dtype=np.int8)
    ours_right.M[:] = rng.integers(0, 20, ours_right.m, dtype=np.int8)
    upstream_left.M[:] = ours_left.M
    upstream_right.M[:] = ours_right.M

    def ours_merge():
        for _ in range(1000):
            ours_left.update(ours_right)

    def upstream_merge():
        for _ in range(1000):
            upstream_left.update(upstream_right)

    rows.append(
        (
            "`update` p=16, 1000 merges",
            ours_merge,
            upstream_merge,
            "upstream",
        )
    )

    measured = []
    for name, ours_fn, reference_fn, reference_name in rows:
        ours_fn()
        reference_fn()
        ours_time = timeit(ours_fn)
        reference_time = timeit(reference_fn)
        measured.append((name, ours_time, reference_time, reference_name))
    return measured


def machine():
    cpu = platform.processor()
    if cpu in ("", "x86_64", "AMD64") and os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    return f"{cpu or platform.machine()}, {platform.system()} {platform.release()}"


def main():
    print(f"Machine: {machine()}")
    print()
    print("| operation | mojo-hyperloglog | reference | speedup |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours_time, reference_time, reference_name in benchmark():
        speedup = reference_time / ours_time
        print(
            f"| {name} | {ours_time * 1000:.2f} ms | "
            f"{reference_time * 1000:.2f} ms ({reference_name}) | {speedup:.2f}x |"
        )


if __name__ == "__main__":
    main()
