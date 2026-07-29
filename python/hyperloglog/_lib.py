"""ctypes bindings for the Mojo HyperLogLog kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.path.join(ROOT, "dist", "libmojo-hyperloglog.so")

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mhll_add_hashes": ([I, I, I, I, I], None),
    "mhll_merge": ([I, I, I], None),
    "mhll_register_stats": ([I, I, I], F),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJO_HYPERLOGLOG_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    manifest = os.path.join(ROOT, "pixi.toml")
    if os.path.exists(pixi) and os.path.exists(manifest):
        return [pixi, "run", "--manifest-path", manifest, "mojo"]
    raise BuildError("mojo not found; set MOJO_HYPERLOGLOG_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        if os.path.getmtime(LIB) >= max(os.path.getmtime(path) for path in sources):
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + [
        "build",
        "--emit",
        "shared-lib",
        os.path.join(SRC, "capi.mojo"),
        "-o",
        LIB,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    return array.ctypes.data


def _register_array(
    array: np.ndarray, *, writable: bool, name: str
) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if array.dtype != np.dtype(np.int8):
        raise TypeError(f"{name} must have dtype numpy.int8")
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if writable and not array.flags.writeable:
        raise ValueError(f"{name} must be writable")
    if array.size and (np.any(array < 0) or np.any(array > 63)):
        raise ValueError(f"{name} contains an invalid register rank")
    return array


def add_hashes(registers: np.ndarray, hashes: np.ndarray, p: int) -> None:
    registers = _register_array(registers, writable=True, name="registers")
    if not isinstance(hashes, np.ndarray):
        raise TypeError("hashes must be a numpy.ndarray")
    if hashes.dtype != np.dtype(np.uint64):
        raise TypeError("hashes must have dtype numpy.uint64")
    if hashes.ndim != 1:
        raise ValueError("hashes must be one-dimensional")
    if not hashes.flags.c_contiguous:
        raise ValueError("hashes must be C-contiguous")
    if not 4 <= p <= 16 or registers.size != 1 << p:
        raise ValueError("register length does not match precision")
    if hashes.size == 0:
        return
    lib().mhll_add_hashes(
        addr(registers), addr(hashes), hashes.size, p, registers.size
    )


def merge(registers: np.ndarray, other: np.ndarray) -> None:
    registers = _register_array(registers, writable=True, name="registers")
    other = _register_array(other, writable=False, name="other")
    if registers.size != other.size:
        raise ValueError("register arrays must have equal lengths")
    lib().mhll_merge(addr(registers), addr(other), registers.size)


def register_stats(registers: np.ndarray) -> tuple[float, int]:
    registers = _register_array(registers, writable=False, name="registers")
    zero_count = np.empty(1, dtype=np.int64)
    harmonic = lib().mhll_register_stats(
        addr(registers), registers.size, addr(zero_count)
    )
    return harmonic, int(zero_count[0])


def main() -> int:
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
