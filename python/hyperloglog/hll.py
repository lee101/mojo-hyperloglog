"""HyperLogLog cardinality estimation with Mojo register kernels."""

from __future__ import annotations

import math
from hashlib import sha1

import numpy as np
from msgpack import Packer, packb

from ._bias import load_bias_data
from ._lib import add_hashes as _add_hashes
from ._lib import _merge_trusted
from ._lib import register_stats as _register_stats

HLL_COUNTER_TYPE = np.int8

_THRESHOLDS = np.array(
    [10, 20, 40, 80, 220, 400, 900, 1800, 3100, 6500, 11500, 20000, 50000],
    dtype=float,
)
rawEstimateData, biasData = load_bias_data()


def get_treshold(p):
    return _THRESHOLDS[p - 4]


def get_nearest_neighbors(E, estimate_vector):
    return np.argsort((E - estimate_vector) ** 2)[:6]


def estimate_bias(E, p):
    bias_vector = biasData[p - 4]
    nearest_neighbors = get_nearest_neighbors(E, rawEstimateData[p - 4])
    return np.sum(bias_vector[nearest_neighbors]) / len(nearest_neighbors)


def get_alpha(p):
    if not (4 <= p <= 16):
        raise ValueError("p=%d should be in range [4 : 16]" % p)
    if p == 4:
        return 0.673
    if p == 5:
        return 0.697
    if p == 6:
        return 0.709
    return 0.7213 / (1.0 + 1.079 / (1 << p))


def get_rho(w, max_width):
    rho = max_width - w.bit_length() + 1
    if rho <= 0:
        raise ValueError("w overflow")
    return rho


def bit_length_vec(arr):
    arr = np.asarray(arr, dtype=np.uint64)
    bits = arr >> np.uint64(1)
    bits |= arr
    bits |= bits >> np.uint64(2)
    bits |= bits >> np.uint64(4)
    bits |= bits >> np.uint64(8)
    bits |= bits >> np.uint64(16)
    bits |= bits >> np.uint64(32)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(bits)
    _, high_exp = np.frexp(arr >> np.uint64(32))
    _, low_exp = np.frexp(arr & np.uint64(0xFFFFFFFF))
    return np.where(high_exp, high_exp + 32, low_exp)


def get_rho_vec(w, max_width):
    bits = bit_length_vec(w)
    if np.count_nonzero(bits > max_width):
        raise ValueError("w overflow")
    return max_width - bits + 1


def _normalized_registers(M):
    return np.ascontiguousarray(M, dtype=HLL_COUNTER_TYPE)


def get_estimate(M, m, p, alpha):
    registers = _normalized_registers(M)
    harmonic, zeros = _register_stats(registers)
    if zeros > 0:
        linear = m * math.log(m / zeros)
        if linear <= get_treshold(p):
            return linear
    raw = alpha * (m**2) / harmonic
    return raw - estimate_bias(raw, p) if raw <= 5 * m else raw


def _hash64(value) -> int:
    return int.from_bytes(sha1(packb(value)).digest()[:8], byteorder="big")


class HyperLogLog:
    """HyperLogLog cardinality counter."""

    __slots__ = ("alpha", "p", "m", "M")

    def __init__(self, error_rate):
        if not (0 < error_rate < 1):
            raise ValueError("Error_Rate must be between 0 and 1.")
        p = math.ceil(math.log((1.04 / error_rate) ** 2, 2))
        self.alpha = get_alpha(p)
        self.p = p
        self.m = 1 << p
        self.M = np.zeros(self.m, HLL_COUNTER_TYPE)

    def __getstate__(self):
        return {
            "alpha": self.alpha,
            "p": self.p,
            "m": self.m,
            "M": self.M,
        }

    def __setstate__(self, state):
        for key in state:
            setattr(self, key, state[key])
        self.M = _normalized_registers(self.M)

    def _registers(self):
        if (
            not isinstance(self.M, np.ndarray)
            or self.M.dtype != np.dtype(HLL_COUNTER_TYPE)
            or self.M.ndim != 1
            or not self.M.flags.c_contiguous
            or not self.M.flags.writeable
        ):
            self.M = _normalized_registers(self.M)
        if self.M.size != self.m:
            raise ValueError("register length does not match precision")
        return self.M

    def add(self, value):
        x = _hash64(value)
        index = x & (self.m - 1)
        rho = get_rho(x >> self.p, 64 - self.p)
        if rho > self.M[index]:
            self.M[index] = rho

    def add_bulk(self, values):
        if isinstance(values, (bytes, str)) or not hasattr(values, "__iter__"):
            raise TypeError("values must be a non-string iterable")
        pack = Packer().pack
        digest = sha1
        from_bytes = int.from_bytes
        hashes = np.fromiter(
            (
                from_bytes(digest(pack(value)).digest()[:8], "big")
                for value in values
            ),
            dtype=np.uint64,
        )
        _add_hashes(self._registers(), hashes, self.p)

    def add_hashes(self, hashes):
        """Add precomputed unsigned 64-bit hashes without sorting or allocation."""
        array = np.asarray(hashes)
        if array.ndim != 1:
            raise ValueError("hashes must be one-dimensional")
        if array.dtype.kind not in "iu":
            raise TypeError("hashes must contain unsigned 64-bit integers")
        if array.dtype.kind == "i" and array.size and np.any(array < 0):
            raise ValueError("hashes must be non-negative")
        array = np.ascontiguousarray(array, dtype=np.uint64)
        _add_hashes(self._registers(), array, self.p)

    def update(self, *others):
        for item in others:
            if self.m != item.m:
                raise ValueError("Counters precisions should be equal")
            if isinstance(item, HyperLogLog):
                other_registers = item._registers()
            else:
                other_registers = _normalized_registers(item.M)
                if other_registers.size != self.m:
                    raise ValueError("register length does not match precision")
            _merge_trusted(self._registers(), other_registers)

    def __eq__(self, other):
        if not isinstance(other, HyperLogLog):
            return NotImplemented
        return self.m == other.m and np.array_equal(self.M, other.M)

    def __len__(self):
        return round(self.card())

    def card(self):
        return get_estimate(self.M, self.m, self.p, self.alpha)
