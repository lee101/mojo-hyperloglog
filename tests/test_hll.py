from __future__ import annotations

import pickle
from hashlib import sha1

import numpy as np
import pytest
from msgpack import packb

from hyperloglog import HyperLogLog
from hyperloglog.hll import (
    bit_length_vec,
    estimate_bias,
    get_alpha,
    get_estimate,
    get_nearest_neighbors,
    get_rho,
    get_rho_vec,
    get_treshold,
)
from hyperloglog._lib import merge
from hyperloglog._lib import add_hashes as ffi_add_hashes
from hyperloglog._lib import register_stats


def test_upstream_published_register_vector():
    counter = HyperLogLog(0.05)
    counter.add_bulk(str(i) for i in range(10))
    occupied = [(i, int(value)) for i, value in enumerate(counter.M) if value > 0]
    assert occupied == [
        (40, 1),
        (121, 1),
        (197, 4),
        (200, 4),
        (247, 2),
        (260, 3),
        (377, 2),
        (444, 1),
        (500, 3),
    ]


def test_constructor_and_helpers_match_upstream():
    counter = HyperLogLog(0.05)
    assert (counter.p, counter.m, counter.alpha) == (
        9,
        512,
        0.7197831133217303,
    )
    assert [get_alpha(p) for p in range(4, 10)] == [
        0.673,
        0.697,
        0.709,
        0.7152704932638152,
        0.7182725932495458,
        0.7197831133217303,
    ]
    assert [get_rho(value, 32) for value in (0, 1, 2, 3, 4, 1 << 31)] == [
        33,
        32,
        31,
        31,
        30,
        1,
    ]
    with pytest.raises(ValueError, match="range"):
        get_alpha(17)
    with pytest.raises(ValueError, match="overflow"):
        get_rho(1 << 32, 32)


@pytest.mark.parametrize("error_rate", [0, 1, -0.1, 0.0001])
def test_constructor_errors_match_upstream(error_rate):
    with pytest.raises(ValueError):
        HyperLogLog(error_rate)


def test_add_matches_real_upstream(upstream_reference):
    values = [None, True, 42, -7, 3.5, "alpha", b"bytes", [1, 2, 3]]
    counter = HyperLogLog(0.05)
    for value in values:
        counter.add(value)
    expected = upstream_reference["single"]
    assert counter.M.tolist() == expected["M"]
    assert counter.card() == pytest.approx(expected["card"], abs=1e-12)


def test_add_bulk_numerical_parity_with_real_upstream(upstream_reference):
    counter = HyperLogLog(0.01)
    counter.add_bulk(range(25000))
    expected = upstream_reference["bulk"]
    assert counter.M.tolist() == expected["M"]
    assert counter.card() == pytest.approx(expected["card"], abs=1e-12)
    assert len(counter) == expected["length"]


def test_bulk_and_scalar_are_identical():
    scalar = HyperLogLog(0.02)
    bulk = HyperLogLog(0.02)
    values = [f"value-{index}" for index in range(5000)]
    for value in values:
        scalar.add(value)
    bulk.add_bulk(values)
    assert scalar == bulk
    assert scalar.card() == bulk.card()


def test_bulk_reused_packer_matches_scalar_for_mixed_values():
    values = [None, True, 42, -7, 3.5, "alpha", b"bytes", [1, 2, 3]]
    scalar = HyperLogLog(0.05)
    bulk = HyperLogLog(0.05)
    for value in values:
        scalar.add(value)
    bulk.add_bulk(values)
    assert bulk == scalar


@pytest.mark.parametrize("bad_values", ["abc", b"abc", 12])
def test_add_bulk_rejects_non_iterable_or_string(bad_values):
    with pytest.raises(TypeError, match="non-string iterable"):
        HyperLogLog(0.05).add_bulk(bad_values)


def test_prehashed_extension_matches_compatibility_hashing():
    values = [f"event-{index}" for index in range(20000)]
    hashes = np.fromiter(
        (
            int.from_bytes(sha1(packb(value)).digest()[:8], "big")
            for value in values
        ),
        dtype=np.uint64,
    )
    normal = HyperLogLog(0.01)
    prehashed = HyperLogLog(0.01)
    normal.add_bulk(values)
    prehashed.add_hashes(hashes)
    assert prehashed == normal
    with pytest.raises(ValueError, match="one-dimensional"):
        prehashed.add_hashes(hashes.reshape(100, 200))


def test_prehashed_kernel_matches_scalar_reference_at_uint64_edges():
    rng = np.random.default_rng(7)
    hashes = np.concatenate(
        [
            np.array([0, 1, 2, 2**63, 2**64 - 1], dtype=np.uint64),
            rng.integers(0, 2**64 - 1, 10000, dtype=np.uint64),
        ]
    )
    counter = HyperLogLog(0.01)
    expected = np.zeros(counter.m, dtype=np.int8)
    for value in hashes:
        x = int(value)
        index = x & (counter.m - 1)
        rank = get_rho(x >> counter.p, 64 - counter.p)
        expected[index] = max(expected[index], rank)
    counter.add_hashes(hashes)
    assert np.array_equal(counter.M, expected)


def test_prehashed_input_rejects_silent_narrowing():
    counter = HyperLogLog(0.05)
    counter.add_hashes(np.array([], dtype=np.uint64))
    assert not np.any(counter.M)
    with pytest.raises(TypeError, match="unsigned 64-bit"):
        counter.add_hashes([1.5, 2.5])
    with pytest.raises(ValueError, match="non-negative"):
        counter.add_hashes([-1, 2])


def test_ffi_rejects_invalid_buffers_before_crossing_boundary():
    registers = np.zeros(16, dtype=np.int8)
    hashes = np.zeros(1, dtype=np.uint64)
    with pytest.raises(TypeError, match="numpy.int8"):
        ffi_add_hashes(registers.astype(np.int16), hashes, 4)
    with pytest.raises(ValueError, match="C-contiguous"):
        ffi_add_hashes(np.zeros(32, dtype=np.int8)[::2], hashes, 4)
    with pytest.raises(ValueError, match="writable"):
        readonly = registers.copy()
        readonly.flags.writeable = False
        ffi_add_hashes(readonly, hashes, 4)
    with pytest.raises(TypeError, match="numpy.uint64"):
        ffi_add_hashes(registers, hashes.astype(np.int64), 4)
    with pytest.raises(ValueError, match="register length"):
        ffi_add_hashes(np.zeros(15, dtype=np.int8), hashes, 4)
    invalid = registers.copy()
    invalid[0] = -1
    with pytest.raises(ValueError, match="invalid register rank"):
        register_stats(invalid)


def test_replaced_and_resized_register_buffers_fail_safely():
    counter = HyperLogLog(0.05)
    counter.M = np.zeros(counter.m * 2, dtype=np.int8)[::2]
    counter.add_hashes(np.array([1], dtype=np.uint64))
    assert counter.M.flags.c_contiguous
    counter.M.resize(counter.m // 2, refcheck=False)
    with pytest.raises(ValueError, match="register length"):
        counter.add_hashes(np.array([1], dtype=np.uint64))


def test_update_matches_real_upstream(upstream_reference):
    left = HyperLogLog(0.02)
    right = HyperLogLog(0.02)
    left.add_bulk(range(0, 10000, 2))
    right.add_bulk(range(1, 10000, 2))
    left.update(right)
    expected = upstream_reference["merged"]
    assert left.M.tolist() == expected["M"]
    assert left.card() == pytest.approx(expected["card"], abs=1e-12)
    with pytest.raises(ValueError, match="precisions"):
        left.update(HyperLogLog(0.01))


def test_merge_tail_for_minimum_precision():
    left = HyperLogLog(0.3)
    right = HyperLogLog(0.3)
    assert left.p == 4
    left.add_bulk(range(0, 100, 2))
    right.add_bulk(range(1, 100, 2))
    expected = np.maximum(left.M, right.M)
    left.update(right)
    assert np.array_equal(left.M, expected)


def test_merge_simd_tail_and_replaced_register_buffers():
    left_registers = np.arange(35, dtype=np.int8)
    right_registers = np.arange(35, 0, -1, dtype=np.int8)
    expected = np.maximum(left_registers, right_registers)
    merge(left_registers, right_registers)
    assert np.array_equal(left_registers, expected)

    left = HyperLogLog(0.05)
    right = HyperLogLog(0.05)
    left.M = np.full(left.m, 2, dtype=np.int8)
    right.M = np.full(right.m, 7, dtype=np.int8)
    left.update(right)
    assert np.all(left.M == 7)


def test_pickle_and_legacy_state_normalization():
    counter = HyperLogLog(0.05)
    counter.add_bulk(range(1000))
    restored = pickle.loads(pickle.dumps(counter))
    assert restored == counter
    restored.__setstate__({**counter.__getstate__(), "M": counter.M.tolist()})
    assert restored == counter
    assert restored.M.dtype == np.int8 and restored.M.flags.c_contiguous


def test_vector_rank_helpers():
    values = np.array([0, 1, 2, 3, 4, 255, 256, 2**63], dtype=np.uint64)
    expected_bits = np.array([value.item().bit_length() for value in values])
    assert np.array_equal(bit_length_vec(values), expected_bits)
    assert np.array_equal(get_rho_vec(values[:-1], 64), 65 - expected_bits[:-1])
    with pytest.raises(ValueError, match="overflow"):
        get_rho_vec(values[-1:], 63)


def test_bias_helpers():
    assert get_treshold(4) == 10
    estimates = np.array([1.0, 4.0, 8.0, 10.0, 12.0, 20.0, 30.0])
    neighbors = get_nearest_neighbors(10.0, estimates)
    assert set(neighbors) == {0, 1, 2, 3, 4, 5}
    assert np.array_equal(
        np.sort((10.0 - estimates[neighbors]) ** 2),
        np.array([0.0, 4.0, 4.0, 36.0, 81.0, 100.0]),
    )
    assert np.isfinite(estimate_bias(100.0, 4))


def test_estimator_matches_real_upstream_bias_correction(upstream_reference):
    registers = np.array(upstream_reference["bulk"]["M"], dtype=np.int8)
    counter = HyperLogLog(0.01)
    estimate = get_estimate(registers, counter.m, counter.p, counter.alpha)
    assert estimate == pytest.approx(upstream_reference["bulk"]["card"], abs=1e-12)


@pytest.mark.parametrize("cardinality", [1, 10, 100, 1000, 10000, 100000])
def test_deterministic_estimation_accuracy(cardinality):
    counter = HyperLogLog(0.02)
    counter.add_bulk(range(cardinality))
    tolerance = max(2, cardinality * 0.06)
    assert counter.card() == pytest.approx(cardinality, abs=tolerance)
