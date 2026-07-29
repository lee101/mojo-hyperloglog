from __future__ import annotations

import pickle

import pytest

from hyperloglog import SlidingHyperLogLog


def test_sliding_published_boundary_behavior():
    counter = SlidingHyperLogLog(0.05, 100)
    counter.add(1, "k1")
    assert int(counter.card(1)) == 1
    assert int(counter.card(101)) == 1
    assert int(counter.card(102)) == 0
    counter.add(2, "k2")
    counter.add(3, "k3")
    assert [int(counter.card(time)) for time in (3, 101, 102, 103, 104)] == [
        3,
        3,
        2,
        1,
        0,
    ]


def test_sliding_state_and_cards_match_real_upstream(upstream_reference):
    counter = SlidingHyperLogLog(0.05, 100)
    for timestamp in range(300):
        counter.add(timestamp, "key-%d" % (timestamp % 80))
    expected = upstream_reference["sliding"]
    state = [[list(item) for item in register] for register in counter.LPFM]
    assert state == expected["LPFM"]
    assert [
        counter.card(299, window) for window in (10, 50, 100)
    ] == pytest.approx(expected["cards"], abs=1e-12)
    assert counter.card_wlist(299, [100, 10, 50]) == pytest.approx(
        expected["card_wlist"], abs=1e-12
    )


def test_from_list_card_wlist_and_pickle():
    original = SlidingHyperLogLog(0.05, 100)
    for timestamp in range(500):
        original.add(timestamp, f"key-{timestamp % 120}")
    copied = SlidingHyperLogLog.from_list(original.LPFM, 100)
    assert copied == original
    windows = [100, 3, 5, 25]
    assert copied.card_wlist(499, windows) == [
        copied.card(499, window) for window in windows
    ]
    assert pickle.loads(pickle.dumps(original)) == original


def test_sliding_update_and_errors():
    left = SlidingHyperLogLog(0.05, 100)
    right = SlidingHyperLogLog(0.05, 100)
    combined = SlidingHyperLogLog(0.05, 100)
    for timestamp in range(1000):
        left.add(timestamp, f"left-{timestamp}")
        combined.add(timestamp, f"left-{timestamp}")
        right.add(timestamp, f"right-{timestamp}")
        combined.add(timestamp, f"right-{timestamp}")
    left.update(right)
    assert left == combined
    with pytest.raises(ValueError, match="precisions"):
        left.update(SlidingHyperLogLog(0.01, 100))
    with pytest.raises(ValueError, match="windows"):
        left.update(SlidingHyperLogLog(0.05, 99))
    with pytest.raises(ValueError, match="0 < window"):
        left.card(1000, 101)
    with pytest.raises(NotImplementedError, match="card"):
        len(left)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((0.05, 0), "window"),
        ((0, 100), "Error_Rate"),
        ((0.05, 100, []), "power of 2"),
        ((0.05, 100, [()] * 3), "power of 2"),
    ],
)
def test_sliding_constructor_errors(args, message):
    with pytest.raises(ValueError, match=message):
        SlidingHyperLogLog(*args)
