"""Sliding-window HyperLogLog compatible with the upstream package."""

from __future__ import annotations

import heapq
import math
from hashlib import sha1

import numpy as np
from msgpack import packb

from .hll import get_alpha, get_estimate, get_rho


class SlidingHyperLogLog:
    __slots__ = ("window", "alpha", "p", "m", "LPFM")

    def __init__(self, error_rate, window, lpfm=None):
        self.window = window
        if not window > 0:
            raise ValueError("window must be > 0")
        if lpfm is not None:
            m = len(lpfm)
            if m == 0 or (m & (m - 1)) != 0:
                raise ValueError("List length is not power of 2")
            p = m.bit_length() - 1
            self.LPFM = list(lpfm)
        else:
            if not (0 < error_rate < 1):
                raise ValueError("Error_Rate must be between 0 and 1.")
            p = math.ceil(math.log((1.04 / error_rate) ** 2, 2))
            m = 1 << p
            self.LPFM = [tuple() for _ in range(m)]
        self.alpha = get_alpha(p)
        self.p = p
        self.m = m

    def __getstate__(self):
        return {name: getattr(self, name) for name in self.__slots__}

    def __setstate__(self, state):
        for key in state:
            setattr(self, key, state[key])
        self.LPFM = [
            tuple(sorted(register, reverse=True)) if register else tuple()
            for register in self.LPFM
        ]

    @classmethod
    def from_list(cls, lpfm, window):
        return cls(None, window, lpfm)

    def add(self, timestamp, value):
        x = int.from_bytes(sha1(packb(value)).digest()[:8], byteorder="big")
        index = x & (self.m - 1)
        rank = get_rho(x >> self.p, 64 - self.p)
        rank_max = None
        merged = []
        time_min = None
        for time, item_rank in heapq.merge(
            self.LPFM[index], ((timestamp, rank),), reverse=True
        ):
            if time_min is None:
                time_min = time - self.window
            if time < time_min:
                break
            if rank_max is None or item_rank > rank_max:
                merged.append((time, item_rank))
                rank_max = item_rank
        self.LPFM[index] = tuple(merged)

    def update(self, *others):
        for item in others:
            if self.m != item.m:
                raise ValueError("Counters precisions should be equal")
            if self.window != item.window:
                raise ValueError("Counters windows should be equal")
        sources = [item.LPFM for item in others]
        for index, registers in enumerate(zip(self.LPFM, *sources)):
            rank_max = None
            merged = []
            time_max = None
            for time, rank in heapq.merge(*registers, reverse=True):
                if time_max is None:
                    time_max = time
                if time < time_max - self.window:
                    break
                if rank_max is None or rank > rank_max:
                    merged.append((time, rank))
                    rank_max = rank
            self.LPFM[index] = tuple(merged)

    def __eq__(self, other):
        if not isinstance(other, SlidingHyperLogLog):
            return NotImplemented
        return (
            self.m == other.m
            and self.window == other.window
            and self.LPFM == other.LPFM
        )

    def __len__(self):
        raise NotImplementedError("use card(timestamp) to estimate cardinality")

    def card(self, timestamp, window=None):
        if window is None:
            window = self.window
        if not 0 < window <= self.window:
            raise ValueError("0 < window <= W")
        time_min = timestamp - window
        registers = np.fromiter(
            (
                max(
                    (rank for time, rank in lpfm if time >= time_min),
                    default=0,
                )
                for lpfm in self.LPFM
            ),
            dtype=np.int8,
            count=self.m,
        )
        return get_estimate(registers, self.m, self.p, self.alpha)

    def card_wlist(self, timestamp, window_list):
        for window in window_list:
            if not 0 < window <= self.window:
                raise ValueError("0 < window <= W")
        thresholds = sorted(
            (timestamp - window, index)
            for index, window in enumerate(window_list)
        )
        register_lists = [[] for _ in window_list]
        for lpfm in self.LPFM:
            rank_max = 0
            position = len(thresholds) - 1
            for time, rank in lpfm:
                while position >= 0:
                    time_min, result_index = thresholds[position]
                    if time >= time_min:
                        break
                    register_lists[result_index].append(rank_max)
                    position -= 1
                if position < 0:
                    break
                rank_max = rank
            for index in range(position + 1):
                register_lists[thresholds[index][1]].append(rank_max)
        return [
            get_estimate(np.array(registers, dtype=np.int8), self.m, self.p, self.alpha)
            for registers in register_lists
        ]
