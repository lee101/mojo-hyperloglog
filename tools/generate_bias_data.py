"""Regenerate the compact HLL++ bias dataset from hyperloglog 0.1.8."""

from __future__ import annotations

import base64
import struct
import textwrap
import zlib
from pathlib import Path

import numpy as np
from hyperloglog import hll

blob = bytearray()
for estimates, biases in zip(hll.rawEstimateData, hll.biasData):
    estimates = np.asarray(estimates, dtype="<f8")
    biases = np.asarray(biases, dtype="<f8")
    blob += struct.pack("<H", len(estimates))
    blob += estimates.tobytes()
    blob += biases.tobytes()

encoded = base64.b85encode(zlib.compress(bytes(blob), 9)).decode("ascii")
lines = "\n".join(f'    "{line}"' for line in textwrap.wrap(encoded, 100))
target = Path(__file__).parents[1] / "python" / "hyperloglog" / "_bias.py"
target.write_text(
    '''"""HLL++ empirical bias data distributed by hyperloglog 0.1.8."""

from __future__ import annotations

import base64
import struct
import zlib

import numpy as np

_ENCODED = (
'''
    + lines
    + """
)


def load_bias_data():
    payload = memoryview(zlib.decompress(base64.b85decode(_ENCODED)))
    estimates = []
    biases = []
    offset = 0
    for _ in range(13):
        length = struct.unpack_from("<H", payload, offset)[0]
        offset += 2
        byte_count = length * 8
        estimates.append(np.frombuffer(payload[offset : offset + byte_count], dtype="<f8"))
        offset += byte_count
        biases.append(np.frombuffer(payload[offset : offset + byte_count], dtype="<f8"))
        offset += byte_count
    return tuple(estimates), tuple(biases)
"""
)
