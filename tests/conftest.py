from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest


@pytest.fixture(scope="session")
def upstream_reference(tmp_path_factory):
    script = r"""
import json
import hyperloglog

result = {}

h = hyperloglog.HyperLogLog(0.01)
h.add_bulk(range(25000))
result["bulk"] = {"M": h.M.tolist(), "card": h.card(), "length": len(h)}

single = hyperloglog.HyperLogLog(0.05)
for value in [None, True, 42, -7, 3.5, "alpha", b"bytes", [1, 2, 3]]:
    single.add(value)
result["single"] = {"M": single.M.tolist(), "card": single.card()}

left = hyperloglog.HyperLogLog(0.02)
right = hyperloglog.HyperLogLog(0.02)
left.add_bulk(range(0, 10000, 2))
right.add_bulk(range(1, 10000, 2))
left.update(right)
result["merged"] = {"M": left.M.tolist(), "card": left.card()}

sliding = hyperloglog.SlidingHyperLogLog(0.05, 100)
for timestamp in range(300):
    sliding.add(timestamp, "key-%d" % (timestamp % 80))
result["sliding"] = {
    "LPFM": [[list(item) for item in register] for register in sliding.LPFM],
    "cards": [sliding.card(299, window) for window in (10, 50, 100)],
    "card_wlist": sliding.card_wlist(299, [100, 10, 50]),
}

print(json.dumps(result))
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    workdir = tmp_path_factory.mktemp("upstream")
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return json.loads(proc.stdout)
