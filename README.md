# mojo-hyperloglog

`mojo-hyperloglog` is a standalone Mojo port of the
[`hyperloglog`](https://github.com/svpcom/hyperloglog) Python package's
cardinality estimators. It preserves the upstream Python API and hash format
while moving register-heavy work into a small compiled shared library.

The ordinary compatibility path accepts the same Python values as upstream and
uses the same SHA-1-over-MessagePack hash. For pipelines that already have
64-bit hashes, the additional `add_hashes()` method avoids Python object
hashing and updates registers directly.

## Coverage

The complete top-level API exported by `hyperloglog 0.1.8` is covered:

- `HyperLogLog(error_rate)`: `add`, `add_bulk`, `update`, `card`, `len`,
  equality, and pickle state
- `SlidingHyperLogLog(error_rate, window, lpfm=None)`: `from_list`, `add`,
  `update`, `card`, `card_wlist`, equality, and pickle state
- HLL++ linear counting and empirical bias correction for precisions 4 through
  16
- the estimator helpers in `hyperloglog.hll`: `bit_length_vec`,
  `estimate_bias`, `get_alpha`, `get_estimate`, `get_nearest_neighbors`,
  `get_rho`, `get_rho_vec`, and `get_treshold`
- `HyperLogLog.add_hashes()` as a Mojo-specific high-throughput extension

Mojo handles bulk register updates, register-wise merges, and the harmonic scan
used by cardinality estimation. Sliding-window LPFM maintenance remains Python
because it operates on variable-length timestamp tuples rather than dense
numeric buffers. The upstream package's internal `hyperloglog.const` module,
its data symbols, and its raw source literals are not compatibility targets;
the estimator data needed by the supported functions is included in compact
form.

## Install

The repository pins the tested Mojo nightly and manages all dependencies with
Pixi:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-hyperloglog.so`. The Python wrapper also
rebuilds it on first import when the library is missing or older than the Mojo
source.

## Usage

```python
from hyperloglog import HyperLogLog

counter = HyperLogLog(0.01)
counter.add("first")
counter.add_bulk(range(100_000))

print(len(counter))       # 100433
print(counter.card())     # 100433.42836524459

other = HyperLogLog(0.01)
other.add_bulk(range(50_000, 150_000))
counter.update(other)
print(len(counter))
```

Prehashed streams can skip Python serialization and SHA-1:

```python
import numpy as np
from hyperloglog import HyperLogLog

hashes = np.random.default_rng(0).integers(
    0, np.iinfo(np.uint64).max, size=2_000_000, dtype=np.uint64
)
counter = HyperLogLog(0.01)
counter.add_hashes(hashes)
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-136-generic. Times are the best of three warmed runs in one process.
The installed reference is `hyperloglog 0.1.8`.

| operation | mojo-hyperloglog | reference | speedup |
| --- | ---: | ---: | ---: |
| `add_bulk` 200k integers | 203.74 ms | 344.62 ms (upstream) | 1.69x |
| `add_hashes` 2M uint64 | 10.84 ms | 1441.11 ms (equivalent NumPy update) | 132.99x |
| `card` p=16, 1000 calls | 112.48 ms | 1808.59 ms (upstream) | 16.08x |
| `update` p=16, 1000 merges | 11.05 ms | 23.60 ms (upstream) | 2.14x |

`add_bulk` includes MessagePack serialization and SHA-1 in both implementations,
so Python object hashing dominates that row. The bulk path reuses one MessagePack
packer instead of constructing one per value. `add_hashes` isolates the register
update that Mojo replaces: it is a single O(n) pass, where the equivalent
upstream NumPy algorithm groups registers through `unique`, sorting, and
`maximum.reduceat`.

There is no GPU or multithreaded path. An HLL has at most 65,536 one-byte
registers, so thread-launch overhead is not justified, while merge has only one
comparison per two bytes read and prehashed insertion performs random register
updates. None of these kernels approaches the roughly two-flops-per-byte
arithmetic intensity needed to justify GPU transfer and launch overhead. The
merge path validates buffer metadata once, then calls the existing Mojo
byte-SIMD kernel directly without allocating or repeating full-array rank scans.

Benchmark results depend on processor and system load. Run `pixi run bench` on
the target machine rather than treating this table as a universal claim.

## How it works

Python owns every allocation. A `HyperLogLog` stores its `2**p` registers as a
C-contiguous `numpy.int8` array, matching upstream on NumPy 2. Precomputed
hashes cross as contiguous `numpy.uint64`; sliding registers remain Python
tuples of `(timestamp, rank)` pairs.

The `ctypes` layer validates array dtype, dimensionality, contiguity,
writability, lengths, and register ranks as applicable, then passes buffer
addresses as integer values. Python keeps live references to all arrays for the
duration of each call. The single Mojo compilation unit reconstructs addresses as
`UnsafePointer[..., AnyOrigin[mut=True]]`, updates caller-owned memory, and
exports non-parametric C ABI functions. No allocation or object ownership
crosses the FFI boundary. Merges use the host's native byte SIMD width with a
scalar tail, and estimation returns the zero-register count plus the harmonic
sum to the Python HLL++ correction logic.

Parity tests compare exact register states, estimates, merges, sliding-window
state, error behavior, and serialization against the real PyPI package.
The empirical bias dataset carries its upstream notice in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md); original source code is MIT
licensed.
