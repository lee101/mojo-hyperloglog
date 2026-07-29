"""Allocation-free HyperLogLog kernels exported through a C ABI."""

from std.sys.info import simd_width_of

comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime U64Ptr = UnsafePointer[UInt64, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def bit_length_64(value: UInt64) -> Int:
    if value == 0:
        return 0
    var x = value
    var width = 0
    if x >= (UInt64(1) << 32):
        x >>= 32
        width += 32
    if x >= (UInt64(1) << 16):
        x >>= 16
        width += 16
    if x >= (UInt64(1) << 8):
        x >>= 8
        width += 8
    if x >= (UInt64(1) << 4):
        x >>= 4
        width += 4
    if x >= (UInt64(1) << 2):
        x >>= 2
        width += 2
    if x >= UInt64(2):
        width += 1
    return width + 1


def add_hashes(registers: BPtr, hashes: U64Ptr, n: Int, p: Int, m: Int):
    var mask = UInt64(m - 1)
    var max_width = 64 - p
    for i in range(n):
        var x = hashes[i]
        var index = Int(x & mask)
        var rho = max_width - bit_length_64(x >> UInt64(p)) + 1
        if rho > Int(registers[index]):
            registers[index] = UInt8(rho)


def merge_registers(registers: BPtr, other: BPtr, m: Int):
    comptime W = simd_width_of[DType.uint8]()
    var i = 0
    while i + W <= m:
        registers.store(
            i,
            max(
                registers.load[width=W](i),
                other.load[width=W](i),
            ),
        )
        i += W
    while i < m:
        if other[i] > registers[i]:
            registers[i] = other[i]
        i += 1


def register_stats(registers: BPtr, m: Int, zeros: I64Ptr) -> Float64:
    var harmonic = 0.0
    var zero_count = Int64(0)
    for i in range(m):
        var rank = Int(registers[i])
        if rank == 0:
            zero_count += 1
        harmonic += 1.0 / Float64(UInt64(1) << UInt64(rank))
    zeros[0] = zero_count
    return harmonic


@export("mhll_add_hashes")
def mhll_add_hashes(
    registers_addr: Int, hashes_addr: Int, n: Int, p: Int, m: Int
) abi("C"):
    add_hashes(
        BPtr(unsafe_from_address=registers_addr),
        U64Ptr(unsafe_from_address=hashes_addr),
        n,
        p,
        m,
    )


@export("mhll_merge")
def mhll_merge(registers_addr: Int, other_addr: Int, m: Int) abi("C"):
    merge_registers(
        BPtr(unsafe_from_address=registers_addr),
        BPtr(unsafe_from_address=other_addr),
        m,
    )


@export("mhll_register_stats")
def mhll_register_stats(
    registers_addr: Int, m: Int, zeros_addr: Int
) abi("C") -> Float64:
    return register_stats(
        BPtr(unsafe_from_address=registers_addr),
        m,
        I64Ptr(unsafe_from_address=zeros_addr),
    )
