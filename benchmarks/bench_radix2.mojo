"""Forward radix-2 throughput benchmark with deterministic inputs."""

from shuhafft import FFTDirection, FFTNormalization, FFTPlan
from std.benchmark import keep
from std.complex import ComplexSIMD
from std.time import perf_counter_ns

comptime _MIN_EXPONENT = 8
comptime _MAX_EXPONENT = 16
comptime _ITERATION_BUDGET = 1 << 22
comptime _TRANSFORMS_PER_BLOCK = 4
comptime _WARMUP_ITERATIONS = 4
comptime _LCG_SEED = UInt64(0x5A17_C0DE)


def _lcg_sample(mut state: UInt64) -> Float64:
    # Numerical Recipes LCG: state = 1664525 * state + 1013904223 (mod 2^32).
    state = (state * UInt64(1_664_525) + UInt64(1_013_904_223)) % UInt64(4_294_967_296)
    return Float64(state) / 2147483647.5 - 1.0


def _random_values[
    dtype: DType
](size: Int) -> List[ComplexSIMD[dtype, 1]] where dtype.is_floating_point():
    var state = _LCG_SEED
    var values = List[ComplexSIMD[dtype, 1]](capacity=size)
    for _ in range(size):
        var real = Scalar[dtype](_lcg_sample(state))
        var imaginary = Scalar[dtype](_lcg_sample(state))
        values.append(ComplexSIMD[dtype, 1](real, imaginary))
    return values^


def _rescale_after_block[
    dtype: DType
](mut values: List[ComplexSIMD[dtype, 1]], size: Int) where dtype.is_floating_point():
    # Four unscaled DFTs multiply the original sequence by n^2. Rescaling once
    # per four-transform block bounds magnitudes without entering timed regions.
    var typed_size = Scalar[dtype](size)
    var scale = Scalar[dtype](1.0) / (typed_size * typed_size)
    for index in range(size):
        values[index] *= scale


def _run_cell[
    dtype: DType
](dtype_name: StringLiteral, size: Int) raises where dtype.is_floating_point():
    var values = _random_values[dtype](size)
    # The reusable plan and input buffer are constructed outside all timings.
    var plan = FFTPlan[dtype](size, FFTDirection.FORWARD, FFTNormalization.BACKWARD)
    var iterations = _ITERATION_BUDGET // size
    debug_assert(iterations % _TRANSFORMS_PER_BLOCK == 0, "incomplete timing block")

    # Repeated transforms use the evolving in-place buffer: radix-2 work is
    # data-independent, and this keeps allocation and input refills out of the
    # measurement. Warmup is one complete rescaled block.
    for _ in range(_WARMUP_ITERATIONS):
        plan.execute_in_place(values)
    _rescale_after_block(values, size)

    var total_ns = 0
    var checksum = Float64(0.0)
    for block in range(iterations // _TRANSFORMS_PER_BLOCK):
        var started = perf_counter_ns()
        for _ in range(_TRANSFORMS_PER_BLOCK):
            plan.execute_in_place(values)
        total_ns += perf_counter_ns() - started

        # Rescaling and checksum work are excluded from the transform timings.
        _rescale_after_block(values, size)
        var checksum_index = block % size
        checksum += Float64(values[checksum_index].re)
        checksum += Float64(values[checksum_index].im)

    keep(checksum)
    print(
        "BENCH shuhafft radix2 forward dtype=",
        dtype_name,
        " n=",
        size,
        " iters=",
        iterations,
        " total_ns=",
        total_ns,
        " ns_per_transform=",
        Float64(total_ns) / Float64(iterations),
        " checksum=",
        checksum,
        sep="",
    )


def _run_dtype[
    dtype: DType
](dtype_name: StringLiteral) raises where dtype.is_floating_point():
    for exponent in range(_MIN_EXPONENT, _MAX_EXPONENT + 1):
        _run_cell[dtype](dtype_name, 1 << exponent)


def main() raises:
    print(
        'BENCH_HEADER shuhafft radix2 mojo=1.0.0 command="pixi run bench" ',
        "statistic=mean_ns_per_transform warmup_iters=4 ",
        "iteration_rule=2^22/n rescale=outside_timing_after_4_transforms",
        sep="",
    )
    _run_dtype[DType.float32]("float32")
    _run_dtype[DType.float64]("float64")
