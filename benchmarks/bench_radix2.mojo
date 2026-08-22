"""Complex/real FFT direction and allocation-path microbenchmarks."""

from shuhafft import FFTDirection, FFTNormalization, FFTPlan, RealFFTPlan
from std.benchmark import keep
from std.complex import ComplexSIMD
from std.time import perf_counter_ns

comptime _THROUGHPUT_BUDGET = 1 << 20
comptime _ALLOCATION_BUDGET = 1 << 18
comptime _TRANSFORMS_PER_BLOCK = 4
comptime _WARMUP_ITERATIONS = 4
comptime _LCG_SEED = UInt64(0x5A17_C0DE)


def _lcg_sample(mut state: UInt64) -> Float64:
    # Numerical Recipes LCG: state = 1664525 * state + 1013904223 (mod 2^32).
    state = (state * UInt64(1_664_525) + UInt64(1_013_904_223)) % UInt64(4_294_967_296)
    return Float64(state) / 2147483647.5 - 1.0


def _random_complex[
    dtype: DType
](size: Int) -> List[ComplexSIMD[dtype, 1]] where dtype.is_floating_point():
    var state = _LCG_SEED
    var values = List[ComplexSIMD[dtype, 1]](capacity=size)
    for _ in range(size):
        var real = Scalar[dtype](_lcg_sample(state))
        var imaginary = Scalar[dtype](_lcg_sample(state))
        values.append(ComplexSIMD[dtype, 1](real, imaginary))
    return values^


def _random_real[
    dtype: DType
](size: Int) -> List[Scalar[dtype]] where dtype.is_floating_point():
    var state = _LCG_SEED
    var values = List[Scalar[dtype]](capacity=size)
    for _ in range(size):
        values.append(Scalar[dtype](_lcg_sample(state)))
    return values^


def _rescale_after_block[
    dtype: DType
](mut values: List[ComplexSIMD[dtype, 1]], size: Int) where dtype.is_floating_point():
    # Four unscaled DFTs multiply the original sequence by n^2. Rescaling once
    # per block bounds magnitudes without entering timed regions.
    var typed_size = Scalar[dtype](size)
    var scale = Scalar[dtype](1.0) / (typed_size * typed_size)
    for index in range(size):
        values[index] *= scale


def _throughput_iterations(size: Int) -> Int:
    var iterations = _THROUGHPUT_BUDGET // size
    if iterations < _TRANSFORMS_PER_BLOCK:
        return _TRANSFORMS_PER_BLOCK
    return iterations - iterations % _TRANSFORMS_PER_BLOCK


def _allocation_iterations(size: Int) -> Int:
    var iterations = _ALLOCATION_BUDGET // size
    return iterations if iterations >= 4 else 4


def _report(
    family: StringLiteral,
    direction: StringLiteral,
    api: StringLiteral,
    allocation: StringLiteral,
    dtype_name: StringLiteral,
    size: Int,
    iterations: Int,
    total_ns: Int,
    checksum: Float64,
):
    print(
        "BENCH shuhafft family=",
        family,
        " direction=",
        direction,
        " api=",
        api,
        " allocation=",
        allocation,
        " dtype=",
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


def _run_complex_in_place[
    dtype: DType
](
    dtype_name: StringLiteral,
    direction_name: StringLiteral,
    size: Int,
    direction: FFTDirection,
) raises where dtype.is_floating_point():
    var values = _random_complex[dtype](size)
    var plan = FFTPlan[dtype](size, direction, FFTNormalization.NONE)
    var iterations = _throughput_iterations(size)

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
        _rescale_after_block(values, size)
        var checksum_index = block % size
        checksum += Float64(values[checksum_index].re)
        checksum += Float64(values[checksum_index].im)

    keep(checksum)
    _report(
        "complex",
        direction_name,
        "execute_in_place",
        "none",
        dtype_name,
        size,
        iterations,
        total_ns,
        checksum,
    )


def _run_complex_allocating[
    dtype: DType
](
    dtype_name: StringLiteral,
    direction_name: StringLiteral,
    size: Int,
    direction: FFTDirection,
) raises where dtype.is_floating_point():
    var values = _random_complex[dtype](size)
    var plan = FFTPlan[dtype](size, direction, FFTNormalization.NONE)
    var output = plan.execute(values)
    var iterations = _allocation_iterations(size)

    var total_ns = 0
    var checksum = Float64(0.0)
    for iteration in range(iterations):
        var started = perf_counter_ns()
        output = plan.execute(values)
        total_ns += perf_counter_ns() - started
        var checksum_index = iteration % size
        checksum += Float64(output[checksum_index].re)
        checksum += Float64(output[checksum_index].im)

    keep(checksum)
    _report(
        "complex",
        direction_name,
        "execute",
        "return_list",
        dtype_name,
        size,
        iterations,
        total_ns,
        checksum,
    )


def _run_real_forward_into[
    dtype: DType
](dtype_name: StringLiteral, size: Int) raises where dtype.is_floating_point():
    var signal = _random_real[dtype](size)
    var plan = RealFFTPlan[dtype](size)
    var spectrum = plan.make_spectrum()
    var iterations = _throughput_iterations(size)
    for _ in range(_WARMUP_ITERATIONS):
        plan.forward_into(signal, spectrum)

    var total_ns = 0
    var checksum = Float64(0.0)
    for iteration in range(iterations):
        var started = perf_counter_ns()
        plan.forward_into(signal, spectrum)
        total_ns += perf_counter_ns() - started
        var checksum_index = iteration % len(spectrum)
        checksum += Float64(spectrum[checksum_index].re)
        checksum += Float64(spectrum[checksum_index].im)

    keep(checksum)
    _report(
        "real",
        "forward",
        "forward_into",
        "none",
        dtype_name,
        size,
        iterations,
        total_ns,
        checksum,
    )


def _run_real_inverse_into[
    dtype: DType
](dtype_name: StringLiteral, size: Int) raises where dtype.is_floating_point():
    var signal = _random_real[dtype](size)
    var plan = RealFFTPlan[dtype](size)
    var spectrum = plan.forward(signal)
    var output = plan.make_input()
    var iterations = _throughput_iterations(size)
    for _ in range(_WARMUP_ITERATIONS):
        plan.inverse_into(spectrum, output)

    var total_ns = 0
    var checksum = Float64(0.0)
    for iteration in range(iterations):
        var started = perf_counter_ns()
        plan.inverse_into(spectrum, output)
        total_ns += perf_counter_ns() - started
        checksum += Float64(output[iteration % size])

    keep(checksum)
    _report(
        "real",
        "inverse",
        "inverse_into",
        "none",
        dtype_name,
        size,
        iterations,
        total_ns,
        checksum,
    )


def _run_real_allocating[
    dtype: DType
](
    dtype_name: StringLiteral,
    direction_name: StringLiteral,
    size: Int,
    inverse: Bool,
) raises where dtype.is_floating_point():
    var signal = _random_real[dtype](size)
    var plan = RealFFTPlan[dtype](size)
    var spectrum = plan.forward(signal)
    var complex_output = plan.forward(signal)
    var real_output = plan.inverse(spectrum)
    var iterations = _allocation_iterations(size)
    var total_ns = 0
    var checksum = Float64(0.0)

    if inverse:
        for iteration in range(iterations):
            var started = perf_counter_ns()
            real_output = plan.inverse(spectrum)
            total_ns += perf_counter_ns() - started
            checksum += Float64(real_output[iteration % size])
    else:
        for iteration in range(iterations):
            var started = perf_counter_ns()
            complex_output = plan.forward(signal)
            total_ns += perf_counter_ns() - started
            var checksum_index = iteration % len(complex_output)
            checksum += Float64(complex_output[checksum_index].re)
            checksum += Float64(complex_output[checksum_index].im)

    keep(checksum)
    if inverse:
        _report(
            "real",
            direction_name,
            "inverse",
            "return_list",
            dtype_name,
            size,
            iterations,
            total_ns,
            checksum,
        )
    else:
        _report(
            "real",
            direction_name,
            "forward",
            "return_list",
            dtype_name,
            size,
            iterations,
            total_ns,
            checksum,
        )


def _run_cell[
    dtype: DType
](dtype_name: StringLiteral, size: Int) raises where dtype.is_floating_point():
    _run_complex_in_place[dtype](dtype_name, "forward", size, FFTDirection.FORWARD)
    _run_complex_in_place[dtype](dtype_name, "inverse", size, FFTDirection.INVERSE)
    _run_complex_allocating[dtype](dtype_name, "forward", size, FFTDirection.FORWARD)
    _run_complex_allocating[dtype](dtype_name, "inverse", size, FFTDirection.INVERSE)
    _run_real_forward_into[dtype](dtype_name, size)
    _run_real_inverse_into[dtype](dtype_name, size)
    _run_real_allocating[dtype](dtype_name, "forward", size, False)
    _run_real_allocating[dtype](dtype_name, "inverse", size, True)


def _run_dtype[
    dtype: DType
](dtype_name: StringLiteral) raises where dtype.is_floating_point():
    for exponent in [8, 12, 16]:
        _run_cell[dtype](dtype_name, 1 << exponent)


def main() raises:
    print(
        'BENCH_HEADER shuhafft fft-api mojo=1.0.0 command="pixi run bench" ',
        "statistic=mean_ns_per_transform caller_buffer_warmup_iters=4 ",
        "return_list_warmup_iters=1 ",
        "sizes=2^8,2^12,2^16 allocation=declared_api_behavior",
        sep="",
    )
    _run_dtype[DType.float32]("float32")
    _run_dtype[DType.float64]("float64")
