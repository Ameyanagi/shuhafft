"""Compiled p50/p95 and sampling-profiler workload for FFT kernels."""

from shuhafft import (
    BluesteinFFTPlan,
    FFTDirection,
    FFTNormalization,
    FFTPlan,
    RealFFTPlan,
)
from shuhafft._radix2 import (
    _radix2_in_place,
    _radix2_interleaved_in_place,
    _radix2_interleaved_scalar_in_place,
    _radix2_prefix_in_place,
)
from std.benchmark import keep
from std.complex import ComplexSIMD
from std.sys import argv
from std.time import perf_counter_ns


comptime _SAMPLES = 101


def _input[
    dtype: DType
](size: Int) -> List[ComplexSIMD[dtype, 1]] where dtype.is_floating_point():
    var values = List[ComplexSIMD[dtype, 1]](capacity=size)
    for index in range(size):
        values.append(
            ComplexSIMD[dtype, 1](
                Scalar[dtype](Float64((index * 17 + 3) % 29) / 17.0 - 0.75),
                Scalar[dtype](Float64((index * 11 + 5) % 23) / 13.0 - 0.625),
            )
        )
    return values^


def _sort(mut values: List[Int]):
    for index in range(1, len(values)):
        var value = values[index]
        var cursor = index
        while cursor > 0 and values[cursor - 1] > value:
            values[cursor] = values[cursor - 1]
            cursor -= 1
        values[cursor] = value


def _report(
    algorithm: StringLiteral,
    direction: StringLiteral,
    dtype_name: StringLiteral,
    normalization: StringLiteral,
    size: Int,
    batch: Int,
    mut samples: List[Int],
):
    _sort(samples)
    var p50_batch = samples[len(samples) // 2]
    var p95_batch = samples[(len(samples) * 95 + 99) // 100 - 1]
    print(
        "PROFILE shuhafft algorithm=",
        algorithm,
        " direction=",
        direction,
        " dtype=",
        dtype_name,
        " normalization=",
        normalization,
        " n=",
        size,
        " batch_transforms=",
        batch,
        " samples=",
        len(samples),
        " p50_batch_ns=",
        p50_batch,
        " p95_batch_ns=",
        p95_batch,
        " p50_mean_ns_per_transform=",
        Float64(p50_batch) / Float64(batch),
        " p95_mean_ns_per_transform=",
        Float64(p95_batch) / Float64(batch),
        sep="",
    )


def _measure_radix2(size: Int, batch: Int) raises:
    var forward_values = _input[DType.float64](size)
    var inverse_values = _input[DType.float64](size)
    var forward = FFTPlan[DType.float64](
        size, FFTDirection.FORWARD, FFTNormalization.ORTHO
    )
    var inverse = FFTPlan[DType.float64](
        size, FFTDirection.INVERSE, FFTNormalization.ORTHO
    )
    var forward_samples = List[Int](capacity=_SAMPLES)
    var inverse_samples = List[Int](capacity=_SAMPLES)
    for _ in range(4):
        forward.execute_in_place(forward_values)
        inverse.execute_in_place(inverse_values)
    for sample_index in range(_SAMPLES):
        if sample_index % 2 == 0:
            var started = perf_counter_ns()
            for _ in range(batch):
                forward.execute_in_place(forward_values)
            forward_samples.append(perf_counter_ns() - started)
            started = perf_counter_ns()
            for _ in range(batch):
                inverse.execute_in_place(inverse_values)
            inverse_samples.append(perf_counter_ns() - started)
        else:
            var started = perf_counter_ns()
            for _ in range(batch):
                inverse.execute_in_place(inverse_values)
            inverse_samples.append(perf_counter_ns() - started)
            started = perf_counter_ns()
            for _ in range(batch):
                forward.execute_in_place(forward_values)
            forward_samples.append(perf_counter_ns() - started)
    keep(forward_values[0].re + inverse_values[0].re)
    _report("radix2", "forward", "float64", "ortho", size, batch, forward_samples)
    _report("radix2", "inverse", "float64", "ortho", size, batch, inverse_samples)


def _measure_twiddle_layouts(size: Int) raises:
    """Compare separate scalar tables with directly loadable complex values."""
    var source = _input[DType.float64](size)
    var plan = FFTPlan[DType.float64](
        size, FFTDirection.FORWARD, FFTNormalization.NONE
    )
    var scalar_samples = List[Int](capacity=_SAMPLES)
    var complex_samples = List[Int](capacity=_SAMPLES)
    var twiddle_re = List[Float64](capacity=size - 1)
    var twiddle_im = List[Float64](capacity=size - 1)
    for index in range(size - 1):
        twiddle_re.append(plan._twiddles[index].re)
        twiddle_im.append(plan._twiddles[index].im)
    for _ in range(4):
        var scalar_values = List[ComplexSIMD[DType.float64, 1]](copy=source)
        var complex_values = List[ComplexSIMD[DType.float64, 1]](copy=source)
        _radix2_in_place(
            scalar_values,
            twiddle_re,
            twiddle_im,
            plan._bit_reversal,
        )
        _radix2_prefix_in_place(
            complex_values, size, plan._twiddles, plan._bit_reversal
        )
        keep(scalar_values[0].re + complex_values[0].re)
    for sample_index in range(_SAMPLES):
        var scalar_values = List[ComplexSIMD[DType.float64, 1]](copy=source)
        var complex_values = List[ComplexSIMD[DType.float64, 1]](copy=source)
        if sample_index % 2 == 0:
            var started = perf_counter_ns()
            _radix2_in_place(
                scalar_values,
                twiddle_re,
                twiddle_im,
                plan._bit_reversal,
            )
            scalar_samples.append(perf_counter_ns() - started)
            started = perf_counter_ns()
            _radix2_prefix_in_place(
                complex_values, size, plan._twiddles, plan._bit_reversal
            )
            complex_samples.append(perf_counter_ns() - started)
        else:
            var started = perf_counter_ns()
            _radix2_prefix_in_place(
                complex_values, size, plan._twiddles, plan._bit_reversal
            )
            complex_samples.append(perf_counter_ns() - started)
            started = perf_counter_ns()
            _radix2_in_place(
                scalar_values,
                twiddle_re,
                twiddle_im,
                plan._bit_reversal,
            )
            scalar_samples.append(perf_counter_ns() - started)
        keep(scalar_values[0].re + complex_values[0].re)
    _report(
        "radix2-scalar-twiddle",
        "forward",
        "float64",
        "none",
        size,
        1,
        scalar_samples,
    )
    _report(
        "radix2-complex-twiddle",
        "forward",
        "float64",
        "none",
        size,
        1,
        complex_samples,
    )


def _measure_bluestein(size: Int, batch: Int) raises:
    var values = _input[DType.float64](size)
    var spectrum = List[ComplexSIMD[DType.float64, 1]](
        length=size, fill=ComplexSIMD[DType.float64, 1](0.0)
    )
    var restored = List[ComplexSIMD[DType.float64, 1]](
        length=size, fill=ComplexSIMD[DType.float64, 1](0.0)
    )
    var forward = BluesteinFFTPlan[DType.float64](
        size, FFTDirection.FORWARD, FFTNormalization.BACKWARD
    )
    var inverse = BluesteinFFTPlan[DType.float64](
        size, FFTDirection.INVERSE, FFTNormalization.BACKWARD
    )
    var forward_samples = List[Int](capacity=_SAMPLES)
    var inverse_samples = List[Int](capacity=_SAMPLES)
    for _ in range(4):
        forward.execute_into(values, spectrum)
        inverse.execute_into(spectrum, restored)
    for sample_index in range(_SAMPLES):
        if sample_index % 2 == 0:
            var started = perf_counter_ns()
            for _ in range(batch):
                forward.execute_into(values, spectrum)
            forward_samples.append(perf_counter_ns() - started)
            started = perf_counter_ns()
            for _ in range(batch):
                inverse.execute_into(spectrum, restored)
            inverse_samples.append(perf_counter_ns() - started)
        else:
            var started = perf_counter_ns()
            for _ in range(batch):
                inverse.execute_into(spectrum, restored)
            inverse_samples.append(perf_counter_ns() - started)
            started = perf_counter_ns()
            for _ in range(batch):
                forward.execute_into(values, spectrum)
            forward_samples.append(perf_counter_ns() - started)
    keep(restored[0].re)
    _report(
        "bluestein", "forward", "float64", "backward", size, batch, forward_samples
    )
    _report(
        "bluestein", "inverse", "float64", "backward", size, batch, inverse_samples
    )


def _measure_interleaved_simd[
    dtype: DType
](dtype_name: StringLiteral, size: Int) raises where dtype.is_floating_point():
    var plan = RealFFTPlan[dtype](2 * size, FFTNormalization.NONE)
    var source = List[Scalar[dtype]](capacity=2 * size)
    for value in _input[dtype](size):
        source.append(value.re)
        source.append(value.im)
    var scalar_samples = List[Int](capacity=_SAMPLES)
    var simd_samples = List[Int](capacity=_SAMPLES)
    for _ in range(4):
        var scalar_values = List[Scalar[dtype]](copy=source)
        var simd_values = List[Scalar[dtype]](copy=source)
        _radix2_interleaved_scalar_in_place(
            scalar_values,
            size,
            plan._inverse_twiddle_interleaved,
            plan._inverse_plan._bit_reversal,
        )
        _radix2_interleaved_in_place(
            simd_values,
            size,
            plan._inverse_twiddle_interleaved,
            plan._inverse_plan._bit_reversal,
        )
        keep(scalar_values[0] + simd_values[0])
    for sample_index in range(_SAMPLES):
        var scalar_values = List[Scalar[dtype]](copy=source)
        var simd_values = List[Scalar[dtype]](copy=source)
        if sample_index % 2 == 0:
            var started = perf_counter_ns()
            _radix2_interleaved_scalar_in_place(
                scalar_values,
                size,
                plan._inverse_twiddle_interleaved,
                plan._inverse_plan._bit_reversal,
            )
            scalar_samples.append(perf_counter_ns() - started)
            started = perf_counter_ns()
            _radix2_interleaved_in_place(
                simd_values,
                size,
                plan._inverse_twiddle_interleaved,
                plan._inverse_plan._bit_reversal,
            )
            simd_samples.append(perf_counter_ns() - started)
        else:
            var started = perf_counter_ns()
            _radix2_interleaved_in_place(
                simd_values,
                size,
                plan._inverse_twiddle_interleaved,
                plan._inverse_plan._bit_reversal,
            )
            simd_samples.append(perf_counter_ns() - started)
            started = perf_counter_ns()
            _radix2_interleaved_scalar_in_place(
                scalar_values,
                size,
                plan._inverse_twiddle_interleaved,
                plan._inverse_plan._bit_reversal,
            )
            scalar_samples.append(perf_counter_ns() - started)
        keep(scalar_values[0] + simd_values[0])
    _report(
        "interleaved-scalar", "inverse", dtype_name, "none", size, 1, scalar_samples
    )
    _report("interleaved-simd", "inverse", dtype_name, "none", size, 1, simd_samples)


def _radix2_profiler_burn() raises:
    # ORTHO keeps repeated same-direction transforms bounded while the profiler
    # observes one deterministic radix-2 phase and no Bluestein work.
    var radix_values = _input[DType.float64](65536)
    var radix_forward = FFTPlan[DType.float64](
        65536, FFTDirection.FORWARD, FFTNormalization.ORTHO
    )
    print(
        "PROFILE_BURN_READY shuhafft phase=radix2-burn dtype=float64 n=65536",
        flush=True,
    )
    for _ in range(4_000):
        radix_forward.execute_in_place(radix_values)
    keep(radix_values[0].re)


def _bluestein_profiler_burn() raises:
    # Fixed input and caller-owned output isolate one deterministic Bluestein
    # forward phase; construction remains outside the repeated region.
    var awkward_values = _input[DType.float64](65521)
    var spectrum = List[ComplexSIMD[DType.float64, 1]](
        length=65521, fill=ComplexSIMD[DType.float64, 1](0.0)
    )
    var awkward_forward = BluesteinFFTPlan[DType.float64](
        65521, FFTDirection.FORWARD, FFTNormalization.BACKWARD
    )
    print(
        "PROFILE_BURN_READY shuhafft phase=bluestein-burn dtype=float64 n=65521",
        flush=True,
    )
    for _ in range(800):
        awkward_forward.execute_into(awkward_values, spectrum)
    keep(spectrum[0].re)


def main() raises:
    var arguments = argv()
    if len(arguments) == 2:
        var phase = String(arguments[1])
        print(
            "PROFILE_BURN_HEADER shuhafft mojo=1.0.0 phase=",
            phase,
            " deterministic_single_phase=true",
            sep="",
        )
        if phase == "radix2-burn":
            _radix2_profiler_burn()
            return
        if phase == "bluestein-burn":
            _bluestein_profiler_burn()
            return
        raise Error("unknown profiler phase; use radix2-burn or bluestein-burn")
    if len(arguments) != 1:
        raise Error("profile_fft accepts at most one profiler phase")

    print(
        "PROFILE_HEADER shuhafft mojo=1.0.0 build=mojo-build ",
        "statistic=nearest-rank-p50-p95-of-batch-elapsed ",
        "normalized_metric=mean-ns-per-transform-within-percentile-batch ",
        "samples=101 warmup=4 timer=single-per-batch-perf_counter_ns ",
        "profiler=macOS-sample",
        sep="",
    )
    _measure_radix2(256, 32)
    _measure_radix2(4096, 4)
    _measure_radix2(65536, 1)
    _measure_twiddle_layouts(4096)
    _measure_twiddle_layouts(65536)
    _measure_bluestein(257, 16)
    _measure_bluestein(4093, 2)
    _measure_bluestein(65521, 1)
    _measure_interleaved_simd[DType.float32]("float32", 4096)
    _measure_interleaved_simd[DType.float32]("float32", 65536)
    _measure_interleaved_simd[DType.float64]("float64", 4096)
    _measure_interleaved_simd[DType.float64]("float64", 65536)
