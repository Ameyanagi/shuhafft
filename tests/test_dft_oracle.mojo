from shuhafft import (
    ComplexFloat64,
    FFTDirection,
    FFTNormalization,
    FFTPlan,
)
from std.complex import ComplexSIMD
from std.math import cos, log2, sin, sqrt
from std.testing import TestSuite, assert_true


def _lcg_sample(mut state: UInt64) -> Float64:
    # Numerical Recipes LCG: state = 1664525 * state + 1013904223 (mod 2^32).
    state = (state * UInt64(1_664_525) + UInt64(1_013_904_223)) % UInt64(4_294_967_296)
    return Float64(state) / 2147483647.5 - 1.0


def _random_values[
    dtype: DType
](size: Int, seed: UInt64) -> List[
    ComplexSIMD[dtype, 1]
] where dtype.is_floating_point():
    var state = seed
    var values = List[ComplexSIMD[dtype, 1]](capacity=size)
    for _ in range(size):
        var real = Scalar[dtype](_lcg_sample(state))
        var imaginary = Scalar[dtype](_lcg_sample(state))
        values.append(ComplexSIMD[dtype, 1](real, imaginary))
    return values^


def _naive_dft[
    dtype: DType
](values: List[ComplexSIMD[dtype, 1]], direction: FFTDirection) -> List[
    ComplexFloat64
] where dtype.is_floating_point():
    # Evaluate every exponential directly. Inputs of either dtype are promoted
    # before all products and accumulation, so the oracle is always Float64.
    var size = len(values)
    var result = List[ComplexFloat64](capacity=size)
    var sign = Float64(1.0) if direction.is_inverse() else Float64(-1.0)
    var two_pi = Float64(6.283185307179586476925286766559)
    for k in range(size):
        var sum_real = Float64(0.0)
        var sum_imaginary = Float64(0.0)
        for j in range(size):
            # Reducing first keeps the trigonometric argument in [0, 2*pi).
            var phase_index = (j * k) % size
            var angle = sign * two_pi * Float64(phase_index) / Float64(size)
            var cosine = cos(angle)
            var sine = sin(angle)
            var input_real = Float64(values[j].re)
            var input_imaginary = Float64(values[j].im)
            sum_real += input_real * cosine - input_imaginary * sine
            sum_imaginary += input_real * sine + input_imaginary * cosine
        result.append(ComplexFloat64(sum_real, sum_imaginary))
    return result^


def _relative_l2_oracle_error[
    dtype: DType
](
    actual: List[ComplexSIMD[dtype, 1]], expected: List[ComplexFloat64]
) -> Float64 where dtype.is_floating_point():
    var error_energy = Float64(0.0)
    var reference_energy = Float64(0.0)
    for index in range(len(actual)):
        var difference_real = Float64(actual[index].re) - expected[index].re
        var difference_imaginary = Float64(actual[index].im) - expected[index].im
        error_energy += (
            difference_real * difference_real
            + difference_imaginary * difference_imaginary
        )
        reference_energy += expected[index].squared_norm()
    return sqrt(error_energy) / sqrt(reference_energy)


def _relative_l2_round_trip_error[
    dtype: DType
](
    actual: List[ComplexSIMD[dtype, 1]],
    expected: List[ComplexSIMD[dtype, 1]],
) -> Float64 where dtype.is_floating_point():
    var error_energy = Float64(0.0)
    var reference_energy = Float64(0.0)
    for index in range(len(actual)):
        var expected_real = Float64(expected[index].re)
        var expected_imaginary = Float64(expected[index].im)
        var difference_real = Float64(actual[index].re) - expected_real
        var difference_imaginary = Float64(actual[index].im) - expected_imaginary
        error_energy += (
            difference_real * difference_real
            + difference_imaginary * difference_imaginary
        )
        reference_energy += (
            expected_real * expected_real + expected_imaginary * expected_imaginary
        )
    return sqrt(error_energy) / sqrt(reference_energy)


def _float64_tight_bound(size: Int) -> Float64:
    # Calibration finalized with K = 4 for both dtypes on 2026-08-21 from the
    # host (osx-arm64, Mojo 1.0.0). Pre-Milestone-2 measured relative L2 errors
    # with multiplicative twiddle recurrence (n:error):
    #   forward   f64  4:1.13e-16  8:1.63e-16  64:6.11e-16
    #                  256:1.89e-15  4096:2.90e-14
    #   forward   f32  4:4.76e-08  64:3.34e-07  512:1.80e-06
    #   round-trip f64 4:5.24e-17  8:1.67e-16  64:7.99e-16
    #                  256:3.13e-15  4096:4.80e-14  16384:2.16e-13
    #   round-trip f32 4:3.31e-08  64:5.02e-07  512:2.69e-06
    # Milestone 2 replaced the recurrence with plan-owned direct twiddle tables.
    return 4.0 * log2(Float64(size)) * 2.220446049250313e-16


def _float32_tight_bound(size: Int) -> Float64:
    return 4.0 * log2(Float64(size)) * 1.1920928955078125e-7


def _report_error(
    kind: StringLiteral, dtype: StringLiteral, size: Int, error: Float64, bound: Float64
):
    print(
        "MEASURED",
        kind,
        dtype,
        "n =",
        size,
        "relative L2 error =",
        error,
        "bound =",
        bound,
    )


def test_float64_forward_against_naive_dft() raises:
    # Backward normalization leaves the forward transform unscaled.
    var errors = List[Float64]()
    var bounds = List[Float64]()
    for size in [4, 8, 64, 256, 4096]:
        var values = _random_values[DType.float64](size, UInt64(0x64D0F7A1))
        var expected = _naive_dft(values, FFTDirection.forward())
        var actual = FFTPlan[DType.float64](
            size, FFTDirection.forward(), FFTNormalization.backward()
        ).execute(values)
        var error = _relative_l2_oracle_error(actual, expected)
        var bound = _float64_tight_bound(size)
        _report_error("forward", "float64", size, error, bound)
        errors.append(error)
        bounds.append(bound)
    for index in range(len(errors)):
        assert_true(errors[index] <= bounds[index])


def test_float32_forward_against_naive_dft() raises:
    var errors = List[Float64]()
    var bounds = List[Float64]()
    for size in [4, 64, 512]:
        var values = _random_values[DType.float32](size, UInt64(0x32D0F7A1))
        var expected = _naive_dft(values, FFTDirection.forward())
        var actual = FFTPlan[DType.float32](
            size, FFTDirection.forward(), FFTNormalization.backward()
        ).execute(values)
        var error = _relative_l2_oracle_error(actual, expected)
        var bound = _float32_tight_bound(size)
        _report_error("forward", "float32", size, error, bound)
        errors.append(error)
        bounds.append(bound)
    for index in range(len(errors)):
        assert_true(errors[index] <= bounds[index])


def test_float64_backward_normalized_round_trip() raises:
    var errors = List[Float64]()
    var bounds = List[Float64]()
    for size in [4, 8, 64, 256, 4096, 16384]:
        var original = _random_values[DType.float64](size, UInt64(0x64BACC01))
        var spectrum = FFTPlan[DType.float64](
            size, FFTDirection.forward(), FFTNormalization.backward()
        ).execute(original)
        var restored = FFTPlan[DType.float64](
            size, FFTDirection.inverse(), FFTNormalization.backward()
        ).execute(spectrum)
        var error = _relative_l2_round_trip_error(restored, original)
        var bound = _float64_tight_bound(size)
        _report_error("round-trip", "float64", size, error, bound)
        errors.append(error)
        bounds.append(bound)
    for index in range(len(errors)):
        assert_true(errors[index] <= bounds[index])


def test_float32_backward_normalized_round_trip() raises:
    var errors = List[Float64]()
    var bounds = List[Float64]()
    for size in [4, 64, 512]:
        var original = _random_values[DType.float32](size, UInt64(0x32BACC01))
        var spectrum = FFTPlan[DType.float32](
            size, FFTDirection.forward(), FFTNormalization.backward()
        ).execute(original)
        var restored = FFTPlan[DType.float32](
            size, FFTDirection.inverse(), FFTNormalization.backward()
        ).execute(spectrum)
        var error = _relative_l2_round_trip_error(restored, original)
        var bound = _float32_tight_bound(size)
        _report_error("round-trip", "float32", size, error, bound)
        errors.append(error)
        bounds.append(bound)
    for index in range(len(errors)):
        assert_true(errors[index] <= bounds[index])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
