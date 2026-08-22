from shuhafft import BluesteinFFTPlan, FFTDirection, FFTNormalization, fft, ifft
from std.complex import ComplexFloat64, ComplexSIMD
from std.math import cos, sin, sqrt
from std.testing import TestSuite, assert_equal, assert_raises, assert_true


def _values(size: Int) -> List[ComplexSIMD[DType.float64, 1]]:
    var values = List[ComplexSIMD[DType.float64, 1]](capacity=size)
    for index in range(size):
        values.append(
            ComplexSIMD[DType.float64, 1](
                Float64((index * 17 + 3) % 11) / 7.0 - 0.5,
                Float64((index * 13 + 5) % 9) / 6.0 - 0.75,
            )
        )
    return values^


def _dft[
    dtype: DType
](values: List[ComplexSIMD[dtype, 1]], direction: FFTDirection) -> List[ComplexFloat64]:
    var size = len(values)
    var output = List[ComplexFloat64](capacity=size)
    var sign = 1.0 if direction.is_inverse() else -1.0
    var two_pi = 6.283185307179586476925286766559
    for k in range(size):
        var sum_re = 0.0
        var sum_im = 0.0
        for j in range(size):
            var angle = sign * two_pi * Float64((j * k) % size) / Float64(size)
            var weight_re = cos(angle)
            var weight_im = sin(angle)
            sum_re += (
                Float64(values[j].re) * weight_re - Float64(values[j].im) * weight_im
            )
            sum_im += (
                Float64(values[j].re) * weight_im + Float64(values[j].im) * weight_re
            )
        output.append(ComplexFloat64(sum_re, sum_im))
    return output^


def _relative_error[
    dtype: DType
](actual: List[ComplexSIMD[dtype, 1]], expected: List[ComplexFloat64]) -> Float64:
    var difference_energy = 0.0
    var reference_energy = 0.0
    for index in range(len(actual)):
        var difference_re = Float64(actual[index].re) - expected[index].re
        var difference_im = Float64(actual[index].im) - expected[index].im
        difference_energy += (
            difference_re * difference_re + difference_im * difference_im
        )
        reference_energy += expected[index].squared_norm()
    if reference_energy == 0.0:
        return sqrt(difference_energy)
    return sqrt(difference_energy / reference_energy)


def test_bluestein_matches_scalar_dft_for_awkward_lengths() raises:
    for size in [1, 3, 5, 6, 7, 10, 17, 31]:
        var values = _values(size)
        for direction in [FFTDirection.FORWARD, FFTDirection.INVERSE]:
            var plan = BluesteinFFTPlan[DType.float64](
                size, direction, FFTNormalization.NONE
            )
            var actual = plan.execute(values)
            var expected = _dft(values, direction)
            assert_true(_relative_error(actual, expected) <= 2.0e-14)


def test_bluestein_float32_matches_scalar_dft_for_prime_and_composite_lengths() raises:
    for size in [3, 5, 10, 17, 31]:
        var values64 = _values(size)
        var values32 = List[ComplexSIMD[DType.float32, 1]](capacity=size)
        for value in values64:
            values32.append(
                ComplexSIMD[DType.float32, 1](Float32(value.re), Float32(value.im))
            )
        for direction in [FFTDirection.FORWARD, FFTDirection.INVERSE]:
            var expected = _dft(values32, direction)
            var plan = BluesteinFFTPlan[DType.float32](
                size, direction, FFTNormalization.NONE
            )
            var actual = plan.execute(values32)
            assert_true(_relative_error(actual, expected) <= 8.0e-6)


def test_bluestein_float32_round_trip_for_normalized_modes() raises:
    for size in [3, 10, 17, 31]:
        var values64 = _values(size)
        var original = List[ComplexSIMD[DType.float32, 1]](capacity=size)
        for value in values64:
            original.append(
                ComplexSIMD[DType.float32, 1](Float32(value.re), Float32(value.im))
            )
        for normalization in [
            FFTNormalization.BACKWARD,
            FFTNormalization.FORWARD,
            FFTNormalization.ORTHO,
        ]:
            var forward_plan = BluesteinFFTPlan[DType.float32](
                size, FFTDirection.FORWARD, normalization
            )
            var inverse_plan = BluesteinFFTPlan[DType.float32](
                size, FFTDirection.INVERSE, normalization
            )
            var spectrum = forward_plan.execute(original)
            var restored = inverse_plan.execute(spectrum)
            var expected = List[ComplexFloat64](capacity=size)
            for value in original:
                expected.append(ComplexFloat64(Float64(value.re), Float64(value.im)))
            assert_true(_relative_error(restored, expected) <= 1.2e-5)


def test_bluestein_normalization_matches_scaled_scalar_dft() raises:
    var values = _values(7)
    for direction in [FFTDirection.FORWARD, FFTDirection.INVERSE]:
        for normalization in [
            FFTNormalization.NONE,
            FFTNormalization.BACKWARD,
            FFTNormalization.FORWARD,
            FFTNormalization.ORTHO,
        ]:
            var expected = _dft(values, direction)
            var scale = Float64(normalization.factor[DType.float64](direction, 7))
            for index in range(len(expected)):
                expected[index] *= scale
            var plan = BluesteinFFTPlan[DType.float64](7, direction, normalization)
            var actual = plan.execute(values)
            assert_true(_relative_error(actual, expected) <= 2.0e-14)


def test_bluestein_reuses_workspace_for_into_and_in_place() raises:
    var original = _values(11)
    var plan = BluesteinFFTPlan[DType.float64](
        11, FFTDirection.FORWARD, FFTNormalization.BACKWARD
    )
    var first = plan.make_buffer()
    var second = plan.make_buffer()
    plan.execute_into(original, first)
    plan.execute_into(original, second)
    assert_equal(first, second)

    var in_place = List[ComplexSIMD[DType.float64, 1]](copy=original)
    plan.execute_in_place(in_place)
    assert_equal(first, in_place)


def test_bluestein_normalized_round_trip_and_one_shot_dispatch() raises:
    for size in [3, 9, 19]:
        var original = _values(size)
        var forward = fft(original)
        var restored = ifft(forward)
        var error_energy = 0.0
        for index in range(size):
            var difference = restored[index] - original[index]
            error_energy += Float64(difference.squared_norm())
        assert_true(sqrt(error_energy) <= 2.0e-13 * sqrt(Float64(size)))


def test_bluestein_rejects_invalid_lengths_and_buffers() raises:
    with assert_raises():
        _ = BluesteinFFTPlan[DType.float64](
            0, FFTDirection.FORWARD, FFTNormalization.BACKWARD
        )
    with assert_raises(contains="Bluestein FFT length must be in [1, 1048576]"):
        _ = BluesteinFFTPlan[DType.float64](
            1_048_577, FFTDirection.FORWARD, FFTNormalization.BACKWARD
        )
    var plan = BluesteinFFTPlan[DType.float64](
        7, FFTDirection.FORWARD, FFTNormalization.BACKWARD
    )
    var short_input = _values(6)
    var output = plan.make_buffer()
    with assert_raises():
        plan.execute_into(short_input, output)
    var input = _values(7)
    var short_output = _values(6)
    with assert_raises():
        plan.execute_into(input, short_output)


def test_bluestein_metadata_copy_and_validation_contracts() raises:
    var plan = BluesteinFFTPlan[DType.float64](
        17, FFTDirection.INVERSE, FFTNormalization.ORTHO
    )
    assert_equal(plan.size(), 17)
    assert_equal(plan.workspace_size(), 64)
    assert_true(plan.direction() == FFTDirection.INVERSE)
    assert_true(plan.normalization() == FFTNormalization.ORTHO)
    assert_true(
        plan
        == BluesteinFFTPlan[DType.float64](
            17, FFTDirection.INVERSE, FFTNormalization.ORTHO
        )
    )
    assert_true(
        String(plan) == "BluesteinFFTPlan(size=17, workspace_size=64, inverse, ortho)"
    )
    plan.validate()
    var copied = BluesteinFFTPlan[DType.float64](copy=plan)
    copied._chirp = List[ComplexFloat64]()
    with assert_raises(contains="Bluestein plan tables must match length 17"):
        copied.validate()
    plan.validate()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
