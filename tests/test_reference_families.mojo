from shuhafft import (
    ComplexFloat32,
    ComplexFloat64,
    FFTDirection,
    FFTPlan,
)
from std.testing import TestSuite, assert_almost_equal


def _assert_complex32(
    actual: ComplexFloat32, expected_re: Float32, expected_im: Float32
) raises:
    assert_almost_equal(actual.re, expected_re, atol=1e-5, rtol=1e-5)
    assert_almost_equal(actual.im, expected_im, atol=1e-5, rtol=1e-5)


def _assert_complex64(
    actual: ComplexFloat64, expected_re: Float64, expected_im: Float64
) raises:
    assert_almost_equal(actual.re, expected_re, atol=1e-12, rtol=1e-12)
    assert_almost_equal(actual.im, expected_im, atol=1e-12, rtol=1e-12)


def test_float32_origin_delta_reference() raises:
    # A delta at the origin transforms to its complex amplitude in every bin.
    var delta = List[ComplexFloat32](length=16, fill=ComplexFloat32(0.0))
    delta[0] = ComplexFloat32(1.25, -0.5)
    var spectrum = FFTPlan[DType.float32](16, FFTDirection.FORWARD).execute(delta)

    for index in range(16):
        _assert_complex32(spectrum[index], 1.25, -0.5)
    _assert_complex32(delta[0], 1.25, -0.5)
    for index in range(1, 16):
        _assert_complex32(delta[index], 0.0, 0.0)


def test_float64_delta_inverse_dual_and_plan_reuse() raises:
    # Besides the analytic reference, reuse one plan and mutate a prior output
    # to prove executions do not share result storage or alias their input.
    var delta = List[ComplexFloat64](length=32, fill=ComplexFloat64(0.0))
    delta[0] = ComplexFloat64(2.25, -0.75)
    var forward = FFTPlan[DType.float64](32, FFTDirection.FORWARD)
    var inverse = FFTPlan[DType.float64](32, FFTDirection.INVERSE)

    var first_spectrum = forward.execute(delta)
    for index in range(32):
        _assert_complex64(first_spectrum[index], 2.25, -0.75)

    var restored = inverse.execute(first_spectrum)
    _assert_complex64(restored[0], 2.25, -0.75)
    for index in range(1, 32):
        _assert_complex64(restored[index], 0.0, 0.0)

    var second_spectrum = forward.execute(delta)
    for index in range(32):
        _assert_complex64(second_spectrum[index], 2.25, -0.75)

    # Mutate a completed result only after all executions. Neither the later
    # result nor the inverse result may share its backing storage.
    first_spectrum[7] = ComplexFloat64(99.0, 101.0)
    _assert_complex64(second_spectrum[7], 2.25, -0.75)
    _assert_complex64(restored[7], 0.0, 0.0)
    _assert_complex64(delta[0], 2.25, -0.75)
    for index in range(1, 32):
        _assert_complex64(delta[index], 0.0, 0.0)


def test_float32_constant_reference() raises:
    # A constant sequence has only its DC bin, equal to length * amplitude.
    var values = List[ComplexFloat32](length=32, fill=ComplexFloat32(0.75, -0.125))
    var spectrum = FFTPlan[DType.float32](32, FFTDirection.FORWARD).execute(values)

    _assert_complex32(spectrum[0], 24.0, -4.0)
    for index in range(1, 32):
        _assert_complex32(spectrum[index], 0.0, 0.0)


def test_float64_constant_reference() raises:
    var values = List[ComplexFloat64](length=64, fill=ComplexFloat64(-0.375, 0.0625))
    var spectrum = FFTPlan[DType.float64](64, FFTDirection.FORWARD).execute(values)

    _assert_complex64(spectrum[0], -24.0, 4.0)
    for index in range(1, 64):
        _assert_complex64(spectrum[index], 0.0, 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
