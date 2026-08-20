from shuhafft import (
    ComplexFloat32,
    ComplexFloat64,
    FFTDirection,
    FFTNormalization,
    FFTPlan,
)
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_raises,
    assert_true,
)


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


def _real_values64(
    a: Float64, b: Float64, c: Float64, d: Float64
) -> List[ComplexFloat64]:
    var values = List[ComplexFloat64](capacity=4)
    values.append(ComplexFloat64(a))
    values.append(ComplexFloat64(b))
    values.append(ComplexFloat64(c))
    values.append(ComplexFloat64(d))
    return values^


def test_direction_and_normalization_contracts() raises:
    var forward = FFTDirection.forward()
    var inverse = FFTDirection.inverse()
    assert_true(forward.is_forward())
    assert_true(inverse.is_inverse())
    assert_true(forward != inverse)
    assert_almost_equal(
        FFTNormalization.backward().factor[DType.float64](inverse, 4), 0.25
    )
    assert_almost_equal(
        FFTNormalization.forward().factor[DType.float64](forward, 4), 0.25
    )
    assert_almost_equal(FFTNormalization.ortho().factor[DType.float64](forward, 4), 0.5)
    assert_almost_equal(FFTNormalization.none().factor[DType.float64](inverse, 4), 1.0)


def test_every_mutated_normalization_representation_is_valid() raises:
    # Mojo 1.0 permits direct field mutation. Each Bool pair must retain one of
    # the four documented meanings rather than becoming an invalid sentinel.
    var normalization = FFTNormalization.none()
    normalization._scale_forward = False
    normalization._scale_inverse = False
    assert_almost_equal(
        normalization.factor[DType.float64](FFTDirection.forward(), 4), 1.0
    )
    assert_almost_equal(
        normalization.factor[DType.float64](FFTDirection.inverse(), 4), 1.0
    )

    normalization._scale_inverse = True
    assert_almost_equal(
        normalization.factor[DType.float64](FFTDirection.forward(), 4), 1.0
    )
    assert_almost_equal(
        normalization.factor[DType.float64](FFTDirection.inverse(), 4), 0.25
    )

    normalization._scale_forward = True
    normalization._scale_inverse = False
    assert_almost_equal(
        normalization.factor[DType.float64](FFTDirection.forward(), 4), 0.25
    )
    assert_almost_equal(
        normalization.factor[DType.float64](FFTDirection.inverse(), 4), 1.0
    )

    normalization._scale_inverse = True
    assert_almost_equal(
        normalization.factor[DType.float64](FFTDirection.forward(), 4), 0.5
    )
    assert_almost_equal(
        normalization.factor[DType.float64](FFTDirection.inverse(), 4), 0.5
    )


def test_plan_rejects_invalid_lengths() raises:
    with assert_raises(contains="non-zero power of two"):
        _ = FFTPlan[DType.float64](0, FFTDirection.forward())
    with assert_raises(contains="non-zero power of two"):
        _ = FFTPlan[DType.float64](3, FFTDirection.forward())


def test_plan_rejects_mismatched_input() raises:
    var plan = FFTPlan[DType.float64](4, FFTDirection.forward())
    var values = List[ComplexFloat64](capacity=2)
    values.append(ComplexFloat64(1.0))
    values.append(ComplexFloat64(2.0))
    with assert_raises(contains="does not match"):
        plan.execute_in_place(values)


def test_plan_validate_provides_explicit_invariant_checkpoint() raises:
    var plan = FFTPlan[DType.float64](4, FFTDirection.forward())
    plan.validate()
    plan._size = 3
    with assert_raises(contains="must remain a non-zero power of two"):
        plan.validate()


def test_float64_four_point_reference_and_input_preservation() raises:
    # DFT([1, 2, 3, 4]) = [10, -2+2i, -2, -2-2i].
    var values = _real_values64(1.0, 2.0, 3.0, 4.0)
    var plan = FFTPlan[DType.float64](4, FFTDirection.forward())
    var result = plan.execute(values)

    _assert_complex64(values[0], 1.0, 0.0)
    _assert_complex64(values[3], 4.0, 0.0)
    _assert_complex64(result[0], 10.0, 0.0)
    _assert_complex64(result[1], -2.0, 2.0)
    _assert_complex64(result[2], -2.0, 0.0)
    _assert_complex64(result[3], -2.0, -2.0)


def test_normalization_modes_scale_transform_execution() raises:
    var values = _real_values64(1.0, 2.0, 3.0, 4.0)

    var forward_scaled = FFTPlan[DType.float64](
        4, FFTDirection.forward(), FFTNormalization.forward()
    ).execute(values)
    _assert_complex64(forward_scaled[0], 2.5, 0.0)
    _assert_complex64(forward_scaled[1], -0.5, 0.5)
    _assert_complex64(forward_scaled[2], -0.5, 0.0)
    _assert_complex64(forward_scaled[3], -0.5, -0.5)

    var spectrum = FFTPlan[DType.float64](
        4, FFTDirection.forward(), FFTNormalization.none()
    ).execute(values)
    var inverse_unscaled = FFTPlan[DType.float64](
        4, FFTDirection.inverse(), FFTNormalization.none()
    ).execute(spectrum)
    for index in range(4):
        _assert_complex64(
            inverse_unscaled[index], values[index].re * 4.0, values[index].im * 4.0
        )

    var ortho_spectrum = FFTPlan[DType.float64](
        4, FFTDirection.forward(), FFTNormalization.ortho()
    ).execute(values)
    _assert_complex64(ortho_spectrum[0], 5.0, 0.0)
    _assert_complex64(ortho_spectrum[1], -1.0, 1.0)
    _assert_complex64(ortho_spectrum[2], -1.0, 0.0)
    _assert_complex64(ortho_spectrum[3], -1.0, -1.0)
    var ortho_round_trip = FFTPlan[DType.float64](
        4, FFTDirection.inverse(), FFTNormalization.ortho()
    ).execute(ortho_spectrum)
    for index in range(4):
        _assert_complex64(ortho_round_trip[index], values[index].re, values[index].im)


def test_float32_normalization_modes_scale_transform_execution() raises:
    # An origin delta has the same unscaled value in every output bin, making
    # every direction/convention combination an exact scaling reference.
    var values = List[ComplexFloat32](length=4, fill=ComplexFloat32(0.0))
    values[0] = ComplexFloat32(2.0, -1.0)

    var none_forward = FFTPlan[DType.float32](
        4, FFTDirection.forward(), FFTNormalization.none()
    ).execute(values)
    var none_inverse = FFTPlan[DType.float32](
        4, FFTDirection.inverse(), FFTNormalization.none()
    ).execute(values)
    var backward_forward = FFTPlan[DType.float32](
        4, FFTDirection.forward(), FFTNormalization.backward()
    ).execute(values)
    var backward_inverse = FFTPlan[DType.float32](
        4, FFTDirection.inverse(), FFTNormalization.backward()
    ).execute(values)
    var forward_forward = FFTPlan[DType.float32](
        4, FFTDirection.forward(), FFTNormalization.forward()
    ).execute(values)
    var forward_inverse = FFTPlan[DType.float32](
        4, FFTDirection.inverse(), FFTNormalization.forward()
    ).execute(values)
    var ortho_forward = FFTPlan[DType.float32](
        4, FFTDirection.forward(), FFTNormalization.ortho()
    ).execute(values)
    var ortho_inverse = FFTPlan[DType.float32](
        4, FFTDirection.inverse(), FFTNormalization.ortho()
    ).execute(values)

    for index in range(4):
        _assert_complex32(none_forward[index], 2.0, -1.0)
        _assert_complex32(none_inverse[index], 2.0, -1.0)
        _assert_complex32(backward_forward[index], 2.0, -1.0)
        _assert_complex32(backward_inverse[index], 0.5, -0.25)
        _assert_complex32(forward_forward[index], 0.5, -0.25)
        _assert_complex32(forward_inverse[index], 2.0, -1.0)
        _assert_complex32(ortho_forward[index], 1.0, -0.5)
        _assert_complex32(ortho_inverse[index], 1.0, -0.5)


def test_float32_complex_round_trip_in_place() raises:
    var original = List[ComplexFloat32](capacity=8)
    original.append(ComplexFloat32(1.0, -0.5))
    original.append(ComplexFloat32(-2.0, 0.25))
    original.append(ComplexFloat32(0.5, 1.5))
    original.append(ComplexFloat32(4.0, -1.0))
    original.append(ComplexFloat32(-0.75, 2.0))
    original.append(ComplexFloat32(1.25, 0.0))
    original.append(ComplexFloat32(3.5, -2.5))
    original.append(ComplexFloat32(-1.0, 0.75))
    var values = List[ComplexFloat32](copy=original)
    var forward = FFTPlan[DType.float32](8, FFTDirection.forward())
    var inverse = FFTPlan[DType.float32](8, FFTDirection.inverse())
    forward.execute_in_place(values)
    inverse.execute_in_place(values)
    for index in range(len(values)):
        _assert_complex32(values[index], original[index].re, original[index].im)


def test_float32_shifted_impulse_forward_inverse_reference() raises:
    # Direct references make the opposite forward/inverse phase signs visible;
    # default backward normalization contributes 1/8 to the inverse result.
    var values = List[ComplexFloat32](length=8, fill=ComplexFloat32(0.0))
    values[1] = ComplexFloat32(1.0)
    var forward = FFTPlan[DType.float32](8, FFTDirection.forward()).execute(values)
    var inverse = FFTPlan[DType.float32](8, FFTDirection.inverse()).execute(values)
    var root_half = Float32(0.707106781186547524400844362105)
    var scaled_root_half = root_half / 8.0

    _assert_complex32(forward[0], 1.0, 0.0)
    _assert_complex32(forward[1], root_half, -root_half)
    _assert_complex32(forward[2], 0.0, -1.0)
    _assert_complex32(forward[3], -root_half, -root_half)
    _assert_complex32(forward[4], -1.0, 0.0)
    _assert_complex32(forward[5], -root_half, root_half)
    _assert_complex32(forward[6], 0.0, 1.0)
    _assert_complex32(forward[7], root_half, root_half)

    _assert_complex32(inverse[0], 0.125, 0.0)
    _assert_complex32(inverse[1], scaled_root_half, scaled_root_half)
    _assert_complex32(inverse[2], 0.0, 0.125)
    _assert_complex32(inverse[3], -scaled_root_half, scaled_root_half)
    _assert_complex32(inverse[4], -0.125, 0.0)
    _assert_complex32(inverse[5], -scaled_root_half, -scaled_root_half)
    _assert_complex32(inverse[6], 0.0, -0.125)
    _assert_complex32(inverse[7], scaled_root_half, -scaled_root_half)


def test_float64_shifted_impulse_forward_inverse_reference() raises:
    # Validate all bins, inverse phase signs, and the three-stage
    # bit-reversal/butterfly ordering directly rather than through a round trip.
    var values = List[ComplexFloat64](length=8, fill=ComplexFloat64(0.0))
    values[1] = ComplexFloat64(1.0)
    var forward = FFTPlan[DType.float64](8, FFTDirection.forward()).execute(values)
    var inverse = FFTPlan[DType.float64](8, FFTDirection.inverse()).execute(values)
    var root_half = Float64(0.707106781186547524400844362105)
    var scaled_root_half = root_half / 8.0

    _assert_complex64(forward[0], 1.0, 0.0)
    _assert_complex64(forward[1], root_half, -root_half)
    _assert_complex64(forward[2], 0.0, -1.0)
    _assert_complex64(forward[3], -root_half, -root_half)
    _assert_complex64(forward[4], -1.0, 0.0)
    _assert_complex64(forward[5], -root_half, root_half)
    _assert_complex64(forward[6], 0.0, 1.0)
    _assert_complex64(forward[7], root_half, root_half)

    _assert_complex64(inverse[0], 0.125, 0.0)
    _assert_complex64(inverse[1], scaled_root_half, scaled_root_half)
    _assert_complex64(inverse[2], 0.0, 0.125)
    _assert_complex64(inverse[3], -scaled_root_half, scaled_root_half)
    _assert_complex64(inverse[4], -0.125, 0.0)
    _assert_complex64(inverse[5], -scaled_root_half, -scaled_root_half)
    _assert_complex64(inverse[6], 0.0, -0.125)
    _assert_complex64(inverse[7], scaled_root_half, -scaled_root_half)


def test_float64_sixty_four_point_round_trip() raises:
    # Exercise six butterfly stages and recursively generated twiddles with a
    # deterministic, non-symmetric complex fixture.
    var original = List[ComplexFloat64](capacity=64)
    for index in range(64):
        var real = Float64((index * 17) % 23 - 11) / 8.0
        var imaginary = Float64((index * 7) % 19 - 9) / 9.0
        original.append(ComplexFloat64(real, imaginary))

    var spectrum = FFTPlan[DType.float64](64, FFTDirection.forward()).execute(original)
    var restored = FFTPlan[DType.float64](64, FFTDirection.inverse()).execute(spectrum)
    for index in range(64):
        _assert_complex64(restored[index], original[index].re, original[index].im)


def test_float64_parseval_for_backward_normalization() raises:
    var values = List[ComplexFloat64](capacity=4)
    values.append(ComplexFloat64(1.0, 2.0))
    values.append(ComplexFloat64(-3.0, 0.5))
    values.append(ComplexFloat64(2.5, -1.0))
    values.append(ComplexFloat64(0.0, 4.0))
    var result = FFTPlan[DType.float64](4, FFTDirection.forward()).execute(values)
    var time_energy = Float64(0.0)
    var frequency_energy = Float64(0.0)
    for index in range(4):
        time_energy += values[index].squared_norm()
        frequency_energy += result[index].squared_norm()
    assert_almost_equal(time_energy, frequency_energy / 4.0, atol=1e-12, rtol=1e-12)


def test_singleton_is_identity_for_all_normalizations() raises:
    var value = List[ComplexFloat64](capacity=1)
    value.append(ComplexFloat64(3.0, -2.0))
    for normalization in [
        FFTNormalization.none(),
        FFTNormalization.backward(),
        FFTNormalization.forward(),
        FFTNormalization.ortho(),
    ]:
        var result = FFTPlan[DType.float64](
            1, FFTDirection.forward(), normalization
        ).execute(value)
        _assert_complex64(result[0], 3.0, -2.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
