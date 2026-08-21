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
    var forward = FFTDirection.FORWARD
    var inverse = FFTDirection.INVERSE
    assert_true(forward.is_forward())
    assert_true(inverse.is_inverse())
    assert_true(forward != inverse)
    assert_almost_equal(
        FFTNormalization.BACKWARD.factor[DType.float64](inverse, 4), 0.25
    )
    assert_almost_equal(
        FFTNormalization.FORWARD.factor[DType.float64](forward, 4), 0.25
    )
    assert_almost_equal(FFTNormalization.ORTHO.factor[DType.float64](forward, 4), 0.5)
    assert_almost_equal(FFTNormalization.NONE.factor[DType.float64](inverse, 4), 1.0)


def test_plan_rejects_invalid_lengths() raises:
    with assert_raises(contains="non-zero power of two"):
        _ = FFTPlan[DType.float64](0, FFTDirection.FORWARD)
    with assert_raises(contains="non-zero power of two"):
        _ = FFTPlan[DType.float64](3, FFTDirection.FORWARD)


def test_plan_invalid_length_message_names_nearest_powers() raises:
    with assert_raises(contains="got 1000 (nearest are 512 and 1024)"):
        _ = FFTPlan[DType.float64](1000, FFTDirection.FORWARD)


def test_plan_equality_and_writable_contracts() raises:
    var plan = FFTPlan[DType.float64](4, FFTDirection.FORWARD)
    var equal_plan = FFTPlan[DType.float64](4, FFTDirection.FORWARD)
    var different_size = FFTPlan[DType.float64](8, FFTDirection.FORWARD)
    var different_direction = FFTPlan[DType.float64](4, FFTDirection.INVERSE)
    var different_normalization = FFTPlan[DType.float64](
        4, FFTDirection.FORWARD, FFTNormalization.NONE
    )

    assert_true(plan == equal_plan)
    assert_true(plan != different_size)
    assert_true(plan != different_direction)
    assert_true(plan != different_normalization)
    assert_true(String(plan) == "FFTPlan(size=4, forward, backward)")


def test_plan_rejects_mismatched_input() raises:
    var plan = FFTPlan[DType.float64](4, FFTDirection.FORWARD)
    var values = List[ComplexFloat64](capacity=2)
    values.append(ComplexFloat64(1.0))
    values.append(ComplexFloat64(2.0))
    with assert_raises(contains="input length 2 does not match FFT plan length 4"):
        plan.execute_in_place(values)


def test_plan_make_buffer_returns_zero_filled_execution_storage() raises:
    var plan = FFTPlan[DType.float64](8, FFTDirection.FORWARD)
    var buffer = plan.make_buffer()
    assert_true(len(buffer) == plan.size())
    for value in buffer:
        _assert_complex64(value, 0.0, 0.0)

    buffer[0] = ComplexFloat64(1.0)
    plan.execute_in_place(buffer)
    for value in buffer:
        _assert_complex64(value, 1.0, 0.0)


def test_plan_validate_provides_explicit_invariant_checkpoint() raises:
    var plan = FFTPlan[DType.float64](4, FFTDirection.FORWARD)
    plan.validate()
    plan._size = 3
    with assert_raises(contains="must remain a non-zero power of two; got 3"):
        plan.validate()

    var plan_with_missing_table = FFTPlan[DType.float64](4, FFTDirection.FORWARD)
    plan_with_missing_table._twiddle_im = List[Float64]()
    with assert_raises(
        contains=(
            "tables must match the plan length; plan length 4 expects twiddle_re"
            " length 3, twiddle_im length 3, and bit_reversal length 4; got"
            " twiddle_re length 3, twiddle_im length 0, and bit_reversal length 4"
        )
    ):
        plan_with_missing_table.validate()


def test_plan_precomputed_state_layout_and_copy_independence() raises:
    var plan = FFTPlan[DType.float64](8, FFTDirection.FORWARD)
    assert_true(len(plan._twiddle_re) == 7)
    assert_true(len(plan._twiddle_im) == 7)
    assert_true(len(plan._bit_reversal) == 8)
    # The stage-size-4 table starts at half - 1 = 1.
    assert_almost_equal(plan._twiddle_re[1], 1.0)
    assert_almost_equal(plan._twiddle_im[1], 0.0)
    assert_almost_equal(plan._twiddle_re[2], 0.0, atol=1e-15)
    assert_almost_equal(plan._twiddle_im[2], -1.0)
    var expected_permutation: List[Int] = [0, 4, 2, 6, 1, 5, 3, 7]
    for index in range(8):
        assert_true(plan._bit_reversal[index] == expected_permutation[index])

    var plan_copy = FFTPlan[DType.float64](copy=plan)
    plan_copy._twiddle_re[0] = 2.0
    plan_copy._bit_reversal[0] = 7
    assert_almost_equal(plan._twiddle_re[0], 1.0)
    assert_true(plan._bit_reversal[0] == 0)

    var singleton = FFTPlan[DType.float64](1, FFTDirection.FORWARD)
    assert_true(len(singleton._twiddle_re) == 0)
    assert_true(len(singleton._twiddle_im) == 0)
    assert_true(len(singleton._bit_reversal) == 1)
    assert_true(singleton._bit_reversal[0] == 0)


def test_float64_four_point_reference_and_input_preservation() raises:
    # DFT([1, 2, 3, 4]) = [10, -2+2i, -2, -2-2i].
    var values = _real_values64(1.0, 2.0, 3.0, 4.0)
    var plan = FFTPlan[DType.float64](4, FFTDirection.FORWARD)
    var result = plan.execute(values)

    _assert_complex64(values[0], 1.0, 0.0)
    _assert_complex64(values[3], 4.0, 0.0)
    _assert_complex64(result[0], 10.0, 0.0)
    _assert_complex64(result[1], -2.0, 2.0)
    _assert_complex64(result[2], -2.0, 0.0)
    _assert_complex64(result[3], -2.0, -2.0)


def test_execute_accepts_span_window_without_copy() raises:
    var larger = List[ComplexFloat64](length=6, fill=ComplexFloat64(0.0))
    larger[0] = ComplexFloat64(99.0)
    larger[1] = ComplexFloat64(1.0)
    larger[2] = ComplexFloat64(2.0)
    larger[3] = ComplexFloat64(3.0)
    larger[4] = ComplexFloat64(4.0)
    larger[5] = ComplexFloat64(88.0)

    var result = FFTPlan[DType.float64](4, FFTDirection.FORWARD).execute(larger[1:5])

    _assert_complex64(result[0], 10.0, 0.0)
    _assert_complex64(result[1], -2.0, 2.0)
    _assert_complex64(result[2], -2.0, 0.0)
    _assert_complex64(result[3], -2.0, -2.0)
    _assert_complex64(larger[0], 99.0, 0.0)
    _assert_complex64(larger[1], 1.0, 0.0)
    _assert_complex64(larger[4], 4.0, 0.0)
    _assert_complex64(larger[5], 88.0, 0.0)


def test_normalization_modes_scale_transform_execution() raises:
    var values = _real_values64(1.0, 2.0, 3.0, 4.0)

    var forward_scaled = FFTPlan[DType.float64](
        4, FFTDirection.FORWARD, FFTNormalization.FORWARD
    ).execute(values)
    _assert_complex64(forward_scaled[0], 2.5, 0.0)
    _assert_complex64(forward_scaled[1], -0.5, 0.5)
    _assert_complex64(forward_scaled[2], -0.5, 0.0)
    _assert_complex64(forward_scaled[3], -0.5, -0.5)

    var spectrum = FFTPlan[DType.float64](
        4, FFTDirection.FORWARD, FFTNormalization.NONE
    ).execute(values)
    var inverse_unscaled = FFTPlan[DType.float64](
        4, FFTDirection.INVERSE, FFTNormalization.NONE
    ).execute(spectrum)
    for index in range(4):
        _assert_complex64(
            inverse_unscaled[index], values[index].re * 4.0, values[index].im * 4.0
        )

    var ortho_spectrum = FFTPlan[DType.float64](
        4, FFTDirection.FORWARD, FFTNormalization.ORTHO
    ).execute(values)
    _assert_complex64(ortho_spectrum[0], 5.0, 0.0)
    _assert_complex64(ortho_spectrum[1], -1.0, 1.0)
    _assert_complex64(ortho_spectrum[2], -1.0, 0.0)
    _assert_complex64(ortho_spectrum[3], -1.0, -1.0)
    var ortho_round_trip = FFTPlan[DType.float64](
        4, FFTDirection.INVERSE, FFTNormalization.ORTHO
    ).execute(ortho_spectrum)
    for index in range(4):
        _assert_complex64(ortho_round_trip[index], values[index].re, values[index].im)


def test_float32_normalization_modes_scale_transform_execution() raises:
    # An origin delta has the same unscaled value in every output bin, making
    # every direction/convention combination an exact scaling reference.
    var values = List[ComplexFloat32](length=4, fill=ComplexFloat32(0.0))
    values[0] = ComplexFloat32(2.0, -1.0)

    var none_forward = FFTPlan[DType.float32](
        4, FFTDirection.FORWARD, FFTNormalization.NONE
    ).execute(values)
    var none_inverse = FFTPlan[DType.float32](
        4, FFTDirection.INVERSE, FFTNormalization.NONE
    ).execute(values)
    var backward_forward = FFTPlan[DType.float32](
        4, FFTDirection.FORWARD, FFTNormalization.BACKWARD
    ).execute(values)
    var backward_inverse = FFTPlan[DType.float32](
        4, FFTDirection.INVERSE, FFTNormalization.BACKWARD
    ).execute(values)
    var forward_forward = FFTPlan[DType.float32](
        4, FFTDirection.FORWARD, FFTNormalization.FORWARD
    ).execute(values)
    var forward_inverse = FFTPlan[DType.float32](
        4, FFTDirection.INVERSE, FFTNormalization.FORWARD
    ).execute(values)
    var ortho_forward = FFTPlan[DType.float32](
        4, FFTDirection.FORWARD, FFTNormalization.ORTHO
    ).execute(values)
    var ortho_inverse = FFTPlan[DType.float32](
        4, FFTDirection.INVERSE, FFTNormalization.ORTHO
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
    var forward = FFTPlan[DType.float32](8, FFTDirection.FORWARD)
    var inverse = FFTPlan[DType.float32](8, FFTDirection.INVERSE)
    forward.execute_in_place(values)
    inverse.execute_in_place(values)
    for index in range(len(values)):
        _assert_complex32(values[index], original[index].re, original[index].im)


def test_float32_shifted_impulse_forward_inverse_reference() raises:
    # Direct references make the opposite forward/inverse phase signs visible;
    # default backward normalization contributes 1/8 to the inverse result.
    var values = List[ComplexFloat32](length=8, fill=ComplexFloat32(0.0))
    values[1] = ComplexFloat32(1.0)
    var forward = FFTPlan[DType.float32](8, FFTDirection.FORWARD).execute(values)
    var inverse = FFTPlan[DType.float32](8, FFTDirection.INVERSE).execute(values)
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
    var forward = FFTPlan[DType.float64](8, FFTDirection.FORWARD).execute(values)
    var inverse = FFTPlan[DType.float64](8, FFTDirection.INVERSE).execute(values)
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
    # Exercise six butterfly stages and plan-owned twiddle tables with a
    # deterministic, non-symmetric complex fixture.
    var original = List[ComplexFloat64](capacity=64)
    for index in range(64):
        var real = Float64((index * 17) % 23 - 11) / 8.0
        var imaginary = Float64((index * 7) % 19 - 9) / 9.0
        original.append(ComplexFloat64(real, imaginary))

    var spectrum = FFTPlan[DType.float64](64, FFTDirection.FORWARD).execute(original)
    var restored = FFTPlan[DType.float64](64, FFTDirection.INVERSE).execute(spectrum)
    for index in range(64):
        _assert_complex64(restored[index], original[index].re, original[index].im)


def test_float64_parseval_for_backward_normalization() raises:
    var values = List[ComplexFloat64](capacity=4)
    values.append(ComplexFloat64(1.0, 2.0))
    values.append(ComplexFloat64(-3.0, 0.5))
    values.append(ComplexFloat64(2.5, -1.0))
    values.append(ComplexFloat64(0.0, 4.0))
    var result = FFTPlan[DType.float64](4, FFTDirection.FORWARD).execute(values)
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
        FFTNormalization.NONE,
        FFTNormalization.BACKWARD,
        FFTNormalization.FORWARD,
        FFTNormalization.ORTHO,
    ]:
        var result = FFTPlan[DType.float64](
            1, FFTDirection.FORWARD, normalization
        ).execute(value)
        _assert_complex64(result[0], 3.0, -2.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
