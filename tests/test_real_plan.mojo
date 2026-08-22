from shuhafft import (
    ComplexFloat64,
    FFTDirection,
    FFTNormalization,
    FFTPlan,
    RealFFTPlan,
)
from shuhafft._radix2 import (
    _radix2_interleaved_in_place,
    _radix2_interleaved_scalar_in_place,
)
from std.complex import ComplexSIMD
from std.math import cos, sin
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
)


def _lcg_sample(mut state: UInt64) -> Float64:
    # Numerical Recipes LCG, matching test_dft_oracle.mojo.
    state = (state * UInt64(1_664_525) + UInt64(1_013_904_223)) % UInt64(4_294_967_296)
    return Float64(state) / 2147483647.5 - 1.0


def _random_real[
    dtype: DType
](size: Int, seed: UInt64) -> List[Scalar[dtype]] where dtype.is_floating_point():
    var state = seed
    var signal = List[Scalar[dtype]](capacity=size)
    for _ in range(size):
        signal.append(Scalar[dtype](_lcg_sample(state)))
    return signal^


def _complex_signal[
    dtype: DType
](
    signal: List[Scalar[dtype]],
) -> List[
    ComplexSIMD[dtype, 1]
] where dtype.is_floating_point():
    var values = List[ComplexSIMD[dtype, 1]](capacity=len(signal))
    for sample in signal:
        values.append(ComplexSIMD[dtype, 1](sample, 0.0))
    return values^


def _direct_irfft_none[
    dtype: DType
](spectrum: List[ComplexSIMD[dtype, 1]], size: Int) -> List[
    Float64
] where dtype.is_floating_point():
    """Evaluate the unnormalized compact-spectrum inverse definition."""
    var half_size = size // 2
    var output = List[Float64](capacity=size)
    var two_pi = Float64(6.283185307179586476925286766559)
    for sample_index in range(size):
        var nyquist_sign = Float64(1.0) if sample_index % 2 == 0 else Float64(-1.0)
        var value = Float64(spectrum[0].re) + (
            nyquist_sign * Float64(spectrum[half_size].re)
        )
        for bin_index in range(1, half_size):
            var angle = two_pi * Float64(sample_index * bin_index) / Float64(size)
            value += Float64(2.0) * (
                Float64(spectrum[bin_index].re) * cos(angle)
                - Float64(spectrum[bin_index].im) * sin(angle)
            )
        output.append(value)
    return output^


def _assert_inverse_matches_direct[
    dtype: DType
](
    size: Int, seed: UInt64, atol: Float64, rtol: Float64
) raises where dtype.is_floating_point():
    var spectrum = List[ComplexSIMD[dtype, 1]](capacity=size // 2 + 1)
    var state = seed
    for _ in range(size // 2 + 1):
        spectrum.append(
            ComplexSIMD[dtype, 1](
                Scalar[dtype](_lcg_sample(state)),
                Scalar[dtype](_lcg_sample(state)),
            )
        )
    var expected = _direct_irfft_none(spectrum, size)
    var actual = RealFFTPlan[dtype](size, FFTNormalization.NONE).inverse(spectrum)
    for index in range(size):
        assert_almost_equal(
            Float64(actual[index]), expected[index], atol=atol, rtol=rtol
        )


def _assert_interleaved_simd_matches_scalar[
    dtype: DType
](
    size: Int, seed: UInt64, atol: Float64, rtol: Float64
) raises where dtype.is_floating_point():
    var plan = RealFFTPlan[dtype](2 * size, FFTNormalization.NONE)
    var state = seed
    var scalar_values = List[Scalar[dtype]](capacity=2 * size)
    for _ in range(2 * size):
        scalar_values.append(Scalar[dtype](_lcg_sample(state)))
    var simd_values = List[Scalar[dtype]](copy=scalar_values)

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
    for index in range(2 * size):
        assert_almost_equal(
            Float64(simd_values[index]),
            Float64(scalar_values[index]),
            atol=atol,
            rtol=rtol,
        )


def _assert_real_matches_full_complex_float64(size: Int) raises:
    var signal = _random_real[DType.float64](size, UInt64(0x64A11CE5))
    var complex_signal = _complex_signal(signal)
    var expected = FFTPlan[DType.float64](
        size, FFTDirection.FORWARD, FFTNormalization.NONE
    ).execute(complex_signal)
    var actual = RealFFTPlan[DType.float64](size, FFTNormalization.NONE).forward(signal)
    assert_equal(len(actual), size // 2 + 1)
    for index in range(len(actual)):
        assert_almost_equal(
            Float64(actual[index].re),
            Float64(expected[index].re),
            atol=1e-9 * Float64(size),
            rtol=1e-9,
        )
        assert_almost_equal(
            Float64(actual[index].im),
            Float64(expected[index].im),
            atol=1e-9 * Float64(size),
            rtol=1e-9,
        )


def test_float64_r2c_agrees_with_full_complex_fft() raises:
    for size in [8, 16, 64, 256, 1024, 4096]:
        _assert_real_matches_full_complex_float64(size)


def test_float32_r2c_agrees_with_full_complex_fft() raises:
    var size = 64
    var signal = _random_real[DType.float32](size, UInt64(0x32A11CE5))
    var complex_signal = _complex_signal(signal)
    var expected = FFTPlan[DType.float32](
        size, FFTDirection.FORWARD, FFTNormalization.NONE
    ).execute(complex_signal)
    var actual = RealFFTPlan[DType.float32](size, FFTNormalization.NONE).forward(signal)
    for index in range(len(actual)):
        assert_almost_equal(
            Float64(actual[index].re),
            Float64(expected[index].re),
            atol=1e-4,
            rtol=1e-4,
        )
        assert_almost_equal(
            Float64(actual[index].im),
            Float64(expected[index].im),
            atol=1e-4,
            rtol=1e-4,
        )


def test_simd_inverse_matches_direct_dft_arm64_widths() raises:
    # These powers exercise native 4-lane float32 and 2-lane float64 butterfly
    # chunks on ARM64, plus their scalar early stages and interleave tails.
    for size in [8, 16, 32, 64]:
        _assert_inverse_matches_direct[DType.float32](
            size, UInt64(0x32D1FF00 + size), 2e-5 * Float64(size), 2e-5
        )
        _assert_inverse_matches_direct[DType.float64](
            size, UInt64(0x64D1FF00 + size), 1e-11, 1e-11
        )


def test_interleaved_simd_matches_scalar_for_float32_and_float64() raises:
    # Size one exercises an all-scalar stage, while the larger powers cross the
    # native f32/f64 complex widths and exercise complete SIMD groups.
    for size in [1, 2, 4, 8, 16, 32, 64]:
        _assert_interleaved_simd_matches_scalar[DType.float32](
            size, UInt64(0x32A1B000 + size), 2e-5, 2e-5
        )
        _assert_interleaved_simd_matches_scalar[DType.float64](
            size, UInt64(0x64A1B000 + size), 1e-12, 1e-12
        )


def test_simd_float32_round_trip_arm64_widths() raises:
    for size in [8, 16, 32, 64]:
        var original = _random_real[DType.float32](size, UInt64(0x32BACC00 + size))
        var plan = RealFFTPlan[DType.float32](size)
        var spectrum = plan.forward(original)
        var restored = plan.inverse(spectrum)
        for index in range(size):
            assert_almost_equal(
                Float64(restored[index]),
                Float64(original[index]),
                atol=2e-5,
                rtol=2e-5,
            )


def test_forward_contracts_and_exact_endpoint_imaginaries() raises:
    var signal: List[Float64] = [1.0, -2.0, 3.0, 4.0, -1.0, 0.5, 2.5, -3.0]
    var plan = RealFFTPlan[DType.float64](8)
    var spectrum = plan.forward(signal)

    assert_equal(plan.size(), 8)
    assert_equal(plan.spectrum_size(), 5)
    assert_equal(len(spectrum), 5)
    assert_equal(spectrum[0].im, 0.0)
    assert_equal(spectrum[4].im, 0.0)
    assert_almost_equal(spectrum[0].re, 5.0, atol=1e-12, rtol=1e-12)


def test_plan_buffer_makers_support_forward_into_round_trip() raises:
    var plan = RealFFTPlan[DType.float64](8)
    var signal = plan.make_input()
    var spectrum = plan.make_spectrum()
    assert_equal(len(signal), plan.size())
    assert_equal(len(spectrum), plan.spectrum_size())
    for sample in signal:
        assert_equal(sample, 0.0)
    for bin_value in spectrum:
        assert_equal(bin_value.re, 0.0)
        assert_equal(bin_value.im, 0.0)

    for index in range(len(signal)):
        signal[index] = Float64(index) - 3.0
    plan.forward_into(signal, spectrum)
    var restored = plan.make_input()
    plan.inverse_into(spectrum, restored)
    for index in range(len(signal)):
        assert_almost_equal(restored[index], signal[index], atol=1e-12, rtol=1e-12)


def test_two_sample_special_case() raises:
    var signal: List[Float64] = [3.5, -1.5]
    var plan = RealFFTPlan[DType.float64](2)
    var spectrum = plan.forward(signal)
    assert_equal(len(spectrum), 2)
    assert_equal(spectrum[0].re, 2.0)
    assert_equal(spectrum[0].im, 0.0)
    assert_equal(spectrum[1].re, 5.0)
    assert_equal(spectrum[1].im, 0.0)
    var restored = plan.inverse(spectrum)
    assert_equal(restored[0], signal[0])
    assert_equal(restored[1], signal[1])


def _assert_round_trip_float64(size: Int, normalization: FFTNormalization) raises:
    var original = _random_real[DType.float64](size, UInt64(0x64BACC02))
    var plan = RealFFTPlan[DType.float64](size, normalization)
    var spectrum = plan.forward(original)
    var restored = plan.inverse(spectrum)
    for index in range(size):
        assert_almost_equal(restored[index], original[index], atol=1e-10, rtol=1e-10)


def test_default_backward_round_trip() raises:
    _assert_round_trip_float64(8, FFTNormalization.BACKWARD)
    _assert_round_trip_float64(256, FFTNormalization.BACKWARD)


def test_ortho_round_trip() raises:
    _assert_round_trip_float64(16, FFTNormalization.ORTHO)
    _assert_round_trip_float64(64, FFTNormalization.ORTHO)


def test_inverse_ignores_dc_and_nyquist_imaginaries_exactly() raises:
    var signal = _random_real[DType.float64](32, UInt64(0x1A60BE5))
    var plan = RealFFTPlan[DType.float64](32)
    var clean_spectrum = plan.forward(signal)
    var dirty_spectrum = List[ComplexFloat64](copy=clean_spectrum)
    dirty_spectrum[0] = ComplexFloat64(dirty_spectrum[0].re, 12345.0)
    dirty_spectrum[16] = ComplexFloat64(dirty_spectrum[16].re, -98765.0)

    var clean = plan.inverse(clean_spectrum)
    var dirty = plan.inverse(dirty_spectrum)
    for index in range(32):
        assert_equal(dirty[index], clean[index])


def test_constructor_rejects_invalid_real_lengths() raises:
    with assert_raises(contains="power of two >= 2; got 0"):
        _ = RealFFTPlan[DType.float64](0)
    with assert_raises(contains="power of two >= 2; got 1"):
        _ = RealFFTPlan[DType.float64](1)
    with assert_raises(contains="power of two >= 2; got -8"):
        _ = RealFFTPlan[DType.float64](-8)
    with assert_raises(contains="got 1000 (nearest are 512 and 1024)"):
        _ = RealFFTPlan[DType.float64](1000)


def test_constructor_rejects_invalid_normalization_discriminant() raises:
    with assert_raises(contains="normalization discriminant 12 is invalid"):
        _ = RealFFTPlan[DType.float64](8, FFTNormalization(_value=12))


def test_execution_rejects_mismatched_lengths() raises:
    var plan = RealFFTPlan[DType.float64](8)
    var short_signal = List[Float64](length=7, fill=0.0)
    with assert_raises(contains="signal length 7 does not match real FFT plan size 8"):
        _ = plan.forward(short_signal)

    var signal = List[Float64](length=8, fill=0.0)
    var short_output = List[ComplexFloat64](length=4, fill=ComplexFloat64(0.0))
    with assert_raises(
        contains="spectrum length 4 does not match plan spectrum size 5 (= 8 // 2 + 1)"
    ):
        plan.forward_into(signal, short_output)

    var large_plan = RealFFTPlan[DType.float64](1024)
    var large_signal = List[Float64](length=1024, fill=0.0)
    var hundred_bins = List[ComplexFloat64](length=100, fill=ComplexFloat64(0.0))
    with assert_raises(
        contains=(
            "spectrum length 100 does not match plan spectrum size 513 (= 1024 // 2"
            " + 1)"
        )
    ):
        large_plan.forward_into(large_signal, hundred_bins)

    var short_spectrum = List[ComplexFloat64](length=4, fill=ComplexFloat64(0.0))
    with assert_raises(
        contains="spectrum length 4 does not match plan spectrum size 5 (= 8 // 2 + 1)"
    ):
        _ = plan.inverse(short_spectrum)

    var spectrum = List[ComplexFloat64](length=5, fill=ComplexFloat64(0.0))
    var short_inverse_output = List[Float64](length=7, fill=0.0)
    with assert_raises(contains="signal length 7 does not match real FFT plan size 8"):
        plan.inverse_into(spectrum, short_inverse_output)


def test_equality_writable_and_validation_contracts() raises:
    var plan = RealFFTPlan[DType.float64](1024)
    var equal_plan = RealFFTPlan[DType.float64](1024)
    var different_size = RealFFTPlan[DType.float64](512)
    var different_normalization = RealFFTPlan[DType.float64](
        1024, FFTNormalization.ORTHO
    )

    assert_true(plan == equal_plan)
    assert_true(plan != different_size)
    assert_true(plan != different_normalization)
    assert_true(plan.normalization() == FFTNormalization.BACKWARD)
    assert_equal(String(plan), "RealFFTPlan(size=1024, backward)")
    plan.validate()
    plan._size = 1000
    with assert_raises(contains="must remain a power of two >= 2; got 1000"):
        plan.validate()

    var plan_with_invalid_inner_plan = RealFFTPlan[DType.float64](8)
    plan_with_invalid_inner_plan._forward_plan._size = 2
    plan_with_invalid_inner_plan._forward_plan._direction = FFTDirection.INVERSE
    plan_with_invalid_inner_plan._forward_plan._normalization = (
        FFTNormalization.BACKWARD
    )
    with assert_raises(
        contains=(
            "internal plans must match the real plan length; expected half size 4,"
            " forward direction forward, inverse direction inverse, and"
            " normalization none; got forward plan (size 2, direction inverse,"
            " normalization backward) and inverse plan (size 4, direction inverse,"
            " normalization none)"
        )
    ):
        plan_with_invalid_inner_plan.validate()

    var plan_with_missing_recombination_table = RealFFTPlan[DType.float64](8)
    plan_with_missing_recombination_table._recombination_twiddle_im = List[Float64]()
    with assert_raises(
        contains=(
            "recombination tables must match the plan length; plan length 8 expects"
            " half_size + 1 = 5; got recombination_twiddle_re length 5 and"
            " recombination_twiddle_im length 0"
        )
    ):
        plan_with_missing_recombination_table.validate()

    var plan_with_missing_inverse_twiddles = RealFFTPlan[DType.float64](8)
    plan_with_missing_inverse_twiddles._inverse_twiddle_interleaved = List[Float64]()
    with assert_raises(
        contains=(
            "inverse interleaved twiddle table must match the plan length; plan"
            " length 8 expects 6 scalar entries; got 0"
        )
    ):
        plan_with_missing_inverse_twiddles.validate()


def test_forward_accepts_span_window() raises:
    var larger: List[Float64] = [99.0, 1.0, 2.0, 3.0, 4.0, 88.0]
    var spectrum = RealFFTPlan[DType.float64](4).forward(larger[1:5])
    assert_almost_equal(spectrum[0].re, 10.0, atol=1e-12, rtol=1e-12)
    assert_almost_equal(spectrum[1].re, -2.0, atol=1e-12, rtol=1e-12)
    assert_almost_equal(spectrum[1].im, 2.0, atol=1e-12, rtol=1e-12)
    assert_almost_equal(spectrum[2].re, -2.0, atol=1e-12, rtol=1e-12)
    assert_equal(spectrum[0].im, 0.0)
    assert_equal(spectrum[2].im, 0.0)
    assert_equal(larger[0], 99.0)
    assert_equal(larger[5], 88.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
