from shuhafft import (
    ComplexFloat64,
    FFTDirection,
    FFTPlan,
    RealFFTPlan,
    fft,
    ifft,
    irfft,
    rfft,
)
from std.math import sin
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
)


def test_fft_matches_plan_and_ifft_round_trip_float64() raises:
    var values: List[ComplexFloat64] = [
        ComplexFloat64(1.0, -0.5),
        ComplexFloat64(-2.0, 3.0),
        ComplexFloat64(0.25, 1.5),
        ComplexFloat64(4.0, -1.0),
        ComplexFloat64(-3.5, 2.25),
        ComplexFloat64(0.0, -2.0),
        ComplexFloat64(1.25, 0.75),
        ComplexFloat64(-0.5, 1.0),
    ]
    var actual = fft[DType.float64](values)
    var expected = FFTPlan[DType.float64](len(values), FFTDirection.FORWARD).execute(
        values
    )
    for index in range(len(values)):
        assert_almost_equal(actual[index].re, expected[index].re, atol=1e-12)
        assert_almost_equal(actual[index].im, expected[index].im, atol=1e-12)

    var restored = ifft[DType.float64](actual)
    for index in range(len(values)):
        assert_almost_equal(restored[index].re, values[index].re, atol=1e-12)
        assert_almost_equal(restored[index].im, values[index].im, atol=1e-12)


def test_rfft_matches_real_plan_float64() raises:
    var signal: List[Float64] = [1.0, -2.0, 3.0, 4.0, -1.0, 0.5, 2.5, -3.0]
    var actual = rfft[DType.float64](signal)
    var expected = RealFFTPlan[DType.float64](len(signal)).forward(signal)
    assert_equal(len(actual), len(expected))
    for index in range(len(actual)):
        assert_almost_equal(actual[index].re, expected[index].re, atol=1e-12)
        assert_almost_equal(actual[index].im, expected[index].im, atol=1e-12)


def test_real_one_shot_round_trip_float64_and_float32() raises:
    var signal64: List[Float64] = [
        0.25,
        -1.5,
        2.0,
        3.25,
        -0.75,
        1.125,
        -2.5,
        0.5,
    ]
    var restored64 = irfft[DType.float64](rfft[DType.float64](signal64))
    for index in range(len(signal64)):
        assert_almost_equal(restored64[index], signal64[index], atol=1e-12)

    var signal32: List[Float32] = [
        0.25,
        -1.5,
        2.0,
        3.25,
        -0.75,
        1.125,
        -2.5,
        0.5,
    ]
    var restored32 = irfft[DType.float32](rfft[DType.float32](signal32))
    for index in range(len(signal32)):
        assert_almost_equal(restored32[index], signal32[index], atol=1e-5)


def test_readme_sine_peak_is_bin_50() raises:
    var sample_count = 1024
    var signal = List[Float64](capacity=sample_count)
    var two_pi = 6.283185307179586476925286766559
    for sample_index in range(sample_count):
        signal.append(
            sin(two_pi * 50.0 * Float64(sample_index) / Float64(sample_count))
        )

    var spectrum = rfft[DType.float64](signal)
    var peak_bin = 1
    var peak_magnitude_squared = Float64(spectrum[peak_bin].squared_norm())
    var runner_up_magnitude_squared: Float64 = 0.0
    for bin_index in range(2, len(spectrum)):
        var magnitude_squared = Float64(spectrum[bin_index].squared_norm())
        if magnitude_squared > peak_magnitude_squared:
            runner_up_magnitude_squared = peak_magnitude_squared
            peak_magnitude_squared = magnitude_squared
            peak_bin = bin_index
        elif magnitude_squared > runner_up_magnitude_squared:
            runner_up_magnitude_squared = magnitude_squared

    assert_equal(peak_bin, 50)
    assert_true(
        peak_magnitude_squared > 10_000.0 * runner_up_magnitude_squared,
        msg="bin 50 magnitude must exceed every other non-DC bin by more than 100x",
    )


def test_irfft_rejects_spectrum_too_short_with_teaching_message() raises:
    var empty = List[ComplexFloat64]()
    with assert_raises(
        contains=(
            "irfft spectrum length 0 cannot imply a supported real length; at least"
            " 2 bins are required"
        )
    ):
        _ = irfft[DType.float64](empty)

    var singleton = List[ComplexFloat64](length=1, fill=ComplexFloat64(0.0))
    with assert_raises(
        contains=(
            "irfft spectrum length 1 cannot imply a supported real length; at least"
            " 2 bins are required (for example, spectrum length 2 reconstructs 2"
            " real samples)"
        )
    ):
        _ = irfft[DType.float64](singleton)


def test_irfft_rejects_non_power_of_two_implied_length() raises:
    var spectrum = List[ComplexFloat64](length=4, fill=ComplexFloat64(0.0))
    with assert_raises(
        contains=(
            "irfft spectrum length 4 implies real length 6, which must be a power"
            " of two >= 2; for example, spectrum length 3 reconstructs 4 real"
            " samples"
        )
    ):
        _ = irfft[DType.float64](spectrum)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
