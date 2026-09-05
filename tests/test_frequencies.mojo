from shuhafft import fftfreq, rfftfreq
from std.math import isfinite
from std.testing import TestSuite, assert_almost_equal, assert_equal, assert_raises


def test_fftfreq_even_and_odd_bin_order() raises:
    var even = fftfreq[DType.float64](8, 0.125)
    var expected_even: List[Float64] = [0.0, 1.0, 2.0, 3.0, -4.0, -3.0, -2.0, -1.0]
    assert_equal(len(even), len(expected_even))
    for index in range(len(even)):
        assert_almost_equal(even[index], expected_even[index], atol=1e-12)

    var odd = fftfreq(5)
    var expected_odd: List[Float64] = [0.0, 0.2, 0.4, -0.4, -0.2]
    assert_equal(len(odd), len(expected_odd))
    for index in range(len(odd)):
        assert_almost_equal(odd[index], expected_odd[index], atol=1e-12)


def test_rfftfreq_even_odd_and_singleton_bin_order() raises:
    var even = rfftfreq[DType.float64](8, 0.125)
    var expected_even: List[Float64] = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert_equal(len(even), len(expected_even))
    for index in range(len(even)):
        assert_almost_equal(even[index], expected_even[index], atol=1e-12)

    var odd = rfftfreq[DType.float64](5, 0.5)
    var expected_odd: List[Float64] = [0.0, 0.4, 0.8]
    assert_equal(len(odd), len(expected_odd))
    for index in range(len(odd)):
        assert_almost_equal(odd[index], expected_odd[index], atol=1e-12)

    var singleton = rfftfreq(1)
    assert_equal(len(singleton), 1)
    assert_equal(singleton[0], 0.0)


def test_frequency_helpers_reject_nonpositive_n() raises:
    with assert_raises(
        contains="fftfreq bin count n must be a positive integer; got 0"
    ):
        _ = fftfreq(0)
    with assert_raises(
        contains="rfftfreq bin count n must be a positive integer; got -3"
    ):
        _ = rfftfreq(-3)


def test_frequency_helpers_reject_zero_sample_spacing() raises:
    with assert_raises(
        contains=(
            "fftfreq sample spacing d must be nonzero; got 0.0; pass"
            " d=1/sample_rate to label bins in hertz"
        )
    ):
        _ = fftfreq[DType.float64](8, 0.0)
    with assert_raises(
        contains=(
            "rfftfreq sample spacing d must be nonzero; got 0.0; pass"
            " d=1/sample_rate to label bins in hertz"
        )
    ):
        _ = rfftfreq[DType.float64](8, 0.0)


def _check_extreme_spacing[
    dtype: DType
](large: Scalar[dtype], small: Scalar[dtype]) raises where dtype.is_floating_point():
    for n in range(4, 6):
        for sign in range(-1, 2, 2):
            var spacing = Scalar[dtype](sign) * large
            var full = fftfreq[dtype](n, spacing)
            var real = rfftfreq[dtype](n, spacing)
            # Normalize subnormal labels before comparison: an absolute tolerance
            # at the original scale would incorrectly accept the old all-zero result.
            for k in range(1, n):
                var signed_bin = k if k < (n + 1) // 2 else k - n
                assert_almost_equal(
                    Float64(full[k]) * Float64(spacing),
                    Float64(signed_bin) / Float64(n),
                    atol=1e-6,
                )
            for k in range(1, n // 2 + 1):
                assert_almost_equal(
                    Float64(real[k]) * Float64(spacing),
                    Float64(k) / Float64(n),
                    atol=1e-6,
                )
            var small_full = fftfreq[dtype](n, Scalar[dtype](sign) * small)
            var small_real = rfftfreq[dtype](n, Scalar[dtype](sign) * small)
            assert_equal(isfinite(small_full[1]), True)
            assert_equal(isfinite(small_real[1]), True)
            assert_almost_equal(
                Float64(small_real[1]) * Float64(small) * Float64(sign),
                1.0 / Float64(n),
                atol=1e-6,
            )
            # Division by zero exposes the sign bit while preserving an exact
            # assertion; negative spacing must still produce positive DC zero.
            assert_equal(1.0 / Float64(full[0]), Float64("inf"))
            assert_equal(1.0 / Float64(real[0]), Float64("inf"))


def test_extreme_finite_spacing_preserves_float32_and_float64_bins() raises:
    _check_extreme_spacing[DType.float32](Float32(1e38), Float32(Float64("2e-39")))
    _check_extreme_spacing[DType.float64](Float64(1e308), Float64(2e-309))


def test_frequency_helpers_reject_nonfinite_spacing() raises:
    var invalid: List[Float64] = [Float64("nan"), Float64("inf"), Float64("-inf")]
    for spacing in invalid:
        with assert_raises(contains="sample spacing d must be finite; got"):
            _ = fftfreq(4, spacing)
        with assert_raises(contains="sample spacing d must be finite; got"):
            _ = rfftfreq(5, spacing)
        with assert_raises(contains="sample spacing d must be finite; got"):
            _ = fftfreq[DType.float32](5, Float32(spacing))
        with assert_raises(contains="sample spacing d must be finite; got"):
            _ = rfftfreq[DType.float32](4, Float32(spacing))


def _check_overflow[
    dtype: DType
](spacing: Scalar[dtype]) raises where dtype.is_floating_point():
    var full = fftfreq[dtype](4, spacing)
    var real = rfftfreq[dtype](5, -spacing)
    assert_equal(Float64(full[1]), Float64("inf"))
    assert_equal(Float64(full[2]), Float64("-inf"))
    assert_equal(Float64(real[1]), Float64("-inf"))
    assert_equal(1.0 / Float64(full[0]), Float64("inf"))
    assert_equal(1.0 / Float64(real[0]), Float64("inf"))
    assert_equal(fftfreq[dtype](1, spacing)[0], Scalar[dtype](0.0))
    assert_equal(rfftfreq[dtype](1, spacing)[0], Scalar[dtype](0.0))


def test_unrepresentable_bins_use_signed_infinity_and_exact_dc_zero() raises:
    _check_overflow[DType.float32](Float32(Float64("1e-45")))
    _check_overflow[DType.float64](Float64("5e-324"))


def test_narrow_dtype_underflow_rounds_only_final_labels() raises:
    var labels = rfftfreq[DType.float16](1024, Scalar[DType.float16](65504))
    assert_equal(labels[1], Scalar[DType.float16](0.0))
    assert_equal(labels[512] > Scalar[DType.float16](0.0), True)
    # The length may exceed the output dtype's finite range. It is a count,
    # so it must not turn into infinity before computing normalized bins.
    var many = rfftfreq[DType.float16](65536)
    assert_equal(Float64(many[32768]), 0.5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
