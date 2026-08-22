from shuhafft import fftfreq, rfftfreq
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
