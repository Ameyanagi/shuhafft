from shuhafft import BluesteinFFTPlan, ComplexFloat64, FFTDirection, FFTPlan
from std.testing import assert_almost_equal, assert_equal


def main() raises:
    var values = List[ComplexFloat64](capacity=4)
    values.append(ComplexFloat64(1.0))
    values.append(ComplexFloat64(2.0))
    values.append(ComplexFloat64(3.0))
    values.append(ComplexFloat64(4.0))
    var spectrum = FFTPlan[DType.float64](4, FFTDirection.FORWARD).execute(values)
    assert_equal(len(spectrum), 4)
    assert_almost_equal(spectrum[0].re, 10.0, atol=1e-12, rtol=1e-12)
    assert_almost_equal(spectrum[1].re, -2.0, atol=1e-12, rtol=1e-12)
    assert_almost_equal(spectrum[1].im, 2.0, atol=1e-12, rtol=1e-12)

    var awkward = List[ComplexFloat64](length=5, fill=ComplexFloat64(0.0))
    awkward[0] = ComplexFloat64(1.0)
    var awkward_plan = BluesteinFFTPlan[DType.float64](5, FFTDirection.FORWARD)
    var awkward_spectrum = awkward_plan.execute(awkward)
    assert_equal(len(awkward_spectrum), 5)
    for value in awkward_spectrum:
        assert_almost_equal(value.re, 1.0, atol=1e-12, rtol=1e-12)
        assert_almost_equal(value.im, 0.0, atol=1e-12, rtol=1e-12)
