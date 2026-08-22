from shuhafft import (
    BluesteinFFTPlan,
    ComplexFloat64,
    FFTDirection,
    FFTPlan,
    fft,
    fftfreq,
    ifft,
    irfft,
    rfft,
    rfftfreq,
)
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
    var restored = ifft[DType.float64](fft[DType.float64](values))
    for index in range(len(values)):
        assert_almost_equal(restored[index].re, values[index].re, atol=1e-12)
        assert_almost_equal(restored[index].im, values[index].im, atol=1e-12)

    var awkward = List[ComplexFloat64](length=5, fill=ComplexFloat64(0.0))
    awkward[0] = ComplexFloat64(1.0)
    var awkward_plan = BluesteinFFTPlan[DType.float64](5, FFTDirection.FORWARD)
    var awkward_spectrum = awkward_plan.execute(awkward)
    assert_equal(len(awkward_spectrum), 5)
    for value in awkward_spectrum:
        assert_almost_equal(value.re, 1.0, atol=1e-12, rtol=1e-12)
        assert_almost_equal(value.im, 0.0, atol=1e-12, rtol=1e-12)

    var real_values: List[Float64] = [1.0, -2.0, 3.0, 4.0]
    var half_spectrum = rfft[DType.float64](real_values)
    var restored_real = irfft[DType.float64](half_spectrum)
    assert_equal(len(restored_real), len(real_values))
    for index in range(len(real_values)):
        assert_almost_equal(restored_real[index], real_values[index], atol=1e-12)

    var full_frequencies = fftfreq[DType.float64](4, 0.25)
    var half_frequencies = rfftfreq[DType.float64](4, 0.25)
    assert_equal(len(full_frequencies), 4)
    assert_equal(len(half_frequencies), 3)
    assert_almost_equal(full_frequencies[2], -2.0, atol=1e-12)
    assert_almost_equal(half_frequencies[2], 2.0, atol=1e-12)
