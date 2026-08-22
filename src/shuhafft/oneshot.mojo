"""One-shot Fourier transform conveniences."""

from std.complex import ComplexSIMD

from .bluestein import BluesteinFFTPlan
from .direction import FFTDirection
from .normalization import FFTNormalization
from .plan import FFTPlan, _is_power_of_two, _lower_bounding_power_of_two
from .real_plan import RealFFTPlan


def fft[
    dtype: DType
](
    values: Span[ComplexSIMD[dtype, 1], _],
    *,
    normalization: FFTNormalization = FFTNormalization.BACKWARD,
) raises -> List[ComplexSIMD[dtype, 1]] where dtype.is_floating_point():
    """Return the one-shot complex FFT with the requested normalization.

    NumPy `norm="backward"`, `"ortho"`, and `"forward"` map to
    `FFTNormalization.BACKWARD`, `ORTHO`, and `FORWARD`. Power-of-two lengths
    use radix-2; other lengths use Bluestein convolution. Use `FFTPlan` for
    repeated radix-2 transforms or `BluesteinFFTPlan` for repeated arbitrary-
    length transforms. Only float32 and float64 are supported.
    """
    if _is_power_of_two(len(values)):
        return FFTPlan[dtype](len(values), FFTDirection.FORWARD, normalization).execute(
            values
        )
    var plan = BluesteinFFTPlan[dtype](len(values), FFTDirection.FORWARD, normalization)
    return plan.execute(values)


def fft[
    dtype: DType
](
    values: Span[Scalar[dtype], _],
    *,
    normalization: FFTNormalization = FFTNormalization.BACKWARD,
) raises -> List[ComplexSIMD[dtype, 1]] where dtype.is_floating_point():
    """Return the full-spectrum FFT of real input.

    Real samples are promoted to complex values with zero imaginary parts for
    NumPy parity and full-spectrum needs. Use `rfft` for the efficient
    half-spectrum path for real signals. NumPy `norm="backward"`, `"ortho"`,
    and `"forward"` map to `FFTNormalization.BACKWARD`, `ORTHO`, and `FORWARD`.
    Power-of-two lengths use radix-2 and other lengths use Bluestein
    convolution. Only float32 and float64 are supported.
    """
    var complex_values = List[ComplexSIMD[dtype, 1]](capacity=len(values))
    for value in values:
        complex_values.append(ComplexSIMD[dtype, 1](value, 0.0))
    return fft[dtype](complex_values, normalization=normalization)


def ifft[
    dtype: DType
](
    values: Span[ComplexSIMD[dtype, 1], _],
    *,
    normalization: FFTNormalization = FFTNormalization.BACKWARD,
) raises -> List[ComplexSIMD[dtype, 1]] where dtype.is_floating_point():
    """Return the one-shot complex inverse FFT with the requested normalization.

    With backward normalization, `ifft(fft(x)) == x`. NumPy `norm="backward"`,
    `"ortho"`, and `"forward"` map to `FFTNormalization.BACKWARD`, `ORTHO`, and
    `FORWARD`. Power-of-two lengths use radix-2 and other lengths use Bluestein
    convolution. Use a reusable plan for repeated transforms. Only float32 and
    float64 are supported.
    """
    if _is_power_of_two(len(values)):
        return FFTPlan[dtype](len(values), FFTDirection.INVERSE, normalization).execute(
            values
        )
    var plan = BluesteinFFTPlan[dtype](len(values), FFTDirection.INVERSE, normalization)
    return plan.execute(values)


def ifft[
    dtype: DType
](
    values: Span[Scalar[dtype], _],
    *,
    normalization: FFTNormalization = FFTNormalization.BACKWARD,
) raises -> List[ComplexSIMD[dtype, 1]] where dtype.is_floating_point():
    """Return the full-spectrum inverse FFT of real input.

    Real samples are promoted to complex values with zero imaginary parts for
    NumPy parity and full-spectrum needs. Use `rfft` and `irfft` for the
    efficient half-spectrum path for real signals. NumPy `norm="backward"`,
    `"ortho"`, and `"forward"` map to `FFTNormalization.BACKWARD`, `ORTHO`, and
    `FORWARD`. Power-of-two lengths use radix-2 and other lengths use Bluestein
    convolution. Only float32 and float64 are supported.
    """
    var complex_values = List[ComplexSIMD[dtype, 1]](capacity=len(values))
    for value in values:
        complex_values.append(ComplexSIMD[dtype, 1](value, 0.0))
    return ifft[dtype](complex_values, normalization=normalization)


def rfft[
    dtype: DType
](
    signal: Span[Scalar[dtype], _],
    *,
    normalization: FFTNormalization = FFTNormalization.BACKWARD,
) raises -> List[ComplexSIMD[dtype, 1]] where dtype.is_floating_point():
    """Return the one-shot real FFT with the requested normalization.

    For `n` samples, the result has `n // 2 + 1` DC-first,
    ascending-frequency bins; bin `k` is `k * sample_rate / n`. DC and
    Nyquist have exactly-zero imaginary parts, and `irfft` ignores imaginary
    input there. With backward normalization, `irfft(rfft(x)) == x`. NumPy
    `norm="backward"`, `"ortho"`, and `"forward"` map to
    `FFTNormalization.BACKWARD`, `ORTHO`, and `FORWARD`; use `RealFFTPlan` for
    repeated transforms. Only float32 and float64 are supported.
    """
    return RealFFTPlan[dtype](len(signal), normalization).forward(signal)


def irfft[
    dtype: DType
](
    spectrum: Span[ComplexSIMD[dtype, 1], _],
    *,
    normalization: FFTNormalization = FFTNormalization.BACKWARD,
) raises -> List[Scalar[dtype]] where dtype.is_floating_point():
    """Return the one-shot real inverse FFT with the requested normalization.

    Input is `n // 2 + 1` DC-first, ascending-frequency bins; bin `k` is
    `k * sample_rate / n`. Forward DC and Nyquist imaginaries are exactly zero,
    and nonzero input imaginaries there are ignored. With backward
    normalization, `irfft(rfft(x)) == x`. NumPy `norm="backward"`, `"ortho"`,
    and `"forward"` map to `FFTNormalization.BACKWARD`, `ORTHO`, and `FORWARD`;
    use `RealFFTPlan` for repeated transforms. Only float32 and float64 are
    supported.
    """
    var spectrum_length = len(spectrum)
    if spectrum_length < 2:
        raise Error(
            String(
                "irfft spectrum length ",
                spectrum_length,
                " cannot imply a supported real length; at least 2 bins are ",
                "required (for example, spectrum length 2 reconstructs 2 real ",
                "samples)",
            )
        )

    var real_length = (spectrum_length - 1) * 2
    if not _is_power_of_two(real_length):
        var nearby_real_length = _lower_bounding_power_of_two(real_length)
        raise Error(
            String(
                "irfft spectrum length ",
                spectrum_length,
                " implies real length ",
                real_length,
                ", which must be a power of two >= 2; for example, spectrum ",
                "length ",
                nearby_real_length // 2 + 1,
                " reconstructs ",
                nearby_real_length,
                " real samples",
            )
        )

    return RealFFTPlan[dtype](real_length, normalization).inverse(spectrum)
