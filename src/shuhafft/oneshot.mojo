"""One-shot Fourier transform conveniences with backward normalization."""

from std.complex import ComplexSIMD

from .direction import FFTDirection
from .normalization import FFTNormalization
from .plan import FFTPlan, _is_power_of_two, _lower_bounding_power_of_two
from .real_plan import RealFFTPlan


def fft[
    dtype: DType
](values: Span[ComplexSIMD[dtype, 1], _]) raises -> List[
    ComplexSIMD[dtype, 1]
] where dtype.is_floating_point():
    """Return the one-shot complex FFT using backward normalization.

    Use `FFTPlan` for repeated transforms or another normalization convention.
    Only float32 and float64 are supported.
    """
    return FFTPlan[dtype](
        len(values), FFTDirection.FORWARD, FFTNormalization.BACKWARD
    ).execute(values)


def ifft[
    dtype: DType
](values: Span[ComplexSIMD[dtype, 1], _]) raises -> List[
    ComplexSIMD[dtype, 1]
] where dtype.is_floating_point():
    """Return the one-shot complex inverse FFT using backward normalization.

    `ifft(fft(x)) == x`. Use `FFTPlan` for repeated transforms or another
    normalization convention. Only float32 and float64 are supported.
    """
    var spectrum_length = len(values)
    if (
        spectrum_length >= 3
        and not _is_power_of_two(spectrum_length)
        and _is_power_of_two(spectrum_length - 1)
    ):
        var lower = _lower_bounding_power_of_two(spectrum_length)
        var higher = lower * 2
        raise Error(
            String(
                "FFT length must be a non-zero power of two; got ",
                spectrum_length,
                " (nearest are ",
                lower,
                " and ",
                higher,
                "); if this spectrum came from rfft, use irfft to reconstruct the ",
                (spectrum_length - 1) * 2,
                " real samples",
            )
        )
    return FFTPlan[dtype](
        spectrum_length, FFTDirection.INVERSE, FFTNormalization.BACKWARD
    ).execute(values)


def rfft[
    dtype: DType
](signal: Span[Scalar[dtype], _]) raises -> List[
    ComplexSIMD[dtype, 1]
] where dtype.is_floating_point():
    """Return the one-shot real FFT using backward normalization.

    For `n` samples, the result has `n // 2 + 1` DC-first,
    ascending-frequency bins; bin `k` is `k * sample_rate / n`. DC and
    Nyquist have exactly-zero imaginary parts, and `irfft` ignores imaginary
    input there. With this default, `irfft(rfft(x)) == x`. Use `RealFFTPlan`
    for repeated transforms or another normalization convention. Only
    float32 and float64 are supported.
    """
    return RealFFTPlan[dtype](len(signal), FFTNormalization.BACKWARD).forward(signal)


def irfft[
    dtype: DType
](spectrum: Span[ComplexSIMD[dtype, 1], _]) raises -> List[
    Scalar[dtype]
] where dtype.is_floating_point():
    """Return the one-shot real inverse FFT using backward normalization.

    Input is `n // 2 + 1` DC-first, ascending-frequency bins; bin `k` is
    `k * sample_rate / n`. Forward DC and Nyquist imaginaries are exactly zero,
    and nonzero input imaginaries there are ignored. With this default,
    `irfft(rfft(x)) == x`. Use `RealFFTPlan` for repeated transforms or another
    normalization convention. Only float32 and float64 are supported.
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

    return RealFFTPlan[dtype](real_length, FFTNormalization.BACKWARD).inverse(spectrum)
