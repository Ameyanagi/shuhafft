"""Frequency-bin labels matching NumPy transform ordering."""


from std.math import isfinite


def fftfreq[
    dtype: DType = DType.float64
](n: Int, d: Scalar[dtype] = 1.0) raises -> List[
    Scalar[dtype]
] where dtype.is_floating_point():
    """Return full-spectrum frequency labels in `fft` bin order.

    The labels are zero through the positive-frequency bins, followed by the
    negative-frequency bins down to -1, all divided by `n * d`. `d` is the
    sample spacing; pass `d=1/sample_rate` to label `fft` bins in hertz.
    Spacing must be finite and nonzero; negative spacing reverses label signs.
    Labels use Float64 intermediates before conversion to `dtype`. Bins beyond
    the output dtype range become signed infinity (overflow) or signed zero
    (underflow). DC is always exactly positive zero.
    """
    if n <= 0:
        raise Error(
            String(
                "fftfreq bin count n must be a positive integer; got ",
                n,
            )
        )
    if d == Scalar[dtype](0.0):
        raise Error(
            String(
                "fftfreq sample spacing d must be nonzero; got ",
                d,
                "; pass d=1/sample_rate to label bins in hertz",
            )
        )

    if not isfinite(d):
        raise Error(
            String(
                "fftfreq sample spacing d must be finite; got ",
                d,
                "; pass a finite nonzero spacing",
            )
        )

    # Normalize the signed bin before dividing by spacing. Forming n*d can
    # overflow, and multiplying by a precomputed reciprocal can underflow.
    # Float64 intermediates also keep n representable for narrow output dtypes.
    var count = Float64(n)
    var spacing = Float64(d)
    var frequencies = List[Scalar[dtype]](capacity=n)
    var positive_count = (n + 1) // 2
    frequencies.append(Scalar[dtype](0.0))
    for index in range(1, positive_count):
        frequencies.append(Scalar[dtype]((Float64(index) / count) / spacing))
    var negative_count = n // 2
    for index in range(negative_count):
        frequencies.append(
            Scalar[dtype]((Float64(index - negative_count) / count) / spacing)
        )
    return frequencies^


def rfftfreq[
    dtype: DType = DType.float64
](n: Int, d: Scalar[dtype] = 1.0) raises -> List[
    Scalar[dtype]
] where dtype.is_floating_point():
    """Return nonnegative frequency labels in `rfft` bin order.

    The labels run from zero through `n // 2`, each divided by `n * d`, matching
    the DC-first half-spectrum returned by `rfft`. `d` is the sample spacing;
    pass `d=1/sample_rate` to label bins in hertz.
    Spacing must be finite and nonzero; negative spacing reverses label signs.
    Labels use Float64 intermediates before conversion to `dtype`. Bins beyond
    the output dtype range become signed infinity (overflow) or signed zero
    (underflow). DC is always exactly positive zero.
    """
    if n <= 0:
        raise Error(
            String(
                "rfftfreq bin count n must be a positive integer; got ",
                n,
            )
        )
    if d == Scalar[dtype](0.0):
        raise Error(
            String(
                "rfftfreq sample spacing d must be nonzero; got ",
                d,
                "; pass d=1/sample_rate to label bins in hertz",
            )
        )

    if not isfinite(d):
        raise Error(
            String(
                "rfftfreq sample spacing d must be finite; got ",
                d,
                "; pass a finite nonzero spacing",
            )
        )

    # Normalize the signed bin before dividing by spacing. Forming n*d can
    # overflow, and multiplying by a precomputed reciprocal can underflow.
    # Float64 intermediates also keep n representable for narrow output dtypes.
    var count = Float64(n)
    var spacing = Float64(d)
    var frequencies = List[Scalar[dtype]](capacity=n // 2 + 1)
    frequencies.append(Scalar[dtype](0.0))
    for index in range(1, n // 2 + 1):
        frequencies.append(Scalar[dtype]((Float64(index) / count) / spacing))
    return frequencies^
