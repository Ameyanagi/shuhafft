"""Frequency-bin labels matching NumPy transform ordering."""


def fftfreq[
    dtype: DType = DType.float64
](n: Int, d: Scalar[dtype] = 1.0) raises -> List[
    Scalar[dtype]
] where dtype.is_floating_point():
    """Return full-spectrum frequency labels in `fft` bin order.

    The labels are zero through the positive-frequency bins, followed by the
    negative-frequency bins down to -1, all divided by `n * d`. `d` is the
    sample spacing; pass `d=1/sample_rate` to label `fft` bins in hertz.
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

    var scale = Scalar[dtype](1.0) / (Scalar[dtype](n) * d)
    var frequencies = List[Scalar[dtype]](capacity=n)
    var positive_count = (n + 1) // 2
    for index in range(positive_count):
        frequencies.append(Scalar[dtype](index) * scale)
    var negative_count = n // 2
    for index in range(negative_count):
        frequencies.append(Scalar[dtype](index - negative_count) * scale)
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

    var scale = Scalar[dtype](1.0) / (Scalar[dtype](n) * d)
    var frequencies = List[Scalar[dtype]](capacity=n // 2 + 1)
    for index in range(n // 2 + 1):
        frequencies.append(Scalar[dtype](index) * scale)
    return frequencies^
