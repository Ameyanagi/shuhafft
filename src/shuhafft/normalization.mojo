"""Normalization conventions for Fourier transforms."""

from std.math import sqrt

from .direction import FFTDirection


struct FFTNormalization(Equatable, TrivialRegisterPassable):
    """A nominal FFT scaling convention.

    `backward()` scales only inverse transforms by `1 / n`. `forward()` scales
    only forward transforms by `1 / n`. `ortho()` scales both directions by
    `1 / sqrt(n)`. `none()` leaves both directions unscaled.
    """

    # Mojo 1.0 fields remain externally reachable. Two booleans deliberately
    # make all constructible and mutated representations meaningful: none,
    # backward, forward, or ortho. There is no invalid discriminant.
    var _scale_forward: Bool
    var _scale_inverse: Bool

    def __init__(out self, *, _scale_forward: Bool, _scale_inverse: Bool):
        self._scale_forward = _scale_forward
        self._scale_inverse = _scale_inverse

    @staticmethod
    def none() -> Self:
        """Return the convention that applies no scaling."""
        return Self(_scale_forward=False, _scale_inverse=False)

    @staticmethod
    def backward() -> Self:
        """Return the convention that scales inverse transforms by `1 / n`."""
        return Self(_scale_forward=False, _scale_inverse=True)

    @staticmethod
    def forward() -> Self:
        """Return the convention that scales forward transforms by `1 / n`."""
        return Self(_scale_forward=True, _scale_inverse=False)

    @staticmethod
    def ortho() -> Self:
        """Return the convention that scales both directions by `1 / sqrt(n)`."""
        return Self(_scale_forward=True, _scale_inverse=True)

    def factor[
        dtype: DType
    ](self, direction: FFTDirection, size: Int) -> Scalar[
        dtype
    ] where dtype.is_floating_point():
        """Return the multiplicative scale for a validated transform length."""
        debug_assert(size > 0, "normalization length must be positive")
        var one = Scalar[dtype](1.0)
        var n = Scalar[dtype](size)
        if self._scale_forward and self._scale_inverse:
            return one / sqrt(n)
        if self._scale_inverse and direction.is_inverse():
            return one / n
        if self._scale_forward and direction.is_forward():
            return one / n
        return one

    def __eq__(self, other: Self) -> Bool:
        return (
            self._scale_forward == other._scale_forward
            and self._scale_inverse == other._scale_inverse
        )

    def __ne__(self, other: Self) -> Bool:
        return not self == other
