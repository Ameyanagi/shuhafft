"""Transform direction as a nominal value."""


struct FFTDirection(Equatable, TrivialRegisterPassable):
    """The sign convention used by a complex Fourier transform.

    Use `forward()` or `inverse()` instead of constructing this value from an
    implementation detail.
    """

    var _is_inverse: Bool

    def __init__(out self, *, _is_inverse: Bool):
        self._is_inverse = _is_inverse

    @staticmethod
    def forward() -> Self:
        """Return the negative-exponent transform direction."""
        return Self(_is_inverse=False)

    @staticmethod
    def inverse() -> Self:
        """Return the positive-exponent transform direction."""
        return Self(_is_inverse=True)

    def is_forward(self) -> Bool:
        """Return whether this is the forward transform."""
        return not self._is_inverse

    def is_inverse(self) -> Bool:
        """Return whether this is the inverse transform."""
        return self._is_inverse

    def __eq__(self, other: Self) -> Bool:
        return self._is_inverse == other._is_inverse

    def __ne__(self, other: Self) -> Bool:
        return not self == other
