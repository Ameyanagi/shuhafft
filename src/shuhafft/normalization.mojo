"""Normalization conventions for Fourier transforms."""

from std.math import sqrt

from .direction import FFTDirection


struct FFTNormalization(Equatable, TrivialRegisterPassable, Writable):
    """A nominal FFT scaling convention.

    `BACKWARD` scales only inverse transforms by `1 / n`. `FORWARD` scales only
    forward transforms by `1 / n`. `ORTHO` scales both directions by
    `1 / sqrt(n)`. `NONE` leaves both directions unscaled. Direct mutation of
    `_value` is out of contract.
    """

    var _value: Int

    comptime NONE = FFTNormalization(_value=0)
    comptime BACKWARD = FFTNormalization(_value=1)
    comptime FORWARD = FFTNormalization(_value=2)
    comptime ORTHO = FFTNormalization(_value=3)

    def __init__(out self, *, _value: Int):
        self._value = _value

    def factor[
        dtype: DType
    ](self, direction: FFTDirection, size: Int) -> Scalar[
        dtype
    ] where dtype.is_floating_point():
        """Return the multiplicative scale for a validated transform length."""
        debug_assert(size > 0, "normalization length must be positive")
        var one = Scalar[dtype](1.0)
        var n = Scalar[dtype](size)
        if self._value == 3:
            return one / sqrt(n)
        if self._value == 1 and direction.is_inverse():
            return one / n
        if self._value == 2 and direction.is_forward():
            return one / n
        return one

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not self == other

    def __str__(self) -> String:
        var result = String()
        self.write_to(result)
        return result^

    def write_to[W: Writer](self, mut writer: W):
        if self._value == 0:
            writer.write("none")
        elif self._value == 1:
            writer.write("backward")
        elif self._value == 2:
            writer.write("forward")
        else:
            writer.write("ortho")
