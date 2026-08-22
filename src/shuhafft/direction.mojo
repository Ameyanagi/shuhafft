"""Transform direction as a nominal value."""


struct FFTDirection(Equatable, TrivialRegisterPassable, Writable):
    """The sign convention used by a complex Fourier transform.

    Use `FORWARD` or `INVERSE` instead of constructing this value from its
    implementation detail.
    """

    var _value: Int

    comptime FORWARD = FFTDirection(_value=0)
    comptime INVERSE = FFTDirection(_value=1)

    def __init__(out self, *, _value: Int):
        self._value = _value

    def is_forward(self) -> Bool:
        """Return whether this is the forward transform."""
        return self._value == 0

    def is_inverse(self) -> Bool:
        """Return whether this is the inverse transform."""
        return self._value == 1

    def validate(self) raises:
        """Reject a discriminant other than the two public constants."""
        if self._value != 0 and self._value != 1:
            raise Error(
                String(
                    "direction discriminant ",
                    self._value,
                    " is invalid; use FFTDirection.FORWARD (0) or ",
                    "FFTDirection.INVERSE (1)",
                )
            )

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not self == other

    def __str__(self) -> String:
        var result = String()
        self.write_to(result)
        return result^

    def write_to[W: Writer](self, mut writer: W):
        writer.write("forward" if self._value == 0 else "inverse")
