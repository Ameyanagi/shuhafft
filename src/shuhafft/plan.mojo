"""Validated FFT plans and public execution contracts."""

from std.complex import ComplexSIMD

from ._radix2 import _radix2_in_place
from .direction import FFTDirection
from .normalization import FFTNormalization


def _is_power_of_two(value: Int) -> Bool:
    return value > 0 and (value & (value - 1)) == 0


struct FFTPlan[dtype: DType](Copyable, Movable) where dtype.is_floating_point():
    """A validated scalar CPU radix-2 complex transform plan.

    The dtype must be `DType.float32` or `DType.float64`. The plan length must
    be a non-zero power of two. A plan can execute repeatedly and forms the
    semantic seam for future optimized backends.
    """

    var _size: Int
    var _direction: FFTDirection
    var _normalization: FFTNormalization

    def __init__(
        out self,
        size: Int,
        direction: FFTDirection,
        normalization: FFTNormalization = FFTNormalization.backward(),
    ) raises:
        comptime assert (
            Self.dtype == DType.float32 or Self.dtype == DType.float64
        ), "FFTPlan supports only float32 and float64 in v0.1"
        if not _is_power_of_two(size):
            raise Error("FFT length must be a non-zero power of two")
        self._size = size
        self._direction = direction
        self._normalization = normalization

    def __init__(out self, *, copy: Self):
        self._size = copy._size
        self._direction = copy._direction
        self._normalization = copy._normalization

    def size(self) -> Int:
        """Return the exact number of complex values accepted by this plan."""
        return self._size

    def direction(self) -> FFTDirection:
        """Return this plan's transform direction."""
        return self._direction

    def normalization(self) -> FFTNormalization:
        """Return this plan's normalization convention."""
        return self._normalization

    def _validate_execution(self, input_length: Int) raises:
        # Mojo 1.0 fields remain externally reachable. Revalidate the stored
        # length at every execution boundary so direct mutation cannot feed
        # unsafe state to the radix-2 kernel.
        if not _is_power_of_two(self._size):
            raise Error("FFT plan length must remain a non-zero power of two")
        if input_length != self._size:
            raise Error("input length does not match FFT plan length")

    def execute(
        self, values: List[ComplexSIMD[Self.dtype, 1]]
    ) raises -> List[ComplexSIMD[Self.dtype, 1]]:
        """Return a transformed deep copy while preserving `values`."""
        # Validate before allocating and copying the out-of-place result.
        self._validate_execution(len(values))
        var output = List[ComplexSIMD[Self.dtype, 1]](copy=values)
        self.execute_in_place(output)
        return output^

    def execute_in_place(self, mut values: List[ComplexSIMD[Self.dtype, 1]]) raises:
        """Transform `values` in place without changing its length."""
        self._validate_execution(len(values))
        _radix2_in_place(values, self._direction)
        var scale = self._normalization.factor[Self.dtype](self._direction, self._size)
        if scale != Scalar[Self.dtype](1.0):
            for index in range(self._size):
                values[index] *= scale
