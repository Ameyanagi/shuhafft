"""Validated FFT plans and public execution contracts."""

from std.complex import ComplexSIMD
from std.math import cos, sin

from ._radix2 import _radix2_in_place
from .direction import FFTDirection
from .normalization import FFTNormalization


def _is_power_of_two(value: Int) -> Bool:
    return value > 0 and (value & (value - 1)) == 0


def _lower_bounding_power_of_two(value: Int) -> Int:
    """Return the greatest power of two below a positive non-power-of-two value."""
    var lower = 1
    while lower * 2 < value:
        lower *= 2
    return lower


struct FFTPlan[dtype: DType](
    Copyable, Equatable, Movable, Writable
) where dtype.is_floating_point():
    """A validated scalar CPU radix-2 complex transform plan.

    The dtype must be `DType.float32` or `DType.float64`. The plan length must
    be a non-zero power of two. A plan can execute repeatedly and forms the
    semantic seam for future optimized backends. Direct mutation of
    underscore-prefixed fields is out of contract; call `validate()` for an
    explicit invariant checkpoint after unusual operations.
    """

    var _size: Int
    var _direction: FFTDirection
    var _normalization: FFTNormalization
    # Stage half-width `half` starts at flat index `half - 1`, with `half`
    # contiguous entries. Thus stages cover [0, size - 1) without gaps.
    var _twiddle_re: List[Scalar[Self.dtype]]
    var _twiddle_im: List[Scalar[Self.dtype]]
    var _bit_reversal: List[Int]

    def __init__(
        out self,
        size: Int,
        direction: FFTDirection,
        normalization: FFTNormalization = FFTNormalization.BACKWARD,
    ) raises:
        comptime assert (
            Self.dtype == DType.float32 or Self.dtype == DType.float64
        ), "FFTPlan supports only float32 and float64 in v0.1"
        if size <= 0:
            raise Error(
                String(
                    "FFT length must be a non-zero power of two; got ",
                    size,
                    "; the smallest valid length is 1",
                )
            )
        if not _is_power_of_two(size):
            var lower = _lower_bounding_power_of_two(size)
            var higher = lower * 2
            raise Error(
                String(
                    "FFT length must be a non-zero power of two; got ",
                    size,
                    " (nearest are ",
                    lower,
                    " and ",
                    higher,
                    ")",
                )
            )
        self._size = size
        self._direction = direction
        self._normalization = normalization
        self._twiddle_re = List[Scalar[Self.dtype]](capacity=size - 1)
        self._twiddle_im = List[Scalar[Self.dtype]](capacity=size - 1)
        self._bit_reversal = List[Int](length=size, fill=0)

        var sign = Scalar[Self.dtype](1.0 if direction.is_inverse() else -1.0)
        var two_pi = Scalar[Self.dtype](6.283185307179586476925286766559)
        var stage_size = 2
        while stage_size <= size:
            var half = stage_size // 2
            for offset in range(half):
                var angle = (
                    sign
                    * two_pi
                    * Scalar[Self.dtype](offset)
                    / Scalar[Self.dtype](stage_size)
                )
                self._twiddle_re.append(cos(angle))
                self._twiddle_im.append(sin(angle))
            stage_size *= 2

        var reversed_index = 0
        for index in range(1, size):
            var bit = size // 2
            while reversed_index & bit:
                reversed_index ^= bit
                bit //= 2
            reversed_index ^= bit
            self._bit_reversal[index] = reversed_index

    def __init__(out self, *, copy: Self):
        self._size = copy._size
        self._direction = copy._direction
        self._normalization = copy._normalization
        self._twiddle_re = List[Scalar[Self.dtype]](copy=copy._twiddle_re)
        self._twiddle_im = List[Scalar[Self.dtype]](copy=copy._twiddle_im)
        self._bit_reversal = List[Int](copy=copy._bit_reversal)

    def size(self) -> Int:
        """Return the exact number of complex values accepted by this plan."""
        return self._size

    def direction(self) -> FFTDirection:
        """Return this plan's transform direction."""
        return self._direction

    def normalization(self) -> FFTNormalization:
        """Return this plan's normalization convention."""
        return self._normalization

    def __eq__(self, other: Self) -> Bool:
        return (
            self._size == other._size
            and self._direction == other._direction
            and self._normalization == other._normalization
        )

    def __ne__(self, other: Self) -> Bool:
        return not self == other

    def __str__(self) -> String:
        var result = String()
        self.write_to(result)
        return result^

    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            "FFTPlan(size=",
            self._size,
            ", ",
            self._direction,
            ", ",
            self._normalization,
            ")",
        )

    def validate(self) raises:
        """Validate the stored plan invariants explicitly."""
        if not _is_power_of_two(self._size):
            raise Error(
                String(
                    "FFT plan length must remain a non-zero power of two; got ",
                    self._size,
                )
            )
        if (
            len(self._twiddle_re) != self._size - 1
            or len(self._twiddle_im) != self._size - 1
            or len(self._bit_reversal) != self._size
        ):
            raise Error(
                String(
                    "FFT plan tables must match the plan length; plan length ",
                    self._size,
                    " expects twiddle_re length ",
                    self._size - 1,
                    ", twiddle_im length ",
                    self._size - 1,
                    ", and bit_reversal length ",
                    self._size,
                    "; got twiddle_re length ",
                    len(self._twiddle_re),
                    ", twiddle_im length ",
                    len(self._twiddle_im),
                    ", and bit_reversal length ",
                    len(self._bit_reversal),
                )
            )

    def _validate_input_length(self, input_length: Int) raises:
        if input_length != self._size:
            raise Error(
                String(
                    "input length ",
                    input_length,
                    " does not match FFT plan length ",
                    self._size,
                )
            )

    def execute(
        self, values: Span[ComplexSIMD[Self.dtype, 1], _]
    ) raises -> List[ComplexSIMD[Self.dtype, 1]]:
        """Return a transformed deep copy while preserving `values`."""
        # Validate before allocating and copying the out-of-place result.
        self._validate_input_length(len(values))
        var output = List[ComplexSIMD[Self.dtype, 1]](capacity=len(values))
        output.extend(values)
        self.execute_in_place(output)
        return output^

    def execute_in_place(self, mut values: List[ComplexSIMD[Self.dtype, 1]]) raises:
        """Transform `values` in place without changing its length."""
        self._validate_input_length(len(values))
        _radix2_in_place(values, self._twiddle_re, self._twiddle_im, self._bit_reversal)
        var scale = self._normalization.factor[Self.dtype](self._direction, self._size)
        if scale != Scalar[Self.dtype](1.0):
            for index in range(self._size):
                values[index] *= scale
