"""Reusable arbitrary-length complex FFT plans using Bluestein convolution."""

from std.complex import ComplexSIMD
from std.math import cos, sin

from .direction import FFTDirection
from .normalization import FFTNormalization
from .plan import FFTPlan


comptime _MAX_BLUESTEIN_LENGTH = 1 << 20


def _bluestein_workspace_size(size: Int) -> Int:
    """Return the radix-2 convolution length for a validated input length."""
    var required = 2 * size - 1
    var workspace_size = 1
    while workspace_size < required:
        workspace_size *= 2
    return workspace_size


struct BluesteinFFTPlan[dtype: DType](
    Copyable, Equatable, Movable, Writable
) where dtype.is_floating_point():
    """A reusable arbitrary-length complex FFT plan.

    Bluestein's identity converts an `n`-point transform into a linear
    convolution of length `2*n - 1`, evaluated by two plan-owned radix-2 FFTs.
    Construction precomputes the chirp and convolution spectrum. Repeated
    `execute_into` and `execute_in_place` calls reuse one plan-owned workspace.

    Lengths from 1 through 2^20 are supported. The explicit bound prevents a
    surprising multi-gigabyte allocation from a malformed length. This plan is
    most useful for prime and otherwise awkward lengths; `FFTPlan` remains the
    lower-memory, lower-constant path for powers of two. `validate()` checks
    configuration, nested-plan contracts, and table shapes; it deliberately
    does not recompute chirp or convolution-spectrum contents.
    """

    var _size: Int
    var _workspace_size: Int
    var _direction: FFTDirection
    var _normalization: FFTNormalization
    var _chirp: List[ComplexSIMD[Self.dtype, 1]]
    var _kernel_spectrum: List[ComplexSIMD[Self.dtype, 1]]
    var _workspace: List[ComplexSIMD[Self.dtype, 1]]
    var _forward_plan: FFTPlan[Self.dtype]
    var _inverse_plan: FFTPlan[Self.dtype]

    def __init__(
        out self,
        size: Int,
        direction: FFTDirection,
        normalization: FFTNormalization = FFTNormalization.BACKWARD,
    ) raises:
        comptime assert (
            Self.dtype == DType.float32 or Self.dtype == DType.float64
        ), "BluesteinFFTPlan supports only float32 and float64"
        if size <= 0 or size > _MAX_BLUESTEIN_LENGTH:
            raise Error(
                String(
                    "Bluestein FFT length must be in [1, ",
                    _MAX_BLUESTEIN_LENGTH,
                    "]; got ",
                    size,
                )
            )
        direction.validate()
        normalization.validate()

        self._size = size
        self._workspace_size = _bluestein_workspace_size(size)
        self._direction = direction
        self._normalization = normalization
        self._chirp = List[ComplexSIMD[Self.dtype, 1]](capacity=size)
        self._kernel_spectrum = List[ComplexSIMD[Self.dtype, 1]]()
        self._workspace = List[ComplexSIMD[Self.dtype, 1]](
            length=self._workspace_size, fill=ComplexSIMD[Self.dtype, 1](0.0)
        )
        self._forward_plan = FFTPlan[Self.dtype](
            self._workspace_size, FFTDirection.FORWARD, FFTNormalization.NONE
        )
        self._inverse_plan = FFTPlan[Self.dtype](
            self._workspace_size, FFTDirection.INVERSE, FFTNormalization.BACKWARD
        )

        var kernel = List[ComplexSIMD[Self.dtype, 1]](
            length=self._workspace_size, fill=ComplexSIMD[Self.dtype, 1](0.0)
        )
        var sign = Scalar[Self.dtype](1.0 if direction.is_inverse() else -1.0)
        var pi = Scalar[Self.dtype](3.1415926535897932384626433832795)
        var period = 2 * size
        for index in range(size):
            # Reducing k^2 modulo 2n keeps the trigonometric argument bounded.
            var phase_index = (index * index) % period
            var angle = (
                sign * pi * Scalar[Self.dtype](phase_index) / Scalar[Self.dtype](size)
            )
            var chirp = ComplexSIMD[Self.dtype, 1](cos(angle), sin(angle))
            self._chirp.append(chirp)
            var kernel_value = ComplexSIMD[Self.dtype, 1](chirp.re, -chirp.im)
            kernel[index] = kernel_value
            if index != 0:
                kernel[self._workspace_size - index] = kernel_value
        self._kernel_spectrum = self._forward_plan.execute(kernel)

    def __init__(out self, *, copy: Self):
        self._size = copy._size
        self._workspace_size = copy._workspace_size
        self._direction = copy._direction
        self._normalization = copy._normalization
        self._chirp = List[ComplexSIMD[Self.dtype, 1]](copy=copy._chirp)
        self._kernel_spectrum = List[ComplexSIMD[Self.dtype, 1]](
            copy=copy._kernel_spectrum
        )
        self._workspace = List[ComplexSIMD[Self.dtype, 1]](copy=copy._workspace)
        self._forward_plan = FFTPlan[Self.dtype](copy=copy._forward_plan)
        self._inverse_plan = FFTPlan[Self.dtype](copy=copy._inverse_plan)

    def size(self) -> Int:
        """Return the exact number of complex values accepted by this plan."""
        return self._size

    def workspace_size(self) -> Int:
        """Return the internal radix-2 convolution length."""
        return self._workspace_size

    def direction(self) -> FFTDirection:
        """Return this plan's transform direction."""
        return self._direction

    def normalization(self) -> FFTNormalization:
        """Return this plan's normalization convention."""
        return self._normalization

    def make_buffer(self) -> List[ComplexSIMD[Self.dtype, 1]]:
        """Return zero-filled storage sized for this plan's public result."""
        return List[ComplexSIMD[Self.dtype, 1]](
            length=self._size, fill=ComplexSIMD[Self.dtype, 1](0.0)
        )

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
            "BluesteinFFTPlan(size=",
            self._size,
            ", workspace_size=",
            self._workspace_size,
            ", ",
            self._direction,
            ", ",
            self._normalization,
            ")",
        )

    def validate(self) raises:
        """Validate table shapes and nested plans, not numerical table contents."""
        if self._size <= 0 or self._size > _MAX_BLUESTEIN_LENGTH:
            raise Error(
                String(
                    "Bluestein FFT plan length must remain in [1, ",
                    _MAX_BLUESTEIN_LENGTH,
                    "]; got ",
                    self._size,
                )
            )
        self._direction.validate()
        self._normalization.validate()
        var expected_workspace_size = _bluestein_workspace_size(self._size)
        if (
            self._workspace_size != expected_workspace_size
            or len(self._chirp) != self._size
            or len(self._kernel_spectrum) != self._workspace_size
            or len(self._workspace) != self._workspace_size
        ):
            raise Error(
                String(
                    "Bluestein plan tables must match length ",
                    self._size,
                    "; expected workspace_size ",
                    expected_workspace_size,
                    ", chirp length ",
                    self._size,
                    ", and workspace tables of that size; got workspace_size ",
                    self._workspace_size,
                    ", chirp length ",
                    len(self._chirp),
                    ", kernel spectrum length ",
                    len(self._kernel_spectrum),
                    ", and workspace length ",
                    len(self._workspace),
                )
            )
        self._forward_plan.validate()
        self._inverse_plan.validate()
        if (
            self._forward_plan.size() != self._workspace_size
            or self._forward_plan.direction() != FFTDirection.FORWARD
            or self._forward_plan.normalization() != FFTNormalization.NONE
            or self._inverse_plan.size() != self._workspace_size
            or self._inverse_plan.direction() != FFTDirection.INVERSE
            or self._inverse_plan.normalization() != FFTNormalization.BACKWARD
        ):
            raise Error(
                "Bluestein nested radix-2 plans do not match the convolution contract"
            )

    def _validate_input_length(self, input_length: Int) raises:
        if input_length != self._size:
            raise Error(
                String(
                    "input length ",
                    input_length,
                    " does not match Bluestein FFT plan length ",
                    self._size,
                )
            )

    def _validate_output_length(self, output_length: Int) raises:
        if output_length != self._size:
            raise Error(
                String(
                    "output length ",
                    output_length,
                    " does not match Bluestein FFT plan length ",
                    self._size,
                )
            )

    def _load_input(mut self, values: Span[ComplexSIMD[Self.dtype, 1], _]):
        for index in range(self._size):
            self._workspace[index] = values[index] * self._chirp[index]
        for index in range(self._size, self._workspace_size):
            self._workspace[index] = ComplexSIMD[Self.dtype, 1](0.0)

    def _transform_workspace(mut self) raises:
        self._forward_plan.execute_in_place(self._workspace)
        for index in range(self._workspace_size):
            self._workspace[index] *= self._kernel_spectrum[index]
        self._inverse_plan.execute_in_place(self._workspace)

    def _write_result(self, mut output: List[ComplexSIMD[Self.dtype, 1]]):
        var scale = self._normalization.factor[Self.dtype](self._direction, self._size)
        for index in range(self._size):
            output[index] = self._workspace[index] * self._chirp[index] * scale

    def execute_into(
        mut self,
        values: Span[ComplexSIMD[Self.dtype, 1], _],
        mut output: List[ComplexSIMD[Self.dtype, 1]],
    ) raises:
        """Transform into caller-owned output while reusing plan workspace."""
        self._validate_input_length(len(values))
        self._validate_output_length(len(output))
        self._load_input(values)
        self._transform_workspace()
        self._write_result(output)

    def execute(
        mut self, values: Span[ComplexSIMD[Self.dtype, 1], _]
    ) raises -> List[ComplexSIMD[Self.dtype, 1]]:
        """Return a transformed copy while reusing plan convolution storage."""
        self._validate_input_length(len(values))
        var output = self.make_buffer()
        self.execute_into(values, output)
        return output^

    def execute_in_place(mut self, mut values: List[ComplexSIMD[Self.dtype, 1]]) raises:
        """Transform the exact-length input and output buffer in place."""
        self._validate_input_length(len(values))
        self._load_input(values)
        self._transform_workspace()
        self._write_result(values)
