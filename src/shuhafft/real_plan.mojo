"""Validated real-input FFT plans with compact Hermitian spectra."""

from std.complex import ComplexSIMD
from std.math import cos, sin

from ._radix2 import _radix2_prefix_in_place
from .direction import FFTDirection
from .normalization import FFTNormalization
from .plan import FFTPlan, _is_power_of_two


struct RealFFTPlan[dtype: DType](
    Copyable, Equatable, Movable, Writable
) where dtype.is_floating_point():
    """A validated scalar CPU radix-2 real transform plan.

    The dtype must be `DType.float32` or `DType.float64`, and the real sample
    count must be a power of two at least 2. Forward output contains exactly
    `n // 2 + 1` bins in DC-first ascending-frequency order: bin `k`
    represents `k * (sample_rate / n)`, through the Nyquist bin at `n // 2`.
    Direct mutation of underscore-prefixed fields is out of contract; call
    `validate()` for an explicit invariant checkpoint after unusual operations.
    """

    var _size: Int
    var _normalization: FFTNormalization
    var _forward_plan: FFTPlan[Self.dtype]
    var _inverse_plan: FFTPlan[Self.dtype]
    var _recombination_twiddle_re: List[Scalar[Self.dtype]]
    var _recombination_twiddle_im: List[Scalar[Self.dtype]]

    def __init__(
        out self,
        size: Int,
        normalization: FFTNormalization = FFTNormalization.BACKWARD,
    ) raises:
        comptime assert (
            Self.dtype == DType.float32 or Self.dtype == DType.float64
        ), "RealFFTPlan supports only float32 and float64"
        if size < 2:
            raise Error(
                String(
                    "real FFT length must be a power of two >= 2; got ",
                    size,
                    "; the smallest valid length is 2",
                )
            )
        if not _is_power_of_two(size):
            var lower = 1
            while lower * 2 < size:
                lower *= 2
            var higher = lower * 2
            raise Error(
                String(
                    "real FFT length must be a power of two >= 2; got ",
                    size,
                    " (nearest are ",
                    lower,
                    " and ",
                    higher,
                    ")",
                )
            )

        self._size = size
        self._normalization = normalization
        var half_size = size // 2
        self._forward_plan = FFTPlan[Self.dtype](
            half_size, FFTDirection.FORWARD, FFTNormalization.NONE
        )
        self._inverse_plan = FFTPlan[Self.dtype](
            half_size, FFTDirection.INVERSE, FFTNormalization.NONE
        )
        self._recombination_twiddle_re = List[Scalar[Self.dtype]](
            capacity=half_size + 1
        )
        self._recombination_twiddle_im = List[Scalar[Self.dtype]](
            capacity=half_size + 1
        )
        var pi = Scalar[Self.dtype](3.1415926535897932384626433832795)
        for index in range(half_size + 1):
            var angle = -pi * Scalar[Self.dtype](index) / Scalar[Self.dtype](half_size)
            self._recombination_twiddle_re.append(cos(angle))
            self._recombination_twiddle_im.append(sin(angle))

    def __init__(out self, *, copy: Self):
        self._size = copy._size
        self._normalization = copy._normalization
        self._forward_plan = FFTPlan[Self.dtype](copy=copy._forward_plan)
        self._inverse_plan = FFTPlan[Self.dtype](copy=copy._inverse_plan)
        self._recombination_twiddle_re = List[Scalar[Self.dtype]](
            copy=copy._recombination_twiddle_re
        )
        self._recombination_twiddle_im = List[Scalar[Self.dtype]](
            copy=copy._recombination_twiddle_im
        )

    def size(self) -> Int:
        """Return the number of real samples accepted and produced."""
        return self._size

    def spectrum_size(self) -> Int:
        """Return the compact DC-through-Nyquist spectrum length."""
        return self._size // 2 + 1

    def normalization(self) -> FFTNormalization:
        """Return this plan's normalization convention."""
        return self._normalization

    def __eq__(self, other: Self) -> Bool:
        return self._size == other._size and self._normalization == other._normalization

    def __ne__(self, other: Self) -> Bool:
        return not self == other

    def __str__(self) -> String:
        var result = String()
        self.write_to(result)
        return result^

    def write_to[W: Writer](self, mut writer: W):
        writer.write("RealFFTPlan(size=", self._size, ", ", self._normalization, ")")

    def validate(self) raises:
        """Validate the stored plan invariants explicitly."""
        if self._size < 2 or not _is_power_of_two(self._size):
            raise Error("real FFT plan length must remain a power of two >= 2")
        var half_size = self._size // 2
        if (
            self._forward_plan.size() != half_size
            or self._forward_plan.direction() != FFTDirection.FORWARD
            or self._forward_plan.normalization() != FFTNormalization.NONE
            or self._inverse_plan.size() != half_size
            or self._inverse_plan.direction() != FFTDirection.INVERSE
            or self._inverse_plan.normalization() != FFTNormalization.NONE
        ):
            raise Error("real FFT internal plans must match the real plan length")
        self._forward_plan.validate()
        self._inverse_plan.validate()
        if (
            len(self._recombination_twiddle_re) != half_size + 1
            or len(self._recombination_twiddle_im) != half_size + 1
        ):
            raise Error("real FFT recombination tables must match the plan length")

    def _validate_signal_length(self, signal_length: Int) raises:
        if signal_length != self._size:
            raise Error(
                String(
                    "signal length ",
                    signal_length,
                    " does not match real FFT plan size ",
                    self._size,
                )
            )

    def _validate_spectrum_length(self, spectrum_length: Int) raises:
        var expected = self.spectrum_size()
        if spectrum_length != expected:
            raise Error(
                String(
                    "spectrum length ",
                    spectrum_length,
                    " does not match plan spectrum size ",
                    expected,
                    " (= ",
                    self._size,
                    " // 2 + 1)",
                )
            )

    def _forward_recombined_bin(
        self,
        packed: List[ComplexSIMD[Self.dtype, 1]],
        index: Int,
    ) -> ComplexSIMD[Self.dtype, 1]:
        var half_size = self._size // 2
        var value = packed[index]
        var paired = packed[half_size - index]
        var mirrored = ComplexSIMD[Self.dtype, 1](paired.re, -paired.im)
        # X[k] = ((Z[k] + conj(Z[M-k]))
        #          - i W[k] (Z[k] - conj(Z[M-k]))) / 2.
        var minus_i_twiddle = ComplexSIMD[Self.dtype, 1](
            self._recombination_twiddle_im[index],
            -self._recombination_twiddle_re[index],
        )
        return ((value + mirrored) + (value - mirrored) * minus_i_twiddle) * Scalar[
            Self.dtype
        ](0.5)

    def forward(
        self, signal: Span[Scalar[Self.dtype], _]
    ) raises -> List[ComplexSIMD[Self.dtype, 1]]:
        """Return the compact real-input spectrum in DC-first bin order.

        The result has exactly `n // 2 + 1` ascending-frequency bins, where
        bin `k` represents `k * (sample_rate / n)`. The DC and Nyquist bins
        always have exactly-zero imaginary parts.
        """
        self._validate_signal_length(len(signal))
        var spectrum = List[ComplexSIMD[Self.dtype, 1]](
            length=self.spectrum_size(),
            fill=ComplexSIMD[Self.dtype, 1](0.0),
        )
        self.forward_into(signal, spectrum)
        return spectrum^

    def forward_into(
        self,
        signal: Span[Scalar[Self.dtype], _],
        mut spectrum: List[ComplexSIMD[Self.dtype, 1]],
    ) raises:
        """Write a compact DC-first spectrum using the output as workspace.

        `signal` must contain `n` samples and `spectrum` must contain exactly
        `n // 2 + 1` bins. No scratch storage is allocated. The imaginary parts
        of DC and Nyquist are assigned the literal zero after recombination.
        """
        self._validate_signal_length(len(signal))
        self._validate_spectrum_length(len(spectrum))
        var half_size = self._size // 2
        var scale = self._normalization.factor[Self.dtype](
            FFTDirection.FORWARD, self._size
        )

        if self._size == 2:
            spectrum[0] = ComplexSIMD[Self.dtype, 1](
                (signal[0] + signal[1]) * scale, 0.0
            )
            spectrum[1] = ComplexSIMD[Self.dtype, 1](
                (signal[0] - signal[1]) * scale, 0.0
            )
            return

        for index in range(half_size):
            spectrum[index] = ComplexSIMD[Self.dtype, 1](
                signal[2 * index], signal[2 * index + 1]
            )
        _radix2_prefix_in_place(
            spectrum,
            half_size,
            self._forward_plan._twiddle_re,
            self._forward_plan._twiddle_im,
            self._forward_plan._bit_reversal,
        )

        var packed_zero = spectrum[0]
        var index = 1
        while index < half_size - index:
            var low_bin = self._forward_recombined_bin(spectrum, index)
            var high_bin = self._forward_recombined_bin(spectrum, half_size - index)
            spectrum[index] = low_bin * scale
            spectrum[half_size - index] = high_bin * scale
            index += 1

        var middle = half_size // 2
        spectrum[middle] = self._forward_recombined_bin(spectrum, middle) * scale
        spectrum[0] = ComplexSIMD[Self.dtype, 1](
            (packed_zero.re + packed_zero.im) * scale, 0.0
        )
        spectrum[half_size] = ComplexSIMD[Self.dtype, 1](
            (packed_zero.re - packed_zero.im) * scale, 0.0
        )

    def inverse(
        self, spectrum: Span[ComplexSIMD[Self.dtype, 1], _]
    ) raises -> List[Scalar[Self.dtype]]:
        """Return `n` real samples reconstructed from a compact spectrum.

        Input bins are interpreted in DC-first ascending-frequency order and
        must number exactly `n // 2 + 1`. Any imaginary components supplied at
        DC or Nyquist are ignored, matching `scipy.fft.irfft` behavior.
        """
        self._validate_spectrum_length(len(spectrum))
        var signal = List[Scalar[Self.dtype]](
            length=self._size, fill=Scalar[Self.dtype](0.0)
        )
        self.inverse_into(spectrum, signal)
        return signal^

    def inverse_into(
        self,
        spectrum: Span[ComplexSIMD[Self.dtype, 1], _],
        mut signal: List[Scalar[Self.dtype]],
    ) raises:
        """Write `n` real samples reconstructed from a compact spectrum.

        `spectrum` must contain exactly `n // 2 + 1` DC-first bins and `signal`
        must contain exactly `n` samples. Imaginary components at DC and
        Nyquist are deliberately ignored. This Mojo 1.0 implementation uses
        one local `n // 2` complex workspace because scalar `List` storage
        cannot be safely reinterpreted as complex values.
        """
        self._validate_spectrum_length(len(spectrum))
        self._validate_signal_length(len(signal))
        var half_size = self._size // 2
        var packed = List[ComplexSIMD[Self.dtype, 1]](
            length=half_size, fill=ComplexSIMD[Self.dtype, 1](0.0)
        )
        var half = Scalar[Self.dtype](0.5)
        packed[0] = ComplexSIMD[Self.dtype, 1](
            (spectrum[0].re + spectrum[half_size].re) * half,
            (spectrum[0].re - spectrum[half_size].re) * half,
        )
        for index in range(1, half_size):
            var value = spectrum[index]
            var paired = spectrum[half_size - index]
            var mirrored = ComplexSIMD[Self.dtype, 1](paired.re, -paired.im)
            # Z[k] = ((X[k] + conj(X[M-k]))
            #          + i conj(W[k]) (X[k] - conj(X[M-k]))) / 2.
            var i_conjugate_twiddle = ComplexSIMD[Self.dtype, 1](
                self._recombination_twiddle_im[index],
                self._recombination_twiddle_re[index],
            )
            packed[index] = (
                (value + mirrored) + (value - mirrored) * i_conjugate_twiddle
            ) * half

        self._inverse_plan.execute_in_place(packed)
        # A length-M inverse produces M*x. Multiply by 2 so normalization has
        # the same unnormalized n*x basis as a length-n inverse complex DFT.
        var scale = Scalar[Self.dtype](2.0) * self._normalization.factor[Self.dtype](
            FFTDirection.INVERSE, self._size
        )
        for index in range(half_size):
            signal[2 * index] = packed[index].re * scale
            signal[2 * index + 1] = packed[index].im * scale
