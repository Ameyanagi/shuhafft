"""Internal scalar radix-2 implementation."""

from std.complex import ComplexSIMD
from std.math import cos, sin

from .direction import FFTDirection


def _radix2_in_place[
    dtype: DType
](
    mut values: List[ComplexSIMD[dtype, 1]], direction: FFTDirection
) where dtype.is_floating_point():
    """Apply an unnormalized radix-2 DIT transform to validated data."""
    var size = len(values)

    # Place inputs in bit-reversed order. This recurrence avoids materializing
    # an index table while remaining deterministic and allocation-free.
    var reversed_index = 0
    for index in range(1, size):
        var bit = size // 2
        while reversed_index & bit:
            reversed_index ^= bit
            bit //= 2
        reversed_index ^= bit
        if index < reversed_index:
            values.swap_elements(index, reversed_index)

    var stage_size = 2
    while stage_size <= size:
        var sign = Scalar[dtype](1.0 if direction.is_inverse() else -1.0)
        var angle = (
            sign
            * Scalar[dtype](6.283185307179586476925286766559)
            / Scalar[dtype](stage_size)
        )
        var stage_twiddle = ComplexSIMD[dtype, 1](cos(angle), sin(angle))
        var half = stage_size // 2

        for block_start in range(0, size, stage_size):
            var twiddle = ComplexSIMD[dtype, 1](Scalar[dtype](1.0))
            for offset in range(half):
                var even = values[block_start + offset]
                var odd = values[block_start + offset + half] * twiddle
                values[block_start + offset] = even + odd
                values[block_start + offset + half] = even - odd
                twiddle *= stage_twiddle
        stage_size *= 2
