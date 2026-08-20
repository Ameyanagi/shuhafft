"""Internal scalar radix-2 implementation."""

from std.complex import ComplexSIMD


def _radix2_in_place[
    dtype: DType
](
    mut values: List[ComplexSIMD[dtype, 1]],
    twiddle_re: List[Scalar[dtype]],
    twiddle_im: List[Scalar[dtype]],
    bit_reversal: List[Int],
) where dtype.is_floating_point():
    """Apply an unnormalized radix-2 DIT transform with plan-owned tables."""
    var size = len(values)
    debug_assert(len(twiddle_re) == size - 1, "invalid twiddle real table")
    debug_assert(len(twiddle_im) == size - 1, "invalid twiddle imaginary table")
    debug_assert(len(bit_reversal) == size, "invalid bit-reversal table")

    # Place inputs in the plan's precomputed bit-reversed order.
    for index in range(1, size):
        if index < bit_reversal[index]:
            values.swap_elements(index, bit_reversal[index])

    var stage_size = 2
    while stage_size <= size:
        var half = stage_size // 2
        var stage_twiddle_start = half - 1

        for block_start in range(0, size, stage_size):
            for offset in range(half):
                var even = values[block_start + offset]
                var twiddle_index = stage_twiddle_start + offset
                var twiddle = ComplexSIMD[dtype, 1](
                    twiddle_re[twiddle_index], twiddle_im[twiddle_index]
                )
                var odd = values[block_start + offset + half] * twiddle
                values[block_start + offset] = even + odd
                values[block_start + offset + half] = even - odd
        stage_size *= 2
