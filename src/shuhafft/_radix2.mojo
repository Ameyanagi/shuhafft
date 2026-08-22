"""Internal scalar and SIMD radix-2 implementations."""

from std.complex import ComplexSIMD
from std.sys import simd_width_of


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


def _radix2_prefix_in_place[
    dtype: DType
](
    mut values: List[ComplexSIMD[dtype, 1]],
    size: Int,
    twiddles: List[ComplexSIMD[dtype, 1]],
    bit_reversal: List[Int],
) where dtype.is_floating_point():
    """Transform the first `size` values using matching radix-2 tables."""
    debug_assert(len(values) >= size, "radix-2 workspace is too short")
    debug_assert(len(twiddles) == size - 1, "invalid complex twiddle table")
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
                var odd = (
                    values[block_start + offset + half]
                    * twiddles[stage_twiddle_start + offset]
                )
                values[block_start + offset] = even + odd
                values[block_start + offset + half] = even - odd
        stage_size *= 2


def _swap_adjacent_lanes[
    dtype: DType, width: Int
](value: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Swap real/imaginary lanes in each native-width complex pair."""
    comptime if width == 2:
        return value.shuffle[1, 0]()
    elif width == 4:
        return value.shuffle[1, 0, 3, 2]()
    elif width == 8:
        return value.shuffle[1, 0, 3, 2, 5, 4, 7, 6]()
    elif width == 16:
        return value.shuffle[1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]()
    else:
        comptime assert False, "FFT interleaved SIMD supports widths 2, 4, 8, and 16"


def _duplicate_twiddle_real_lanes[
    dtype: DType, width: Int
](value: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Duplicate each interleaved twiddle real lane into its complex pair."""
    comptime if width == 2:
        return value.shuffle[0, 0]()
    elif width == 4:
        return value.shuffle[0, 0, 2, 2]()
    elif width == 8:
        return value.shuffle[0, 0, 2, 2, 4, 4, 6, 6]()
    elif width == 16:
        return value.shuffle[0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14]()
    else:
        comptime assert False, "FFT interleaved SIMD supports widths 2, 4, 8, and 16"


def _signed_twiddle_imaginary_lanes[
    dtype: DType, width: Int
](value: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Return [-wi, wi] for every interleaved complex twiddle pair."""
    comptime if width == 2:
        return (-value).shuffle[1, 3](value)
    elif width == 4:
        return (-value).shuffle[1, 5, 3, 7](value)
    elif width == 8:
        return (-value).shuffle[1, 9, 3, 11, 5, 13, 7, 15](value)
    elif width == 16:
        return (-value).shuffle[
            1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31
        ](value)
    else:
        comptime assert False, "FFT interleaved SIMD supports widths 2, 4, 8, and 16"


def _radix2_interleaved_in_place[
    dtype: DType
](
    mut values: List[Scalar[dtype]],
    size: Int,
    twiddle_interleaved: List[Scalar[dtype]],
    bit_reversal: List[Int],
) where dtype.is_floating_point():
    """Transform interleaved scalar pairs with native-width butterflies."""
    debug_assert(len(values) >= 2 * size, "interleaved radix-2 workspace is too short")
    debug_assert(
        len(twiddle_interleaved) == 2 * (size - 1),
        "invalid interleaved twiddle table",
    )
    debug_assert(len(bit_reversal) == size, "invalid bit-reversal table")

    for index in range(1, size):
        var reversed_index = bit_reversal[index]
        if index < reversed_index:
            values.swap_elements(2 * index, 2 * reversed_index)
            values.swap_elements(2 * index + 1, 2 * reversed_index + 1)

    # Safety: the assertions establish live 2*size scalar values, 2*(size-1)
    # scalar twiddles, and `size` reversal entries. Each vector iteration starts
    # only when a complete `complex_width` group remains inside its butterfly
    # half. Every load/store therefore stays within typed contiguous storage;
    # no ComplexSIMD storage is reinterpreted.
    var values_ptr = values.unsafe_ptr()
    var twiddle_ptr = twiddle_interleaved.unsafe_ptr()
    comptime width = simd_width_of[dtype]()
    comptime complex_width = width // 2
    var stage_size = 2
    while stage_size <= size:
        var half = stage_size // 2
        var stage_twiddle_start = half - 1
        for block_start in range(0, size, stage_size):
            var offset = 0
            while offset + complex_width <= half:
                var even_index = 2 * (block_start + offset)
                var odd_index = even_index + 2 * half
                var twiddle_index = 2 * (stage_twiddle_start + offset)
                var even = values_ptr.unsafe_load[width=width](even_index)
                var odd = values_ptr.unsafe_load[width=width](odd_index)
                var twiddle = twiddle_ptr.unsafe_load[width=width](twiddle_index)
                var weight_re = _duplicate_twiddle_real_lanes(twiddle)
                var signed_weight_im = _signed_twiddle_imaginary_lanes(twiddle)
                var weighted = odd * weight_re + (
                    _swap_adjacent_lanes(odd) * signed_weight_im
                )
                values_ptr.unsafe_store[width=width](even_index, even + weighted)
                values_ptr.unsafe_store[width=width](odd_index, even - weighted)
                offset += complex_width

            while offset < half:
                var even_index = 2 * (block_start + offset)
                var odd_index = even_index + 2 * half
                var twiddle_index = 2 * (stage_twiddle_start + offset)
                var even_re = values[even_index]
                var even_im = values[even_index + 1]
                var odd_re = values[odd_index]
                var odd_im = values[odd_index + 1]
                var weight_re = twiddle_interleaved[twiddle_index]
                var weight_im = twiddle_interleaved[twiddle_index + 1]
                var weighted_re = odd_re * weight_re - odd_im * weight_im
                var weighted_im = odd_re * weight_im + odd_im * weight_re
                values[even_index] = even_re + weighted_re
                values[even_index + 1] = even_im + weighted_im
                values[odd_index] = even_re - weighted_re
                values[odd_index + 1] = even_im - weighted_im
                offset += 1
        stage_size *= 2


def _radix2_interleaved_scalar_in_place[
    dtype: DType
](
    mut values: List[Scalar[dtype]],
    size: Int,
    twiddle_interleaved: List[Scalar[dtype]],
    bit_reversal: List[Int],
) where dtype.is_floating_point():
    """Scalar reference for the typed interleaved radix-2 layout."""
    debug_assert(len(values) >= 2 * size, "interleaved radix-2 workspace is too short")
    debug_assert(
        len(twiddle_interleaved) == 2 * (size - 1),
        "invalid interleaved twiddle table",
    )
    debug_assert(len(bit_reversal) == size, "invalid bit-reversal table")

    for index in range(1, size):
        var reversed_index = bit_reversal[index]
        if index < reversed_index:
            values.swap_elements(2 * index, 2 * reversed_index)
            values.swap_elements(2 * index + 1, 2 * reversed_index + 1)

    var stage_size = 2
    while stage_size <= size:
        var half = stage_size // 2
        var stage_twiddle_start = half - 1
        for block_start in range(0, size, stage_size):
            for offset in range(half):
                var even_index = 2 * (block_start + offset)
                var odd_index = even_index + 2 * half
                var twiddle_index = 2 * (stage_twiddle_start + offset)
                var even_re = values[even_index]
                var even_im = values[even_index + 1]
                var odd_re = values[odd_index]
                var odd_im = values[odd_index + 1]
                var weight_re = twiddle_interleaved[twiddle_index]
                var weight_im = twiddle_interleaved[twiddle_index + 1]
                var weighted_re = odd_re * weight_re - odd_im * weight_im
                var weighted_im = odd_re * weight_im + odd_im * weight_re
                values[even_index] = even_re + weighted_re
                values[even_index + 1] = even_im + weighted_im
                values[odd_index] = even_re - weighted_re
                values[odd_index + 1] = even_im - weighted_im
        stage_size *= 2


def _scale_in_place[
    dtype: DType
](
    mut values: List[Scalar[dtype]], scale: Scalar[dtype]
) where dtype.is_floating_point():
    """Scale a scalar workspace with native-width SIMD and a scalar tail."""
    comptime width = simd_width_of[dtype]()
    var vector_end = len(values) - len(values) % width
    var values_ptr = values.unsafe_ptr()
    var vector_scale = SIMD[dtype, width](scale)
    for index in range(0, vector_end, width):
        var chunk = values_ptr.unsafe_load[width=width](index)
        values_ptr.unsafe_store[width=width](index, chunk * vector_scale)
    for index in range(vector_end, len(values)):
        values[index] *= scale
