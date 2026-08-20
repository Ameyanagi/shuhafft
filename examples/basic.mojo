from shuhafft import rfft
from std.math import sin


def main() raises:
    var sample_rate = 1024
    var signal = List[Float64](capacity=sample_rate)
    var two_pi = 6.283185307179586476925286766559
    for sample_index in range(sample_rate):
        signal.append(sin(two_pi * 50.0 * Float64(sample_index) / Float64(sample_rate)))

    var spectrum = rfft[DType.float64](signal)
    var peak_bin = 1
    for bin_index in range(2, len(spectrum)):
        if Float64(spectrum[bin_index].squared_norm()) > Float64(
            spectrum[peak_bin].squared_norm()
        ):
            peak_bin = bin_index
    print("Peak bin:", peak_bin)
