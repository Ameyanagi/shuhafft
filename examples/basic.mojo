from shuhafft import BluesteinFFTPlan, ComplexFloat64, FFTDirection, rfft
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

    var awkward = List[ComplexFloat64](length=1009, fill=ComplexFloat64(0.0))
    awkward[0] = ComplexFloat64(1.0)
    var awkward_plan = BluesteinFFTPlan[DType.float64](1009, FFTDirection.FORWARD)
    var awkward_spectrum = awkward_plan.make_buffer()
    awkward_plan.execute_into(awkward, awkward_spectrum)
    print("Prime-length bins:", len(awkward_spectrum))
