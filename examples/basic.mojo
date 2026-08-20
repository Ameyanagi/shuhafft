from shuhafft import ComplexFloat64, FFTDirection, FFTPlan


def main() raises:
    var samples = List[ComplexFloat64](capacity=4)
    samples.append(ComplexFloat64(1.0))
    samples.append(ComplexFloat64(2.0))
    samples.append(ComplexFloat64(3.0))
    samples.append(ComplexFloat64(4.0))
    var spectrum = FFTPlan[DType.float64](len(samples), FFTDirection.forward()).execute(
        samples
    )
    for bin in spectrum:
        print(bin)
