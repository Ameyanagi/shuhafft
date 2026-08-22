# ShuhaFFT

> **Experimental — API not yet released.**

Production-quality fast Fourier transforms for Mojo.

## Install

In a [Pixi](https://pixi.sh/) project, add the Mojo ecosystem channel and the
ShuhaFFT package:

```sh
pixi project channel add https://ameyanagi.github.io/mojo-channel
pixi add mojo-shuhafft
```

Alternatively, run ShuhaFFT from a source checkout:

```sh
git clone https://github.com/Ameyanagi/shuhafft.git
cd shuhafft
pixi install --locked
```

To run a file of your own against the library from the checkout, save it in the
checkout and run `pixi run mojo run -I src your_file.mojo`.

The Conda package is `mojo-shuhafft`; its Mojo import is `shuhafft`.

## Quickstart

One-shot complex transforms accept any length from 1 through 2^20: power-of-two
inputs use radix-2 and other lengths dispatch to Bluestein convolution. Reusable
`FFTPlan` remains the lean radix-2 plan; use `BluesteinFFTPlan` when repeatedly
transforming a prime or otherwise awkward length. Real transforms currently
require a power-of-two length of at least 2. This example generates one second
of a 50 Hz sine sampled at 1024 Hz. With 1024 samples the bin resolution is 1
Hz, so the peak lands exactly in bin 50:

```mojo
from shuhafft import rfft, rfftfreq
from std.math import sin


def main() raises:
    var sample_rate = 1024
    var signal = List[Float64](capacity=sample_rate)
    var two_pi = 6.283185307179586476925286766559
    for sample_index in range(sample_rate):
        signal.append(
            sin(two_pi * 50.0 * Float64(sample_index) / Float64(sample_rate))
        )

    var spectrum = rfft(signal)
    var peak_bin = 1
    for bin_index in range(2, len(spectrum)):
        if Float64(spectrum[bin_index].squared_norm()) > Float64(
            spectrum[peak_bin].squared_norm()
        ):
            peak_bin = bin_index
    var freqs = rfftfreq(sample_rate, 1.0 / Float64(sample_rate))
    print("Peak frequency:", freqs[peak_bin], "Hz")  # Peak frequency: 50.0 Hz
```

Save this as `spectrum.mojo` in a checkout and run
`pixi run mojo run -I src spectrum.mojo`.

The explicit form `rfft[DType.float64](...)` is available as a disambiguation
escape hatch when input inference is not enough.

## Reuse an arbitrary-length plan

`BluesteinFFTPlan` precomputes its chirp and convolution spectrum once. Its
`execute_into` method reuses plan-owned convolution workspace and caller-owned
output, including for prime lengths:

```mojo
from shuhafft import BluesteinFFTPlan, FFTDirection
from std.complex import ComplexFloat64


def main() raises:
    var values = List[ComplexFloat64](length=1009, fill=ComplexFloat64(0.0))
    values[0] = ComplexFloat64(1.0)
    var plan = BluesteinFFTPlan[DType.float64](1009, FFTDirection.FORWARD)
    var spectrum = plan.make_buffer()
    plan.execute_into(values, spectrum)
    print(len(spectrum))  # 1009
```

## Reuse a plan for Welch-style frames

For Welch or spectrogram-style work, construct one plan and reuse its output
buffer across frames of a longer signal:

```mojo
from shuhafft import RealFFTPlan


def main() raises:
    var frame_size = 1024
    var plan = RealFFTPlan[DType.float64](frame_size)
    var spectrum = plan.make_spectrum()
    var long_signal = List[Float64](length=4 * frame_size, fill=0.0)
    for frame_index in range(4):
        var start = frame_index * frame_size
        plan.forward_into(long_signal[start : start + frame_size], spectrum)
        # Consume `spectrum` here before the next frame overwrites it.
```

For more detail, see [the architecture](docs/architecture.md),
[design principles](docs/design.md), the
[executable v0.1 plan](docs/v0.1-plan.md), and [roadmap](docs/roadmap.md).

## Transform contract

Backward normalization is the default, so forward transforms are unscaled,
inverse transforms divide by the length, and round trips work without extra
scaling. `FORWARD` instead divides only the forward transform by the length,
`ORTHO` divides both directions by `sqrt(n)`, and `NONE` leaves both directions
unscaled, so a forward/inverse pair under `NONE` returns `n * x`. The
real-transform contract is:

- Bins are DC-first in ascending frequency; bin `k` is
  `k * sample_rate / n`.
- An `n`-sample real signal produces `n // 2 + 1` complex bins, from DC through
  Nyquist inclusive.
- Forward output gives DC and Nyquist exactly-zero imaginary parts; inverse
  ignores nonzero imaginaries supplied at those endpoints, like SciPy `irfft`.
- With default BACKWARD normalization, `irfft(rfft(x)) == x` and a plan's
  `inverse(forward(x)) == x`.

Use the one-shot `fft`, `ifft`, `rfft`, and `irfft` functions for exploratory
work; they accept `normalization=` (default `FFTNormalization.BACKWARD`), and
`fft`/`ifft` also accept real input. Reuse `FFTPlan` or `RealFFTPlan` when
running repeated transforms. The API is experimental and may change before
v0.1.

## Scope

ShuhaFFT owns transform semantics, planning, normalization, and optimized
backends without absorbing signal-processing policy.

The current implementation provides CPU radix-2 complex and real transforms,
arbitrary-length complex Bluestein transforms, native-width SIMD for the typed
interleaved real-inverse kernel, one-shot conveniences, and reusable plans for
Float32 and Float64. The project is independently installable and does not
require any application from the wider ecosystem.

## Development

From a source checkout, run:

```sh
pixi run check
pixi run example
```

The exact stable Mojo compiler and all development dependencies are captured in
`pixi.lock`. Runtime and library code is Mojo-first and pure Mojo wherever
practical. Build-time data generation may use another language when justified,
but generated outputs must be deterministic, checksum-pinned, licensed, and
documented.

## Name

Shuha (周波) means frequency or cycle in Japanese, matching this project's
Japanese-named sibling libraries. Public functions keep the NumPy/SciPy
spellings (`fft`, `rfft`) so the ecosystem reads consistently to scientists.

## Repository map

- `src/shuhafft/`: library source; `__init__.mojo` defines the package boundary
- `tests/`: TestSuite unit, reference-value, and invariant tests
- `examples/`: small compilable usage programs
- `benchmarks/`: reproducible throughput, latency, and profiler workloads
- `docs/`: architecture, design, compatibility, roadmap, and release policy
- `conda.recipe/`: local Rattler build recipe

## License

Licensed under either Apache-2.0 or MIT, at your option.
