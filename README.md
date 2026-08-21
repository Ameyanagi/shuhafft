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

ShuhaFFT currently provides radix-2 transforms, so transform lengths must be
powers of two; real transforms need at least 2 samples. This example generates
one second of a 50 Hz sine sampled at 1024 Hz. With 1024 samples the bin
resolution is 1 Hz, so the peak lands exactly in bin 50:

```mojo
from shuhafft import rfft
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
    print("Peak bin:", peak_bin)  # Peak bin: 50
```

Save this as `spectrum.mojo` in a checkout and run
`pixi run mojo run -I src spectrum.mojo`.

The explicit form `rfft[DType.float64](...)` is available as a disambiguation
escape hatch when input inference is not enough.

## Reuse a plan for Welch-style frames

For Welch or spectrogram-style work, construct one plan and reuse its output
buffer across frames of a longer signal:

```mojo
from shuhafft import ComplexFloat64, RealFFTPlan


def main() raises:
    var frame_size = 1024
    var plan = RealFFTPlan[DType.float64](frame_size)
    var spectrum = List[ComplexFloat64](
        length=plan.spectrum_size(), fill=ComplexFloat64(0.0)
    )
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
scaling. The real-transform contract is:

- Bins are DC-first in ascending frequency; bin `k` is
  `k * sample_rate / n`.
- An `n`-sample real signal produces `n // 2 + 1` complex bins, from DC through
  Nyquist inclusive.
- Forward output gives DC and Nyquist exactly-zero imaginary parts; inverse
  ignores nonzero imaginaries supplied at those endpoints, like SciPy `irfft`.
- With default BACKWARD normalization, `irfft(rfft(x)) == x` and a plan's
  `inverse(forward(x)) == x`.

Use the one-shot `fft`, `ifft`, `rfft`, and `irfft` functions for exploratory
work. Reuse `FFTPlan` or `RealFFTPlan` when running repeated transforms or when
you need ORTHO, FORWARD, or NONE normalization. The API is experimental and may
change before v0.1.

## Scope

ShuhaFFT owns transform semantics, planning, normalization, and optimized
backends without absorbing signal-processing policy.

The current implementation provides CPU radix-2 complex and real transforms for
Float32 and Float64, with one-shot conveniences and reusable plans. The project
is independently installable and does not require any application from the
wider ecosystem.

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
- `benchmarks/`: reproducible methodology and later benchmark programs
- `docs/`: architecture, design, compatibility, roadmap, and release policy
- `conda.recipe/`: local Rattler build recipe

## License

Licensed under either Apache-2.0 or MIT, at your option.
