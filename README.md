# ShuhaFFT

> **Experimental — API not yet released.**

Production-quality fast Fourier transforms for Mojo.

## Scope

ShuhaFFT owns transform semantics, planning, normalization, and optimized backends without absorbing signal-processing policy.

The first implementation milestone is intentionally narrow: implement CPU radix-2 complex-to-complex forward and inverse transforms for Float32 and Float64, both in-place and out-of-place, with strong numerical invariants.
The project is independently installable and does not require any application
from the wider ecosystem.

## Development

Install [Pixi](https://pixi.sh/), then run:

```sh
pixi install --locked
pixi run check
pixi run example
```

The exact stable Mojo compiler and all development dependencies are captured in
`pixi.lock`. Runtime and library code is Mojo-first and pure Mojo wherever
practical. Build-time data generation may use another language when justified,
but generated outputs must be deterministic, checksum-pinned, licensed, and
documented.

## Package

The Mojo import is `shuhafft`. The eventual Conda distribution is
`mojo-shuhafft`. Source lives under `src/shuhafft/`, whose
`__init__.mojo` defines the package boundary.

The first correctness slice exposes validated scalar CPU radix-2 plans for
`Float32` and `Float64` complex values:

```mojo
from shuhafft import ComplexFloat64, FFTDirection, FFTPlan

var samples = List[ComplexFloat64](capacity=4)
samples.append(ComplexFloat64(1.0))
samples.append(ComplexFloat64(2.0))
samples.append(ComplexFloat64(3.0))
samples.append(ComplexFloat64(4.0))
var spectrum = FFTPlan[DType.float64](
    len(samples), FFTDirection.forward()
).execute(samples)
```

Lengths must be non-zero powers of two. Backward normalization is the default,
so forward transforms are unscaled and inverse transforms divide by the length.
The API is experimental and may change before v0.1.

## Repository map

- `src/shuhafft/`: library or application source
- `tests/`: TestSuite unit, reference-value, and invariant tests
- `examples/`: small compilable usage programs
- `benchmarks/`: reproducible methodology and later benchmark programs
- `docs/`: architecture, design, compatibility, roadmap, and release policy
- `conda.recipe/`: local Rattler build recipe

See [the architecture](docs/architecture.md), [design principles](docs/design.md),
the [executable v0.1 plan](docs/v0.1-plan.md), and [roadmap](docs/roadmap.md)
before proposing a new dependency or feature.

## License

Licensed under either Apache-2.0 or MIT, at your option.
