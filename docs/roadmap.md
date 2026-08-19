# Roadmap

## v0.1 — Foundation

- implement CPU radix-2 complex-to-complex forward and inverse transforms for Float32 and Float64, both in-place and out-of-place, with strong numerical invariants.
- Define the smallest useful public API and its invariants.
- Add unit, reference-value, and property/invariant coverage.
- Build and test the precompiled package on supported targets.

## v0.2 — Usability

- Add ergonomic APIs only after v0.1 usage demonstrates repeated friction.
- Expand examples and integration fixtures.
- Publish the first modular-community recipe when the package is useful alone.

## v0.3 — Performance

- Add reproducible benchmarks and representative datasets.
- Optimize measured bottlenecks without weakening correctness or API clarity.
- Add SIMD or specialized backends only behind the same semantic contract.

## v1.0 — Stability

- Document every public symbol and error contract.
- Provide a compatibility and deprecation policy.
- Support the declared OS and architecture matrix in CI.
- Require downstream proof from at least one independent consumer.

## Not planned

Signal processing, plotting, file formats, GPU kernels, mixed radix, Bluestein, multidimensional transforms, and distributed execution are outside v0.1.
