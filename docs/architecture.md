# Architecture

ShuhaFFT owns FFT direction, normalization, plans, twiddles, complex and real transform contracts, and backend kernels.

## Dependency boundary

Allowed ecosystem dependencies: Mojo standard library numeric, complex, buffer, and SIMD facilities only in the foundation.
Expected downstream consumers: Nami and independent scientific or numerical applications.

Dependencies point from applications and higher-level packages toward smaller
foundations. This repository must never import a downstream consumer. New
dependencies require a documented need and must not force unrelated users to
install an application, renderer, language layer, or scientific stack.

## Layers

Planned implementation areas: direction, normalization, planner-compatible plans, twiddles, complex FFT/IFFT, later real transforms, algorithm kernels, SIMD, and isolated GPU backends.

The current correctness slice has four concrete layers:

```text
package root
    -> FFTPlan (validation, ownership and scaling contract)
        -> internal scalar radix-2 kernel
            -> Mojo standard ComplexSIMD and math primitives
```

`FFTDirection` and `FFTNormalization` are backend-independent nominal values.
`FFTPlan` is the future backend seam: optimized implementations may replace
the scalar kernel, but they must preserve its validation, sign, scaling,
in-place, and out-of-place behavior. Backend selection is deliberately absent
until there is a second implementation to select.

The package root exports only the small documented public surface. Algorithms,
generated tables, platform details, and backend implementations remain in
their owning modules. Generic Mojo-native buffers, spans, strings, and
collections are preferred over an ecosystem-specific universal container.

The [reference architecture](reference-architecture.md) records the exact
primary FFT sources used to evaluate future planner, scratch, algorithm, and
backend seams. It is design evidence only; no reference implementation is a
runtime or source dependency.

## Data flow

Input validation occurs at the public boundary. Internal layers operate on
explicit typed values, produce deterministic outputs for deterministic inputs,
and report invalid state rather than silently replacing it with a default.
I/O, clocks, randomness, terminal queries, filesystem access, and accelerator
selection stay at explicit effect or backend boundaries.
