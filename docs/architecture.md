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

The current implementation has four concrete layers:

```text
package root
    -> one-shot fft/ifft dispatcher
        -> FFTPlan for power-of-two lengths
        -> BluesteinFFTPlan for arbitrary complex lengths
            -> plan-owned chirp, convolution spectrum, and reusable workspace
    -> RealFFTPlan for power-of-two compact real transforms
        -> internal radix-2 kernels
            -> ComplexSIMD path for public complex buffers
            -> native-width SIMD path for interleaved real-inverse workspace
```

`FFTDirection` and `FFTNormalization` are backend-independent nominal values.
`FFTPlan` is the radix-2 backend seam: optimized implementations may replace its
kernel, but they must preserve construction-time validation, explicit
checkpoints, input-length validation, sign, scaling, in-place, and out-of-place
behavior. `BluesteinFFTPlan` preserves the same direction and normalization
contract while transforming lengths from 1 through 2^20. It chooses the
smallest power-of-two convolution workspace that holds `2*n - 1` values,
precomputes the chirp convolution spectrum, and reuses workspace on every
`execute_into` or `execute_in_place` call. The explicit length bound prevents
malformed input from requesting multi-gigabyte plan storage.

Public complex buffers remain `List[ComplexSIMD[dtype, 1]]`. Radix-2 plans store
twiddles in that same complex layout, avoiding reconstruction and a second table
load in the measured dominant kernel. Mojo 1.0 does not
provide a sound contiguous wider-lane view of that array-of-structs layout, so
the complex kernel does not reinterpret it. `RealFFTPlan.inverse_into` instead
reuses its caller-owned scalar output as `[re, im, ...]` workspace. A plan-owned
interleaved twiddle table gives both buffers the same typed contiguous layout;
complete native-width butterfly chunks use SIMD and scalar code handles tails.
This path allocates no execution-time scratch storage.

The general complex kernel deliberately stays scalar-width. A sound
native-width view of the public array-of-structs complex layout is unavailable
in Mojo 1.0, and an unsafe reinterpretation would weaken its ownership contract.
SIMD is used only for the typed interleaved scalar layout where p50/p95 A/B
profiles show a benefit and scalar tails have direct-DFT coverage.

The package root exports only the small documented public surface. Algorithms,
generated tables, platform details, and backend implementations remain in
their owning modules. Generic Mojo-native buffers, spans, strings, and
collections are preferred over an ecosystem-specific universal container.

## Data flow

Plan invariants are established at construction and trusted during execution;
`validate()` is an explicit structural checkpoint after unusual operations. It
checks configuration, nested-plan contracts, and table shapes without
recomputing numerical twiddle, chirp, or convolution-spectrum contents. Direct
mutation of those underscore-prefixed contents remains out of contract.
Execution still validates caller-owned input lengths at the public boundary.
Internal layers operate on explicit typed values and produce deterministic
outputs for deterministic inputs. I/O, clocks, randomness, terminal queries,
filesystem access, and accelerator selection stay at explicit effect or backend
boundaries.
