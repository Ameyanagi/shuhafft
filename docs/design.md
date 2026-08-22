# Design

## Principles

- Mojo is the runtime implementation language.
- Prefer pure Mojo and safe standard-library APIs.
- Keep the root API small, typed, documented, and testable.
- Separate semantic contracts from optimized CPU, SIMD, GPU, terminal, or
  rendering backends.
- Establish correctness and reference fixtures before optimization.
- Make invalid public configuration unrepresentable when practical; otherwise
  reject it explicitly.
- Establish stored invariants at construction and trust them thereafter.
  Direct mutation of underscore-prefixed fields is out of contract; validated
  types provide `validate()` for explicit structural checkpoints. Plan
  validation checks configuration, nested-plan contracts, and table lengths; it
  does not recompute numerical twiddle, chirp, or spectrum contents.
- Represent nominal modes as Int-backed structs with `comptime` constants.
- Accept contiguous buffer inputs as `Span` values so callers can pass windows
  without an intermediate copy.
- Preserve source mappings, numerical tolerances, ownership, and provenance as
  first-class data when the domain requires them.
- Do not add a framework-wide array, executor, renderer, or application model.

## Tradeoffs

The project accepts a narrower initial feature set in exchange for reviewable
contracts and sparse dependencies. Generated tables are acceptable when their
sources, Unicode or data version, licenses, checksums, and deterministic update
procedure are committed. Consumers must not need the generator toolchain.

## Out of scope

Signal processing, plotting, file formats, GPU kernels, mixed radix,
multidimensional transforms, and distributed execution remain outside the
current package. Bounded Bluestein convolution is included because it preserves
the existing complex transform contract without adding a backend-selection API.
