# Changelog

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and uses semantic versioning after the first public release.

## [Unreleased]

### Added

- Initial experimental repository scaffold.
- Cross-precision analytic delta and constant FFT reference families, including
  inverse duality and reusable-plan ownership coverage.
- Cross-precision shifted-impulse references for both transform directions and
  focused `Float32` normalization execution coverage.
- Reusable bounded `BluesteinFFTPlan` support for arbitrary complex lengths,
  with automatic one-shot dispatch and caller-owned output APIs.
- Compiled p50/p95 workloads and a sampling-profiler workflow covering forward,
  inverse, radix-2, Bluestein, twiddle-layout, and SIMD paths.
- Direct `Float32`/`Float64` scalar-versus-SIMD interleaved-kernel tests and
  `Float32` Bluestein inverse and normalized round-trip coverage.

### Changed

- Store radix-2 twiddles in directly loadable complex form based on profiler-led
  A/B measurements.
- Use measured native-width SIMD for typed interleaved real-inverse butterflies
  while retaining scalar reference and tail coverage.
- Measure one clock interval per benchmark batch, label batch percentiles and
  normalized batch means separately, and isolate profiler burns by algorithm.
- Document plan `validate()` as a structural checkpoint and correct the
  Int-backed normalization contract in the v0.1 plan.
