# Benchmarks

No performance benchmark is published before the first real algorithm exists.
When benchmarks are added, record the CPU, OS, Mojo version, compiler options,
dataset provenance, warmup, iterations, statistic, and exact command.

Benchmark programs belong in `bench_*.mojo`. Results are development evidence,
not permanent marketing claims.

## Radix-2 forward transforms

Run `pixi run bench`. The benchmark reports mean nanoseconds per unscaled forward
transform for `float32` and `float64` sizes from 2^8 through 2^16. Plan
construction is excluded. Each cell reuses one in-place buffer, repeatedly
transforms its evolving values, and emits a checksum to prevent dead-code
elimination. After four warmup transforms, each cell measures `2^22 / n`
transforms. Magnitudes are rescaled outside the timed regions after every four
transforms.

Inputs use the Numerical Recipes LCG
`state = 1664525 * state + 1013904223 (mod 2^32)` with fixed seed `0x5A17C0DE`;
successive states are mapped to real and imaginary values in `[-1, 1]`. Record
benchmark numbers in PRs or commits, together with the required machine and run
metadata above, as development evidence only.
