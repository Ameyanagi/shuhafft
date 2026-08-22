# Benchmarks

No performance benchmark is published before the first real algorithm exists.
When benchmarks are added, record the CPU, OS, Mojo version, compiler options,
dataset provenance, warmup, iterations, statistic, and exact command.

Throughput programs belong in `bench_*.mojo`; sampling-profiler workloads belong
in `profile_*.mojo`. Results are development evidence, not permanent marketing
claims.

The deterministic three-second radix-2 burn collected 2563 main-thread samples;
2340 (91.3%) landed directly in the complex radix-2 prefix kernel. The separate
Bluestein burn collected 2554 main-thread samples: 2101 landed in its nested
radix-2 kernels, while 143/137/99 landed in convolution orchestration,
`execute_into`, and result writing. That evidence selects twiddle loads and
radix-2 execution as the optimization target rather than plan setup, and the
separate phases make the attribution reproducible.

## FFT API matrix

Run `pixi run bench`. The benchmark isolates complex and real, forward and
inverse, and caller-buffer and return-list APIs for `float32` and `float64` at
sizes 2^8, 2^12, and 2^16. Plan construction and deterministic input generation
are excluded. `allocation=none` means all buffers are caller-owned and reused;
`allocation=return_list` includes the documented result allocation. Checksums
prevent dead-code elimination.

Complex `execute_in_place` cells apply four unnormalized transforms to an
evolving buffer, then rescale outside the timed region. All other cells transform
fixed inputs into overwritten or newly returned outputs. Caller-buffer cells use
an iteration budget of `2^20 / n`; return-list cells use `2^18 / n`, with a
minimum of four iterations per cell. Four untimed iterations warm caller-buffer
paths.

Inputs use the Numerical Recipes LCG
`state = 1664525 * state + 1013904223 (mod 2^32)` with fixed seed `0x5A17C0DE`;
successive states are mapped to values in `[-1, 1]`; complex inputs consume two
successive states per element. Record benchmark numbers in PRs or commits,
together with the required machine and run metadata above, as development
evidence only.

## Compiled latency and profiler workflow

Run `pixi run profile` to compile `profile_fft.mojo` and collect 101 warmed
samples per direction. One `perf_counter_ns` interval encloses each complete
batch; no per-transform timer reads are summed. Nearest-rank p50 and p95 are
computed from total batch durations. Output keeps those batch percentiles
(`p50_batch_ns`, `p95_batch_ns`) distinct from the corresponding batch-average
per-transform values (`p50_mean_ns_per_transform`,
`p95_mean_ns_per_transform`). The latter are not percentiles of individually
timed calls. Plans, deterministic inputs, output buffers, and Bluestein
convolution workspace are created outside timed regions. Repeated in-place
radix-2 cells use orthonormal scaling to keep same-direction batches bounded;
the normalization is included in every result row.

The matrix covers radix-2 sizes 256, 4096, and 65536; awkward Bluestein sizes
257, 4093, and 65521; and alternating-order A/B cells for complex twiddle layout
and native-width interleaved SIMD. SIMD A/B cells cover both `float32` and
`float64` at 4096 and 65536 complex values.

For call-stack evidence, compile the same executable and sample each deterministic
single-algorithm phase separately:

```sh
pixi run mojo build -I src benchmarks/profile_fft.mojo -o .pixi/profile_fft
.pixi/profile_fft radix2-burn >/tmp/shuhafft-radix2-profile.txt &
profile_pid=$!
until grep -q '^PROFILE_BURN_READY ' /tmp/shuhafft-radix2-profile.txt; do
  kill -0 "$profile_pid" 2>/dev/null || { wait "$profile_pid"; exit 1; }
  sleep 0.01
done
sample "$profile_pid" 3 1 -file /tmp/shuhafft-radix2-sample.txt
wait "$profile_pid"

.pixi/profile_fft bluestein-burn >/tmp/shuhafft-bluestein-profile.txt &
profile_pid=$!
until grep -q '^PROFILE_BURN_READY ' /tmp/shuhafft-bluestein-profile.txt; do
  kill -0 "$profile_pid" 2>/dev/null || { wait "$profile_pid"; exit 1; }
  sleep 0.01
done
sample "$profile_pid" 3 1 -file /tmp/shuhafft-bluestein-sample.txt
wait "$profile_pid"
```

Each burn flushes a `PROFILE_BURN_READY` record only after its input, output,
plan, twiddles, and reusable workspace are constructed. The polling handshake
therefore starts macOS `sample` only after setup and before the first repeated
transform. Each burn then executes only the named algorithm and lasts long
enough for a 1 ms interval; benchmark setup and the other algorithm cannot land
in the captured phase. On other platforms, use the same ready record with the
native sampling profiler while keeping the compiled workload and phase names
unchanged.

### Development profile, 2026-08-22

Apple M4 (10 cores), 32 GB, macOS 26.5.1, arm64, Mojo 1.0.0. The radix-2 and
Bluestein development rows use `float64`; the A/B results below state their
`float32` or `float64` dtype explicitly. Exact command: `pixi run profile`.
Times vary with power and thermal state and are not release claims.

The table reports the normalized mean per transform within the p50/p95 batch;
raw batch percentiles are emitted by the executable.

| Algorithm | Dtype | Normalization | n | Direction | p50 batch mean | p95 batch mean |
|---|---|---|---:|---|---:|---:|
| radix-2 | float64 | ortho | 256 | forward | 1.41 us | 1.41 us |
| radix-2 | float64 | ortho | 256 | inverse | 1.41 us | 1.41 us |
| radix-2 | float64 | ortho | 4096 | forward | 29.3 us | 29.8 us |
| radix-2 | float64 | ortho | 4096 | inverse | 29.3 us | 29.5 us |
| radix-2 | float64 | ortho | 65536 | forward | 1.15 ms | 1.24 ms |
| radix-2 | float64 | ortho | 65536 | inverse | 0.796 ms | 0.841 ms |
| Bluestein | float64 | backward | 257 | forward | 17.5 us | 18.5 us |
| Bluestein | float64 | backward | 257 | inverse | 17.4 us | 18.9 us |
| Bluestein | float64 | backward | 4093 | forward | 181 us | 185 us |
| Bluestein | float64 | backward | 4093 | inverse | 181 us | 184 us |
| Bluestein | float64 | backward | 65521 | forward | 4.26 ms | 4.61 ms |
| Bluestein | float64 | backward | 65521 | inverse | 4.29 ms | 4.66 ms |

In alternating-order matched A/B cells, directly loadable complex twiddles
reduced p50/p95 from 30/31 to 28/29 us at n=4096 and from 0.800/0.850 to
0.752/0.801 ms at n=65536. Native-width interleaved SIMD reduced float32 p50/p95
from 44/44 to 19/19 us at n=4096 and from 1.11/1.11 to 0.539/0.544 ms at n=65536.
For float64, it reduced 44/45 to 22/22 us and 1.15/1.17 to 0.714/0.731 ms at the
same sizes. Direct scalar-versus-SIMD tests cover both dtypes across scalar-only
early stages and native-width groups. The complex public array-of-structs layout
was not unsafely reinterpreted for wider SIMD.
