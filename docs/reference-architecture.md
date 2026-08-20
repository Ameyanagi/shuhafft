# Reference architecture

This document records design evidence for ShuhaFFT without importing source,
generated code, or runtime dependencies from another FFT implementation. The
references were shallow-cloned and read at the exact revisions below. The
result is an independent Mojo design: mathematical contracts and architectural
ideas are adopted where they fit, while implementation code is not copied.

## Primary-reference ledger

| Reference | Revision inspected | License | Material inspected | Architectural value |
| --- | --- | --- | --- | --- |
| [RustFFT](https://github.com/ejmahler/RustFFT/tree/4758ab0dd6f256c50ac8987c75c9cb96152dc2ca) | `4758ab0dd6f256c50ac8987c75c9cb96152dc2ca`, tag `6.4.1`, 2025-09-17 | MIT OR Apache-2.0 | [`src/lib.rs`](https://github.com/ejmahler/RustFFT/blob/4758ab0dd6f256c50ac8987c75c9cb96152dc2ca/src/lib.rs), [`src/plan.rs`](https://github.com/ejmahler/RustFFT/blob/4758ab0dd6f256c50ac8987c75c9cb96152dc2ca/src/plan.rs), scalar algorithms, and AVX/SSE/Neon/WASM planners | Planner-selected kernels, plan reuse, explicit scratch requirements, and runtime CPU-feature seams |
| [pocketfft](https://github.com/mreineck/pocketfft/tree/c90e55b3d529f8efa40ed01a20de22405f45fc65) | `c90e55b3d529f8efa40ed01a20de22405f45fc65`, 2026-06-30 | BSD-3-Clause | [`README.md`](https://github.com/mreineck/pocketfft/blob/c90e55b3d529f8efa40ed01a20de22405f45fc65/README.md) and [`pocketfft_hdronly.h`](https://github.com/mreineck/pocketfft/blob/c90e55b3d529f8efa40ed01a20de22405f45fc65/pocketfft_hdronly.h) | Compact shape/stride contracts, cost-based FFTPACK-versus-Bluestein selection, accurate twiddles, and bounded optional plan caching |
| [FFTW](https://github.com/FFTW/fftw3/tree/93ed4c786934aec9946f8dda4b4e3eb08f8be41c) | `93ed4c786934aec9946f8dda4b4e3eb08f8be41c`, 2026-06-10 | GPL-2.0-or-later | [`doc/reference.texi`](https://github.com/FFTW/fftw3/blob/93ed4c786934aec9946f8dda4b4e3eb08f8be41c/doc/reference.texi), [`api/apiplan.c`](https://github.com/FFTW/fftw3/blob/93ed4c786934aec9946f8dda4b4e3eb08f8be41c/api/apiplan.c), and [`kernel/planner.c`](https://github.com/FFTW/fftw3/blob/93ed4c786934aec9946f8dda4b4e3eb08f8be41c/kernel/planner.c) | Mature plan lifetime, planning-rigor and time-budget concepts, reusable execution, problem/solver separation, alignment, and wisdom tradeoffs |
| [AbstractFFTs.jl](https://github.com/JuliaMath/AbstractFFTs.jl/tree/31fc5f4de1d5a4f0f1565e879ff519d46255b02a) | `31fc5f4de1d5a4f0f1565e879ff519d46255b02a`, 2026-06-09 | MIT | [`src/definitions.jl`](https://github.com/JuliaMath/AbstractFFTs.jl/blob/31fc5f4de1d5a4f0f1565e879ff519d46255b02a/src/definitions.jl) | A small common transform vocabulary, convenience calls implemented through plans, exact shape/type applicability, preallocated output, inverse-plan reuse, and normalization wrappers |

The local research clones remain outside this repository; they are evidence,
not package inputs. Licenses in the table describe the inspected repositories,
not ShuhaFFT, which remains MIT OR Apache-2.0.

### Organization observed at those revisions

- **RustFFT:** `src/lib.rs` defines direction and the execution trait, including
  separate in-place, immutable-input, out-of-place, and caller-scratch paths.
  `src/plan.rs` selects AVX, SSE, Neon, WASM SIMD, or scalar planners. The scalar
  and ISA planners cache recipes separately from instantiated algorithms, while
  `src/algorithm/` owns DFT, butterflies, radix, mixed-radix, Good-Thomas,
  Rader, and Bluestein kernels. Transforms are unnormalized.
- **pocketfft:** the public shape/stride/axes functions and internal machinery
  occupy one header, but the responsibilities are distinct. `cfftp`/`rfftp`
  implement factorized complex/real kernels, `fftblue` owns Bluestein, and
  `pocketfft_c`/`pocketfft_r` select between them from estimated cost. The
  multidimensional driver obtains per-axis plans, manages temporary buffers,
  and applies the caller's explicit scale factor once.
- **FFTW:** the public `api/` converts shape, sign, placement, strides, and
  planning flags into a problem. `api/apiplan.c` drives escalating planning
  rigor and returns a reference-counted API plan. `kernel/planner.c` registers
  and searches problem-specific solvers with memoized wisdom; `dft/`, `rdft/`,
  `reodft/`, and generated codelets own transform kernels. Execution reuses an
  opaque plan; transforms are not normalized.
- **AbstractFFTs.jl:** `src/definitions.jl` defines a backend-neutral `Plan`,
  shape/dimension vocabulary, convenience calls routed through `plan_*`,
  preallocated application, inverse-plan caching, and a scaled-plan wrapper.
  It intentionally contains no FFT kernel or planner heuristic; downstream
  packages implement those pieces behind the shared API.

## Contracts before kernels

ShuhaFFT separates four concerns:

```text
public convenience API
        -> validated transform specification
                -> planner and selected plan
                        -> scalar, SIMD, or accelerator kernel
```

The public contract must not change when the selected kernel changes. A caller
chooses the transform; the planner chooses how to compute it. Direction,
normalization, input/output shape, element type, placement, and ownership are
semantic inputs. Radix decomposition, codelets, vector width, twiddle storage,
and launch geometry are implementation decisions.

The current `FFTPlan[dtype]` already combines the validated specification and
the only available scalar radix-2 implementation. That is appropriate for
v0.1. The layers should be separated internally only when a second algorithm or
backend makes selection real. A public `FFTPlanner` before then would expose an
abstraction with no decision to make. Even after selection becomes real,
`FFTPlan(...)` remains the public deterministic planning entry point. The
planner, transform specification, algorithm selector, and CPU backend policy
remain internal until a concrete caller requirement proves that exposing one is
better than adding a narrow constructor or diagnostic.

## Planner and convenience API

### Convenience path

Convenience functions should construct a deterministic default plan, execute
once, and return owned output. Their role is discoverability, not a separate
execution engine. They may allocate output and required scratch. They must not
populate a hidden process-global cache.

The first convenience surface after v0.1 should remain one-dimensional and
complex-to-complex:

```mojo
def fft[dtype: DType](
    values: List[ComplexSIMD[dtype, 1]],
    normalization: FFTNormalization = FFTNormalization.backward(),
) raises -> List[ComplexSIMD[dtype, 1]]

def ifft[dtype: DType](
    values: List[ComplexSIMD[dtype, 1]],
    normalization: FFTNormalization = FFTNormalization.backward(),
) raises -> List[ComplexSIMD[dtype, 1]]
```

These signatures are architectural targets, not a v0.1 commitment. They should
be added only after Mojo ownership and generic inference make them at least as
clear as direct `FFTPlan` construction.

### Planned path

Repeated transforms use an explicit plan:

```mojo
var plan = FFTPlan[DType.float64](
    size,
    FFTDirection.forward(),
    FFTNormalization.backward(),
)
var output = plan.execute(input)
plan.execute_in_place(buffer)
```

Plan construction may validate, factor a length, choose a backend and
algorithm, and precompute immutable data. Execution must not repeat planning.
Plans are reusable for the exact shape, type, direction, normalization, and
selected CPU implementation with which they were created. Placement remains an
execution method: one plan may expose both in-place and owned out-of-place paths
when its selected kernel supports both.

When general lengths exist, an internal planner should take a value-like
specification and return a plan containing a selected kernel:

```text
TransformSpec
  dtype: Float32 | Float64
  length: positive Int
  direction: forward | inverse
  normalization: none | backward | forward | ortho
  selected CPU family: scalar | supported SIMD family
```

`TransformSpec` and the selected-family field are internal records, not root
exports or caller configuration. Automatic selection always has a scalar
fallback. Tests and benchmarks may force an internal family, but production
callers do not acquire an unstable backend-selection API merely to make those
tests possible.

Normalization is intentionally part of the reusable public plan but not of the
kernel-selection key unless a measured optimization proves fused scaling
changes the best kernel. This prevents four copies of identical twiddle and
factorization state.

No measured planning, time limit, persistent wisdom, or user-selectable
algorithm belongs in the first planner. Selection begins as deterministic,
versioned heuristics. Benchmarked planning can be evaluated later as a separate
policy because FFTW demonstrates both its power and its costs: planning can be
slow, mutate planning buffers, depend on the host, and require global wisdom
and cleanup rules.

### Plan integrity under Mojo 1.0

Mojo 1.0 underscore-prefixed fields remain externally reachable. A reusable
plan must therefore never rely on the apparent privacy of duplicated mutable
fields. Once precomputed state exists, `FFTPlan` is a small public façade over
one package-controlled CPU plan state. Length, direction, selected kernel,
normalization, twiddle/table dimensions, and scratch requirements have one
canonical stored representation. Public accessors derive from that state rather
than keeping independent copies that can drift.

The current scalar plan may retain direct fields only while it has no cached
kernel or twiddle state: its length is revalidated and every reachable
direction/normalization bit pattern is meaningful. Extracting the internal
kernel contract is also the gate for replacing those direct fields with the
coherent state above.

Before every execution, one complete validation pass proves all of the
following before input, output, or scratch is mutated:

- the public plan state is structurally valid for its dtype and transform kind;
- its length, direction, normalization, selected-kernel metadata, and every
  required twiddle or table dimension agree;
- the actual input and output lengths match the canonical plan length;
- the selected placement has enough correctly typed scratch;
- all required non-aliasing relationships hold; and
- the selected kernel supports the canonical dtype, length, direction, and
  placement.

A mismatch raises. Execution does not silently repair the plan, select another
kernel, or rebuild tables. Raw table storage, unchecked pointers, and device
resources are not exposed as reachable fields on the public CPU façade. If
Mojo cannot provide a package-controlled owner that preserves that boundary,
precomputed state stays in a separately validated safe value and each execution
checks its structural agreement; an optimized representation is not permitted
to weaken memory safety.

Internal kernels receive actual buffer lengths alongside validated metadata.
They use checked indexing or prove bounds from those actual lengths, validate
every auxiliary table before its first access, and remain bounds-safe if a
caller corrupts any externally reachable plan field. The public validation is
the semantic error boundary, not the kernel's only memory-safety boundary.

### Public-surface limit

The root execution surface grows only to `fft`, `ifft`, `FFTPlan`, the existing
nominal direction/normalization values, and the explicit scratch query/execution
methods described below. `FFTPlanner`, `TransformSpec`, `PlannedKernel`,
algorithm names, twiddle storage, backend policy, and CPU feature values remain
internal. A diagnostics API may later report the selected family without
letting callers construct invalid combinations.

## Shape, type, direction, and normalization

### Shape and layout

The v0.1 shape is exactly one contiguous dimension of `n` complex elements:

- `n` is a non-zero power of two;
- the execution input length equals the planned length;
- in-place execution does not change length;
- out-of-place execution returns exactly `n` elements;
- a singleton transform is the identity under every normalization.

General positive lengths come before batches, strides, or multiple dimensions.
Later shape support needs an explicit `Shape`/`Strides` contract and checked
size-product arithmetic. pocketfft and FFTW both show that axes, negative or
byte strides, alignment, aliasing, and real-transform packing quickly dominate
the API. ShuhaFFT should not encode those policies in a one-dimensional plan.

Batching should first accept contiguous fixed-size chunks. Arbitrary strides
are a later API and must reject duplicate axes, overlapping writable addresses,
rank mismatches, overflow in address calculations, and in-place layouts whose
input/output representations differ.

### Element types

v0.1 accepts only `ComplexSIMD[DType.float32, 1]` and
`ComplexSIMD[DType.float64, 1]`. Plan dtype and buffer dtype match exactly;
execution never narrows, widens, or converts integer/real input implicitly.
Convenience conversion belongs in a higher-level numerical package or in a
future separately documented overload.

### Direction

`FFTDirection.forward()` uses the negative complex-exponent sign.
`FFTDirection.inverse()` uses the positive sign. Direction is nominal rather
than an integer or Boolean at the public boundary. A plan has one direction;
callers create or cache the opposite plan explicitly. This follows the clear
semantic split shared by all four references without adopting C-style signed
constants.

### Normalization

For transform length `n`, ShuhaFFT supports:

| Convention | Forward scale | Inverse scale |
| --- | ---: | ---: |
| `none()` | `1` | `1` |
| `backward()` | `1` | `1 / n` |
| `forward()` | `1 / n` | `1` |
| `ortho()` | `1 / sqrt(n)` | `1 / sqrt(n)` |

`backward()` remains the default. The kernel computes an unnormalized
transform and scaling occurs once at the plan boundary until profiling proves
that fusing it is useful. RustFFT, pocketfft, FFTW, and AbstractFFTs differ in
where normalization lives; ShuhaFFT makes it explicit so callers never need to
infer a backend convention.

Finite values receive the documented numerical accuracy. Non-finite input is
accepted as floating-point data and may propagate according to Mojo operations;
the library does not promise a particular NaN payload or exceptional-value
layout. Length, shape, aliasing, and resource errors are still validated even
when data contains non-finite values.

## Ownership and scratch space

`execute(values)` preserves `values` and returns independent owned storage.
`execute_in_place(values)` mutates only the supplied list. The current radix-2
kernel is allocation-free after the out-of-place copy and needs no public
scratch contract.

Future algorithms may require work space. RustFFT's split between an allocating
convenience call and execution with caller-provided scratch is adopted in
principle, with these ShuhaFFT rules:

- the plan separately reports `in_place_scratch_len()` and
  `out_of_place_scratch_len()` because placement requirements may differ;
- counts are measured in `ComplexSIMD[plan dtype, 1]` elements, may be zero,
  and describe the required prefix rather than an exact caller-list length;
- a caller may provide more elements than required; an empty list is valid when
  the reported count is zero and is never accessed;
- standard Mojo `List` element alignment is sufficient for every public CPU
  path; SIMD kernels must handle it or fall back internally rather than impose
  a hidden stronger alignment precondition;
- an allocating convenience execution may allocate the placement-specific
  amount;
- a no-allocation execution accepts caller-owned scratch and rejects an
  undersized buffer before mutating input, output, or scratch;
- scratch contents are unspecified after execution;
- scratch must not alias input or output unless an API explicitly permits it;
- input and output must not alias for the out-of-place method;
- an out-of-place input remains unchanged;
- hidden per-execution allocation is forbidden in an API documented as using
  caller-provided scratch;
- required scratch may change between library versions, so callers query the
  plan instead of hard-coding it.

The concrete caller-scratch surface is intentionally small:

```mojo
plan.execute_in_place_with_scratch(buffer, scratch)
plan.execute_into(input, output, scratch)
```

These are architectural method names rather than implemented signatures. Both
use caller-owned `List[ComplexSIMD[plan dtype, 1]]` storage. `execute_into`
requires output length exactly equal to the planned length and preserves input.
Both methods perform no allocation after plan construction. If Mojo cannot
prove or check the required non-aliasing for a proposed generic buffer type,
that overload is not added; the initial explicit-scratch API stays on a storage
type whose ownership contract can be enforced.

CPU plans own or share immutable twiddles, factorization, and selected-kernel
metadata required for repeated execution. Dropping an internal planner must not
invalidate a returned plan. Copying a CPU plan either deep-copies this immutable
state or shares it through an explicit safe owner; dropping or replacing one
copy never invalidates another. A CPU plan owns no mutable per-execution
workspace, counters, temporary buffers, or hidden stream state.

CPU execution is reentrant: the same plan and its copies may execute
concurrently when every input, output, and scratch region is disjoint. Scratch
is caller-owned per execution. If a future optimization needs mutation, it must
use local/caller storage or introduce an explicitly synchronized execution
object rather than making ordinary plan reuse history-dependent.

There is no unbounded global plan cache. A later cache, if benchmarks justify
one, is opt-in, bounded, keyed by the complete semantic and backend contract,
and exposes clear lifetime/reset behavior. pocketfft's disabled-by-default
bounded cache is a safer starting point than implicit process-wide wisdom.

## Algorithm selection

Selection evolves in correctness gates:

1. `n == 1`: identity kernel.
2. Power-of-two `n`: scalar radix-2 baseline; later measured radix-4 or split
   kernels may replace it behind the same plan.
3. General composite `n`: factorization plus independently tested small-radix
   butterflies and mixed-radix composition.
4. Prime or awkward `n`: direct DFT for very small sizes; Rader only after a
   primitive-root implementation and numerical fixtures exist; Bluestein for
   large-prime cases using a checked convolution length.
5. SIMD: select only when runtime capabilities, dtype, length, alignment, and
   kernel coverage all match; otherwise fall back to scalar.

RustFFT demonstrates recipe graphs and per-ISA planners. pocketfft demonstrates
a smaller alternative: estimate the factorized transform cost against a
Bluestein convolution of a nearby efficient length. ShuhaFFT should begin with
the latter style of deterministic cost model, record its thresholds in tests,
and grow a recipe graph only when composition requires it.

The selector must always have a correctness baseline. Unsupported acceleration
is not an error when automatic selection can run scalar code. A forced SIMD
selector remains internal to tests and benchmarks. A future public resource
policy requires its own API review and must raise rather than silently changing
an explicit caller request.

Planner heuristics are implementation details. Tests assert selected families
only at boundary cases needed to prevent catastrophic complexity; ordinary
tests assert mathematical results so tuning can change without breaking the
public API.

## CPU, SIMD, and GPU seams

The internal plan boundary should conceptually expose:

```text
PlannedKernel
  length
  dtype
  in-place scratch elements
  out-of-place scratch elements
  execute in place
  execute out of place
```

The concrete scalar and SIMD kernels implement that contract. CPU feature
detection and algorithm availability belong to the CPU planner, not to
direction or normalization values. SIMD modules may specialize butterflies and
data movement, but may not introduce a different sign, scale, ordering, or
failure contract.

GPU support is a separate backend with the same mathematical specification but
an explicit device-storage and execution model. It does not implement or extend
the CPU `FFTPlan`/`List` methods. A CPU `List` API must not silently copy to a
GPU. GPU support uses a distinct `GPUFFTPlan`-like resource-owning type and
device-buffer type, preferably in a separate package or isolated module that
CPU-only users never import.

GPU plan construction binds one explicit device/context and records compatible
stream or queue semantics. Device input, output, and workspace must belong to
that context; cross-device use raises before enqueue. GPU plans are non-copyable
by default. If an explicit `clone()` is later added, it shares a reference-
counted context and immutable plan resources, and dropping one clone cannot
invalidate another or any already-returned completion event. The final owner
releases resources through normal lifetime semantics without a global cleanup
call.

Execution names its stream/queue or uses the stream fixed at construction and
returns a completion event/fence for asynchronous work. A separate explicit
wait/synchronize operation reports deferred launch errors. No method presents
as synchronous unless it waits before returning. Device scratch/workspace size,
alignment, aliasing, lifetime through completion, and reuse-after-event rules
are queryable from the GPU plan rather than inferred from CPU scratch methods.
CPU-only users must not initialize an accelerator runtime or import a GPU
package.

This keeps GPU code from contaminating the CPU API while allowing higher-level
code to share direction, normalization, shape, and result conventions.

## Error and resource policy

Public fallibility uses `raises`. Invalid lengths, mismatched buffers,
unsupported dtypes, shape-product overflow, illegal axes/strides, insufficient
scratch, forbidden aliasing, incoherent plan/kernel/table state, unavailable
explicitly requested device resources, and resource-creation failures are
errors.

Every predictable contract error is checked before observable mutation. This
includes complete plan coherence, shape, dtype, placement, destination length,
scratch length/type/alignment, aliasing, and device/context compatibility. CPU
kernels have no expected fallible step after this validation; floating-point
NaN or overflow remains data behavior rather than an execution error.

An allocating `execute` never returns a partially valid owned result. If an
unexpected internal CPU failure occurs after in-place or caller-output mutation,
the mutated output and scratch contents are unspecified and the error is
reported; the out-of-place input remains unchanged. GPU enqueue failure before
work is accepted leaves buffers unchanged. A launch or asynchronous failure
reported after acceptance leaves destination and workspace contents unspecified
until the event completes or reports failure; out-of-place input preservation
still holds. These states are documented by each device API rather than hidden
behind a generic success value.

Programmer-facing assertions remain internal proofs after public validation;
they are not a substitute for rejecting reachable invalid public state. Mojo
1.0 fields that remain externally reachable are revalidated at every semantic
operation, as the current plan length already is.

Planning has no filesystem, environment, clock, randomness, or global-state
effect in the deterministic policy. A later measured policy must make its time
budget and input-preservation behavior explicit. Persistent tuning data would
require a versioned schema, CPU/compiler fingerprint, bounded storage, atomic
updates, provenance, and an opt-in I/O boundary; FFTW wisdom is therefore
rejected for the near-term design.

Allocation failure and accelerator resource errors propagate without returning
a partially valid plan. Plans release owned resources through the CPU or GPU
lifetime rules above. No public `cleanup()` invalidates unrelated live plans.

## Adopted and rejected ideas

### Adopted

- From RustFFT: a plan selects a kernel, can be reused independently of the
  planner, and reports placement-specific scratch requirements; CPU ISA
  planners stay behind one semantic execution contract.
- From pocketfft: validate shape/stride/axis relationships together, keep plan
  caching bounded and optional, compute twiddles carefully, and compare a
  factorized algorithm with Bluestein through a deterministic cost model.
- From FFTW: distinguish the transform problem from candidate solvers; make
  plan reuse, placement, alignment, and resource lifetime explicit; treat
  planning effort as policy rather than transform semantics.
- From AbstractFFTs.jl: implement convenience calls through plans, require plan
  shape/type applicability, support preallocated output, and keep scaling
  composable with an unnormalized kernel.

### Rejected or deferred

- RustFFT's public algorithm constructors, public ISA-specific planners, and
  panic-based shape errors are not adopted. ShuhaFFT keeps planners and
  algorithms internal and reports reachable input errors with `raises`.
- pocketfft's Boolean direction, raw byte strides, silent thread-count fallback,
  and single header organization are not a public Mojo model.
- FFTW's GPL source is evidence only. Its global planner, mutable planning
  buffers, default measured planning, wisdom, manual cleanup, raw pointers,
  bitwise flags, and nullable plan failure are rejected for the deterministic
  foundation.
- AbstractFFTs.jl's operator-overloaded plan application, implicit numeric
  promotion/copying, multidimensional default, and inverse-plan mutation are
  not adopted. Named methods and explicit ownership are clearer for Mojo.
- User-selected algorithms, arbitrary strides, threads, real transforms,
  multidimensional transforms, and GPU execution remain deferred until each
  lower-level contract has correctness and benchmark evidence.

## Numerical verification

Every algorithm and backend must pass the same semantic suite before selection
is enabled.

### Reference tests

- exact or analytic transforms: singleton, origin delta, shifted delta,
  constants, alternating signs, and individual Fourier modes;
- small direct-DFT oracle implemented independently from FFT kernels for all
  direction/normalization combinations;
- committed external fixtures only when generated by a documented primary
  implementation, with generator version, exact revision, dtype, convention,
  checksum, and tolerance recorded;
- Float32 and Float64 cases at every supported algorithm boundary.

### Properties and invariants

- forward/inverse round trips with the expected factor for all four
  normalization conventions;
- Parseval/energy identities adjusted for the selected normalization;
- linearity, circular-shift phase, conjugation, and repeat-plan determinism;
- in-place and out-of-place agreement;
- input preservation and independent output ownership;
- scratch and plan reuse without history-dependent results;
- coherent rejection of every reachable plan/kernel/twiddle mismatch before
  buffer mutation;
- concurrent CPU-plan reuse with disjoint input/output/scratch storage;
- exact output length and no mutation after rejected validation;
- scalar/SIMD/GPU agreement within dtype- and size-specific tolerances;
- later RFFT Hermitian symmetry, even/odd original-length handling, and packed
  shape tests.

Tolerance scales with precision, length, and the numerical path. The current
starting tolerances are `1e-5` absolute/relative for Float32 and `1e-12` for
Float64 reference cases. Tests may use a documented `O(epsilon * log(n))` or
`O(epsilon * n)` bound where appropriate; tolerances are never relaxed solely
to make a new optimized backend pass.

Boundary tests include zero length, non-power-of-two lengths while radix-2 is
the only algorithm, mismatched input, mutated plan state, extreme finite
magnitudes that avoid reference overflow, subnormal-scale inputs where the
platform supports them, and non-finite propagation without NaN-payload claims.

## Benchmark design

Benchmarks separate costs that references sometimes combine:

- plan construction, including factorization and twiddle generation;
- cold first execution and warmed repeated execution;
- allocating convenience, owned out-of-place, in-place, and caller-scratch
  execution;
- Float32 and Float64;
- scalar versus every enabled SIMD/backend path;
- sizes around algorithm thresholds, not only favorable powers of two;
- later smooth composite, prime, and Bluestein convolution sizes;
- batches and dimensions only after those APIs exist.

Each run records exact ShuhaFFT commit, Mojo/compiler options, CPU and detected
ISA, OS, backend/device, data alignment and shape, warmup, sample count,
statistic, and timer. Report throughput and latency without excluding planning
cost unless the result is clearly labeled execution-only. Correctness is checked
outside the timed region. Methodology and runnable programs are committed;
machine-specific results are evidence, not permanent performance claims.

Cross-library comparisons use each primary library's documented convention and
record any normalization or allocation performed outside its timed call. A
comparison is omitted when license, build, alignment, threading, or planning
differences cannot be made explicit.

## Dependency-ordered issue sequence

Issues should be opened and implemented in this order. A later gate does not
begin until the prior gate is validated on the supported CI matrix.

1. **Complete v0.1 numerical confidence.** Add the independent small direct-DFT
   oracle, full normalization round trips, ownership/rejection invariants, and
   installed-package verification. Keep the current scalar radix-2 API.
2. **Record a scalar baseline.** Add benchmark programs and methodology for
   current plan creation and execution without claiming superiority.
3. **Extract an internal kernel contract.** Separate validated specification
   from scalar execution without a public API change; move future plan metadata
   into one coherent state and prove corrupted reachable fields cannot bypass
   buffer validation or kernel bounds safety.
4. **Add immutable twiddle planning.** Measure recurrence drift against
   precomputed, symmetry-reduced twiddles before selecting a representation;
   validate table/spec/kernel agreement before every first table access.
5. **Add explicit scratch execution.** Introduce separate in-place/out-of-place
   scratch queries and the two caller-provided execution methods only when an
   algorithm needs workspace; test zero/excess/undersized counts, alignment,
   every forbidden alias, no hidden allocation, and pre-mutation rejection.
6. **Add scalar general composite lengths.** Land small butterflies and
   mixed-radix composition incrementally, each behind direct-DFT and property
   tests.
7. **Add Bluestein for awkward lengths.** Implement checked convolution-size
   selection and reuse the trusted FFT kernel; consider Rader only with
   separate evidence.
8. **Introduce the deterministic planner.** Select among the now-real scalar
   families, cache recipes within explicit planner ownership, and test only
   meaningful selection boundaries.
9. **Add convenience `fft`/`ifft`.** Route through the deterministic default
   planner after ownership and allocation behavior are stable.
10. **Add CPU SIMD backends.** One ISA/dtype family at a time, with runtime
    detection, scalar fallback, cross-backend properties, and measured wins.
11. **Add real transforms.** Specify even/odd output shapes and original-length
    recovery before RFFT/IRFFT kernels.
12. **Add batches, axes, and multidimensional plans.** Establish checked shape,
    stride, aliasing, and ordering semantics before implementation.
13. **Evaluate a separate GPU backend.** Require distinct plan/buffer types and
    explicit context, stream, copy/drop, workspace, event, synchronization, and
    failure semantics; do not alter or initialize the CPU path.
14. **Evaluate measured planning and persistent tuning last.** Proceed only if
    deterministic heuristics leave material measured performance and the I/O,
    reproducibility, privacy, and invalidation policies are solved.

This order preserves a trusted scalar oracle beneath every optimization and
keeps planner complexity proportional to algorithms that actually exist.
