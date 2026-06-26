# 0002 — Ultracode adversarial audit pass

**Push date:** 2026-06-25
**Status:** Verified — all RFC 8439 tests still pass on an NVIDIA RTX 2080 SUPER
in **both** the command-line (nvcc) and Visual Studio (Debug + Release) builds;
kernel throughput preserved at ~23 GB/s.

---

## Summary

The initial implementation (push 0001) was written and verified in Claude
Code's standard mode — single-agent, no multi-agent orchestration. This push
adds the layer that "Ultracode" mode would have applied from the start: a
**multi-agent adversarial audit** of the existing code, and the fixes that
survived that audit.

Six independent reviewer agents each audited one dimension (crypto correctness,
CUDA safety/UB, comment factual accuracy, build system, docs/completeness,
performance claims). Every finding was then handed to **two independent skeptic
verifiers** instructed to *refute* it; only findings that survived verification
were kept. Of 11 raw findings, **7 survived** (6 unanimous).

A notable result: the audit found **no cryptographic correctness bug** — the
cipher itself was already correct, consistent with the passing RFC 8439 vectors.
Every confirmed finding was a **teaching-accuracy or safety/robustness** issue,
which is exactly the class that matters most for study material whose comments
are the deliverable.

> Transparency note: the audit run hit the account's session usage limit near
> the end, which cut short the build-system / docs / perf verifier panels and
> the auto-synthesis step. The three most important dimensions (crypto, CUDA
> safety, comment accuracy) completed their full review + double-verification;
> the remaining findings were adjudicated manually.

---

## What was fixed

### 1. `__restrict__` aliasing contract (the headline finding)

**Problem.** `chacha20_kernel` marks `in`/`out` as `__restrict__`, but the code
and comments advertised in-place use (`out == in`). Passing aliasing pointers to
`__restrict__` parameters is **undefined behavior** in C/C++, so the comment
"in-place is still correct" was technically false. (No in-repo path actually
triggered it — `chacha20_xor_cuda` already stages through two separate device
buffers — but the low-level API left the contract undocumented.)

**The interesting part — why I did NOT take the audit's preferred fix.** Both
verifiers recommended *dropping* `__restrict__`, asserting it "costs nothing
measurable" for a memory-bound kernel. I applied that, rebuilt, and measured:
throughput **collapsed from ~24.6 GB/s to 4.2 GB/s — a ~6× regression**. Without
`__restrict__`, the compiler must assume each store to `out` may alias the next
load from `in`, so it cannot widen/vectorize the byte serialization loop. The
audit's confident "free" claim was simply wrong for this loop.

**Fix actually applied (the audit's option B).** Keep `__restrict__` for the 6×
performance, and make the *contract* honest instead:
- Kernel comment now states `in`/`out` **must not alias**, explains the measured
  6× cost of removing the qualifier, and notes that in-place is supported at the
  library level because `chacha20_xor_cuda` double-buffers on the device.
- The low-level `chacha20_xor_device` header now documents the no-alias
  requirement and points in-place callers to the high-level API.
- The in-place test comment now explains *why* it is safe (host-level aliasing,
  separate device buffers) rather than implying the kernel handles aliasing.

**Lesson (very much in the spirit of this repo): measure, don't trust.** A
plausible, unanimously "verified" recommendation was a 6× regression. Rebuilding
and benchmarking caught it.

### 2. Benchmark ignored CUDA error codes
`demo_benchmark` discarded the `cudaError_t` of `cudaMemset`, `cudaEventCreate`,
the warm-up launch, and `cudaEventElapsedTime` — contradicting the project's own
headline lesson (chacha20.cu's `CUDA_CHECK` rationale). Now each is checked, with
the warm-up launch and the elapsed-time gating the validity of the printed
numbers (and a guard so a timing failure can no longer print an "inf" GB/s).

### 3. `cudaSetDevice(0)` return discarded
`main.cu` now checks the bind step (the call that decides which GPU every later
kernel runs on), matching the surrounding error-handling discipline.

### 4. "Round function" vs "block function" (a real conceptual error)
A comment claimed the feed-forward makes "the round function non-invertible."
The 20 rounds are in fact **fully invertible** (add/xor/rotate are each
reversible); it is the **block function** (state → keystream) that becomes
one-way, because the unknown original state is summed in. Reworded to attribute
one-wayness correctly (the Davies–Meyer construction).

### 5. Visual Studio Debug comment said `-G` but emitted `-lineinfo`
The Debug `CudaCompile` group set `<GenerateLineInfo>` (nvcc `-lineinfo`,
source-line info only) while the comment promised `-G` "step into kernels"
debugging — two different, mutually exclusive flags. Switched Debug to
`<GPUDebugInfo>true</GPUDebugInfo>` (real `-G`: kernel breakpoints, single-step,
unoptimized device code) and corrected the comment. Release keeps optimized
device code.

### 6. Incomplete intermediate-output comment in the vcxproj
The comment claimed all intermediates land in `$(IntDir)`; the CUDA build
customization actually stages `.cu.obj` device objects under
`vs\ChaCha20CUDA\x64\<Config>\`. Comment corrected (`.gitignore` already covered
both locations).

### 7. README `uint4` "coalescing" framing
Clarified that switching the serialization to `uint4` cuts transaction/
instruction count but does **not** by itself produce warp-level coalescing under
the one-block-per-thread mapping (adjacent threads stay 64 bytes apart); full
coalescing needs a mapping change such as a `uint4`-granular grid-stride layout.

---

## Files touched

- `src/chacha20.cu` — `__restrict__` contract comment; feed-forward
  invertibility comment.
- `include/chacha20.cuh` — low-level API no-alias documentation.
- `tests/test_vectors.cu` — in-place test rationale comment.
- `demo/demo.cu` — error checking in the benchmark path.
- `src/main.cu` — `cudaSetDevice` return check.
- `vs/ChaCha20CUDA.vcxproj` — Debug `-G` via `GPUDebugInfo`; intermediate-output
  comment.
- `README.md` — `uint4` coalescing nuance.

---

## Verification

- `build.ps1 -Run` (nvcc, Release): **ALL TESTS PASSED**, throughput **23.14 GB/s**
  (statistically unchanged from 0001's 24.58; the dip-to-4.2 happened only on the
  abandoned drop-`__restrict__` attempt).
- MSBuild `Debug|x64` (now `-G`): builds; **ALL TESTS PASSED** at runtime.
- MSBuild `Release|x64`: builds.

No cipher output changed (the section 2.4.2 ciphertext is byte-identical to
0001); every change here is a comment, a contract clarification, an error check,
or a Debug-only build flag.

---

## Audit methodology (for reference)

- Orchestrated with the Workflow tool: `review → adversarially verify → synthesize`.
- 6 review dimensions, fan-out in parallel.
- Each finding verified by 2 independent skeptics with distinct lenses
  (correctness; impact/fix-soundness), defaulting to "refuted" unless
  independently confirmed.
- 29 agents, ~187 tool calls. Findings adjudicated and applied in the main loop,
  then re-verified by rebuilding and re-running on hardware.
