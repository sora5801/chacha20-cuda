# 0001 — Initial ChaCha20 CUDA implementation

**Push date:** 2026-06-25
**Status:** Verified — all RFC 8439 known-answer tests and GPU/CPU cross-checks
pass on an NVIDIA RTX 2080 SUPER (CUDA 13.3, Visual Studio 2026).

---

## Summary

This is the first push: a complete, heavily-commented, didactic implementation
of the **ChaCha20 stream cipher (RFC 8439)** in **CUDA C++**, together with an
independent CPU reference, a known-answer test suite, a live demo with a
throughput benchmark, and both a Visual Studio solution and a command-line
build script.

The project is intended as CUDA C++ study material, so every file is commented
far beyond what production code would carry — explaining the cryptography, the
CUDA execution model, the memory layout, and the reasoning behind each choice.

---

## What was added

### Core cipher

- **`include/chacha20.cuh`** — the single public header. Defines the fixed sizes
  (32-byte key, 12-byte nonce, 64-byte block), the four `"expand 32-byte k"`
  constants, the `ChaCha20State` 4×4-matrix struct, and the prototypes for the
  host-callable API. Includes a long preamble explaining ChaCha20 and *why*
  counter mode is ideal for GPUs.
- **`src/chacha20.cu`** — the GPU implementation:
  - `ROTL32` / `QUARTERROUND` — the ARX primitives as device-side helpers.
  - `chacha20_kernel` — the `__global__` kernel: **one 64-byte block per
    thread**, 20 rounds in registers, feed-forward, little-endian
    serialization, byte-wise XOR with a correct partial-final-block path, and a
    grid-overhang guard.
  - `chacha20_init_state` — host-side builder for the initial state
    (little-endian, portable across CPU endianness).
  - `chacha20_xor_device` — host launcher that chooses the grid/block
    configuration and launches on a given stream (device pointers in/out).
  - `chacha20_xor_cuda` — host convenience wrapper that owns the full GPU memory
    lifecycle (allocate → H2D copy → launch → sync → D2H copy → free on every
    path, including errors).

### Independent reference (the verification oracle)

- **`src/chacha20_reference.h` / `src/chacha20_reference.cpp`** — a separate,
  deliberately simple, single-threaded, pure-C++ ChaCha20. It shares **no code**
  with the GPU version, so when the two agree we have strong evidence both are
  correct. It is also the best place to learn the algorithm without any GPU
  concepts in the way.

### Tests

- **`tests/test_vectors.cu` / `.cuh`** — the correctness suite, run before the
  demo:
  - RFC 8439 §2.1.1 quarter-round known answer.
  - RFC 8439 §2.3.2 block-function keystream (checked on **both** the CPU oracle
    and the GPU, the latter by encrypting 64 zero bytes).
  - RFC 8439 §2.4.2 full "sunscreen" encryption vector (114 bytes → exercises
    the partial final block) on the GPU.
  - Encrypt→decrypt round trip.
  - GPU-vs-CPU agreement across lengths `{1, 63, 64, 65, 127, 128, 200, 4096,
    1 MiB}` (every block boundary).
  - In-place encryption (output buffer aliases input).

### Demo

- **`demo/demo.cu` / `.cuh`** — two acts:
  1. Encrypt a readable message, print the ciphertext as hex, decrypt it back,
     and verify the round trip.
  2. A **kernel-only throughput benchmark** (256 MiB × 50 iterations) timed with
     CUDA events, reporting GB/s and blocks/second, with PCIe copies excluded.

### Entry point

- **`src/main.cu`** — checks for a CUDA device, runs the test suite (aborting
  with a non-zero exit code on any failure), then runs the demo.

### Build system

- **`vs/ChaCha20CUDA.sln`, `.vcxproj`, `.vcxproj.filters`** — a Visual Studio
  2026 solution wired to the **CUDA 13.3** build customization, PlatformToolset
  **v145**, x64, default target `compute_75,sm_75`. Open and press Ctrl+F5.
- **`build.ps1`** — a one-command command-line build. It imports the MSVC
  developer environment and invokes `nvcc`, producing
  `build/chacha20_demo.exe`. Supports `-Arch` and `-Run`.

### Project docs

- **`README.md`** — full didactic overview: the algorithm, the CUDA mapping,
  layout, build/run instructions, performance and security notes.
- **`LICENSE`** — MIT, with an explicit "this is study code, not for production
  secrets" note.
- **`.gitignore`** — keeps build artifacts (objects, exes, nvcc temporaries, VS
  output dirs) out of the repo.
- **`docs/README.md`** — describes this per-push change-log convention.

---

## Why these choices

- **One thread per block (counter mode).** ChaCha20 blocks are independent, so
  this is the simplest mapping that is also the most parallel — no shared
  memory, no synchronization. It is the clearest demonstration of why counter
  mode and GPUs fit together.
- **State passed by value to the kernel.** The 64-byte state lands in parameter
  space; every thread gets the key/nonce/constants for free and only customizes
  its counter. No global-memory key fetch.
- **A second, independent CPU implementation.** Shared helper code would let a
  single bug hide in both paths. Independence turns "they agree" into real
  evidence of correctness.
- **Clarity over peak speed.** The byte-wise serialization loop is easy to read
  and obviously correct for partial blocks; the README lists vectorization
  (`uint4`) and stream pipelining as explicit next-step exercises.

---

## Verification

Built and run two ways on the reference machine (RTX 2080 SUPER, CUDA 13.3,
VS 2026):

- `build.ps1 -Run` (nvcc command-line build), and
- MSBuild on `vs/ChaCha20CUDA.sln` (the IDE path).

Both produce a binary that reports **`ALL TESTS PASSED`** and a benchmark of
**~24.6 GB/s** kernel-only throughput.

---

## Next ideas (not yet done)

- Vectorized (`uint4`) load/XOR/store path for full blocks.
- ChaCha20-Poly1305 AEAD (authentication).
- Multi-stream pipelining to overlap PCIe transfers with compute.
- Grid-stride kernel variant for arbitrarily large buffers with a fixed grid.
