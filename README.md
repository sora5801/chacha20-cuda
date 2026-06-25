# ChaCha20 in CUDA C++

A heavily-commented, **didactic** implementation of the [ChaCha20 stream
cipher](https://www.rfc-editor.org/rfc/rfc8439) (RFC 8439) written in CUDA C++,
built as study material for learning GPU programming through a real, useful,
correctness-verified algorithm.

Every source file is commented to excess on purpose: the goal is not just a
working cipher, but a file you can *read top to bottom* and come away
understanding both the cryptography and the CUDA. If you have never written a
GPU kernel, you can learn the algorithm from the plain-C++ reference first
([`src/chacha20_reference.cpp`](src/chacha20_reference.cpp)) and then see "the
same thing, one block per thread" in the kernel
([`src/chacha20.cu`](src/chacha20.cu)).

> **Status:** Verified against the official RFC 8439 known-answer vectors and
> cross-checked GPU-vs-CPU on a real GPU (NVIDIA RTX 2080 SUPER, CUDA 13.3).
> All tests pass. See [Running it](#running-it).

> **Not for production secrets.** This is teaching code: correct, but
> unaudited and *unauthenticated*. See [Security notes](#security-notes).

---

## Table of contents

- [Why ChaCha20 on a GPU?](#why-chacha20-on-a-gpu)
- [The algorithm in five minutes](#the-algorithm-in-five-minutes)
- [How it maps onto CUDA](#how-it-maps-onto-cuda)
- [Repository layout](#repository-layout)
- [Building](#building)
- [Running it](#running-it)
- [Performance notes](#performance-notes)
- [Security notes](#security-notes)
- [The "document every push" convention](#the-document-every-push-convention)
- [References](#references)

---

## Why ChaCha20 on a GPU?

ChaCha20 runs in **counter mode**: the 64 bytes of keystream for block *N*
depend only on `(key, nonce, N)` — never on block *N − 1*. That independence is
the whole game for a GPU:

```
block 0   -> thread 0     (counter = base + 0)
block 1   -> thread 1     (counter = base + 1)
block 2   -> thread 2     (counter = base + 2)
   ...         ...
block k   -> thread k     (counter = base + k)
```

A million blocks become a million independent threads with **zero
communication, zero shared memory, zero `__syncthreads()`**. This is the
"embarrassingly parallel" ideal, and it is why stream ciphers in counter mode
(ChaCha20, AES-CTR) are a natural first real-world CUDA kernel. Compare that to
AES-CBC, where block *N* needs block *N − 1*'s output and is therefore stuck
being serial.

---

## The algorithm in five minutes

ChaCha20 is a **stream cipher**: it turns a key + nonce into a long
pseudo-random **keystream**, and encryption is just XOR:

```
ciphertext = plaintext  XOR keystream
plaintext  = ciphertext XOR keystream      (XOR is its own inverse)
```

So **encryption and decryption are the same operation** — this library exposes
one `xor` function that does both.

The keystream is produced 64 bytes at a time by the **block function**, which
operates on a **4×4 matrix of 32-bit words** (the "state"):

```
 +----------+----------+----------+----------+
 | const 0  | const 1  | const 2  | const 3  |   row 0: "expand 32-byte k"
 +----------+----------+----------+----------+
 | key 0    | key 1    | key 2    | key 3    |   row 1: key words 0..3
 +----------+----------+----------+----------+
 | key 4    | key 5    | key 6    | key 7    |   row 2: key words 4..7
 +----------+----------+----------+----------+
 | counter  | nonce 0  | nonce 1  | nonce 2  |   row 3: counter + 96-bit nonce
 +----------+----------+----------+----------+
```

- **Constants** are the ASCII bytes of `"expand 32-byte k"` — fixed, public,
  "nothing-up-my-sleeve" numbers that diversify the state.
- **Key** is 256 bits (32 bytes), read as eight little-endian words.
- **Counter** is a 32-bit block index (which 64-byte chunk we are on).
- **Nonce** is 96 bits (12 bytes), unique per message under a given key.

The state is stirred by the **quarter round**, the cipher's only moving part.
It uses just **A**dd, **R**otate, **X**or (so-called **ARX**) — no S-boxes, no
table lookups, no data-dependent branches (hence fast, GPU-friendly, and
naturally constant-time):

```
a += b;  d ^= a;  d <<<= 16;     // <<<= n means "rotate left by n bits"
c += d;  b ^= c;  b <<<= 12;
a += b;  d ^= a;  d <<<=  8;
c += d;  b ^= c;  b <<<=  7;
```

One full round applies the quarter round four times. ChaCha20 alternates:

- **column rounds** — stir the four vertical columns, then
- **diagonal rounds** — stir the four diagonals (spreading each column's
  changes across the whole matrix).

That alternation is what diffuses every input bit to every output word.
**20 rounds = 10 (column + diagonal) "double rounds"** — that "20" is the name.

Finally, the original state is **added back** into the stirred state (the
"feed-forward", which makes the function non-invertible), and the 16 words are
serialized little-endian into 64 keystream bytes.

The cleanest place to read all of this as ordinary code is the reference
implementation: [`src/chacha20_reference.cpp`](src/chacha20_reference.cpp).

---

## How it maps onto CUDA

The kernel in [`src/chacha20.cu`](src/chacha20.cu) assigns **one 64-byte block
to one thread**:

1. `block_index = blockIdx.x * blockDim.x + threadIdx.x` — the global thread id
   *is* the block number.
2. Each thread copies the shared initial state (passed **by value** as a kernel
   argument, so no global-memory read is needed to fetch the key) and sets
   **only its own counter word** to `base_counter + block_index`.
3. It runs the 20 rounds entirely in **registers** (16 × `uint32_t` = 64 bytes).
4. It serializes its keystream and XORs it with its slice of the input.

Threads that fall off the end of the buffer return early (the "grid-overhang"
guard), and the final partial block (when the length is not a multiple of 64)
is handled by a single `min(64, remaining)` — no padding, no special case.

Two API layers are provided (see [`include/chacha20.cuh`](include/chacha20.cuh)):

| Function | Use it when |
|---|---|
| `chacha20_xor_cuda(...)` | You have **host** buffers and want it to "just work" — it allocates GPU memory, copies in/out, launches, and frees for you. |
| `chacha20_xor_device(...)` | Your data is **already on the device** and you want pure kernel performance with no PCIe copies (used by the benchmark). |

---

## Repository layout

```
CHACHA20/
├── include/
│   └── chacha20.cuh            # Public API, constants, state struct (start here)
├── src/
│   ├── chacha20.cu             # The CUDA kernel + host launchers (the GPU code)
│   ├── chacha20_reference.h    # Plain-C++ reference (oracle) interface
│   ├── chacha20_reference.cpp  # Plain-C++ reference: learn the algorithm here
│   └── main.cu                 # Entry point: run tests, then the demo
├── tests/
│   ├── test_vectors.cuh
│   └── test_vectors.cu         # RFC 8439 known-answer tests + GPU/CPU cross-checks
├── demo/
│   ├── demo.cuh
│   └── demo.cu                 # Encrypt/decrypt showcase + throughput benchmark
├── vs/
│   ├── ChaCha20CUDA.sln        # Visual Studio 2026 solution
│   ├── ChaCha20CUDA.vcxproj    # ... project (CUDA 13.3 build integration)
│   └── ChaCha20CUDA.vcxproj.filters
├── docs/
│   ├── README.md               # The change-log convention
│   └── 0001-initial-implementation.md
├── build.ps1                   # One-command command-line build (nvcc)
├── .gitignore
├── LICENSE                     # MIT (+ a note that this is study code)
└── README.md                   # You are here
```

**Suggested reading order for study:**
`include/chacha20.cuh` → `src/chacha20_reference.cpp` → `src/chacha20.cu` →
`tests/test_vectors.cu` → `demo/demo.cu`.

---

## Building

You need an **NVIDIA GPU**, the **CUDA Toolkit** (developed against 13.3), and
**Visual Studio 2026** (the reference setup). Adjust the target GPU
architecture to match your card — the project defaults to `sm_75` (Turing,
RTX 20-series). For other cards: Ampere `sm_86`, Ada `sm_89`, Hopper `sm_90`.

### Option A — Visual Studio (IDE)

1. Open `vs/ChaCha20CUDA.sln`.
2. Pick the **Release / x64** configuration.
3. Press **Ctrl+F5** (*Start Without Debugging*, so the console stays open).

To change the target GPU: Project → Properties → CUDA C/C++ → Device →
*Code Generation*, e.g. `compute_86,sm_86`.

### Option B — command line (nvcc)

From a normal PowerShell prompt (the script sets up the MSVC environment for
you):

```powershell
.\build.ps1            # build for sm_75 -> .\build\chacha20_demo.exe
.\build.ps1 -Run       # build, then run
.\build.ps1 -Arch sm_86 -Run   # target a different GPU
```

---

## Running it

The program runs the **test suite first** (it refuses to demo a cipher it has
not proven correct), then the **demo**. Real output on the reference machine:

```
##################################################################
#           ChaCha20 stream cipher, implemented in CUDA          #
##################################################################
Using GPU 0: NVIDIA GeForce RTX 2080 SUPER (sm_75)

 ChaCha20 CUDA -- correctness test suite (RFC 8439 known answers)
  [ PASS ] RFC 8439 2.1.1  quarter-round known answer
  [ PASS ] RFC 8439 2.3.2  block keystream (CPU oracle)
  [ PASS ] RFC 8439 2.3.2  block keystream (GPU kernel)
  [ PASS ] RFC 8439 2.4.2  full encryption (GPU)
  [ PASS ] Round trip  encrypt then decrypt recovers plaintext
  [ PASS ] GPU vs CPU oracle  (lengths 1..1MiB, all block boundaries)
  [ PASS ] In-place  encrypt-in-place changes then restores buffer
 RESULT: ALL TESTS PASSED
   ...
 DEMO 2: Kernel-only throughput benchmark
  Throughput       : 24.58 GB/s
```

The tests check three independent things:

1. **RFC 8439 known answers** — our output equals the bytes the entire world
   agrees on (the standard's quarter-round, block-function, and encryption
   vectors). Passing means we implemented the *actual* standard.
2. **GPU vs CPU oracle** — the parallel kernel and the independent
   single-threaded reference agree byte-for-byte across many awkward lengths
   (1, 63, 64, 65, 127, 128, 200, 4096, 1 MiB), stressing every block boundary.
3. **Round trips & in-place** — encrypt-then-decrypt recovers the original, and
   `output == input` aliasing is safe.

---

## Performance notes

The reference run hits **~24.6 GB/s** of plaintext on an RTX 2080 SUPER, with
**kernel-only timing** (PCIe copies excluded). This kernel is **memory-bound**,
not compute-bound: it reads `len` bytes and writes `len` bytes, and the 20
rounds of ARX are cheap next to that traffic.

It is written for **clarity first**. Deliberate didactic simplifications that
also leave performance on the table (good exercises if you want to go faster):

- **Byte-wise load/XOR/store.** The serialization loop touches memory one byte
  at a time. Vectorizing to `uint4` (16 bytes/transaction) for full blocks
  would dramatically improve memory coalescing and throughput; only the partial
  final block needs the byte path.
- **One block per thread.** Fine here, but very large buffers can also use a
  grid-stride loop so a fixed-size grid covers any length.
- **No `__restrict__`-driven overlap of compute and copy.** A streaming version
  could pipeline H2D copy, kernel, and D2H copy across CUDA streams to hide PCIe
  latency.

These are intentionally left as next steps — the comments in `src/chacha20.cu`
point them out where relevant.

---

## Security notes

This is **study code**. It is correct against the spec but:

- **No authentication.** A real protocol must detect tampering. Pair ChaCha20
  with Poly1305 (the AEAD construction **ChaCha20-Poly1305**, RFC 8439 §2.8)
  so modified ciphertext is rejected. This project implements the cipher only.
- **Never reuse a (key, nonce) pair.** Two messages encrypted under the same
  key *and* nonce are XORed against the *same* keystream, which catastrophically
  breaks confidentiality. The nonce must be unique for every message.
- **Unaudited.** No side-channel review, no constant-time guarantees beyond
  ARX's inherent branch-freedom, no key-zeroization.
- For anything real, use **libsodium**, **BoringSSL**, or your platform's vetted
  crypto. Don't ship your own.

---

## The "document every push" convention

Every time something new is pushed to this repository, a numbered Markdown file
is added under [`docs/`](docs/) explaining **what changed and why** — a
human-readable changelog that grows alongside the code. The first entry,
[`docs/0001-initial-implementation.md`](docs/0001-initial-implementation.md),
documents this initial implementation. See [`docs/README.md`](docs/README.md)
for the convention.

---

## References

- **RFC 8439** — *ChaCha20 and Poly1305 for IETF Protocols*:
  <https://www.rfc-editor.org/rfc/rfc8439>
- D. J. Bernstein — *ChaCha, a variant of Salsa20*:
  <https://cr.yp.to/chacha/chacha-20080128.pdf>
- NVIDIA CUDA C++ Programming Guide:
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>

---

*Built as one of many CUDA C++ study projects. Read the comments — that's where
the learning is.*
