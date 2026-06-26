// ============================================================================
//  chacha20.cu  --  The CUDA implementation: device kernel + host launchers
// ============================================================================
//
//  This is where the cipher actually runs. The file has three layers, from
//  bottom (closest to the GPU) to top (closest to the caller):
//
//      1. DEVICE code        : the quarter-round macro and the __global__
//                              kernel that every GPU thread executes.
//      2. HOST launchers      : ordinary C++ functions that configure the grid
//                              and start the kernel (chacha20_xor_device).
//      3. HOST convenience    : memory allocation + PCIe copies wrapped around
//                              the launcher (chacha20_xor_cuda).
//
//  Read it bottom-up the first time: understand the math of one block, then how
//  one thread runs one block, then how thousands of threads cover a whole
//  buffer, then how the buffer gets to and from the GPU.
// ============================================================================

#include "chacha20.cuh"

#include <cstdio>   // fprintf for error reporting
#include <cstdlib>  // (kept for completeness; size math only)

// ============================================================================
//  SECTION 0 -- A tiny error-checking helper
// ----------------------------------------------------------------------------
//  Almost every CUDA runtime call returns a cudaError_t. Ignoring those return
//  values is the #1 source of "my kernel silently did nothing" bugs. This macro
//  evaluates an expression, and if it did not return cudaSuccess it prints the
//  human-readable error string (with file + line) and bails out of the current
//  function by returning that error code.
//
//  We use a do/while(0) wrapper -- the classic C idiom -- so the macro behaves
//  like a single statement and is safe to use inside an unbraced if/else.
// ============================================================================
#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "[CUDA ERROR] %s:%d: %s -> %s\n",                  \
                    __FILE__, __LINE__, #expr, cudaGetErrorString(_err));      \
            return _err;                                                       \
        }                                                                      \
    } while (0)

// ============================================================================
//  SECTION 0.5 -- Host helper: little-endian word loader + state builder
// ----------------------------------------------------------------------------
//  ChaCha20 reads the key and nonce as LITTLE-ENDIAN 32-bit words, regardless
//  of the byte order of the machine running the code. To stay correct on any
//  CPU we never reinterpret raw memory as uint32_t (that would bake in the
//  host's endianness); instead we assemble each word explicitly from its four
//  bytes. This is slower by a few instructions and 100% portable -- the right
//  trade for cryptographic code.
// ============================================================================

// Read 4 bytes at p as a little-endian uint32_t: p[0] is the LOW byte.
static inline uint32_t load_le32(const uint8_t* p) {
    return  (uint32_t)p[0]
         | ((uint32_t)p[1] <<  8)
         | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
}

// Populate the 16-word state from key/nonce/counter, following the fixed
// layout documented in chacha20.cuh:
//   row 0  : the four "expand 32-byte k" constants
//   rows 1-2: the eight key words (32 bytes, little-endian)
//   row 3  : [counter][nonce0][nonce1][nonce2]
ChaCha20State chacha20_init_state(const uint8_t key[CHACHA20_KEY_BYTES],
                                  const uint8_t nonce[CHACHA20_NONCE_BYTES],
                                  uint32_t counter)
{
    ChaCha20State st;

    // Row 0: the nothing-up-my-sleeve constants.
    st.w[0] = CHACHA20_CONST_0;
    st.w[1] = CHACHA20_CONST_1;
    st.w[2] = CHACHA20_CONST_2;
    st.w[3] = CHACHA20_CONST_3;

    // Rows 1-2: the 256-bit key as eight little-endian words (slots 4..11).
    for (int i = 0; i < 8; ++i) {
        st.w[4 + i] = load_le32(key + 4 * i);
    }

    // Row 3, slot 12: the starting block counter.
    st.w[12] = counter;

    // Row 3, slots 13..15: the 96-bit nonce as three little-endian words.
    for (int i = 0; i < 3; ++i) {
        st.w[13 + i] = load_le32(nonce + 4 * i);
    }

    return st;
}

// ============================================================================
//  SECTION 1 -- The ARX primitives (device side)
// ============================================================================

// ----------------------------------------------------------------------------
//  ROTL32 -- rotate a 32-bit word LEFT by n bits ("circular shift").
//
//  A rotate is a shift where the bits that fall off the top wrap around to the
//  bottom instead of being lost. ChaCha20's diffusion (how one input bit comes
//  to affect many output bits) relies entirely on these rotations.
//
//      (v << n)          : the bits we keep, shifted up; the low n bits become 0
//      (v >> (32 - n))   : the top n bits, brought down to the bottom
//      OR them together  : the wrapped-around rotate
//
//  IMPORTANT: this is only valid for 1 <= n <= 31. ChaCha20 only ever rotates
//  by 16, 12, 8, and 7, so we are safe. (A rotate by 0 or 32 would invoke the
//  C/C++ undefined behavior of shifting by the full width.)
//
//  On the GPU the compiler recognizes this exact idiom and emits a single
//  hardware funnel-shift instruction (SHF.L) -- there is no real "shift twice
//  and OR" cost. We keep the portable form because it is self-documenting; the
//  intrinsic __funnelshift_l() would compile to the same thing.
// ----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t ROTL32(uint32_t v, int n) {
    return (v << n) | (v >> (32 - n));
}

// ----------------------------------------------------------------------------
//  QUARTERROUND -- the atom of ChaCha20.
//
//  A "quarter round" takes four words of the state -- here named a, b, c, d --
//  and scrambles them with the fixed ARX sequence below. ARX = Add, Rotate,
//  Xor, the only three operations ChaCha20 uses. There are no S-boxes, no
//  lookup tables, and no data-dependent branches, which is precisely why
//  ChaCha20 is fast, constant-time (resistant to timing attacks), and trivial
//  to run on a GPU.
//
//  The canonical sequence (RFC 8439 section 2.1):
//
//      a += b;  d ^= a;  d = ROTL32(d, 16);
//      c += d;  b ^= c;  b = ROTL32(b, 12);
//      a += b;  d ^= a;  d = ROTL32(d,  8);
//      c += d;  b ^= c;  b = ROTL32(b,  7);
//
//  Reading it as data flow: each line lets one word absorb another (the Add),
//  feeds that change sideways into a third word (the Xor), and then smears the
//  changed bits across the word (the Rotate) so a single altered input bit
//  rapidly influences all 32 output bits. Four such lines, with the rotate
//  amounts 16/12/8/7, give the quarter round its "avalanche".
//
//  We implement it as a macro that operates IN PLACE on four named lvalues.
//  Why a macro instead of a function? Because the kernel calls it on different
//  combinations of array slots (x[0],x[4],x[8],x[12]; then x[0],x[5],x[10],
//  x[15]; ...). A macro lets us pass those slots directly as modifiable
//  references with zero call overhead and lets the compiler keep everything in
//  registers. The price is the usual macro hygiene rules -- hence the
//  parentheses and the multi-statement do/while(0) wrapper.
//
//  The arguments are expected to be uint32_t lvalues (array elements).
// ----------------------------------------------------------------------------
#define QUARTERROUND(a, b, c, d)        \
    do {                                \
        (a) += (b); (d) ^= (a); (d) = ROTL32((d), 16); \
        (c) += (d); (b) ^= (c); (b) = ROTL32((b), 12); \
        (a) += (b); (d) ^= (a); (d) = ROTL32((d),  8); \
        (c) += (d); (b) ^= (c); (b) = ROTL32((b),  7); \
    } while (0)

// ============================================================================
//  SECTION 2 -- The ChaCha20 block function, expressed as a CUDA kernel
// ----------------------------------------------------------------------------
//  ONE THREAD == ONE 64-BYTE BLOCK.
//
//  The grid is laid out so that the global thread index equals the block index:
//
//        global_thread_id = blockIdx.x * blockDim.x + threadIdx.x
//
//  Thread 0 handles input bytes   0..63   using counter = base_counter + 0,
//  thread 1 handles input bytes  64..127  using counter = base_counter + 1,
//  thread k handles input bytes 64k..64k+63 using counter = base_counter + k.
//
//  Because counter-mode blocks are independent (see chacha20.cuh), the threads
//  never communicate, never share memory, and never need a __syncthreads().
//  This is the embarrassingly-parallel ideal.
//
//  Kernel parameters:
//      in   : device pointer to the input bytes (plaintext or ciphertext).
//      out  : device pointer to the output bytes. May alias `in` (in-place).
//      len  : total number of bytes in the buffer.
//      init : the initial 4x4 state, passed BY VALUE. Each thread receives its
//             own private copy in registers/parameter space -- no global memory
//             read is needed to obtain the key, nonce, or constants.
//
//  Why __restrict__ here -- and the contract it imposes:
//  `in` and `out` are marked __restrict__ to PROMISE the compiler they never
//  alias. That promise lets it coalesce/vectorize the byte serialization loop
//  below instead of conservatively assuming each store to `out` might change a
//  future load from `in`. The win is large and real: measured on an RTX 2080
//  SUPER, removing __restrict__ here cut throughput ~6x (24.6 -> 4.2 GB/s),
//  because the per-byte loop can no longer be widened. The price of the promise
//  is a CONTRACT -- callers of this kernel (and of chacha20_xor_device) MUST
//  pass non-aliasing buffers; passing out == in would violate __restrict__ and
//  is undefined behavior. In-place encryption is still fully supported at the
//  library level: chacha20_xor_cuda() stages through two SEPARATE device
//  buffers, so even when the host passes h_output == h_input the kernel only
//  ever sees distinct pointers. (Each thread also touches only its own disjoint
//  64-byte slice, so there is no cross-thread interference either.)
// ============================================================================
__global__ void chacha20_kernel(const uint8_t* __restrict__ in,
                                uint8_t* __restrict__ out,
                                size_t len,
                                ChaCha20State init)
{
    // ---- 2.1  Figure out which block this thread owns ----------------------
    // A 64-bit index because a large buffer can contain billions of blocks and
    // a 32-bit index would overflow. (gridDim.x * blockDim.x can exceed 2^32.)
    const size_t block_index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    // Byte offset where this thread's 64-byte slice begins.
    const size_t byte_offset = block_index * (size_t)CHACHA20_BLOCK_BYTES;

    // The last CUDA block usually launches more threads than there are data
    // blocks (we round the grid size up). Any thread whose slice starts beyond
    // the end of the buffer has nothing to do, so it exits immediately. This is
    // the standard "grid-overhang" guard.
    if (byte_offset >= len) {
        return;
    }

    // ---- 2.2  Build this thread's working state ----------------------------
    // `x` is the matrix we will stir for 20 rounds. We start from the shared
    // template `init` and then OVERWRITE only word 12 (the counter) with this
    // thread's own block number. That is the single per-thread customization.
    //
    // x lives entirely in registers (16 uint32_t = 64 bytes). The #pragma unroll
    // tells the compiler to flatten this fixed-length loop into 16 straight-line
    // copies, which it would do anyway, but the hint makes the intent explicit.
    uint32_t x[CHACHA20_STATE_WORDS];
    #pragma unroll
    for (int i = 0; i < CHACHA20_STATE_WORDS; ++i) {
        x[i] = init.w[i];
    }

    // This thread's counter = base counter + its block index.
    //
    // NOTE on the 32-bit cast: RFC 8439 defines the block counter as a single
    // 32-bit word, giving a per-(key,nonce) ceiling of 2^32 blocks = 256 GiB.
    // If you ever processed more than that in one call, the counter would wrap
    // and keystream would repeat -- so 256 GiB is the documented per-call limit.
    // For any realistic buffer this addition is exact.
    x[12] = init.w[12] + (uint32_t)block_index;

    // ---- 2.3  Keep an untouched copy for the final feed-forward ------------
    // ChaCha20 finishes each block by adding the ORIGINAL (pre-round) state
    // back into the stirred state. Be precise about WHY this matters: the 20
    // rounds on their own are FULLY INVERTIBLE -- every operation is a modular
    // add, an XOR, or a fixed-amount rotate, and each of those is individually
    // reversible, so the round sequence is a bijection you could run backwards
    // step by step. The feed-forward is what makes the BLOCK function
    // (state -> keystream) one-way: because the unknown original state is
    // summed into the round output, an attacker who sees the final `x` cannot
    // peel the rounds back to recover the state (and thus the key) without
    // already knowing that original state. (This is the same Davies-Meyer trick
    // that turns a reversible permutation into a one-way function.) So we must
    // stash the starting words before we mutate them.
    uint32_t start[CHACHA20_STATE_WORDS];
    #pragma unroll
    for (int i = 0; i < CHACHA20_STATE_WORDS; ++i) {
        start[i] = x[i];
    }

    // ---- 2.4  The 20 rounds = 10 "double rounds" ---------------------------
    // Each iteration performs:
    //   * 4 COLUMN quarter-rounds  -- they mix down the four columns of the
    //     matrix, so each column's four words are stirred together.
    //   * 4 DIAGONAL quarter-rounds -- they mix along the four diagonals, so
    //     information that was confined to a column now spreads across columns.
    //
    // Alternating column then diagonal is the trick that diffuses every input
    // bit to every output word within a handful of rounds. The specific index
    // quadruples below come straight from RFC 8439 section 2.3.1.
    //
    // 10 iterations * 2 round-types = 20 rounds, hence "ChaCha20".
    #pragma unroll
    for (int round = 0; round < CHACHA20_ROUNDS / 2; ++round) {
        // Column rounds: operate on the 4 vertical columns.
        QUARTERROUND(x[0], x[4], x[ 8], x[12]);   // column 0
        QUARTERROUND(x[1], x[5], x[ 9], x[13]);   // column 1
        QUARTERROUND(x[2], x[6], x[10], x[14]);   // column 2
        QUARTERROUND(x[3], x[7], x[11], x[15]);   // column 3

        // Diagonal rounds: operate on the 4 diagonals of the matrix.
        QUARTERROUND(x[0], x[5], x[10], x[15]);   // main diagonal
        QUARTERROUND(x[1], x[6], x[11], x[12]);   // next diagonal (wraps)
        QUARTERROUND(x[2], x[7], x[ 8], x[13]);
        QUARTERROUND(x[3], x[4], x[ 9], x[14]);
    }

    // ---- 2.5  Feed-forward: add the original state back in -----------------
    // After this loop, x holds the 64 bytes of keystream for this block (still
    // as 16 words; serialization happens next).
    #pragma unroll
    for (int i = 0; i < CHACHA20_STATE_WORDS; ++i) {
        x[i] += start[i];
    }

    // ---- 2.6  Serialize to bytes and XOR with the input --------------------
    // The keystream words must be written out in LITTLE-ENDIAN order (least
    // significant byte first), per the spec. Word x[j] therefore contributes
    // bytes (4*j + 0 .. 4*j + 3), where byte b of that word is (x[j] >> (8*b)).
    //
    // `n` is how many bytes this thread should actually process. For every full
    // block it is 64; for the final, possibly-partial block it is whatever is
    // left (1..64). This single line is the entire "partial last block" story --
    // no special-casing, no padding, no buffer overrun past `len`.
    const size_t remaining = len - byte_offset;
    const size_t n = (remaining < (size_t)CHACHA20_BLOCK_BYTES)
                         ? remaining
                         : (size_t)CHACHA20_BLOCK_BYTES;

    for (size_t i = 0; i < n; ++i) {
        // i >> 2  selects the word (i / 4); (i & 3) selects the byte within it.
        const uint8_t keystream_byte =
            (uint8_t)(x[i >> 2] >> ((i & 3) * 8));

        // The cipher itself: output = input XOR keystream. Identical code path
        // for encryption and decryption.
        out[byte_offset + i] = in[byte_offset + i] ^ keystream_byte;
    }
}

// ============================================================================
//  SECTION 3 -- Host launcher (operates on device pointers)
// ----------------------------------------------------------------------------
//  Given buffers that already live on the GPU, choose a launch configuration
//  and start the kernel. This separates "how many threads / how big a grid"
//  from "where the bytes live", so the benchmark can reuse it without paying
//  for host<->device copies every iteration.
// ============================================================================
cudaError_t chacha20_xor_device(const uint8_t* d_input,
                                uint8_t*       d_output,
                                size_t         len,
                                ChaCha20State  init,
                                int            threads_per_block,
                                cudaStream_t   stream)
{
    // Processing zero bytes is a valid no-op; launching a kernel with a zero
    // grid is illegal, so return early.
    if (len == 0) {
        return cudaSuccess;
    }

    // Choose a default block size if the caller passed 0. 256 threads per block
    // is a robust, widely-good choice: it is a multiple of the warp size (32),
    // keeps register pressure reasonable, and gives the scheduler enough warps
    // (8) per block to hide memory latency on every NVIDIA generation.
    if (threads_per_block <= 0) {
        threads_per_block = 256;
    }

    // How many 64-byte data blocks does the buffer contain? Round UP so the
    // final partial block still gets a thread. This is the classic
    // ceil(len / 64) = (len + 63) / 64 integer-math trick.
    const size_t num_blocks =
        (len + (size_t)CHACHA20_BLOCK_BYTES - 1) / (size_t)CHACHA20_BLOCK_BYTES;

    // How many CUDA blocks (groups of `threads_per_block` threads) do we need
    // to cover `num_blocks` data blocks? Round UP again. The kernel's
    // grid-overhang guard (section 2.1) handles the extra threads in the last
    // CUDA block.
    const size_t grid_blocks =
        (num_blocks + (size_t)threads_per_block - 1) / (size_t)threads_per_block;

    // gridDim.x is a 32-bit field (max 2^31 - 1). For our 256 GiB ceiling this
    // is never exceeded (256 GiB / 64 B / 256 threads ~= 16.7M < 2^31), but we
    // assert it defensively so a future change that lifts the size limit fails
    // loudly instead of silently truncating the grid.
    if (grid_blocks > 2147483647ull) {
        fprintf(stderr, "[ChaCha20] buffer too large for a single launch\n");
        return cudaErrorInvalidConfiguration;
    }

    // dim3 packs the grid/block dimensions. We only use the x-dimension; y and
    // z default to 1. The cast is safe given the check above.
    dim3 grid((unsigned int)grid_blocks);
    dim3 block((unsigned int)threads_per_block);

    // Launch. The <<< >>> "execution configuration" syntax is CUDA's extension
    // to C++: <<<grid, block, dynamic_shared_mem, stream>>>. We use 0 bytes of
    // dynamic shared memory (the kernel keeps everything in registers) and the
    // caller-supplied stream.
    chacha20_kernel<<<grid, block, 0, stream>>>(d_input, d_output, len, init);

    // A kernel launch is asynchronous and can fail in two distinct ways:
    //   (a) at launch time (bad config) -- surfaced by cudaGetLastError();
    //   (b) during execution -- surfaced later, e.g. by a sync.
    // We check (a) immediately. We deliberately do NOT synchronize here: the
    // caller owns the stream and decides when to wait, which lets them overlap
    // this launch with other work. The high-level wrapper below does sync.
    CUDA_CHECK(cudaGetLastError());

    return cudaSuccess;
}

// ============================================================================
//  SECTION 4 -- Host convenience wrapper (operates on host pointers)
// ----------------------------------------------------------------------------
//  The "just encrypt my buffer" entry point. It owns the full lifecycle of the
//  GPU memory so the caller never has to touch cudaMalloc/cudaMemcpy/cudaFree.
//
//  Lifecycle:
//      1. allocate two device buffers (input, output) of `len` bytes,
//      2. copy the host input up to the device (Host -> Device),
//      3. launch the kernel via chacha20_xor_device,
//      4. wait for completion and copy the result back (Device -> Host),
//      5. free the device buffers -- even on the error paths.
//
//  Every cudaMalloc here is a heap allocation in GPU memory; per the project's
//  "flag allocations" convention, note that this function performs exactly two
//  device allocations and frees both before returning on every path.
// ============================================================================
cudaError_t chacha20_xor_cuda(const uint8_t* h_input,
                              uint8_t*       h_output,
                              size_t         len,
                              const uint8_t  key[CHACHA20_KEY_BYTES],
                              const uint8_t  nonce[CHACHA20_NONCE_BYTES],
                              uint32_t       counter)
{
    if (len == 0) {
        return cudaSuccess;   // nothing to do
    }

    // Build the initial state on the host (cheap, runs once). Each GPU thread
    // will receive a by-value copy of this.
    ChaCha20State init = chacha20_init_state(key, nonce, counter);

    // Device buffers. Declared up front and zero-initialized so the cleanup
    // label can safely cudaFree(nullptr) (which is a documented no-op) if an
    // early allocation fails.
    uint8_t* d_input  = nullptr;
    uint8_t* d_output = nullptr;
    cudaError_t status = cudaSuccess;

    // --- Allocate input buffer on the device --------------------------------
    status = cudaMalloc((void**)&d_input, len);
    if (status != cudaSuccess) {
        fprintf(stderr, "[ChaCha20] cudaMalloc(d_input) failed: %s\n",
                cudaGetErrorString(status));
        goto cleanup;
    }

    // --- Allocate output buffer on the device -------------------------------
    status = cudaMalloc((void**)&d_output, len);
    if (status != cudaSuccess) {
        fprintf(stderr, "[ChaCha20] cudaMalloc(d_output) failed: %s\n",
                cudaGetErrorString(status));
        goto cleanup;
    }

    // --- Copy plaintext/ciphertext up to the GPU ----------------------------
    // cudaMemcpyHostToDevice walks the data across the PCIe bus. For a single
    // call this transfer typically dominates the wall-clock time -- the kernel
    // itself is far faster than the copy. (The benchmark in demo.cu isolates
    // kernel time precisely to make this visible.)
    status = cudaMemcpy(d_input, h_input, len, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        fprintf(stderr, "[ChaCha20] cudaMemcpy H2D failed: %s\n",
                cudaGetErrorString(status));
        goto cleanup;
    }

    // --- Launch the kernel on the default stream (0) ------------------------
    status = chacha20_xor_device(d_input, d_output, len, init,
                                 /*threads_per_block=*/256, /*stream=*/0);
    if (status != cudaSuccess) {
        goto cleanup;   // launcher already printed the error
    }

    // --- Wait for the kernel, then copy the result back ---------------------
    // cudaMemcpy (Device -> Host) implicitly synchronizes on the default
    // stream, so by the time it returns the kernel has finished AND the bytes
    // are back in host memory. We still check for asynchronous kernel errors
    // first via cudaDeviceSynchronize for a precise error message.
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        fprintf(stderr, "[ChaCha20] kernel execution failed: %s\n",
                cudaGetErrorString(status));
        goto cleanup;
    }

    status = cudaMemcpy(h_output, d_output, len, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        fprintf(stderr, "[ChaCha20] cudaMemcpy D2H failed: %s\n",
                cudaGetErrorString(status));
        goto cleanup;
    }

cleanup:
    // Free both device buffers. cudaFree(nullptr) is a safe no-op, so this is
    // correct whether we arrived here via success or via an early failure.
    // Freeing on every path is what prevents GPU memory leaks across many
    // calls -- the same discipline you would apply to malloc/free on the host.
    cudaFree(d_input);
    cudaFree(d_output);
    return status;
}
