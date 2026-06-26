// ============================================================================
//  chacha20.cuh  --  Public interface for the CUDA ChaCha20 stream cipher
// ============================================================================
//
//  WHAT THIS FILE IS
//  -----------------
//  This header is the single "front door" to the project. Every other source
//  file (the kernel, the tests, the demo) includes this header to learn:
//
//      * the numeric constants that define ChaCha20,
//      * the in-memory layout of the cipher "state" (the 4x4 matrix), and
//      * the prototypes of the host-callable functions that launch the GPU
//        work.
//
//  Keeping all of this in one header means there is exactly one authoritative
//  definition of the data structures, so the host code and the device code can
//  never disagree about, say, how many bytes are in a key.
//
//
//  A 60-SECOND TOUR OF ChaCha20  (so the constants below make sense)
//  -----------------------------------------------------------------
//  ChaCha20 is a *stream cipher* designed by Daniel J. Bernstein. A stream
//  cipher does not encrypt your data directly. Instead it generates a long,
//  unpredictable pseudo-random byte stream called the "keystream", and you
//  encrypt by XOR-ing your plaintext with that keystream:
//
//      ciphertext = plaintext XOR keystream
//      plaintext  = ciphertext XOR keystream      (XOR is its own inverse)
//
//  Because encryption and decryption are *the same operation* (XOR with the
//  same keystream), this library exposes a single "xor" function that does
//  both. Run it on plaintext to encrypt; run it again on the ciphertext with
//  the identical key/nonce/counter to decrypt.
//
//  The keystream is produced 64 bytes at a time by the "ChaCha20 block
//  function". The block function takes:
//
//      * a 256-bit (32-byte) secret key,
//      * a 96-bit (12-byte) "nonce" (number-used-once),
//      * a 32-bit block counter,
//
//  arranges them into a 4x4 matrix of 32-bit words (the "state"), stirs that
//  matrix with 20 rounds of add / rotate / XOR ("ARX") operations, and emits
//  64 keystream bytes. Bump the counter by 1 and you get the next 64 bytes.
//
//
//  WHY ChaCha20 IS A GREAT FIT FOR A GPU  (the whole point of this project)
//  -----------------------------------------------------------------------
//  The block for counter N depends ONLY on (key, nonce, N). It does NOT depend
//  on the block for counter N-1. This is "counter mode": every 64-byte block
//  is completely independent of every other block. That independence is exactly
//  what a GPU craves -- we can hand block 0 to thread 0, block 1 to thread 1,
//  ..., block 1,000,000 to thread 1,000,000, and they all run in parallel with
//  zero coordination. Contrast this with a cipher like AES-CBC, where block N
//  needs the output of block N-1 and is therefore inherently serial.
//
//  This file (and the project) is written to be read top-to-bottom as study
//  material, so comments err heavily on the side of "too much explanation".
//
//  Reference: RFC 8439, "ChaCha20 and Poly1305 for IETF Protocols".
//             https://www.rfc-editor.org/rfc/rfc8439
// ============================================================================

#ifndef CHACHA20_CUH
#define CHACHA20_CUH

// <cstdint>  -> fixed-width integer types (uint8_t, uint32_t, ...). Crypto code
//               must be exact about bit widths; "int" or "long" vary by platform
//               and would silently corrupt the math. We only ever use the sized
//               types below.
// <cstddef>  -> size_t, the unsigned type used for byte counts / buffer lengths.
#include <cstdint>
#include <cstddef>

// cuda_runtime.h gives us cudaError_t, cudaStream_t, and the launch machinery.
// We include it here so callers that only include this header still get the
// CUDA types named in our prototypes.
#include <cuda_runtime.h>

// ----------------------------------------------------------------------------
//  Fixed sizes, expressed as named constants so the rest of the code never
//  contains a "magic number" like 32 whose meaning you have to guess.
// ----------------------------------------------------------------------------

// A ChaCha20 key is always 256 bits = 32 bytes. (The "20" in ChaCha20 is the
// number of rounds, NOT the key size; the key is 256-bit.)
#define CHACHA20_KEY_BYTES    32

// The IETF/RFC 8439 nonce is 96 bits = 12 bytes. (Bernstein's original design
// used a 64-bit nonce; this project follows the modern RFC 8439 variant, which
// is what TLS, WireGuard, SSH, etc. all use.)
#define CHACHA20_NONCE_BYTES  12

// One run of the block function emits exactly 64 bytes of keystream. This is
// also the size of the cipher state: 16 words x 4 bytes = 64 bytes.
#define CHACHA20_BLOCK_BYTES  64

// The state is a 4x4 grid, i.e. 16 thirty-two-bit words.
#define CHACHA20_STATE_WORDS  16

// ChaCha20 performs 20 rounds. We implement them as 10 "double rounds", where
// each double round is 1 column round + 1 diagonal round (see chacha20.cu).
#define CHACHA20_ROUNDS       20

// ----------------------------------------------------------------------------
//  The four "magic constants" that occupy the first row of the state.
//
//  They are simply the ASCII bytes of the string "expand 32-byte k", read as
//  four little-endian 32-bit words:
//
//        "expa" -> 0x61707865     ('e'=0x65,'x'=0x78,'p'=0x70,'a'=0x61)
//        "nd 3" -> 0x3320646e
//        "2-by" -> 0x79622d32
//        "te k" -> 0x6b206574
//
//  Why a fixed, publicly known string? These "nothing-up-my-sleeve" numbers
//  diversify the state and ensure the first row is never attacker-controlled.
//  They carry no secret; they exist purely to give the mixing function a known,
//  asymmetric starting point. The phrase encodes the design: "expand" a
//  "32-byte" (256-bit) "k"(ey).
// ----------------------------------------------------------------------------
#define CHACHA20_CONST_0  0x61707865u   // "expa"
#define CHACHA20_CONST_1  0x3320646eu   // "nd 3"
#define CHACHA20_CONST_2  0x79622d32u   // "2-by"
#define CHACHA20_CONST_3  0x6b206574u   // "te k"

// ----------------------------------------------------------------------------
//  ChaCha20State -- the 4x4 working matrix, flattened into a 16-word array.
//
//  Memory layout (row-major). Each cell is one uint32_t (4 bytes):
//
//        index:   0    1    2    3
//                 4    5    6    7
//                 8    9   10   11
//                12   13   14   15
//
//  Semantic meaning of each cell once the state is initialized:
//
//        [ 0] const0   [ 1] const1   [ 2] const2   [ 3] const3      <- constants
//        [ 4] key0     [ 5] key1     [ 6] key2     [ 7] key3        <- key words 0..3
//        [ 8] key4     [ 9] key5     [10] key6     [11] key7        <- key words 4..7
//        [12] counter  [13] nonce0   [14] nonce1   [15] nonce2      <- counter + nonce
//
//  We wrap the array in a struct (rather than using a bare uint32_t[16]) for
//  two practical reasons:
//
//    1. We can pass it BY VALUE as a kernel argument. The whole 64-byte state
//       is copied into the kernel's parameter space, so every GPU thread starts
//       from an identical, read-only template and then customizes only its own
//       counter word. No global memory traffic is needed to fetch the key.
//
//    2. A struct documents intent and prevents array-to-pointer "decay" from
//       quietly turning a 64-byte value into an 8-byte pointer at a call site.
//
//  __host__ __device__ is not needed on a plain struct definition; the type is
//  usable on both sides. We only annotate *functions* with execution-space
//  qualifiers.
// ----------------------------------------------------------------------------
typedef struct ChaCha20State {
    uint32_t w[CHACHA20_STATE_WORDS];   // the 16 words, indices 0..15 as above
} ChaCha20State;

// ----------------------------------------------------------------------------
//  Host-side helper: build the initial state from raw key/nonce/counter bytes.
//
//  This runs on the CPU (it is plain host code). It reads the 32 key bytes and
//  12 nonce bytes as LITTLE-ENDIAN 32-bit words and drops them, together with
//  the constants and the starting counter, into the 16 state slots shown above.
//
//  Parameters:
//      key     : pointer to exactly 32 bytes of secret key.
//      nonce   : pointer to exactly 12 bytes of nonce (unique per message).
//      counter : the 32-bit block counter for the FIRST block. RFC 8439's
//                worked examples start at 1; many protocols start at 0. The
//                caller decides.
//
//  Returns a fully-populated ChaCha20State, ready to be handed to a kernel.
// ----------------------------------------------------------------------------
ChaCha20State chacha20_init_state(const uint8_t key[CHACHA20_KEY_BYTES],
                                  const uint8_t nonce[CHACHA20_NONCE_BYTES],
                                  uint32_t counter);

// ----------------------------------------------------------------------------
//  HIGH-LEVEL API  --  the function 95% of callers want.
//
//  chacha20_xor_cuda() encrypts OR decrypts `len` bytes. It does everything:
//  allocates GPU memory, copies the input up to the device, launches the
//  kernel, copies the result back, and frees the GPU memory. The caller just
//  provides ordinary host (CPU) pointers.
//
//  Parameters:
//      h_input  : host pointer to `len` plaintext  (encrypt) or
//                 ciphertext (decrypt) bytes to read.
//      h_output : host pointer to `len` bytes to write the result into. It is
//                 legal for h_output == h_input (encrypt in place).
//      len      : number of bytes to process. Any length is allowed, including
//                 lengths that are NOT a multiple of 64; the final partial
//                 block is handled correctly.
//      key      : 32-byte key.
//      nonce    : 12-byte nonce. CRITICAL: never reuse the same (key, nonce)
//                 pair for two different messages -- doing so XORs two plaintexts
//                 under the same keystream and breaks the cipher completely.
//      counter  : starting 32-bit block counter (see chacha20_init_state).
//
//  Returns cudaSuccess on success, or the first CUDA error encountered. The
//  return type is cudaError_t (rather than void) so callers can detect and
//  report GPU failures instead of silently producing garbage.
// ----------------------------------------------------------------------------
cudaError_t chacha20_xor_cuda(const uint8_t* h_input,
                              uint8_t*       h_output,
                              size_t         len,
                              const uint8_t  key[CHACHA20_KEY_BYTES],
                              const uint8_t  nonce[CHACHA20_NONCE_BYTES],
                              uint32_t       counter);

// ----------------------------------------------------------------------------
//  LOW-LEVEL API  --  for callers that manage their own GPU memory.
//
//  chacha20_xor_device() assumes the input is ALREADY in device memory and the
//  output buffer is ALREADY allocated in device memory. It performs no host
//  <-> device copies. The benchmark in demo.cu uses this so it can measure pure
//  kernel throughput without counting PCIe transfer time.
//
//  Parameters:
//      d_input          : DEVICE pointer to `len` input bytes.
//      d_output         : DEVICE pointer to `len` output bytes. MUST NOT alias
//                         d_input: the kernel marks its pointers __restrict__
//                         (worth ~6x throughput), so the two buffers have to be
//                         distinct. For IN-PLACE encryption use the high-level
//                         chacha20_xor_cuda(), which double-buffers on the
//                         device so the kernel never sees aliasing.
//      len              : number of bytes to process.
//      init             : the initial state, already built with
//                         chacha20_init_state(). Passed by value; the kernel
//                         receives its own copy.
//      threads_per_block: CUDA block size (e.g. 256). If you pass 0 we pick a
//                         sensible default. Must be a multiple of the warp size
//                         (32) for best efficiency.
//      stream           : the CUDA stream to launch on (0 = default stream).
//
//  Returns the result of the launch / any synchronous error.
// ----------------------------------------------------------------------------
cudaError_t chacha20_xor_device(const uint8_t* d_input,
                                uint8_t*       d_output,
                                size_t         len,
                                ChaCha20State  init,
                                int            threads_per_block,
                                cudaStream_t   stream);

#endif // CHACHA20_CUH
