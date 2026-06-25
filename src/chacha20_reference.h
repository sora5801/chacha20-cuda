// ============================================================================
//  chacha20_reference.h  --  Interface for the plain-C++ reference ("oracle")
// ============================================================================
//
//  WHY A SECOND, SLOWER IMPLEMENTATION EXISTS
//  ------------------------------------------
//  The CUDA kernel in chacha20.cu is the implementation we actually want to
//  ship. This file declares a completely SEPARATE, dead-simple, single-threaded
//  CPU version of the same cipher. Its only job is to be OBVIOUSLY correct so
//  the tests can trust it as a reference -- an "oracle" -- to check the GPU
//  output against.
//
//  Crucially, the two implementations share NO code. If they shared a buggy
//  helper, both would be wrong in the same way and the bug would hide. By
//  writing the reference independently (different style, no GPU constructs) we
//  get a genuine cross-check: the GPU result is only trusted when it matches
//  BOTH the published RFC 8439 vectors AND this independent CPU computation.
//
//  This is ordinary host C++ -- no CUDA. It compiles as a .cpp and can be read
//  by anyone who has never touched a GPU, which also makes it the best place to
//  learn the algorithm before tackling the parallel version.
// ============================================================================

#ifndef CHACHA20_REFERENCE_H
#define CHACHA20_REFERENCE_H

#include <cstdint>
#include <cstddef>

// ----------------------------------------------------------------------------
//  chacha20_quarterround_ref -- run ONE quarter round on four state words.
//
//  Exposed mainly so the test suite can reproduce RFC 8439's standalone
//  quarter-round known-answer test (section 2.1.1). It operates in place on the
//  four uint32_t values passed by reference.
// ----------------------------------------------------------------------------
void chacha20_quarterround_ref(uint32_t& a, uint32_t& b,
                               uint32_t& c, uint32_t& d);

// ----------------------------------------------------------------------------
//  chacha20_block_ref -- produce ONE 64-byte keystream block.
//
//  Given the key, nonce, and a specific block counter, compute the 64 keystream
//  bytes for that block and write them to `out64`. This is the CPU twin of the
//  GPU kernel's per-thread work, isolated so the test can compare a single
//  block against RFC 8439 section 2.3.2.
//
//      key     : 32 bytes.
//      nonce   : 12 bytes.
//      counter : the 32-bit block counter for THIS block.
//      out64   : caller-provided buffer of at least 64 bytes.
// ----------------------------------------------------------------------------
void chacha20_block_ref(const uint8_t key[32],
                        const uint8_t nonce[12],
                        uint32_t      counter,
                        uint8_t       out64[64]);

// ----------------------------------------------------------------------------
//  chacha20_xor_ref -- encrypt/decrypt `len` bytes entirely on the CPU.
//
//  Same contract as chacha20_xor_cuda() but single-threaded host code. The
//  tests run both and demand identical output. Handles arbitrary lengths,
//  including a partial final block.
// ----------------------------------------------------------------------------
void chacha20_xor_ref(const uint8_t* input,
                      uint8_t*       output,
                      size_t         len,
                      const uint8_t  key[32],
                      const uint8_t  nonce[12],
                      uint32_t       counter);

#endif // CHACHA20_REFERENCE_H
