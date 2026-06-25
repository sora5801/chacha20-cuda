// ============================================================================
//  chacha20_reference.cpp  --  Plain-C++ reference implementation (the oracle)
// ============================================================================
//
//  This file re-implements ChaCha20 from scratch in the most boring, readable
//  way possible, using only the standard library and a single CPU thread. It is
//  deliberately NOT optimized: clarity beats speed here because its entire
//  purpose is to be a trustworthy yardstick for the GPU version.
//
//  If you have never implemented ChaCha20 before, READ THIS FILE FIRST. Every
//  step maps one-to-one onto RFC 8439, and there are no GPU concepts to
//  distract you. Once this makes sense, the CUDA kernel in chacha20.cu will
//  read as "the same thing, but one block per thread".
// ============================================================================

#include "chacha20_reference.h"

#include <cstring>   // std::memcpy for copying the keystream block

// ----------------------------------------------------------------------------
//  rotl32 -- rotate a 32-bit value left by c bits.
//
//  Identical in spirit to the GPU's ROTL32, but written as an ordinary inline
//  function. Valid for 1..31; ChaCha20 only uses 16/12/8/7. We declare it
//  `static` so it has internal linkage and cannot collide with the GPU file's
//  symbol of a similar name at link time.
// ----------------------------------------------------------------------------
static inline uint32_t rotl32(uint32_t v, int c) {
    return (v << c) | (v >> (32 - c));
}

// ----------------------------------------------------------------------------
//  load_le32 / store_le32 -- portable little-endian (de)serialization.
//
//  Same reasoning as in chacha20.cu: we assemble/disassemble words byte by byte
//  so the code is correct on both little-endian and big-endian CPUs. Never cast
//  a byte pointer to a uint32_t* for this -- that would hard-code host
//  endianness and is also a strict-aliasing violation.
// ----------------------------------------------------------------------------
static inline uint32_t load_le32(const uint8_t* p) {
    return  (uint32_t)p[0]
         | ((uint32_t)p[1] <<  8)
         | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
}

static inline void store_le32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)(v        & 0xff);
    p[1] = (uint8_t)((v >> 8 ) & 0xff);
    p[2] = (uint8_t)((v >> 16) & 0xff);
    p[3] = (uint8_t)((v >> 24) & 0xff);
}

// ----------------------------------------------------------------------------
//  The quarter round, written as plain statements (not a macro) for clarity.
//  Compare line-for-line with the QUARTERROUND macro in chacha20.cu: they are
//  the same four ARX lines.
// ----------------------------------------------------------------------------
void chacha20_quarterround_ref(uint32_t& a, uint32_t& b,
                               uint32_t& c, uint32_t& d) {
    a += b;  d ^= a;  d = rotl32(d, 16);
    c += d;  b ^= c;  b = rotl32(b, 12);
    a += b;  d ^= a;  d = rotl32(d,  8);
    c += d;  b ^= c;  b = rotl32(b,  7);
}

// ----------------------------------------------------------------------------
//  Internal helper: run a quarter round addressed by INDEX into a state array.
//  The kernel calls QUARTERROUND on array slots directly; here we wrap the
//  reference quarter round so we can write the column/diagonal pattern with
//  index quadruples, matching RFC 8439 section 2.3.1 exactly.
// ----------------------------------------------------------------------------
static inline void qr(uint32_t state[16], int a, int b, int c, int d) {
    chacha20_quarterround_ref(state[a], state[b], state[c], state[d]);
}

// ----------------------------------------------------------------------------
//  chacha20_block_ref -- the full 20-round block function for one counter.
//
//  Steps (mirroring chacha20.cu section 2):
//    1. Lay out the initial state: constants, key, counter, nonce.
//    2. Copy it to a working array.
//    3. Apply 10 double rounds (column + diagonal) = 20 rounds.
//    4. Add the original state back in (the feed-forward).
//    5. Serialize all 16 words little-endian into the 64-byte output.
// ----------------------------------------------------------------------------
void chacha20_block_ref(const uint8_t key[32],
                        const uint8_t nonce[12],
                        uint32_t      counter,
                        uint8_t       out64[64]) {
    // --- 1. Initial state ---------------------------------------------------
    uint32_t state[16];
    state[0] = 0x61707865u;   // "expa"  (see chacha20.cuh for the derivation)
    state[1] = 0x3320646eu;   // "nd 3"
    state[2] = 0x79622d32u;   // "2-by"
    state[3] = 0x6b206574u;   // "te k"
    for (int i = 0; i < 8; ++i) {
        state[4 + i] = load_le32(key + 4 * i);     // key words 0..7
    }
    state[12] = counter;                            // block counter
    for (int i = 0; i < 3; ++i) {
        state[13 + i] = load_le32(nonce + 4 * i);   // nonce words 0..2
    }

    // --- 2. Working copy ----------------------------------------------------
    uint32_t working[16];
    std::memcpy(working, state, sizeof(working));

    // --- 3. 20 rounds = 10 double rounds ------------------------------------
    for (int i = 0; i < 10; ++i) {
        // Column rounds
        qr(working, 0, 4,  8, 12);
        qr(working, 1, 5,  9, 13);
        qr(working, 2, 6, 10, 14);
        qr(working, 3, 7, 11, 15);
        // Diagonal rounds
        qr(working, 0, 5, 10, 15);
        qr(working, 1, 6, 11, 12);
        qr(working, 2, 7,  8, 13);
        qr(working, 3, 4,  9, 14);
    }

    // --- 4. Feed-forward: working += original state -------------------------
    for (int i = 0; i < 16; ++i) {
        working[i] += state[i];
    }

    // --- 5. Serialize little-endian into the 64-byte block ------------------
    for (int i = 0; i < 16; ++i) {
        store_le32(out64 + 4 * i, working[i]);
    }
}

// ----------------------------------------------------------------------------
//  chacha20_xor_ref -- stream the whole buffer, one 64-byte block at a time.
//
//  For each chunk of up to 64 input bytes we generate the matching keystream
//  block (incrementing the counter each time) and XOR byte by byte. The final
//  chunk may be shorter than 64 bytes; we simply stop at `len`.
// ----------------------------------------------------------------------------
void chacha20_xor_ref(const uint8_t* input,
                      uint8_t*       output,
                      size_t         len,
                      const uint8_t  key[32],
                      const uint8_t  nonce[12],
                      uint32_t       counter) {
    uint8_t keystream[64];
    size_t  offset = 0;
    uint32_t block_counter = counter;

    while (offset < len) {
        // Keystream for this block.
        chacha20_block_ref(key, nonce, block_counter, keystream);

        // How many bytes remain in this (possibly final, partial) block.
        size_t chunk = len - offset;
        if (chunk > 64) {
            chunk = 64;
        }

        // XOR plaintext with keystream.
        for (size_t i = 0; i < chunk; ++i) {
            output[offset + i] = input[offset + i] ^ keystream[i];
        }

        offset        += chunk;
        block_counter += 1;   // next block uses the next counter value
    }
}
