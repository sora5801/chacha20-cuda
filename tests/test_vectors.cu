// ============================================================================
//  test_vectors.cu  --  Known-answer tests proving the cipher is correct
// ============================================================================
//
//  A cipher that is fast but wrong is worthless, so this is arguably the most
//  important file in the project. It verifies the implementation three ways:
//
//    (A) Against the PUBLISHED constants in RFC 8439. These "known-answer
//        tests" pin our output to bytes that the whole world agrees on, so a
//        passing run means we implemented the actual standard, not some
//        look-alike.
//
//    (B) GPU vs CPU. We run the same inputs through the parallel CUDA kernel
//        and the independent single-threaded reference and demand identical
//        output. Because the two were written separately, agreement is strong
//        evidence both are right.
//
//    (C) Round trips. Encrypt then decrypt random data of awkward lengths and
//        confirm we recover the original bytes -- this exercises the partial
//        final block and the "encryption == decryption" property.
//
//  Each test prints "[ PASS ]" or "[ FAIL ]". run_all_tests() ANDs them all.
// ============================================================================

#include "test_vectors.cuh"

#include "chacha20.cuh"             // GPU API + chacha20_init_state
#include "../src/chacha20_reference.h"  // CPU oracle

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>

// ----------------------------------------------------------------------------
//  Small console helpers (kept local to this file with `static`).
// ----------------------------------------------------------------------------

// Print a byte buffer as spaced hex, wrapping at 16 bytes per line. Used to
// show the actual-vs-expected diff when a test fails.
static void print_hex(const char* label, const uint8_t* data, size_t len) {
    printf("    %s (%zu bytes):\n", label, len);
    for (size_t i = 0; i < len; ++i) {
        if (i % 16 == 0) printf("        ");
        printf("%02x ", data[i]);
        if (i % 16 == 15) printf("\n");
    }
    if (len % 16 != 0) printf("\n");
}

// Compare two buffers; on mismatch, dump both so the failure is debuggable.
static bool bytes_equal(const char* what,
                        const uint8_t* got, const uint8_t* expected, size_t len) {
    if (std::memcmp(got, expected, len) == 0) {
        return true;
    }
    printf("    MISMATCH in %s\n", what);
    print_hex("got     ", got, len);
    print_hex("expected", expected, len);
    return false;
}

// Uniform PASS/FAIL line so the output reads like a checklist.
static bool report(const char* name, bool ok) {
    printf("  [ %s ] %s\n", ok ? "PASS" : "FAIL", name);
    return ok;
}

// ----------------------------------------------------------------------------
//  Shared test data from RFC 8439.
// ----------------------------------------------------------------------------

// The 32-byte key 00 01 02 ... 1f, used by the section 2.3.2 and 2.4.2 vectors.
static const uint8_t RFC_KEY[32] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f
};

// ============================================================================
//  TEST 1 -- Quarter-round known answer (RFC 8439, section 2.1.1)
// ----------------------------------------------------------------------------
//  The smallest possible check: feed the four documented input words through a
//  single quarter round and confirm the four documented output words. If this
//  fails, every higher-level test will too, so we run it first.
// ============================================================================
static bool test_quarterround() {
    uint32_t a = 0x11111111u, b = 0x01020304u, c = 0x9b8d6f43u, d = 0x01234567u;
    chacha20_quarterround_ref(a, b, c, d);

    const bool ok = (a == 0xea2a92f4u) && (b == 0xcb1cf8ceu)
                 && (c == 0x4581472eu) && (d == 0x5881c4bbu);
    if (!ok) {
        printf("    got      a=%08x b=%08x c=%08x d=%08x\n", a, b, c, d);
        printf("    expected a=ea2a92f4 b=cb1cf8ce c=4581472e d=5881c4bb\n");
    }
    return report("RFC 8439 2.1.1  quarter-round known answer", ok);
}

// ============================================================================
//  TEST 2 -- Block function keystream (RFC 8439, section 2.3.2)
// ----------------------------------------------------------------------------
//  With the RFC key, nonce 00..00 09 ... 4a ..., and counter = 1, the block
//  function must emit exactly these 64 keystream bytes. We check the CPU oracle
//  here; TEST 3 checks the GPU against the same constant.
// ============================================================================
static bool test_block_keystream_cpu() {
    const uint8_t nonce[12] = {
        0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x4a, 0x00, 0x00, 0x00, 0x00
    };
    const uint8_t expected[64] = {
        0x10, 0xf1, 0xe7, 0xe4, 0xd1, 0x3b, 0x59, 0x15,
        0x50, 0x0f, 0xdd, 0x1f, 0xa3, 0x20, 0x71, 0xc4,
        0xc7, 0xd1, 0xf4, 0xc7, 0x33, 0xc0, 0x68, 0x03,
        0x04, 0x22, 0xaa, 0x9a, 0xc3, 0xd4, 0x6c, 0x4e,
        0xd2, 0x82, 0x64, 0x46, 0x07, 0x9f, 0xaa, 0x09,
        0x14, 0xc2, 0xd7, 0x05, 0xd9, 0x8b, 0x02, 0xa2,
        0xb5, 0x12, 0x9c, 0xd1, 0xde, 0x16, 0x4e, 0xb9,
        0xcb, 0xd0, 0x83, 0xe8, 0xa2, 0x50, 0x3c, 0x4e
    };

    uint8_t out[64];
    chacha20_block_ref(RFC_KEY, nonce, /*counter=*/1, out);
    return report("RFC 8439 2.3.2  block keystream (CPU oracle)",
                  bytes_equal("keystream", out, expected, 64));
}

// ============================================================================
//  TEST 3 -- Same keystream, but produced by the GPU kernel
// ----------------------------------------------------------------------------
//  Trick: XOR-ing 64 ZERO bytes with the keystream yields the keystream itself
//  (x ^ 0 == x). So encrypting an all-zero block is a direct way to read the
//  raw keystream out of the GPU and compare it to RFC 8439 section 2.3.2.
// ============================================================================
static bool test_block_keystream_gpu() {
    const uint8_t nonce[12] = {
        0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x4a, 0x00, 0x00, 0x00, 0x00
    };
    const uint8_t expected[64] = {
        0x10, 0xf1, 0xe7, 0xe4, 0xd1, 0x3b, 0x59, 0x15,
        0x50, 0x0f, 0xdd, 0x1f, 0xa3, 0x20, 0x71, 0xc4,
        0xc7, 0xd1, 0xf4, 0xc7, 0x33, 0xc0, 0x68, 0x03,
        0x04, 0x22, 0xaa, 0x9a, 0xc3, 0xd4, 0x6c, 0x4e,
        0xd2, 0x82, 0x64, 0x46, 0x07, 0x9f, 0xaa, 0x09,
        0x14, 0xc2, 0xd7, 0x05, 0xd9, 0x8b, 0x02, 0xa2,
        0xb5, 0x12, 0x9c, 0xd1, 0xde, 0x16, 0x4e, 0xb9,
        0xcb, 0xd0, 0x83, 0xe8, 0xa2, 0x50, 0x3c, 0x4e
    };

    uint8_t zeros[64] = {0};
    uint8_t out[64]   = {0};
    cudaError_t st = chacha20_xor_cuda(zeros, out, 64, RFC_KEY, nonce, /*counter=*/1);
    if (st != cudaSuccess) {
        printf("    GPU error: %s\n", cudaGetErrorString(st));
        return report("RFC 8439 2.3.2  block keystream (GPU kernel)", false);
    }
    return report("RFC 8439 2.3.2  block keystream (GPU kernel)",
                  bytes_equal("keystream", out, expected, 64));
}

// ============================================================================
//  TEST 4 -- Full encryption vector (RFC 8439, section 2.4.2)
// ----------------------------------------------------------------------------
//  The famous "sunscreen" paragraph (114 bytes -> 1 full block + a 50-byte
//  partial block) encrypted with counter = 1 must produce exactly the RFC's
//  ciphertext. This is the end-to-end proof on the GPU, and it specifically
//  exercises the partial-final-block path.
// ============================================================================
static bool test_encryption_gpu() {
    const uint8_t nonce[12] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x4a, 0x00, 0x00, 0x00, 0x00
    };
    const char* plaintext_str =
        "Ladies and Gentlemen of the class of '99: If I could offer you only "
        "one tip for the future, sunscreen would be it.";
    const size_t len = std::strlen(plaintext_str);   // 114 bytes

    const uint8_t expected[114] = {
        0x6e, 0x2e, 0x35, 0x9a, 0x25, 0x68, 0xf9, 0x80,
        0x41, 0xba, 0x07, 0x28, 0xdd, 0x0d, 0x69, 0x81,
        0xe9, 0x7e, 0x7a, 0xec, 0x1d, 0x43, 0x60, 0xc2,
        0x0a, 0x27, 0xaf, 0xcc, 0xfd, 0x9f, 0xae, 0x0b,
        0xf9, 0x1b, 0x65, 0xc5, 0x52, 0x47, 0x33, 0xab,
        0x8f, 0x59, 0x3d, 0xab, 0xcd, 0x62, 0xb3, 0x57,
        0x16, 0x39, 0xd6, 0x24, 0xe6, 0x51, 0x52, 0xab,
        0x8f, 0x53, 0x0c, 0x35, 0x9f, 0x08, 0x61, 0xd8,
        0x07, 0xca, 0x0d, 0xbf, 0x50, 0x0d, 0x6a, 0x61,
        0x56, 0xa3, 0x8e, 0x08, 0x8a, 0x22, 0xb6, 0x5e,
        0x52, 0xbc, 0x51, 0x4d, 0x16, 0xcc, 0xf8, 0x06,
        0x81, 0x8c, 0xe9, 0x1a, 0xb7, 0x79, 0x37, 0x36,
        0x5a, 0xf9, 0x0b, 0xbf, 0x74, 0xa3, 0x5b, 0xe6,
        0xb4, 0x0b, 0x8e, 0xed, 0xf2, 0x78, 0x5e, 0x42,
        0x87, 0x4d
    };

    std::vector<uint8_t> cipher(len);
    cudaError_t st = chacha20_xor_cuda(
        reinterpret_cast<const uint8_t*>(plaintext_str),
        cipher.data(), len, RFC_KEY, nonce, /*counter=*/1);
    if (st != cudaSuccess) {
        printf("    GPU error: %s\n", cudaGetErrorString(st));
        return report("RFC 8439 2.4.2  full encryption (GPU)", false);
    }
    return report("RFC 8439 2.4.2  full encryption (GPU)",
                  bytes_equal("ciphertext", cipher.data(), expected, len));
}

// ============================================================================
//  TEST 5 -- Decryption round trip (encrypt, then decrypt, recover original)
// ----------------------------------------------------------------------------
//  Because decryption IS encryption with the same key/nonce/counter, feeding
//  the ciphertext from TEST 4 back through the cipher must reproduce the
//  plaintext. This proves the symmetry property end to end on the GPU.
// ============================================================================
static bool test_roundtrip_gpu() {
    const uint8_t nonce[12] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x4a, 0x00, 0x00, 0x00, 0x00
    };
    const char* plaintext_str =
        "Ladies and Gentlemen of the class of '99: If I could offer you only "
        "one tip for the future, sunscreen would be it.";
    const size_t len = std::strlen(plaintext_str);

    std::vector<uint8_t> cipher(len), recovered(len);

    cudaError_t st = chacha20_xor_cuda(
        reinterpret_cast<const uint8_t*>(plaintext_str),
        cipher.data(), len, RFC_KEY, nonce, 1);
    if (st == cudaSuccess) {
        st = chacha20_xor_cuda(cipher.data(), recovered.data(), len,
                               RFC_KEY, nonce, 1);
    }
    if (st != cudaSuccess) {
        printf("    GPU error: %s\n", cudaGetErrorString(st));
        return report("Round trip  encrypt then decrypt recovers plaintext", false);
    }
    const bool ok = std::memcmp(recovered.data(), plaintext_str, len) == 0;
    return report("Round trip  encrypt then decrypt recovers plaintext", ok);
}

// ============================================================================
//  TEST 6 -- GPU vs CPU oracle across many awkward lengths
// ----------------------------------------------------------------------------
//  We fill a buffer with a deterministic pseudo-random pattern (no real RNG
//  needed; we just want non-trivial bytes) and encrypt it with BOTH the GPU and
//  the CPU reference for several lengths chosen to stress block boundaries:
//  empty-ish, sub-block, exactly one block, block+1, several blocks, and a
//  large multi-thread buffer. Every length must agree byte-for-byte.
// ============================================================================
static bool test_gpu_vs_cpu_many_lengths() {
    const uint8_t key[32] = {
        0x9a,0x1f,0x3c,0x77,0x52,0xab,0x10,0x4d,
        0x88,0x21,0x6e,0xff,0x03,0xc4,0xbe,0x59,
        0x12,0x34,0x56,0x78,0x9a,0xbc,0xde,0xf0,
        0x0f,0x1e,0x2d,0x3c,0x4b,0x5a,0x69,0x78
    };
    const uint8_t nonce[12] = {
        0x07,0x00,0x00,0x00, 0x40,0x41,0x42,0x43, 0x44,0x45,0x46,0x47
    };
    const uint32_t counter = 0;

    // Lengths picked to cover: partial block, exact block, block+1, multi-block,
    // and enough data to span many CUDA threads/blocks.
    const size_t lengths[] = { 1, 63, 64, 65, 127, 128, 200, 4096, 1 << 20 };

    bool all_ok = true;
    for (size_t li = 0; li < sizeof(lengths) / sizeof(lengths[0]); ++li) {
        const size_t len = lengths[li];

        // Deterministic filler: a simple byte ramp xored with the index high
        // bits. Not cryptographic -- just a varied, reproducible input.
        std::vector<uint8_t> input(len);
        for (size_t i = 0; i < len; ++i) {
            input[i] = (uint8_t)((i * 31u + (i >> 8)) & 0xff);
        }

        std::vector<uint8_t> gpu_out(len), cpu_out(len);

        cudaError_t st = chacha20_xor_cuda(input.data(), gpu_out.data(), len,
                                           key, nonce, counter);
        if (st != cudaSuccess) {
            printf("    GPU error at len=%zu: %s\n", len, cudaGetErrorString(st));
            all_ok = false;
            continue;
        }
        chacha20_xor_ref(input.data(), cpu_out.data(), len, key, nonce, counter);

        if (std::memcmp(gpu_out.data(), cpu_out.data(), len) != 0) {
            printf("    GPU/CPU disagree at len=%zu\n", len);
            all_ok = false;
        }
    }
    return report("GPU vs CPU oracle  (lengths 1..1MiB, all block boundaries)",
                  all_ok);
}

// ============================================================================
//  TEST 7 -- In-place encryption (host output buffer aliases the host input)
// ----------------------------------------------------------------------------
//  Many real callers encrypt a buffer in place to save memory. The HIGH-LEVEL
//  chacha20_xor_cuda() supports this: even when output == input on the host, it
//  stages through two SEPARATE device buffers, so the kernel (whose pointers
//  are __restrict__ and therefore must not alias) never actually sees aliasing.
//  This test proves the round trip by encrypting in place and decrypting back.
// ============================================================================
static bool test_inplace_gpu() {
    const uint8_t key[32] = {
        0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,
        0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0x10,
        0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,
        0x19,0x1a,0x1b,0x1c,0x1d,0x1e,0x1f,0x20
    };
    const uint8_t nonce[12] = {0,0,0,0,0,0,0,0,0,0,0,1};
    const size_t len = 1000;

    std::vector<uint8_t> buf(len), original(len);
    for (size_t i = 0; i < len; ++i) {
        buf[i] = original[i] = (uint8_t)(i & 0xff);
    }

    // Encrypt in place: input pointer == output pointer.
    cudaError_t st = chacha20_xor_cuda(buf.data(), buf.data(), len, key, nonce, 0);
    bool changed = (std::memcmp(buf.data(), original.data(), len) != 0);
    // Decrypt in place: must come back to the original.
    if (st == cudaSuccess) {
        st = chacha20_xor_cuda(buf.data(), buf.data(), len, key, nonce, 0);
    }
    bool restored = (std::memcmp(buf.data(), original.data(), len) == 0);

    if (st != cudaSuccess) {
        printf("    GPU error: %s\n", cudaGetErrorString(st));
    }
    return report("In-place  encrypt-in-place changes then restores buffer",
                  st == cudaSuccess && changed && restored);
}

// ============================================================================
//  run_all_tests -- run the whole suite, AND the results, report a summary.
// ============================================================================
bool run_all_tests() {
    printf("==================================================================\n");
    printf(" ChaCha20 CUDA -- correctness test suite (RFC 8439 known answers)\n");
    printf("==================================================================\n");

    bool ok = true;
    ok &= test_quarterround();
    ok &= test_block_keystream_cpu();
    ok &= test_block_keystream_gpu();
    ok &= test_encryption_gpu();
    ok &= test_roundtrip_gpu();
    ok &= test_gpu_vs_cpu_many_lengths();
    ok &= test_inplace_gpu();

    printf("------------------------------------------------------------------\n");
    printf(" RESULT: %s\n", ok ? "ALL TESTS PASSED" : "*** SOME TESTS FAILED ***");
    printf("==================================================================\n\n");
    return ok;
}
