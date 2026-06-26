// ============================================================================
//  demo.cu  --  Interactive showcase: round-trip encryption + throughput bench
// ============================================================================
//
//  Two acts:
//
//    ACT 1 -- "It works and you can read it."
//      Encrypt a readable message, print the ciphertext as hex, decrypt it back,
//      and confirm we recovered the original text. This is the human-facing
//      proof that complements the byte-exact test suite.
//
//    ACT 2 -- "And here is how fast the GPU does it."
//      Allocate a large buffer ON the device, run the kernel many times, and
//      time ONLY the kernel with CUDA events (no PCIe copies in the measured
//      region). Report throughput in GB/s and blocks/second so you get an
//      intuition for the parallel speedup counter mode buys you.
//
//  Everything here uses the public API from chacha20.cuh; the demo never
//  reaches into kernel internals, which is exactly how a real caller would use
//  the library.
// ============================================================================

#include "demo.cuh"
#include "chacha20.cuh"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>

// ----------------------------------------------------------------------------
//  Local helper: print bytes as hex with a label, 16 per line.
// ----------------------------------------------------------------------------
static void dump_hex(const char* label, const uint8_t* data, size_t len) {
    printf("  %s:\n", label);
    for (size_t i = 0; i < len; ++i) {
        if (i % 16 == 0) printf("    ");
        printf("%02x ", data[i]);
        if (i % 16 == 15) printf("\n");
    }
    if (len % 16 != 0) printf("\n");
}

// ----------------------------------------------------------------------------
//  ACT 1 -- Encrypt a message, show it, decrypt it, verify.
// ----------------------------------------------------------------------------
static int demo_roundtrip() {
    printf("------------------------------------------------------------------\n");
    printf(" DEMO 1: Encrypt -> show ciphertext -> decrypt -> verify\n");
    printf("------------------------------------------------------------------\n");

    // A made-up 256-bit key and 96-bit nonce. In real use the key is secret and
    // the nonce must be unique for every message under that key.
    const uint8_t key[32] = {
        0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,
        0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff,
        0x0f,0x1e,0x2d,0x3c,0x4b,0x5a,0x69,0x78,
        0x87,0x96,0xa5,0xb4,0xc3,0xd2,0xe1,0xf0
    };
    const uint8_t nonce[12] = {
        0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x01, 0x02,0x03,0x04,0x05
    };
    const uint32_t counter = 1;   // matches the RFC convention of starting at 1

    const char* message =
        "ChaCha20 on CUDA: every 64-byte block is encrypted by its own GPU "
        "thread, fully in parallel. This sentence is the plaintext.";
    const size_t len = std::strlen(message);

    std::vector<uint8_t> ciphertext(len);
    std::vector<uint8_t> decrypted(len);

    printf("  Plaintext (%zu bytes):\n    \"%s\"\n\n", len, message);

    // --- Encrypt ---
    cudaError_t st = chacha20_xor_cuda(
        reinterpret_cast<const uint8_t*>(message),
        ciphertext.data(), len, key, nonce, counter);
    if (st != cudaSuccess) {
        printf("  encryption failed: %s\n", cudaGetErrorString(st));
        return 1;
    }
    dump_hex("Ciphertext (hex)", ciphertext.data(), len);

    // --- Decrypt (same call, ciphertext in) ---
    st = chacha20_xor_cuda(ciphertext.data(), decrypted.data(), len,
                           key, nonce, counter);
    if (st != cudaSuccess) {
        printf("  decryption failed: %s\n", cudaGetErrorString(st));
        return 1;
    }

    // The decrypted bytes are not NUL-terminated, so print exactly `len` chars.
    printf("\n  Decrypted back to plaintext:\n    \"%.*s\"\n",
           (int)len, decrypted.data());

    const bool ok = std::memcmp(decrypted.data(), message, len) == 0;
    printf("\n  Round trip %s\n\n", ok ? "SUCCEEDED (decrypted == original)"
                                       : "FAILED");
    return ok ? 0 : 1;
}

// ----------------------------------------------------------------------------
//  ACT 2 -- Kernel-only throughput benchmark.
//
//  We measure pure compute, deliberately excluding host<->device copies, by:
//    1. allocating the input/output buffers directly on the device,
//    2. doing a warm-up launch (the first launch pays one-time costs like
//       JIT/module load that would otherwise pollute the measurement),
//    3. timing N launches with cudaEvent timers (GPU-side timestamps, far more
//       accurate for kernel timing than a host clock),
//    4. converting elapsed time into GB/s and blocks/s.
// ----------------------------------------------------------------------------
static int demo_benchmark() {
    printf("------------------------------------------------------------------\n");
    printf(" DEMO 2: Kernel-only throughput benchmark\n");
    printf("------------------------------------------------------------------\n");

    // Report what we are running on so the numbers have context.
    int dev = 0;
    cudaDeviceProp prop;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        printf("  Device           : %s (compute capability %d.%d)\n",
               prop.name, prop.major, prop.minor);
        printf("  SMs              : %d\n", prop.multiProcessorCount);
        printf("  Global memory    : %.1f GiB\n",
               (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }

    // 256 MiB of data: big enough to saturate the GPU and dwarf launch overhead,
    // small enough to fit comfortably in an 8 GiB card alongside the output.
    const size_t len = (size_t)256 * 1024 * 1024;
    const int    iterations = 50;

    const uint8_t key[32] = {
        0x9a,0x1f,0x3c,0x77,0x52,0xab,0x10,0x4d,
        0x88,0x21,0x6e,0xff,0x03,0xc4,0xbe,0x59,
        0x12,0x34,0x56,0x78,0x9a,0xbc,0xde,0xf0,
        0x0f,0x1e,0x2d,0x3c,0x4b,0x5a,0x69,0x78
    };
    const uint8_t nonce[12] = {0,1,2,3,4,5,6,7,8,9,10,11};
    ChaCha20State init = chacha20_init_state(key, nonce, /*counter=*/0);

    // --- Device buffers (two allocations, both freed before returning) ------
    uint8_t* d_in  = nullptr;
    uint8_t* d_out = nullptr;
    cudaError_t st;

    st = cudaMalloc((void**)&d_in, len);
    if (st != cudaSuccess) { printf("  cudaMalloc(in) failed: %s\n",
                                    cudaGetErrorString(st)); return 1; }
    st = cudaMalloc((void**)&d_out, len);
    if (st != cudaSuccess) { printf("  cudaMalloc(out) failed: %s\n",
                                    cudaGetErrorString(st));
                             cudaFree(d_in); return 1; }

    // Fill the input with something non-trivial. cudaMemset is plenty -- the
    // cipher's speed does not depend on the input values. We check its return:
    // this project's whole thesis (see the CUDA_CHECK note in chacha20.cu) is
    // that ignored CUDA errors are how kernels "silently do nothing", so the
    // benchmark holds itself to the same standard it preaches. (We cannot reuse
    // CUDA_CHECK here -- it lives in chacha20.cu and returns cudaError_t, while
    // this function returns int -- so we mirror the local cudaMalloc checks.)
    st = cudaMemset(d_in, 0xA5, len);
    if (st != cudaSuccess) {
        printf("  cudaMemset failed: %s\n", cudaGetErrorString(st));
        cudaFree(d_in); cudaFree(d_out); return 1;
    }

    // --- CUDA event timers --------------------------------------------------
    // Events are recorded into the stream and timestamped by the GPU itself, so
    // the measured interval reflects actual device execution, not host jitter.
    cudaEvent_t start = nullptr, stop = nullptr;
    if (cudaEventCreate(&start) != cudaSuccess ||
        cudaEventCreate(&stop)  != cudaSuccess) {
        printf("  cudaEventCreate failed\n");
        cudaEventDestroy(start); cudaEventDestroy(stop);
        cudaFree(d_in); cudaFree(d_out); return 1;
    }

    // Warm-up: absorbs first-launch one-time costs so they don't skew timing.
    // If even this launch fails, every measured number below would be garbage,
    // so we stop here with a clear message instead of printing a bogus GB/s.
    st = chacha20_xor_device(d_in, d_out, len, init, 256, 0);
    if (st == cudaSuccess) st = cudaDeviceSynchronize();
    if (st != cudaSuccess) {
        printf("  warm-up launch failed: %s\n", cudaGetErrorString(st));
        cudaEventDestroy(start); cudaEventDestroy(stop);
        cudaFree(d_in); cudaFree(d_out); return 1;
    }

    // --- Timed region: `iterations` kernel launches back to back ------------
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; ++i) {
        chacha20_xor_device(d_in, d_out, len, init, 256, 0);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);   // block host until `stop` has been reached

    float ms = 0.0f;
    // If timing failed, ms would stay 0 and the throughput math below would
    // divide by zero and print "inf" -- guard against reporting a meaningless
    // number, which is exactly the silent-garbage failure mode we warn about.
    st = cudaEventElapsedTime(&ms, start, stop);
    if (st != cudaSuccess || ms <= 0.0f) {
        printf("  timing failed (cudaEventElapsedTime): %s\n",
               cudaGetErrorString(st));
        cudaEventDestroy(start); cudaEventDestroy(stop);
        cudaFree(d_in); cudaFree(d_out); return 1;
    }

    // --- Convert to human numbers -------------------------------------------
    const double seconds      = (ms / 1000.0);
    const double total_bytes  = (double)len * (double)iterations;
    const double gb_per_sec   = total_bytes / seconds / 1.0e9;   // decimal GB
    const double per_iter_ms  = (double)ms / (double)iterations;
    const double blocks       = (double)((len + 63) / 64) * iterations;
    const double gblocks_sec  = blocks / seconds / 1.0e9;

    printf("\n  Buffer size      : %zu MiB\n", len / (1024 * 1024));
    printf("  Iterations       : %d\n", iterations);
    printf("  Total processed  : %.2f GiB\n",
           total_bytes / (1024.0 * 1024.0 * 1024.0));
    printf("  Time / iteration : %.3f ms\n", per_iter_ms);
    printf("  Throughput       : %.2f GB/s\n", gb_per_sec);
    printf("  Block rate       : %.3f billion 64-byte blocks/s\n", gblocks_sec);
    printf("  (Kernel time only; host<->device PCIe copies are excluded.)\n\n");

    // --- Cleanup ------------------------------------------------------------
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

// ----------------------------------------------------------------------------
//  run_demo -- public entry point: run both acts in order.
// ----------------------------------------------------------------------------
int run_demo() {
    printf("\n==================================================================\n");
    printf(" ChaCha20 CUDA -- demonstration\n");
    printf("==================================================================\n");

    int rc = demo_roundtrip();
    if (rc == 0) {
        rc = demo_benchmark();
    }
    return rc;
}
