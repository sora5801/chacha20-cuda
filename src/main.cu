// ============================================================================
//  main.cu  --  Program entry point
// ============================================================================
//
//  This is the file Visual Studio runs when you press F5 (or that ./chacha20
//  runs from the command line). Its job is intentionally tiny -- it is just an
//  orchestrator:
//
//      1. Print a banner and confirm a CUDA-capable GPU is present.
//      2. Run the correctness test suite (test_vectors.cu). If ANY test fails,
//         stop immediately with a non-zero exit code -- we refuse to "demo" a
//         cipher we have not proven correct.
//      3. Run the demonstration (demo.cu): a readable encrypt/decrypt round
//         trip followed by a throughput benchmark.
//
//  Keeping main() this small is deliberate: all the substance lives in focused,
//  independently-readable translation units, and main just wires them together.
// ============================================================================

#include "chacha20.cuh"
#include "../tests/test_vectors.cuh"
#include "../demo/demo.cuh"

#include <cstdio>

// ----------------------------------------------------------------------------
//  Verify a usable GPU exists before we attempt any CUDA work, so the failure
//  message is friendly ("no GPU") instead of a cryptic error deep in a kernel
//  launch. cudaGetDeviceCount is the canonical "is there a GPU?" probe.
// ----------------------------------------------------------------------------
static bool ensure_cuda_device() {
    int count = 0;
    cudaError_t st = cudaGetDeviceCount(&count);
    if (st != cudaSuccess) {
        printf("No CUDA-capable device available: %s\n", cudaGetErrorString(st));
        return false;
    }
    if (count == 0) {
        printf("No CUDA-capable device found (device count is 0).\n");
        return false;
    }

    // Bind to device 0 and announce it. A multi-GPU box could select another,
    // but for a study demo device 0 is the right default. We check the return
    // even though ordinal 0 is provably valid here (count >= 1 just above): the
    // bind step decides which GPU every later kernel runs on, so it deserves the
    // same error-checking discipline as the count probe.
    cudaError_t set_st = cudaSetDevice(0);
    if (set_st != cudaSuccess) {
        printf("cudaSetDevice(0) failed: %s\n", cudaGetErrorString(set_st));
        return false;
    }
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        printf("Using GPU 0: %s (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);
    }
    return true;
}

int main() {
    printf("##################################################################\n");
    printf("#           ChaCha20 stream cipher, implemented in CUDA          #\n");
    printf("#        (study build: tests first, then a live demo)            #\n");
    printf("##################################################################\n\n");

    // Step 1: GPU presence check.
    if (!ensure_cuda_device()) {
        return 2;   // distinct exit code: environment problem, not a logic bug
    }

    // Step 2: correctness gate. We will not proceed to the demo unless every
    // RFC 8439 known-answer and cross-check test passes.
    if (!run_all_tests()) {
        printf("Aborting: correctness tests failed.\n");
        return 1;
    }

    // Step 3: the showcase.
    int demo_rc = run_demo();

    printf("Done.\n");
    return demo_rc;
}
