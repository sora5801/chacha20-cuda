// ============================================================================
//  demo.cuh  --  Interface to the showcase demo
// ============================================================================
//
//  Declares run_demo(), which main.cu calls after the tests pass. The demo is
//  the "show, don't tell" half of this study project: it prints a real
//  encrypt/decrypt round trip in human-readable form and then benchmarks raw
//  kernel throughput on the GPU so you can see how fast counter-mode ChaCha20
//  runs when every block is handled by its own thread.
//
//  Returns 0 on success, non-zero if a CUDA call failed during the demo.
// ============================================================================

#ifndef DEMO_CUH
#define DEMO_CUH

int run_demo();

#endif // DEMO_CUH
