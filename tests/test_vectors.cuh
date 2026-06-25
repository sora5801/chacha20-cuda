// ============================================================================
//  test_vectors.cuh  --  Interface to the known-answer test suite
// ============================================================================
//
//  Declares a single entry point, run_all_tests(), which executes every
//  correctness check (RFC 8439 known-answer vectors + GPU/CPU cross-checks +
//  round-trip tests) and prints a PASS/FAIL line for each. main.cu calls this
//  before running the demo: there is no point showcasing a cipher we have not
//  proven correct first.
//
//  Returns true only if EVERY test passed. main.cu treats a false return as a
//  hard failure and exits non-zero, so an automated build can detect breakage.
// ============================================================================

#ifndef TEST_VECTORS_CUH
#define TEST_VECTORS_CUH

bool run_all_tests();

#endif // TEST_VECTORS_CUH
