#pragma once
// Test-only diagnostics for the streaming C-API's mel sliding window.
//
// Deliberately NOT in parakeet_capi.h: these are a regression-test seam, not
// part of the ABI. They live in their own internal header so the definition in
// parakeet_capi.cpp and the test that calls them cannot drift apart.
//
// Both are per-stream high-water marks, reset by simply beginning a new stream:
//   max_copy         floats moved by a single append_mel_frames call. An append
//                    that only copies its own new frames keeps this
//                    proportional to one chunk, so it stays bounded however
//                    long the session runs; the pre-fix whole-buffer rebuild
//                    made it grow with the session (O(N^2) total).
//   max_frames_held  mel frames the window retained. Bounded by the chunk
//                    schedule's left context, not by session length.
//
// See tests/test_streaming_mel_bounded.cpp.

#include "parakeet_capi.h"

#ifdef __cplusplus
extern "C" {
#endif

unsigned long long parakeet_capi_test_mel_max_copy(parakeet_stream* s);
unsigned long long parakeet_capi_test_mel_max_frames_held(parakeet_stream* s);

#ifdef __cplusplus
}
#endif
