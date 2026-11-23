/* parakeet_scene_opts must keep the ABI v11 layout (60 bytes on the usual
 * 64-bit and 32-bit targets), so a caller built against an older header, or a
 * binding that mirrors the struct (Go, Java, Python), is not overwritten by
 * parakeet_capi_scene_opts_default and is not read past its end.
 *
 * `struct v11_scene_opts` copies the v11 layout field by field. The test fills
 * one inside a wrapper with a canary after it, calls the library, and checks
 * that the canary is intact, that the size the library wrote is the v11 size,
 * and that a stream begun with these options behaves like one begun with NULL
 * options (both refuse an all-NULL call). No model needed. */
#include "parakeet_capi.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>

struct v11_sound_opts { int size; float window_sec, hop_sec, threshold_on, threshold_off, min_duration_sec; int top_k; };
struct v11_scene_opts {
    int size;
    int diar_latency;
    struct v11_sound_opts sound;
    int flags;
    float speaker_accept_threshold, speaker_margin, speaker_min_voice_sec, speaker_refresh_sec,
        speaker_max_voice_sec;
};

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { fprintf(stderr, "FAIL: %s\n", msg); ++fails; } } while (0)

int main(void) {
    CHECK(sizeof(parakeet_sound_opts) == sizeof(struct v11_sound_opts), "parakeet_sound_opts layout changed");
    CHECK(sizeof(parakeet_scene_opts) == sizeof(struct v11_scene_opts), "parakeet_scene_opts size changed");
    CHECK(offsetof(parakeet_scene_opts, speaker_max_voice_sec) ==
              offsetof(struct v11_scene_opts, speaker_max_voice_sec),
          "parakeet_scene_opts field offsets changed");

    struct { struct v11_scene_opts o; unsigned char canary[32]; } w;
    memset(&w, 0, sizeof w);
    memset(w.canary, 0xAB, sizeof w.canary);
    parakeet_capi_scene_opts_default((parakeet_scene_opts*)&w.o);
    int intact = 1;
    for (size_t i = 0; i < sizeof w.canary; ++i) intact &= w.canary[i] == 0xAB;
    CHECK(intact, "scene_opts_default wrote past the v11 struct");
    CHECK(w.o.size == (int)sizeof(struct v11_scene_opts), "scene_opts_default wrote another size");
    CHECK(w.o.flags == 0, "default flags are not 0");

    /* An all-NULL begin returns NULL with these options and with NULL options. */
    CHECK(parakeet_capi_scene_stream_begin(NULL, NULL, NULL, (const parakeet_scene_opts*)&w.o) == NULL,
          "all-NULL begin did not fail");
    CHECK(parakeet_capi_scene_stream_begin_lang(NULL, NULL, NULL, NULL, NULL, NULL, "en") == NULL,
          "all-NULL begin_lang did not fail");

    if (fails) { fprintf(stderr, "%d failure(s)\n", fails); return 1; }
    fprintf(stderr, "PASS (parakeet_scene_opts = %zu bytes)\n", sizeof(parakeet_scene_opts));
    return 0;
}
