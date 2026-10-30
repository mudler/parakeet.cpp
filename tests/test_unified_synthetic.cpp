// CPU-only checks that need no checkpoint, so they run in CI and are never skipped.
// Each case writes a small KV-only GGUF (no weights: the mel frontend falls back to
// zero filterbanks, and the loader reads only the config). Covers:
//   - the unified valid-length gate: trimmed mel frame count T == floor(S/hop),
//     including sub-hop and empty clips, on the CPU and the GPU mel frontends;
//   - the gate off (and a non-unified style): T == 1 + S/hop, as before;
//   - StreamingMel / StreamingEncoder throw (not abort) on a per_feature or a
//     non-streaming config;
//   - a unified GGUF with a KV of the wrong type is refused at load, not aborted;
//   - a buffered window out of bounds (negative, zero chunk, over 4000 frames, values
//     near INT_MAX) is refused by BufferedStreamingSession and by the C-API stream begin,
//     before any weight is read.
#include "buffered_streaming.hpp"
#include "model_loader.hpp"
#include "mel.hpp"
#include "mel_gpu.hpp"
#include "streaming_encoder.hpp"
#include "subsampling.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "parakeet_capi.h"

#include "ggml.h"
#include "gguf.h"

using namespace pk;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

namespace {

constexpr int kHop = 160;
constexpr int kNMels = 8;

struct Spec {
    bool unified = true;        // style chunked_limited_with_rc (else "regular")
    bool valid_length = true;   // parakeet.unified.valid_length
    std::string normalize = "per_feature";
};

// Writes the KV-only GGUF for spec to path. `extra` may replace keys for the wrong-type cases.
static bool write_gguf(const std::string& path, const Spec& s,
                       const std::function<void(gguf_context*)>& extra = nullptr) {
    gguf_context* g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "parakeet");
    gguf_set_val_str(g, "parakeet.arch", "rnnt");
    gguf_set_val_u32(g, "parakeet.encoder.feat_in", kNMels);
    gguf_set_val_u32(g, "parakeet.encoder.d_model", 8);
    gguf_set_val_u32(g, "parakeet.encoder.n_layers", 1);
    gguf_set_val_u32(g, "parakeet.encoder.n_heads", 2);
    gguf_set_val_u32(g, "parakeet.encoder.ff_dim", 16);
    gguf_set_val_u32(g, "parakeet.encoder.conv_kernel", 9);
    gguf_set_val_u32(g, "parakeet.encoder.subsampling_factor", 8);
    gguf_set_val_u32(g, "parakeet.encoder.subsampling_conv_channels", 4);
    gguf_set_val_str(g, "parakeet.encoder.att_context_style", s.unified ? "chunked_limited_with_rc" : "regular");
    gguf_set_val_u32(g, "parakeet.vocab_size", 8);  // the loader requires a vocabulary
    gguf_set_val_u32(g, "parakeet.preprocessor.sample_rate", 16000);
    gguf_set_val_u32(g, "parakeet.preprocessor.n_mels", kNMels);
    gguf_set_val_u32(g, "parakeet.preprocessor.n_fft", 512);
    gguf_set_val_u32(g, "parakeet.preprocessor.win_length", 400);
    gguf_set_val_u32(g, "parakeet.preprocessor.hop_length", kHop);
    gguf_set_val_f32(g, "parakeet.preprocessor.preemph", 0.97f);
    gguf_set_val_f32(g, "parakeet.preprocessor.mag_power", 2.0f);
    gguf_set_val_f32(g, "parakeet.preprocessor.log_zero_guard", 5.960464477539063e-08f);
    gguf_set_val_str(g, "parakeet.preprocessor.normalize", s.normalize.c_str());
    if (s.unified) {
        gguf_set_val_bool(g, "parakeet.unified.present", true);
        gguf_set_val_bool(g, "parakeet.unified.valid_length", s.valid_length);
        gguf_set_val_str(g, "parakeet.encoder.conv_context_style", "dcc");
        const int32_t presets[6] = {70, 13, 13, 70, 7, 7};
        gguf_set_arr_data(g, "parakeet.unified.presets", GGUF_TYPE_INT32, presets, 6);
        gguf_set_val_u32(g, "parakeet.unified.default_preset", 1);
    } else if (s.valid_length) {
        // Not a unified checkpoint: the key is written but must be ignored.
        gguf_set_val_bool(g, "parakeet.unified.valid_length", true);
    }
    // The mel kernel reads the filterbank and the window as tensors: small
    // deterministic ones (the values do not matter for the frame counts).
    ggml_init_params ip{ 4u * 1024u * 1024u, nullptr, /*no_alloc*/ false };
    ggml_context* ctx = ggml_init(ip);
    const int n_bins = 512 / 2 + 1;
    ggml_tensor* fb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_bins, kNMels);  // ne = [bins, mels]
    ggml_tensor* win = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 400);
    float* fbd = (float*)fb->data;
    for (int m = 0; m < kNMels; ++m)
        for (int b = 0; b < n_bins; ++b)
            fbd[(size_t)m * n_bins + b] = (b % (m + 3) == 0) ? 0.01f * (float)(m + 1) : 0.0f;
    float* wd = (float*)win->data;
    for (int i = 0; i < 400; ++i) wd[i] = 0.5f - 0.5f * std::cos(2.0f * 3.14159265f * (float)i / 399.0f);
    ggml_set_name(fb, "preprocessor.featurizer.fb");
    ggml_set_name(win, "preprocessor.featurizer.window");
    gguf_add_tensor(g, fb);
    gguf_add_tensor(g, win);
    if (extra) extra(g);
    const bool ok = gguf_write_to_file(g, path.c_str(), /*only_meta=*/false);
    gguf_free(g);
    ggml_free(ctx);
    return ok;
}

// Removes the file on every exit path.
struct TempFile {
    std::string path;
    explicit TempFile(const std::string& tag) {
        path = "pk_unified_synth_" + tag + ".gguf";
    }
    ~TempFile() { std::remove(path.c_str()); }
};

static bool load_spec(const Spec& s, const std::string& tag, ModelLoader& ml,
                      const std::function<void(gguf_context*)>& extra = nullptr) {
    TempFile tf(tag);
    if (!write_gguf(tf.path, s, extra)) return false;
    return ml.load(tf.path);
}

static std::vector<float> ramp(int S) {
    std::vector<float> x((size_t)S);
    for (int i = 0; i < S; ++i) x[(size_t)i] = 0.1f * (float)((i * 7919) % 201 - 100) / 100.0f;
    return x;
}

// Frame counts the trimmed (valid-length) frontend must produce: floor(S/hop).
static const int kSizes[] = {0, 1, kHop - 1, kHop, kHop + 1, 5 * kHop, 5 * kHop + 37};

}  // namespace

int main() {
    // ---- subsampling valid length: the trimmed mel has no extra trailing frame ----
    // A unified mel is trimmed to its valid length, so every frame is valid and the
    // valid output length equals the full output length. A mel of 8k+1 frames
    // (for example 625, a clip of 100000..100159 samples) used to lose its last
    // output frame because the entry length was taken as T-1. The gate off keeps
    // the T-1 convention of the untrimmed offline mel.
    {
        ModelLoader on;
        CHECK(load_spec(Spec{}, "sub_on", on));
        Subsampling sub_on(on);
        bool all_equal = true;
        for (int T = 1; T <= 2000; ++T)
            if (sub_on.valid_out_len(T) != sub_on.subsample_len(T)) all_equal = false;
        CHECK(all_equal);
        CHECK(sub_on.valid_out_len(625) == 79);
        CHECK(sub_on.valid_out_len(624) == 78);
        CHECK(sub_on.valid_out_len(626) == 79);

        Spec off_spec;
        off_spec.valid_length = false;
        ModelLoader off;
        CHECK(load_spec(off_spec, "sub_off", off));
        Subsampling sub_off(off);
        CHECK(sub_off.valid_out_len(625) == 78);   // T-1 = 624 -> 312 -> 156 -> 78
        CHECK(sub_off.valid_out_len(626) == 79);
        CHECK(sub_off.offline_entry_valid(625) == 624);
        CHECK(sub_on.offline_entry_valid(625) == 625);
    }

    // ---- the gate on: T == floor(S/hop) on the CPU and the GPU frontends ----
    {
        ModelLoader ml;
        CHECK(load_spec(Spec{}, "gate_on", ml));
        CHECK(ml.config().unified.present);
        CHECK(ml.config().unified.valid_length);
        MelFrontend cpu(ml);
        GpuMel gpu(ml);
        for (int S : kSizes) {
            const std::vector<float> x = ramp(S);
            const int want = S / kHop;
            std::vector<float> feats; int n_mels = 0, T = -1;
            cpu.compute(x, feats, n_mels, T);
            CHECK(T == want);
            CHECK(feats.size() == (size_t)n_mels * (size_t)T);
            std::vector<float> gfeats; int gn = 0, gT = -1;
            gpu.compute(x, gfeats, gn, gT);
            CHECK(gT == want);
            CHECK(gfeats.size() == (size_t)gn * (size_t)gT);
            if (T == gT && T > 0) {
                // The two frontends agree on the trimmed matrix.
                float worst = 0.0f;
                for (size_t i = 0; i < feats.size(); ++i)
                    worst = std::max(worst, std::fabs(feats[i] - gfeats[i]));
                CHECK(worst < 1e-3f);
            }
            std::fprintf(stderr, "[gate on] S=%d T=%d (want %d)\n", S, T, want);
        }
    }

    // ---- the trimmed matrix is the first S/hop columns of the untrimmed one ----
    // The normalization uses the same valid count either way, so the values must
    // be identical, not only the shape.
    {
        ModelLoader on, off;
        CHECK(load_spec(Spec{}, "cmp_on", on));
        Spec s; s.valid_length = false;
        CHECK(load_spec(s, "cmp_off", off));
        MelFrontend fon(on), foff(off);
        const std::vector<float> x = ramp(5 * kHop + 37);
        std::vector<float> a, b; int na = 0, nb = 0, Ta = 0, Tb = 0;
        fon.compute(x, a, na, Ta);
        foff.compute(x, b, nb, Tb);
        CHECK(na == nb && Ta == 5 && Tb == 6);
        bool same = true;
        for (int m = 0; m < na && same; ++m)
            for (int t = 0; t < Ta && same; ++t)
                same = a[(size_t)m * Ta + t] == b[(size_t)m * Tb + t];
        CHECK(same);
    }

    // ---- the gate off: T == 1 + S/hop (and 0 for an empty clip), as before ----
    {
        ModelLoader ml;
        Spec s; s.valid_length = false;
        CHECK(load_spec(s, "gate_off", ml));
        CHECK(!ml.config().unified.valid_length);
        MelFrontend cpu(ml);
        GpuMel gpu(ml);
        for (int S : kSizes) {
            const std::vector<float> x = ramp(S);
            const int want = (S == 0) ? 0 : 1 + S / kHop;
            std::vector<float> feats; int n_mels = 0, T = -1;
            cpu.compute(x, feats, n_mels, T);
            CHECK(T == want);
            std::vector<float> gfeats; int gn = 0, gT = -1;
            gpu.compute(x, gfeats, gn, gT);
            CHECK(gT == want);
        }
    }

    // ---- a non-unified style ignores the key: T == 1 + S/hop ----
    {
        ModelLoader ml;
        Spec s; s.unified = false; s.valid_length = true;
        CHECK(load_spec(s, "regular_style", ml));
        CHECK(!ml.config().unified.present);
        MelFrontend cpu(ml);
        const std::vector<float> x = ramp(5 * kHop + 37);
        std::vector<float> feats; int n_mels = 0, T = -1;
        cpu.compute(x, feats, n_mels, T);
        CHECK(T == 1 + 5);
    }

    // ---- a file without the key loads with the gate off ----
    {
        ModelLoader ml;
        Spec s;
        CHECK(load_spec(s, "no_key", ml, [](gguf_context* g) { gguf_remove_key(g, "parakeet.unified.valid_length"); }));
        CHECK(ml.config().unified.present);
        CHECK(!ml.config().unified.valid_length);
    }

    // ---- StreamingMel refuses a per_feature config with an exception ----
    {
        ModelLoader ml;
        Spec s; s.unified = false; s.normalize = "per_feature";
        CHECK(load_spec(s, "pf", ml));
        bool threw = false;
        try { StreamingMel sm(ml); } catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
    }
    {
        ModelLoader ml;
        Spec s; s.unified = false; s.normalize = "NA";
        CHECK(load_spec(s, "na", ml));
        bool threw = false;
        try { StreamingMel sm(ml); } catch (const std::runtime_error&) { threw = true; }
        CHECK(!threw);
    }

    // ---- StreamingEncoder refuses a non-streaming config with an exception ----
    {
        ModelLoader ml;
        Spec s; s.unified = false;
        CHECK(load_spec(s, "nonstream", ml));
        bool threw = false;
        try { StreamingEncoder se(ml); } catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
    }

    // ---- a wrong KV type is a load error, not an abort ----
    {
        ModelLoader ml;
        Spec s;
        CHECK(!load_spec(s, "bad_default", ml, [](gguf_context* g) {
            gguf_remove_key(g, "parakeet.unified.default_preset");
            gguf_set_val_str(g, "parakeet.unified.default_preset", "1");
        }));
    }
    {
        ModelLoader ml;
        Spec s;
        CHECK(!load_spec(s, "bad_presets", ml, [](gguf_context* g) {
            gguf_remove_key(g, "parakeet.unified.presets");
            gguf_set_val_u32(g, "parakeet.unified.presets", 70);
        }));
    }
    {
        ModelLoader ml;
        Spec s;
        CHECK(!load_spec(s, "bad_valid_length", ml, [](gguf_context* g) {
            gguf_remove_key(g, "parakeet.unified.valid_length");
            gguf_set_val_i32(g, "parakeet.unified.valid_length", 1);
        }));
    }
    {
        ModelLoader ml;
        Spec s;
        CHECK(!load_spec(s, "bad_conv_style", ml, [](gguf_context* g) {
            gguf_remove_key(g, "parakeet.encoder.conv_context_style");
            gguf_set_val_i32(g, "parakeet.encoder.conv_context_style", 3);
        }));
    }
    {
        // A presets array of the wrong element type (FLOAT32).
        ModelLoader ml;
        Spec s;
        CHECK(!load_spec(s, "bad_presets_elem", ml, [](gguf_context* g) {
            gguf_remove_key(g, "parakeet.unified.presets");
            const float f[3] = {70.0f, 7.0f, 7.0f};
            gguf_set_arr_data(g, "parakeet.unified.presets", GGUF_TYPE_FLOAT32, f, 3);
        }));
    }

    // ---- buffered window bounds: refused before any weight is read ----
    {
        ModelLoader ml;
        CHECK(load_spec(Spec{}, "window_ctor", ml));
        auto refused = [&](int L, int C, int R) {
            try {
                BufferedStreamingSession sess(ml, L, C, R);
            } catch (const std::invalid_argument&) {
                return true;
            } catch (...) {
            }
            return false;
        };
        CHECK(refused(-1, 7, 7));
        CHECK(refused(70, 0, 13));
        CHECK(refused(0, 4001, 0));
        CHECK(refused(2147483647, 2147483647, 2147483647));
    }
    {
        TempFile tf("window_capi");
        CHECK(write_gguf(tf.path, Spec{}));
        parakeet_ctx* ctx = parakeet_capi_load(tf.path.c_str());
        CHECK(ctx != nullptr);
        if (ctx) {
            // {L, C, R, the message must contain}: the total limit or the minimum sizes.
            struct Case { int L, C, R; const char* want; };
            const Case cases[] = {
                {0, 0, 0, "left >= 0, chunk >= 1"},
                {-1, 7, 7, "left >= 0, chunk >= 1"},
                {0, 4001, 0, "4000 frames"},
                {2147483647, 2147483647, 2147483647, "4000 frames"},
                {2147483647, 1, 0, "4000 frames"},
            };
            for (const Case& c : cases) {
                parakeet_stream_opts o;
                parakeet_capi_stream_opts_default(&o);
                o.preset = PARAKEET_STREAM_PRESET_CUSTOM;
                o.left_frames = c.L;
                o.chunk_frames = c.C;
                o.right_frames = c.R;
                parakeet_stream* st = parakeet_capi_stream_begin_opts(ctx, &o);
                CHECK(st == nullptr);
                const char* err = parakeet_capi_last_error(ctx);
                CHECK(err != nullptr && std::strstr(err, c.want) != nullptr);
                if (st) parakeet_capi_stream_free(st);
            }
            parakeet_capi_free(ctx);
        }
    }

    if (failures) {
        std::fprintf(stderr, "test_unified_synthetic: %d failure(s)\n", failures);
        return 1;
    }
    std::puts("test_unified_synthetic: OK");
    return 0;
}
