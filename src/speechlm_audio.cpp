#include "speechlm_audio.hpp"
#include "backend.hpp"
#include "common.hpp"
#include "ggml.h"
#include "ggml_graph.hpp"
#include "graph_builder.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pk {

namespace {

// Fixed sinusoidal speaker table S [n_speakers, d] (sin on even columns, cos on odd),
// stored transposed (out[c * n_spk + s]) so it can be the weight of a mul_mat.
std::vector<float> speaker_table_t(int n_spk, int d) {
    const float step = (float)(-(std::log(10000.0) / (double)d));
    std::vector<float> t((size_t)n_spk * d);
    for (int s = 0; s < n_spk; ++s) {
        for (int i = 0; i < d / 2; ++i) {
            const float inv_freq = std::exp((float)(2 * i) * step);
            const float a = (float)s * inv_freq;
            t[(size_t)(2 * i) * n_spk + s] = std::sin(a);
            t[(size_t)(2 * i + 1) * n_spk + s] = std::cos(a);
        }
    }
    return t;
}

ggml_tensor* affine_norm(ggml_context* ctx, const ModelLoader& ml, ggml_tensor* x,
                         const std::string& name) {
    ggml_tensor* g = clone_weight(ctx, ml, (name + ".weight").c_str());
    ggml_tensor* b = clone_weight(ctx, ml, (name + ".bias").c_str());
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, 1e-5f), g), b);
}

ggml_tensor* input_f32(ggml_context* ctx, GraphInputPool& pool, const float* data, int64_t ne0, int64_t ne1) {
    std::vector<float>& buf = pool.alloc_f32((size_t)(ne0 * ne1));
    std::copy(data, data + (size_t)(ne0 * ne1), buf.begin());
    int64_t ne[2] = {ne0, ne1};
    return graph_input_tensor(ctx, GGML_TYPE_F32, 2, ne, buf.data(), buf.size() * sizeof(float));
}

}  // namespace

std::unique_ptr<SpeechLmAudio> SpeechLmAudio::load(const std::string& audio_path,
                                                   const std::string& audio_component,
                                                   const std::string& diar_path,
                                                   const std::string& diar_component,
                                                   std::string* err) {
    auto fail = [&](const std::string& m) -> std::unique_ptr<SpeechLmAudio> {
        if (err) *err = m;
        return nullptr;
    };
    std::unique_ptr<SpeechLmAudio> m(new (std::nothrow) SpeechLmAudio());
    if (!m) return fail("out of memory");
    if (!(audio_component.empty() ? m->audio_.load(audio_path)
                                  : m->audio_.load_component(audio_path, audio_component)))
        return fail("cannot load the speech LLM audio model from " + audio_path);
    const ParakeetConfig& ac = m->audio_.config();
    if (ac.arch != "speechlm_audio" || !ac.speechlm.present)
        return fail("not a speech LLM audio model (arch '" + ac.arch + "'): " + audio_path);
    if (!(diar_component.empty() ? m->diar_.load(diar_path)
                                 : m->diar_.load_component(diar_path, diar_component)))
        return fail("cannot load the diarization model from " + diar_path);
    const ParakeetConfig& dc = m->diar_.config();
    if (dc.arch != "diarization" || !dc.diarization.present)
        return fail("not a diarization model (arch '" + dc.arch + "'): " + diar_path);
    if (ac.normalize == "per_feature")
        return fail("the audio model must store the raw log-mel (normalize NA); the engine "
                    "normalises the ASR branch itself: " + audio_path);
    if (dc.n_mels != ac.n_mels || dc.subsampling_factor != ac.subsampling_factor)
        return fail("the audio and diarization models disagree on mel bins or subsampling");
    if (dc.diarization.n_speakers != ac.speechlm.num_speakers)
        return fail("the audio and diarization models disagree on the number of speakers");
    if (ac.subsampling_factor == 0 || ac.d_model % 2 != 0)
        return fail("invalid speech LLM audio config");
    try {
        ensure_weights_realized(m->audio_);
        ensure_weights_realized(m->diar_);
        m->mel_ = std::make_unique<MelFrontend>(m->audio_);
        m->encoder_ = std::make_unique<DiarizationEncoder>(m->audio_);
        // The graphs below look these up by name; fail at load, not at the first call.
        for (const char* t : {"speechlm.asr_norm.weight", "speechlm.asr_norm.bias",
                              "speechlm.diar_norm.weight", "speechlm.diar_norm.bias",
                              "speechlm.proj.weight", "speechlm.proj.bias"})
            if (!m->audio_.tensor(t)) return fail(std::string("missing tensor ") + t);
    } catch (const std::exception& e) {
        return fail(e.what());
    }
    m->subsampling_ = (int)ac.subsampling_factor;
    return m;
}

int SpeechLmAudio::tokens_for_samples(long long n_samples) const {
    const long long hop = audio_.config().hop_length;
    const long long valid = hop > 0 ? n_samples / hop : 0;
    return (int)((valid + subsampling_ - 1) / subsampling_);
}

// Encoder windows over the normalised, padded mel [n_mels, T_padded]. Writes
// hidden [n_frames, d_model]; every frame comes from the window it belongs to.
void SpeechLmAudio::asr_chunks(const std::vector<float>& mel, int T_padded,
                               std::vector<float>& hidden) const {
    const SpeechLmCfg& sl = audio_.config().speechlm;
    const int f = subsampling_, d = hidden_dim(), n_mels = (int)audio_.config().n_mels;
    const int n_frames = T_padded / f;
    hidden.assign((size_t)n_frames * d, 0.0f);
    std::vector<float> win, enc;
    for (int start = 0; start < n_frames; start += sl.chunk_len) {
        const int end = std::min(start + sl.chunk_len, n_frames);
        const int c0 = std::max(start - sl.chunk_left_context, 0);
        const int c1 = std::min(end + sl.chunk_right_context, n_frames);
        const int T = (c1 - c0) * f;
        win.resize((size_t)n_mels * T);
        for (int m = 0; m < n_mels; ++m)
            std::copy_n(mel.begin() + (size_t)m * T_padded + (size_t)c0 * f, T,
                        win.begin() + (size_t)m * T);
        int T_enc = 0;
        encoder_->forward(win, n_mels, T, enc, T_enc);
        if (T_enc != c1 - c0) throw std::runtime_error("parakeet: speech LLM encoder frame count mismatch");
        std::copy_n(enc.begin() + (size_t)(start - c0) * d, (size_t)(end - start) * d,
                    hidden.begin() + (size_t)start * d);
    }
}

// Speaker probabilities per encoder frame (mean over the 8 sub-frames), [n_frames, n_spk].
void SpeechLmAudio::speaker_activity(const std::vector<float>& mel, int T_padded, int n_frames,
                                     std::vector<float>& probs80) const {
    const ParakeetConfig& dc = diar_.config();
    const SpeechLmCfg& sl = audio_.config().speechlm;
    const int n_spk = (int)dc.diarization.n_speakers, f = subsampling_;
    const int n_mels = (int)audio_.config().n_mels;
    DiarStreamConfig scfg = DiarStreamConfig::from_model(dc);
    scfg.left_context = sl.chunk_left_context;
    scfg.right_context = sl.chunk_right_context;
    StreamingDiarization sd(diar_, scfg);
    sd.feed_mel(mel, n_mels, T_padded, /*is_last=*/true);
    const std::vector<float>& p = sd.last_probs();       // [n_spk, last_frames]
    const int T = sd.last_frames();
    if (T < n_frames * f) throw std::runtime_error("parakeet: speaker branch returned too few frames");
    probs80.assign((size_t)n_frames * n_spk, 0.0f);
    for (int s = 0; s < n_spk; ++s)
        for (int t = 0; t < n_frames; ++t) {
            float acc = 0.0f;
            for (int j = 0; j < f; ++j) acc += p[(size_t)s * T + (size_t)t * f + j];
            probs80[(size_t)t * n_spk + s] = acc / (float)f;
        }
}

void SpeechLmAudio::fuse_and_project(const std::vector<float>& hidden, const std::vector<float>& activity,
                                     int n_frames, std::vector<float>& fused,
                                     std::vector<float>& out) const {
    const int d = hidden_dim(), n_spk = num_speakers();
    const std::vector<float> table_t = speaker_table_t(n_spk, d);
    ensure_weights_realized(audio_);
    {   // fused = asr_norm(hidden) + diar_norm(activity) @ S
        GraphInputPool pool;
        const bool ok = run_graph(0, 0, [&](ggml_context* ctx) -> ggml_tensor* {
            ggml_tensor* h = input_f32(ctx, pool, hidden.data(), d, n_frames);
            ggml_tensor* a = input_f32(ctx, pool, activity.data(), n_spk, n_frames);
            ggml_tensor* tab = input_f32(ctx, pool, table_t.data(), n_spk, d);
            ggml_tensor* spk = ggml_mul_mat(ctx, tab, affine_norm(ctx, audio_, a, "speechlm.diar_norm"));
            return ggml_add(ctx, affine_norm(ctx, audio_, h, "speechlm.asr_norm"), spk);
        }, fused);
        if (!ok) throw std::runtime_error("parakeet: speech LLM fusion graph failed");
    }
    {   // out = proj(fused)
        GraphInputPool pool;
        const bool ok = run_graph(0, 0, [&](ggml_context* ctx) -> ggml_tensor* {
            ggml_tensor* x = input_f32(ctx, pool, fused.data(), d, n_frames);
            x = ggml_mul_mat(ctx, clone_weight(ctx, audio_, "speechlm.proj.weight"), x);
            return ggml_add(ctx, x, clone_weight(ctx, audio_, "speechlm.proj.bias"));
        }, out);
        if (!ok) throw std::runtime_error("parakeet: speech LLM adapter graph failed");
    }
}

void SpeechLmAudio::embed_pcm(const std::vector<float>& pcm, std::vector<float>& out, int& n_tokens,
                              SpeechLmAudioStages* stages) const {
    const ParakeetConfig& cfg = audio_.config();
    const int n_mels = (int)cfg.n_mels, f = subsampling_, hop = (int)cfg.hop_length;
    out.clear();
    n_tokens = 0;
    if (stages) *stages = SpeechLmAudioStages();
    if (pcm.empty() || hop <= 0) return;

    // 1. Log-mel (the audio GGUF has normalize NA, so this is the raw log-mel), trimmed
    //    to floor(samples / hop) frames.
    std::vector<float> mel;
    int nm = 0, T = 0;
    mel_->compute(pcm, mel, nm, T);
    trim_mel_frames(mel, nm, T, std::min(T, (int)(pcm.size() / hop)));
    if (T <= 0) return;

    // 2. Zero-pad to a multiple of the subsampling factor (what the reference
    //    processor does); every encoder frame then has at least one real mel frame.
    const int T_padded = (T + f - 1) / f * f;
    const int n_frames = T_padded / f;
    std::vector<float> raw((size_t)nm * T_padded, 0.0f);
    for (int m = 0; m < nm; ++m)
        std::copy_n(mel.begin() + (size_t)m * T, T, raw.begin() + (size_t)m * T_padded);
    std::vector<float> norm = raw;
    normalize_per_feature(norm, nm, T_padded, T);

    // 3. ASR branch and 4. speaker branch.
    std::vector<float> hidden, probs80;
    asr_chunks(norm, T_padded, hidden);
    speaker_activity(raw, T_padded, n_frames, probs80);
    const float thr = cfg.speechlm.activity_threshold;
    std::vector<float> activity(probs80.size());
    for (size_t i = 0; i < activity.size(); ++i) activity[i] = probs80[i] > thr ? 1.0f : 0.0f;

    // 5. Fusion and adapter.
    std::vector<float> fused;
    fuse_and_project(hidden, activity, n_frames, fused, out);
    n_tokens = n_frames;

    if (stages) {
        stages->n_frames = n_frames;
        stages->n_valid_mel = T;
        stages->mel.resize((size_t)T * nm);
        for (int m = 0; m < nm; ++m)
            for (int t = 0; t < T; ++t) stages->mel[(size_t)t * nm + m] = mel[(size_t)m * T + t];
        stages->asr_hidden = std::move(hidden);
        stages->speaker_probs = std::move(probs80);
        stages->fused = std::move(fused);
    }
}

}  // namespace pk
