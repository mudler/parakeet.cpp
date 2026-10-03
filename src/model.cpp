#include "model.hpp"

#include "audio_io.hpp"
#include "common.hpp"
#include "ternary.hpp"
#include "mel.hpp"
#include "mel_gpu.hpp"
#include "encoder.hpp"
#include "subsampling.hpp"
#include "vad_head.hpp"
#include "vad_segmenter.hpp"
#include "ctc_decoder.hpp"
#include "search.hpp"
#include "tokenizer.hpp"
#include "prediction.hpp"
#include "joint.hpp"
#include "prompt_kernel.hpp"
#include "tdt.hpp"
#include "rnnt.hpp"
#include "transducer_batch.hpp"
#include "transcription.hpp"
#include "decode_types.hpp"
#include "backend.hpp"
#include "backend_pool.hpp"
#include "ggml_graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <vector>

namespace pk {

namespace {
// Returns true when the arch string indicates a TDT/RNNT transducer head should
// be used by default (NeMo's cur_decoder='rnnt' for hybrid models).
bool arch_prefers_tdt(const std::string& arch) {
    return arch == "tdt"
        || arch == "hybrid_tdt_ctc"
        || arch == "rnnt"
        || arch == "hybrid_rnnt_ctc";
}
} // namespace

std::unique_ptr<Model> Model::load(const std::string& gguf_path, const std::string& component) {
    // unique_ptr<Model> via private ctor: construct then load. We avoid
    // std::make_unique (private ctor) and never throw out of here.
    std::unique_ptr<Model> m(new (std::nothrow) Model());
    if (!m) return nullptr;
    if (!(component.empty() ? m->loader_.load(gguf_path) : m->loader_.load_component(gguf_path, component))) {
        return nullptr;
    }
    // Model is the ASR entry point — reject diarization models so the C-API
    // can fall through to DiarizationModel::load.
    if (m->loader_.config().arch == "diarization") {
        return nullptr;
    }
    // A GGUF whose packed tensors and ternary flag disagree would skip the GPU
    // refusal and the validation below, so refuse it outright.
    {
        const std::string err = ternary_flag_consistency_error(m->loader_);
        if (!err.empty()) {
            PK_LOG("%s; refusing to load it", err.c_str());
            return nullptr;
        }
    }
    // Packed ternary weights run on the CPU kernel only. Fail at load with a
    // clear message instead of crashing inside a GPU graph.
    if (m->loader_.config().ternary.present &&
        std::string(pk::global_backend().device_name()) != "cpu") {
        PK_LOG("this GGUF holds packed ternary weights, which run on the CPU backend only; "
               "re-convert with --ternary dequant to use a GPU backend, or set PARAKEET_DEVICE=cpu");
        return nullptr;
    }
    // Validate and repack every packed linear now so graph building never throws.
    if (m->loader_.config().ternary.present) {
        try {
            ternary_prepare(m->loader_);
        } catch (const std::exception& e) {
            PK_LOG("invalid packed ternary GGUF: %s", e.what());
            return nullptr;
        }
    }
    // Give the weights a CPU backend buffer ONCE so graphs reference them
    // directly as leaves (zero per-call copy). Done at load (vs. lazily on first
    // clone_weight) so the cost is paid up front, not per utterance.
    ensure_weights_realized(m->loader_);
    return m;
}

std::shared_ptr<BackendPool> Model::pool_snapshot() const {
    std::lock_guard<std::mutex> lk(pool_mu_);
    return pool_;
}

int Model::set_concurrency(int backends, int threads_each) {
    if (backends < 1) backends = 1;
    // GPU keeps one backend and the global mutex: the pool is CPU only.
    if (backends > 1 && std::string(pk::global_backend().device_name()) != "cpu") {
        PK_LOG("set_concurrency(%d): the compute device is not the CPU; keeping 1 backend", backends);
        backends = 1;
    }
    std::shared_ptr<BackendPool> next;
    if (backends > 1) {
        if (threads_each < 1) {
            // Split the default thread budget across the backends.
            const int total = pk::effective_threads();
            threads_each = std::max(1, total / backends);
        }
        next = std::make_shared<BackendPool>(backends, threads_each);
    }
    std::shared_ptr<BackendPool> old;
    {
        std::lock_guard<std::mutex> lk(pool_mu_);
        old = std::move(pool_);
        pool_ = std::move(next);
    }
    // `old` is released here; requests still holding a lease keep it alive and
    // it is destroyed after the last of them finishes.
    return backends;
}

int Model::concurrency() const {
    std::shared_ptr<BackendPool> p = pool_snapshot();
    return p ? p->size() : 1;
}

int Model::threads_per_backend() const {
    std::shared_ptr<BackendPool> p = pool_snapshot();
    return p ? p->threads_each() : 0;
}

size_t Model::pool_working_set_bytes() const {
    std::shared_ptr<BackendPool> p = pool_snapshot();
    return p ? p->working_set_bytes() : 0;
}

const Model::DecoderObjects& Model::decoder_objects() const {
    std::call_once(decoder_once_, [this]() {
        decoder_ = std::make_unique<DecoderObjects>(loader_);
    });
    return *decoder_;
}

// Forward declarations: subsampling-tiling helpers are defined below (after the
// batched staging helpers) but used by the single-clip transcribe entry points.
static int safe_mel_window(const pk::ParakeetConfig& cfg);
static int subsampling_tile_for(const pk::ParakeetConfig& cfg,
                                const pk::ModelLoader& ml, int T_max);

int Model::resolve_prompt_index(const std::string& target_lang) const {
    const ParakeetConfig& cfg = loader_.config();
    if (!cfg.prompt.present) return -1;
    return cfg.prompt.resolve_index_or_throw(target_lang);
}

// Apply the prompt-conditioning projection in place on a channels-first encoder
// output [d_model, Tout], if the model is prompt-conditioned. No-op otherwise.
static void maybe_apply_prompt(const ModelLoader& loader, std::vector<float>& enc_out,
                               int d_model, int Tout, int prompt_index) {
    if (!loader.config().prompt.present) return;
    PromptKernel pk(loader);
    std::vector<float> projected;
    pk.apply(enc_out, d_model, Tout, prompt_index, projected);
    enc_out.swap(projected);
}

struct EncodedAudio {
    std::vector<float> channels_first;
    int d_model = 0;
    int frames = 0;
};

static void validate_nbest_request(const ParakeetConfig& cfg,
                                   int beam_size, int nbest) {
    if (cfg.tdt_durations.empty())
        throw std::runtime_error(
            "parakeet: N-best decoding requires a TDT duration table");
    if (beam_size < 1 || nbest < 1 || nbest > beam_size)
        throw std::invalid_argument(
            "parakeet: require beam_size >= nbest >= 1");
}

// Shared single-clip frontend + encoder path. Keeping this in one helper makes
// the opt-in N-best decoder consume exactly the same encoder output as greedy
// and timestamped transcription.
static EncodedAudio encode_16k(const ModelLoader& loader,
                               const std::vector<float>& pcm16k,
                               int prompt_index) {
    const ParakeetConfig& cfg = loader.config();

    std::vector<float> feats;
    int n_mels = 0, T = 0;
    if (std::string(pk::global_backend().device_name()) != "cpu") {
        GpuMel gmel(loader);
        gmel.compute(pcm16k, feats, n_mels, T);
    } else {
        MelFrontend mel(loader);
        mel.compute(pcm16k, feats, n_mels, T);
    }

    Encoder encoder(loader);
    EncodedAudio encoded;
    const int sub_tile = subsampling_tile_for(cfg, loader, T);
    if (sub_tile > 0) {
        MelBatch mb1;
        mb1.B = 1;
        mb1.n_mels = n_mels;
        mb1.T_max = T;
        mb1.valid_T = {T};
        mb1.data = std::move(feats);
        std::vector<std::vector<float>> outputs;
        std::vector<int> valid_frames;
        int padded_frames = 0;
        encoder.forward_batch_tiled(
            mb1, outputs, encoded.d_model, padded_frames, valid_frames, sub_tile);
        encoded.channels_first = std::move(outputs[0]);
        encoded.frames = valid_frames[0];
    } else {
        encoder.forward(feats, n_mels, T, encoded.channels_first,
                        encoded.d_model, encoded.frames);
    }

    maybe_apply_prompt(loader, encoded.channels_first,
                       encoded.d_model, encoded.frames, prompt_index);
    return encoded;
}

// Decode one item's encoder output (row-major [d_model, Tout], channels-first)
// into a transcript. Mirrors the tail of transcribe_16k exactly.
static std::string decode_enc_out(const ModelLoader& loader,
                                  const PredictionNet* dpred, const Joint* djoint,
                                  const std::vector<float>& enc_out,
                                  int d_model, int Tout, bool use_tdt) {
    const ParakeetConfig& cfg = loader.config();
    if (use_tdt) {
        std::vector<float> enc_row((size_t)Tout * d_model);
        for (int t = 0; t < Tout; ++t)
            for (int c = 0; c < d_model; ++c)
                enc_row[(size_t)t * d_model + c] = enc_out[(size_t)c * Tout + t];
        const PredictionNet& pred  = *dpred;
        const Joint&         joint = *djoint;
        const int max_symbols = static_cast<int>(cfg.max_symbols);
        std::vector<int32_t> ids;
        if (!cfg.tdt_durations.empty())
            ids = tdt_greedy(pred, joint, enc_row, Tout, d_model,
                             cfg.tdt_durations, (int)cfg.blank_id, max_symbols);
        else
            ids = rnnt_greedy(pred, joint, enc_row, Tout, d_model,
                              (int)cfg.blank_id, max_symbols);
        return detokenize(loader.tokenizer_pieces(), strip_special_tokens(loader.tokenizer_pieces(), ids));
    } else {
        CTCDecoder ctc(loader);
        std::vector<float> logits; int vocab_plus_1 = 0;
        ctc.forward(enc_out, d_model, Tout, logits, vocab_plus_1);
        std::vector<int32_t> ids = ctc_greedy(logits, Tout, vocab_plus_1,
                                              (int)cfg.blank_id);
        return detokenize(loader.tokenizer_pieces(), strip_special_tokens(loader.tokenizer_pieces(), ids));
    }
}

std::string Model::transcribe_16k(const std::vector<float>& pcm16k,
                                  Decoder decoder,
                                  const std::string& target_lang) const {
    PoolLease lease(pool_snapshot());
    const ParakeetConfig& cfg = loader_.config();
    const int prompt_index = resolve_prompt_index(target_lang);
    EncodedAudio encoded = encode_16k(loader_, pcm16k, prompt_index);

    // Decide which head to use.
    const bool use_tdt = (decoder == Decoder::kTDT)
        || (decoder == Decoder::kDefault && arch_prefers_tdt(cfg.arch));

    return decode_enc_out(loader_,
                          use_tdt ? &decoder_objects().pred : nullptr,
                          use_tdt ? &decoder_objects().joint : nullptr,
                          encoded.channels_first,
                          encoded.d_model, encoded.frames, use_tdt);
}

void Model::transcribe_16k_ctc_logits(const std::vector<float>& pcm16k,
                                      std::vector<float>& logits, int& T,
                                      int& vocab_plus_1,
                                      const std::string& target_lang) const {
    PoolLease lease(pool_snapshot());
    const ParakeetConfig& cfg = loader_.config();
    const int prompt_index = resolve_prompt_index(target_lang);

    // 1. Log-mel front end -> feats [n_mels, T]. Mirrors transcribe_16k exactly.
    std::vector<float> feats;
    int n_mels = 0, Tmel = 0;
    if (std::string(pk::global_backend().device_name()) != "cpu") {
        GpuMel gmel(loader_);
        gmel.compute(pcm16k, feats, n_mels, Tmel);
    } else {
        MelFrontend mel(loader_);
        mel.compute(pcm16k, feats, n_mels, Tmel);
    }

    // 2. FastConformer encoder -> enc_out [d_model, Tout] (channels-first).
    //    Long audio: tile the subsampling stage exactly as transcribe_16k does.
    Encoder encoder(loader_);
    std::vector<float> enc_out;
    int d_model = 0, Tout = 0;
    const int sub_tile = subsampling_tile_for(cfg, loader_, Tmel);
    if (sub_tile > 0) {
        MelBatch mb1;
        mb1.B = 1; mb1.n_mels = n_mels; mb1.T_max = Tmel; mb1.valid_T = { Tmel };
        mb1.data = std::move(feats);
        std::vector<std::vector<float>> eo; std::vector<int> vT;
        int dm1 = 0, To1 = 0;
        encoder.forward_batch_tiled(mb1, eo, dm1, To1, vT, sub_tile);
        enc_out = std::move(eo[0]);
        d_model = dm1;
        Tout = vT[0];
    } else {
        encoder.forward(feats, n_mels, Tmel, enc_out, d_model, Tout);
    }

    // 2b. Prompt conditioning (multilingual nemotron): project the encoder
    //     output with the selected language one-hot before decoding. No-op
    //     for other models (prompt.present == false).
    maybe_apply_prompt(loader_, enc_out, d_model, Tout, prompt_index);

    // 3. CTC head only — always, regardless of the model's preferred decoder.
    //    Throws std::runtime_error (from ctc_head_tensor, via CTCDecoder::forward)
    //    if the model has no CTC head, e.g. a TDT/RNNT-only streaming model.
    CTCDecoder ctc(loader_);
    ctc.forward(enc_out, d_model, Tout, logits, vocab_plus_1);
    T = Tout;
}

namespace {
// The VAD runs on blocks of this many seconds, each with its own mel
// normalisation, so the statistics follow the audio and memory stays bounded.
constexpr double kVadBlockSec = 120.0;
// A trailing block shorter than this is folded into the previous one.
constexpr double kVadMinTailSec = 5.0;
}  // namespace

std::vector<float> Model::vad_probabilities(const std::vector<float>& pcm16k,
                                            const VadVariant* v) const {
    PoolLease lease(pool_snapshot());
    const ParakeetConfig& cfg = loader_.config();
    if (!cfg.vad.present) throw std::runtime_error("model has no VAD head");
    const double fs = cfg.vad.frame_sec;
    if (!(fs > 0.0) || !std::isfinite(fs)) throw std::runtime_error("invalid VAD frame size");
    const bool gpu_mel = std::string(pk::global_backend().device_name()) != "cpu";
    const size_t block = (size_t)std::llround(kVadBlockSec * 16000.0);
    const size_t min_tail = (size_t)std::llround(kVadMinTailSec * 16000.0);
    const size_t block_frames = (size_t)std::llround(kVadBlockSec / fs);
    VadHead head(loader_);
    std::vector<float> all;
    size_t pos = 0;
    do {
        size_t len = std::min(block, pcm16k.size() - pos);
        if (pcm16k.size() - pos - len < min_tail) len = pcm16k.size() - pos;  // fold a short tail in
        const bool last = pos + len >= pcm16k.size();
        const std::vector<float> chunk(pcm16k.begin() + (std::ptrdiff_t)pos,
                                       pcm16k.begin() + (std::ptrdiff_t)(pos + len));
        std::vector<float> feats;
        int n_mels = 0, T = 0;
        if (gpu_mel) {
            GpuMel gmel(loader_);
            gmel.compute(chunk, feats, n_mels, T);
        } else {
            MelFrontend mel(loader_);
            mel.compute(chunk, feats, n_mels, T);
        }
        Subsampling sub(loader_);
        std::vector<float> out;
        int Tout = 0, d_model = 0, valid = 0;
        const int tile = subsampling_tile_for(cfg, loader_, T);
        if (tile > 0) sub.forward_tiled(feats, n_mels, T, tile, out, Tout, d_model, valid);
        else          sub.forward(feats, n_mels, T, out, Tout, d_model, valid);
        if ((uint32_t)d_model != cfg.vad.d_in)
            throw std::runtime_error("VAD head input width does not match the subsampler output");
        std::vector<float> p = head.probabilities(out.data(), valid, v);
        // Keep the frame grid of the whole clip: a full block contributes exactly
        // block_frames probabilities.
        if (!last && p.size() > block_frames) p.resize(block_frames);
        if (!last && p.size() < block_frames) p.resize(block_frames, p.empty() ? 0.0f : p.back());
        all.insert(all.end(), p.begin(), p.end());
        pos += len;
    } while (pos < pcm16k.size());
    return all;
}

namespace {

// Slices the caller decodes: [first_sample, last_sample) of the 16 kHz PCM.
struct Slice { std::vector<float> pcm; double start_sec; int start_frame; };

// `ext` replaces the model's own VAD head: it returns one probability per
// opts.frame_sec frame for the 16 kHz PCM. With `ext`, opts_in.frame_sec is the
// frame period of those probabilities.
std::vector<Slice> vad_slices(const Model& m, const std::vector<float>& pcm16k,
                              const SegmenterOpts& opts_in, const Model::VadProbabilityFn* ext) {
    SegmenterOpts opts = opts_in;
    if (!ext) opts.frame_sec = m.config().vad.frame_sec;
    // Token frame offsets are in ENCODER frames (same formula as the JSON writer).
    const ParakeetConfig& cfg = m.config();
    const double enc_frame_sec =
        (double)cfg.hop_length * (double)cfg.subsampling_factor / (double)cfg.sample_rate;
    if (!(enc_frame_sec > 0.0) || !std::isfinite(enc_frame_sec))
        throw std::runtime_error("invalid encoder frame size");
    const double total_sec = (double)pcm16k.size() / 16000.0;
    const std::vector<float> p = ext ? (*ext)(pcm16k) : m.vad_probabilities(pcm16k);
    const std::vector<VadSegment> segs = segment_by_vad(p, total_sec, opts);
    std::vector<Slice> out;
    const size_t n = pcm16k.size();
    auto at = [&](double sec) {
        const long long v = std::llround(sec * 16000.0);
        return (size_t)std::min<long long>(std::max<long long>(v, 0), (long long)n);
    };
    for (size_t i = 0; i < segs.size(); ++i) {
        // Kept segments are ordered and disjoint; segments without speech were dropped.
        const size_t a = at(segs[i].start);
        const size_t b = std::max(a, at(segs[i].end));
        Slice s;
        s.pcm.assign(pcm16k.begin() + (std::ptrdiff_t)a, pcm16k.begin() + (std::ptrdiff_t)b);
        if (s.pcm.size() < 3200) s.pcm.resize(3200, 0.0f);  // 0.2 s minimum
        s.start_sec = segs[i].start;
        s.start_frame = (int)std::llround(segs[i].start / enc_frame_sec);
        out.push_back(std::move(s));
    }
    return out;
}

}  // namespace

std::string Model::transcribe_pcm_vad(const std::vector<float>& pcm, int sample_rate,
                                      Decoder decoder, const std::string& target_lang,
                                      const SegmenterOpts& opts,
                                      const VadProbabilityFn* external_vad) const {
    PoolLease lease(pool_snapshot());
    if (!external_vad && !loader_.config().vad.present) throw std::runtime_error("model has no VAD head");
    const std::vector<float> pcm16k =
        sample_rate == 16000 ? pcm : resample_linear(pcm, sample_rate, 16000);
    if ((double)pcm16k.size() / 16000.0 <= opts.max_seg_sec)
        return transcribe_16k(pcm16k, decoder, target_lang);
    const std::vector<Slice> slices = vad_slices(*this, pcm16k, opts, external_vad);
    if (slices.empty()) return std::string();  // no speech found
    std::vector<const std::vector<float>*> pcms;
    for (const Slice& s : slices) pcms.push_back(&s.pcm);
    const std::vector<Transcription> parts = transcribe_16k_grouped(pcms, decoder, target_lang, false);
    std::string text;
    for (const Transcription& t : parts) {
        if (t.text.empty()) continue;
        if (!text.empty()) text += ' ';
        text += t.text;
    }
    return text;
}

Transcription Model::transcribe_pcm_vad_with_timestamps(const std::vector<float>& pcm, int sample_rate,
                                                        Decoder decoder, const std::string& target_lang,
                                                        const SegmenterOpts& opts,
                                      const VadProbabilityFn* external_vad) const {
    PoolLease lease(pool_snapshot());
    if (!external_vad && !loader_.config().vad.present) throw std::runtime_error("model has no VAD head");
    const std::vector<float> pcm16k =
        sample_rate == 16000 ? pcm : resample_linear(pcm, sample_rate, 16000);
    if ((double)pcm16k.size() / 16000.0 <= opts.max_seg_sec)
        return transcribe_with_timestamps(pcm16k, 16000, decoder, target_lang);
    const std::vector<Slice> slices = vad_slices(*this, pcm16k, opts, external_vad);
    if (slices.empty()) return Transcription();  // no speech found
    Transcription all;
    std::vector<const std::vector<float>*> pcms;
    for (const Slice& s : slices) pcms.push_back(&s.pcm);
    std::vector<Transcription> parts = transcribe_16k_grouped(pcms, decoder, target_lang, true);
    for (size_t i = 0; i < slices.size(); ++i) {
        const Slice& s = slices[i];
        Transcription& t = parts[i];
        for (Word& w : t.words) { w.start += (float)s.start_sec; w.end += (float)s.start_sec; }
        for (TokenInfo& k : t.tokens) k.frame += s.start_frame;
        if (!t.text.empty()) {
            if (!all.text.empty()) all.text += ' ';
            all.text += t.text;
        }
        all.words.insert(all.words.end(), t.words.begin(), t.words.end());
        all.tokens.insert(all.tokens.end(), t.tokens.begin(), t.tokens.end());
    }
    return all;
}

// Max mel frames per encoder pass before the first subsampling conv output
// (n_mels/2 * T/2 * conv_channels) approaches INT_MAX. ggml's CUDA unary (relu)
// kernel indexes elements with int32, so a tensor > 2^31 elements crashes
// ("invalid configuration argument"). Bound the per-pass first-conv tensor to a
// safe 1.5e9 elements: (n_mels/2)*(T/2)*C < 1.5e9  =>  T < 2*1.5e9 / ((n_mels/2)*C).
static int safe_mel_window(const pk::ParakeetConfig& cfg) {
    const long long per_t = (long long)((int)cfg.n_mels / 2) * (int)cfg.subsampling_conv_channels; // first-conv elems per output mel-row pair
    if (per_t <= 0) return 1 << 30;            // unknown config -> effectively no cap
    const long long bound = 1500000000LL;      // 1.5e9 elements, safe margin under 2^31
    long long win = (2 * bound) / per_t;        // T such that (n_mels/2)*(T/2)*C ~= bound
    if (win < 16384) win = 16384;               // floor for tiny/odd configs
    if (win > (1LL<<30)) win = (1LL<<30);
    return (int)win;
}

// Decide whether to tile the subsampling stage for a mel of T_max frames.
// Returns the tile size (output frames per tile, >0) to tile, or 0 to use the
// fused path. Tiling engages for long audio (first subsampling conv would exceed
// ggml's 2^31 element limit) or when PARAKEET_SUBSAMPLING_TILE forces it (testing).
static int subsampling_tile_for(const pk::ParakeetConfig& cfg,
                                const pk::ModelLoader& ml, int T_max) {
    if (const char* e = std::getenv("PARAKEET_SUBSAMPLING_TILE")) {
        const int t = std::atoi(e);
        if (t > 0) return t;
    }
    const int win = safe_mel_window(cfg);
    if (T_max > win) return pk::Subsampling(ml).subsample_len(win);
    return 0;
}

// Stage a batch of 16 kHz mono clips into a MelBatch: per-clip log-mel
// (GpuMel on a non-CPU backend, else the byte-identical FFT MelFrontend),
// zero-padded and stacked to the batch's longest clip (T_max). data layout is
// [B][n_mels][T_max], index (b*n_mels+m)*T_max+t; valid_T[b] is each clip's
// true frame count.
static MelBatch build_mel_batch(const ModelLoader& loader,
                                const std::vector<std::vector<float>>& pcms16k) {
    const bool gpu = std::string(pk::global_backend().device_name()) != "cpu";
    MelBatch mb;
    mb.B = (int)pcms16k.size();
    std::vector<std::vector<float>> feats(mb.B);
    std::vector<int> Ts(mb.B, 0);
    int n_mels = 0;
    for (int b = 0; b < mb.B; ++b) {
        int nm = 0, T = 0;
        if (gpu) { GpuMel g(loader); g.compute(pcms16k[b], feats[b], nm, T); }
        else     { MelFrontend m(loader); m.compute(pcms16k[b], feats[b], nm, T); }
        n_mels = nm; Ts[b] = T;
    }
    mb.n_mels = n_mels;
    mb.T_max = 0;
    for (int b = 0; b < mb.B; ++b) mb.T_max = std::max(mb.T_max, Ts[b]);
    mb.valid_T = Ts;
    mb.data.assign((size_t)mb.B * n_mels * mb.T_max, 0.0f);
    for (int b = 0; b < mb.B; ++b)
        for (int m = 0; m < n_mels; ++m)
            for (int t = 0; t < Ts[b]; ++t)
                mb.data[((size_t)b * n_mels + m) * mb.T_max + t] =
                    feats[b][(size_t)m * Ts[b] + t];
    return mb;
}

// Transpose the batched encoder outputs (channels-first [d_model, valid_Tout[b]])
// into per-item row-major [valid_Tout[b], d_model] for the transducer decoder.
static void batch_enc_to_row_major(const std::vector<std::vector<float>>& enc_outs,
                                   const std::vector<int>& valid_Tout, int d_model,
                                   std::vector<std::vector<float>>& encs,
                                   std::vector<int>& Ts) {
    const int B = (int)enc_outs.size();
    encs.assign(B, {}); Ts.assign(B, 0);
    for (int b = 0; b < B; ++b) {
        const int tb = valid_Tout[b];
        Ts[b] = tb;
        encs[b].resize((size_t)tb * d_model);
        for (int t = 0; t < tb; ++t)
            for (int c = 0; c < d_model; ++c)
                encs[b][(size_t)t * d_model + c] = enc_outs[b][(size_t)c * tb + t];
    }
}

std::vector<std::string> Model::transcribe_16k_batch(
    const std::vector<std::vector<float>>& pcms16k, Decoder decoder,
    const std::string& target_lang) const {
    PoolLease lease(pool_snapshot());
    const ParakeetConfig& cfg = loader_.config();
    const int prompt_index = resolve_prompt_index(target_lang);
    const bool use_tdt = (decoder == Decoder::kTDT)
        || (decoder == Decoder::kDefault && arch_prefers_tdt(cfg.arch));

    // 1. Per-clip mel, then stack to T_max.
    MelBatch mb = build_mel_batch(loader_, pcms16k);

    // 2. Batched encoder.
    Encoder encoder(loader_);
    std::vector<std::vector<float>> enc_outs; int d_model = 0, Tout = 0;
    std::vector<int> valid_Tout;
    // Long audio: the first subsampling conv would exceed ggml's 2^31 element limit.
    // Tile the subsampling stage (faithful; see Encoder::forward_batch_tiled). An env
    // override forces the tiled path for testing on short clips.
    const int sub_tile = subsampling_tile_for(cfg, loader_, mb.T_max);
    if (sub_tile > 0) {
        encoder.forward_batch_tiled(mb, enc_outs, d_model, Tout, valid_Tout, sub_tile);
    } else {
        encoder.forward_batch(mb, enc_outs, d_model, Tout, valid_Tout);
    }

    // 2b. Prompt conditioning per item (one language for the whole batch). No-op
    //     for non-prompt models.
    for (int b = 0; b < mb.B; ++b)
        maybe_apply_prompt(loader_, enc_outs[b], d_model, valid_Tout[b], prompt_index);

    // 3. Decode (each enc_out is [d_model, valid_Tout[b]]).
    std::vector<std::string> outs(mb.B);
    if (use_tdt) {
        // Batched transducer (TDT/RNNT) greedy decode: build per-item row-major
        // [T, d_model] from the channels-first [d_model, T] encoder outputs.
        std::vector<std::vector<float>> encs;
        std::vector<int> Ts;
        batch_enc_to_row_major(enc_outs, valid_Tout, d_model, encs, Ts);
        const PredictionNet& pred  = decoder_objects().pred;
        const Joint&         joint = decoder_objects().joint;
        std::vector<std::vector<int32_t>> ids;
        pk::transducer_greedy_batch(pred, joint, encs, Ts, d_model,
                                    cfg.tdt_durations, (int)cfg.blank_id,
                                    (int)cfg.max_symbols, ids, nullptr);
        for (int b = 0; b < mb.B; ++b)
            outs[b] = detokenize(loader_.tokenizer_pieces(),
                                 strip_special_tokens(loader_.tokenizer_pieces(), ids[b]));
    } else {
        // CTC stays per-item (no autoregressive decode to batch).
        for (int b = 0; b < mb.B; ++b)
            outs[b] = decode_enc_out(loader_,
                                   use_tdt ? &decoder_objects().pred : nullptr,
                                   use_tdt ? &decoder_objects().joint : nullptr,
                                   enc_outs[b], d_model, valid_Tout[b], use_tdt);
    }
    return outs;
}

std::vector<std::string> Model::transcribe_pcm_batch(
    const std::vector<std::vector<float>>& pcms, int sample_rate,
    Decoder decoder, const std::string& target_lang) const {
    if (sample_rate <= 0) {
        throw std::runtime_error("parakeet: invalid sample_rate");
    }
    std::vector<std::vector<float>> r(pcms.size());
    for (size_t i = 0; i < pcms.size(); ++i)
        r[i] = (sample_rate == 16000) ? pcms[i]
                                      : resample_linear(pcms[i], sample_rate, 16000);
    return transcribe_16k_batch(r, decoder, target_lang);
}

// Decode one item's encoder output (channels-first [d_model, Tout]) into a
// Transcription (text + per-word timestamps + tokens). Mirrors the decode tail
// of transcribe_16k_with_timestamps exactly.
static Transcription decode_enc_out_with_timestamps(
        const ModelLoader& loader, const PredictionNet* dpred, const Joint* djoint,
        const std::vector<float>& enc_out,
        int d_model, int Tout, bool use_tdt, float frame_sec) {
    const ParakeetConfig& cfg = loader.config();
    Transcription result;
    std::vector<TokenInfo> toks;
    if (use_tdt) {
        std::vector<float> enc_row((size_t)Tout * d_model);
        for (int t = 0; t < Tout; ++t)
            for (int c = 0; c < d_model; ++c)
                enc_row[(size_t)t * d_model + c] = enc_out[(size_t)c * Tout + t];
        const PredictionNet& pred  = *dpred;
        const Joint&         joint = *djoint;
        const int max_symbols = (int)cfg.max_symbols;
        if (!cfg.tdt_durations.empty())
            tdt_greedy(pred, joint, enc_row, Tout, d_model, cfg.tdt_durations,
                       (int)cfg.blank_id, max_symbols, &toks);
        else
            rnnt_greedy(pred, joint, enc_row, Tout, d_model,
                        (int)cfg.blank_id, max_symbols, &toks);
    } else {
        CTCDecoder ctc(loader);
        std::vector<float> logits; int vocab_plus_1 = 0;
        ctc.forward(enc_out, d_model, Tout, logits, vocab_plus_1);
        ctc_greedy(logits, Tout, vocab_plus_1, (int)cfg.blank_id, &toks);
        // NeMo CTC word end_offset is the NEXT collapsed token's start frame
        // (cumulative run lengths), not start+1. ctc_greedy emits span == 1;
        // rewrite each token's span to (next_frame - frame) so group_words'
        // frame+span rule reproduces NeMo's end_offset. The final token keeps
        // span == 1 (its true run length is unknown to the collapse, and within
        // the 1-frame word-end tolerance).
        for (size_t i = 0; i + 1 < toks.size(); ++i)
            toks[i].span = toks[i + 1].frame - toks[i].frame;
    }
    std::vector<int32_t> ids;
    ids.reserve(toks.size());
    for (const TokenInfo& ti : toks) ids.push_back(ti.id);
    result.text   = detokenize(loader.tokenizer_pieces(), strip_special_tokens(loader.tokenizer_pieces(), ids));
    result.words  = group_words(toks, loader.tokenizer_pieces(), frame_sec);
    result.tokens = std::move(toks);
    return result;
}

Transcription Model::transcribe_16k_with_timestamps(
    const std::vector<float>& pcm16k, Decoder decoder,
    const std::string& target_lang) const {
    PoolLease lease(pool_snapshot());
    const ParakeetConfig& cfg = loader_.config();
    const int prompt_index = resolve_prompt_index(target_lang);

    // frame_sec = hop_length * subsampling_factor / sample_rate (= 0.08 s here).
    // This is NeMo's window_stride * subsampling_factor (window_stride =
    // hop_length / sample_rate).
    const float frame_sec =
        (float)cfg.hop_length * (float)cfg.subsampling_factor / (float)cfg.sample_rate;
    EncodedAudio encoded = encode_16k(loader_, pcm16k, prompt_index);

    const bool use_tdt = (decoder == Decoder::kTDT)
        || (decoder == Decoder::kDefault && arch_prefers_tdt(cfg.arch));

    Transcription result = decode_enc_out_with_timestamps(
        loader_, use_tdt ? &decoder_objects().pred : nullptr,
        use_tdt ? &decoder_objects().joint : nullptr, encoded.channels_first, encoded.d_model, encoded.frames,
        use_tdt, frame_sec);
    return result;
}

namespace {
// Segments decoded together on the VAD path.
constexpr size_t kVadDecodeGroup = 16;
}  // namespace

std::vector<Transcription> Model::transcribe_16k_grouped(
        const std::vector<const std::vector<float>*>& pcms16k, Decoder decoder,
        const std::string& target_lang, bool with_timestamps) const {
    PoolLease lease(pool_snapshot());
    const ParakeetConfig& cfg = loader_.config();
    const int prompt_index = resolve_prompt_index(target_lang);
    const float frame_sec =
        (float)cfg.hop_length * (float)cfg.subsampling_factor / (float)cfg.sample_rate;
    const bool use_tdt = (decoder == Decoder::kTDT)
        || (decoder == Decoder::kDefault && arch_prefers_tdt(cfg.arch));

    std::vector<Transcription> outs(pcms16k.size());
    for (size_t g0 = 0; g0 < pcms16k.size(); g0 += kVadDecodeGroup) {
        const size_t g1 = std::min(pcms16k.size(), g0 + kVadDecodeGroup);
        // Encode one by one, exactly as the single clip path does.
        std::vector<EncodedAudio> enc;
        enc.reserve(g1 - g0);
        for (size_t i = g0; i < g1; ++i) enc.push_back(encode_16k(loader_, *pcms16k[i], prompt_index));
        if (!use_tdt) {  // CTC: no autoregressive decode to batch
            for (size_t i = g0; i < g1; ++i) {
                const EncodedAudio& e = enc[i - g0];
                if (with_timestamps)
                    outs[i] = decode_enc_out_with_timestamps(loader_, nullptr, nullptr, e.channels_first,
                                                             e.d_model, e.frames, false, frame_sec);
                else
                    outs[i].text = decode_enc_out(loader_, nullptr, nullptr, e.channels_first,
                                                  e.d_model, e.frames, false);
            }
            continue;
        }
        std::vector<std::vector<float>> chans;
        std::vector<int> frames;
        chans.reserve(enc.size());
        for (EncodedAudio& e : enc) { chans.push_back(std::move(e.channels_first)); frames.push_back(e.frames); }
        const int d_model = enc.front().d_model;
        std::vector<std::vector<float>> encs;
        std::vector<int> Ts;
        batch_enc_to_row_major(chans, frames, d_model, encs, Ts);
        std::vector<std::vector<int32_t>> ids;
        std::vector<std::vector<TokenInfo>> toks;
        pk::transducer_greedy_batch(decoder_objects().pred, decoder_objects().joint, encs, Ts, d_model,
                                    cfg.tdt_durations, (int)cfg.blank_id, (int)cfg.max_symbols, ids,
                                    with_timestamps ? &toks : nullptr);
        for (size_t i = g0; i < g1; ++i) {
            const size_t k = i - g0;
            Transcription& r = outs[i];
            r.text = detokenize(loader_.tokenizer_pieces(),
                                strip_special_tokens(loader_.tokenizer_pieces(), ids[k]));
            if (with_timestamps) {
                r.words = group_words(toks[k], loader_.tokenizer_pieces(), frame_sec);
                r.tokens = std::move(toks[k]);
            }
        }
    }
    return outs;
}

std::vector<Transcription> Model::transcribe_16k_batch_with_timestamps(
        const std::vector<std::vector<float>>& pcms16k, Decoder decoder,
        const std::string& target_lang) const {
    PoolLease lease(pool_snapshot());
    const ParakeetConfig& cfg = loader_.config();
    const int prompt_index = resolve_prompt_index(target_lang);
    const float frame_sec =
        (float)cfg.hop_length * (float)cfg.subsampling_factor / (float)cfg.sample_rate;
    const bool use_tdt = (decoder == Decoder::kTDT)
        || (decoder == Decoder::kDefault && arch_prefers_tdt(cfg.arch));

    MelBatch mb = build_mel_batch(loader_, pcms16k);

    Encoder encoder(loader_);
    std::vector<std::vector<float>> enc_outs; int d_model = 0, Tout = 0;
    std::vector<int> valid_Tout;
    // Long audio: the first subsampling conv would exceed ggml's 2^31 element limit.
    // Tile the subsampling stage (faithful; see Encoder::forward_batch_tiled). An env
    // override forces the tiled path for testing on short clips.
    const int sub_tile = subsampling_tile_for(cfg, loader_, mb.T_max);
    if (sub_tile > 0) {
        encoder.forward_batch_tiled(mb, enc_outs, d_model, Tout, valid_Tout, sub_tile);
    } else {
        encoder.forward_batch(mb, enc_outs, d_model, Tout, valid_Tout);
    }

    // Prompt conditioning per item (one language for the whole batch). No-op
    // for non-prompt models.
    for (int b = 0; b < mb.B; ++b)
        maybe_apply_prompt(loader_, enc_outs[b], d_model, valid_Tout[b], prompt_index);

    std::vector<Transcription> outs(mb.B);
    if (use_tdt) {
        // Batched transducer (TDT/RNNT) greedy decode with timestamps. Build
        // per-item row-major [T, d_model] from channels-first [d_model, T].
        std::vector<std::vector<float>> encs;
        std::vector<int> Ts;
        batch_enc_to_row_major(enc_outs, valid_Tout, d_model, encs, Ts);
        const PredictionNet& pred  = decoder_objects().pred;
        const Joint&         joint = decoder_objects().joint;
        std::vector<std::vector<int32_t>> ids;
        std::vector<std::vector<TokenInfo>> toks;
        pk::transducer_greedy_batch(pred, joint, encs, Ts, d_model,
                                    cfg.tdt_durations, (int)cfg.blank_id,
                                    (int)cfg.max_symbols, ids, &toks);
        // Assemble each Transcription exactly as decode_enc_out_with_timestamps'
        // transducer tail does.
        for (int b = 0; b < mb.B; ++b) {
            Transcription& result = outs[b];
            result.text   = detokenize(loader_.tokenizer_pieces(),
                                       strip_special_tokens(loader_.tokenizer_pieces(), ids[b]));
            result.words  = group_words(toks[b], loader_.tokenizer_pieces(), frame_sec);
            result.tokens = std::move(toks[b]);
        }
    } else {
        // CTC stays per-item (not a transducer; no autoregressive decode).
        for (int b = 0; b < mb.B; ++b)
            outs[b] = decode_enc_out_with_timestamps(
                loader_, use_tdt ? &decoder_objects().pred : nullptr,
                use_tdt ? &decoder_objects().joint : nullptr, enc_outs[b], d_model, valid_Tout[b], use_tdt, frame_sec);
    }
    return outs;
}

std::vector<Transcription> Model::transcribe_pcm_batch_with_timestamps(
        const std::vector<std::vector<float>>& pcms, int sample_rate,
        Decoder decoder, const std::string& target_lang) const {
    if (sample_rate <= 0) {
        throw std::runtime_error("parakeet: invalid sample_rate");
    }
    std::vector<std::vector<float>> r(pcms.size());
    for (size_t i = 0; i < pcms.size(); ++i)
        r[i] = (sample_rate == 16000) ? pcms[i]
                                      : resample_linear(pcms[i], sample_rate, 16000);
    return transcribe_16k_batch_with_timestamps(r, decoder, target_lang);
}

std::vector<NBestTranscription> Model::transcribe_16k_nbest(
        const std::vector<float>& pcm16k, int beam_size, int nbest,
        bool score_norm, const std::string& target_lang) const {
    PoolLease lease(pool_snapshot());
    const ParakeetConfig& cfg = loader_.config();

    const int prompt_index = resolve_prompt_index(target_lang);
    EncodedAudio encoded = encode_16k(loader_, pcm16k, prompt_index);

    std::vector<float> enc_row(
        (size_t)encoded.frames * encoded.d_model);
    for (int t = 0; t < encoded.frames; ++t)
        for (int c = 0; c < encoded.d_model; ++c)
            enc_row[(size_t)t * encoded.d_model + c] =
                encoded.channels_first[(size_t)c * encoded.frames + t];

    const PredictionNet& pred  = decoder_objects().pred;
    const Joint&         joint = decoder_objects().joint;
    std::vector<TdtBeamHypothesis> beam = tdt_beam_search(
        pred, joint, enc_row, encoded.frames, encoded.d_model,
        cfg.tdt_durations, (int)cfg.blank_id,
        beam_size, nbest, score_norm);

    std::vector<NBestTranscription> result;
    result.reserve(beam.size());
    for (TdtBeamHypothesis& hyp : beam) {
        std::vector<int32_t> ids;
        ids.reserve(hyp.tokens.size());
        for (const TdtBeamToken& token : hyp.tokens)
            ids.push_back(token.id);

        NBestTranscription item;
        item.text = detokenize(
            loader_.tokenizer_pieces(),
            strip_special_tokens(loader_.tokenizer_pieces(), ids));
        item.tokens = std::move(hyp.tokens);
        item.score = hyp.score;
        item.normalized_score = hyp.normalized_score;
        result.push_back(std::move(item));
    }
    return result;
}

std::vector<NBestTranscription> Model::transcribe_pcm_nbest(
        const std::vector<float>& pcm, int sample_rate,
        int beam_size, int nbest, bool score_norm,
        const std::string& target_lang) const {
    validate_nbest_request(loader_.config(), beam_size, nbest);
    if (sample_rate <= 0)
        throw std::runtime_error("parakeet: invalid sample_rate");
    if (sample_rate == 16000)
        return transcribe_16k_nbest(
            pcm, beam_size, nbest, score_norm, target_lang);
    return transcribe_16k_nbest(
        resample_linear(pcm, sample_rate, 16000),
        beam_size, nbest, score_norm, target_lang);
}

std::vector<NBestTranscription> Model::transcribe_path_nbest(
        const std::string& wav_path, int beam_size, int nbest,
        bool score_norm, const std::string& target_lang) const {
    validate_nbest_request(loader_.config(), beam_size, nbest);
    Audio audio;
    if (!load_audio_16k_mono(wav_path, audio))
        throw std::runtime_error(
            "parakeet: failed to load audio: " + wav_path);
    return transcribe_16k_nbest(
        audio.samples, beam_size, nbest, score_norm, target_lang);
}

std::string Model::transcribe_pcm(const std::vector<float>& pcm, int sample_rate,
                                  Decoder decoder, const std::string& target_lang) const {
    if (sample_rate <= 0) {
        throw std::runtime_error("parakeet: invalid sample_rate");
    }
    if (sample_rate == 16000) {
        return transcribe_16k(pcm, decoder, target_lang);
    }
    std::vector<float> pcm16k = resample_linear(pcm, sample_rate, 16000);
    return transcribe_16k(pcm16k, decoder, target_lang);
}

void Model::transcribe_pcm_ctc_logits(const std::vector<float>& pcm, int sample_rate,
                                      std::vector<float>& logits, int& T,
                                      int& vocab_plus_1,
                                      const std::string& target_lang) const {
    if (sample_rate <= 0) {
        throw std::runtime_error("parakeet: invalid sample_rate");
    }
    if (sample_rate == 16000) {
        transcribe_16k_ctc_logits(pcm, logits, T, vocab_plus_1, target_lang);
        return;
    }
    std::vector<float> pcm16k = resample_linear(pcm, sample_rate, 16000);
    transcribe_16k_ctc_logits(pcm16k, logits, T, vocab_plus_1, target_lang);
}

std::string Model::transcribe_path(const std::string& wav_path,
                                   Decoder decoder, const std::string& target_lang) const {
    Audio audio;
    if (!load_audio_16k_mono(wav_path, audio)) {
        throw std::runtime_error("parakeet: failed to load audio: " + wav_path);
    }
    // load_audio_16k_mono already resamples to 16 kHz mono.
    return transcribe_16k(audio.samples, decoder, target_lang);
}

Transcription Model::transcribe_with_timestamps(
    const std::vector<float>& pcm, int sample_rate, Decoder decoder,
    const std::string& target_lang) const {
    if (sample_rate <= 0) {
        throw std::runtime_error("parakeet: invalid sample_rate");
    }
    if (sample_rate == 16000) {
        return transcribe_16k_with_timestamps(pcm, decoder, target_lang);
    }
    std::vector<float> pcm16k = resample_linear(pcm, sample_rate, 16000);
    return transcribe_16k_with_timestamps(pcm16k, decoder, target_lang);
}

Transcription Model::transcribe_path_with_timestamps(
    const std::string& wav_path, Decoder decoder,
    const std::string& target_lang) const {
    Audio audio;
    if (!load_audio_16k_mono(wav_path, audio)) {
        throw std::runtime_error("parakeet: failed to load audio: " + wav_path);
    }
    return transcribe_16k_with_timestamps(audio.samples, decoder, target_lang);
}

} // namespace pk
