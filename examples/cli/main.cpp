#include "parakeet.h"
#include "parakeet_capi.h"
#include "model.hpp"
#include <cmath>
#include "model_loader.hpp"
#include "audio_io.hpp"
#include "streaming.hpp"
#include "transcription.hpp"
#include "ggml_graph.hpp"   // pk::set_num_threads, pk::global_backend
#include "backend.hpp"      // pk::ensure_weights_realized
#include "encoder.hpp"
#include "prediction.hpp"
#include "joint.hpp"
#include "tdt.hpp"
#include "rnnt.hpp"
#include "transducer_batch.hpp"
#include "mel.hpp"
#include "mel_gpu.hpp"
#include "ggml.h"
#include "gguf.h"
#include "transcription_json.hpp"
#include "common.hpp"   // pk::write_file_atomic
#include "diarization.hpp"
#include "ced_tagger.hpp"
#include "scene_stream.hpp"
#include "scene_render.hpp"
#include "vad_head.hpp"
#include "vad_json.hpp"
#include "silero_vad.hpp"
#include "bundle.hpp"
#include "speaker_encoder.hpp"
#include "speaker_registry.hpp"
#include <atomic>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

// The ASR component that --model/--component name, "" for a plain single-model
// file. Set once by the commands that take --component; load_asr() uses it.
static std::string g_asr_component;

// Resolves `component` (may be empty) against `model`. On a bundle with no
// component named, picks the default ASR component. Returns false with a message
// when the file is a plain GGUF but a component was named, or when there is no
// unique default.
static bool resolve_asr_component(const std::string& model, const std::string& component, std::string& out) {
    out.clear();
    if (!pk::gguf_is_bundle(model)) {
        if (!component.empty()) {
            std::fprintf(stderr, "parakeet-cli: %s is not a bundle GGUF, so --component does not apply\n", model.c_str());
            return false;
        }
        return true;
    }
    pk::BundleInfo info;
    std::string err;
    if (!pk::read_bundle_info(model, info, &err)) { std::fprintf(stderr, "parakeet-cli: %s\n", err.c_str()); return false; }
    if (!component.empty()) {
        const pk::BundleComponent* c = info.find(component);
        if (!c) {
            std::fprintf(stderr, "parakeet-cli: bundle has no component '%s'; components: %s\n", component.c_str(),
                         pk::bundle_component_names(info).c_str());
            return false;
        }
        if (c->kind != pk::kBundleKindAsr) {
            std::fprintf(stderr, "parakeet-cli: component '%s' has kind %s, not an ASR model\n", component.c_str(), c->kind.c_str());
            return false;
        }
        out = component;
        return true;
    }
    if (!pk::select_default_component(info, out, &err)) { std::fprintf(stderr, "parakeet-cli: %s\n", err.c_str()); return false; }
    return true;
}

static std::unique_ptr<pk::Model> load_asr(const std::string& model) {
    return pk::Model::load(model, g_asr_component);
}

static parakeet_ctx* load_asr_ctx(const std::string& model) {
    parakeet_ctx* ctx = parakeet_capi_load_component(model.c_str(), g_asr_component.c_str());
    if (!ctx) std::fprintf(stderr, "parakeet-cli: %s\n", parakeet_capi_load_error());
    return ctx;
}

static int cmd_info_bundle(const char* path, const pk::BundleInfo& info) {
    std::printf("parakeet.cpp %s\n", parakeet_version());
    std::printf("bundle: %s\n", path);
    std::printf("  name            : %s\n", info.name.c_str());
    std::printf("  format version  : %u\n", info.version);
    for (const pk::BundleComponent& c : info.components) {
        std::printf("  component %s\n", c.name.c_str());
        std::printf("    kind          : %s\n", c.kind.c_str());
        std::printf("    licence       : %s (%s)\n", c.license.c_str(), c.license_url.c_str());
        std::printf("    source        : %s\n", c.source.c_str());
        std::printf("    attribution   : %s\n", c.attribution.c_str());
        std::printf("    changes       : %s\n", c.changes.c_str());
        std::printf("    tensors       : %llu (%.1f MB)\n", (unsigned long long)c.n_tensors, (double)c.n_bytes / 1e6);
        if (!c.content_sha256.empty()) std::printf("    content sha256: %s\n", c.content_sha256.c_str());
    }
    std::string def, err;
    if (pk::select_default_component(info, def, &err)) std::printf("  default for load: %s\n", def.c_str());
    else std::printf("  default for load: none (%s)\n", err.c_str());
    std::printf("  (use --component NAME for the details of one component)\n");
    return 0;
}

static int cmd_info(int argc, char** argv) {
    const char* path = argv[0];
    std::string component;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--component") == 0 && i + 1 < argc) component = argv[++i];
        else { std::fprintf(stderr, "usage: parakeet-cli info <model.gguf> [--component NAME]\n"); return 2; }
    }
    if (pk::gguf_is_bundle(path)) {
        pk::BundleInfo info;
        std::string err;
        if (!pk::read_bundle_info(path, info, &err)) { std::fprintf(stderr, "parakeet-cli: %s\n", err.c_str()); return 1; }
        if (component.empty()) return cmd_info_bundle(path, info);
        const pk::BundleComponent* bc = info.find(component);
        if (!bc) {
            std::fprintf(stderr, "parakeet-cli: bundle has no component '%s'; components: %s\n", component.c_str(),
                         pk::bundle_component_names(info).c_str());
            return 1;
        }
        std::printf("component: %s (kind %s, licence %s)\n", bc->name.c_str(), bc->kind.c_str(), bc->license.c_str());
        if (bc->kind == pk::kBundleKindVad) {
            std::string err2;
            std::unique_ptr<pk::SileroVad> sv = pk::SileroVad::load(path, &err2, component);
            if (!sv) { std::fprintf(stderr, "parakeet-cli: %s\n", err2.c_str()); return 1; }
            std::printf("  silero VAD, sample rates:");
            for (int r : sv->sample_rates()) std::printf(" %d", r);
            std::printf("\n");
            return 0;
        }
    } else if (!component.empty()) {
        std::fprintf(stderr, "parakeet-cli: %s is not a bundle GGUF, so --component does not apply\n", path);
        return 1;
    }
    pk::ModelLoader ml;
    if (!(component.empty() ? ml.load(path) : ml.load_component(path, component))) {
        std::fprintf(stderr, "failed to load %s\n", path);
        return 1;
    }
    const pk::ParakeetConfig& c = ml.config();
    std::printf("parakeet.cpp %s\n", parakeet_version());
    std::printf("model: %s\n", path);
    std::printf("  arch            : %s\n", c.arch.c_str());
    if (c.vad.present) std::printf("  vad head        : yes (%.3f s frames)\n", (double)c.vad.frame_sec);
    std::printf("  d_model/layers/heads: %u / %u / %u\n", c.d_model, c.n_layers, c.n_heads);
    std::printf("  conv_kernel/norm: %u / %s\n", c.conv_kernel, c.conv_norm_type.c_str());
    std::printf("  xscaling        : %s\n", c.xscaling ? "true" : "false");
    std::printf("  subsampling     : x%u (ch=%u)\n", c.subsampling_factor, c.subsampling_conv_channels);
    std::printf("  mel/n_fft/win/hop: %u / %u / %u / %u\n", c.n_mels, c.n_fft, c.win_length, c.hop_length);
    std::printf("  vocab/blank     : %u / %u\n", c.vocab_size, c.blank_id);
    if (!c.tdt_durations.empty()) {
        std::printf("  tdt durations   : [");
        for (size_t i=0;i<c.tdt_durations.size();++i) std::printf("%s%d", i?",":"", c.tdt_durations[i]);
        std::printf("]\n");
    }
    // Cache-aware streaming / causal config (Phase 5). Only meaningful for
    // streaming models; offline models report style "regular" with full context.
    std::printf("  att_context     : [%d,%d] %s\n",
                c.att_context_left, c.att_context_right, c.att_context_style.c_str());
    std::printf("  causal ds/conv  : %s / %s\n",
                c.causal_downsampling ? "true" : "false",
                c.conv_causal ? "true" : "false");
    if (c.streaming.present) {
        const pk::StreamingCfg& s = c.streaming;
        auto print_ivec = [](const char* label, const std::vector<int32_t>& v) {
            std::printf("  %-15s : [", label);
            for (size_t i = 0; i < v.size(); ++i) std::printf("%s%d", i ? "," : "", v[i]);
            std::printf("]\n");
        };
        std::printf("  streaming       : enabled\n");
        print_ivec("  chunk_size", s.chunk_size);
        print_ivec("  shift_size", s.shift_size);
        print_ivec("  pre_enc_cache", s.pre_encode_cache_size);
        std::printf("    cache_drop     : %d\n", s.cache_drop_size);
        std::printf("    last_ch_cache  : %d\n", s.last_channel_cache_size);
        std::printf("    valid_out_len  : %d\n", s.valid_out_len);
        std::printf("    drop_extra_pre : %d\n", s.drop_extra_pre_encoded);
    }
    return 0;
}

static bool is_stdin_input(const std::string& input) {
    return input == "-";
}

static std::string input_display_name(const std::string& input) {
    return is_stdin_input(input) ? "stdin" : input;
}

static bool read_stdin_bytes(std::vector<unsigned char>& bytes) {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
#endif
    std::istreambuf_iterator<char> begin(std::cin);
    std::istreambuf_iterator<char> end;
    bytes.assign(begin, end);
    return !bytes.empty() && !std::cin.bad();
}

static bool load_audio_arg_16k_mono(const std::string& input, pk::Audio& out) {
    if (!is_stdin_input(input)) return pk::load_audio_16k_mono(input, out);

    std::vector<unsigned char> bytes;
    if (!read_stdin_bytes(bytes)) {
        std::fprintf(stderr, "parakeet-cli: failed to read WAV bytes from stdin\n");
        return false;
    }
    return pk::load_audio_16k_mono_from_memory(bytes.data(), bytes.size(), out);
}

static float model_frame_sec(const pk::Model& model) {
    const pk::ParakeetConfig& cfg = model.config();
    return (float)cfg.hop_length * (float)cfg.subsampling_factor / (float)cfg.sample_rate;
}

// Cache-aware streaming transcription for the EOU streaming model. Feeds the WAV
// to a pk::StreamingSession in the model's exact chunk schedule, printing partial
// text incrementally and `[EOU @ <t>s]` / `[EOB @ <t>s]` markers when events
// fire, then the finalize() tail. Returns 0/1.
//
// When `timestamps` is set, also prints one line per finalized word
// (`<start>-<end>  <word>  (<conf>)`) after the running text/EOU line.
static int cmd_transcribe_stream(const std::string& model, const std::string& input,
                                 bool timestamps, const std::string& lang) {
    pk::ModelLoader ml;
    if (!ml.load(model)) {
        std::fprintf(stderr, "parakeet-cli: failed to load model %s\n", model.c_str());
        return 1;
    }
    if (!ml.config().streaming.present) {
        std::fprintf(stderr,
            "parakeet-cli: --stream requires a cache-aware streaming model "
            "(e.g. parakeet_realtime_eou_120m-v1); %s is not one\n", model.c_str());
        return 1;
    }
    pk::Audio audio;
    if (!load_audio_arg_16k_mono(input, audio)) {
        std::string display = input_display_name(input);
        std::fprintf(stderr, "parakeet-cli: failed to load audio %s\n", display.c_str());
        return 1;
    }

    try {
        // `lang` selects the language prompt for multilingual (nemotron) prompt
        // models; empty -> the model default, and non-prompt models ignore it.
        // This is exactly what parakeet_capi_stream_begin_lang forwards to the
        // StreamingSession ctor — done directly here so the CLI keeps its rich
        // per-word / EOU-timestamp output the flat stream C-API does not expose.
        pk::StreamingSession sess(ml, lang);
        std::vector<pk::Word> all_words;  // collected for the --timestamps recap
        std::printf("[stream] ");
        std::fflush(stdout);
        pk::run_stream_over_pcm(
            sess, ml, audio.samples,
            [&](const std::string& new_text, const std::vector<pk::EouEvent>& evs,
                const std::vector<pk::Word>& words) {
                if (!new_text.empty()) {
                    std::printf("%s", new_text.c_str());
                    std::fflush(stdout);
                }
                for (const pk::EouEvent& e : evs) {
                    std::printf(" [%s @ %.2fs]", e.is_eob ? "EOB" : "EOU", e.time_sec);
                    std::fflush(stdout);
                }
                if (timestamps)
                    all_words.insert(all_words.end(), words.begin(), words.end());
            });
        // Flush the end-of-stream tail (no extra <EOU> is fabricated if NeMo's
        // streaming would not emit one for this clip).
        std::string tail = sess.finalize();
        if (!tail.empty()) std::printf("%s", tail.c_str());
        if (timestamps) {
            // The trailing open word finalizes only at finalize(); drain it now.
            std::vector<pk::Word> last = sess.drain_words();
            all_words.insert(all_words.end(), last.begin(), last.end());
        }
        std::printf("\n");
        // Also print the full transcript on its own line for easy capture.
        std::printf("[stream:final] %s\n", sess.text().c_str());
        if (timestamps) {
            for (const pk::Word& w : all_words)
                std::printf("%.2f-%.2f  %s  (%.2f)\n", w.start, w.end,
                            w.text.c_str(), w.conf);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "parakeet-cli: streaming failed: %s\n", e.what());
        return 1;
    }
    return 0;
}

// Segmenter options the user set on the command line. A value left unset keeps
// the default of the VAD in use (Ultra/Redux head or Silero).
struct VadOverrides {
    std::optional<double> threshold, min_pause, min_speech, max_seg, pad;
    void apply(pk::SegmenterOpts& o) const {
        if (threshold) o.threshold = (float)*threshold;
        if (min_pause) o.min_pause_sec = *min_pause;
        if (min_speech) o.min_speech_sec = *min_speech;
        if (max_seg) o.max_seg_sec = *max_seg;
        if (pad) o.pad_sec = *pad;
    }
};

static int cmd_transcribe_vad(const std::string& model, const std::string& input, pk::Decoder dec,
                              const std::string& lang, bool timestamps, bool json,
                              const VadOverrides& ov, const std::string& vad_model,
                              const std::string& vad_component) {
    pk::Audio audio;
    if (!load_audio_arg_16k_mono(input, audio)) {
        std::fprintf(stderr, "parakeet-cli: failed to load audio %s\n", input.c_str());
        return 1;
    }
    try {
        std::unique_ptr<pk::Model> m = load_asr(model);
        if (!m) { std::fprintf(stderr, "parakeet-cli: failed to load model %s\n", model.c_str()); return 1; }
        // VAD source: --vad-model (a Silero GGUF, or a bundle with --vad-component),
        // else the Silero component of the --model bundle, else the ASR head.
        std::string silero_path = vad_model, silero_comp;
        if (vad_model.empty() && pk::gguf_is_bundle(model)) {
            pk::BundleInfo info;
            std::string berr;
            if (!pk::read_bundle_info(model, info, &berr)) { std::fprintf(stderr, "parakeet-cli: %s\n", berr.c_str()); return 1; }
            if (!vad_component.empty()) {
                const pk::BundleComponent* c = info.find(vad_component);
                if (!c) { std::fprintf(stderr, "parakeet-cli: bundle has no component '%s'; components: %s\n", vad_component.c_str(), pk::bundle_component_names(info).c_str()); return 1; }
                if (c->kind == pk::kBundleKindVad) { silero_path = model; silero_comp = vad_component; }
                else if (vad_component != g_asr_component) { std::fprintf(stderr, "parakeet-cli: --vad-component %s is not a VAD component\n", vad_component.c_str()); return 1; }
            } else {
                for (const pk::BundleComponent& c : info.components)
                    if (c.kind == pk::kBundleKindVad) { silero_path = model; silero_comp = c.name; break; }
            }
        } else if (!vad_model.empty() && !vad_component.empty()) {
            silero_comp = vad_component;
        }
        pk::SegmenterOpts opts = pk::default_segmenter_opts(silero_path.empty() ? pk::VadKind::kHead : pk::VadKind::kSilero);
        ov.apply(opts);
        std::unique_ptr<pk::SileroVad> silero;
        pk::Model::VadProbabilityFn fn;
        if (!silero_path.empty()) {
            std::string err;
            silero = pk::SileroVad::load(silero_path, &err, silero_comp);
            if (!silero) { std::fprintf(stderr, "parakeet-cli: failed to load VAD model %s: %s\n", silero_path.c_str(), err.c_str()); return 1; }
            const pk::SileroVad* sv = silero.get();
            fn = [sv](const std::vector<float>& pcm16k) {
                std::vector<float> p = sv->probabilities(pcm16k.data(), pcm16k.size(), 16000);
                if (p.empty() && !pcm16k.empty()) throw std::runtime_error("Silero VAD failed");
                return p;
            };
        }
        const pk::Model::VadProbabilityFn* ext = silero ? &fn : nullptr;
        if (json || timestamps) {
            pk::Transcription tr =
                m->transcribe_pcm_vad_with_timestamps(audio.samples, audio.sample_rate, dec, lang, opts, ext);
            if (json) {
                std::printf("%s\n", pk::transcription_to_json(tr, model_frame_sec(*m)).c_str());
            } else {
                for (const pk::Word& w : tr.words)
                    std::printf("%.2f-%.2f  %s  (%.2f)\n", w.start, w.end, w.text.c_str(), w.conf);
            }
        } else {
            std::printf("%s\n", m->transcribe_pcm_vad(audio.samples, audio.sample_rate, dec, lang, opts, ext).c_str());
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "parakeet-cli: transcribe failed: %s\n", e.what());
        return 1;
    }
    return 0;
}

// parakeet-cli transcribe --model <m.gguf> --input <wav|-> [--decoder ctc|tdt]
//                         [--stream]
// Prints the transcript. Default decoder is chosen by arch (TDT for transducer
// archs, CTC for ctc arch — matching NeMo's cur_decoder default). --stream uses
// the cache-aware streaming path (EOU streaming model only).
static int cmd_transcribe(int argc, char** argv) {
    std::string model, input, decoder_str, lang;
    bool stream = false;
    bool timestamps = false;
    bool json = false;
    bool vad = false;
    std::string vad_model, vad_component, component;
    VadOverrides vad_ov;
    double d = 0.0;
    auto parse_pos = [](const char* str, double& out) {
        char* end = nullptr;
        out = std::strtod(str, &end);
        return end != str && *end == '\0' && std::isfinite(out) && out > 0.0 && out <= 1e6;
    };
    bool score_norm = true;
    int beam_size = 0;
    int nbest = 0;
    int threads = 0;  // 0 == unset -> use the persistent-backend default
    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model = argv[++i];
        } else if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input = argv[++i];
        } else if (std::strcmp(argv[i], "--decoder") == 0 && i + 1 < argc) {
            decoder_str = argv[++i];
        } else if (std::strcmp(argv[i], "--lang") == 0 && i + 1 < argc) {
            lang = argv[++i];
        } else if (std::strcmp(argv[i], "--stream") == 0) {
            stream = true;
        } else if (std::strcmp(argv[i], "--timestamps") == 0) {
            timestamps = true;
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--json") == 0) {
            json = true;
        } else if (std::strcmp(argv[i], "--beam-size") == 0 && i + 1 < argc) {
            beam_size = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--nbest") == 0 && i + 1 < argc) {
            nbest = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--no-score-norm") == 0) {
            score_norm = false;
        } else if (std::strcmp(argv[i], "--vad") == 0) {
            vad = true;
        } else if (std::strcmp(argv[i], "--component") == 0 && i + 1 < argc) {
            component = argv[++i];
        } else if (std::strcmp(argv[i], "--vad-component") == 0 && i + 1 < argc) {
            vad_component = argv[++i];
            vad = true;
        } else if (std::strcmp(argv[i], "--vad-model") == 0 && i + 1 < argc) {
            vad_model = argv[++i];
            vad = true;
        } else if (std::strcmp(argv[i], "--vad-threshold") == 0 && i + 1 < argc) {
            if (!parse_pos(argv[++i], d) || d > 1.0) { std::fprintf(stderr, "parakeet-cli: --vad-threshold must be in (0,1]\n"); return 2; }
            vad_ov.threshold = d;
        } else if (std::strcmp(argv[i], "--vad-min-pause") == 0 && i + 1 < argc) {
            if (!parse_pos(argv[++i], d)) { std::fprintf(stderr, "parakeet-cli: --vad-min-pause must be > 0\n"); return 2; }
            vad_ov.min_pause = d;
        } else if (std::strcmp(argv[i], "--vad-min-speech") == 0 && i + 1 < argc) {
            if (!parse_pos(argv[++i], d)) { std::fprintf(stderr, "parakeet-cli: --vad-min-speech must be > 0\n"); return 2; }
            vad_ov.min_speech = d;
        } else if (std::strcmp(argv[i], "--vad-max-seg") == 0 && i + 1 < argc) {
            if (!parse_pos(argv[++i], d)) { std::fprintf(stderr, "parakeet-cli: --vad-max-seg must be > 0\n"); return 2; }
            vad_ov.max_seg = d;
        }
    }
    if (model.empty() || input.empty()) {
        std::fprintf(stderr,
            "usage: parakeet-cli transcribe --model <m.gguf> --input <wav|-> "
            "[--decoder ctc|tdt] [--lang <locale>] [--stream] [--timestamps] "
            "[--threads N] [--json] "
            "[--component NAME] "
            "[--vad [--vad-model <silero.gguf>] [--vad-component NAME] [--vad-threshold F=0.5] [--vad-min-pause SEC] "
            "[--vad-min-speech SEC] [--vad-max-seg SEC=30]] "
            "[--beam-size N [--nbest N] [--no-score-norm]]\n");
        return 2;
    }
    // A bundle GGUF: pick the ASR component (the only "asr" one unless --component).
    if (!resolve_asr_component(model, component, g_asr_component)) return 2;
    // Apply the thread override (offline + streaming graph compute). When unset
    // the persistent-backend default (kDefaultThreads) is used.
    if (threads > 0) pk::set_num_threads(threads);

    if (stream) {
        if (vad) {
            std::fprintf(stderr, "parakeet-cli: --vad is offline only\n");
            return 2;
        }
        if (beam_size != 0 || nbest != 0) {
            std::fprintf(stderr,
                "parakeet-cli: --beam-size/--nbest are offline TDT only\n");
            return 2;
        }
        if (!decoder_str.empty()) {
            std::fprintf(stderr,
                "parakeet-cli: --stream is RNN-T only; --decoder is ignored\n");
        }
        if (json) {
            std::fprintf(stderr,
                "parakeet-cli: --json is not supported with --stream\n");
            return 2;
        }
        return cmd_transcribe_stream(model, input, timestamps, lang);
    }

    // Resolve the decoder selector.
    pk::Decoder dec = pk::Decoder::kDefault;
    int dec_int = 0;  // C-API decoder selector (0 default, 1 ctc, 2 tdt)
    if (!decoder_str.empty()) {
        if (decoder_str == "ctc") {
            dec = pk::Decoder::kCTC;
            dec_int = 1;
        } else if (decoder_str == "tdt") {
            dec = pk::Decoder::kTDT;
            dec_int = 2;
        } else {
            std::fprintf(stderr, "parakeet-cli: unknown --decoder '%s' (want ctc|tdt)\n",
                         decoder_str.c_str());
            return 2;
        }
    }

    if (vad) {
        if (beam_size != 0 || nbest != 0) {
            std::fprintf(stderr, "parakeet-cli: --vad works with greedy decoding only\n");
            return 2;
        }
        return cmd_transcribe_vad(model, input, dec, lang, timestamps, json, vad_ov, vad_model, vad_component);
    }
    if (nbest != 0 && beam_size == 0) {
        std::fprintf(stderr,
            "parakeet-cli: --nbest requires --beam-size\n");
        return 2;
    }
    if (!score_norm && beam_size == 0) {
        std::fprintf(stderr,
            "parakeet-cli: --no-score-norm requires --beam-size\n");
        return 2;
    }
    if (beam_size != 0) {
        if (beam_size < 1) {
            std::fprintf(stderr,
                "parakeet-cli: --beam-size must be at least 1\n");
            return 2;
        }
        if (nbest == 0) nbest = beam_size;
        if (nbest < 1 || nbest > beam_size) {
            std::fprintf(stderr,
                "parakeet-cli: require beam-size >= nbest >= 1\n");
            return 2;
        }
        if (decoder_str == "ctc") {
            std::fprintf(stderr,
                "parakeet-cli: TDT N-best cannot use --decoder ctc\n");
            return 2;
        }
        if (timestamps) {
            std::fprintf(stderr,
                "parakeet-cli: --timestamps is redundant with N-best JSON\n");
            return 2;
        }

        if (is_stdin_input(input)) {
            pk::Audio audio;
            if (!load_audio_arg_16k_mono(input, audio)) {
                std::fprintf(stderr, "parakeet-cli: failed to load audio stdin\n");
                return 1;
            }
            try {
                std::unique_ptr<pk::Model> m = load_asr(model);
                if (!m) {
                    std::fprintf(stderr,
                        "parakeet-cli: failed to load model %s\n", model.c_str());
                    return 1;
                }
                std::vector<pk::NBestTranscription> hypotheses =
                    m->transcribe_pcm_nbest(
                        audio.samples, audio.sample_rate,
                        beam_size, nbest, score_norm, lang);
                std::string output = pk::nbest_transcriptions_to_json(
                    hypotheses, beam_size, score_norm, model_frame_sec(*m));
                std::printf("%s\n", output.c_str());
            } catch (const std::exception& e) {
                std::fprintf(stderr,
                    "parakeet-cli: N-best transcription failed: %s\n", e.what());
                return 1;
            }
            return 0;
        }

        parakeet_ctx* ctx = load_asr_ctx(model);
        if (!ctx) {
            std::fprintf(stderr,
                "parakeet-cli: failed to load model %s\n", model.c_str());
            return 1;
        }
        char* output = parakeet_capi_transcribe_path_nbest_json(
            ctx, input.c_str(), beam_size, nbest, score_norm ? 1 : 0,
            lang.empty() ? nullptr : lang.c_str());
        if (!output) {
            std::fprintf(stderr,
                "parakeet-cli: N-best transcription failed: %s\n",
                parakeet_capi_last_error(ctx));
            parakeet_capi_free(ctx);
            return 1;
        }
        std::printf("%s\n", output);
        parakeet_capi_free_string(output);
        parakeet_capi_free(ctx);
        return 0;
    }

    // --json: emit the C-API JSON document (text + word/token timestamps + conf).
    if (json) {
        if (is_stdin_input(input)) {
            pk::Audio audio;
            if (!load_audio_arg_16k_mono(input, audio)) {
                std::fprintf(stderr, "parakeet-cli: failed to load audio stdin\n");
                return 1;
            }
            try {
                std::unique_ptr<pk::Model> m = load_asr(model);
                if (!m) {
                    std::fprintf(stderr, "parakeet-cli: failed to load model %s\n", model.c_str());
                    return 1;
                }
                pk::Transcription tr =
                    m->transcribe_with_timestamps(audio.samples, audio.sample_rate, dec, lang);
                std::string j = pk::transcription_to_json(tr, model_frame_sec(*m));
                std::printf("%s\n", j.c_str());
            } catch (const std::exception& e) {
                std::fprintf(stderr, "parakeet-cli: transcribe failed: %s\n", e.what());
                return 1;
            }
            return 0;
        }

        parakeet_ctx* ctx = load_asr_ctx(model);
        if (!ctx) {
            std::fprintf(stderr, "parakeet-cli: failed to load model %s\n", model.c_str());
            return 1;
        }
        char* j = parakeet_capi_transcribe_path_json(ctx, input.c_str(), dec_int);
        if (!j) {
            std::fprintf(stderr, "parakeet-cli: transcribe failed: %s\n",
                         parakeet_capi_last_error(ctx));
            parakeet_capi_free(ctx);
            return 1;
        }
        std::printf("%s\n", j);
        parakeet_capi_free_string(j);
        parakeet_capi_free(ctx);
        return 0;
    }

    // --timestamps: print one line per word `<start>-<end>  <word>  (<conf>)`.
    if (timestamps) {
        pk::Audio audio;
        if (is_stdin_input(input) && !load_audio_arg_16k_mono(input, audio)) {
            std::fprintf(stderr, "parakeet-cli: failed to load audio stdin\n");
            return 1;
        }
        try {
            std::unique_ptr<pk::Model> m = load_asr(model);
            if (!m) {
                std::fprintf(stderr, "parakeet-cli: failed to load model %s\n", model.c_str());
                return 1;
            }
            // `lang` (empty -> model default) selects the language prompt for
            // multilingual models; ignored by non-prompt models.
            pk::Transcription tr = is_stdin_input(input)
                ? m->transcribe_with_timestamps(audio.samples, audio.sample_rate, dec, lang)
                : m->transcribe_path_with_timestamps(input, dec, lang);
            for (const pk::Word& w : tr.words)
                std::printf("%.2f-%.2f  %s  (%.2f)\n", w.start, w.end,
                            w.text.c_str(), w.conf);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "transcribe failed: %s\n", e.what());
            return 1;
        }
        return 0;
    }

    // Plain transcript. When --lang is given, go through the load-once C-API
    // language variant so the language prompt is selected (and an unknown locale
    // surfaces as a clean error). With no --lang keep the existing free-function
    // path so behavior for every other model is byte-for-byte unchanged.
    if (!lang.empty() && !is_stdin_input(input)) {
        parakeet_ctx* ctx = load_asr_ctx(model);
        if (!ctx) {
            std::fprintf(stderr, "parakeet-cli: failed to load model %s\n", model.c_str());
            return 1;
        }
        char* t = parakeet_capi_transcribe_path_lang(ctx, input.c_str(), dec_int,
                                                     lang.c_str());
        if (!t) {
            std::fprintf(stderr, "transcribe failed: %s\n", parakeet_capi_last_error(ctx));
            parakeet_capi_free(ctx);
            return 1;
        }
        std::printf("%s\n", t);
        parakeet_capi_free_string(t);
        parakeet_capi_free(ctx);
        return 0;
    }

    std::string text;
    try {
        if (is_stdin_input(input)) {
            pk::Audio audio;
            if (!load_audio_arg_16k_mono(input, audio)) {
                std::fprintf(stderr, "parakeet-cli: failed to load audio stdin\n");
                return 1;
            }
            std::unique_ptr<pk::Model> m = load_asr(model);
            if (!m) {
                std::fprintf(stderr, "parakeet-cli: failed to load model %s\n", model.c_str());
                return 1;
            }
            text = m->transcribe_pcm(audio.samples, audio.sample_rate, dec, lang);
        } else {
            // Same as pk::transcribe(), with the bundle component applied.
            std::unique_ptr<pk::Model> m = load_asr(model);
            if (!m) throw std::runtime_error("parakeet: failed to load model: " + model);
            text = m->transcribe_path(input, dec);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "transcribe failed: %s\n", e.what());
        return 1;
    }
    std::printf("%s\n", text.c_str());
    return 0;
}

// ---------------------------------------------------------------------------
// quantize: re-quantize the SAME allowlisted linear weights as the converter's
// f16/q8_0 path (docs/quantization.md) to a target ggml type -- including the
// K-quants (Q4_K/Q5_K/Q6_K) that the Python gguf writer can't produce. Every
// non-allowlisted tensor (and any allowlisted tensor that isn't shape-eligible)
// is copied verbatim in its stored type. All KV metadata is copied unchanged.
// ---------------------------------------------------------------------------

// Returns true if `name` is on the Task-2 quantization allowlist (mul_mat src0
// weights only). Mirrors _QUANTIZABLE_PATTERNS in convert_parakeet_to_gguf.py.
//   ^encoder\.layers\.\d+\.feed_forward[12]\.linear[12]\.weight$
//   ^encoder\.layers\.\d+\.self_attn\.linear_(q|k|v|out|pos)\.weight$
//   ^encoder\.pre_encode\.out\.weight$
//   ^joint\.enc\.weight$
//   ^joint\.pred\.weight$
static bool is_quantizable_name(const std::string& n) {
    if (n == "encoder.pre_encode.out.weight") return true;
    if (n == "joint.enc.weight") return true;
    if (n == "joint.pred.weight") return true;
    // encoder.layers.<d+>.<rest>
    const char* prefix = "encoder.layers.";
    if (n.rfind(prefix, 0) != 0) return false;
    size_t i = std::strlen(prefix);
    size_t start = i;
    while (i < n.size() && std::isdigit(static_cast<unsigned char>(n[i]))) ++i;
    if (i == start || i >= n.size() || n[i] != '.') return false;  // need digits then '.'
    std::string rest = n.substr(i + 1);
    // feed_forward{1,2}.linear{1,2}.weight
    if ((rest == "feed_forward1.linear1.weight") ||
        (rest == "feed_forward1.linear2.weight") ||
        (rest == "feed_forward2.linear1.weight") ||
        (rest == "feed_forward2.linear2.weight")) return true;
    // self_attn.linear_{q,k,v,out,pos}.weight
    if ((rest == "self_attn.linear_q.weight")   ||
        (rest == "self_attn.linear_k.weight")   ||
        (rest == "self_attn.linear_v.weight")   ||
        (rest == "self_attn.linear_out.weight") ||
        (rest == "self_attn.linear_pos.weight")) return true;
    return false;
}

static bool parse_quant_type(const std::string& s, ggml_type& out) {
    std::string t = s;
    std::transform(t.begin(), t.end(), t.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    if (t == "q4_0") { out = GGML_TYPE_Q4_0; return true; }
    if (t == "q5_0") { out = GGML_TYPE_Q5_0; return true; }
    if (t == "q8_0") { out = GGML_TYPE_Q8_0; return true; }
    if (t == "q4_k") { out = GGML_TYPE_Q4_K; return true; }
    if (t == "q5_k") { out = GGML_TYPE_Q5_K; return true; }
    if (t == "q6_k") { out = GGML_TYPE_Q6_K; return true; }
    return false;
}

static bool is_k_quant(ggml_type t) {
    return t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q6_K;
}

static int cmd_quantize(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: parakeet-cli quantize <in.gguf> <out.gguf> "
            "<q4_0|q5_0|q8_0|q4_k|q5_k|q6_k>\n");
        return 2;
    }
    const std::string in_path  = argv[0];
    const std::string out_path = argv[1];
    ggml_type qtype;
    if (!parse_quant_type(argv[2], qtype)) {
        std::fprintf(stderr,
            "parakeet-cli quantize: unknown type '%s' "
            "(want q4_0|q5_0|q8_0|q4_k|q5_k|q6_k)\n", argv[2]);
        return 2;
    }

    // Load the source GGUF the same way ModelLoader does: gguf_init_from_file
    // with a backing ggml_context so the tensor data is read into memory.
    struct ggml_context* src_ctx = nullptr;
    struct gguf_init_params p{ /*no_alloc*/ false, /*ctx*/ &src_ctx };
    struct gguf_context* src = gguf_init_from_file(in_path.c_str(), p);
    if (!src) {
        std::fprintf(stderr, "failed to open %s\n", in_path.c_str());
        return 1;
    }

    const int64_t block = ggml_blck_size(qtype);  // 32 for q*_0, 256 for K-quants

    // Destination: empty gguf, copy all KV verbatim, then add tensors.
    struct gguf_context* dst = gguf_init_empty();
    gguf_set_kv(dst, src);  // copies every KV pair unchanged

    const int64_t nt = gguf_get_n_tensors(src);
    // Holds quantized buffers alive until gguf_write_to_file copies them out.
    std::vector<std::vector<uint8_t>> quant_bufs;
    quant_bufs.reserve(static_cast<size_t>(nt));
    // We must reset each quantized tensor's type/data on the in-memory ggml
    // tensor so gguf_add_tensor records the new type and points at our buffer.
    struct ggml_init_params meta_p{ /*mem_size*/ ggml_tensor_overhead() * (nt + 1),
                                    /*mem_buffer*/ nullptr, /*no_alloc*/ true };
    struct ggml_context* meta = ggml_init(meta_p);

    int n_quant = 0, n_kept = 0, n_skipped = 0;
    for (int64_t i = 0; i < nt; ++i) {
        const char* name = gguf_get_tensor_name(src, i);
        struct ggml_tensor* t = ggml_get_tensor(src_ctx, name);

        bool do_quant = false;
        if (is_quantizable_name(name) && t->type == GGML_TYPE_F32) {
            const bool two_d   = (t->ne[2] == 1 && t->ne[3] == 1 &&
                                  t->ne[0] >= 32 && t->ne[1] >= 32);
            const bool blk_ok  = (t->ne[0] % block == 0);
            if (two_d && blk_ok) {
                do_quant = true;
            } else if (two_d && !blk_ok) {
                std::fprintf(stderr,
                    "  keep F32: %-48s ne0=%lld not divisible by %s block %lld\n",
                    name, (long long)t->ne[0],
                    is_k_quant(qtype) ? "K-quant superblock" : "block",
                    (long long)block);
                ++n_skipped;
            }
        }

        if (do_quant) {
            const int64_t n_per_row = t->ne[0];
            const int64_t nrows     = t->ne[1];
            const size_t  out_bytes = ggml_row_size(qtype, n_per_row) * (size_t)nrows;
            quant_bufs.emplace_back(out_bytes);
            const float* fsrc = (const float*)t->data;
            // imatrix = NULL: none of q4_0/q5_0/q8_0/q4_k/q5_k/q6_k require one.
            ggml_quantize_chunk(qtype, fsrc, quant_bufs.back().data(),
                                /*start*/ 0, nrows, n_per_row, /*imatrix*/ nullptr);
            // Build a fresh meta tensor in the new type pointing at our buffer.
            struct ggml_tensor* q = ggml_new_tensor_2d(meta, qtype, n_per_row, nrows);
            ggml_set_name(q, name);
            q->data = quant_bufs.back().data();
            gguf_add_tensor(dst, q);
            ++n_quant;
        } else {
            // Copy verbatim in its stored type/data.
            gguf_add_tensor(dst, t);
            ++n_kept;
        }
    }

    const bool ok = gguf_write_to_file(dst, out_path.c_str(), /*only_meta*/ false);

    std::printf("quantize: %s -> %s [%s]\n", in_path.c_str(), out_path.c_str(), argv[2]);
    std::printf("  quantized %d tensor(s), copied %d verbatim, %d allowlisted kept F32 (block).\n",
                n_quant, n_kept, n_skipped);

    ggml_quantize_free();
    ggml_free(meta);
    gguf_free(dst);
    gguf_free(src);
    ggml_free(src_ctx);

    if (!ok) {
        std::fprintf(stderr, "failed to write %s\n", out_path.c_str());
        return 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// bench: clean per-file transcription timing for the NeMo-vs-ours benchmark.
//
// Loads the model ONCE (timed -> load_ms), then for each audio path in a
// manifest: loads the WAV, computes audio_sec = samples/16000, and times ONLY
// pk::Model::transcribe_path (steady_clock, ms). Emits a single JSON document
// so the Python runner can compute RTFx (audio_sec / proc_sec) fairly without
// the one-time model-load cost polluting the per-file numbers.
//
// --threads N controls the ggml compute threads for EVERY graph computation
// (via pk::set_num_threads, read inside pk::run_graph). When unset we leave the
// process-global override clear so the components' built-in default is used.
// ---------------------------------------------------------------------------

// Append `s` to `out` as a JSON string literal (quoted), escaping per RFC 8259.
// Mirrors append_json_string in src/transcription_json.cpp (UTF-8 >= 0x80 passes
// through verbatim).
static void bench_json_string(std::string& out, const std::string& s) {
    out += '"';
    char esc[8];
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    std::snprintf(esc, sizeof(esc), "\\u%04x", (unsigned)c);
                    out += esc;
                } else {
                    out += (char)c;
                }
        }
    }
    out += '"';
}

// Reads one audio path per line from the manifest. Blank lines and lines whose
// first non-space char is '#' are ignored. A tab-separated `path\tref` line is
// accepted -- only the first field (the path) is taken. Leading/trailing
// whitespace on the path field is trimmed.
static std::vector<std::string> read_manifest(const std::string& path, bool& ok) {
    std::vector<std::string> out;
    std::ifstream f(path);
    if (!f) { ok = false; return out; }
    ok = true;
    std::string line;
    while (std::getline(f, line)) {
        // Strip a trailing CR (CRLF manifests).
        if (!line.empty() && line.back() == '\r') line.pop_back();
        // First field only (tab-separated path\tref is allowed).
        size_t tab = line.find('\t');
        std::string p = (tab == std::string::npos) ? line : line.substr(0, tab);
        // Trim surrounding whitespace.
        size_t b = p.find_first_not_of(" \t");
        if (b == std::string::npos) continue;                 // blank
        size_t e = p.find_last_not_of(" \t");
        p = p.substr(b, e - b + 1);
        if (p.empty() || p[0] == '#') continue;               // blank / comment
        out.push_back(p);
    }
    return out;
}

static int cmd_bench(int argc, char** argv) {
    std::string model, manifest, decoder_str, json_out, lang;
    int threads = 0;  // 0 == unset -> use the components' built-in default
    int concurrency = 1;  // >1: worker threads over a pool of CPU backends
    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model = argv[++i];
        } else if (std::strcmp(argv[i], "--manifest") == 0 && i + 1 < argc) {
            manifest = argv[++i];
        } else if (std::strcmp(argv[i], "--decoder") == 0 && i + 1 < argc) {
            decoder_str = argv[++i];
        } else if (std::strcmp(argv[i], "--lang") == 0 && i + 1 < argc) {
            lang = argv[++i];
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--concurrency") == 0 && i + 1 < argc) {
            concurrency = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            json_out = argv[++i];
        }
    }
    if (model.empty() || manifest.empty()) {
        std::fprintf(stderr,
            "usage: parakeet-cli bench --model <m.gguf> --manifest <file> "
            "[--decoder ctc|tdt] [--lang <locale>] [--threads N] [--concurrency K] "
            "[--json <out>]\n");
        return 2;
    }

    // Resolve the decoder selector (matches `transcribe`).
    pk::Decoder dec = pk::Decoder::kDefault;
    if (!decoder_str.empty()) {
        if (decoder_str == "ctc") {
            dec = pk::Decoder::kCTC;
        } else if (decoder_str == "tdt") {
            dec = pk::Decoder::kTDT;
        } else {
            std::fprintf(stderr,
                "parakeet-cli bench: unknown --decoder '%s' (want ctc|tdt)\n",
                decoder_str.c_str());
            return 2;
        }
    }

    // Apply the thread count to EVERY ggml graph computation. When --threads is
    // omitted we report the components' built-in default in the JSON so the
    // runner records the thread count that was actually used.
    // With --concurrency K > 1, --threads is the thread count of EACH backend and
    // the global override stays unset (pooled backends ignore it).
    if (concurrency < 1) concurrency = 1;
    int reported_threads = threads;
    if (concurrency > 1) {
        if (reported_threads <= 0) reported_threads = std::max(1, 8 / concurrency);
    } else if (threads > 0) {
        pk::set_num_threads(threads);
    } else {
        reported_threads = 8;  // the persistent-backend default (kDefaultThreads)
    }

    bool man_ok = false;
    std::vector<std::string> paths = read_manifest(manifest, man_ok);
    if (!man_ok) {
        std::fprintf(stderr, "parakeet-cli bench: failed to read manifest %s\n",
                     manifest.c_str());
        return 1;
    }
    if (paths.empty()) {
        std::fprintf(stderr, "parakeet-cli bench: manifest %s has no audio paths\n",
                     manifest.c_str());
        return 1;
    }

    using clock = std::chrono::steady_clock;
    auto ms_since = [](clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    };

    // Load the model ONCE -- timed, and excluded from per-file proc_ms.
    auto t_load = clock::now();
    std::unique_ptr<pk::Model> m = pk::Model::load(model);
    double load_ms = ms_since(t_load);
    if (!m) {
        std::fprintf(stderr, "parakeet-cli bench: failed to load model %s\n",
                     model.c_str());
        return 1;
    }

    if (concurrency > 1) {
        const int eff = m->set_concurrency(concurrency, reported_threads);
        if (eff != concurrency) {
            std::fprintf(stderr,
                "parakeet-cli bench: --concurrency %d not available on this device; using %d\n",
                concurrency, eff);
            concurrency = eff;
        }
    }

    struct FileResult { std::string path; double audio_sec; double proc_ms; std::string text; };
    std::vector<FileResult> results;
    results.reserve(paths.size());

    // Warm up once (untimed): the first transcribe pays one-time costs — lazy
    // device weight upload + CUDA kernel/cuBLAS init on a GPU backend (~100x the
    // steady-state per-file time), or weight realization on CPU. Excluding it
    // keeps per-file proc_ms (and RTFx) steady-state and fair vs other engines.
    // With a pool, every backend allocates its own graph buffers on first use, so
    // run the warmup from `concurrency` threads at once, twice.
    {
        pk::Audio warm;
        if (pk::load_audio_16k_mono(paths[0], warm)) {
            for (int round = 0; round < (concurrency > 1 ? 2 : 1); ++round) {
                std::vector<std::thread> ws;
                for (int w = 0; w < concurrency; ++w)
                    ws.emplace_back([&] {
                        try { (void)m->transcribe_pcm(warm.samples, 16000, dec, lang); }
                        catch (...) {}
                    });
                for (auto& t : ws) t.join();
            }
        }
    }

    // Decode all audio up front so the timed region has no file IO.
    std::vector<pk::Audio> audios(paths.size());
    for (size_t i = 0; i < paths.size(); ++i) {
        if (!pk::load_audio_16k_mono(paths[i], audios[i])) {
            std::fprintf(stderr, "parakeet-cli bench: failed to load audio %s\n",
                         paths[i].c_str());
            return 1;
        }
    }
    results.assign(paths.size(), FileResult{});

    // `concurrency` workers pull clips from a shared index. With one worker this
    // is the original sequential loop. Each clip is timed on its own.
    std::atomic<size_t> next_clip{0};
    std::atomic<bool> failed{false};
    auto worker = [&]() {
        for (;;) {
            const size_t i = next_clip.fetch_add(1);
            if (i >= paths.size() || failed.load()) return;
            // audio_sec from the decoded 16 kHz sample count.
            const double audio_sec = (double)audios[i].samples.size() / 16000.0;
            // Time ONLY the transcription (model already loaded).
            auto t_proc = clock::now();
            std::string text;
            try {
                text = m->transcribe_pcm(audios[i].samples, 16000, dec, lang);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "parakeet-cli bench: transcribe failed on %s: %s\n",
                             paths[i].c_str(), e.what());
                failed = true;
                return;
            }
            results[i] = {paths[i], audio_sec, ms_since(t_proc), text};
        }
    };
    auto t_wall = clock::now();
    if (concurrency > 1) {
        std::vector<std::thread> ws;
        for (int w = 0; w < concurrency; ++w) ws.emplace_back(worker);
        for (auto& t : ws) t.join();
    } else {
        worker();
    }
    const double wall_ms = ms_since(t_wall);
    if (failed) return 1;

    // Hand-roll the JSON document.
    std::string out;
    out.reserve(256 + results.size() * 128);
    out += "{\"model\":";
    bench_json_string(out, model);
    char numbuf[64];
    std::snprintf(numbuf, sizeof(numbuf), ",\"threads\":%d", reported_threads);
    out += numbuf;
    std::snprintf(numbuf, sizeof(numbuf), ",\"load_ms\":%.3f", load_ms);
    out += numbuf;
    if (concurrency > 1) {
        // Throughput of the whole run: wall time of all workers together.
        double total_audio = 0;
        for (const FileResult& r : results) total_audio += r.audio_sec;
        std::snprintf(numbuf, sizeof(numbuf), ",\"concurrency\":%d", concurrency);
        out += numbuf;
        std::snprintf(numbuf, sizeof(numbuf), ",\"wall_ms\":%.3f", wall_ms);
        out += numbuf;
        std::snprintf(numbuf, sizeof(numbuf), ",\"aggregate_rtfx\":%.3f",
                      wall_ms > 0 ? total_audio / (wall_ms / 1000.0) : 0.0);
        out += numbuf;
    }
    out += ",\"files\":[";
    for (size_t i = 0; i < results.size(); ++i) {
        if (i) out += ',';
        out += "{\"path\":";
        bench_json_string(out, results[i].path);
        std::snprintf(numbuf, sizeof(numbuf), ",\"audio_sec\":%.6f", results[i].audio_sec);
        out += numbuf;
        std::snprintf(numbuf, sizeof(numbuf), ",\"proc_ms\":%.3f", results[i].proc_ms);
        out += numbuf;
        out += ",\"text\":";
        bench_json_string(out, results[i].text);
        out += '}';
    }
    out += "]}";

    if (!json_out.empty()) {
        std::ofstream of(json_out, std::ios::binary | std::ios::trunc);
        if (!of) {
            std::fprintf(stderr, "parakeet-cli bench: failed to write %s\n",
                         json_out.c_str());
            return 1;
        }
        of << out << '\n';
    } else {
        std::printf("%s\n", out.c_str());
    }
    return 0;
}

// Measures BATCHED encoder throughput at one or more batch sizes. Mirrors
// cmd_bench's arg parsing / model load / manifest read / warmup, but instead of
// timing one clip at a time it groups the clips into batches of size B and times
// the wall-clock cost of running every batch through transcribe_pcm_batch.
//
// The B=1 row goes through the SAME fused batched encoder path with one-clip
// batches, so it is the apples-to-apples baseline against which the batching win
// (B=4, B=8, ...) is read.
static int cmd_bench_batch(int argc, char** argv) {
    std::string model, manifest, decoder_str, json_out;
    std::string batch_sizes_str = "1,4,8";
    int threads = 0;  // 0 == unset -> use the components' built-in default
    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model = argv[++i];
        } else if (std::strcmp(argv[i], "--manifest") == 0 && i + 1 < argc) {
            manifest = argv[++i];
        } else if (std::strcmp(argv[i], "--decoder") == 0 && i + 1 < argc) {
            decoder_str = argv[++i];
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--batch-sizes") == 0 && i + 1 < argc) {
            batch_sizes_str = argv[++i];
        } else if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            json_out = argv[++i];
        }
    }
    if (model.empty() || manifest.empty()) {
        std::fprintf(stderr,
            "usage: parakeet-cli bench-batch --model <m.gguf> --manifest <file> "
            "[--decoder ctc|tdt] [--threads N] [--batch-sizes 1,4,8] [--json <out>]\n");
        return 2;
    }

    // Parse --batch-sizes (comma-separated positive ints).
    std::vector<int> batch_sizes;
    {
        std::stringstream ss(batch_sizes_str);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            // Trim surrounding whitespace.
            size_t b = tok.find_first_not_of(" \t");
            if (b == std::string::npos) continue;
            size_t e = tok.find_last_not_of(" \t");
            int v = std::atoi(tok.substr(b, e - b + 1).c_str());
            if (v > 0) batch_sizes.push_back(v);
        }
    }
    if (batch_sizes.empty()) {
        std::fprintf(stderr,
            "parakeet-cli bench-batch: no valid --batch-sizes (want e.g. 1,4,8)\n");
        return 2;
    }

    // Resolve the decoder selector (matches `transcribe` / `bench`).
    pk::Decoder dec = pk::Decoder::kDefault;
    if (!decoder_str.empty()) {
        if (decoder_str == "ctc") {
            dec = pk::Decoder::kCTC;
        } else if (decoder_str == "tdt") {
            dec = pk::Decoder::kTDT;
        } else {
            std::fprintf(stderr,
                "parakeet-cli bench-batch: unknown --decoder '%s' (want ctc|tdt)\n",
                decoder_str.c_str());
            return 2;
        }
    }

    // Apply the thread count to EVERY ggml graph computation. When --threads is
    // omitted we report the components' built-in default.
    int reported_threads = threads;
    if (threads > 0) {
        pk::set_num_threads(threads);
    } else {
        reported_threads = 8;  // the persistent-backend default (kDefaultThreads)
    }

    bool man_ok = false;
    std::vector<std::string> paths = read_manifest(manifest, man_ok);
    if (!man_ok) {
        std::fprintf(stderr, "parakeet-cli bench-batch: failed to read manifest %s\n",
                     manifest.c_str());
        return 1;
    }
    if (paths.empty()) {
        std::fprintf(stderr, "parakeet-cli bench-batch: manifest %s has no audio paths\n",
                     manifest.c_str());
        return 1;
    }

    using clock = std::chrono::steady_clock;
    auto ms_since = [](clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    };

    // Load ALL clips into memory ONCE (untimed) so the per-batch loop only times
    // the encoder/decoder work, not audio decode/IO.
    std::vector<std::vector<float>> clips;
    clips.reserve(paths.size());
    double total_audio_sec = 0.0;
    for (const std::string& p : paths) {
        pk::Audio audio;
        if (!pk::load_audio_16k_mono(p, audio)) {
            std::fprintf(stderr, "parakeet-cli bench-batch: failed to load audio %s\n",
                         p.c_str());
            return 1;
        }
        total_audio_sec += (double)audio.samples.size() / 16000.0;
        clips.push_back(std::move(audio.samples));
    }

    // Load the model ONCE -- timed, and excluded from per-batch proc_ms.
    auto t_load = clock::now();
    std::unique_ptr<pk::Model> m = pk::Model::load(model);
    double load_ms = ms_since(t_load);
    if (!m) {
        std::fprintf(stderr, "parakeet-cli bench-batch: failed to load model %s\n",
                     model.c_str());
        return 1;
    }

    // Warm up once (untimed): pays the one-time lazy weight upload / kernel init
    // so per-batch timings are steady-state.
    {
        std::vector<std::vector<float>> warm{clips[0]};
        try {
            (void)m->transcribe_pcm_batch(warm, 16000, dec);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "parakeet-cli bench-batch: warmup failed: %s\n", e.what());
            return 1;
        }
    }

    struct BatchResult { int batch_size; double proc_ms; size_t n_clips;
                         double clips_per_sec; double rtfx; };
    std::vector<BatchResult> results;
    results.reserve(batch_sizes.size());

    for (int B : batch_sizes) {
        auto t_proc = clock::now();
        for (size_t s = 0; s < clips.size(); s += (size_t)B) {
            size_t end = std::min(clips.size(), s + (size_t)B);
            std::vector<std::vector<float>> chunk(clips.begin() + (long)s,
                                                  clips.begin() + (long)end);
            try {
                (void)m->transcribe_pcm_batch(chunk, 16000, dec);
            } catch (const std::exception& e) {
                std::fprintf(stderr,
                    "parakeet-cli bench-batch: transcribe failed at batch_size=%d: %s\n",
                    B, e.what());
                return 1;
            }
        }
        double proc_ms = ms_since(t_proc);
        double secs = proc_ms / 1000.0;
        double clips_per_sec = secs > 0.0 ? (double)clips.size() / secs : 0.0;
        double rtfx = secs > 0.0 ? total_audio_sec / secs : 0.0;
        results.push_back({B, proc_ms, clips.size(), clips_per_sec, rtfx});
    }

    // Human-readable summary table to stderr.
    std::fprintf(stderr,
        "\nbench-batch: %zu clips, %.2f s audio, decoder=%s, threads=%d, load_ms=%.1f\n",
        clips.size(), total_audio_sec,
        decoder_str.empty() ? "default" : decoder_str.c_str(),
        reported_threads, load_ms);
    std::fprintf(stderr, "  %-12s %-14s %-14s %-10s\n",
                 "batch_size", "proc_ms", "clips/sec", "RTFx");
    for (const BatchResult& r : results) {
        std::fprintf(stderr, "  %-12d %-14.1f %-14.2f %-10.2f\n",
                     r.batch_size, r.proc_ms, r.clips_per_sec, r.rtfx);
    }

    // Hand-roll the JSON document.
    std::string out;
    out.reserve(256 + results.size() * 96);
    out += "{\"model\":";
    bench_json_string(out, model);
    out += ",\"decoder\":";
    bench_json_string(out, decoder_str.empty() ? std::string("default") : decoder_str);
    char numbuf[96];
    std::snprintf(numbuf, sizeof(numbuf), ",\"threads\":%d", reported_threads);
    out += numbuf;
    std::snprintf(numbuf, sizeof(numbuf), ",\"n_clips\":%zu", clips.size());
    out += numbuf;
    std::snprintf(numbuf, sizeof(numbuf), ",\"total_audio_sec\":%.6f", total_audio_sec);
    out += numbuf;
    out += ",\"results\":[";
    for (size_t i = 0; i < results.size(); ++i) {
        if (i) out += ',';
        std::snprintf(numbuf, sizeof(numbuf),
            "{\"batch_size\":%d,\"proc_ms\":%.3f,\"clips_per_sec\":%.6f,\"rtfx\":%.6f}",
            results[i].batch_size, results[i].proc_ms,
            results[i].clips_per_sec, results[i].rtfx);
        out += numbuf;
    }
    out += "]}";

    if (!json_out.empty()) {
        std::ofstream of(json_out, std::ios::binary | std::ios::trunc);
        if (!of) {
            std::fprintf(stderr, "parakeet-cli bench-batch: failed to write %s\n",
                         json_out.c_str());
            return 1;
        }
        of << out << '\n';
    } else {
        std::printf("%s\n", out.c_str());
    }
    return 0;
}

// ---------------------------------------------------------------------------
// bench-decode: encode ONE clip once, then time DECODE only -- serial (N
// separate tdt_greedy/rnnt_greedy calls over N copies of the encoder output)
// vs batched (transducer_greedy_batch over the same N copies) -- at several
// batch sizes. Reports decode wall-clock and the batched/serial speedup so the
// GPU win from batched decode can be measured in isolation from the encoder.
//
// The encoder cost is paid once and excluded; only the transducer decode loop
// is timed. Each rep is averaged over R repetitions (best/min recorded). The
// b=0 batched ids are compared against the serial ids (same clip, decode is
// deterministic) as a correctness sanity check.
// ---------------------------------------------------------------------------
static int cmd_bench_decode(int argc, char** argv) {
    std::string model, audio, json_out;
    std::string batch_sizes_str = "1,4,8,16";
    int threads = 0;  // 0 == unset -> use the persistent-backend default
    int reps = 5;
    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model = argv[++i];
        } else if (std::strcmp(argv[i], "--audio") == 0 && i + 1 < argc) {
            audio = argv[++i];
        } else if (std::strcmp(argv[i], "--batch-sizes") == 0 && i + 1 < argc) {
            batch_sizes_str = argv[++i];
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
            reps = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            json_out = argv[++i];
        }
    }
    if (model.empty() || audio.empty()) {
        std::fprintf(stderr,
            "usage: parakeet-cli bench-decode --model <m.gguf> --audio <wav> "
            "[--batch-sizes 1,4,8,16] [--threads N] [--reps R] [--json <out>]\n");
        return 2;
    }
    if (reps < 1) reps = 1;

    // Parse --batch-sizes (comma-separated positive ints).
    std::vector<int> batch_sizes;
    {
        std::stringstream ss(batch_sizes_str);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            size_t b = tok.find_first_not_of(" \t");
            if (b == std::string::npos) continue;
            size_t e = tok.find_last_not_of(" \t");
            int v = std::atoi(tok.substr(b, e - b + 1).c_str());
            if (v > 0) batch_sizes.push_back(v);
        }
    }
    if (batch_sizes.empty()) {
        std::fprintf(stderr,
            "parakeet-cli bench-decode: no valid --batch-sizes (want e.g. 1,4,8,16)\n");
        return 2;
    }

    if (threads > 0) pk::set_num_threads(threads);
    int reported_threads = threads > 0 ? threads : 8;  // kDefaultThreads

    // Load the model components over the lower-level loader (we need the encoder
    // / prediction / joint pieces, not the high-level Model::transcribe path).
    pk::ModelLoader ml;
    if (!ml.load(model)) {
        std::fprintf(stderr, "parakeet-cli bench-decode: failed to load model %s\n",
                     model.c_str());
        return 1;
    }
    pk::ensure_weights_realized(ml);
    pk::Encoder       enc(ml);
    pk::PredictionNet pred(ml);
    pk::Joint         joint(ml);
    const auto& cfg = ml.config();
    const int blank = (int)cfg.blank_id;
    const int maxs  = (int)cfg.max_symbols;
    const std::vector<int32_t> durations = cfg.tdt_durations;

    // Load the WAV.
    pk::Audio a;
    if (!pk::load_audio_16k_mono(audio, a)) {
        std::fprintf(stderr, "parakeet-cli bench-decode: failed to load audio %s\n",
                     audio.c_str());
        return 1;
    }

    // Mel front end (GpuMel on a non-CPU backend, else FFT MelFrontend), exactly
    // as model.cpp's transcribe path does.
    std::vector<float> feats;
    int n_mels = 0, T = 0;
    if (std::string(pk::global_backend().device_name()) != "cpu") {
        pk::GpuMel gmel(ml);
        gmel.compute(a.samples, feats, n_mels, T);
    } else {
        pk::MelFrontend mel(ml);
        mel.compute(a.samples, feats, n_mels, T);
    }

    // Encoder -> enc_out [d_model, Tout] (channels-first); transpose to row-major
    // enc_row [Tout, d_model] as the decoders expect.
    std::vector<float> enc_out;
    int dm = 0, Tout = 0;
    enc.forward(feats, n_mels, T, enc_out, dm, Tout);
    std::vector<float> enc_row((size_t)Tout * dm);
    for (int t = 0; t < Tout; ++t)
        for (int c = 0; c < dm; ++c)
            enc_row[(size_t)t * dm + c] = enc_out[(size_t)c * Tout + t];

    const bool use_tdt = !durations.empty();
    auto decode_serial_one = [&]() -> std::vector<int32_t> {
        return use_tdt
            ? pk::tdt_greedy(pred, joint, enc_row, Tout, dm, durations, blank, maxs, nullptr)
            : pk::rnnt_greedy(pred, joint, enc_row, Tout, dm, blank, maxs, nullptr);
    };

    using clock = std::chrono::steady_clock;
    auto ms_since = [](clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    };

    // Warm up (untimed): realize weights / CUDA kernels for both paths.
    std::vector<int32_t> serial_ref = decode_serial_one();
    {
        std::vector<std::vector<float>> encs1{enc_row};
        std::vector<int> Ts1{Tout};
        std::vector<std::vector<int32_t>> ids1;
        pk::transducer_greedy_batch(pred, joint, encs1, Ts1, dm, durations,
                                    blank, maxs, ids1, nullptr);
    }

    struct Row { int B; double serial_ms; double batched_ms; double speedup;
                 double serial_cps; double batched_cps; };
    std::vector<Row> rows;
    rows.reserve(batch_sizes.size());
    bool sanity_ok = true;

    for (int B : batch_sizes) {
        std::vector<std::vector<float>> encs((size_t)B, enc_row);
        std::vector<int> Ts((size_t)B, Tout);

        // SERIAL: B separate single-clip decodes, best of R reps.
        double serial_ms = 1e300;
        for (int r = 0; r < reps; ++r) {
            auto t0 = clock::now();
            for (int b = 0; b < B; ++b) (void)decode_serial_one();
            serial_ms = std::min(serial_ms, ms_since(t0));
        }

        // BATCHED: one transducer_greedy_batch over the B copies, best of R reps.
        double batched_ms = 1e300;
        std::vector<std::vector<int32_t>> ids_last;
        for (int r = 0; r < reps; ++r) {
            std::vector<std::vector<int32_t>> ids;
            auto t0 = clock::now();
            pk::transducer_greedy_batch(pred, joint, encs, Ts, dm, durations,
                                        blank, maxs, ids, nullptr);
            batched_ms = std::min(batched_ms, ms_since(t0));
            ids_last = std::move(ids);
        }

        // Sanity: batched ids[0] must equal the serial decode of the same clip.
        if (!ids_last.empty() && ids_last[0] != serial_ref) {
            sanity_ok = false;
            std::fprintf(stderr,
                "parakeet-cli bench-decode: WARN B=%d batched ids[0] != serial "
                "(%zu vs %zu tokens) -- batched decode may be buggy\n",
                B, ids_last[0].size(), serial_ref.size());
        }

        double speedup     = batched_ms > 0.0 ? serial_ms / batched_ms : 0.0;
        double serial_cps  = serial_ms  > 0.0 ? (double)B / (serial_ms  / 1000.0) : 0.0;
        double batched_cps = batched_ms > 0.0 ? (double)B / (batched_ms / 1000.0) : 0.0;
        rows.push_back({B, serial_ms, batched_ms, speedup, serial_cps, batched_cps});
    }

    // Human-readable table to stderr.
    std::fprintf(stderr,
        "\nbench-decode: clip Tout=%d frames, d_model=%d, decoder=%s, threads=%d, "
        "reps=%d (best-of), backend=%s\n",
        Tout, dm, use_tdt ? "tdt" : "rnnt", reported_threads, reps,
        pk::global_backend().device_name());
    std::fprintf(stderr, "  %-6s %-12s %-12s %-10s %-14s %-14s\n",
                 "B", "serial_ms", "batched_ms", "speedup", "serial_cps", "batched_cps");
    for (const Row& r : rows) {
        std::fprintf(stderr, "  %-6d %-12.2f %-12.2f %-10.2f %-14.1f %-14.1f\n",
                     r.B, r.serial_ms, r.batched_ms, r.speedup,
                     r.serial_cps, r.batched_cps);
    }
    std::fprintf(stderr, "  sanity (batched ids[0]==serial): %s\n",
                 sanity_ok ? "OK" : "MISMATCH (see WARN above)");

    // Optional machine-readable JSON document (hand-rolled, same style as
    // cmd_bench). Written ONLY when --json <out> is given; the human table above
    // always prints regardless.
    if (!json_out.empty()) {
        // basename of the model gguf path.
        std::string model_base = model;
        size_t slash = model_base.find_last_of("/\\");
        if (slash != std::string::npos) model_base = model_base.substr(slash + 1);

        std::string out;
        out.reserve(512 + rows.size() * 96);
        out += "{\"model\":";
        bench_json_string(out, model_base);
        out += ",\"decoder\":";
        bench_json_string(out, use_tdt ? std::string("tdt") : std::string("ctc"));
        out += ",\"backend\":";
        bench_json_string(out, std::string(pk::global_backend().device_name()));
        char nb[64];
        std::snprintf(nb, sizeof(nb), ",\"threads\":%d", reported_threads);
        out += nb;
        std::snprintf(nb, sizeof(nb), ",\"reps\":%d", reps);
        out += nb;
        std::snprintf(nb, sizeof(nb), ",\"clip_frames\":%d", Tout);
        out += nb;
        std::snprintf(nb, sizeof(nb), ",\"d_model\":%d", dm);
        out += nb;
        out += ",\"batch_sizes\":[";
        for (size_t i = 0; i < batch_sizes.size(); ++i) {
            if (i) out += ',';
            std::snprintf(nb, sizeof(nb), "%d", batch_sizes[i]);
            out += nb;
        }
        out += "],\"rows\":[";
        for (size_t i = 0; i < rows.size(); ++i) {
            if (i) out += ',';
            const Row& r = rows[i];
            std::snprintf(nb, sizeof(nb), "{\"B\":%d", r.B);
            out += nb;
            std::snprintf(nb, sizeof(nb), ",\"serial_ms\":%.2f", r.serial_ms);
            out += nb;
            std::snprintf(nb, sizeof(nb), ",\"batched_ms\":%.2f", r.batched_ms);
            out += nb;
            std::snprintf(nb, sizeof(nb), ",\"speedup\":%.2f", r.speedup);
            out += nb;
            std::snprintf(nb, sizeof(nb), ",\"serial_cps\":%.1f", r.serial_cps);
            out += nb;
            std::snprintf(nb, sizeof(nb), ",\"batched_cps\":%.1f", r.batched_cps);
            out += nb;
            out += '}';
        }
        out += "]}";

        std::ofstream of(json_out, std::ios::binary | std::ios::trunc);
        if (!of) {
            std::fprintf(stderr, "parakeet-cli bench-decode: failed to write %s\n",
                         json_out.c_str());
            return 1;
        }
        of << out << '\n';
    }
    return 0;
}

// Reads a whole file. Returns 0 on success, ENOENT when the path does not
// exist, EISDIR when it is a directory, else the errno of the failure.
static int read_file_bytes(const std::string& path, std::string& out) {
    std::error_code ec;
    if (std::filesystem::is_directory(path, ec)) return EISDIR;
    errno = 0;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return errno ? errno : EIO;
    out.clear();
    char buf[4096];
    size_t k;
    while ((k = std::fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, k);
    const int e = std::ferror(f) ? (errno ? errno : EIO) : 0;
    std::fclose(f);
    return e;
}

// One line for a registry file that cannot be read.
static std::string registry_read_error(const std::string& path, int e) {
    if (e == EISDIR) return path + " is a directory, not a speaker registry";
    return "cannot read registry " + path + ": " + std::strerror(e);
}

static const char* kEnrollUsage =
    "usage: parakeet-cli enroll --model <speaker.gguf> --name <name> "
    "--input <wav> [--input <wav> ...] --registry <file>\n";

// parakeet-cli enroll --model <speaker.gguf> --name <name> --input <wav> [--input <wav> ...]
//                     --registry <file>
// Embeds each input as one clip of <name> and adds it to the registry file
// (created when missing). The file is written only after every clip embedded.
static int cmd_enroll(int argc, char** argv) {
    std::string model, name, registry_path;
    std::vector<std::string> inputs;
    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) model = argv[++i];
        else if (std::strcmp(argv[i], "--name") == 0 && i + 1 < argc) name = argv[++i];
        else if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) inputs.push_back(argv[++i]);
        else if (std::strcmp(argv[i], "--registry") == 0 && i + 1 < argc) registry_path = argv[++i];
        else { std::fprintf(stderr, "%s", kEnrollUsage); return 2; }
    }
    if (model.empty() || name.empty() || inputs.empty() || registry_path.empty()) {
        std::fprintf(stderr, "%s", kEnrollUsage);
        return 2;
    }
    if (!pk::SpeakerEncoder::available()) {
        std::fprintf(stderr, "parakeet-cli: built without speaker identification (PARAKEET_WITH_VOICEDETECT=OFF)\n");
        return 2;
    }
    auto enc = pk::SpeakerEncoder::load(model);
    if (!enc) {
        std::fprintf(stderr, "parakeet-cli enroll: failed to load speaker model %s\n", model.c_str());
        return 1;
    }
    pk::SpeakerRegistry reg;
    std::string blob;
    const int rerr = read_file_bytes(registry_path, blob);
    if (rerr != 0 && rerr != ENOENT) {   // only a missing file means "start a new registry"
        std::fprintf(stderr, "parakeet-cli enroll: %s\n", registry_read_error(registry_path, rerr).c_str());
        return 1;
    }
    if (rerr == 0) {   // add to an existing registry
        try { reg = pk::SpeakerRegistry::deserialize(blob); }
        catch (const std::exception& e) {
            std::fprintf(stderr, "parakeet-cli enroll: %s is not a speaker registry: %s\n",
                         registry_path.c_str(), e.what());
            return 1;
        }
    }
    int clips = 0;
    for (const std::string& in : inputs) {
        pk::Audio audio;
        if (!load_audio_arg_16k_mono(in, audio)) {
            std::fprintf(stderr, "parakeet-cli enroll: failed to load audio %s\n",
                         input_display_name(in).c_str());
            return 1;
        }
        std::vector<float> emb;
        if (!enc->embed(audio.samples.data(), (int)audio.samples.size(), emb)) {
            std::fprintf(stderr, "parakeet-cli enroll: %s: %s\n", input_display_name(in).c_str(),
                         enc->last_error().c_str());
            return 1;
        }
        try { reg.enroll(name, emb); }
        catch (const std::exception& e) {
            std::fprintf(stderr, "parakeet-cli enroll: %s\n", e.what());
            return 1;
        }
        ++clips;
    }
    // Written next to the target and moved over it, so a failed write never
    // costs the user the registry they already had.
    std::string werr;
    if (!pk::write_file_atomic(registry_path, reg.serialize(), &werr)) {
        std::fprintf(stderr, "parakeet-cli enroll: %s\n", werr.c_str());
        return 1;
    }
    std::printf("enrolled %s (%d clip(s)), registry has %zu speaker(s)\n", name.c_str(), clips,
                reg.size());
    return 0;
}

static const char* kSceneUsage =
    "usage: parakeet-cli scene [--model <m.gguf>] [--diar <diar.gguf>] "
    "[--sound <ced.gguf>] [--speakers <speaker.gguf> --registry <file> "
    "[--speaker-threshold F]] --input <wav|-> "
    "[--latency model|low|very_low|ultra_low] [--chunk-ms N] "
    "[--show-speech] [--json]\n"
    "  --speaker-threshold: default 0.5; ECAPA needs about 0.7, see docs/speaker.md\n";

// parakeet-cli scene [--model <m.gguf>] [--diar <diar.gguf>] [--sound <ced.gguf>]
//                    [--speakers <speaker.gguf> --registry <file> [--speaker-threshold F]]
//                    --input <wav|-> [--latency model|low|very_low|ultra_low]
//                    [--chunk-ms N] [--show-speech] [--json]
// --speakers names diarized speakers from the enrolled voices in --registry
// (made by `parakeet-cli enroll`); it needs --diar and --registry.
// Streams the WAV through pk::SceneStream (ASR + diarization + sound events,
// each optional -- at least one is required) and prints a time-ordered
// transcript with sound annotations. --json prints scene_update_to_json per
// update (one JSON document per line) instead of the rendered transcript.
static int cmd_scene(int argc, char** argv) {
    std::string model, diar, sound, input, latency_str;
    std::string speakers, registry_path;
    bool have_threshold = false;
    float speaker_threshold = 0.0f;
    bool json = false;
    bool show_speech = false;
    int chunk_ms = 200;
    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model = argv[++i];
        } else if (std::strcmp(argv[i], "--diar") == 0 && i + 1 < argc) {
            diar = argv[++i];
        } else if (std::strcmp(argv[i], "--sound") == 0 && i + 1 < argc) {
            sound = argv[++i];
        } else if (std::strcmp(argv[i], "--speakers") == 0 && i + 1 < argc) {
            speakers = argv[++i];
        } else if (std::strcmp(argv[i], "--registry") == 0 && i + 1 < argc) {
            registry_path = argv[++i];
        } else if (std::strcmp(argv[i], "--speaker-threshold") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* txt = argv[++i];
            speaker_threshold = std::strtof(txt, &end);
            if (end == txt || *end != '\0') {
                std::fprintf(stderr, "parakeet-cli scene: --speaker-threshold needs a number, got '%s'\n", txt);
                return 2;
            }
            have_threshold = true;
        } else if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input = argv[++i];
        } else if (std::strcmp(argv[i], "--latency") == 0 && i + 1 < argc) {
            latency_str = argv[++i];
        } else if (std::strcmp(argv[i], "--chunk-ms") == 0 && i + 1 < argc) {
            chunk_ms = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--json") == 0) {
            json = true;
        } else if (std::strcmp(argv[i], "--show-speech") == 0) {
            show_speech = true;
        } else {
            std::fprintf(stderr, "%s", kSceneUsage);
            return 2;
        }
    }

    if ((model.empty() && diar.empty() && sound.empty()) || input.empty()) {
        std::fprintf(stderr, "%s", kSceneUsage);
        return 2;
    }
    if (chunk_ms <= 0) {
        std::fprintf(stderr, "parakeet-cli scene: --chunk-ms must be > 0\n");
        return 2;
    }
    // chunk_ms * 16 (samples/ms at 16 kHz) must not overflow int; 60 s is far
    // above any sane chunk size (the default is 200 ms) and leaves headroom.
    if (chunk_ms > 60000) {
        std::fprintf(stderr, "parakeet-cli scene: --chunk-ms must be <= 60000\n");
        return 2;
    }
    pk::DiarLatency latency = pk::DiarLatency::Model;
    if (!latency_str.empty()) {
        if (latency_str == "model") {
            latency = pk::DiarLatency::Model;
        } else if (latency_str == "low") {
            latency = pk::DiarLatency::Low;
        } else if (latency_str == "very_low") {
            latency = pk::DiarLatency::VeryLow;
        } else if (latency_str == "ultra_low") {
            latency = pk::DiarLatency::UltraLow;
        } else {
            std::fprintf(stderr,
                "parakeet-cli scene: unknown --latency '%s' (want model|low|very_low|ultra_low)\n",
                latency_str.c_str());
            return 2;
        }
    }
    if (!sound.empty() && !pk::CedTagger::available()) {
        std::fprintf(stderr, "parakeet-cli: built without sound tagging (PARAKEET_WITH_CED=OFF)\n");
        return 2;
    }
    if (speakers.empty() && (!registry_path.empty() || have_threshold)) {
        std::fprintf(stderr, "parakeet-cli scene: --registry and --speaker-threshold need --speakers\n");
        return 2;
    }
    pk::SpeakerIdOpts speaker_opts;
    if (!speakers.empty()) {
        if (diar.empty()) {
            std::fprintf(stderr, "parakeet-cli scene: --speakers needs --diar\n");
            return 2;
        }
        if (registry_path.empty()) {
            std::fprintf(stderr, "parakeet-cli scene: --speakers needs --registry\n");
            return 2;
        }
        if (!pk::SpeakerEncoder::available()) {
            std::fprintf(stderr, "parakeet-cli: built without speaker identification (PARAKEET_WITH_VOICEDETECT=OFF)\n");
            return 2;
        }
        if (have_threshold) speaker_opts.accept_threshold = speaker_threshold;
        const std::string bad = pk::validate_speaker_opts(speaker_opts);
        if (!bad.empty()) {
            std::fprintf(stderr, "parakeet-cli scene: invalid speaker options: %s\n", bad.c_str());
            return 2;
        }
    }

    std::unique_ptr<pk::Model> asr_model;
    if (!model.empty()) {
        asr_model = pk::Model::load(model);
        if (!asr_model) {
            std::fprintf(stderr, "parakeet-cli scene: failed to load model %s\n", model.c_str());
            return 1;
        }
    }
    std::unique_ptr<pk::DiarizationModel> diar_model;
    if (!diar.empty()) {
        diar_model = pk::DiarizationModel::load(diar);
        if (!diar_model) {
            std::fprintf(stderr, "parakeet-cli scene: failed to load diarization model %s\n",
                         diar.c_str());
            return 1;
        }
    }
    std::unique_ptr<pk::CedTagger> tagger;
    if (!sound.empty()) {
        tagger = pk::CedTagger::load(sound);
        if (!tagger) {
            std::fprintf(stderr, "parakeet-cli scene: failed to load sound model %s\n", sound.c_str());
            return 1;
        }
    }

    std::unique_ptr<pk::SpeakerEncoder> speaker_enc;
    pk::SpeakerRegistry registry;
    if (!speakers.empty()) {
        speaker_enc = pk::SpeakerEncoder::load(speakers);
        if (!speaker_enc) {
            std::fprintf(stderr, "parakeet-cli scene: failed to load speaker model %s\n",
                         speakers.c_str());
            return 1;
        }
        std::string blob;
        const int rerr = read_file_bytes(registry_path, blob);
        if (rerr != 0) {
            std::fprintf(stderr, "parakeet-cli scene: %s\n", registry_read_error(registry_path, rerr).c_str());
            return 1;
        }
        try { registry = pk::SpeakerRegistry::deserialize(blob); }
        catch (const std::exception& e) {
            std::fprintf(stderr, "parakeet-cli scene: %s is not a speaker registry: %s\n",
                         registry_path.c_str(), e.what());
            return 1;
        }
        if (registry.dim() != speaker_enc->dim()) {
            std::fprintf(stderr,
                "parakeet-cli scene: registry %s holds %d-dim voices but %s makes %d-dim embeddings "
                "(enroll again with this model)\n",
                registry_path.c_str(), registry.dim(), speakers.c_str(), speaker_enc->dim());
            return 1;
        }
    }

    pk::Audio audio;
    if (!load_audio_arg_16k_mono(input, audio)) {
        std::string display = input_display_name(input);
        std::fprintf(stderr, "parakeet-cli scene: failed to load audio %s\n", display.c_str());
        return 1;
    }

    pk::SceneParts parts;
    parts.asr = asr_model.get();
    parts.diar = diar_model.get();
    parts.diar_latency = latency;
    parts.tagger = tagger.get();
    if (speaker_enc) {
        parts.speaker_embed = speaker_enc->embedder();
        parts.registry = &registry;
        parts.speaker_opts = speaker_opts;
    }

    // scene_update_to_json's label(i) may return nullptr (emitted as ""); the
    // same lambda drives the renderer's --json-less line formatting.
    auto label = [&](int i) -> const char* { return tagger ? tagger->label(i) : nullptr; };

    std::unique_ptr<pk::SceneStream> stream;
    try {
        stream.reset(new pk::SceneStream(parts));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "parakeet-cli scene: %s\n", e.what());
        return 1;
    }

    pk::SceneRenderer renderer(diar_model != nullptr, show_speech, label, asr_model != nullptr);

    const int chunk_samples = chunk_ms * 16;  // 16 samples/ms at 16 kHz
    const int n = (int)audio.samples.size();
    for (int lo = 0; lo < n || lo == 0; lo += chunk_samples) {
        const int len = std::min(chunk_samples, n - lo);
        const bool is_last = lo + len >= n;
        pk::SceneUpdate u;
        try {
            u = stream->feed(audio.samples.data() + lo, len, is_last);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "parakeet-cli scene: feed failed: %s\n", e.what());
            return 1;
        }
        if (json) {
            std::printf("%s\n", pk::scene_update_to_json(u, label).c_str());
        } else {
            renderer.add(u);
            for (const std::string& line : renderer.flush(u.safe_until))
                std::printf("%s\n", line.c_str());
            if (is_last)
                for (const std::string& line : renderer.flush_all())
                    std::printf("%s\n", line.c_str());
        }
        if (is_last) break;
    }
    return 0;
}

// Run a subcommand, then free the process-global backend while the GPU driver is
// still alive (the subcommand's local Model is already destroyed by the time it
// returns, releasing its device weight buffer). Avoids the CUDA "driver shutting
// down" abort caused by static-destruction teardown ordering.
static int run_and_shutdown(int (*fn)(int, char**), int argc, char** argv) {
    int rc = fn(argc, argv);
    pk::shutdown_backend();
    return rc;
}

// parakeet-cli vad-probe --model <m.gguf> --input <wav|-> [--variant N]
// Prints "t_sec,p" for every 80 ms frame. For inspecting the VAD head.
// --variant N picks a debug wiring (0 is the default: SiLU, SiLU, no residual;
// bit 0 ReLU after proj, bit 1 residual, bit 2 ReLU after ctx).
static int cmd_vad_probe(int argc, char** argv) {
    std::string model, input;
    int variant = -1;
    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) model = argv[++i];
        else if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) input = argv[++i];
        else if (std::strcmp(argv[i], "--variant") == 0 && i + 1 < argc) variant = std::atoi(argv[++i]);
    }
    if (model.empty() || input.empty()) {
        std::fprintf(stderr, "usage: parakeet-cli vad-probe --model <m.gguf> --input <wav|-> [--variant N]\n");
        return 2;
    }
    pk::Audio audio;
    if (!load_audio_arg_16k_mono(input, audio)) {
        std::fprintf(stderr, "parakeet-cli: failed to load audio %s\n", input.c_str());
        return 1;
    }
    try {
        std::unique_ptr<pk::Model> m = pk::Model::load(model);
        if (!m) { std::fprintf(stderr, "parakeet-cli: failed to load model %s\n", model.c_str()); return 1; }
        const pk::VadVariant v = pk::VadVariant::from_index(variant < 0 ? 0 : variant);
        const std::vector<float> p = m->vad_probabilities(audio.samples, variant < 0 ? nullptr : &v);
        const float fs = m->config().vad.frame_sec;
        for (size_t i = 0; i < p.size(); ++i) std::printf("%.2f,%.4f\n", (double)i * fs, p[i]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "parakeet-cli: vad-probe failed: %s\n", e.what());
        return 1;
    }
    return 0;
}

// parakeet-cli vad --model <m.gguf> --input <wav|-> [--threshold F=0.5]
//   [--min-pause SEC] [--min-speech SEC] [--speech-pad SEC] [--max-segment SEC=30]
//   [--mode speech|segments] [--probabilities] [--threads N]
// Prints the same JSON as parakeet_capi_vad_path_json. The model is an ASR GGUF
// with a VAD head (Ultra, Redux) or a Silero VAD GGUF. Unset options keep the
// defaults of that model kind.
static int cmd_vad(int argc, char** argv) {
    std::string model, input, component;
    VadOverrides ov;
    std::optional<pk::VadRequest::Mode> mode;
    bool want_probs = false;
    int threads = 0;
    auto bad = [](const char* what) {
        std::fprintf(stderr, "parakeet-cli: %s\n", what);
        return 2;
    };
    auto num = [](const char* s, double& out, bool allow_zero) {
        char* end = nullptr;
        out = std::strtod(s, &end);
        return end != s && *end == '\0' && std::isfinite(out) && (allow_zero ? out >= 0.0 : out > 0.0);
    };
    for (int i = 0; i < argc; ++i) {
        double d = 0.0;
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) model = argv[++i];
        else if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) input = argv[++i];
        else if (std::strcmp(argv[i], "--component") == 0 && i + 1 < argc) component = argv[++i];
        else if (std::strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) {
            if (!num(argv[++i], d, false) || d > 1.0) return bad("--threshold must be in (0,1]");
            ov.threshold = d;
        } else if (std::strcmp(argv[i], "--min-pause") == 0 && i + 1 < argc) {
            if (!num(argv[++i], d, false)) return bad("--min-pause must be > 0");
            ov.min_pause = d;
        } else if (std::strcmp(argv[i], "--min-speech") == 0 && i + 1 < argc) {
            if (!num(argv[++i], d, false)) return bad("--min-speech must be > 0");
            ov.min_speech = d;
        } else if (std::strcmp(argv[i], "--speech-pad") == 0 && i + 1 < argc) {
            if (!num(argv[++i], d, true)) return bad("--speech-pad must be >= 0");
            ov.pad = d;
        } else if (std::strcmp(argv[i], "--max-segment") == 0 && i + 1 < argc) {
            if (!num(argv[++i], d, false)) return bad("--max-segment must be > 0");
            ov.max_seg = d;
        } else if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            const char* v = argv[++i];
            if (std::strcmp(v, "speech") == 0) mode = pk::VadRequest::Mode::kSpeech;
            else if (std::strcmp(v, "segments") == 0) mode = pk::VadRequest::Mode::kSegments;
            else return bad("--mode must be speech or segments");
        } else if (std::strcmp(argv[i], "--probabilities") == 0) want_probs = true;
        else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) threads = std::atoi(argv[++i]);
        else return bad("unknown or incomplete option for vad");
    }
    if (model.empty() || input.empty()) {
        std::fprintf(stderr,
            "usage: parakeet-cli vad --model <asr-with-vad-head.gguf|silero.gguf|bundle.gguf> --input <wav|-> "
            "[--component NAME] [--threshold F=0.5] [--min-pause SEC] [--min-speech SEC] [--speech-pad SEC] "
            "[--max-segment SEC=30] [--mode speech|segments] [--probabilities] [--threads N]\n");
        return 2;
    }
    if (threads > 0) pk::set_num_threads(threads);
    pk::Audio audio;
    if (!load_audio_arg_16k_mono(input, audio)) {
        std::fprintf(stderr, "parakeet-cli: failed to load audio %s\n", input.c_str());
        return 1;
    }
    try {
        // A bundle: --component names the VAD source (a Silero component, or the
        // ASR component whose head is used). Without it, the Silero component
        // if there is one, else the default ASR component.
        std::string comp;
        bool is_silero = pk::gguf_is_silero(model);
        if (pk::gguf_is_bundle(model)) {
            pk::BundleInfo info;
            std::string berr;
            if (!pk::read_bundle_info(model, info, &berr)) { std::fprintf(stderr, "parakeet-cli: %s\n", berr.c_str()); return 1; }
            const pk::BundleComponent* pick = component.empty() ? nullptr : info.find(component);
            if (!component.empty() && !pick) {
                std::fprintf(stderr, "parakeet-cli: bundle has no component '%s'; components: %s\n", component.c_str(), pk::bundle_component_names(info).c_str());
                return 1;
            }
            if (component.empty())
                for (const pk::BundleComponent& c : info.components)
                    if (c.kind == pk::kBundleKindVad) { pick = &c; break; }
            if (!pick) {
                if (!pk::select_default_component(info, comp, &berr)) { std::fprintf(stderr, "parakeet-cli: %s\n", berr.c_str()); return 1; }
            } else {
                comp = pick->name;
            }
            const pk::BundleComponent* chosen = info.find(comp);
            is_silero = chosen && chosen->kind == pk::kBundleKindVad;
            if (chosen && chosen->kind != pk::kBundleKindVad && chosen->kind != pk::kBundleKindAsr) {
                std::fprintf(stderr, "parakeet-cli: component '%s' (kind %s) is not a VAD source\n", comp.c_str(), chosen->kind.c_str());
                return 1;
            }
        } else if (!component.empty()) {
            std::fprintf(stderr, "parakeet-cli: %s is not a bundle GGUF, so --component does not apply\n", model.c_str());
            return 2;
        }
        pk::VadRequest req;
        req.kind = is_silero ? pk::VadKind::kSilero : pk::VadKind::kHead;
        req.opts = pk::default_segmenter_opts(req.kind);
        ov.apply(req.opts);
        if (mode) req.mode = *mode;
        req.probabilities = want_probs;
        if (is_silero) {
            std::string err;
            std::unique_ptr<pk::SileroVad> sv = pk::SileroVad::load(model, &err, comp);
            if (!sv) { std::fprintf(stderr, "parakeet-cli: failed to load model %s: %s\n", model.c_str(), err.c_str()); return 1; }
            std::printf("%s\n", pk::silero_vad_to_json(*sv, audio.samples, 16000, req).c_str());
            return 0;
        }
        std::unique_ptr<pk::Model> m = pk::Model::load(model, comp);
        if (!m) { std::fprintf(stderr, "parakeet-cli: failed to load model %s\n", model.c_str()); return 1; }
        std::printf("%s\n", pk::vad_to_json(*m, audio.samples, req).c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "parakeet-cli: vad failed: %s\n", e.what());
        return 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 2 && (std::strcmp(argv[1], "--version") == 0 ||
                      std::strcmp(argv[1], "-V") == 0)) {
        std::printf("parakeet-cli %s\n", parakeet_version());
        return 0;
    }
    if (argc >= 3 && std::strcmp(argv[1], "info") == 0)
        return run_and_shutdown(cmd_info, argc - 2, argv + 2);
    if (argc >= 2 && std::strcmp(argv[1], "transcribe") == 0)
        return run_and_shutdown(cmd_transcribe, argc - 2, argv + 2);
    if (argc >= 2 && std::strcmp(argv[1], "quantize") == 0)
        return run_and_shutdown(cmd_quantize, argc - 2, argv + 2);
    if (argc >= 2 && std::strcmp(argv[1], "bench-batch") == 0)
        return run_and_shutdown(cmd_bench_batch, argc - 2, argv + 2);
    if (argc >= 2 && std::strcmp(argv[1], "bench-decode") == 0)
        return run_and_shutdown(cmd_bench_decode, argc - 2, argv + 2);
    if (argc >= 2 && std::strcmp(argv[1], "bench") == 0)
        return run_and_shutdown(cmd_bench, argc - 2, argv + 2);
    if (argc >= 2 && std::strcmp(argv[1], "enroll") == 0)
        return run_and_shutdown(cmd_enroll, argc - 2, argv + 2);
    if (argc >= 2 && std::strcmp(argv[1], "scene") == 0)
        return run_and_shutdown(cmd_scene, argc - 2, argv + 2);
    if (argc >= 2 && std::strcmp(argv[1], "vad") == 0)
        return run_and_shutdown(cmd_vad, argc - 2, argv + 2);
    if (argc >= 2 && std::strcmp(argv[1], "vad-probe") == 0)
        return run_and_shutdown(cmd_vad_probe, argc - 2, argv + 2);
    std::fprintf(stderr,
        "usage:\n"
        "  parakeet-cli vad --model <asr-with-vad-head.gguf|silero.gguf> --input <wav|-> "
        "[--threshold F=0.5] [--min-pause SEC] [--min-speech SEC] [--speech-pad SEC] "
        "[--max-segment SEC=30] [--mode speech|segments] [--probabilities] [--threads N]\n"
        "  parakeet-cli vad-probe --model <m.gguf> --input <wav|-> [--variant N]\n"
        "  parakeet-cli info <model.gguf> [--component NAME]\n"
        "  parakeet-cli transcribe --model <model.gguf> --input <wav|-> "
        "[--decoder ctc|tdt] [--lang <locale>] [--stream] [--timestamps] "
        "[--threads N] [--json] "
        "[--component NAME] "
            "[--vad [--vad-model <silero.gguf>] [--vad-component NAME] [--vad-threshold F=0.5] [--vad-min-pause SEC] "
            "[--vad-min-speech SEC] [--vad-max-seg SEC=30]] "
        "[--beam-size N [--nbest N] [--no-score-norm]]\n"
        "  parakeet-cli quantize <in.gguf> <out.gguf> "
        "<q4_0|q5_0|q8_0|q4_k|q5_k|q6_k>\n"
        "  parakeet-cli bench --model <model.gguf> --manifest <file> "
        "[--decoder ctc|tdt] [--lang <locale>] [--threads N] [--concurrency K] [--json <out>]\n"
        "  parakeet-cli bench-batch --model <model.gguf> --manifest <file> "
        "[--decoder ctc|tdt] [--threads N] [--batch-sizes 1,4,8] [--json <out>]\n"
        "  parakeet-cli bench-decode --model <model.gguf> --audio <wav> "
        "[--batch-sizes 1,4,8,16] [--threads N] [--reps R] [--json <out>]\n"
        "  parakeet-cli scene [--model <m.gguf>] [--diar <diar.gguf>] "
        "[--sound <ced.gguf>] [--speakers <speaker.gguf> --registry <file> "
        "[--speaker-threshold F]] --input <wav|-> "
        "[--latency model|low|very_low|ultra_low] [--chunk-ms N] "
        "[--show-speech] [--json]\n"
        "      --speaker-threshold: default 0.5; ECAPA needs about 0.7, see docs/speaker.md\n"
        "  parakeet-cli enroll --model <speaker.gguf> --name <name> "
        "--input <wav> [--input <wav> ...] --registry <file>\n");
    return 2;
}
