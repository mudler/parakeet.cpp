#include "parakeet.h"
#include "parakeet_capi.h"
#include "model.hpp"
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
#include <chrono>
#include <fstream>
#include <memory>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

static int cmd_info(const char* path) {
    pk::ModelLoader ml;
    if (!ml.load(path)) { std::fprintf(stderr, "failed to load %s\n", path); return 1; }
    const pk::ParakeetConfig& c = ml.config();
    std::printf("parakeet.cpp %s\n", parakeet_version());
    std::printf("model: %s\n", path);
    std::printf("  arch            : %s\n", c.arch.c_str());
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
    if (!pk::load_audio_16k_mono(input, audio)) {
        std::fprintf(stderr, "parakeet-cli: failed to load audio %s\n", input.c_str());
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

// parakeet-cli transcribe --model <m.gguf> --input <wav> [--decoder ctc|tdt]
//                         [--stream]
// Prints the transcript. Default decoder is chosen by arch (TDT for transducer
// archs, CTC for ctc arch — matching NeMo's cur_decoder default). --stream uses
// the cache-aware streaming path (EOU streaming model only).
static int cmd_transcribe(int argc, char** argv) {
    std::string model, input, decoder_str, lang;
    bool stream = false;
    bool timestamps = false;
    bool json = false;
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
        }
    }
    if (model.empty() || input.empty()) {
        std::fprintf(stderr,
            "usage: parakeet-cli transcribe --model <m.gguf> --input <wav> "
            "[--decoder ctc|tdt] [--lang <locale>] [--stream] [--timestamps] "
            "[--threads N] [--json]\n");
        return 2;
    }
    // Apply the thread override (offline + streaming graph compute). When unset
    // the persistent-backend default (kDefaultThreads) is used.
    if (threads > 0) pk::set_num_threads(threads);

    if (stream) {
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

    // --json: emit the C-API JSON document (text + word/token timestamps + conf).
    if (json) {
        parakeet_ctx* ctx = parakeet_capi_load(model.c_str());
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
        try {
            std::unique_ptr<pk::Model> m = pk::Model::load(model);
            if (!m) {
                std::fprintf(stderr, "parakeet-cli: failed to load model %s\n", model.c_str());
                return 1;
            }
            // `lang` (empty -> model default) selects the language prompt for
            // multilingual models; ignored by non-prompt models.
            pk::Transcription tr = m->transcribe_path_with_timestamps(input, dec, lang);
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
    if (!lang.empty()) {
        parakeet_ctx* ctx = parakeet_capi_load(model.c_str());
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
        text = pk::transcribe(model, input, dec);
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
// Mirrors append_json_string in src/parakeet_capi.cpp (UTF-8 >= 0x80 passes
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
        } else if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            json_out = argv[++i];
        }
    }
    if (model.empty() || manifest.empty()) {
        std::fprintf(stderr,
            "usage: parakeet-cli bench --model <m.gguf> --manifest <file> "
            "[--decoder ctc|tdt] [--lang <locale>] [--threads N] [--json <out>]\n");
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
    int reported_threads = threads;
    if (threads > 0) {
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

    struct FileResult { std::string path; double audio_sec; double proc_ms; std::string text; };
    std::vector<FileResult> results;
    results.reserve(paths.size());

    // Warm up once (untimed): the first transcribe pays one-time costs — lazy
    // device weight upload + CUDA kernel/cuBLAS init on a GPU backend (~100x the
    // steady-state per-file time), or weight realization on CPU. Excluding it
    // keeps per-file proc_ms (and RTFx) steady-state and fair vs other engines.
    {
        pk::Audio warm;
        if (pk::load_audio_16k_mono(paths[0], warm)) {
            (void)m->transcribe_pcm(warm.samples, 16000, dec, lang);
        }
    }

    for (const std::string& p : paths) {
        pk::Audio audio;
        if (!pk::load_audio_16k_mono(p, audio)) {
            std::fprintf(stderr, "parakeet-cli bench: failed to load audio %s\n",
                         p.c_str());
            return 1;
        }
        // audio_sec from the decoded 16 kHz sample count.
        double audio_sec = (double)audio.samples.size() / 16000.0;

        // Time ONLY the transcription (model already loaded).
        auto t_proc = clock::now();
        std::string text;
        try {
            text = m->transcribe_pcm(audio.samples, 16000, dec, lang);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "parakeet-cli bench: transcribe failed on %s: %s\n",
                         p.c_str(), e.what());
            return 1;
        }
        double proc_ms = ms_since(t_proc);
        results.push_back({p, audio_sec, proc_ms, text});
    }

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

// Run a subcommand, then free the process-global backend while the GPU driver is
// still alive (the subcommand's local Model is already destroyed by the time it
// returns, releasing its device weight buffer). Avoids the CUDA "driver shutting
// down" abort caused by static-destruction teardown ordering.
static int run_and_shutdown(int (*fn)(int, char**), int argc, char** argv) {
    int rc = fn(argc, argv);
    pk::shutdown_backend();
    return rc;
}

int main(int argc, char** argv) {
    if (argc == 2 && (std::strcmp(argv[1], "--version") == 0 ||
                      std::strcmp(argv[1], "-V") == 0)) {
        std::printf("parakeet-cli %s\n", parakeet_version());
        return 0;
    }
    if (argc >= 3 && std::strcmp(argv[1], "info") == 0)
        return run_and_shutdown([](int, char** a) { return cmd_info(a[0]); }, 1, argv + 2);
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
    std::fprintf(stderr,
        "usage:\n"
        "  parakeet-cli info <model.gguf>\n"
        "  parakeet-cli transcribe --model <model.gguf> --input <wav> "
        "[--decoder ctc|tdt] [--lang <locale>] [--stream] [--timestamps] "
        "[--threads N] [--json]\n"
        "  parakeet-cli quantize <in.gguf> <out.gguf> "
        "<q4_0|q5_0|q8_0|q4_k|q5_k|q6_k>\n"
        "  parakeet-cli bench --model <model.gguf> --manifest <file> "
        "[--decoder ctc|tdt] [--lang <locale>] [--threads N] [--json <out>]\n"
        "  parakeet-cli bench-batch --model <model.gguf> --manifest <file> "
        "[--decoder ctc|tdt] [--threads N] [--batch-sizes 1,4,8] [--json <out>]\n"
        "  parakeet-cli bench-decode --model <model.gguf> --audio <wav> "
        "[--batch-sizes 1,4,8,16] [--threads N] [--reps R] [--json <out>]\n");
    return 2;
}
