#include "model_loader.hpp"
#include "common.hpp"
#include "bundle.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <utility>
#include <stdexcept>
namespace pk {

int PromptCfg::resolve_index_or_throw(const std::string& target_lang) const {
    const std::string lang = target_lang.empty() ? default_lang : target_lang;
    int idx = lang_to_index(lang);
    if (idx < 0) {
        std::string sample;
        for (size_t i = 0; i < dict_keys.size() && i < 8; ++i)
            sample += (i ? ", " : "") + dict_keys[i];
        throw std::runtime_error("parakeet: unknown target_lang '" + lang +
                                 "'. Valid examples: " + sample + ", ...");
    }
    return idx;
}

// Key lookup through an optional component prefix. A bundle GGUF stores the
// keys of component "asr" as "asr.<original key>"; a plain file has prefix "".
struct KvView {
    gguf_context* g;
    std::string prefix;
    int64_t find(const char* k) const { return gguf_find_key(g, (prefix + k).c_str()); }
};
static uint32_t kv_u32(const KvView& v, const char* k, uint32_t d=0){ gguf_context* g=v.g;
    int64_t id = v.find(k); return id<0 ? d : (uint32_t)gguf_get_val_u32(g,id);
}
static int32_t kv_i32(const KvView& v, const char* k, int32_t d=0){ gguf_context* g=v.g;
    int64_t id = v.find(k); return id<0 ? d : gguf_get_val_i32(g,id);
}
static std::vector<int32_t> kv_i32_arr(const KvView& v, const char* k){ gguf_context* g=v.g;
    std::vector<int32_t> out;
    int64_t id = v.find(k);
    if(id>=0 && gguf_get_arr_type(g,id)==GGUF_TYPE_INT32){
        size_t n = gguf_get_arr_n(g,id);
        const int32_t* a = (const int32_t*)gguf_get_arr_data(g,id);
        out.assign(a, a+n);
    }
    return out;
}
static std::vector<std::string> kv_str_arr(const KvView& v, const char* k){ gguf_context* g=v.g;
    std::vector<std::string> out;
    int64_t id = v.find(k);
    if(id>=0 && gguf_get_arr_type(g,id)==GGUF_TYPE_STRING){
        size_t n = gguf_get_arr_n(g,id);
        out.resize(n);
        for(size_t i=0;i<n;++i) out[i] = gguf_get_arr_str(g,id,i);
    }
    return out;
}
static float kv_f32(const KvView& v, const char* k, float d=0){ gguf_context* g=v.g;
    int64_t id = v.find(k); return id<0 ? d : gguf_get_val_f32(g,id);
}
static bool kv_bool(const KvView& v, const char* k, bool d=false){ gguf_context* g=v.g;
    int64_t id = v.find(k); return id<0 ? d : gguf_get_val_bool(g,id);
}
static std::string kv_str(const KvView& v, const char* k, const char* d=""){ gguf_context* g=v.g;
    int64_t id = v.find(k); return id<0 ? std::string(d) : std::string(gguf_get_val_str(g,id));
}
ModelLoader::~ModelLoader(){
    // Free the weight buffer BEFORE the contexts. For the CPU path it is a
    // from_ptr buffer (free_buffer == NULL) that does NOT own its memory
    // (ctx_ does); for the device path it OWNS the device buffer. The device
    // upload path releases ctx_ as soon as the upload completes.
    if(ggml_backend_buffer_t wb = weights_buf_.load()) ggml_backend_buffer_free(wb);
    if(device_ctx_) ggml_free(device_ctx_);
    if(gguf_) gguf_free(gguf_); if(ctx_) ggml_free(ctx_);
}
bool ModelLoader::realize_weights(ggml_backend_t backend){
    if(weights_buf_.load(std::memory_order_acquire)) return true;   // idempotent
    if(!backend || !ctx_){ PK_LOG("realize_weights: null backend/ctx"); return false; }

    if (ggml_backend_is_cpu(backend)) {
        // Fast path: borrow the host ctx memory directly (no copy).
        // The GGUF is loaded with no_alloc=false, so every tensor's data lives
        // in one contiguous ctx mem_buffer. Wrap that exact memory as a CPU
        // backend buffer (zero-copy: ggml_backend_cpu_buffer_from_ptr borrows
        // the ptr) and point every tensor's ->buffer at it, so graphs can
        // reference the loader tensors DIRECTLY as leaves (the gallocr treats
        // data!=NULL tensors as already-allocated and never copies them;
        // reshapes/views resolve at build time). Eliminates per-call recopy.
        void*  base = ggml_get_mem_buffer(ctx_);
        size_t size = ggml_get_mem_size(ctx_);
        ggml_backend_buffer_t wb = ggml_backend_cpu_buffer_from_ptr(base, size);
        if(!wb){ PK_LOG("realize_weights: buffer_from_ptr failed"); return false; }
        for(auto& kv : tensors_) kv.second->buffer = wb;
        // Publish last: a reader that sees the buffer sees the tensors set up.
        weights_buf_.store(wb, std::memory_order_release);
        return true;
    }

    // Device path (CUDA/Metal/Vulkan/...): weights must live in a backend buffer.
    // ctx_ was created no_alloc=false (host-resident data), which
    // ggml_backend_alloc_ctx_tensors rejects (it asserts the ctx is no_alloc).
    // So mirror every weight into a no_alloc=true ctx, allocate THAT on the
    // backend, upload each tensor's bytes from the host source, and repoint the
    // name->tensor map at the device tensors. The host context is only a staging
    // allocation and can be released after all uploads complete.
    const size_t n = tensors_.size();
    struct ggml_init_params dp = {
        /*.mem_size  =*/ ggml_tensor_overhead() * (n + 8),
        /*.mem_buffer=*/ nullptr,
        /*.no_alloc  =*/ true,
    };
    device_ctx_ = ggml_init(dp);
    if(!device_ctx_){ PK_LOG("realize_weights: device ctx init failed"); return false; }

    std::vector<std::pair<ggml_tensor*, const void*>> ups; ups.reserve(n);
    std::unordered_map<std::string, ggml_tensor*> devmap; devmap.reserve(n);
    for (auto& kv : tensors_) {
        ggml_tensor* s = kv.second;
        ggml_tensor* d = ggml_new_tensor(device_ctx_, s->type, GGML_MAX_DIMS, s->ne);
        ggml_set_name(d, kv.first.c_str());
        devmap.emplace(kv.first, d);
        ups.emplace_back(d, s->data);   // host source (valid in ctx_ mem buffer)
    }
    ggml_backend_buffer_t wb = ggml_backend_alloc_ctx_tensors(device_ctx_, backend);
    if(!wb){ PK_LOG("realize_weights: alloc_ctx_tensors failed"); return false; }
    for (auto& pr : ups)
        ggml_backend_tensor_set(pr.first, pr.second, 0, ggml_nbytes(pr.first));
    tensors_.swap(devmap);   // graphs now reference the device-resident tensors

    // The device tensors and their backend buffer own everything needed for
    // inference. Keeping the no_alloc=false loader context alive retained a
    // second full copy of every model weight (notably ~708 MiB for the 0.6B Q5_K
    // model) for the lifetime of the process.
    ggml_free(ctx_);
    ctx_ = nullptr;
    weights_buf_.store(wb, std::memory_order_release);
    return true;
}
bool ModelLoader::load(const std::string& path){
    // A bundle holds several models. Say so instead of failing later on a
    // missing key, and do not read its tensors.
    if(gguf_is_bundle(path)){
        PK_LOG("%s is a bundle GGUF (several models in one file); open one component of it "
               "(load_component, parakeet_capi_load_component, or --component)", path.c_str());
        return false;
    }
    struct gguf_init_params p{ /*no_alloc*/false, /*ctx*/&ctx_ };
    gguf_ = gguf_init_from_file(path.c_str(), p);
    if(!gguf_){ PK_LOG("gguf open failed: %s", path.c_str()); return false; }
    return parse_config(KvView{gguf_, ""});
}
// Bundle path: open the file with no_alloc (header and descriptors only), then
// read only the tensors of one component into a private ctx. Components that
// were not asked for are never read from disk. The tensors keep their original
// (unprefixed) names, so the rest of the engine sees a plain single-model file.
bool ModelLoader::load_component(const std::string& path, const std::string& component){
    BundleInfo info; std::string err;
    if(!read_bundle_info(path, info, &err)){ PK_LOG("%s", err.c_str()); return false; }
    if(!info.find(component)){
        PK_LOG("bundle %s has no component '%s'; components: %s", path.c_str(), component.c_str(),
               bundle_component_names(info).c_str());
        return false;
    }
    struct ggml_context* meta = nullptr;
    struct gguf_init_params p{ /*no_alloc*/true, /*ctx*/&meta };
    gguf_ = gguf_init_from_file(path.c_str(), p);
    if(!gguf_){ PK_LOG("gguf open failed: %s", path.c_str()); return false; }
    const std::string pre = component + ".";
    const int64_t nt = gguf_get_n_tensors(gguf_);
    size_t need = 0; int64_t count = 0;
    for(int64_t i=0;i<nt;++i){
        const char* nm = gguf_get_tensor_name(gguf_,i);
        if(strncmp(nm, pre.c_str(), pre.size())!=0) continue;
        need += GGML_PAD(gguf_get_tensor_size(gguf_,i), GGML_MEM_ALIGN) + ggml_tensor_overhead();
        ++count;
    }
    if(count==0){ PK_LOG("bundle %s: component '%s' has no tensors", path.c_str(), component.c_str()); ggml_free(meta); return false; }
    struct ggml_init_params ip{ need + (1u<<16), nullptr, /*no_alloc*/false };
    ctx_ = ggml_init(ip);
    FILE* f = ctx_ ? fopen(path.c_str(), "rb") : nullptr;
    if(!f){ ggml_free(meta); return false; }
    // Size of the file, to refuse a truncated bundle before reading any tensor.
    fseeko(f, 0, SEEK_END);
    const uint64_t file_size = (uint64_t)ftello(f);
    bool ok = true;
    for(int64_t i=0;i<nt && ok;++i){
        const char* nm = gguf_get_tensor_name(gguf_,i);
        if(strncmp(nm, pre.c_str(), pre.size())!=0) continue;
        ggml_tensor* src = ggml_get_tensor(meta, nm);
        ggml_tensor* t = src ? ggml_new_tensor(ctx_, src->type, GGML_MAX_DIMS, src->ne) : nullptr;
        if(!t){ ok = false; break; }
        ggml_set_name(t, nm + pre.size());
        const uint64_t off = gguf_get_data_offset(gguf_) + gguf_get_tensor_offset(gguf_,i);
        if(off + ggml_nbytes(t) > file_size){
            PK_LOG("bundle %s is truncated (tensor %s)", path.c_str(), nm); ok = false; break;
        }
        if(fseeko(f, (off_t)off, SEEK_SET)!=0 || fread(t->data, 1, ggml_nbytes(t), f)!=ggml_nbytes(t)) ok = false;
    }
    fclose(f);
    ggml_free(meta);
    if(!ok){ PK_LOG("bundle read failed for component '%s'", component.c_str()); return false; }
    return parse_config(KvView{gguf_, pre});
}
bool ModelLoader::parse_config(const KvView& kv){
    cfg_.arch        = kv_str(kv, "parakeet.arch");
    cfg_.feat_in     = kv_u32(kv, "parakeet.encoder.feat_in");
    cfg_.d_model     = kv_u32(kv, "parakeet.encoder.d_model");
    cfg_.n_layers    = kv_u32(kv, "parakeet.encoder.n_layers");
    cfg_.n_heads     = kv_u32(kv, "parakeet.encoder.n_heads");
    cfg_.ff_dim      = kv_u32(kv, "parakeet.encoder.ff_dim");
    cfg_.conv_kernel = kv_u32(kv, "parakeet.encoder.conv_kernel");
    cfg_.conv_norm_type = kv_str(kv, "parakeet.encoder.conv_norm_type", "batch_norm");
    cfg_.subsampling_factor = kv_u32(kv, "parakeet.encoder.subsampling_factor");
    cfg_.subsampling_conv_channels = kv_u32(kv, "parakeet.encoder.subsampling_conv_channels");
    cfg_.xscaling    = kv_bool(kv, "parakeet.encoder.xscaling", true);
    cfg_.pos_emb_max_len = kv_u32(kv, "parakeet.encoder.pos_emb_max_len", 5000);
    // cache-aware streaming / causal config (Phase 5). Absent for offline models
    // -> offline-safe defaults (regular style, no causal, streaming.present=false).
    cfg_.att_context_left  = kv_i32(kv, "parakeet.encoder.att_context_left", -1);
    cfg_.att_context_right = kv_i32(kv, "parakeet.encoder.att_context_right", -1);
    cfg_.att_context_style = kv_str(kv, "parakeet.encoder.att_context_style", "regular");
    cfg_.causal_downsampling = kv_bool(kv, "parakeet.encoder.causal_downsampling", false);
    cfg_.conv_causal = kv_bool(kv, "parakeet.encoder.conv_causal", false);
    // encoder.use_bias: false for nemotron (the attention/FFN linear projections
    // carry no bias tensor). Defaults true so existing models are unaffected.
    cfg_.use_bias = kv_bool(kv, "parakeet.encoder.use_bias", true);
    // Transformer encoder config (diarization models with RoPE attention).
    // Absent for ASR (FastConformer) models → safe defaults.
    cfg_.self_attention_model = kv_str(kv, "parakeet.encoder.self_attention_model", "");
    cfg_.qkv_bias = kv_bool(kv, "parakeet.encoder.qkv_bias", false);
    cfg_.pre_block_norm = kv_bool(kv, "parakeet.encoder.pre_block_norm", true);
    cfg_.rope_base = kv_f32(kv, "parakeet.encoder.rope_base", 10000.0f);
    cfg_.rotary_fraction = kv_f32(kv, "parakeet.encoder.rotary_fraction", 1.0f);
    // Prompt conditioning (multilingual nemotron). Orthogonal capability flag;
    // absent -> present=false and the engine skips the prompt stage entirely.
    cfg_.prompt.present = kv_bool(kv, "parakeet.prompt.present", false);
    if(cfg_.prompt.present){
        cfg_.prompt.num_prompts  = kv_u32(kv, "parakeet.prompt.num_prompts", 0);
        cfg_.prompt.default_lang = kv_str(kv, "parakeet.prompt.default_lang", "");
        cfg_.prompt.dict_keys = kv_str_arr(kv, "parakeet.prompt.dictionary.keys");
        cfg_.prompt.dict_vals = kv_i32_arr(kv, "parakeet.prompt.dictionary.values");
    }
    cfg_.ternary.present = kv_bool(kv, "parakeet.ternary.present", false);
    if(cfg_.ternary.present){
        cfg_.ternary.group_size = kv_u32(kv, "parakeet.ternary.group_size", 128);
        if(cfg_.ternary.group_size != 128){
            PK_LOG("invalid packed ternary GGUF: parakeet.ternary.group_size is %u, only 128 is supported",
                   (unsigned)cfg_.ternary.group_size);
            return false;
        }
    }
    cfg_.vad.present = kv_bool(kv, "parakeet.vad.present", false);
    if(cfg_.vad.present){
        cfg_.vad.d_in      = kv_u32(kv, "parakeet.vad.d_in", 0);
        cfg_.vad.hidden    = kv_u32(kv, "parakeet.vad.hidden", 0);
        cfg_.vad.kernel    = kv_u32(kv, "parakeet.vad.kernel", 0);
        cfg_.vad.frame_sec = kv_f32(kv, "parakeet.vad.frame_sec", 0.08f);
        if(!std::isfinite(cfg_.vad.frame_sec) || !(cfg_.vad.frame_sec > 0.0f) ||
           cfg_.vad.d_in==0 || cfg_.vad.hidden==0 || cfg_.vad.kernel==0 || (cfg_.vad.kernel % 2)==0){
            PK_LOG("invalid VAD config: frame_sec=%g d_in=%u hidden=%u kernel=%u (need finite frame_sec > 0, "
                   "non-zero sizes and an odd kernel)", (double)cfg_.vad.frame_sec,
                   (unsigned)cfg_.vad.d_in, (unsigned)cfg_.vad.hidden, (unsigned)cfg_.vad.kernel);
            return false;
        }
    }
    if(cfg_.att_context_style != "regular"){
        StreamingCfg& s = cfg_.streaming;
        s.chunk_size = kv_i32_arr(kv, "parakeet.streaming.chunk_size");
        s.shift_size = kv_i32_arr(kv, "parakeet.streaming.shift_size");
        s.pre_encode_cache_size = kv_i32_arr(kv, "parakeet.streaming.pre_encode_cache_size");
        s.cache_drop_size = kv_i32(kv, "parakeet.streaming.cache_drop_size", 0);
        s.last_channel_cache_size = kv_i32(kv, "parakeet.streaming.last_channel_cache_size", 0);
        s.valid_out_len = kv_i32(kv, "parakeet.streaming.valid_out_len", 0);
        s.drop_extra_pre_encoded = kv_i32(kv, "parakeet.streaming.drop_extra_pre_encoded", 0);
        s.present = true;
    }
    cfg_.sample_rate = kv_u32(kv, "parakeet.preprocessor.sample_rate", 16000);
    cfg_.n_mels      = kv_u32(kv, "parakeet.preprocessor.n_mels");
    cfg_.n_fft       = kv_u32(kv, "parakeet.preprocessor.n_fft");
    cfg_.win_length  = kv_u32(kv, "parakeet.preprocessor.win_length");
    cfg_.hop_length  = kv_u32(kv, "parakeet.preprocessor.hop_length");
    if(cfg_.vad.present){
        const double enc_frame = (double)cfg_.hop_length * (double)cfg_.subsampling_factor / (double)cfg_.sample_rate;
        const double ratio = enc_frame > 0.0 ? (double)cfg_.vad.frame_sec / enc_frame : 0.0;
        if(!(enc_frame > 0.0) || std::fabs(ratio - std::round(ratio)) > 1e-3 || std::round(ratio) < 1.0){
            PK_LOG("invalid VAD config: frame_sec=%g is not a whole multiple of the encoder frame (%g s)",
                   (double)cfg_.vad.frame_sec, enc_frame);
            return false;
        }
    }
    cfg_.preemph     = kv_f32(kv, "parakeet.preprocessor.preemph", 0.0f);
    cfg_.mag_power   = kv_f32(kv, "parakeet.preprocessor.mag_power", 2.0f);
    cfg_.normalize   = kv_str(kv, "parakeet.preprocessor.normalize", "per_feature");
    cfg_.log_zero_guard = kv_f32(kv, "parakeet.preprocessor.log_zero_guard", 0.0f);
    cfg_.pred_hidden = kv_u32(kv, "parakeet.decoder.pred_hidden");
    cfg_.pred_rnn_layers = kv_u32(kv, "parakeet.decoder.pred_rnn_layers");
    cfg_.joint_hidden = kv_u32(kv, "parakeet.joint.joint_hidden");
    cfg_.joint_activation = kv_str(kv, "parakeet.joint.activation");
    cfg_.max_symbols = kv_u32(kv, "parakeet.decoding.max_symbols", 10);
    cfg_.vocab_size  = kv_u32(kv, "parakeet.vocab_size");
    cfg_.blank_id    = kv_u32(kv, "parakeet.blank_id");
    // diarization config (absent for ASR models → present=false)
    if (kv.find("parakeet.diar.n_speakers") >= 0) {
        auto& d = cfg_.diarization;
        d.present = true;
        d.n_speakers = kv_u32(kv, "parakeet.diar.n_speakers");
        d.tf_d_model = kv_u32(kv, "parakeet.diar.tf_d_model");
        d.upsample_factor = kv_u32(kv, "parakeet.diar.upsample_factor");
        d.frame_resolution_sec = kv_f32(kv, "parakeet.diar.frame_resolution_sec", 0.01f);
        d.onset_threshold = kv_f32(kv, "parakeet.diar.onset_threshold", 0.5f);
        d.offset_threshold = kv_f32(kv, "parakeet.diar.offset_threshold", 0.5f);
        // Streaming (speaker cache) config, in ENCODER frames as in NeMo
        // SortformerModules. Defaults are the Nemotron-3-Diarization values,
        // for GGUFs converted before these keys were written.
        d.streaming_mode         = kv_bool(kv, "parakeet.diar.streaming_mode", true);
        d.chunk_len              = (int32_t)kv_u32(kv, "parakeet.diar.chunk_len", 264);
        d.spkcache_len           = (int32_t)kv_u32(kv, "parakeet.diar.spkcache_len", 264);
        d.fifo_len               = (int32_t)kv_u32(kv, "parakeet.diar.fifo_len", 0);
        d.spkcache_update_period = (int32_t)kv_u32(kv, "parakeet.diar.spkcache_update_period", 264);
        d.spkcache_sil_frames_per_spk = (int32_t)kv_u32(kv, "parakeet.diar.spkcache_sil_frames_per_spk", 1);
        d.sil_threshold          = kv_f32(kv, "parakeet.diar.sil_threshold", 0.2f);
        d.pred_score_threshold   = kv_f32(kv, "parakeet.diar.pred_score_threshold", 0.25f);
        d.scores_boost_latest    = kv_f32(kv, "parakeet.diar.scores_boost_latest", 0.05f);
        d.strong_boost_rate      = kv_f32(kv, "parakeet.diar.strong_boost_rate", 0.75f);
        d.weak_boost_rate        = kv_f32(kv, "parakeet.diar.weak_boost_rate", 1.5f);
        d.min_pos_scores_rate    = kv_f32(kv, "parakeet.diar.min_pos_scores_rate", 0.5f);
        d.use_learnable_sil_emb  = kv_bool(kv, "parakeet.diar.use_learnable_sil_emb",
            gguf_find_tensor(gguf_, (kv.prefix + "sortformer_modules.learnable_sil_emb").c_str()) >= 0);
    }
    // durations array (stored as INT32 by the converter)
    { int64_t id = kv.find("parakeet.tdt.durations");
      if(id>=0 && gguf_get_arr_type(gguf_,id)==GGUF_TYPE_INT32){
          size_t n = gguf_get_arr_n(gguf_,id);
          const int32_t* a = (const int32_t*)gguf_get_arr_data(gguf_,id);
          cfg_.tdt_durations.assign(a, a+n); } }
    // tokenizer pieces STRING array
    { int64_t id = kv.find("parakeet.tokenizer.pieces");
      if(id>=0 && gguf_get_arr_type(gguf_,id)==GGUF_TYPE_STRING){
          size_t n = gguf_get_arr_n(gguf_,id);
          cfg_.tokenizer_pieces.resize(n);
          for(size_t i=0;i<n;++i)
              cfg_.tokenizer_pieces[i] = gguf_get_arr_str(gguf_,id,i); } }
    // tensors
    const int64_t nt = gguf_get_n_tensors(gguf_);
    // (a bundle ctx holds only this component's tensors, named without the prefix)
    for(int64_t i=0;i<nt;++i){ const char* nm = gguf_get_tensor_name(gguf_,i);
        if(strncmp(nm, kv.prefix.c_str(), kv.prefix.size())!=0) continue;
        nm += kv.prefix.size();
        ggml_tensor* t = ggml_get_tensor(ctx_, nm); if(t) tensors_[nm]=t; }
    return cfg_.d_model>0 && (cfg_.vocab_size>0 || cfg_.arch=="diarization");
}
ggml_tensor* ModelLoader::tensor(const std::string& n) const {
    auto it = tensors_.find(n); return it==tensors_.end()? nullptr : it->second;
}
}
