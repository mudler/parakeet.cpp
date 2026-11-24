#pragma once
// Byte-level BPE tokenizer (encoder and decoder) and the transcription prompt
// builder for LLM-decoder checkpoints such as Nemotron 3.5 Transcribe.
//
// This sits beside src/tokenizer.* (SentencePiece detokenizer used by the
// transducer models) and does not change it. See docs/bpe-tokenizer.md.
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace pk {

// GGUF metadata keys (the llama.cpp BPE layout plus one extra key).
namespace bpe_keys {
constexpr const char* kModel = "tokenizer.ggml.model";          // "gpt2"
constexpr const char* kPre = "tokenizer.ggml.pre";              // "tekken"
constexpr const char* kPrePattern = "tokenizer.ggml.pre_pattern";  // informative
constexpr const char* kTokens = "tokenizer.ggml.tokens";        // string[]
constexpr const char* kTokenType = "tokenizer.ggml.token_type";  // int32[]
constexpr const char* kMerges = "tokenizer.ggml.merges";        // string[] "a b"
constexpr const char* kBos = "tokenizer.ggml.bos_token_id";
constexpr const char* kEos = "tokenizer.ggml.eos_token_id";
constexpr const char* kUnk = "tokenizer.ggml.unknown_token_id";
constexpr const char* kPad = "tokenizer.ggml.padding_token_id";
constexpr const char* kIgnoreMerges = "tokenizer.ggml.ignore_merges";  // bool
}  // namespace bpe_keys

// token_type values (llama.cpp enum). Types 3 and 4 mark added tokens, which
// are matched as whole strings before BPE. Type 3 is a special token.
enum BpeTokenType : int32_t {
    kTokNormal = 1,
    kTokUnknown = 2,
    kTokControl = 3,
    kTokUserDefined = 4,
};

struct BpeVocab {
    std::vector<std::string> tokens;      // byte-level form; added tokens are literal
    std::vector<int32_t> token_types;     // same size as tokens
    std::vector<std::string> merges;      // "left right" in rank order
    std::string pre = "tekken";
    bool ignore_merges = true;
    int32_t bos_id = -1, eos_id = -1, unk_id = -1, pad_id = -1;
};

struct DecodeOptions {
    bool keep_tags = false;     // keep <spk:N>, <ev:...>, <sty:...> tags in the text
    bool skip_special = true;   // drop other special tokens (<|im_end|>, ...)
    bool stop_at_eos = true;    // stop at the first EOS id (not included)
};

class BpeTokenizer {
public:
    // Load from a GGUF file that carries the keys in bpe_keys. Returns null and
    // fills *err on failure.
    static std::unique_ptr<BpeTokenizer> load_gguf(const std::string& path, std::string* err);
    static std::unique_ptr<BpeTokenizer> from_vocab(BpeVocab v, std::string* err);

    // Text to ids: added tokens as whole strings, then the tekken pre-tokenizer
    // regex, byte-to-unicode, and BPE merges (ignore_merges: a word that is a
    // single vocabulary entry is emitted as is). No BOS/EOS is added.
    std::vector<int32_t> encode(std::string_view text) const;

    // Byte spans [begin, end) of the words the pre-tokenizer regex cuts text
    // into. Added tokens are not looked for. Exposed for tests.
    std::vector<std::pair<size_t, size_t>> pre_tokenize(std::string_view text) const;

    // Ids to UTF-8 text. Invalid byte sequences become U+FFFD.
    std::string decode(const std::vector<int32_t>& ids, const DecodeOptions& opt = {}) const;
    // Same without the UTF-8 repair.
    std::string decode_bytes(const std::vector<int32_t>& ids, const DecodeOptions& opt = {}) const;

    size_t vocab_size() const { return tokens_.size(); }
    int32_t eos_id() const { return eos_id_; }
    int32_t bos_id() const { return bos_id_; }
    int32_t unk_id() const { return unk_id_; }
    int32_t pad_id() const { return pad_id_; }
    // Id of an exact token string (added token or vocabulary entry), or -1.
    int32_t token_id(const std::string& token) const;
    const std::string& token_text(int32_t id) const { return tokens_[(size_t)id]; }
    bool is_special(int32_t id) const;
    bool is_tag(int32_t id) const;  // <spk:..>, <ev:..>, <sty:..>

private:
    BpeTokenizer() = default;
    bool init(BpeVocab v, std::string* err);
    void encode_segment(std::string_view seg, std::vector<int32_t>& out) const;
    void encode_word(const unsigned char* p, size_t n, std::vector<int32_t>& out) const;

    struct TrieNode {
        int32_t id = -1;
        std::vector<std::pair<uint8_t, int32_t>> kids;
    };

    std::vector<std::string> tokens_;
    std::vector<uint8_t> cls_;  // 0 normal, 1 special added, 2 tag, 3 non-special added
    std::unordered_map<std::string, int32_t> vocab_;
    std::unordered_map<uint64_t, uint32_t> pair_rank_;  // (left<<32|right) -> rank
    std::vector<int32_t> rank_result_;                  // rank -> merged id
    std::vector<TrieNode> trie_;
    bool first_byte_[256] = {};
    bool ignore_merges_ = true;
    int32_t bos_id_ = -1, eos_id_ = -1, unk_id_ = -1, pad_id_ = -1;
    int32_t byte_id_[256];
};

// ---- Transcription prompt -------------------------------------------------

// processor_config.json "default_transcription_prompt".
extern const char* const kDefaultTranscriptionPrompt;
// Placeholder text and the id it has in the Nemotron 3.5 Transcribe vocabulary.
extern const char* const kAudioPlaceholder;
constexpr int32_t kNemotronAudioTokenId = 131072;
constexpr int32_t kNemotronEosId = 11;

struct TranscriptionPrompt {
    std::vector<int32_t> ids;
    int audio_begin = -1;  // index of the first audio row, -1 when there is none
    int audio_rows = 0;
};

// The prompt text exactly as chat_template.jinja renders it for one user turn
// with enable_thinking false and add_generation_prompt true:
//   <|im_start|>system\n<|im_end|>\n<|im_start|>user\n{prompt} <|audio|><|im_end|>\n
//   <|im_start|>assistant\n<think></think>
// with_audio false leaves out " <|audio|>".
std::string render_transcription_prompt(const std::string& prompt, bool with_audio);

// Token ids of that prompt. The audio placeholder is emitted audio_rows times
// (once per audio row, no separator); audio_rows 0 renders the text without it.
TranscriptionPrompt build_transcription_prompt(const BpeTokenizer& tok,
                                               const std::string& prompt,
                                               int audio_rows);

}  // namespace pk
