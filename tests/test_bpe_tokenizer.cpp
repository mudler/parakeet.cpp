// Model-free test of the byte-level BPE tokenizer with a tiny synthetic
// vocabulary: merges, ignore_merges, byte alphabet, added tokens, regex
// splitting (non-ASCII letters, digit runs, whitespace), decode, the prompt
// builder and the GGUF loader. Always runs.
#include "bpe_tokenizer.hpp"
#include "unicode_props.hpp"

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

using pk::BpeTokenizer;
using pk::BpeVocab;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define CHECK_IDS(got, ...) do { std::vector<int32_t> exp_ = __VA_ARGS__; if ((got) != exp_) { \
    std::fprintf(stderr, "FAIL %s:%d: ids differ\n", __FILE__, __LINE__); print_ids("got", (got)); print_ids("exp", exp_); ++g_fail; } } while (0)

static void print_ids(const char* n, const std::vector<int32_t>& v) {
    std::fprintf(stderr, "  %s:", n);
    for (int32_t x : v) std::fprintf(stderr, " %d", x);
    std::fprintf(stderr, "\n");
}

static void put_utf8(std::string& s, uint32_t cp) {
    if (cp < 0x80) s += (char)cp;
    else { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F)); }
}

// GPT-2 byte alphabet, written independently of the library.
static std::string bchar(int b) {
    int n = 0;
    for (int i = 0; i < 256; ++i) {
        bool direct = (i >= 33 && i <= 126) || (i >= 161 && i <= 172) || (i >= 174 && i <= 255);
        uint32_t cp = direct ? (uint32_t)i : (uint32_t)(256 + n++);
        if (i == b) { std::string s; put_utf8(s, cp); return s; }
    }
    return "";
}
static std::string bstr(const std::string& raw) {
    std::string s;
    for (unsigned char c : raw) s += bchar(c);
    return s;
}

struct Syn {
    BpeVocab v;
    std::map<std::string, int32_t> id;
    int32_t add(const std::string& tok, int ty) {
        int32_t i = (int32_t)v.tokens.size();
        v.tokens.push_back(tok);
        v.token_types.push_back(ty);
        id[tok] = i;
        return i;
    }
    int32_t b(const std::string& raw) { return id.at(bstr(raw)); }
};

static Syn make_syn(bool ignore_merges) {
    Syn s;
    for (int i = 0; i < 256; ++i) s.add(bchar(i), pk::kTokNormal);   // ids 0..255 = byte values
    auto merge = [&](const std::string& a, const std::string& b) {
        s.v.merges.push_back(bstr(a) + " " + bstr(b));
        s.add(bstr(a) + bstr(b), pk::kTokNormal);
    };
    merge("h", "e");        // he
    merge("l", "l");        // ll
    merge("he", "ll");      // hell
    merge("hell", "o");     // hello
    merge("o", " ");        // "o "  (must never form across a word boundary)
    merge(" ", "w");        // " w"
    merge("\xC3", "\xA9");  // é
    merge("\xC3\xA9", "a"); // éa
    merge("\xC3\xA9", "1"); // é1  (letter and digit are different words)
    merge("1", "2");        // 12  (digits are single words)
    merge("a", "a");        // aa
    merge("aa", "a");       // aaa
    s.add(bstr("ab"), pk::kTokNormal);   // in the vocabulary, no merge produces it
    s.id["<|x|>"] = s.add("<|x|>", pk::kTokControl);
    s.add("<|x|>>", pk::kTokControl);
    s.add("[B]", pk::kTokUserDefined);
    s.add("<spk:1>", pk::kTokControl);
    s.add("<|im_start|>", pk::kTokControl);
    s.add("<|im_end|>", pk::kTokControl);
    s.add("<think>", pk::kTokUserDefined);
    s.add("</think>", pk::kTokUserDefined);
    s.add("<|audio|>", pk::kTokControl);
    s.v.eos_id = s.id["<|im_end|>"];
    s.v.ignore_merges = ignore_merges;
    return s;
}

static std::unique_ptr<BpeTokenizer> build(Syn& s) {
    std::string err;
    auto t = BpeTokenizer::from_vocab(s.v, &err);
    if (!t) { std::fprintf(stderr, "from_vocab: %s\n", err.c_str()); std::exit(1); }
    return t;
}

static void test_encode(bool ignore_merges) {
    Syn s = make_syn(ignore_merges);
    auto t = build(s);
    auto e = [&](const std::string& x) { return t->encode(x); };

    CHECK_IDS(e(""), {});
    CHECK_IDS(e("hello"), {s.b("hello")});
    CHECK_IDS(e("hell"), {s.b("hell")});
    CHECK_IDS(e("he"), {s.b("he")});
    // Same result through the merge path or the whole-word path.
    CHECK_IDS(e("hellohello"), {s.b("hello"), s.b("hello")});
    // ignore_merges only matters for a word that is in the vocabulary but has no merge path.
    if (ignore_merges) CHECK_IDS(e("ab"), {s.b("ab")});
    else CHECK_IDS(e("ab"), {s.b("a"), s.b("b")});
    // Merge priority is the rank, then the leftmost pair: aaaa -> aa aa (rank 10 twice, before aa+a).
    CHECK_IDS(e("aaa"), {s.b("aaa")});
    CHECK_IDS(e("aaaa"), {s.b("aa"), s.b("aa")});
    // " w" forms inside a word start; "o " must not form across words (o | " w").
    CHECK_IDS(e("o w"), {s.b("o"), s.b(" w")});
    CHECK_IDS(e("hello w"), {s.b("hello"), s.b(" w")});
    // Byte alphabet and a non-ASCII letter.
    CHECK_IDS(e("\xC3\xA9"), {s.b("\xC3\xA9")});
    CHECK_IDS(e("\xC3\xA9" "a"), {s.b("\xC3\xA9" "a")});
    // Letter then digit are different words: the "é1" merge must not apply.
    CHECK_IDS(e("\xC3\xA9" "1"), {s.b("\xC3\xA9"), s.b("1")});
    // Digits split one by one: no "12" merge across words.
    CHECK_IDS(e("12"), {s.b("1"), s.b("2")});
    CHECK_IDS(e("123"), {s.b("1"), s.b("2"), s.b("3")});
    CHECK_IDS(e("2024"), {s.b("2"), s.b("0"), s.b("2"), s.b("4")});
    // Bytes missing from merges stay byte tokens (CJK, emoji).
    CHECK_IDS(e("\xE4\xBD\xA0"), {s.b("\xE4"), s.b("\xBD"), s.b("\xA0")});
    // Whitespace: a run before a word keeps one space for the word.
    CHECK_IDS(e("a  b"), {s.b("a"), s.b(" "), s.b(" "), s.b("b")});
    CHECK_IDS(e("a\n\nb"), {s.b("a"), s.b("\n"), s.b("\n"), s.b("b")});
    CHECK_IDS(e("a \nb"), {s.b("a"), s.b(" "), s.b("\n"), s.b("b")});
    // Added tokens: whole-string match, leftmost-longest.
    CHECK_IDS(e("a<|x|>b"), {s.b("a"), s.id["<|x|>"], s.b("b")});
    CHECK_IDS(e("<|x|>>"), {s.id["<|x|>>"]});
    CHECK_IDS(e("<|x|>>>"), {s.id["<|x|>>"], s.b(">")});
    CHECK_IDS(e("<|x|"), {s.b("<"), s.b("|"), s.b("x"), s.b("|")});
    CHECK_IDS(e("[B][B]"), {s.id["[B]"], s.id["[B]"]});
    CHECK_IDS(e("hello<spk:1>hello"), {s.b("hello"), s.id["<spk:1>"], s.b("hello")});
    CHECK_IDS(e("o<|x|> w"), {s.b("o"), s.id["<|x|>"], s.b(" w")});
    // Invalid UTF-8 still encodes byte by byte.
    CHECK_IDS(e("\xFF" "a"), {s.b("\xFF"), s.b("a")});
}

static void test_decode() {
    Syn s = make_syn(true);
    auto t = build(s);
    auto E = [&](const std::string& x) { return t->encode(x); };
    for (const char* x : {"hello world", "a  b\n\nc", "\xC3\xA9" "a 12 \xE4\xBD\xA0\xF0\x9F\x98\x80", "tab\there", ""}) {
        CHECK(t->decode(E(x)) == x);
    }
    // Added tokens: dropped by default (special), kept when asked.
    std::vector<int32_t> ids = E("a<|x|>b[B]<spk:1>c");
    CHECK(t->decode(ids) == "ab[B]c");
    pk::DecodeOptions keep; keep.keep_tags = true;
    CHECK(t->decode(ids, keep) == "ab[B]<spk:1>c");
    pk::DecodeOptions all; all.skip_special = false;
    CHECK(t->decode(ids, all) == "a<|x|>b[B]<spk:1>c");
    CHECK(t->is_tag(s.id["<spk:1>"]));
    CHECK(!t->is_tag(s.id["<|x|>"]) && t->is_special(s.id["<|x|>"]));
    CHECK(!t->is_special(s.id["[B]"]));
    // Stop at EOS.
    std::vector<int32_t> with_eos = E("hi");
    with_eos.push_back(t->eos_id());
    for (int32_t x : E("zz")) with_eos.push_back(x);
    CHECK(t->decode(with_eos) == "hi");
    pk::DecodeOptions noeos; noeos.stop_at_eos = false;
    CHECK(t->decode(with_eos, noeos) == "hizz");
    // A multi-byte character split across ids is joined; a lone byte is repaired.
    CHECK(t->decode({s.b("\xC3"), s.b("\xA9")}) == "\xC3\xA9");
    CHECK(t->decode({s.b("\xC3")}) == "\xEF\xBF\xBD");
    CHECK(t->decode({s.b("\xE4"), s.b("\xBD"), s.b("a")}) == "\xEF\xBF\xBD" "a");
    CHECK(t->decode({s.b("\xFF"), s.b("a")}) == "\xEF\xBF\xBD" "a");
    CHECK(t->decode({-5, 999999, s.b("a")}) == "a");
}

static void test_prompt() {
    Syn s = make_syn(true);
    auto t = build(s);
    const std::string p = "Say it";
    CHECK(pk::render_transcription_prompt(p, true) ==
          "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nSay it <|audio|><|im_end|>\n<|im_start|>assistant\n<think></think>");
    CHECK(pk::render_transcription_prompt(p, false) ==
          "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nSay it<|im_end|>\n<|im_start|>assistant\n<think></think>");
    int32_t au = s.id["<|audio|>"];
    pk::TranscriptionPrompt r1 = pk::build_transcription_prompt(*t, p, 1);
    CHECK(r1.ids == t->encode(pk::render_transcription_prompt(p, true)));
    CHECK(r1.audio_begin >= 0 && r1.ids[(size_t)r1.audio_begin] == au && r1.audio_rows == 1);
    pk::TranscriptionPrompt r7 = pk::build_transcription_prompt(*t, p, 7);
    CHECK(r7.ids.size() == r1.ids.size() + 6);
    CHECK(r7.audio_begin == r1.audio_begin);
    for (int i = 0; i < 7; ++i) CHECK(r7.ids[(size_t)(r7.audio_begin + i)] == au);
    CHECK(r7.ids[(size_t)(r7.audio_begin + 7)] == s.id["<|im_end|>"]);
    int n_au = 0;
    for (int32_t x : r7.ids) n_au += x == au;
    CHECK(n_au == 7);
    pk::TranscriptionPrompt r0 = pk::build_transcription_prompt(*t, p, 0);
    CHECK(r0.audio_begin == -1 && r0.ids == t->encode(pk::render_transcription_prompt(p, false)));
    CHECK(r0.ids.front() == s.id["<|im_start|>"]);
    CHECK(r0.ids.back() == s.id["</think>"]);
}

static void test_unicode() {
    using namespace pk::unicode;
    CHECK(props('A') == kLu && props('a') == kLl && props('5') == kN && props(' ') == kWS);
    CHECK(props(0x00E9) == kLl && props(0x0301) == kM && props(0x4F60) == kLo);
    CHECK(props(0x01C5) == kLt && props(0x02B0) == kLm && props(0x0663) == kN);
    CHECK(props(0x00A0) == kWS && props(0x2003) == kWS && props(0x3000) == kWS);
    CHECK(props('\n') == kWS && props('!') == 0 && props(0x1F600) == 0);
    CHECK(props(0x10FFFF) == 0 && props(0x20000) == kLo);
    CHECK(props(0x10D40) == kN && props(0x10D4A) == kLo && props(0x0897) == kM);  // Unicode 16
    CHECK(std::string(version()) == "16.0.0");
}

static void test_gguf_roundtrip() {
    Syn s = make_syn(true);
    gguf_context* g = gguf_init_empty();
    gguf_set_val_str(g, pk::bpe_keys::kModel, "gpt2");
    gguf_set_val_str(g, pk::bpe_keys::kPre, "tekken");
    std::vector<const char*> toks, merges;
    for (auto& x : s.v.tokens) toks.push_back(x.c_str());
    for (auto& x : s.v.merges) merges.push_back(x.c_str());
    gguf_set_arr_str(g, pk::bpe_keys::kTokens, toks.data(), toks.size());
    gguf_set_arr_str(g, pk::bpe_keys::kMerges, merges.data(), merges.size());
    gguf_set_arr_data(g, pk::bpe_keys::kTokenType, GGUF_TYPE_INT32, s.v.token_types.data(), s.v.token_types.size());
    gguf_set_val_bool(g, pk::bpe_keys::kIgnoreMerges, true);
    gguf_set_val_u32(g, pk::bpe_keys::kEos, (uint32_t)s.v.eos_id);
    const char* td = std::getenv("TMPDIR");
    std::string path = std::string(td && *td ? td : ".") + "/pk_bpe_synthetic.gguf";
    CHECK(gguf_write_to_file(g, path.c_str(), false));
    gguf_free(g);

    std::string err;
    auto t = BpeTokenizer::load_gguf(path, &err);
    if (!t) { std::fprintf(stderr, "load_gguf: %s\n", err.c_str()); ++g_fail; return; }
    auto ref = build(s);
    CHECK(t->vocab_size() == ref->vocab_size() && t->eos_id() == ref->eos_id());
    for (const char* x : {"hello w", "a<|x|>b", "\xC3\xA9" "a12", "aaaa", "ab"}) CHECK(t->encode(x) == ref->encode(x));
    std::remove(path.c_str());

    CHECK(BpeTokenizer::load_gguf("/nonexistent/x.gguf", &err) == nullptr);
    BpeVocab bad = s.v; bad.pre = "other";
    CHECK(BpeTokenizer::from_vocab(bad, &err) == nullptr);
    bad = s.v; bad.merges.push_back("zz yy");
    CHECK(BpeTokenizer::from_vocab(bad, &err) == nullptr);
}

int main() {
    test_unicode();
    test_encode(true);
    test_encode(false);
    test_decode();
    test_prompt();
    test_gguf_roundtrip();
    if (g_fail) { std::fprintf(stderr, "test_bpe_tokenizer: %d failures\n", g_fail); return 1; }
    std::printf("test_bpe_tokenizer: ok\n");
    return 0;
}
