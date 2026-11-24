#include "bpe_tokenizer.hpp"
#include "unicode_props.hpp"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstring>
#include <queue>

namespace pk {

// ---------------------------------------------------------------------------
// Byte-level alphabet (GPT-2 bytes_to_unicode).
// ---------------------------------------------------------------------------
namespace {

struct ByteTable {
    uint32_t to_cp[256];
    int16_t from_cp[324];
    ByteTable() {
        for (auto& v : from_cp) v = -1;
        uint32_t n = 0;
        for (uint32_t b = 0; b < 256; ++b) {
            bool direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            to_cp[b] = direct ? b : 256 + n++;
            from_cp[to_cp[b]] = (int16_t)b;
        }
    }
};
const ByteTable& byte_table() {
    static const ByteTable t;
    return t;
}

void append_utf8(std::string& s, uint32_t cp) {
    if (cp < 0x80) {
        s += (char)cp;
    } else if (cp < 0x800) {
        s += (char)(0xC0 | (cp >> 6));
        s += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += (char)(0xE0 | (cp >> 12));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    } else {
        s += (char)(0xF0 | (cp >> 18));
        s += (char)(0x80 | ((cp >> 12) & 0x3F));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    }
}

// Length of the well-formed UTF-8 sequence at p (bytes left: n), 0 if the
// first byte does not start a well-formed sequence. Sets *cp.
int utf8_len(const unsigned char* p, size_t n, uint32_t* cp) {
    unsigned char c = p[0];
    if (c < 0x80) { *cp = c; return 1; }
    int len; uint32_t v; unsigned char lo = 0x80, hi = 0xBF;
    if (c >= 0xC2 && c <= 0xDF) { len = 2; v = c & 0x1F; }
    else if (c >= 0xE0 && c <= 0xEF) { len = 3; v = c & 0x0F; if (c == 0xE0) lo = 0xA0; if (c == 0xED) hi = 0x9F; }
    else if (c >= 0xF0 && c <= 0xF4) { len = 4; v = c & 0x07; if (c == 0xF0) lo = 0x90; if (c == 0xF4) hi = 0x8F; }
    else return 0;
    if (n < (size_t)len) return 0;
    for (int i = 1; i < len; ++i) {
        unsigned char d = p[i];
        unsigned char l = (i == 1) ? lo : 0x80, h = (i == 1) ? hi : 0xBF;
        if (d < l || d > h) return 0;
        v = (v << 6) | (d & 0x3F);
    }
    *cp = v;
    return len;
}

// Replace ill-formed sequences by U+FFFD, one per maximal ill-formed prefix
// (same practice as Rust's String::from_utf8_lossy and Python's "replace").
std::string repair_utf8(const std::string& in) {
    const unsigned char* p = (const unsigned char*)in.data();
    size_t n = in.size();
    std::string out;
    out.reserve(n);
    size_t i = 0;
    while (i < n) {
        uint32_t cp;
        int l = utf8_len(p + i, n - i, &cp);
        if (l > 0) { out.append((const char*)p + i, (size_t)l); i += (size_t)l; continue; }
        // Maximal prefix of a valid sequence.
        unsigned char c = p[i];
        size_t take = 1;
        int need = (c >= 0xC2 && c <= 0xDF) ? 2 : (c >= 0xE0 && c <= 0xEF) ? 3 : (c >= 0xF0 && c <= 0xF4) ? 4 : 1;
        unsigned char lo = 0x80, hi = 0xBF;
        if (c == 0xE0) lo = 0xA0; else if (c == 0xED) hi = 0x9F;
        else if (c == 0xF0) lo = 0x90; else if (c == 0xF4) hi = 0x8F;
        for (int k = 1; k < need && i + (size_t)k < n; ++k) {
            unsigned char d = p[i + (size_t)k];
            unsigned char l2 = (k == 1) ? lo : 0x80, h2 = (k == 1) ? hi : 0xBF;
            if (d < l2 || d > h2) break;
            take++;
        }
        out += "\xEF\xBF\xBD";
        i += take;
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

bool BpeTokenizer::init(BpeVocab v, std::string* err) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
    if (v.pre != "tekken") return fail("unsupported pre-tokenizer '" + v.pre + "' (only 'tekken')");
    if (v.tokens.empty()) return fail("empty vocabulary");
    if (v.token_types.size() != v.tokens.size()) return fail("token_type size differs from tokens");
    tokens_ = std::move(v.tokens);
    ignore_merges_ = v.ignore_merges;
    bos_id_ = v.bos_id; eos_id_ = v.eos_id; unk_id_ = v.unk_id; pad_id_ = v.pad_id;

    const size_t nt = tokens_.size();
    vocab_.reserve(nt * 2);
    cls_.assign(nt, 0);
    trie_.assign(1, TrieNode());
    for (size_t i = 0; i < nt; ++i) {
        vocab_.emplace(tokens_[i], (int32_t)i);
        int32_t ty = v.token_types[i];
        if (ty != kTokControl && ty != kTokUserDefined) continue;
        const std::string& s = tokens_[i];
        if (s.empty()) continue;
        bool tag = s.rfind("<spk:", 0) == 0 || s.rfind("<ev:", 0) == 0 || s.rfind("<sty:", 0) == 0;
        cls_[i] = tag ? 2 : (ty == kTokControl ? 1 : 3);
        int32_t node = 0;
        for (unsigned char c : s) {
            int32_t nx = -1;
            for (auto& kv : trie_[(size_t)node].kids) if (kv.first == c) { nx = kv.second; break; }
            if (nx < 0) {
                nx = (int32_t)trie_.size();
                trie_.emplace_back();
                trie_[(size_t)node].kids.emplace_back(c, nx);
            }
            node = nx;
        }
        if (trie_[(size_t)node].id < 0) trie_[(size_t)node].id = (int32_t)i;
        first_byte_[(unsigned char)s[0]] = true;
    }

    const ByteTable& bt = byte_table();
    for (int b = 0; b < 256; ++b) {
        std::string s;
        append_utf8(s, bt.to_cp[b]);
        auto it = vocab_.find(s);
        byte_id_[b] = it == vocab_.end() ? -1 : it->second;
    }

    pair_rank_.reserve(v.merges.size() * 2);
    rank_result_.reserve(v.merges.size());
    for (size_t r = 0; r < v.merges.size(); ++r) {
        const std::string& m = v.merges[r];
        size_t sp = m.find(' ', 1);
        if (sp == std::string::npos) return fail("bad merge line " + std::to_string(r));
        auto a = vocab_.find(m.substr(0, sp));
        auto b = vocab_.find(m.substr(sp + 1));
        auto c = vocab_.find(m.substr(0, sp) + m.substr(sp + 1));
        if (a == vocab_.end() || b == vocab_.end() || c == vocab_.end())
            return fail("merge " + std::to_string(r) + " refers to a token outside the vocabulary");
        uint64_t key = ((uint64_t)(uint32_t)a->second << 32) | (uint32_t)b->second;
        if (pair_rank_.emplace(key, (uint32_t)r).second) {}
        rank_result_.push_back(c->second);
    }
    return true;
}

std::unique_ptr<BpeTokenizer> BpeTokenizer::from_vocab(BpeVocab v, std::string* err) {
    std::unique_ptr<BpeTokenizer> t(new BpeTokenizer());
    if (!t->init(std::move(v), err)) return nullptr;
    return t;
}

std::unique_ptr<BpeTokenizer> BpeTokenizer::load_gguf(const std::string& path, std::string* err) {
    gguf_init_params ip{};
    ip.no_alloc = true;
    ip.ctx = nullptr;
    gguf_context* g = gguf_init_from_file(path.c_str(), ip);
    if (!g) { if (err) *err = "cannot open GGUF " + path; return nullptr; }
    struct Guard { gguf_context* g; ~Guard() { gguf_free(g); } } guard{g};

    auto find = [&](const char* k) { return gguf_find_key(g, k); };
    auto str = [&](const char* k, const char* d) -> std::string {
        int64_t i = find(k);
        return i < 0 ? std::string(d) : std::string(gguf_get_val_str(g, i));
    };
    auto i32 = [&](const char* k) -> int32_t {
        int64_t i = find(k);
        if (i < 0) return -1;
        switch (gguf_get_kv_type(g, i)) {
            case GGUF_TYPE_UINT32: return (int32_t)gguf_get_val_u32(g, i);
            case GGUF_TYPE_INT32: return gguf_get_val_i32(g, i);
            default: return -1;
        }
    };
    auto strs = [&](const char* k, std::vector<std::string>& out) {
        int64_t i = find(k);
        if (i < 0 || gguf_get_kv_type(g, i) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g, i) != GGUF_TYPE_STRING)
            return false;
        size_t n = gguf_get_arr_n(g, i);
        out.resize(n);
        for (size_t j = 0; j < n; ++j) out[j] = gguf_get_arr_str(g, i, j);
        return true;
    };

    if (str(bpe_keys::kModel, "") != "gpt2") {
        if (err) *err = std::string(bpe_keys::kModel) + " is not 'gpt2'";
        return nullptr;
    }
    BpeVocab v;
    v.pre = str(bpe_keys::kPre, "");
    if (!strs(bpe_keys::kTokens, v.tokens)) { if (err) *err = "missing tokenizer.ggml.tokens"; return nullptr; }
    if (!strs(bpe_keys::kMerges, v.merges)) { if (err) *err = "missing tokenizer.ggml.merges"; return nullptr; }
    int64_t ti = find(bpe_keys::kTokenType);
    if (ti >= 0 && gguf_get_kv_type(g, ti) == GGUF_TYPE_ARRAY && gguf_get_arr_type(g, ti) == GGUF_TYPE_INT32) {
        const int32_t* a = (const int32_t*)gguf_get_arr_data(g, ti);
        v.token_types.assign(a, a + gguf_get_arr_n(g, ti));
    } else {
        v.token_types.assign(v.tokens.size(), kTokNormal);
    }
    int64_t im = find(bpe_keys::kIgnoreMerges);
    v.ignore_merges = im < 0 ? false : gguf_get_val_bool(g, im);
    v.bos_id = i32(bpe_keys::kBos);
    v.eos_id = i32(bpe_keys::kEos);
    v.unk_id = i32(bpe_keys::kUnk);
    v.pad_id = i32(bpe_keys::kPad);
    return from_vocab(std::move(v), err);
}

int32_t BpeTokenizer::token_id(const std::string& token) const {
    auto it = vocab_.find(token);
    return it == vocab_.end() ? -1 : it->second;
}
bool BpeTokenizer::is_special(int32_t id) const {
    return id >= 0 && (size_t)id < cls_.size() && (cls_[(size_t)id] == 1 || cls_[(size_t)id] == 2);
}
bool BpeTokenizer::is_tag(int32_t id) const {
    return id >= 0 && (size_t)id < cls_.size() && cls_[(size_t)id] == 2;
}

// ---------------------------------------------------------------------------
// Pre-tokenizer: the "tekken" Split regex, written out by hand.
//
//   [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+
// | [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*
// | \p{N}
// | ?[^\s\p{L}\p{N}]+[\r\n/]*
// | \s*[\r\n]+
// | \s+(?!\S)
// | \s+
//
// Alternatives are tried in this order at each position (leftmost, first
// alternative that matches, greedy with backtracking). The converter refuses
// a tokenizer.json whose pattern differs from this one.
// ---------------------------------------------------------------------------
namespace {

struct Cps {
    std::vector<uint32_t> cp;    // code points
    std::vector<uint32_t> off;   // byte offsets, off[n] = size
    std::vector<uint8_t> pr;     // property masks
};

void decode_cps(std::string_view s, Cps& o) {
    const unsigned char* p = (const unsigned char*)s.data();
    size_t n = s.size();
    o.cp.clear(); o.off.clear(); o.pr.clear();
    o.cp.reserve(n); o.off.reserve(n + 1); o.pr.reserve(n);
    size_t i = 0;
    while (i < n) {
        uint32_t cp;
        int l = utf8_len(p + i, n - i, &cp);
        if (l == 0) { cp = 0xFFFD; l = 1; }
        o.off.push_back((uint32_t)i);
        o.cp.push_back(cp);
        o.pr.push_back((uint8_t)unicode::props(cp));
        i += (size_t)l;
    }
    o.off.push_back((uint32_t)n);
}

struct Matcher {
    const Cps& c;
    size_t n;
    explicit Matcher(const Cps& cc) : c(cc), n(cc.cp.size()) {}

    bool is_nl(size_t i) const { return c.cp[i] == '\r' || c.cp[i] == '\n'; }
    bool is_L(size_t i) const { return c.pr[i] & unicode::kLetter; }
    bool is_N(size_t i) const { return c.pr[i] & unicode::kN; }
    bool is_ws(size_t i) const { return c.pr[i] & unicode::kWS; }
    bool is_pre(size_t i) const { return !is_nl(i) && !is_L(i) && !is_N(i); }
    bool is_U(size_t i) const {
        return c.pr[i] & (unicode::kLu | unicode::kLt | unicode::kLm | unicode::kLo | unicode::kM);
    }
    bool is_Lw(size_t i) const {
        return c.pr[i] & (unicode::kLl | unicode::kLm | unicode::kLo | unicode::kM);
    }

    bool alt1(size_t i, size_t* end) const {
        for (int pref = (i < n && is_pre(i)) ? 1 : 0; pref >= 0; --pref) {
            size_t s = i + (size_t)pref;
            size_t k = s;
            while (k < n && is_U(k)) ++k;
            for (size_t j = k + 1; j-- > s;) {
                if (j < n && is_Lw(j)) {
                    size_t e = j;
                    while (e < n && is_Lw(e)) ++e;
                    *end = e;
                    return true;
                }
            }
        }
        return false;
    }
    bool alt2(size_t i, size_t* end) const {
        for (int pref = (i < n && is_pre(i)) ? 1 : 0; pref >= 0; --pref) {
            size_t s = i + (size_t)pref;
            size_t k = s;
            while (k < n && is_U(k)) ++k;
            if (k > s) {
                while (k < n && is_Lw(k)) ++k;
                *end = k;
                return true;
            }
        }
        return false;
    }
    bool alt4(size_t i, size_t* end) const {
        size_t j = i;
        if (c.cp[j] == ' ') ++j;
        size_t k = j;
        while (k < n && !is_ws(k) && !is_L(k) && !is_N(k)) ++k;
        if (k == j) return false;
        while (k < n && (is_nl(k) || c.cp[k] == '/')) ++k;
        *end = k;
        return true;
    }
    // Whitespace alternatives 5, 6, 7 at a whitespace position.
    size_t ws_match(size_t i) const {
        size_t r = i;
        while (r < n && is_ws(r)) ++r;
        for (size_t q = r; q-- > i;)
            if (is_nl(q)) return q + 1;          // \s*[\r\n]+
        if (r == n) return r;                    // \s+(?!\S) at end
        if (r - i >= 2) return r - 1;            // \s+(?!\S) after backtracking
        return r;                                // \s+
    }

    size_t match(size_t i) const {
        size_t e;
        if (alt1(i, &e)) return e;
        if (alt2(i, &e)) return e;
        if (is_N(i)) return i + 1;
        if (alt4(i, &e)) return e;
        if (is_ws(i)) return ws_match(i);
        return i + 1;  // not reachable for valid input; keeps progress
    }
};

}  // namespace

// ---------------------------------------------------------------------------
// BPE
// ---------------------------------------------------------------------------

void BpeTokenizer::encode_word(const unsigned char* p, size_t n, std::vector<int32_t>& out) const {
    if (n == 1) {
        if (byte_id_[p[0]] >= 0) out.push_back(byte_id_[p[0]]);
        return;
    }
    const ByteTable& bt = byte_table();
    if (ignore_merges_) {
        std::string w;
        w.reserve(n * 2);
        for (size_t i = 0; i < n; ++i) append_utf8(w, bt.to_cp[p[i]]);
        auto it = vocab_.find(w);
        if (it != vocab_.end()) { out.push_back(it->second); return; }
    }

    struct Sym { int32_t id, prev, next, len; };
    std::vector<Sym> sy;
    sy.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        int32_t id = byte_id_[p[i]];
        if (id < 0) continue;  // not in the vocabulary: dropped (no unk token)
        sy.push_back({id, (int32_t)sy.size() - 1, (int32_t)sy.size() + 1, 1});
    }
    if (sy.empty()) return;
    sy.back().next = -1;

    struct Cand {
        uint32_t rank; int32_t pos; int32_t new_id;
        bool operator>(const Cand& o) const { return rank != o.rank ? rank > o.rank : pos > o.pos; }
    };
    std::priority_queue<Cand, std::vector<Cand>, std::greater<Cand>> heap;
    auto push = [&](int32_t pos) {
        int32_t nx = sy[(size_t)pos].next;
        if (nx < 0) return;
        uint64_t key = ((uint64_t)(uint32_t)sy[(size_t)pos].id << 32) | (uint32_t)sy[(size_t)nx].id;
        auto it = pair_rank_.find(key);
        if (it != pair_rank_.end()) heap.push({it->second, pos, rank_result_[it->second]});
    };
    for (size_t i = 0; i + 1 < sy.size(); ++i) push((int32_t)i);

    while (!heap.empty()) {
        Cand t = heap.top();
        heap.pop();
        Sym& l = sy[(size_t)t.pos];
        if (l.len == 0 || l.next < 0) continue;
        Sym& r = sy[(size_t)l.next];
        uint64_t key = ((uint64_t)(uint32_t)l.id << 32) | (uint32_t)r.id;
        auto it = pair_rank_.find(key);
        if (it == pair_rank_.end() || rank_result_[it->second] != t.new_id) continue;  // stale
        int32_t right = l.next;
        l.id = t.new_id;
        l.len += r.len;
        l.next = r.next;
        r.len = 0;
        if (l.next >= 0) sy[(size_t)l.next].prev = t.pos;
        if (l.prev >= 0) push(l.prev);
        push(t.pos);
        (void)right;
    }
    for (int32_t i = 0; i >= 0; i = sy[(size_t)i].next) out.push_back(sy[(size_t)i].id);
}

void BpeTokenizer::encode_segment(std::string_view seg, std::vector<int32_t>& out) const {
    if (seg.empty()) return;
    Cps cps;
    decode_cps(seg, cps);
    Matcher m(cps);
    const unsigned char* base = (const unsigned char*)seg.data();
    size_t i = 0, n = cps.cp.size();
    while (i < n) {
        size_t e = m.match(i);
        if (e <= i) e = i + 1;
        encode_word(base + cps.off[i], cps.off[e] - cps.off[i], out);
        i = e;
    }
}

std::vector<std::pair<size_t, size_t>> BpeTokenizer::pre_tokenize(std::string_view text) const {
    std::vector<std::pair<size_t, size_t>> out;
    Cps cps;
    decode_cps(text, cps);
    Matcher m(cps);
    size_t i = 0, n = cps.cp.size();
    while (i < n) {
        size_t e = m.match(i);
        if (e <= i) e = i + 1;
        out.emplace_back(cps.off[i], cps.off[e]);
        i = e;
    }
    return out;
}

std::vector<int32_t> BpeTokenizer::encode(std::string_view text) const {
    std::vector<int32_t> out;
    out.reserve(text.size() / 3 + 4);
    const unsigned char* t = (const unsigned char*)text.data();
    size_t n = text.size(), seg = 0, i = 0;
    while (i < n) {
        if (first_byte_[t[i]]) {
            int32_t node = 0, best = -1;
            size_t j = i, best_end = i;
            while (j < n) {
                int32_t nx = -1;
                for (auto& kv : trie_[(size_t)node].kids) if (kv.first == t[j]) { nx = kv.second; break; }
                if (nx < 0) break;
                node = nx;
                ++j;
                if (trie_[(size_t)node].id >= 0) { best = trie_[(size_t)node].id; best_end = j; }
            }
            if (best >= 0) {
                encode_segment(text.substr(seg, i - seg), out);
                out.push_back(best);
                i = seg = best_end;
                continue;
            }
        }
        ++i;
    }
    encode_segment(text.substr(seg), out);
    return out;
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

std::string BpeTokenizer::decode_bytes(const std::vector<int32_t>& ids, const DecodeOptions& opt) const {
    const ByteTable& bt = byte_table();
    std::string out;
    for (int32_t id : ids) {
        if (opt.stop_at_eos && id == eos_id_) break;
        if (id < 0 || (size_t)id >= tokens_.size()) continue;
        uint8_t c = cls_[(size_t)id];
        if (c == 2 && !(opt.keep_tags || !opt.skip_special)) continue;
        if (c == 1 && opt.skip_special) continue;
        const std::string& s = tokens_[(size_t)id];
        if (c != 0) { out += s; continue; }
        const unsigned char* p = (const unsigned char*)s.data();
        size_t n = s.size(), i = 0;
        while (i < n) {
            uint32_t cp;
            int l = utf8_len(p + i, n - i, &cp);
            if (l == 0) { out += (char)p[i++]; continue; }
            if (cp < 324 && bt.from_cp[cp] >= 0) out += (char)bt.from_cp[cp];
            else out.append((const char*)p + i, (size_t)l);
            i += (size_t)l;
        }
    }
    return out;
}

std::string BpeTokenizer::decode(const std::vector<int32_t>& ids, const DecodeOptions& opt) const {
    return repair_utf8(decode_bytes(ids, opt));
}

// ---------------------------------------------------------------------------
// Prompt
// ---------------------------------------------------------------------------

const char* const kDefaultTranscriptionPrompt =
    "Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. "
    "Formatting: punctuation and capitalization. Output: transcript only.";
const char* const kAudioPlaceholder = "<|audio|>";

std::string render_transcription_prompt(const std::string& prompt, bool with_audio) {
    std::string s = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n";
    s += prompt;
    if (with_audio) { s += " "; s += kAudioPlaceholder; }
    s += "<|im_end|>\n<|im_start|>assistant\n<think></think>";
    return s;
}

TranscriptionPrompt build_transcription_prompt(const BpeTokenizer& tok, const std::string& prompt,
                                               int audio_rows) {
    TranscriptionPrompt r;
    if (audio_rows <= 0) {
        r.ids = tok.encode(render_transcription_prompt(prompt, false));
        return r;
    }
    int32_t aid = tok.token_id(kAudioPlaceholder);
    if (aid < 0) aid = kNemotronAudioTokenId;
    std::vector<int32_t> one = tok.encode(render_transcription_prompt(prompt, true));
    for (int32_t id : one) {
        if (id == aid && r.audio_begin < 0) {
            r.audio_begin = (int)r.ids.size();
            r.ids.insert(r.ids.end(), (size_t)audio_rows, aid);
        } else {
            r.ids.push_back(id);
        }
    }
    r.audio_rows = audio_rows;
    return r;
}

}  // namespace pk
