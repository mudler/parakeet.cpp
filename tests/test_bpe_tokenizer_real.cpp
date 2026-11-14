// Parity of the byte-level BPE tokenizer with HF `tokenizers` on the real
// Nemotron 3.5 Transcribe tokenizer. Needs a tokenizer GGUF made by
// scripts/convert_bpe_tokenizer.py (path in PARAKEET_BPE_GGUF) and the fixture
// tests/fixtures/bpe_tokenizer_parity.tsv (made by
// scripts/gen_bpe_tokenizer_fixture.py). Skips (77) when the GGUF is absent.
#include "bpe_tokenizer.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static bool parse_json_string(const std::string& line, size_t* pos, std::string* out) {
    size_t i = *pos;
    if (i >= line.size() || line[i] != '"') return false;
    ++i;
    auto hex4 = [&](size_t at) { return (uint32_t)std::strtoul(line.substr(at, 4).c_str(), nullptr, 16); };
    auto put = [&](uint32_t cp) {
        if (cp < 0x80) *out += (char)cp;
        else if (cp < 0x800) { *out += (char)(0xC0 | (cp >> 6)); *out += (char)(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { *out += (char)(0xE0 | (cp >> 12)); *out += (char)(0x80 | ((cp >> 6) & 0x3F)); *out += (char)(0x80 | (cp & 0x3F)); }
        else { *out += (char)(0xF0 | (cp >> 18)); *out += (char)(0x80 | ((cp >> 12) & 0x3F)); *out += (char)(0x80 | ((cp >> 6) & 0x3F)); *out += (char)(0x80 | (cp & 0x3F)); }
    };
    while (i < line.size() && line[i] != '"') {
        char c = line[i];
        if (c != '\\') { *out += c; ++i; continue; }
        char e = line[i + 1];
        i += 2;
        switch (e) {
            case 'n': *out += '\n'; break;
            case 'r': *out += '\r'; break;
            case 't': *out += '\t'; break;
            case 'b': *out += '\b'; break;
            case 'f': *out += '\f'; break;
            case '/': *out += '/'; break;
            case '\\': *out += '\\'; break;
            case '"': *out += '"'; break;
            case 'u': {
                uint32_t cp = hex4(i);
                i += 4;
                if (cp >= 0xD800 && cp < 0xDC00 && line[i] == '\\' && line[i + 1] == 'u') {
                    uint32_t lo = hex4(i + 2);
                    i += 6;
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                }
                put(cp);
                break;
            }
            default: return false;
        }
    }
    *pos = i + 1;
    return true;
}

int main() {
    const char* gguf = std::getenv("PARAKEET_BPE_GGUF");
    if (!gguf || !*gguf) { std::fprintf(stderr, "test_bpe_tokenizer_real: PARAKEET_BPE_GGUF not set; skip\n"); return 77; }
    {
        std::ifstream probe(gguf, std::ios::binary);
        if (!probe) { std::fprintf(stderr, "test_bpe_tokenizer_real: %s not found; skip\n", gguf); return 77; }
    }
    const char* fx = std::getenv("PARAKEET_BPE_FIXTURE");
    std::string fixture = fx && *fx ? fx : "tests/fixtures/bpe_tokenizer_parity.tsv";

    std::string err;
    auto t0 = std::chrono::steady_clock::now();
    auto tok = pk::BpeTokenizer::load_gguf(gguf, &err);
    double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!tok) { std::fprintf(stderr, "load failed: %s\n", err.c_str()); return 1; }
    std::fprintf(stderr, "loaded %zu tokens in %.0f ms\n", tok->vocab_size(), load_ms);

    std::ifstream in(fixture);
    if (!in) { std::fprintf(stderr, "cannot open fixture %s\n", fixture.c_str()); return 1; }
    std::string line;
    size_t total = 0, match = 0, rt_ok = 0, nrows = 0;
    std::vector<int32_t> prompt_ids;
    std::string first_text;
    double enc_ms = 0;
    size_t enc_bytes = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::string text;
        size_t pos = 0;
        if (!parse_json_string(line, &pos, &text) || line[pos] != '\t') { std::fprintf(stderr, "bad fixture line %zu\n", total); return 1; }
        std::istringstream is(line.substr(pos + 1));
        std::vector<int32_t> want;
        long v;
        while (is >> v) want.push_back((int32_t)v);

        auto a = std::chrono::steady_clock::now();
        std::vector<int32_t> got = tok->encode(text);
        enc_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
        enc_bytes += text.size();
        ++total;
        if (got == want) ++match;
        else if (total - match <= 10) std::fprintf(stderr, "MISMATCH #%zu text=%.60s\n", total, text.c_str());
        pk::DecodeOptions all;
        all.skip_special = false;
        all.stop_at_eos = false;
        if (tok->decode(got, all) == text) ++rt_ok;
        else if (total - rt_ok <= 10) std::fprintf(stderr, "ROUNDTRIP FAIL #%zu text=%.60s\n", total, text.c_str());
        if (total == 1) { prompt_ids = want; first_text = text; }
        ++nrows;
    }
    std::printf("exact id match %zu/%zu, roundtrip %zu/%zu\n", match, total, rt_ok, total);
    std::printf("encode %.1f MB/s on the corpus (%zu bytes)\n", enc_bytes / 1e6 / (enc_ms / 1e3), enc_bytes);
    int rc = (match == total && rt_ok == total && total >= 500) ? 0 : 1;

    // The prompt builder must give the fixture ids for the default prompt, with
    // the audio placeholder expanded to the requested rows.
    pk::TranscriptionPrompt p1 = pk::build_transcription_prompt(*tok, pk::kDefaultTranscriptionPrompt, 1);
    if (p1.ids != prompt_ids || first_text != pk::render_transcription_prompt(pk::kDefaultTranscriptionPrompt, true)) {
        std::fprintf(stderr, "prompt ids differ from the fixture\n");
        rc = 1;
    }
    pk::TranscriptionPrompt p93 = pk::build_transcription_prompt(*tok, pk::kDefaultTranscriptionPrompt, 93);
    size_t n_au = 0;
    for (int32_t x : p93.ids) n_au += x == pk::kNemotronAudioTokenId;
    if (n_au != 93 || p93.ids.size() != p1.ids.size() + 92 || p93.audio_begin != p1.audio_begin) {
        std::fprintf(stderr, "audio row expansion wrong\n");
        rc = 1;
    }
    if (tok->eos_id() != pk::kNemotronEosId) { std::fprintf(stderr, "eos id is %d\n", tok->eos_id()); rc = 1; }

    // Speaker, event and style tags are kept or stripped on request.
    std::vector<int32_t> g = tok->encode("<spk:0> Hello there.<ev:breath> Bye.<sty:stt:flat>");
    g.push_back(pk::kNemotronEosId);
    g.push_back(1000);
    pk::DecodeOptions keep;
    keep.keep_tags = true;
    if (tok->decode(g, keep) != "<spk:0> Hello there.<ev:breath> Bye.<sty:stt:flat>" ||
        tok->decode(g) != " Hello there. Bye.") {
        std::fprintf(stderr, "tag handling wrong: [%s] [%s]\n", tok->decode(g, keep).c_str(), tok->decode(g).c_str());
        rc = 1;
    }

    // Long inputs: a 100 KB string must encode quickly.
    std::string big;
    const std::string unit = "The quick brown fox jumps over the lazy dog, 12345 times. Il perch\xC3\xA9 non \xC3\xA8 chiaro.\n";
    while (big.size() < 100000) big += unit;
    auto b0 = std::chrono::steady_clock::now();
    std::vector<int32_t> bi = tok->encode(big);
    double big_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - b0).count();
    std::string rt = tok->decode(bi);
    std::string worst(100000, 'a');                      // one word of 100000 symbols
    auto w0 = std::chrono::steady_clock::now();
    std::vector<int32_t> wi = tok->encode(worst);
    double worst_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
    std::printf("100 KB text: %zu ids in %.1f ms; 100 KB single word: %zu ids in %.1f ms\n", bi.size(), big_ms, wi.size(), worst_ms);
    if (rt != big || tok->decode(wi) != worst || big_ms > 1000 || worst_ms > 1000) {
        std::fprintf(stderr, "long input check failed\n");
        rc = 1;
    }
    (void)nrows;
    return rc;
}
