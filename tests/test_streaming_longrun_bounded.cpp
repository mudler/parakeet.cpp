#include "model.hpp"
#include "streaming.hpp"
#include "audio_io.hpp"
#include "tokenizer.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// MODEL: any cache-aware streaming model (e.g. nemotron-3.5-asr-streaming-0.6b
// or parakeet_realtime_eou_120m-v1).
// WORKING_DIRECTORY: the repo root (build/tests run from there).
//
// Regression for the O(N^2) long-session runaway: StreamingSession rebuilt the
// running text (detokenize over ALL non-special tokens) and the word grouping
// (group_words over ALL accumulated word tokens) on EVERY chunk, and neither
// buffer is trimmed across utterances. So the work done per chunk grows with the
// whole session, not with the chunk — O(N^2) total CPU and a steadily climbing
// RSS for a long-lived stream (e.g. a dictation session left open for hours).
//
// We drive a LONG stream (the fixture clip repeated many times — a multi-minute
// continuous session) and assert that the most expensive single chunk reprocess
// a bounded amount of history, NOT a fraction that scales with the total tokens.
// A correct (incremental) decoder keeps max_chunk_reprocess() ~ one chunk / open
// word; the buggy whole-history rebuild makes it ~ total tokens.
//
// Skips (77) unless PARAKEET_TEST_GGUF_STREAM is set (the streaming model is a
// large download, not in CI). Optional:
//   PARAKEET_TEST_STREAM_WAV     clip to repeat (default tests/fixtures/speech.wav)
//   PARAKEET_TEST_STREAM_LANG    language prompt for prompt models (default "")
//   PARAKEET_TEST_STREAM_REPEATS how many times to repeat the clip (default 24)
int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF_STREAM");
    if (!gguf) {
        std::fprintf(stderr,
            "test_streaming_longrun_bounded: PARAKEET_TEST_GGUF_STREAM not set; "
            "skip (streaming model is a large download, not in CI)\n");
        return 77;
    }
    const char* wav  = std::getenv("PARAKEET_TEST_STREAM_WAV");
    const char* lang = std::getenv("PARAKEET_TEST_STREAM_LANG");
    const char* rep  = std::getenv("PARAKEET_TEST_STREAM_REPEATS");
    std::string wav_path = wav ? wav : "tests/fixtures/speech.wav";
    std::string target_lang = lang ? lang : "";
    int repeats = rep ? std::atoi(rep) : 24;
    if (repeats < 4) repeats = 4;

    auto m = pk::Model::load(gguf);
    if (!m) { std::fprintf(stderr, "[longrun] load failed %s\n", gguf); return 1; }
    if (!m->config().streaming.present) {
        std::fprintf(stderr, "[longrun] model has no streaming config\n");
        return 1;
    }

    pk::Audio a;
    if (!pk::load_audio_16k_mono(wav_path, a)) {
        std::fprintf(stderr, "[longrun] audio load failed %s\n", wav_path.c_str());
        return 1;
    }

    // Build a long continuous session: the clip repeated `repeats` times, each
    // separated by 0.6 s of silence so an <EOU> fires between utterances (the
    // accumulated buffers are NOT reset on <EOU>, which is exactly the bug).
    std::vector<float> gap((size_t)(0.6f * 16000.0f), 0.0f);
    std::vector<float> pcm;
    pcm.reserve((a.samples.size() + gap.size()) * (size_t)repeats);
    for (int i = 0; i < repeats; ++i) {
        pcm.insert(pcm.end(), a.samples.begin(), a.samples.end());
        pcm.insert(pcm.end(), gap.begin(), gap.end());
    }

    pk::StreamingSession sess(m->loader(), target_lang);
    sess.reset_instrumentation();
    pk::run_stream_over_pcm(sess, m->loader(), pcm);

    const size_t total   = sess.tokens().size();        // tokens over the session
    const size_t hottest = sess.max_chunk_reprocess();  // worst single-chunk work

    std::fprintf(stderr,
        "[longrun] repeats=%d  total_tokens=%zu  max_chunk_reprocess=%zu\n",
        repeats, total, hottest);

    // PARITY (public API only): the incremental running transcript must be
    // byte-for-byte identical to a full detokenize of every non-special token the
    // session emitted (i.e. what the pre-fix whole-history rebuild produced).
    const auto& pieces = m->loader().config().tokenizer_pieces;
    int eou_id = -1, eob_id = -1;
    for (int i = 0; i < (int)pieces.size(); ++i) {
        if (pieces[i] == "<EOU>") eou_id = i;
        else if (pieces[i] == "<EOB>") eob_id = i;
    }
    std::vector<int32_t> non_special;
    non_special.reserve(sess.tokens().size());
    for (int32_t t : sess.tokens())
        if (t != eou_id && t != eob_id) non_special.push_back(t);
    const std::string full_text = pk::detokenize(pieces, non_special);
    if (sess.text() != full_text) {
        std::fprintf(stderr,
            "[longrun] FAIL: incremental text != full detokenize\n  inc (%zu): %.120s\n  full(%zu): %.120s\n",
            sess.text().size(), sess.text().c_str(), full_text.size(), full_text.c_str());
        return 1;
    }
    std::fprintf(stderr,
        "[longrun] parity OK: incremental text (%zu B) == full detokenize of %zu non-special tokens\n",
        sess.text().size(), non_special.size());

    // Sanity: the repeated clip must produce a substantial token stream, else the
    // signal is too small to be meaningful (bump PARAKEET_TEST_STREAM_REPEATS).
    if (total < 60) {
        std::fprintf(stderr,
            "[longrun] only %zu tokens emitted; increase PARAKEET_TEST_STREAM_REPEATS\n",
            total);
        return 1;
    }

    // The bug: the hottest chunk reprocesses ~the whole session history, so
    // max_chunk_reprocess scales with total tokens. Incremental decoding keeps it
    // proportional to one chunk's new tokens (plus the open word) — far below the
    // total. Require the hottest chunk to touch less than a third of the session.
    if (hottest * 3 >= total) {
        std::fprintf(stderr,
            "[longrun] FAIL: hottest chunk reprocessed %zu of %zu total tokens "
            "(O(N^2): per-chunk work scales with the whole session)\n",
            hottest, total);
        return 1;
    }

    std::fprintf(stderr,
        "[longrun] OK: per-chunk reprocess bounded (%zu << %zu)\n", hottest, total);
    return 0;
}
