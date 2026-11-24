#include "asr_committer.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <utility>

namespace pk {

namespace {

// Parakeet's word start times come from the frame where the first token is
// emitted, often one or two 80 ms encoder frames after the sound actually
// starts, so cutting exactly at a word start can permanently drop the tail
// end of the previous window's onset. Back off by this much when no word
// commits, so the next window re-hears that onset with full context.
constexpr double kOnsetMargin = 0.3;

// quiet_resume: energy frames used to find the quietest point.
constexpr size_t kQuietFrame = 640;   // 40 ms
constexpr size_t kQuietHop = 160;     // 10 ms
constexpr double kQuietSearchSec = 1.0;

// empty_retry_pad_sec: a window counts as silent (no retry) when no 40 ms
// frame reaches this RMS (about -46 dBFS).
constexpr double kSilenceRms = 0.005;

// Lowercase letters and digits only, for comparing a word heard twice.
std::string word_key(const std::string& w) {
    std::string k;
    for (unsigned char c : w)
        if (std::isalnum(c) || c >= 0x80) k += (char)std::tolower(c);
    return k;
}

}  // namespace

AsrCommitter::AsrCommitter(Transcriber t, double min_window_sec, double right_context_sec,
                           AsrCommitterOpts opts)
    : transcribe_(std::move(t)), min_window_sec_(min_window_sec), right_context_sec_(right_context_sec),
      opts_(opts) {}

// True when some 40 ms frame of pcm[a, b) reaches kSilenceRms.
static bool loud(const std::vector<float>& pcm, size_t a, size_t b) {
    for (size_t i = a; i + kQuietFrame <= b; i += kQuietHop) {
        double e = 0.0;
        for (size_t j = i; j < i + kQuietFrame; ++j) e += (double)pcm[j] * pcm[j];
        if (std::sqrt(e / kQuietFrame) >= kSilenceRms) return true;
    }
    return false;
}

// Transcribe audio_[0, span). An offline model can return nothing for a
// window cut out of running speech, even when NeMo runs the same window. With
// the empty_* options a window that gives no word but is not silent is tried
// again: once with digital silence before it, then once split in two at its
// quietest point near the middle, each part transcribed alone. That is at
// most 4 transcriber calls and 3 times the window audio (plus the pad). A
// window that overlaps the previous window, when that one stayed empty after
// its retries, is not retried: a long stretch of loud non-speech costs the
// retries once, not once per window.
std::vector<Word> AsrCommitter::transcribe_window(size_t span) {
    std::vector<float> window(audio_.begin(), audio_.begin() + span);
    std::vector<Word> words = transcribe_(window);
    const double end_sec = commit_sec_ + (double)span / 16000.0;
    const bool retry = opts_.empty_retry_pad_sec > 0.0 || opts_.empty_split_min_sec > 0.0;
    if (!words.empty() || !retry) {
        empty_until_ = -1.0;
        return words;
    }
    if (!loud(window, 0, span)) {   // silence ends the empty stretch
        empty_until_ = -1.0;
        return words;
    }
    if (empty_until_ > commit_sec_) {   // the previous window, over this audio, stayed empty
        empty_until_ = end_sec;
        return words;
    }
    if (opts_.empty_retry_pad_sec > 0.0) {
        const size_t pad = (size_t)std::llround(opts_.empty_retry_pad_sec * 16000.0);
        std::vector<float> padded(pad, 0.0f);
        padded.insert(padded.end(), window.begin(), window.end());
        words = transcribe_(padded);
        const float shift = (float)pad / 16000.0f;
        for (Word& w : words) {
            w.start = std::max(0.0f, w.start - shift);
            w.end = std::max(w.start, w.end - shift);
        }
    }
    const double len = (double)span / 16000.0;
    const double min_part = opts_.empty_split_min_sec;
    if (words.empty() && min_part > 0.0 && len >= 2.0 * min_part) {
        // Cut in the middle third, leaving each part at least min_part long.
        const double lo = std::max(min_part, len / 3.0), hi = std::min(len - min_part, 2.0 * len / 3.0);
        const size_t cut = std::min(span, (size_t)std::llround(quiet_point(lo, hi) * 16000.0));
        if (cut > 0 && cut < span) {
            words = transcribe_(std::vector<float>(window.begin(), window.begin() + cut));
            std::vector<Word> tail = transcribe_(std::vector<float>(window.begin() + cut, window.end()));
            const float off = (float)cut / 16000.0f;
            for (Word& w : tail) { w.start += off; w.end += off; }
            words.insert(words.end(), tail.begin(), tail.end());
        }
    }
    empty_until_ = words.empty() ? end_sec : -1.0;
    return words;
}

// Start (seconds from audio_[0]) of the quietest 40 ms of audio in [from, to].
// Returns `from` when the range holds no full frame.
double AsrCommitter::quiet_point(double from, double to) const {
    const size_t a = (size_t)std::max(0.0, std::floor(from * 16000.0));
    const size_t b = std::min(audio_.size(), (size_t)std::max(0.0, std::floor(to * 16000.0)));
    double best_e = -1.0;
    size_t best = a;
    for (size_t i = a; i + kQuietFrame <= b; i += kQuietHop) {
        double e = 0.0;
        for (size_t j = i; j < i + kQuietFrame; ++j) e += (double)audio_[j] * audio_[j];
        if (best_e < 0.0 || e < best_e) { best_e = e; best = i; }
    }
    return best_e < 0.0 ? from : (double)best / 16000.0;
}

void AsrCommitter::push(const float* pcm, int n) {
    if (n > 0) audio_.insert(audio_.end(), pcm, pcm + n);
}

// Uncommitted samples up to `until` (all of them with is_last).
size_t AsrCommitter::span(double until, bool is_last) const {
    return is_last ? audio_.size()
                   : std::min(audio_.size(), (size_t)std::max(0.0, (until - commit_sec_) * 16000.0));
}

// Offline ASR on a short window loses words, so wait until enough
// uncommitted audio has built up (it bounds how often the text commits).
bool AsrCommitter::ready(double until, bool is_last) const {
    return is_last || span(until, is_last) >= (size_t)(min_window_sec_ * 16000.0);
}

std::vector<Word> AsrCommitter::commit(double until, bool is_last) {
    if (!ready(until, is_last)) return {};
    const size_t span = this->span(until, is_last);
    std::vector<Word> words;
    if (span > 0) words = transcribe_window(span);
    // Commit only words that end right_context_sec before the cut: the ASR
    // needs right context, and a word at the edge may be cut in half.
    size_t keep = words.size();
    double next_commit = commit_sec_ + (double)span / 16000.0;
    if (!is_last) {
        const double limit = (double)span / 16000.0 - right_context_sec_;
        keep = 0;
        while (keep < words.size() && words[keep].end <= limit) ++keep;
        if (keep > 0) {
            // Resume right after the last committed word: audio the ASR
            // skipped this time is heard again with more context.
            double resume = words[keep - 1].end;
            // quiet_resume_min_gap_sec: in a pause before the next word
            // heard, start the next window at the quietest point, not on the
            // tail of the last committed word. Never search past the next
            // word heard, or further than kQuietSearchSec: audio the model
            // skipped must be heard again.
            if (opts_.quiet_resume_min_gap_sec > 0.0 && keep < words.size()) {
                const double next = std::min(limit, (double)words[keep].start);
                if (next - resume >= opts_.quiet_resume_min_gap_sec)
                    resume = quiet_point(resume, std::min(next, resume + kQuietSearchSec));
            }
            next_commit = commit_sec_ + resume;
        } else {
            // No committable word (silence, music, long non-speech). Release
            // the audio before the first word heard, or before the right
            // context when there is none, so the buffer and the cost of each
            // transcription stay bounded. Back off by the onset margin so the
            // first uncommitted word's onset is re-heard with full context
            // next time, not cut at its (possibly late) reported start.
            double first_start = limit;
            if (!words.empty()) first_start = std::min(first_start, (double)words.front().start);
            const double rel = std::max(0.0, first_start - kOnsetMargin);
            next_commit = commit_sec_ + rel;
        }
    }
    if (opts_.output_times) opts_.output_times(words);
    std::vector<Word> committed(words.begin(), words.begin() + keep);
    for (auto& w : committed) { w.start += (float)commit_sec_; w.end += (float)commit_sec_; }
    // ASR timestamps are only accurate to a frame or two, so the tail of the
    // previously committed word can be heard again at the new start.
    if (have_last_ && !committed.empty() &&
        word_key(committed.front().text) == word_key(last_.text) &&
        committed.front().start - last_.start < 0.5f)
        committed.erase(committed.begin());
    if (!committed.empty()) { last_ = committed.back(); have_last_ = true; }
    const size_t drop = std::min(audio_.size(),
                                 (size_t)std::llround((next_commit - commit_sec_) * 16000.0));
    audio_.erase(audio_.begin(), audio_.begin() + drop);
    commit_sec_ += (double)drop / 16000.0;
    return committed;
}

} // namespace pk
