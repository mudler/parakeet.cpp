#pragma once
#include "transcription.hpp"  // pk::Word

#include <functional>
#include <vector>

namespace pk {

// Offline transcription of one 16 kHz mono PCM window; word times are
// relative to the window start.
using Transcriber = std::function<std::vector<Word>(const std::vector<float>& pcm)>;

// Optional guards for models whose offline decode is fragile on a window that
// opens in the middle of speech. All are off by default, and with the default
// options the committer behaves exactly as before.
struct AsrCommitterOpts {
    // When the next word heard starts at least this long after the end of the
    // last committed word (a pause), resume the next window at the quietest
    // point of that pause (at most 1 s into it), not at the end of
    // the last committed word (0 = off). For a model whose word ends are late
    // (an RNN-T model with no durations): the end then lands on the tail of
    // the word, and the next window opens on a word fragment.
    double quiet_resume_min_gap_sec = 0.0;
    // When a window that is not silent gives no word, transcribe it once more
    // with this much digital silence before it (0 = off). Word times stay
    // relative to the real audio.
    double empty_retry_pad_sec = 0.0;
    // When the window still gives no word, split it once, at its quietest
    // point in the middle third, and transcribe the two parts on their own;
    // each part is at least this long (0 = off).
    // Bound of both retries together: at most 4 transcriber calls and 3 times
    // the window audio per window. A window that overlaps the previous one,
    // when that one stayed empty after its retries, gets no retry.
    double empty_split_min_sec = 0.0;
    // Applied to a copy of each window's words before they are committed, to
    // change the reported times only. The commit point and the resume point
    // still use the transcriber's own times.
    std::function<void(std::vector<Word>&)> output_times;
};

// Streaming text commit over an offline transcriber: buffers uncommitted PCM,
// transcribes it once enough has built up, and commits only the words that
// have right context. The rest is transcribed again with the next window.
// When a window holds no committable word, the audio before the first word
// heard (or before the right context) is released, so non-speech does not
// build up.
class AsrCommitter {
public:
    explicit AsrCommitter(Transcriber t, double min_window_sec = 4.0, double right_context_sec = 1.0,
                          AsrCommitterOpts opts = {});
    void push(const float* pcm, int n);
    // Words committed up to stream time `until` (absolute times). Empty when
    // fewer than min_window_sec of uncommitted audio is available (unless is_last).
    std::vector<Word> commit(double until, bool is_last);
    // True when commit(until, is_last) would transcribe (the window is long
    // enough, or is_last); false when it would return early.
    bool ready(double until, bool is_last) const;
    // Stream time of the first uncommitted sample.
    double commit_sec() const { return commit_sec_; }
    // Uncommitted samples held (starting at commit_sec()).
    size_t buffered_samples() const { return audio_.size(); }

private:
    size_t span(double until, bool is_last) const;
    std::vector<Word> transcribe_window(size_t span);
    double quiet_point(double from, double to) const;

    Transcriber transcribe_;
    double min_window_sec_;
    double right_context_sec_;
    AsrCommitterOpts opts_;
    std::vector<float> audio_;   // uncommitted PCM, starting at commit_sec_
    double commit_sec_ = 0.0;    // stream time of audio_[0]
    double empty_until_ = -1.0;  // end (stream time) of the last window that stayed empty after its retries
    Word last_;                  // last committed word (absolute times)
    bool have_last_ = false;
};

} // namespace pk
