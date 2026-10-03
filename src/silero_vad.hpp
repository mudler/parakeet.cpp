#pragma once
// Silero VAD (https://github.com/snakers4/silero-vad, MIT licence) as a small
// standalone ggml model. One GGUF holds the 16 kHz and the 8 kHz weight sets
// (schema in docs/conversion.md, usage in docs/vad.md).
//
// Time base: the model reads fixed chunks of 512 samples at 16 kHz or 256
// samples at 8 kHz. Both are 32 ms. Each chunk gives one speech probability, so
// probability i describes the audio in [i * 0.032 s, (i + 1) * 0.032 s). A
// segmenter can use kSileroFrameSec as its frame period.
//
// Threading: a loaded SileroVad is read-only and can be shared. All streaming
// state lives in a Stream object; use one Stream per audio stream and one
// thread at a time per Stream. Compute goes through the process backend, which
// serialises graph runs, so several threads are safe but they queue.
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;
struct ggml_context;
struct ggml_backend_buffer;

namespace pk {

// Seconds (and milliseconds) of audio behind one probability, at every rate.
constexpr double kSileroFrameSec = 0.032;
constexpr int kSileroFrameMs = 32;

namespace silero {

// Fixed geometry of one sample rate. The model only exists for these two.
struct RateParams {
    int sample_rate;
    int chunk;    // new samples per model call
    int context;  // trailing samples of the previous input, prepended
    int rpad;     // right reflect padding
    int n_fft;    // STFT window
    int hop;      // STFT hop
    int input_len() const { return context + chunk + rpad; }
};
// nullptr for any rate other than 16000 and 8000.
const RateParams* rate_params(int sample_rate);

// x[base + i] = x[base - 2 - i] for i in [0, rpad): reflection without the edge
// sample, like torch's F.pad(mode="reflect"). Needs base >= rpad + 1.
void reflect_pad_right(float* x, int base, int rpad);

// Cuts an arbitrary-sized sample stream into model inputs. Pure and model free.
// Every input is context | chunk | reflect padding; the context is the last
// `context` samples of the previous context | chunk (zeros at the start).
class Framer {
public:
    explicit Framer(const RateParams& p);
    void reset();
    void push(const float* pcm, size_t n);
    size_t pending() const { return buf_.size() - off_; }
    // Writes one input (p.input_len() floats) and consumes one chunk. False when
    // fewer than `chunk` samples are pending.
    bool next(float* input);
    // Same for a trailing partial chunk: zero-pads it to a full chunk. False when
    // nothing is pending.
    bool next_padded(float* input);

private:
    void emit(const float* chunk, float* input);
    RateParams p_;
    std::vector<float> ctx_, buf_;
    size_t off_ = 0;
};

}  // namespace silero

class SileroVad {
public:
    // Returns nullptr on failure and writes a one-line reason to *err (when not null).
    // With a non-empty `component`, `gguf_path` is a bundle GGUF (docs/bundle.md):
    // the Silero weights are read from that component only, and no other
    // component's tensor data is read.
    static std::unique_ptr<SileroVad> load(const std::string& gguf_path, std::string* err = nullptr,
                                           const std::string& component = "");
    ~SileroVad();
    SileroVad(const SileroVad&) = delete;
    SileroVad& operator=(const SileroVad&) = delete;

    // Sample rates in the file (8000 and 16000 for the official model).
    const std::vector<int>& sample_rates() const { return rates_; }
    bool supports(int sample_rate) const;
    // Samples per chunk: 512 at 16 kHz, 256 at 8 kHz. 0 for an unsupported rate.
    int chunk_samples(int sample_rate) const;
    // Seconds of audio per probability: always kSileroFrameSec. 0 if unsupported.
    double chunk_sec(int sample_rate) const;

    // Streaming state for one audio stream: input buffer, context and LSTM
    // state. The SileroVad that made it must outlive it.
    class Stream {
    public:
        Stream() = default;
        // False for a stream made for an unsupported sample rate.
        bool valid() const { return model_ != nullptr && !failed_; }
        int sample_rate() const { return sr_; }
        // Back to the initial state (empty buffer, zero context and LSTM state).
        void reset();
        // Feeds any number of samples in [-1, 1] (0 is fine). Appends one
        // probability to *probs for every full chunk completed. Returns false on
        // failure; the stream is then unusable until reset().
        bool process_chunk(const float* pcm, size_t n, std::vector<float>* probs);
        // Zero-pads the buffered partial chunk, if any, and appends its
        // probability. Call at the end of the audio. Afterwards the stream
        // continues from the padded position, so call reset() to start a new clip.
        bool flush(std::vector<float>* probs);
        // Samples buffered but not yet part of a probability.
        size_t pending_samples() const { return framer_.pending(); }

    private:
        friend class SileroVad;
        Stream(const SileroVad* m, const silero::RateParams* p);
        bool run(const float* input, std::vector<float>* probs);
        const SileroVad* model_ = nullptr;
        int sr_ = 0;
        silero::Framer framer_{*silero::rate_params(16000)};
        std::vector<float> in_, h_, c_;
        bool failed_ = false;
    };
    Stream new_stream(int sample_rate) const;

    // Whole clip from a fresh state; the last partial chunk is zero-padded like
    // the official audio_forward(). Returns ceil(n / chunk) values, or an empty
    // vector for an unsupported rate or on failure.
    std::vector<float> probabilities(const float* pcm, size_t n, int sample_rate) const;

private:
    SileroVad() = default;
    struct RateWeights {
        silero::RateParams p{};
        ggml_tensor* stft = nullptr;
        ggml_tensor* enc_w[4] = {};
        ggml_tensor* enc_b[4] = {};
        ggml_tensor *w_ih = nullptr, *w_hh = nullptr, *b_ih = nullptr, *b_hh = nullptr;
        ggml_tensor *w_out = nullptr, *b_out = nullptr;
    };
    const RateWeights* find(int sr) const;
    // One model call on a full input; updates h and c (128 floats each).
    bool run_chunk(const RateWeights& w, const float* input, std::vector<float>& h, std::vector<float>& c,
                   float* prob) const;

    std::vector<int> rates_;
    std::vector<RateWeights> w_;
    ggml_context* ctx_ = nullptr;          // tensor metadata, F32 weights
    ggml_backend_buffer* buf_ = nullptr;   // weights
};

}  // namespace pk
