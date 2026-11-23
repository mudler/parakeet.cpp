#include "scene_stream.hpp"

#include "ced_tagger.hpp"  // pk::CedTagger
#include "model.hpp"       // pk::Model
#include "transcription_json.hpp"

#include <algorithm>
#include <stdexcept>

namespace pk {

namespace {
// Digital silence put before a loud window that gave no word, for models
// without durations (see AsrCommitterOpts::empty_retry_pad_sec).
constexpr double kEmptyRetryPadSec = 0.16;
// A pause at least this long after the last committed word moves the resume
// point into the pause (AsrCommitterOpts::quiet_resume_min_gap_sec).
constexpr double kQuietResumeMinGapSec = 0.5;
// Shortest part when such a window is split (AsrCommitterOpts::empty_split_min_sec).
constexpr double kEmptySplitMinSec = 2.0;
}  // namespace

SceneStream::SceneStream(const SceneParts& p) {
    if (p.speaker_embed && !p.diar)
        throw std::invalid_argument("speaker identification needs a diarization model");
    if (p.speaker_embed && !p.registry)
        throw std::invalid_argument("speaker identification needs a registry");
    if (p.speaker_embed) {
        const std::string err = validate_speaker_opts(p.speaker_opts);
        if (!err.empty()) throw std::invalid_argument("invalid speaker options: " + err);
    }
    if (!p.asr && !p.diar && !p.tagger)
        throw std::invalid_argument("scene stream needs at least one model");
    if (p.speaker_embed)
        speaker_ = std::make_unique<SpeakerIdentifier>(p.speaker_embed, p.registry, p.speaker_opts);
    if (p.diar) diar_ = std::make_unique<DiarPcmStream>(*p.diar, p.diar_latency);
    if (p.asr) {
        const Model* m = p.asr;
        // A model without durations (RNN-T) reports late, one-frame word
        // times, and the offline decode of the unified checkpoint can return
        // nothing for a window cut out of running speech. Start each window
        // in a pause and retry a loud window that gave no word; report
        // acoustic word times for the unified checkpoint. TDT and CTC models
        // keep the plain committer.
        AsrCommitterOpts opts;
        if (m->word_times_from_emission()) {
            const ParakeetConfig& cfg = m->config();
            const float frame_sec =
                (float)cfg.hop_length * (float)cfg.subsampling_factor / (float)cfg.sample_rate;
            opts.quiet_resume_min_gap_sec = kQuietResumeMinGapSec;
            opts.empty_retry_pad_sec = kEmptyRetryPadSec;
            opts.empty_split_min_sec = kEmptySplitMinSec;
            // The time correction was measured on the unified checkpoint only.
            if (cfg.unified.present)
                opts.output_times = [frame_sec](std::vector<Word>& w) { refine_emission_word_times(w, frame_sec); };
        }
        asr_ = std::make_unique<AsrCommitter>(
            [m](const std::vector<float>& x) { return m->transcribe_with_timestamps(x, 16000).words; },
            4.0, 1.0, std::move(opts));
    }
    if (p.tagger)
        sound_ = std::make_unique<SoundStream>(p.tagger->scorer(), p.tagger->n_classes(), p.sound);
}

SceneStream::~SceneStream() = default;

SceneUpdate SceneStream::feed(const float* pcm, int n, bool is_last) {
    SceneUpdate u;
    std::vector<StreamingSpeakerSegment> closed;
    long long advanced = 0;
    // Until the ASR transcribes, a failure is charged to diarization (as the
    // SAS C-API always did), including the ASR audio push.
    part_ = diar_ ? ScenePart::Diarization : ScenePart::Asr;
    if (diar_) {
        advanced = diar_->feed(pcm, n, is_last, closed);
        for (const auto& c : closed) {
            segs_.push_back({c.speaker, c.start, c.end});
            u.speakers.push_back({c.speaker, c.start, c.end});
        }
    }
    if (speaker_) {
        part_ = ScenePart::Speaker;
        speaker_->push_pcm(pcm, n);
        std::vector<SpeakerSegment> closed_segs, open_segs;
        for (const auto& c : closed) closed_segs.push_back({c.speaker, c.start, c.end});
        for (const auto& o : diar_->open_segments()) open_segs.push_back({o.speaker, o.start, o.end});
        speaker_->update(closed_segs, open_segs, is_last);
        u.names = speaker_->names();
        u.named = true;
        // A failure past this point is charged to the next part.
        part_ = asr_ ? ScenePart::Asr : ScenePart::Diarization;
    }
    if (asr_) {
        asr_->push(pcm, n);
        // With diarization, ASR follows how far diarization has got, and only
        // when it advanced (the SAS behavior). Without it, the audio received.
        // A commit that the min window skips changes nothing, not even the
        // segment pruning below (the SAS behavior).
        const bool tick = !diar_ || advanced > 0 || is_last;
        const double until = diar_ ? diar_->diarized_until() : t_ + (n > 0 ? n : 0) / 16000.0;
        if (tick && asr_->ready(until, is_last)) {
            part_ = ScenePart::Asr;
            auto committed = asr_->commit(until, is_last);
            // Speaker segments known so far: closed ones plus those still open.
            std::vector<SpeakerSegment> segs = segs_;
            if (diar_)
                for (const auto& o : diar_->open_segments()) segs.push_back({o.speaker, o.start, o.end});
            u.words = merge_asr_diarization(committed, segs);
            if (speaker_)
                for (SpeakerWord& w : u.words) {
                    const SlotName sn = speaker_->name(w.speaker);
                    w.name = sn.name;
                    w.name_score = sn.score;
                }
            u.utterances = group_speaker_words(u.words);
            // Segments that ended before the commit point can no longer match a word.
            const double commit_sec = asr_->commit_sec();
            segs_.erase(std::remove_if(segs_.begin(), segs_.end(),
                                       [&](const SpeakerSegment& g) { return g.end < commit_sec; }),
                        segs_.end());
        }
    }
    if (sound_) {
        part_ = ScenePart::Sound;
        u.sounds = sound_->feed(pcm, n, is_last);
        u.active_sounds = sound_->open_segments();
    }
    t_ += (n > 0 ? n : 0) / 16000.0;
    if (is_last) finished_ = true;
    u.t = t_;
    if (diar_) {
        part_ = ScenePart::Diarization;
        u.active_speakers = diar_->open_segments();
    }
    if (finished_) {
        u.safe_until = t_;
    } else if (sound_ && asr_) {
        u.safe_until = std::min(asr_->commit_sec(), sound_->safe_until());
    } else if (sound_) {
        u.safe_until = sound_->safe_until();
    } else if (asr_) {
        u.safe_until = asr_->commit_sec();
    } else {
        u.safe_until = t_;
    }
    part_ = ScenePart::None;
    return u;
}

std::vector<SoundWindow> SceneStream::drain_windows() {
    return sound_ ? sound_->drain_windows() : std::vector<SoundWindow>{};
}

namespace {

// ,"name":"alice","name_score":0.7100 : only when a speaker model ran (`named`).
void append_name(std::string& out, bool named, const std::map<int, SlotName>& names, int slot,
                 const std::string& own_name, float own_score, bool use_own) {
    if (!named) return;
    std::string n = own_name;
    float sc = own_score;
    if (!use_own) {
        auto it = names.find(slot);
        n = it == names.end() ? std::string() : it->second.name;
        sc = it == names.end() ? 0.0f : it->second.score;
    }
    out += ",\"name\":";        append_json_string(out, n);
    out += ",\"name_score\":";  append_json_float(out, "%.4f", sc);
}

void append_speaker_segment(std::string& out, const SpeakerSegment& s,
                            bool named, const std::map<int, SlotName>& names) {
    out += "{\"speaker\":"; append_json_int(out, s.speaker);
    append_name(out, named, names, s.speaker, std::string(), 0.0f, false);
    out += ",\"start\":";   append_json_float(out, "%.3f", s.start);
    out += ",\"end\":";     append_json_float(out, "%.3f", s.end);
    out += "}";
}

void append_active_speaker(std::string& out, const StreamingSpeakerSegment& s,
                           bool named, const std::map<int, SlotName>& names) {
    out += "{\"speaker\":"; append_json_int(out, s.speaker);
    append_name(out, named, names, s.speaker, std::string(), 0.0f, false);
    out += ",\"start\":";   append_json_float(out, "%.3f", s.start);
    out += "}";
}

std::string utterances_to_json(const std::vector<SpeakerUtterance>& utts,
                               bool named, const std::map<int, SlotName>& names) {
    std::string out = "[";
    for (size_t i = 0; i < utts.size(); ++i) {
        if (i) out += ",";
        out += "{\"speaker\":"; append_json_int(out, utts[i].speaker);
        append_name(out, named, names, utts[i].speaker, utts[i].name, utts[i].name_score, true);
        out += ",\"text\":";    append_json_string(out, utts[i].text);
        out += ",\"start\":";   append_json_float(out, "%.3f", utts[i].start);
        out += ",\"end\":";     append_json_float(out, "%.3f", utts[i].end);
        out += ",\"conf\":";    append_json_float(out, "%.4f", utts[i].conf);
        out += "}";
    }
    return out + "]";
}

std::string words_to_json(const std::vector<SpeakerWord>& words,
                          bool named, const std::map<int, SlotName>& names) {
    std::string out = "[";
    for (size_t i = 0; i < words.size(); ++i) {
        if (i) out += ",";
        out += "{\"text\":";    append_json_string(out, words[i].text);
        out += ",\"start\":";   append_json_float(out, "%.3f", words[i].start);
        out += ",\"end\":";     append_json_float(out, "%.3f", words[i].end);
        out += ",\"conf\":";    append_json_float(out, "%.4f", words[i].conf);
        out += ",\"speaker\":"; append_json_int(out, words[i].speaker);
        append_name(out, named, names, words[i].speaker, words[i].name, words[i].name_score, true);
        out += "}";
    }
    return out + "]";
}

std::string speakers_to_json(const std::vector<SpeakerSegment>& segs,
                             bool named, const std::map<int, SlotName>& names) {
    std::string out = "[";
    for (size_t i = 0; i < segs.size(); ++i) {
        if (i) out += ",";
        append_speaker_segment(out, segs[i], named, names);
    }
    return out + "]";
}

}  // namespace

std::string scene_update_to_json(const SceneUpdate& u, const std::function<const char*(int)>& label) {
    std::string out = "{\"t\":";
    append_json_float(out, "%.3f", (float)u.t);
    // No speaker part: no names at all (the shape from before speaker
    // identification). With one: names on every update, possibly empty.
    const bool named = u.named || !u.names.empty();
    if (named) {
        out += ",\"names\":{";
        bool first = true;
        for (const auto& kv : u.names) {
            if (!first) out += ",";
            first = false;
            out += "\"" + std::to_string(kv.first) + "\":{\"name\":";
            append_json_string(out, kv.second.name);
            out += ",\"score\":";
            append_json_float(out, "%.4f", kv.second.score);
            out += "}";
        }
        out += "}";
    }
    out += ",\"utterances\":" + utterances_to_json(u.utterances, named, u.names);
    out += ",\"words\":" + words_to_json(u.words, named, u.names);
    out += ",\"speakers\":" + speakers_to_json(u.speakers, named, u.names);
    out += ",\"sounds\":" + sound_segments_to_json(u.sounds, label);
    out += ",\"active\":{\"speakers\":[";
    for (size_t i = 0; i < u.active_speakers.size(); ++i) {
        if (i) out += ",";
        append_active_speaker(out, u.active_speakers[i], named, u.names);
    }
    out += "],\"sounds\":" + sound_segments_to_json(u.active_sounds, label);
    out += "}}";
    return out;
}

} // namespace pk
