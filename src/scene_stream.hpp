#pragma once
#include "asr_committer.hpp"
#include "diar_pcm_stream.hpp"
#include "speaker_identifier.hpp"  // pk::SlotName
#include "sas_merge.hpp"      // pk::SpeakerWord, pk::SpeakerUtterance
#include "sound_stream.hpp"   // pk::SoundOpts, pk::SoundSegment, pk::SoundWindow

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace pk {

class Model;
class CedTagger;

// The models a scene stream runs over. Any may be null; at least one is needed.
struct SceneParts {
    const Model* asr = nullptr;
    // Language for a prompt-conditioned ASR model ("" = the model default,
    // `auto` for the unified checkpoint). Ignored by models without a prompt.
    std::string asr_lang;
    const DiarizationModel* diar = nullptr;
    DiarLatency diar_latency = DiarLatency::Model;
    CedTagger* tagger = nullptr;     // sound events
    SoundOpts sound;                 // sound events
    SpeakerEmbed speaker_embed;      // speaker identification; needs diar and registry
    const SpeakerRegistry* registry = nullptr;   // borrowed, must outlive the stream
    SpeakerIdOpts speaker_opts;
};

// What one feed finalized. All times are seconds on the stream clock.
struct SceneUpdate {
    double t = 0;                    // stream time consumed
    // Renderer promise: no utterance, word or sound segment returned by a
    // later feed() can start before this. It does NOT cover `speakers`: an
    // already-open speaker segment can still close later with a start
    // earlier than safe_until (diarization does not give that bound, and
    // speaker segments are not the thing a renderer commits to the screen).
    double safe_until = 0;
    std::vector<SpeakerUtterance> utterances;
    std::vector<SpeakerWord> words;
    std::vector<SpeakerSegment> speakers;           // closed this call
    std::vector<SoundSegment> sounds;               // closed this call
    std::vector<StreamingSpeakerSegment> active_speakers;
    std::vector<SoundSegment> active_sounds;
    // Current identity of every diarization slot the speaker part has seen
    // (unknown slots have an empty name). Empty without a speaker model.
    std::map<int, SlotName> names;
    // True on every update of a stream that has a speaker part, even before
    // any slot is seen, so the JSON keeps one shape for the whole stream.
    bool named = false;
};

// The part running when feed() threw, so a caller can attribute the error.
enum class ScenePart { None, Diarization, Asr, Sound, Speaker };

// Speech, speakers and sound events over one live 16 kHz mono PCM stream.
// Each feed gives the PCM to every part, then collects what each one
// finalized. Not thread-safe.
//
// Error paths: feed() runs diarization, then the speaker part (when there
// is one), then ASR, then the sound part, in that order, and does not catch
// between them, so a part that throws loses the rest of that call. If the sound part throws, anything diarization or
// ASR already finalized in this call (including words the commit window
// released) is lost with it, not returned before the exception propagates.
// If ASR throws, the sound part for that chunk never runs (skipped, not
// deferred: it does not see that audio again). If a part throws on the
// is_last feed while diarization is present, finished() is already true
// (diarization takes the is_last chunk before ASR or sound run), so the
// stream ends without flushing whatever the throwing part (or anything
// after it) would otherwise have flushed on that final call.
//
// If the speaker part throws, ASR and sound for that chunk are skipped like
// any other part that throws.
//
// After a feed() that throws, later timestamps may be misaligned: the parts
// that did not see the failed chunk lag behind the ones that did. Callers
// should end the stream after an error rather than keep feeding it.
class SceneStream {
public:
    explicit SceneStream(const SceneParts& p);      // throws std::invalid_argument when no part is given,
                                                    // or a speaker part lacks diarization, a registry or valid options
    ~SceneStream();
    SceneStream(const SceneStream&) = delete;
    SceneStream& operator=(const SceneStream&) = delete;

    SceneUpdate feed(const float* pcm, int n, bool is_last);
    std::vector<SoundWindow> drain_windows();        // empty without a tagger
    // True after an is_last feed. With diarization it turns true as soon as
    // the diarizer takes the is_last chunk, so an is_last feed that throws
    // later (diarizer or transcriber) still ends the stream and is not re-run.
    bool finished() const { return finished_ || (diar_ && diar_->finished()); }
    const DiarPcmStream* diar() const { return diar_.get(); }
    // The part that was running when the last feed() threw.
    ScenePart failed_part() const { return part_; }

private:
    std::unique_ptr<DiarPcmStream> diar_;
    std::unique_ptr<AsrCommitter> asr_;
    std::unique_ptr<SoundStream> sound_;
    std::unique_ptr<SpeakerIdentifier> speaker_;
    std::vector<SpeakerSegment> segs_;   // closed diarization segments not yet behind the commit point
    double t_ = 0.0;                     // stream time consumed
    bool finished_ = false;
    ScenePart part_ = ScenePart::None;
};

// Serialize a SceneUpdate to the scene stream's JSON document shape:
// {"t","utterances","words","speakers","sounds",
//  "active":{"speakers","sounds"}}. `label(i)` may return nullptr (emitted
// as ""); pass a function that always returns nullptr when there is no
// tagger (sounds are then always empty, so it is never called).
std::string scene_update_to_json(const SceneUpdate& u, const std::function<const char*(int)>& label);

} // namespace pk
