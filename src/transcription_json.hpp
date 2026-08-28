#pragma once

#include "transcription.hpp"

#include <string>
#include <vector>

namespace pk {

struct NBestTranscription;

void append_json_string(std::string& out, const std::string& s);
void append_json_int(std::string& out, int v);
void append_json_float(std::string& out, const char* fmt, float v);

// Serialize a Transcription to the C-API JSON document shape:
// {"text", "frame_sec", "words", "tokens"}.
std::string transcription_to_json(const Transcription& tr, float frame_sec);

// Serialize the offline TDT beam result:
// {"beam_size","score_norm","frame_sec","hypotheses":[...]}.
std::string nbest_transcriptions_to_json(
    const std::vector<NBestTranscription>& hypotheses,
    int beam_size, bool score_norm, float frame_sec);

} // namespace pk
