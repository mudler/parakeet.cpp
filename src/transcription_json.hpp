#pragma once

#include "transcription.hpp"

#include <string>

namespace pk {

void append_json_string(std::string& out, const std::string& s);
void append_json_int(std::string& out, int v);
void append_json_float(std::string& out, const char* fmt, float v);

// Serialize a Transcription to the C-API JSON document shape:
// {"text", "frame_sec", "words", "tokens"}.
std::string transcription_to_json(const Transcription& tr, float frame_sec);

} // namespace pk
