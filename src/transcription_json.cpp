#include "transcription_json.hpp"

#include <cstdio>

namespace pk {

// Append `s` to `out` as a JSON string literal (with surrounding quotes),
// escaping `"`, `\\`, and control characters (< 0x20) per RFC 8259. UTF-8
// multibyte sequences (>= 0x80) pass through verbatim.
void append_json_string(std::string& out, const std::string& s) {
    out += '"';
    char esc[8];
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    std::snprintf(esc, sizeof(esc), "\\u%04x", (unsigned)c);
                    out += esc;
                } else {
                    out += (char)c;
                }
        }
    }
    out += '"';
}

// Append an int to `out` as a bare JSON number.
void append_json_int(std::string& out, int v) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%d", v);
    out += buf;
}

// Append a float to `out` formatted with `fmt` (e.g. "%.3f"). NaN/Inf are
// emitted as 0 (JSON has no NaN/Inf literal); confidences/times are finite here.
void append_json_float(std::string& out, const char* fmt, float v) {
    char buf[32];
    if (!(v == v) || v > 1e30f || v < -1e30f) {  // NaN or huge -> 0
        out += '0';
        return;
    }
    std::snprintf(buf, sizeof(buf), fmt, v);
    out += buf;
}

// Serialize a pk::Transcription to the C-API JSON document (see the header doc
// on parakeet_capi_transcribe_path_json). Hand-rolled (no JSON library): times
// (word start/end, token t) with %.3f, confidences with %.4f.
std::string transcription_to_json(const Transcription& tr, float frame_sec) {
    std::string out;
    out.reserve(80 + tr.words.size() * 48 + tr.tokens.size() * 40);
    out += "{\"text\":";
    append_json_string(out, tr.text);
    // Encoder frame stride in seconds; lets consumers convert a frame-unit
    // segment gap threshold (NeMo segment_gap_threshold) to the seconds gap
    // between words when forming segments.
    out += ",\"frame_sec\":";
    append_json_float(out, "%.6f", frame_sec);
    out += ",\"words\":[";
    for (size_t i = 0; i < tr.words.size(); ++i) {
        if (i) out += ',';
        out += "{\"w\":";
        append_json_string(out, tr.words[i].text);
        out += ",\"start\":";
        append_json_float(out, "%.3f", tr.words[i].start);
        out += ",\"end\":";
        append_json_float(out, "%.3f", tr.words[i].end);
        out += ",\"conf\":";
        append_json_float(out, "%.4f", tr.words[i].conf);
        out += '}';
    }
    out += "],\"tokens\":[";
    for (size_t i = 0; i < tr.tokens.size(); ++i) {
        if (i) out += ',';
        out += "{\"id\":";
        append_json_int(out, tr.tokens[i].id);
        out += ",\"t\":";
        append_json_float(out, "%.3f", (float)tr.tokens[i].frame * frame_sec);
        out += ",\"conf\":";
        append_json_float(out, "%.4f", tr.tokens[i].conf);
        out += '}';
    }
    out += "]}";
    return out;
}

} // namespace pk
