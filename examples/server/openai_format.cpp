#include "openai_format.hpp"

#include <cstdio>

namespace pkserver {

bool parse_format(const std::string& s, Format& out) {
    if (s.empty() || s == "json") { out = Format::kJson; return true; }
    if (s == "text")              { out = Format::kText; return true; }
    if (s == "verbose_json")      { out = Format::kVerboseJson; return true; }
    return false;
}

std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\b': o += "\\b";  break;
            case '\f': o += "\\f";  break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    o += buf;
                } else {
                    o += (char)c;
                }
        }
    }
    return o;
}

static std::string fixed(double v, int prec) {
    // JSON has no NaN/Inf literal; emit 0 instead, matching the repo's
    // append_json_float convention in src/transcription_json.cpp.
    if (!(v == v) || v > 1e30 || v < -1e30) return "0";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", prec, v);
    return buf;
}

Response format_transcription(const pk::Transcription& tr, Format fmt,
                              double duration_sec, bool include_words) {
    Response r;
    if (fmt == Format::kText) {
        r.body = tr.text;
        r.content_type = "text/plain; charset=utf-8";
        return r;
    }
    if (fmt == Format::kJson) {
        r.body = "{\"text\":\"" + json_escape(tr.text) + "\"}";
        r.content_type = "application/json";
        return r;
    }
    // verbose_json
    std::string b = "{";
    b += "\"task\":\"transcribe\",";
    b += "\"language\":\"en\",";  // fixed: models here are English (no detection)
    b += "\"duration\":" + fixed(duration_sec, 3) + ",";
    b += "\"text\":\"" + json_escape(tr.text) + "\",";
    // Single synthetic segment: Parakeet emits no native segmentation.
    b += "\"segments\":[{";
    b += "\"id\":0,";
    b += "\"start\":0.000,";
    b += "\"end\":" + fixed(duration_sec, 3) + ",";
    b += "\"text\":\"" + json_escape(tr.text) + "\"";
    b += "}]";
    if (include_words) {
        // Each word carries the NeMo per-word confidence (Word.conf); surface it
        // under the same "conf" key parakeet-cli --json uses so clients can flag
        // low-confidence spans.
        b += ",\"words\":[";
        for (size_t i = 0; i < tr.words.size(); ++i) {
            const pk::Word& w = tr.words[i];
            if (i) b += ",";
            b += "{\"word\":\"" + json_escape(w.text) + "\",";
            b += "\"start\":" + fixed(w.start, 3) + ",";
            b += "\"end\":" + fixed(w.end, 3) + ",";
            b += "\"conf\":" + fixed(w.conf, 4) + "}";
        }
        b += "]";
    }
    b += "}";
    r.body = b;
    r.content_type = "application/json";
    return r;
}

std::string error_body(const std::string& message, const std::string& type) {
    return "{\"error\":{\"message\":\"" + json_escape(message) +
           "\",\"type\":\"" + json_escape(type) + "\"}}";
}

} // namespace pkserver
