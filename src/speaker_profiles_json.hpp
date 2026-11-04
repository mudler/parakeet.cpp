#pragma once
#include "speaker_identifier.hpp"
#include <iomanip>
#include <locale>
#include <sstream>
#include <string_view>

namespace pk {
// Shared production assembler: nullptr preserves the legacy JSON byte for byte.
// identity is generated internally; reasons are the fixed SpeakerProfile codes.
inline std::string named_profiles_json(const std::string& json, const std::string& names,
    const std::map<int, SpeakerProfile>* profiles, const std::string& identity, int dimension) {
    std::string_view json_trimmed(json);
    if (!json_trimmed.empty()) {
        json_trimmed.remove_suffix(1);
    }
    if (!profiles) {
        std::string res;
        res.reserve(json_trimmed.size() + 10 + names.size() + 1);
        res.append(json_trimmed);
        res.append(",\"names\":");
        res.append(names);
        res.push_back('}');
        return res;
    }
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(9);
    out << json_trimmed << ",\"names\":" << names << ",\"speaker_profiles\":{\"version\":1,\"encoder\":{\"identity\":\""
        << identity << "\",\"dimension\":" << dimension << "},\"speakers\":[";
    bool first = true;
    for (const auto& kv : *profiles) {
        if (!first) out << ',';
        first = false;
        const auto& p = kv.second;
        out << "{\"speaker\":" << kv.first << ",\"clean_duration\":" << p.clean_duration
            << ",\"intervals\":[";
        for (size_t i = 0; i < p.intervals.size(); ++i) {
            if (i) out << ',';
            out << "{\"start\":" << p.intervals[i].start << ",\"end\":" << p.intervals[i].end << '}';
        }
        out << "],\"unavailable_reason\":";
        if (p.unavailable_reason.empty()) out << "null";
        else out << '"' << p.unavailable_reason << '"';
        if (!p.embedding.empty()) {
            out << ",\"embedding\":[";
            for (size_t i = 0; i < p.embedding.size(); ++i) {
                if (i) out << ',';
                out << p.embedding[i];
            }
            out << ']';
        }
        out << '}';
    }
    return out.str() + "]}}";
}
} // namespace pk
