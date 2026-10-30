#include "tokenizer.hpp"
namespace pk {
// U+2581 LOWER ONE EIGHTH BLOCK — SentencePiece meta-space marker.
// UTF-8 encoding: 0xE2 0x96 0x81  (3 bytes)
static const char META_SPACE[] = "\xe2\x96\x81";
static const size_t META_SPACE_LEN = 3;

std::string detokenize(const std::vector<std::string>& pieces,
                       const std::vector<int32_t>& ids) {
    // Step 1: concatenate the piece strings for each id.
    std::string result;
    result.reserve(ids.size() * 4);
    for (int32_t id : ids) {
        if (id >= 0 && (size_t)id < pieces.size()) {
            result += pieces[(size_t)id];
        }
    }

    // Step 2: replace every occurrence of META_SPACE (▁) with a regular space, but
    // never two in a row (a lone ▁ before a stripped language tag would double it).
    std::string out;
    out.reserve(result.size());
    for (size_t i = 0; i < result.size(); ) {
        // Check for the 3-byte UTF-8 sequence 0xE2 0x96 0x81
        if (i + META_SPACE_LEN <= result.size() &&
            (unsigned char)result[i]   == 0xE2 &&
            (unsigned char)result[i+1] == 0x96 &&
            (unsigned char)result[i+2] == 0x81) {
            if (out.empty() || out.back() != ' ') {
                out += ' ';
            }
            i += META_SPACE_LEN;
        } else {
            out += result[i++];
        }
    }

    // Step 3: strip a single leading space (SentencePiece decode_ids behavior).
    if (!out.empty() && out[0] == ' ') {
        out.erase(0, 1);
    }
    return out;
}

bool is_special_piece(const std::string& piece) {
    return !piece.empty() &&
           ((piece.front() == '<' && piece.back() == '>') ||
            (piece.front() == '[' && piece.back() == ']'));
}

std::vector<int32_t> strip_special_tokens(const std::vector<std::string>& pieces,
                                          const std::vector<int32_t>& ids) {
    std::vector<int32_t> out;
    out.reserve(ids.size());
    for (int32_t id : ids) {
        if (id >= 0 && (size_t)id < pieces.size()) {
            if (is_special_piece(pieces[(size_t)id])) {
                continue;
            }
        }
        out.push_back(id);
    }
    return out;
}
} // namespace pk
