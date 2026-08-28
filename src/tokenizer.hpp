#pragma once
#include <string>
#include <vector>
#include <cstdint>
namespace pk {
// Map each id to its SentencePiece piece, concatenate, replace the U+2581
// meta-space character (▁, UTF-8: 0xE2 0x96 0x81) with a regular space, and
// strip a single leading space if present.  This matches the behavior of
// NeMo SentencePieceTokenizer::ids_to_text (non-legacy path) which calls
// sentencepiece::SentencePieceProcessor::decode_ids.
std::string detokenize(const std::vector<std::string>& pieces,
                       const std::vector<int32_t>& ids);

// Drop ids whose piece is a bracketed special token (`<...>` or `[...]`, e.g.
// language tags like `<en-US>` or `<EOU>`), so they never reach detokenize().
// Ordinary SentencePiece content tokens (including `▁`-prefixed word starts)
// are left untouched.
std::vector<int32_t> strip_special_tokens(const std::vector<std::string>& pieces,
                                          const std::vector<int32_t>& ids);
} // namespace pk
