#pragma once
// Minimal Unicode property lookup for the byte-level BPE pre-tokenizer.
// Only the classes that regex needs are available: Lu, Ll, Lt, Lm, Lo, M, N
// and White_Space. The data is generated (see scripts/gen_unicode_tables.py
// and docs/bpe-tokenizer.md); no ICU is used.
#include <cstdint>

namespace pk {
namespace unicode {

enum : uint32_t {
    kLu = 1,    // \p{Lu}
    kLl = 2,    // \p{Ll}
    kLt = 4,    // \p{Lt}
    kLm = 8,    // \p{Lm}
    kLo = 16,   // \p{Lo}
    kM = 32,    // \p{M}  (Mn, Mc, Me)
    kN = 64,    // \p{N}  (Nd, Nl, No)
    kWS = 128,  // White_Space, the meaning of \s
    kLetter = kLu | kLl | kLt | kLm | kLo,
};

// Bit mask of the classes above that the code point belongs to (0 for none).
uint32_t props(uint32_t cp);

// Version of the Unicode data the tables were generated from, e.g. "15.0.0".
const char* version();

}  // namespace unicode
}  // namespace pk
