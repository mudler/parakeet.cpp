#include "unicode_props.hpp"
#include "unicode_props_data.hpp"

#include <algorithm>

namespace pk {
namespace unicode {

namespace {
struct LatinCache {
    uint32_t m[256];
    LatinCache() {
        for (uint32_t c = 0; c < 256; ++c) m[c] = lookup_slow(c);
    }
    static uint32_t lookup_slow(uint32_t cp) {
        const unicode_data::Range* b = unicode_data::kRanges;
        const unicode_data::Range* e = b + unicode_data::kNumRanges;
        const unicode_data::Range* it = std::upper_bound(
            b, e, cp, [](uint32_t v, const unicode_data::Range& r) { return v < r.start; });
        if (it == b) return 0;
        return (it - 1)->mask;
    }
};
}  // namespace

uint32_t props(uint32_t cp) {
    static const LatinCache cache;
    if (cp < 256) return cache.m[cp];
    return LatinCache::lookup_slow(cp);
}

const char* version() { return unicode_data::kUnicodeVersion; }

}  // namespace unicode
}  // namespace pk
