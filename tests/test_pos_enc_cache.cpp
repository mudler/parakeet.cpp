// pk::rel_pos_encoding keeps one table grown to the longest clip so far and returns a slice of it. Every
// call, in any order of lengths and model widths, must return the table NeMo's formula gives for that T,
// bit for bit: shorter after longer, the same length twice, T = 1, another d_model and back, and from
// several threads at once.
#include "pos_enc.hpp"

#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

static int failures = 0;

// The table as computed before the cache, the same double arithmetic: row p holds position (T-1)-p.
static std::vector<float> fresh_table(int T, int d) {
    std::vector<float> out((size_t)(2 * T - 1) * d);
    const double factor = -(std::log(10000.0) / (double)d);
    for (int p = 0; p < 2 * T - 1; ++p)
        for (int i = 0; i < d / 2; ++i) {
            const double arg = (double)((T - 1) - p) * std::exp((double)(2 * i) * factor);
            out[(size_t)p * d + 2 * i] = (float)std::sin(arg);
            out[(size_t)p * d + 2 * i + 1] = (float)std::cos(arg);
        }
    return out;
}

static void check(int T, int d) {
    std::vector<float> got;
    pk::rel_pos_encoding(T, d, got);
    if (got != fresh_table(T, d)) {
        std::fprintf(stderr, "FAIL: T=%d d_model=%d differs from a freshly built table\n", T, d);
        ++failures;
    }
}

int main() {
    for (int T : {50, 50, 7, 300, 1, 299, 50, 301, 2}) check(T, 176);
    check(40, 512);   // another width rebuilds the table
    check(400, 512);
    check(10, 176);   // and back
    check(310, 176);

    std::vector<std::thread> threads;
    std::vector<int> bad(8, 0);
    for (int i = 0; i < 8; ++i)
        threads.emplace_back([i, &bad]() {
            for (int r = 0; r < 20; ++r) {
                const int T = 1 + (i * 37 + r * 53) % 400;
                std::vector<float> got;
                pk::rel_pos_encoding(T, 176, got);
                bad[i] += got != fresh_table(T, 176);
            }
        });
    for (std::thread& t : threads) t.join();
    for (int b : bad) failures += b;

    if (failures) {
        std::fprintf(stderr, "%d mismatches\n", failures);
        return 1;
    }
    std::printf("ok: every slice equals a freshly built table\n");
    return 0;
}
