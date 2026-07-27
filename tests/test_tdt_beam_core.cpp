#include "tdt.hpp"

#include <cstdio>
#include <vector>

int main() {
    const std::vector<int32_t> durations{0, 1, 2, 4};
    const std::vector<float> scores{0.0f, -3.0f, -0.1f, -2.0f};

    const int best =
        pk::detail::best_positive_duration_index(scores, durations);
    if (best != 2) {
        std::fprintf(
            stderr,
            "test_tdt_beam_core: expected duration index 2, got %d\n",
            best);
        return 1;
    }
    return 0;
}
