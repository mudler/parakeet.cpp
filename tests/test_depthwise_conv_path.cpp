// When the depthwise conv falls back to im2col. pk::depthwise_conv_takes_ssm is the whole decision of
// DepthwiseConvPath::Auto, with the backend's answers passed in, so every branch runs here without a GPU:
//   - the CPU backend takes im2col, and is never asked about the ops;
//   - a width without a ggml-cuda kernel (any but 3, 4, 5, 9, 15) takes im2col, never asking either;
//   - a GPU backend that declines the pad or the SSM_CONV takes im2col;
//   - only a GPU backend that runs both, at a supported width, takes SSM_CONV.
// Then Auto through the production builder on the CPU backend must build the im2col graph for every
// width. test_depthwise_conv_gpu checks the same decision against real GPU backends.
#include "depthwise_conv_cases.hpp"

#include <cstdio>

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++failures; } } while (0)

int main() {
    for (int K = 1; K <= 16; ++K) {
        const bool has_kernel = K == 3 || K == 4 || K == 5 || K == 9 || K == 15;
        int asked = 0;
        auto runs = [&]() { ++asked; return true; };
        auto declines = [&]() { ++asked; return false; };

        CHECK(!pk::depthwise_conv_takes_ssm(/*backend_is_cpu=*/true, K, runs));
        CHECK(asked == 0);

        CHECK(pk::depthwise_conv_takes_ssm(false, K, runs) == has_kernel);
        CHECK(asked == (has_kernel ? 1 : 0));

        asked = 0;
        CHECK(!pk::depthwise_conv_takes_ssm(false, K, declines));
        CHECK(asked == (has_kernel ? 1 : 0));
    }

    pk::Backend cpu(4, /*cpu_only=*/true);
    for (int K : {3, 5, 7, 9, 15}) {
        const dwtest::Case c{K, 20, 8, 2, false};
        std::vector<float> y;
        ggml_op op;
        CHECK(dwtest::run(cpu, c, dwtest::fill((size_t)c.T * c.C * c.B, 1), dwtest::fill((size_t)K * c.C, 2),
                          pk::DepthwiseConvPath::Auto, y, op));
        if (op != GGML_OP_CONT) {
            std::fprintf(stderr, "FAIL: Auto on the CPU at K=%d built %s, not im2col\n", K, ggml_op_name(op));
            ++failures;
        }
    }

    if (failures) return 1;
    std::printf("ok: im2col on the CPU, at widths without a kernel, and where the backend declines\n");
    return 0;
}
