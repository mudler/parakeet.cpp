// The conv module's depthwise conv through the production builder, both paths, on the CPU backend:
// SSM_CONV (the GPU path, forced here) and im2col (the CPU path) must each equal NeMo's conv computed on
// the host, for causal and symmetric padding, one item and a batch, and short clips down to T = 1 < K.
// Auto must pick im2col on the CPU and give exactly the im2col result.
//
// This validates the SSM_CONV graph's padding and batch layout with ggml-cpu's SSM_CONV kernel. It does
// not run a GPU kernel: test_depthwise_conv_gpu does, and skips where no GPU device exists.
#include "depthwise_conv_cases.hpp"

#include <cstdio>
#include <set>
#include <vector>

int main() {
    pk::Backend cpu(4, /*cpu_only=*/true);
    // Sums of at most 15 products of values in [-1, 1): f32 rounding stays below 1e-5.
    const double tol = 1e-5;
    int failures = 0, cases = 0;
    double worst = 0.0;
    for (bool causal : {true, false}) {
        // ConformerLayer asserts an odd K. The causal builder also takes K = 4, which ggml-cuda has a
        // kernel for; a symmetric even K has no NeMo shape (im2col gives T - 1 frames).
        const std::vector<int> widths = causal ? std::vector<int>{3, 4, 5, 9, 15}
                                               : std::vector<int>{3, 5, 9, 15};
        for (int K : widths)
            for (int B : {1, 3})
                for (int T : std::set<int>{1, 2, K - 1, K, K + 1, 50}) {
                    const dwtest::Case c{K, T, 16, B, causal};
                    const uint32_t seed = (uint32_t)(K * 1000003 + T * 131 + B * 7 + causal);
                    const std::vector<float> x = dwtest::fill((size_t)T * c.C * B, seed);
                    const std::vector<float> w = dwtest::fill((size_t)K * c.C, seed ^ 0x9e3779b9u);
                    const std::vector<float> ref = dwtest::reference(c, x, w);

                    std::vector<float> y_ssm, y_im2col, y_auto;
                    ggml_op op_ssm, op_im2col, op_auto;
                    const bool ran = dwtest::run(cpu, c, x, w, pk::DepthwiseConvPath::SsmConv, y_ssm, op_ssm) &&
                                     dwtest::run(cpu, c, x, w, pk::DepthwiseConvPath::Im2col, y_im2col, op_im2col) &&
                                     dwtest::run(cpu, c, x, w, pk::DepthwiseConvPath::Auto, y_auto, op_auto);
                    ++cases;
                    if (!ran) {
                        std::fprintf(stderr, "FAIL [%s]: compute failed\n", dwtest::name(c));
                        ++failures;
                        continue;
                    }
                    const double d_ssm = dwtest::max_abs_diff(y_ssm, ref);
                    const double d_im2col = dwtest::max_abs_diff(y_im2col, ref);
                    worst = std::fmax(worst, std::fmax(d_ssm, d_im2col));
                    const bool ok = op_ssm == GGML_OP_SSM_CONV && op_im2col == GGML_OP_CONT &&
                                    op_auto == GGML_OP_CONT && y_auto == y_im2col &&
                                    d_ssm <= tol && d_im2col <= tol;
                    if (!ok) {
                        std::fprintf(stderr,
                                     "FAIL [%s]: ops ssm=%s im2col=%s auto=%s, auto==im2col %d, "
                                     "max|d| vs NeMo: ssm %.3g im2col %.3g\n",
                                     dwtest::name(c), ggml_op_name(op_ssm), ggml_op_name(op_im2col),
                                     ggml_op_name(op_auto), (int)(y_auto == y_im2col), d_ssm, d_im2col);
                        ++failures;
                    }
                }
    }
    if (failures) {
        std::fprintf(stderr, "%d of %d cases failed\n", failures, cases);
        return 1;
    }
    std::printf("ok: %d cases on the CPU backend, SSM_CONV and im2col within %.3g of NeMo's conv (limit %.0e)\n",
                cases, worst, tol);
    return 0;
}
