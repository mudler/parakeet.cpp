// The single-clip attention reads NeMo's rel_shift of the position scores through a strided view
// (pk::rel_shift) instead of pad, reshape, copy, slice, copy. On the CPU backend the view, copied out,
// must equal NeMo's rel_shift computed on the host, and the scores ac + rel_shift(bd) that build_graph
// adds must equal ac plus that reference, bit for bit, for T from 1 to 33 and one or four heads.
// test_relpos_attention_gpu runs the same graph on GPUs.
#include "relpos_cases.hpp"

#include <cstdio>
#include <vector>

int main() {
    pk::Backend cpu(4, /*cpu_only=*/true);
    int failures = 0, cases = 0;
    for (int T : {1, 2, 3, 8, 33})
        for (int H : {1, 4}) {
            const std::vector<float> bd = rstest::fill((size_t)(2 * T - 1) * T * H, (uint32_t)(T * 10 + H));
            const std::vector<float> ac = rstest::fill((size_t)T * T * H, (uint32_t)(T * 10 + H + 5));
            const std::vector<float> shifted = rstest::nemo_rel_shift(bd, T, H);
            std::vector<float> want(shifted.size());
            for (size_t i = 0; i < want.size(); ++i) want[i] = ac[i] + shifted[i];

            std::vector<float> view, sum;
            const bool ok = cpu.compute([&](ggml_context* ctx) -> ggml_tensor* {
                const int64_t ne[3] = {2 * T - 1, T, H};
                ggml_tensor* b = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 3, ne, bd.data(), bd.size() * sizeof(float));
                return ggml_cont(ctx, pk::rel_shift(ctx, b));
            }, view) && rstest::scores(cpu, ac, bd, T, H, sum);
            ++cases;
            if (!ok || view != shifted || sum != want) {
                std::fprintf(stderr, "FAIL T=%d H=%d: computed %d, view==NeMo %d, scores==ac+NeMo %d\n", T, H,
                             (int)ok, (int)(view == shifted), (int)(sum == want));
                ++failures;
            }
        }
    if (failures) return 1;
    std::printf("ok: %d cases, the view equals NeMo's rel_shift bit for bit\n", cases);
    return 0;
}
