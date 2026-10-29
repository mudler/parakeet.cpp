// The single-clip attention's two GPU-facing changes, on every GPU device:
//   1. The scores ac + rel_shift(bd) read the position scores through a strided view, so the device's
//      add must take that non-contiguous input: the result must equal ac plus NeMo's rel_shift.
//   2. An unpadded full-context clip runs softmax without a mask. It must equal softmax with the
//      all-zero mask it replaces, and the CPU's softmax.
// T runs from 1 to 200 with four heads. Exits 77 without a GPU device.
#include "relpos_cases.hpp"
#include "gpu_devices.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return INFINITY;
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::fmax(m, std::fabs((double)a[i] - (double)b[i]));
    return m;
}

// softmax(scale * (ac + rel_shift(bd)) + mask) as build_graph computes it, with an all-zero mask or none.
static bool attention(pk::Backend& be, const std::vector<float>& ac, const std::vector<float>& bd, int T,
                      int H, bool zero_mask, std::vector<float>& out) {
    const std::vector<float> zeros((size_t)T * T, 0.0f);
    return be.compute([&](ggml_context* ctx) -> ggml_tensor* {
        const int64_t ac_ne[3] = {T, T, H}, bd_ne[3] = {2 * T - 1, T, H}, m_ne[2] = {T, T};
        ggml_tensor* a = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 3, ac_ne, ac.data(), ac.size() * sizeof(float));
        ggml_tensor* b = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 3, bd_ne, bd.data(), bd.size() * sizeof(float));
        ggml_tensor* m = zero_mask ? pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, m_ne, zeros.data(),
                                                            zeros.size() * sizeof(float))
                                   : nullptr;
        return ggml_soft_max_ext(ctx, ggml_add(ctx, a, pk::rel_shift(ctx, b)), m, 0.125f, 0.0f);
    }, out);
}

int main() {
    const std::vector<std::string> devices = gputest::device_names();
    if (int rc = gputest::skip_without_gpu(devices, "test_relpos_attention_gpu")) return rc;

    pk::Backend cpu(4, /*cpu_only=*/true);
    // An f32 add is exact on any device; exp and the softmax sum differ between kernels by rounding.
    const double tol_scores = 0.0, tol_softmax = 1e-5;
    int failures = 0;
    for (const std::string& name : devices) {
        auto gpu = gputest::backend_on(name);
        if (!gpu) { ++failures; continue; }
        double worst_scores = 0.0, worst_softmax = 0.0;
        for (int T : {1, 2, 33, 200}) {
            const int H = 4;
            const std::vector<float> bd = rstest::fill((size_t)(2 * T - 1) * T * H, (uint32_t)(T * 3 + 1));
            const std::vector<float> ac = rstest::fill((size_t)T * T * H, (uint32_t)(T * 3 + 2));
            const std::vector<float> shifted = rstest::nemo_rel_shift(bd, T, H);
            std::vector<float> want(shifted.size());
            for (size_t i = 0; i < want.size(); ++i) want[i] = ac[i] + shifted[i];

            std::vector<float> sc, sm_none, sm_zero, sm_cpu;
            if (!rstest::scores(*gpu, ac, bd, T, H, sc) || !attention(*gpu, ac, bd, T, H, false, sm_none) ||
                !attention(*gpu, ac, bd, T, H, true, sm_zero) || !attention(cpu, ac, bd, T, H, false, sm_cpu)) {
                std::fprintf(stderr, "FAIL [%s T=%d]: compute failed\n", name.c_str(), T);
                ++failures;
                continue;
            }
            const double d_sc = max_abs_diff(sc, want);
            const double d_sm = std::fmax(max_abs_diff(sm_none, sm_zero), max_abs_diff(sm_none, sm_cpu));
            worst_scores = std::fmax(worst_scores, d_sc);
            worst_softmax = std::fmax(worst_softmax, d_sm);
            if (d_sc > tol_scores || d_sm > tol_softmax) {
                std::fprintf(stderr, "FAIL [%s T=%d]: scores vs ac+NeMo %.3g; softmax unmasked vs zero mask %.3g, "
                             "vs CPU %.3g\n", name.c_str(), T, d_sc, max_abs_diff(sm_none, sm_zero),
                             max_abs_diff(sm_none, sm_cpu));
                ++failures;
            }
        }
        std::printf("%s: scores max|d| %.3g, softmax max|d| %.3g\n", name.c_str(), worst_scores, worst_softmax);
    }
    if (failures) return 1;
    std::printf("ok: tested on %zu GPU device(s)\n", devices.size());
    return 0;
}
