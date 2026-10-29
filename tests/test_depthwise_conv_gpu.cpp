// The depthwise conv on every GPU device of this build (Vulkan, CUDA, ...), through the production
// builder with DepthwiseConvPath::Auto, as the encoder runs it:
//   - at the published models' shapes (K = 9, C a multiple of 128) and every width with a kernel, Auto
//     must take SSM_CONV on Vulkan and CUDA, and match the CPU's im2col result and NeMo's conv;
//   - at K = 7 (no ggml-cuda kernel) it must take im2col on every device;
//   - at C = 100, ggml-cuda declines SSM_CONV (its kernel needs C % 128 == 0), so CUDA must take im2col
//     while Vulkan, which runs any C, keeps SSM_CONV. Other devices must follow what they report.
// Exits 77 without a GPU device.
#include "depthwise_conv_cases.hpp"
#include "gpu_devices.hpp"

#include <cstdio>
#include <string>
#include <vector>

// What the device reports for the two ops Auto asks about, built here in a scratch context.
static bool device_runs_ssm(ggml_backend_dev_t dev, const dwtest::Case& c) {
    ggml_init_params p = {ggml_tensor_overhead() * 8, nullptr, /*no_alloc=*/true};
    ggml_context* ctx = ggml_init(p);
    const int left = c.causal ? c.K - 1 : (c.K - 1) / 2;
    ggml_tensor* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.T, c.C, c.B);
    ggml_tensor* w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.K, c.C);
    ggml_tensor* xp = ggml_pad_ext(ctx, x, left, c.K - 1 - left, 0, 0, 0, 0, 0, 0);
    ggml_tensor* y = ggml_ssm_conv(ctx, xp, w);
    const bool runs = ggml_backend_dev_supports_op(dev, xp) && ggml_backend_dev_supports_op(dev, y);
    ggml_free(ctx);
    return runs;
}

static bool has_kernel(int K) { return K == 3 || K == 4 || K == 5 || K == 9 || K == 15; }

int main() {
    const std::vector<std::string> devices = gputest::device_names();
    if (int rc = gputest::skip_without_gpu(devices, "test_depthwise_conv_gpu")) return rc;

    std::vector<dwtest::Case> cases;
    for (bool causal : {false, true})
        for (int B : {1, 3})
            for (int T : {1, 8, 9, 200}) cases.push_back({9, T, 256, B, causal});
    for (int K : {3, 5, 15}) cases.push_back({K, 37, 128, 2, false});
    for (int K : {3, 4, 5, 15}) cases.push_back({K, 37, 128, 2, true});
    cases.push_back({7, 37, 128, 2, false});
    cases.push_back({9, 37, 100, 2, false});
    cases.push_back({9, 37, 100, 1, true});

    pk::Backend cpu(4, /*cpu_only=*/true);
    // GPU kernels sum in another order than the CPU, so results agree to rounding, not bit for bit.
    const double tol = 1e-4;
    int failures = 0;
    for (const std::string& name : devices) {
        auto gpu = gputest::backend_on(name);
        if (!gpu) { ++failures; continue; }
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(name.c_str());
        const bool cuda = name.rfind("CUDA", 0) == 0, vulkan = name.rfind("Vulkan", 0) == 0;
        double worst = 0.0;
        int n_ssm = 0;
        for (const dwtest::Case& c : cases) {
            const uint32_t seed = (uint32_t)(c.K * 7919 + c.T * 31 + c.C * 3 + c.B + c.causal);
            const std::vector<float> x = dwtest::fill((size_t)c.T * c.C * c.B, seed);
            const std::vector<float> w = dwtest::fill((size_t)c.K * c.C, seed ^ 0x51ed270bu);
            const std::vector<float> ref = dwtest::reference(c, x, w);

            // Vulkan and CUDA are held to what their kernels are known to run; other devices to what they report.
            bool want_ssm = has_kernel(c.K) && device_runs_ssm(dev, c);
            if (has_kernel(c.K) && (cuda || vulkan)) want_ssm = vulkan || c.C % 128 == 0;

            std::vector<float> y_gpu, y_cpu;
            ggml_op op_gpu, op_cpu;
            if (!dwtest::run(*gpu, c, x, w, pk::DepthwiseConvPath::Auto, y_gpu, op_gpu) ||
                !dwtest::run(cpu, c, x, w, pk::DepthwiseConvPath::Im2col, y_cpu, op_cpu)) {
                std::fprintf(stderr, "FAIL [%s %s]: compute failed\n", name.c_str(), dwtest::name(c));
                ++failures;
                continue;
            }
            const double d_cpu = dwtest::max_abs_diff(y_gpu, y_cpu);
            const double d_ref = dwtest::max_abs_diff(y_gpu, ref);
            const ggml_op want_op = want_ssm ? GGML_OP_SSM_CONV : GGML_OP_CONT;
            n_ssm += op_gpu == GGML_OP_SSM_CONV;
            worst = std::fmax(worst, d_cpu);
            if (op_gpu != want_op || d_cpu > tol || d_ref > tol) {
                std::fprintf(stderr, "FAIL [%s %s]: built %s, want %s; max|d| vs CPU im2col %.3g, vs NeMo %.3g\n",
                             name.c_str(), dwtest::name(c), ggml_op_name(op_gpu), ggml_op_name(want_op),
                             d_cpu, d_ref);
                ++failures;
            }
        }
        std::printf("%s: %d of %zu cases took SSM_CONV, max|d| vs CPU im2col %.3g\n", name.c_str(), n_ssm,
                    cases.size(), worst);
    }
    if (failures) return 1;
    std::printf("ok: tested on %zu GPU device(s)\n", devices.size());
    return 0;
}
