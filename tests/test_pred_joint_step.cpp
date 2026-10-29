// The TDT decoder's step graphs, on the CPU or, with --gpu, on the first GPU device:
//   1. pk::pred_joint_step (the prediction net and the joint as one graph, as tdt_greedy runs them) must
//      give what PredictionNet::step followed by Joint::step_logits give, over SOS and three tokens with
//      the state carried: the same bits on the CPU, within 1e-4 on a GPU.
//   2. PredictionNet::step (whose LSTM gates are views of z, without copies) must equal the LSTM
//      computed on the host in double from the GGUF weights, within 1e-4.
//   3. A fresh PredictionNet must have fetched its embedding table after an SOS step, which embeds
//      nothing, so a warm-up on silence pays that read and the first real decode does not.
// Needs PARAKEET_TEST_GGUF; --gpu also needs a GPU device. Each exits 77 without what it needs.
#include "gpu_devices.hpp"
#include "backend.hpp"
#include "ggml_graph.hpp"
#include "joint.hpp"
#include "model_loader.hpp"
#include "prediction.hpp"
#include "tdt.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++failures; } } while (0)

static double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return INFINITY;
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::fmax(m, std::fabs((double)a[i] - (double)b[i]));
    return m;
}

static double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// NeMo's stacked LSTM step in double, from the host copies of the weights. Gate order i, f, g, o.
static void host_lstm_step(const pk::ModelLoader& ml, int H, int L, const std::vector<float>& x0,
                           const pk::PredState& in, pk::PredState& out) {
    out = in;
    std::vector<double> x(x0.begin(), x0.end());
    for (int l = 0; l < L; ++l) {
        const std::string s = "decoder.prediction.dec_rnn.lstm.";
        std::vector<float> wih, whh, bih, bhh;
        pk::weight_to_host_f32(ml, (s + "weight_ih_l" + std::to_string(l)).c_str(), wih);
        pk::weight_to_host_f32(ml, (s + "weight_hh_l" + std::to_string(l)).c_str(), whh);
        pk::weight_to_host_f32(ml, (s + "bias_ih_l" + std::to_string(l)).c_str(), bih);
        pk::weight_to_host_f32(ml, (s + "bias_hh_l" + std::to_string(l)).c_str(), bhh);
        std::vector<double> z((size_t)4 * H);
        for (int r = 0; r < 4 * H; ++r) {
            double acc = (double)bih[r] + (double)bhh[r];
            for (int k = 0; k < H; ++k)
                acc += (double)wih[(size_t)r * H + k] * x[k] + (double)whh[(size_t)r * H + k] * in.h[l][k];
            z[r] = acc;
        }
        for (int k = 0; k < H; ++k) {
            const double c = sigmoid(z[H + k]) * in.c[l][k] + sigmoid(z[k]) * std::tanh(z[2 * H + k]);
            const double h = sigmoid(z[3 * H + k]) * std::tanh(c);
            out.c[l][k] = (float)c;
            out.h[l][k] = (float)h;
            x[k] = h;
        }
    }
}

int main(int argc, char** argv) {
    const bool gpu = argc > 1 && std::strcmp(argv[1], "--gpu") == 0;
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    if (!gguf) { std::fprintf(stderr, "PARAKEET_TEST_GGUF not set; skip\n"); return 77; }
    std::string device = "cpu";
    if (gpu) {
        const std::vector<std::string> devices = gputest::device_names();
        if (int rc = gputest::skip_without_gpu(devices, "test_pred_joint_step --gpu")) return rc;
        device = devices[0];
    }
    // The process-global backend, which holds the weights, is created on the first compute.
    setenv("PARAKEET_DEVICE", device.c_str(), 1);

    pk::ModelLoader ml;
    if (!ml.load(gguf)) { std::fprintf(stderr, "load failed\n"); return 1; }
    pk::Joint joint(ml);
    const int Hj = joint.joint_hidden();

    {
        pk::PredictionNet fresh(ml);
        CHECK(!fresh.embed_table_fetched());
        std::vector<float> enc_proj((size_t)Hj, 0.0f), x0, g, logits;
        pk::PredState out;
        CHECK(pk::pred_joint_step(fresh, joint, 0, /*is_sos=*/true, fresh.zero_state(), enc_proj.data(),
                                  x0, g, out, logits));
        CHECK(fresh.embed_table_fetched());
    }
    if (pk::global_backend().device_name() != device) {
        std::fprintf(stderr, "FAIL: the global backend is on %s, not %s\n",
                     pk::global_backend().device_name(), device.c_str());
        return 1;
    }

    pk::PredictionNet pred(ml);
    const int H = pred.hidden_size(), L = pred.num_layers();
    std::vector<float> embed;
    pk::weight_to_host_f32(ml, "decoder.prediction.embed.weight", embed);

    const double tol = 1e-4;
    double worst_fused = 0.0, worst_host = 0.0;
    pk::PredState state = pred.zero_state();
    const int32_t tokens[] = {-1, 5, 42, 7};  // -1: SOS
    for (int step = 0; step < 4; ++step) {
        const bool sos = tokens[step] < 0;
        const int32_t id = sos ? 0 : tokens[step];
        std::vector<float> enc_proj((size_t)Hj);
        for (int i = 0; i < Hj; ++i) enc_proj[i] = 0.05f * (float)((i * 7 + step * 13) % 41 - 20);

        std::vector<float> x0, g_f, logits_f, g_u, logits_u;
        pk::PredState s_f, s_u;
        CHECK(pk::pred_joint_step(pred, joint, id, sos, state, enc_proj.data(), x0, g_f, s_f, logits_f));
        pred.step(id, sos, state, g_u, s_u);
        joint.step_logits(enc_proj.data(), g_u.data(), (int)g_u.size(), logits_u);

        double d = std::fmax(max_abs_diff(g_f, g_u), max_abs_diff(logits_f, logits_u));
        for (int l = 0; l < L; ++l)
            d = std::fmax(d, std::fmax(max_abs_diff(s_f.h[l], s_u.h[l]), max_abs_diff(s_f.c[l], s_u.c[l])));
        worst_fused = std::fmax(worst_fused, d);
        if (gpu ? d > tol : d != 0.0) {
            std::fprintf(stderr, "FAIL step %d: fused differs from step + step_logits by %.3g\n", step, d);
            ++failures;
        }

        std::vector<float> x_host((size_t)H, 0.0f);
        if (!sos) std::memcpy(x_host.data(), embed.data() + (size_t)id * H, (size_t)H * sizeof(float));
        pk::PredState s_h;
        host_lstm_step(ml, H, L, x_host, state, s_h);
        double dh = max_abs_diff(g_u, s_h.h[L - 1]);
        for (int l = 0; l < L; ++l)
            dh = std::fmax(dh, std::fmax(max_abs_diff(s_u.h[l], s_h.h[l]), max_abs_diff(s_u.c[l], s_h.c[l])));
        worst_host = std::fmax(worst_host, dh);
        if (dh > tol) {
            std::fprintf(stderr, "FAIL step %d: step() differs from the host LSTM by %.3g\n", step, dh);
            ++failures;
        }
        state = s_u;
    }
    std::printf("%s: fused vs separate max|d| %.3g, step() vs host LSTM max|d| %.3g\n", device.c_str(),
                worst_fused, worst_host);
    pk::shutdown_backend();
    if (failures) return 1;
    std::printf("ok\n");
    return 0;
}
