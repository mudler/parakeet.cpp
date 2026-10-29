// Backend::compute on every GPU device: without the scheduler it queues the reads of the captured tensors
// and the output behind the graph and waits once, so each read must hold this compute's values when
// compute returns. Each round changes the inputs, so a read that returned early would hold the previous
// round's values or none. The graphs mix a large output (4M floats) with small captures, as a decode step
// and the encoder do. A graph with an op the GPU lacks (a custom op) takes the scheduler path, whose
// reads stay synchronous; it is checked the same way. Values are small integers, so every result is
// exact on any device. Exits 77 without a GPU device.
#include "gpu_devices.hpp"
#include "ggml.h"

#include <cstdio>
#include <string>
#include <vector>

// Runs on the CPU: GPU backends do not support custom ops, so this graph goes through the scheduler.
static void plus_one(ggml_tensor* dst, const ggml_tensor* a, int ith, int nth, void*) {
    const int64_t n = ggml_nelements(dst);
    const float* src = (const float*)a->data;
    float* out = (float*)dst->data;
    for (int64_t i = ith; i < n; i += nth) out[i] = src[i] + 1.0f;
}

int main() {
    const std::vector<std::string> devices = gputest::device_names();
    if (int rc = gputest::skip_without_gpu(devices, "test_backend_readback_gpu")) return rc;

    const int n_big = 1 << 22, n_small = 5;
    int failures = 0;
    for (const std::string& name : devices) {
        auto gpu = gputest::backend_on(name);
        if (!gpu) { ++failures; continue; }
        for (bool sched : {false, true}) {
            int bad = 0;
            for (int round = 0; round < 20; ++round) {
                std::vector<float> x(n_big), y(n_small);
                for (int i = 0; i < n_big; ++i) x[i] = (float)((i + round) % 1000);
                for (int i = 0; i < n_small; ++i) y[i] = (float)(round * 10 + i);
                std::vector<float> doubled, tripled_small, out;
                const bool ok = gpu->compute([&](ggml_context* ctx) -> ggml_tensor* {
                    const int64_t big_ne[1] = {n_big}, small_ne[1] = {n_small};
                    ggml_tensor* xt = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 1, big_ne, x.data(),
                                                             x.size() * sizeof(float));
                    ggml_tensor* yt = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 1, small_ne, y.data(),
                                                             y.size() * sizeof(float));
                    ggml_tensor* a = ggml_scale(ctx, xt, 2.0f);
                    ggml_tensor* c = ggml_scale(ctx, yt, 3.0f);
                    pk::capture_graph_output(a, &doubled);
                    pk::capture_graph_output(c, &tripled_small);
                    ggml_tensor* o = ggml_add(ctx, a, xt);
                    return sched ? ggml_map_custom1(ctx, o, plus_one, GGML_N_TASKS_MAX, nullptr) : o;
                }, out);
                if (!ok || doubled.size() != (size_t)n_big || tripled_small.size() != (size_t)n_small ||
                    out.size() != (size_t)n_big) {
                    ++bad;
                    continue;
                }
                for (int i = 0; i < n_big; ++i)
                    if (doubled[i] != 2.0f * x[i] || out[i] != 3.0f * x[i] + (sched ? 1.0f : 0.0f)) { ++bad; break; }
                for (int i = 0; i < n_small; ++i)
                    if (tripled_small[i] != 3.0f * y[i]) { ++bad; break; }
            }
            std::printf("%s, %s path: %d of 20 rounds read wrong values\n", name.c_str(),
                        sched ? "scheduler" : "gallocr", bad);
            failures += bad;
        }
    }
    if (failures) return 1;
    std::printf("ok: tested on %zu GPU device(s)\n", devices.size());
    return 0;
}
