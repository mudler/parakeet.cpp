#pragma once
// For the GPU tests: the GPU devices this build and machine have, and a pk::Backend on one of them.
// A test with none exits 77 (ctest SKIP) and says why, so a CPU build never reports a GPU path as tested.
#include "backend.hpp"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace gputest {

constexpr int kSkip = 77;

// Registry names ("CUDA0", "Vulkan1", ...) of the discrete and integrated GPUs, in registry order.
inline std::vector<std::string> device_names() {
    std::vector<std::string> names;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(dev);
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU)
            names.emplace_back(ggml_backend_dev_name(dev));
    }
    return names;
}

// Prints the skip reason and returns kSkip when there is no GPU device.
inline int skip_without_gpu(const std::vector<std::string>& names, const char* test) {
    if (!names.empty()) return 0;
    std::fprintf(stderr,
                 "SKIP %s: no GPU device (this build has no GPU backend, or the machine has no GPU), "
                 "so the GPU path is NOT tested\n", test);
    return kSkip;
}

// A Backend on the named device, through PARAKEET_DEVICE as production selects it. Null, with a message,
// when the Backend landed on another device.
inline std::unique_ptr<pk::Backend> backend_on(const std::string& name) {
    setenv("PARAKEET_DEVICE", name.c_str(), 1);
    auto be = std::make_unique<pk::Backend>(4);
    unsetenv("PARAKEET_DEVICE");
    if (be->device_name() != name || be->is_cpu()) {
        std::fprintf(stderr, "FAIL: asked for device %s, got %s\n", name.c_str(), be->device_name());
        return nullptr;
    }
    return be;
}

} // namespace gputest
