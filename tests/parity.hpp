#pragma once
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace pktest {

// Load an f32 tensor (flattened, row-major) by name from a baseline gguf.
inline bool load_baseline(const std::string& path, const std::string& name,
                          std::vector<float>& out, std::vector<int64_t>& shape) {
    ggml_context* ctx = nullptr;
    gguf_init_params p{ /*no_alloc=*/false, /*ctx=*/&ctx };
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    if (!g) {
        std::fprintf(stderr, "[parity] failed to open baseline: %s\n", path.c_str());
        return false;
    }
    ggml_tensor* t = ggml_get_tensor(ctx, name.c_str());
    if (!t) {
        std::fprintf(stderr, "[parity] tensor '%s' not found in %s\n", name.c_str(), path.c_str());
        gguf_free(g);
        ggml_free(ctx);
        return false;
    }
    shape.clear();
    // Report shape outer..inner (slowest to fastest varying dimension)
    for (int i = ggml_n_dims(t) - 1; i >= 0; --i) {
        shape.push_back(t->ne[i]);
    }
    size_t n = (size_t)ggml_nelements(t);
    out.resize(n);
    std::memcpy(out.data(), t->data, n * sizeof(float));
    gguf_free(g);
    ggml_free(ctx);
    return true;
}

// Compare got vs ref; returns true if all elements are within tolerance.
// Prints max/mean abs diff, worst element, and OK/FAIL to stderr.
inline bool compare(const std::vector<float>& got, const std::vector<float>& ref,
                    const char* label, float atol, float rtol) {
    if (got.size() != ref.size()) {
        std::fprintf(stderr, "[%s] size mismatch got=%zu ref=%zu\n",
                     label, got.size(), ref.size());
        return false;
    }
    double maxabs = 0.0;
    double sumabs = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        double d = std::fabs((double)got[i] - (double)ref[i]);
        sumabs += d;
        if (d > maxabs) { maxabs = d; worst = i; }
    }
    double mean = sumabs / (got.size() ? got.size() : 1);
    bool ok = true;
    for (size_t i = 0; i < got.size() && ok; ++i) {
        double tol = (double)atol + (double)rtol * std::fabs((double)ref[i]);
        if (std::fabs((double)got[i] - (double)ref[i]) > tol) ok = false;
    }
    std::fprintf(stderr,
        "[%s] n=%zu max|d|=%.3e mean|d|=%.3e (worst@%zu got=%.5f ref=%.5f) -> %s\n",
        label, got.size(), maxabs, mean, worst,
        got[worst], ref[worst], ok ? "OK" : "FAIL");
    return ok;
}

// Load an int32 tensor (flattened) by name from a baseline gguf.
inline bool load_baseline_i32(const std::string& path, const std::string& name,
                               std::vector<int32_t>& out) {
    ggml_context* ctx = nullptr;
    gguf_init_params p{ /*no_alloc=*/false, /*ctx=*/&ctx };
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    if (!g) {
        std::fprintf(stderr, "[parity] failed to open baseline: %s\n", path.c_str());
        return false;
    }
    ggml_tensor* t = ggml_get_tensor(ctx, name.c_str());
    if (!t) {
        std::fprintf(stderr, "[parity] tensor '%s' not found in %s\n", name.c_str(), path.c_str());
        gguf_free(g);
        ggml_free(ctx);
        return false;
    }
    size_t n = (size_t)ggml_nelements(t);
    out.resize(n);
    std::memcpy(out.data(), t->data, n * sizeof(int32_t));
    gguf_free(g);
    ggml_free(ctx);
    return true;
}

// Read a uint32 KV entry from a baseline gguf (0 if absent / unopenable).
inline uint32_t pktest_read_u32(const std::string& path, const std::string& key) {
    gguf_init_params p{ /*no_alloc=*/true, /*ctx=*/nullptr };
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    if (!g) return 0;
    int64_t id = gguf_find_key(g, key.c_str());
    uint32_t v = (id < 0) ? 0u : gguf_get_val_u32(g, id);
    gguf_free(g);
    return v;
}

// Load a string KV entry from a baseline gguf.
inline bool load_kv_str(const std::string& path, const std::string& key,
                         std::string& out) {
    gguf_init_params p{ /*no_alloc=*/true, /*ctx=*/nullptr };
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    if (!g) {
        std::fprintf(stderr, "[parity] failed to open baseline: %s\n", path.c_str());
        return false;
    }
    int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) {
        std::fprintf(stderr, "[parity] key '%s' not found in %s\n", key.c_str(), path.c_str());
        gguf_free(g);
        return false;
    }
    out = std::string(gguf_get_val_str(g, id));
    gguf_free(g);
    return true;
}

} // namespace pktest
