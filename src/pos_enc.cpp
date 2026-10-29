#include "pos_enc.hpp"
#include <cmath>
#include <cassert>
#include <mutex>

namespace pk {

// INF_VAL from NeMo multi_head_attention.py (used as the sinusoid base).
static constexpr double kInfVal = 10000.0;

static void build_rel_pos_encoding(int T, int d_model, std::vector<float>& out) {
    assert(T > 0 && d_model > 0 && (d_model % 2) == 0);
    const int P = 2 * T - 1;             // number of relative positions
    const int half = d_model / 2;        // number of (sin,cos) pairs

    // div_term[i] = exp(2i * -(log(INF_VAL)/d_model)) for i in [0, half).
    // Compute in double for parity, store the precomputed factors.
    std::vector<double> div_term(half);
    const double factor = -(std::log(kInfVal) / (double)d_model);
    for (int i = 0; i < half; ++i) {
        div_term[i] = std::exp((double)(2 * i) * factor);
    }

    out.assign((size_t)P * d_model, 0.0f);
    // positions run from +(T-1) down to -(T-1).
    for (int p = 0; p < P; ++p) {
        const double pos = (double)((T - 1) - p);   // (T-1), (T-2), ..., -(T-1)
        float* row = out.data() + (size_t)p * d_model;
        for (int i = 0; i < half; ++i) {
            const double arg = pos * div_term[i];
            row[2 * i]     = (float)std::sin(arg);   // even dims
            row[2 * i + 1] = (float)std::cos(arg);   // odd dims
        }
    }
}

// Row p holds position (T-1)-p, so the table for T is rows [Tmax-T, Tmax+T-1) of the table for any
// Tmax >= T, bit for bit. One table, grown to the longest clip so far, spares the sin/cos per call.
void rel_pos_encoding(int T, int d_model, std::vector<float>& out) {
    static std::mutex mu;
    static int table_T = 0, table_d = 0;
    static std::vector<float> table;
    std::lock_guard<std::mutex> lock(mu);
    if (d_model != table_d || T > table_T) {
        build_rel_pos_encoding(T, d_model, table);
        table_T = T;
        table_d = d_model;
    }
    const auto first = table.begin() + (std::ptrdiff_t)(table_T - T) * d_model;
    out.assign(first, first + (std::ptrdiff_t)(2 * T - 1) * d_model);
}

void local_rel_pos_encoding(int att_left, int att_right, int d_model,
                            std::vector<float>& out) {
    assert(att_left >= 0 && att_right >= 0 && d_model > 0 && (d_model % 2) == 0);
    const int P = att_left + att_right + 1;   // local relative positions
    const int half = d_model / 2;

    std::vector<double> div_term(half);
    const double factor = -(std::log(kInfVal) / (double)d_model);
    for (int i = 0; i < half; ++i) {
        div_term[i] = std::exp((double)(2 * i) * factor);
    }

    out.assign((size_t)P * d_model, 0.0f);
    // positions run from +att_left down to -att_right (NeMo arange(left,-right-1,-1)).
    for (int p = 0; p < P; ++p) {
        const double pos = (double)(att_left - p);
        float* row = out.data() + (size_t)p * d_model;
        for (int i = 0; i < half; ++i) {
            const double arg = pos * div_term[i];
            row[2 * i]     = (float)std::sin(arg);
            row[2 * i + 1] = (float)std::cos(arg);
        }
    }
}

} // namespace pk
