#include "subsampling.hpp"
#include "model_loader.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
int main(){
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    if(!gguf){ std::fprintf(stderr,"env not set; skip\n"); return 77; }
    pk::ModelLoader ml; if(!ml.load(gguf)) return 1;
    pk::Subsampling sub(ml);
    // f(x)=(x-1)/2+1 applied 3x for non-causal; spot-check.
    struct { int T, Tp; } cases[] = {{100, 13}, {1000, 125}, {307848, 38481}};
    for (auto& c : cases) {
        int got = sub.subsample_len(c.T);
        if (got != c.Tp){ std::fprintf(stderr,"T=%d got=%d want=%d\n",c.T,got,c.Tp); return 1; }
    }
    std::printf("subsample_len ok\n");

    // ---- forward_tiled parity: tiled == untiled within the valid region ----
    {
        const int n_mels = (int)ml.config().n_mels;   // 80 for pk110m
        const int T = 4000;                            // -> ~30 tiles at tile=17
        std::vector<float> mel((size_t)n_mels*T);
        // deterministic pseudo-random, mean-subtracted-ish
        unsigned s2 = 1234567u;
        for (size_t i=0;i<mel.size();++i){ s2 = s2*1103515245u + 12345u; mel[i] = ((s2>>9)&0xFFFF)/32768.0f - 1.0f; }
        std::vector<float> full; int Tp=0, dm=0, vl=0;
        sub.forward(mel, n_mels, T, full, Tp, dm, vl);
        std::vector<float> tiled; int Tp2=0, dm2=0, vl2=0;
        sub.forward_tiled(mel, n_mels, T, /*tile_out_frames*/17, tiled, Tp2, dm2, vl2);
        if (Tp2!=Tp || dm2!=dm || vl2!=vl){ std::fprintf(stderr,"tiled shape/valid mismatch Tp=%d/%d dm=%d/%d vl=%d/%d\n",Tp,Tp2,dm,dm2,vl,vl2); return 1; }

        // STRUCTURAL guard (float-noise-immune): a single tile whose halo spans the
        // whole utterance is literally build_graph over the full T, so it MUST be
        // bit-exact vs forward(). This validates the window->global frame mapping
        // (j0 = ws/8, halo alignment) with ZERO tolerance: any off-by-one in the
        // tiling arithmetic breaks it. (tile_out_frames > Tp => one window.)
        // NOTE: keep T such that valid_out_len(T,-1) == Tp (no trailing mask),
        // else forward() masks tail frames to bias and this all-element guard
        // would false-fail. T=4000 satisfies this (vl==Tp).
        {
            std::vector<float> one; int a=0,b=0,c=0;
            sub.forward_tiled(mel, n_mels, T, Tp+1, one, a, b, c);
            double smax=0; for (size_t i=0;i<full.size();++i){ double d=std::fabs((double)one[i]-(double)full[i]); if(d>smax)smax=d; }
            if (smax != 0.0){ std::fprintf(stderr,"single-tile NOT bit-exact vs forward: maxabs=%.3e\n", smax); return 1; }
        }

        // PARITY (multi-tile, tile=17): compare only the valid region [0, vl).
        // We report the raw absolute maxabs (the synthetic full-band-noise mel
        // drives activations to magnitude ~2e3 and the out Linear sums mixed-sign
        // terms, so cancellation leaves an absolute graph-splitting float-noise
        // floor of ~3e-2 here, INDEPENDENT of input scale). The pass/fail gate uses
        // a per-frame relative metric (max|diff| / max|ref| over the frame): real
        // graph-splitting float noise is ~1e-5 relative, while a wrong receptive
        // field / frame misalignment reads uncorrelated data and diverges by
        // O(|ref|) (~1.0 relative), so 1e-2 cleanly separates the two without
        // hiding a boundary error.
        double maxabs=0, worstrel=0; for (int t=0;t<vl;++t){
            double dmax=0, fmax=0;
            for (int c=0;c<dm;++c){
                double ref = full[(size_t)t*dm+c];
                double d = std::fabs((double)tiled[(size_t)t*dm+c] - ref);
                if (d>maxabs) maxabs=d;
                if (d>dmax) dmax=d;
                if (std::fabs(ref)>fmax) fmax=std::fabs(ref);
            }
            double rel = dmax/(fmax+1e-6);
            if (rel>worstrel) worstrel=rel;
        }
        std::printf("forward_tiled maxabs=%.3e (vl=%d dm=%d)\n", maxabs, vl, dm);
        std::printf("forward_tiled worst per-frame rel=%.3e\n", worstrel);
        if (worstrel > 1e-2){ std::fprintf(stderr,"forward_tiled parity FAIL rel=%.3e maxabs=%.3e\n", worstrel, maxabs); return 1; }
    }

    // ---- Causal model tiling check (if PARAKEET_TEST_GGUF_NEMOTRON is provided) ----
    if (const char* causal_gguf = std::getenv("PARAKEET_TEST_GGUF_NEMOTRON")) {
        pk::ModelLoader ml_causal;
        if (ml_causal.load(causal_gguf) && ml_causal.config().causal_downsampling) {
            pk::Subsampling sub_causal(ml_causal);
            const int n_mels = (int)ml_causal.config().n_mels;
            const int T = 4000;
            std::vector<float> mel((size_t)n_mels * T);
            unsigned s2 = 7654321u;
            for (size_t i = 0; i < mel.size(); ++i) {
                s2 = s2 * 1103515245u + 12345u;
                mel[i] = ((s2 >> 9) & 0xFFFF) / 32768.0f - 1.0f;
            }
            std::vector<float> full; int Tp = 0, dm = 0, vl = 0;
            sub_causal.forward(mel, n_mels, T, full, Tp, dm, vl);

            std::vector<float> tiled; int Tp2 = 0, dm2 = 0, vl2 = 0;
            sub_causal.forward_tiled(mel, n_mels, T, /*tile_out_frames*/17, tiled, Tp2, dm2, vl2);

            if (Tp2 != Tp || dm2 != dm || vl2 != vl) {
                std::fprintf(stderr, "causal tiled shape/valid mismatch Tp=%d/%d dm=%d/%d vl=%d/%d\n",
                             Tp, Tp2, dm, dm2, vl, vl2);
                return 1;
            }

            double maxabs = 0, worstrel = 0;
            for (int t = 0; t < vl; ++t) {
                double dmax = 0, fmax = 0;
                for (int c = 0; c < dm; ++c) {
                    double ref = full[(size_t)t * dm + c];
                    double d = std::fabs((double)tiled[(size_t)t * dm + c] - ref);
                    if (d > maxabs) maxabs = d;
                    if (d > dmax) dmax = d;
                    if (std::fabs(ref) > fmax) fmax = std::fabs(ref);
                }
                double rel = dmax / (fmax + 1e-6);
                if (rel > worstrel) worstrel = rel;
            }
            std::printf("causal forward_tiled maxabs=%.3e (vl=%d dm=%d)\n", maxabs, vl, dm);
            std::printf("causal forward_tiled worst per-frame rel=%.3e\n", worstrel);
            if (worstrel > 1e-2) {
                std::fprintf(stderr, "causal forward_tiled parity FAIL rel=%.3e maxabs=%.3e\n", worstrel, maxabs);
                return 1;
            }
        }
    }

    return 0;
}
