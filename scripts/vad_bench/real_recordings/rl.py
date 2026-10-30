"""results loader + bootstrap helpers"""
import json, numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vr

def _to_tuple(x):
    if isinstance(x, list):
        return tuple(_to_tuple(i) for i in x)
    return x

R = json.load(open(f"{vr.ROOT}/results/counts.json", "r"))
SP = [_to_tuple(s) for s in R["specs"]]
C = np.array(R["counts"])
M = R["meta"]
IDX = {s: i for i, s in enumerate(SP)}
SPEECH_DOM = ("vox", "ami", "ava"); NS_DOM = ("music", "noise", "esc")
def sel(domains=None, split=None):
    return np.array([i for i, m in enumerate(M) if (domains is None or m["domain"] in domains) and (split is None or m["split"] == split)])
def f1_of(c):
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    P = tp / np.maximum(1, tp + fp); Rr = tp / np.maximum(1, tp + fn)
    return P, Rr, 2 * P * Rr / np.maximum(1e-12, P + Rr)
def macro_f1(spec, split):
    f = []
    for d in SPEECH_DOM:
        ii = sel((d,), split); c = C[ii, IDX[spec], :4].sum(0); f.append(f1_of(c)[2])
    return float(np.mean(f))
def tune(cands, split="tune"):
    best = max(cands, key=lambda s: macro_f1(s, split)); return best
B = 2000
_rng = np.random.default_rng(12345)
_BI = {}
def boot_indices(ii_by_group):
    """stratified bootstrap indices, cached per group-key so that every system uses the same resamples (paired)."""
    key = tuple(tuple(g) for g in ii_by_group)
    if key not in _BI:
        _BI[key] = [ii[_rng.integers(0, len(ii), size=(B, len(ii)))] for ii in ii_by_group]
    return _BI[key]
def boot_counts(spec, groups):
    """returns (B,4) bootstrap pooled counts (sum over groups) and point counts (4,)"""
    bi = boot_indices(groups); k = IDX[spec]
    tot = sum(C[b, k, :4].sum(1) for b in bi)
    pt = sum(C[g, k, :4].sum(0) for g in groups)
    return tot, pt
def ci(x, lo=2.5, hi=97.5): return np.percentile(x, [lo, hi])
def fmt(point, bs, scale=100, d=1):
    l, h = ci(bs); return f"{point*scale:.{d}f} [{l*scale:.{d}f}, {h*scale:.{d}f}]"
def stat(spec, groups, which=2):
    tot, pt = boot_counts(spec, groups)
    return f1_of(pt)[which], f1_of(tot)[which]
def delta(spec_a, spec_b, groups, which=2):
    ta, pa = boot_counts(spec_a, groups); tb, pb = boot_counts(spec_b, groups)
    return f1_of(pa)[which] - f1_of(pb)[which], f1_of(ta)[which] - f1_of(tb)[which]
