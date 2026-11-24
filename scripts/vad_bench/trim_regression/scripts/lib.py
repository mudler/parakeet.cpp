import os, sys, json
import numpy as np, soundfile as sf, jiwer
sys.path.insert(0, os.path.dirname(__file__)); from common import *; from seg import *; from errio import err_to_json, err_from_json; import dec, variants
MAN = {m["name"]: m for m in manifest()}
def total_sec(name):
    m = MAN[name]
    if "dur" in m and m["kind"] != "insert": return m["dur"]
    return sf.info(f"{W}/data/{name}.wav").frames / 16000
def segs_for(det, name, vn): return segments(probs(det, name), total_sec(name), OPTS[det], variants.get(vn, det))
def hyp(det, name, segs):
    """tokens: list of (token, start, end, seg_idx, conf)"""
    cache = dec.load_cache(det, name); out = []
    for i, (s, e) in enumerate(segs):
        c = cache[dec.key(s, e)]
        for w in c["words"]:
            for t in norm(w[3]): out.append((t, w[0], w[1], i, w[2]))
    return out
def align(ref, hyp_tokens):
    """returns events: list of (kind, ref_idx or None, hyp_idx or None), kind in C,S,D,I"""
    h = [t[0] for t in hyp_tokens]
    if not h: return [("D", j, None) for j in range(len(ref))]
    r = jiwer.process_words(" ".join(ref), " ".join(h))
    ev = []
    for ch in r.alignments[0]:
        if ch.type == "equal":
            ev += [("C", ch.ref_start_idx + k, ch.hyp_start_idx + k) for k in range(ch.ref_end_idx - ch.ref_start_idx)]
        elif ch.type == "substitute":
            ev += [("S", ch.ref_start_idx + k, ch.hyp_start_idx + k) for k in range(ch.ref_end_idx - ch.ref_start_idx)]
        elif ch.type == "delete":
            ev += [("D", ch.ref_start_idx + k, None) for k in range(ch.ref_end_idx - ch.ref_start_idx)]
        else:
            ev += [("I", None, ch.hyp_start_idx + k) for k in range(ch.hyp_end_idx - ch.hyp_start_idx)]
    return ev
def utt_index(m):
    """ref word idx -> utterance idx for sinr files"""
    idx = []
    for u, t in enumerate(m["utts"]): idx += [u] * len(norm(t))
    return idx
BLOCK = 100
def unit_of(m, ref_idx, uidx):
    if m["kind"] == "sinr": return (m["split"], m["layout"], uidx[ref_idx])
    return (m["name"], ref_idx // BLOCK)
_cache = {}
def file_errors(det, name, vn):
    """dict unit -> [errors, nref], plus counts S,D,I and events with times. cached on disk."""
    key = (det, name, vn)
    f = f"{W}/cache/err/{det}/{vn}/{name}.json"
    if os.path.exists(f): return err_from_json(json.load(open(f)))
    m = MAN[name]; ref = norm(m["ref"]); segs = segs_for(det, name, vn); hy = hyp(det, name, segs); ev = align(ref, hy)
    uidx = utt_index(m) if m["kind"] == "sinr" else None
    units = {}; sdi = {"S": 0, "D": 0, "I": 0}; events = []
    last_ref = 0
    for kind, rj, hj in ev:
        if rj is not None: last_ref = rj
        if kind == "C":
            u = unit_of(m, rj, uidx); units.setdefault(u, [0, 0])[1] += 1; continue
        j = rj if rj is not None else last_ref
        u = unit_of(m, min(j, len(ref) - 1), uidx); units.setdefault(u, [0, 0])
        units[u][0] += 1
        if rj is not None: units[u][1] += 1
        sdi[kind] += 1; events.append((kind, rj, hj))
    res = dict(units=units, sdi=sdi, events=events, nref=len(ref), nhyp=len(hy), nseg=len(segs))
    os.makedirs(os.path.dirname(f), exist_ok=True); json.dump(err_to_json(res), open(f, "w"))
    return res
def boot_diff(A, B, B_n=5000, seed=1):
    """A, B: arrays [n_units, 2] (errors, nref) for the same units. Paired bootstrap of WER(B) - WER(A) in percentage points."""
    A = np.asarray(A, float); B = np.asarray(B, float); n = len(A)
    rng = np.random.default_rng(seed); idx = rng.integers(0, n, size=(B_n, n))
    ea, na, eb, nb = A[idx, 0].sum(1), A[idx, 1].sum(1), B[idx, 0].sum(1), B[idx, 1].sum(1)
    d = 100 * (eb / nb - ea / na)
    pt = 100 * (B[:, 0].sum() / B[:, 1].sum() - A[:, 0].sum() / A[:, 1].sum())
    return pt, np.percentile(d, 2.5), np.percentile(d, 97.5)
def boot_ci(A, B_n=5000, seed=1):
    A = np.asarray(A, float); n = len(A); rng = np.random.default_rng(seed); idx = rng.integers(0, n, size=(B_n, n))
    w = 100 * A[idx, 0].sum(1) / A[idx, 1].sum(1)
    return 100 * A[:, 0].sum() / A[:, 1].sum(), np.percentile(w, 2.5), np.percentile(w, 97.5)
def group_units(det, vn, names):
    """merge unit dicts of several files; returns (unit keys, array [n,2]) with consistent order, summed S/D/I"""
    tot = {}; sdi = {"S": 0, "D": 0, "I": 0}
    for nm in names:
        r = file_errors(det, nm, vn)
        for u, (e, n) in r["units"].items():
            t = tot.setdefault(u, [0, 0]); t[0] += e; t[1] += n
        for k in sdi: sdi[k] += r["sdi"][k]
    return tot, sdi
def paired(det, va, vb, names):
    ta, sa = group_units(det, va, names); tb, sb = group_units(det, vb, names)
    keys = sorted(set(ta) | set(tb), key=str)
    A = [ta.get(k, [0, 0]) for k in keys]; B = [tb.get(k, [0, 0]) for k in keys]
    return A, B, sa, sb
