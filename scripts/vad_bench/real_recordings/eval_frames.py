#!/usr/bin/env python3
"""Compute per-recording confusion counts (tp,fp,fn,tn on 10 ms cells) + number of speech regions for every system spec. -> results/counts.json"""
import os, sys, json, itertools, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vr, systems
THR = [0.02, 0.05, 0.075] + [round(x, 2) for x in np.arange(0.1, 0.951, 0.05)]
THR9 = [round(x, 1) for x in np.arange(0.1, 0.91, 0.1)]
HEADS = ("ultra", "redux")
def specs():
    S = []
    S += [("ref",)]
    S += [("sil", t) for t in THR]
    S += [("nat", "silero", t, ms, mp, pad) for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7) for ms in (0.1, 0.25, 0.5) for mp in (0.1, 0.2, 0.5) for pad in (0.0, 0.03, 0.1, 0.2)]
    for h in HEADS:
        S += [("head", h, t) for t in THR]
        S += [("nat", h, 0.5, 0.1, 0.2, 0.0)]
        S += [("or", h, a, b) for a in THR9 for b in THR9]
        S += [("and", h, a, b) for a in THR9 for b in THR9]
        S += [("mean", h, w, t) for w in (0.25, 0.5, 0.75) for t in THR]
        S += [("two", h, ts, th, pre, po, Y) for ts in (0.1, 0.2, 0.3, 0.5, 0.7) for th in (0.3, 0.5, 0.7, 0.9) for pre in (0, 8, 16, 24, 32) for po in (0, 8, 16, 32) for Y in (0, 30, 60)]
        S += [("gate", h, m) for m in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99)]
        S += [("org", h, m) for m in (0.8, 0.92, 0.96)]
        S += [("twog", h, m, pre, po, Y) for m in (0.92,) for pre, po, Y in ((16, 16, 30), (32, 16, 60), (16, 16, 0), (0, 0, 30), (8, 8, 30))]
    return S
PPG = [(ms, mp, pad) for ms in (0.1, 0.25, 0.5) for mp in (0.1, 0.2, 0.5) for pad in (0.0, 0.03, 0.1, 0.2)]
def specs_pp():
    inner = [("sil", t) for t in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)]
    for h in HEADS:
        inner += [("head", h, t) for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
        inner += [("or", h, a, b) for a in (0.1, 0.3, 0.5) for b in (0.3, 0.5, 0.7, 0.9)]
        inner += [("mean", h, w, t) for w in (0.25, 0.5, 0.75) for t in (0.2, 0.3, 0.4, 0.5)]
        inner += [("two", h, ts, th, pre, po, Y) for ts in (0.1, 0.3, 0.5) for th in (0.3, 0.5, 0.9) for pre in (0, 16, 32) for po in (0, 16, 32) for Y in (0, 30, 60)]
        inner += [("gate", h, m) for m in (0.5, 0.8, 0.9, 0.92, 0.96)]
        inner += [("org", h, m) for m in (0.8, 0.92)]
    return [("pp", ms, mp, pad, i) for i in inner for (ms, mp, pad) in PPG]
def work(meta):
    r = systems.Rec(meta); ref = meta["ref"]; out = []
    for sp in SPECS:
        m = systems.mask(r, sp)
        s, e = vr.runs(m)
        out.append(np.concatenate([vr.counts(m, ref), [len(s)]]))
    return meta["id"], np.array(out, np.int64)
SPECS = specs() + specs_pp()
if __name__ == "__main__":
    D = vr.load_recordings(); print(len(D), "recordings", len(SPECS), "specs", flush=True)
    t = time.time()
    with Pool(10) as p: res = dict(p.map(work, D, chunksize=1))
    os.makedirs(f"{vr.ROOT}/results", exist_ok=True)
    meta = [dict(id=m["id"], domain=m["domain"], split=m["split"], dur=m["dur"], nonspeech=m["nonspeech"]) for m in D]
    counts_data = np.stack([res[m["id"]] for m in D]).tolist()
    json.dump(dict(specs=SPECS, meta=meta, counts=counts_data), open(f"{vr.ROOT}/results/counts.json", "w"))
    print("done", time.time() - t)
