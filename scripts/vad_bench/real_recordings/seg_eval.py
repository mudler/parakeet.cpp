#!/usr/bin/env python3
"""Seconds of non-speech sent to the decoder by the transcribe --vad segmentation (segment_by_vad replica, trim .3 default), per detector/system and cut policy.
-> results/seg.json : per recording, per system: [decoded_s, decoded_nonspeech_s, speech_s, speech_in_segments_s, n_segments, hard_cuts]"""
import os, sys, json, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vr, systems
from vr import FS
FUS = {"ultra": ("two", "ultra", 0.5, 0.5, 16, 16, 30), "redux": ("two", "redux", 0.5, 0.5, 16, 16, 30)}
def system_masks(r):
    """name -> (smoothed speech mask, period, min_pause)"""
    out = {}
    out["silero"] = (vr.smooth_native(r.P["silero"], FS["silero"], r.dur, 0.5, 0.1, 0.25), FS["silero"], 0.1)
    for t in (0.1, 0.2, 0.3):
        out[f"silero_t{int(t*10):02d}"] = (vr.smooth_native(r.P["silero"], FS["silero"], r.dur, t, 0.1, 0.25), FS["silero"], 0.1)
    out["silero_p02"] = (vr.smooth_native(r.P["silero"], FS["silero"], r.dur, 0.5, 0.1, 0.25), FS["silero"], 0.2)
    for h in ("ultra", "redux"):
        out[h] = (vr.smooth_native(r.P[h], FS[h], r.dur, 0.5, 0.1, 0.1), FS[h], 0.2)
        out[f"{h}_t09"] = (vr.smooth_native(r.P[h], FS[h], r.dur, 0.9, 0.1, 0.1), FS[h], 0.2)
        out[f"fusion_{h}"] = (systems.mask(r, FUS[h]), vr.GR, 0.2)
        out[f"fusiontuned_{h}"] = (systems.mask(r, ("two", h, 0.1, 0.3, 32, 8, 30)), vr.GR, 0.2)
        out[f"gate98_{h}"] = (systems.mask(r, ("gate", h, 0.98)), vr.GR, 0.2)
        out[f"gate_{h}"] = (systems.mask(r, ("gate", h, 0.92)), vr.GR, 0.2)
        out[f"orgate_{h}"] = (systems.mask(r, ("org", h, 0.92)), vr.GR, 0.2)
    out["oracle"] = (systems.mask(r, ("ref",)), vr.GR, 0.2)
    return out
POL = [("base", dict()), ("longest", dict(policy="longest")), ("soft", dict(policy="soft")), ("gap2", dict(long_gap=2.0)), ("gap5", dict(long_gap=5.0))]
def work(meta):
    r = systems.Rec(meta); ref = meta["ref"]; n = r.n; res = {}
    M = system_masks(r)
    for name, (sp, fs, mp) in M.items():
        for pol, kw in POL:
            if pol != "base" and name not in ("silero", "ultra", "redux", "fusion_ultra", "fusion_redux", "gate_ultra", "gate_redux"): continue
            for trim in (0.3, 0.0):
                if trim == 0.0 and pol != "base": continue
                segs, hard = vr.segment_by_mask(sp, fs, r.dur, min_pause=mp, trim=trim, **kw)
                sm = vr.seg_mask(segs, n)
                cuts = list(vr.LAST_CUTS)
                in_sp = sum(1 for c, hd in cuts if ref[min(n - 1, int(c / vr.GR))]); hard_in_sp = sum(1 for c, hd in cuts if hd and ref[min(n - 1, int(c / vr.GR))])
                # decoded seconds of detector-non-speech by gap class (edge = trim margin at a segment end; internal gaps by length)
                dec = np.zeros(5)   # edge, int<1s, 1-5s, >=5s, total
                if trim == 0.3 and pol == "base":
                    spm = vr.hold(sp.astype(np.float32), fs, n) >= 0.5 if fs != vr.GR else sp[:n]
                    for a, b in segs:
                        ia, ib = int(round(a / vr.GR)), int(round(b / vr.GR)); seg = spm[ia:ib]
                        if len(seg) == 0: continue
                        ss, ee = vr.runs(~seg)
                        for x, y in zip(ss, ee):
                            L = (y - x) * vr.GR
                            if x == 0 or y == len(seg): dec[0] += L
                            elif L < 1: dec[1] += L
                            elif L < 5: dec[2] += L
                            else: dec[3] += L
                        dec[4] += (~seg).sum() * vr.GR
                res[(name, pol, trim)] = np.concatenate([[sm.sum() * vr.GR, (sm & ~ref).sum() * vr.GR, ref.sum() * vr.GR, (sm & ref).sum() * vr.GR, len(segs), hard, len(cuts), in_sp, hard_in_sp], dec])
    return meta["id"], res
if __name__ == "__main__":
    D = vr.load_recordings(); t = time.time()
    with Pool(8) as p: res = dict(p.map(work, D, chunksize=1))
    res_json = {rid: {f"{k[0]}|{k[1]}|{k[2]}": v.tolist() for k, v in rdict.items()} for rid, rdict in res.items()}
    json.dump(dict(meta=[dict(id=m["id"], domain=m["domain"], split=m["split"], dur=m["dur"], nonspeech=m["nonspeech"]) for m in D], res=res_json), open(f"{vr.ROOT}/results/seg.json", "w"))
    print("done", time.time() - t)
