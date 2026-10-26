#!/usr/bin/env python3
import json, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vr
R = json.load(open(f"{vr.ROOT}/results/seg.json", "r")); M = R["meta"]
res = {
    rid: {
        (k.split("|")[0], k.split("|")[1], float(k.split("|")[2])): np.array(v)
        for k, v in rdict.items()
    }
    for rid, rdict in R["res"].items()
}
DOMS = ["vox", "ami", "ava", "music", "noise", "esc"]
rng = np.random.default_rng(11)
IDS = {d: [m["id"] for m in M if m["domain"] == d] for d in DOMS}; HRS = {d: sum(m["dur"] for m in M if m["domain"] == d) / 3600 for d in DOMS}
BI = {d: rng.integers(0, len(IDS[d]), size=(2000, len(IDS[d]))) for d in DOMS}
def vals(dom, key, col):
    return np.array([res[i][key][col] for i in IDS[dom]]), np.array([m["dur"] for m in M if m["domain"] == dom]) / 3600
def rate(dom, key, col, per="hour"):
    v, h = vals(dom, key, col); est = v.sum() / h.sum(); bs = v[BI[dom]].sum(1) / h[BI[dom]].sum(1)
    l, u = np.percentile(bs, [2.5, 97.5]); return est, l, u
def lost(dom, key):
    sp, _ = vals(dom, key, 2); ins, _ = vals(dom, key, 3)
    if sp.sum() == 0: return None
    est = 1 - ins.sum() / sp.sum(); bs = 1 - ins[BI[dom]].sum(1) / sp[BI[dom]].sum(1); l, u = np.percentile(bs, [2.5, 97.5]); return est, l, u
f = lambda t, d=0: f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"
out = ["# Segmentation: non-speech sent to the decoder (segment_by_vad replica, verified against the CLI), per hour of audio", ""]
SYSN = ["silero", "silero_t03", "silero_t02", "silero_t01", "ultra", "redux", "ultra_t09", "redux_t09", "fusion_ultra", "fusion_redux", "fusiontuned_ultra", "fusiontuned_redux", "gate_ultra", "gate_redux", "gate98_redux", "orgate_ultra", "orgate_redux", "oracle"]
out += ["## S1. Seconds of reference non-speech inside the decoded segments, per hour of audio (95% CI over recordings), old behaviour (trim 0) -> default (trim 0.3). Last column pair: share of reference speech outside the segments (speech lost), trim 0.3", ""]
out += ["| system | " + " | ".join(f"{d} trim0 | {d} trim.3 | {d} lost %" for d in DOMS[:3]) + " | " + " | ".join(f"{d} trim0 | {d} trim.3" for d in DOMS[3:]) + " |", "|---|" + "---|" * (9 + 6)]
for s in SYSN:
    row = []
    for d in DOMS[:3]:
        a = rate(d, (s, "base", 0.0), 1); b = rate(d, (s, "base", 0.3), 1); lo = lost(d, (s, "base", 0.3))
        row += [f(a), f(b), f"{100*lo[0]:.1f} [{100*lo[1]:.1f}, {100*lo[2]:.1f}]"]
    for d in DOMS[3:]:
        a = rate(d, (s, "base", 0.0), 1); b = rate(d, (s, "base", 0.3), 1); row += [f(a), f(b)]
    out.append(f"| {s} | " + " | ".join(row) + " |")
out += ["", "Reading: for music/noise/esc every decoded second is non-speech; 3600 means the whole hour was decoded. Short files (<= 30 s) are never cut or trimmed by the segmenter (not present here: all files are >= 6 min).", ""]
out += ["## S2. Cut policy (trim 0.3): non-speech seconds decoded per hour [95% CI], speech lost %, number of segments per hour, hard cuts per hour and cuts that land in reference speech per hour", ""]
POLS = ["base", "soft", "longest", "gap5", "gap2"]
out += ["policies: base = shipped rule; soft = shipped rule, but before a hard cut take the midpoint of the longest silent gap of any length in the window; longest = cut at the longest pause (>= min pause) in the window; gap5 / gap2 = shipped rule plus a cut at every pause of at least 5 s / 2 s (decoded segments can then be shorter than 1 s apart).", ""]
for d in DOMS[:3]:
    out += [f"### {d} ({HRS[d]:.2f} h)", "", "| detector | policy | non-speech s/h | speech lost % | segments/h | hard cuts/h | cuts in ref speech/h (all) | hard cuts in ref speech/h |", "|---|---|---|---|---|---|---|---|"]
    for s in ("silero", "ultra", "redux", "fusion_redux", "gate_redux", "oracle"):
        for p in POLS:
            k = (s, p, 0.3)
            if k not in res[IDS[d][0]]: continue
            lo = lost(d, k)
            out.append(f"| {s} | {p} | {f(rate(d, k, 1))} | {100*lo[0]:.1f} [{100*lo[1]:.1f}, {100*lo[2]:.1f}] | {rate(d, k, 4)[0]:.0f} | {rate(d, k, 5)[0]:.1f} | {rate(d, k, 7)[0]:.1f} | {rate(d, k, 8)[0]:.1f} |")
    out.append("")
out += ["## S3. Where the decoded non-speech sits (trim 0.3, shipped policy), measured on the detector's own speech mask, seconds per hour: trim margins at segment ends, internal pauses of <1 s, 1-5 s and >= 5 s that stay inside a segment", "", "| domain | detector | edge margins | internal <1 s | internal 1-5 s | internal >= 5 s | total non-speech by detector mask |", "|---|---|---|---|---|---|---|"]
for d in DOMS[:3]:
    for s in ("silero", "ultra", "redux", "fusion_redux", "gate_redux"):
        k = (s, "base", 0.3); out.append(f"| {d} | {s} | " + " | ".join(f"{rate(d, k, c)[0]:.0f}" for c in (9, 10, 11, 12, 13)) + " |")
open(f"{vr.ROOT}/results/seg.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
