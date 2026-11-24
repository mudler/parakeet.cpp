#!/usr/bin/env python3
"""Frame-level tables (markdown) from results/counts.json: speech domains P/R/F1 with bootstrap CIs (recordings resampled within domain), tuned-on-tune/held-out, leave-one-domain-out, non-speech false alarms."""
import sys, os, numpy as np, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rl
from rl import *
HEADS = ("ultra", "redux")
out = []
def P(*a): out.append(" ".join(str(x) for x in a))
def groups(split=None, doms=SPEECH_DOM): return [sel((d,), split) for d in doms]
def cell(spec, g, which=2):
    pt, bs = stat(spec, g, which); return fmt(pt, bs)
def rowtxt(name, spec, split=None):
    cells = [cell(spec, [sel((d,), split)]) for d in SPEECH_DOM]
    gall = groups(split)
    pooled = cell(spec, gall, 2); pp = cell(spec, gall, 0); rr = cell(spec, gall, 1)
    return f"| {name} | " + " | ".join(cells) + f" | {pooled} | {pp} | {rr} |"
HDR = "| system | vox F1 | ami F1 | ava F1 | pooled F1 | pooled P | pooled R |\n|---|---|---|---|---|---|---|"
# ---------------- tuning on the tune split (macro F1 over the three domains)
T = {}      # thr/params only, default post
T["sil"] = tune([s for s in SP if s[0] == "sil"])
T["silnat"] = tune([s for s in SP if s[0] == "nat" and s[1] == "silero"])
for h in HEADS:
    for k in ("head", "or", "and", "mean", "two"):
        T[(k, h)] = tune([s for s in SP if s[0] == k and s[1] == h])
    T[("twog", h)] = tune([s for s in SP if s[0] == "twog" and s[1] == h])
    T[("gate", h)] = tune([s for s in SP if s[0] == "gate" and s[1] == h])
def ppfam(kind, h=None): return [s for s in SP if s[0] == "pp" and s[4][0] == kind and (h is None or s[4][1] == h)]
TP = {}     # params + post-processing (min speech, min pause, pad) tuned jointly
TP["sil"] = tune(ppfam("sil"))
for h in HEADS:
    for k in ("head", "or", "mean", "two", "gate", "org"):
        TP[(k, h)] = tune(ppfam(k, h))
P("# Tuned parameters (chosen on the tune split only, objective = mean F1 of vox, ami, ava)")
for k, v in T.items(): P(f"- {k}: {v}  (tune F1 {100*macro_f1(v,'tune'):.2f}, held F1 {100*macro_f1(v,'held'):.2f})")
P("\n# Tuned parameters incl. post-processing (ms, mp, pad), chosen on the tune split only")
for k, v in TP.items(): P(f"- {k}: {v}  (tune F1 {100*macro_f1(v,'tune'):.2f}, held F1 {100*macro_f1(v,'held'):.2f})")
json.dump({str(k): list(map(lambda x: float(x) if isinstance(x, (float, np.floating)) else x, v)) for k, v in T.items()}, open(f"{rl.vr.ROOT}/results/tuned.json", "w"))
DEF = {"sil": ("sil", 0.5), "silnat": ("nat", "silero", 0.5, 0.25, 0.1, 0.03)}
FDEF = lambda h: ("two", h, 0.5, 0.5, 16, 16, 30)
def block(title, split, include_tuned):
    P(f"\n## {title}\n"); P(HDR)
    P(rowtxt("Silero, own defaults (thr .5, min speech .25, pause .1, pad .03)", DEF["silnat"], split))
    P(rowtxt("Silero thr .5, unified post", DEF["sil"], split))
    for h in HEADS:
        P(rowtxt(f"{h} head thr .5 (own defaults)", ("head", h, 0.5), split))
    for h in HEADS:
        P(rowtxt(f"OR(Silero .5, {h} .5)", ("or", h, 0.5, 0.5), split))
        P(rowtxt(f"AND(Silero .5, {h} .5)", ("and", h, 0.5, 0.5), split))
        P(rowtxt(f"mean(Silero, {h}) >= .5", ("mean", h, 0.5, 0.5), split))
        P(rowtxt(f"two-stage fusion {h}, defaults (ts .5, th .5, pre 160 ms, post 160 ms, fill 300 ms)", FDEF(h), split))
    for h in HEADS:
        P(rowtxt(f"{h} head + median gate 0.92", ("gate", h, 0.92), split))
        P(rowtxt(f"two-stage fusion {h} with gated head (.92)", ("twog", h, 0.92, 16, 16, 30), split))
        P(rowtxt(f"OR(Silero .5, gated {h} .92)", ("org", h, 0.92), split))
    if include_tuned:
        P(rowtxt(f"Silero thr tuned {T['sil'][1]}", T["sil"], split))
        P(rowtxt(f"Silero all options tuned {T['silnat'][2:]}", T["silnat"], split))
        for h in HEADS:
            for k, nm in (("head", "thr"), ("or", "OR"), ("and", "AND"), ("mean", "mean"), ("two", "two-stage")):
                P(rowtxt(f"{h} {nm} tuned {T[(k,h)][1:] if k!='head' else T[(k,h)][2]}", T[(k, h)], split))
        P(rowtxt(f"Silero thr + post tuned {TP['sil'][1:4]} {TP['sil'][4][1]}", TP["sil"], split))
        for h in HEADS:
            for k, nm in (("head", "head thr"), ("or", "OR"), ("mean", "mean"), ("two", "two-stage fusion"), ("gate", "median gate"), ("org", "OR with gated head")):
                P(rowtxt(f"{h} {nm}, params + post tuned {TP[(k,h)][1:4]} {TP[(k,h)][4][1:]}", TP[(k, h)], split))
block("A. All recordings, untuned systems (no parameter was fitted to these; 9.9 h of speech-bearing audio)", None, False)
block("B. Held-out recordings only (vox test, ami test, ava hash split), untuned and tuned-on-tune systems", "held", True)
# ---------------- paired deltas
P("\n## C. Paired differences in F1 points (bootstrap over recordings, 95% CI), pooled and per domain\n")
P("| comparison | split | vox | ami | ava | pooled |\n|---|---|---|---|---|---|")
def drow(name, a, b, split):
    cs = []
    for g in [[sel((d,), split)] for d in SPEECH_DOM] + [groups(split)]:
        pt, bs = delta(a, b, g); cs.append(fmt(pt, bs, 100, 2))
    P(f"| {name} | {split or 'all'} | " + " | ".join(cs) + " |")
best_single_def = {}
for h in HEADS:
    drow(f"fusion defaults {h} minus {h} head .5", FDEF(h), ("head", h, 0.5), None)
    drow(f"fusion defaults {h} minus Silero own defaults", FDEF(h), DEF["silnat"], None)
    drow(f"{h} head .5 minus Silero own defaults", ("head", h, 0.5), DEF["silnat"], None)
    drow(f"gate .92 {h} minus {h} head .5", ("gate", h, 0.92), ("head", h, 0.5), None)
for h in HEADS:
    drow(f"fusion tuned {h} (thr/params only, default post) minus {h} head tuned (thr only)", T[("two", h)], T[("head", h)], "held")
    drow(f"fusion tuned {h} (thr/params only, default post) minus Silero thr tuned (default post)", T[("two", h)], T["sil"], "held")
P("\n### C2. Paired differences, everything tuned jointly with post-processing (held-out split)\n")
P("| comparison | split | vox | ami | ava | pooled |\n|---|---|---|---|---|---|")
for h in HEADS:
    best = max([TP["sil"], TP[("head", h)]], key=lambda s: macro_f1(s, "tune"))
    for k, nm in (("two", "fusion"), ("gate", "gated head"), ("org", "OR with gated head"), ("or", "OR"), ("mean", "mean")):
        drow(f"{h} {nm} {TP[(k,h)][1:4]} minus best single detector tuned on tune ({'Silero' if best==TP['sil'] else h+' head'})", TP[(k, h)], best, "held")
    drow(f"{h} head tuned minus Silero tuned", TP[("head", h)], TP["sil"], "held")
    drow(f"{h} fusion minus Silero tuned", TP[("two", h)], TP["sil"], "held")
# ---------------- leave one domain out
P("\n## D. Leave-one-domain-out: parameters tuned on the other two domains (all their recordings), scored on the left-out domain (F1 and 95% CI)\n")
P("| system | vox (tuned on ami+ava) | ami (tuned on vox+ava) | ava (tuned on vox+ami) | mean |\n|---|---|---|---|---|")
def lodo_pick(cands, hold):
    others = [d for d in SPEECH_DOM if d != hold]
    def sc(s):
        f = []
        for d in others:
            ii = sel((d,)); c = C[ii, IDX[s], :4].sum(0); f.append(f1_of(c)[2])
        return np.mean(f)
    return max(cands, key=sc)
fams = {"Silero thr": [s for s in SP if s[0] == "sil"], "Silero all options": ppfam("sil")}
for h in HEADS:
    for k, nm in (("head", "head thr"), ("or", "OR"), ("mean", "mean"), ("two", "two-stage")):
        fams[f"{h} {nm}"] = [s for s in SP if s[0] == k and s[1] == h]
        fams[f"{h} {nm} + post"] = ppfam(k, h)
    fams[f"{h} gate + post"] = ppfam("gate", h)
LODO = {}
for nm, cands in fams.items():
    cs = []; vals = []; picks = []
    for d in SPEECH_DOM:
        b = lodo_pick(cands, d); picks.append(b); pt, bs = stat(b, [sel((d,))]); cs.append(fmt(pt, bs)); vals.append(pt)
    LODO[nm] = picks
    P(f"| {nm} | " + " | ".join(cs) + f" | {100*np.mean(vals):.1f} |")
P("\nPicks: " + "; ".join(f"{k}: {[tuple(x[1:]) if len(x)>2 else x for x in v]}" for k, v in LODO.items()))
def others_score(spec, hold):
    f = []
    for d in SPEECH_DOM:
        if d == hold: continue
        ii = sel((d,)); c = C[ii, IDX[spec], :4].sum(0); f.append(f1_of(c)[2])
    return np.mean(f)
P("\nLODO paired deltas in F1 points (left-out domain; the best single detector is chosen by its score on the two tuning domains; CI over recordings of the left-out domain):\n")
P("| comparison | vox | ami | ava | mean of three |\n|---|---|---|---|---|")
for h in HEADS:
    for fam in ("two-stage", "gate", "OR", "mean"):
        key_a = f"{h} {fam} + post" if fam != "gate" else f"{h} gate + post"
        if key_a not in LODO: continue
        for withpost in (True,):
            cs = []; vals = []
            for d_i, d in enumerate(SPEECH_DOM):
                a = LODO[key_a][d_i]
                singles = [LODO["Silero all options"][d_i], LODO[f"{h} head thr + post"][d_i]]
                bsel = max(singles, key=lambda x: others_score(x, d))
                pt, bs = delta(a, bsel, [sel((d,))]); cs.append(fmt(pt, bs, 100, 2)); vals.append(pt * 100)
            P(f"| {h} {fam} (+post) minus best single (+post) | " + " | ".join(cs) + f" | {np.mean(vals):.2f} |")
    cs = []; vals = []
    for d_i, d in enumerate(SPEECH_DOM):
        pt, bs = delta(LODO[f"{h} head thr + post"][d_i], LODO["Silero all options"][d_i], [sel((d,))]); cs.append(fmt(pt, bs, 100, 2)); vals.append(pt * 100)
    P(f"| {h} head (+post) minus Silero (+post) | " + " | ".join(cs) + f" | {np.mean(vals):.2f} |")
# ---------------- gate sweep
P("\n## E. Median-probability gate sweep (head thr .5, keep a run if its median probability >= m; unified post), all recordings\n")
P("| head | m | vox F1 | ami F1 | ava F1 | pooled F1 | music FA s/h | noise FA s/h | esc FA s/h |\n|---|---|---|---|---|---|---|---|---|")
def fa_rate(spec, dom_list, g=None):
    """speech seconds called per hour of audio, bootstrap over files (ratio of sums)"""
    ii = sel(dom_list) if g is None else g; k = IDX[spec]
    fp = (C[ii, k, 0] + C[ii, k, 1]); n = C[ii, k, :4].sum(1)   # tp is 0 on non-speech recordings; fp = predicted cells
    rng = np.random.default_rng(7); bi = rng.integers(0, len(ii), size=(2000, len(ii)))
    est = fp.sum() * 0.01 / (n.sum() * 0.01 / 3600); bs = fp[bi].sum(1) / n[bi].sum(1) * 3600
    l, h = np.percentile(bs, [2.5, 97.5]); return f"{est:.0f} [{l:.0f}, {h:.0f}]"
for h in HEADS:
    for m in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99):
        sp = ("gate", h, m); g = groups(None)
        P(f"| {h} | {m} | " + " | ".join(cell(sp, [sel((d,))]) for d in SPEECH_DOM) + f" | {cell(sp, g)} | " + " | ".join(fa_rate(sp, (d,)) for d in NS_DOM) + " |")
# ---------------- non-speech FA
P("\n## F. Non-speech audio: seconds called speech per hour of audio and speech regions per hour (95% CI over files; domains: music 16 files, noise 6, esc 5)\n")
P("| system | music s/h | noise s/h | esc s/h | pooled s/h | pooled regions/h |\n|---|---|---|---|---|---|")
def regions_rate(spec, doms):
    ii = sel(doms); k = IDX[spec]; nr = C[ii, k, 4]; n = C[ii, k, :4].sum(1)
    rng = np.random.default_rng(8); bi = rng.integers(0, len(ii), size=(2000, len(ii)))
    est = nr.sum() / (n.sum() * 0.01 / 3600); bs = nr[bi].sum(1) / (n[bi].sum(1) * 0.01 / 3600)
    l, h = np.percentile(bs, [2.5, 97.5]); return f"{est:.0f} [{l:.0f}, {h:.0f}]"
NSR = [("Silero own defaults", DEF["silnat"]), ("Silero thr .5 unified", DEF["sil"]), (f"Silero tuned thr {T['sil'][1]}", T["sil"]), (f"Silero all options tuned", T["silnat"])]
for h in HEADS:
    NSR += [(f"{h} head .5", ("head", h, 0.5)), (f"{h} head tuned {T[('head',h)][2]}", T[("head", h)]), (f"OR(S .5,{h} .5)", ("or", h, 0.5, 0.5)), (f"AND(S .5,{h} .5)", ("and", h, 0.5, 0.5)),
            (f"fusion defaults {h}", FDEF(h)), (f"fusion tuned {h} {T[('two',h)][2:]}", T[("two", h)]), (f"{h} gate .92", ("gate", h, 0.92)), (f"fusion with gated head {h}", ("twog", h, 0.92, 16, 16, 30)),
            (f"OR(S .5, gated {h})", ("org", h, 0.92))]
NSR += [("Silero thr+post tuned " + str(TP['sil'][1:4]) + " thr " + str(TP['sil'][4][1]), TP["sil"])]
for h in HEADS:
    for k, nm in (("head", "head thr"), ("two", "fusion"), ("gate", "median gate"), ("org", "OR with gated head")):
        NSR.append((f"{h} {nm} thr/params+post tuned {TP[(k,h)][1:4]}", TP[(k, h)]))
for nm, sp in NSR:
    P(f"| {nm} | " + " | ".join(fa_rate(sp, (d,)) for d in NS_DOM) + f" | {fa_rate(sp, NS_DOM)} | {regions_rate(sp, NS_DOM)} |")
# music by source
open(f"{rl.vr.ROOT}/results/frames.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
