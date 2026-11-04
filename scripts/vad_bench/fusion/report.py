from analyze import *
import rules
out=[]
KL=(("lr2","5a logistic reg, 2 probs"),("lr6","5b logistic reg, + prev/next frame"),("hgb6","5c gradient boosting, + prev/next frame"),("lr6nz","5d as 5b, trained also on noise-only clips"),("hgb6nz","5e as 5c, trained also on noise-only clips"))
KINDS=[k for k,_ in KL]
def P_(s=""): out.append(s)
def crossfit_fixed(getarr,thr_idx_of):
    rows=np.zeros((N,5))
    for f in range(4):
        ps,arr=getarr("f%d"%f); te=np.flatnonzero(fold==f); rows[te]=arr[thr_idx_of(ps)][te]
    return rows
def shiftfit(getarr,crit):
    return crossfit(getarr,crit,"shift")
sysdefs={}   # name -> (rows, note)
def build(h):
    S={}
    S["Silero thr .5 (unified post)"]=(A_of("silero",(0.5,),h),"fixed")
    S["Silero thr .5 (native defaults: min speech .25, pause .1, pad .03)"]=(CFG[("silero",(0.5,),"ultra",0.25)],"fixed")
    S[f"{h} head thr .5"]=(A_of("head",(0.5,),h),"fixed")
    S["Silero tuned thr (CV)"]=crossfit(static("silero",h),"f1")[0],"cv"
    S[f"{h} head tuned thr (CV)"]=crossfit(static("head",h),"f1")[0],"cv"
    dflt=dict(OR=("or",(0.5,0.5)),AND=("and",(0.5,0.5)),MEAN=("mean",(0.5,0.5)),MAX=("max",(0.5,)),MIN=("min",(0.5,)),TWO=("two",(0.5,0.5,16,16,30)),SWITCH=("switch",(0.5,0.5,12)))
    names=dict(OR="1 OR of decisions",AND="2 AND of decisions",MEAN="3 mean of probs (w=.5)",MAX="4a max of probs",MIN="4b min of probs",TWO="6 two-stage (Silero + head extend/fill)",SWITCH="7 noise-switch (estimated SNR)")
    for k,(r,p) in dflt.items(): S[f"{names[k]} @ defaults {p}"]=(A_of(r,p,h),"fixed")
    for kind,nm in KL:
        S[f"{nm} @ thr .5"]=(crossfit_fixed(lrget(h,kind),lambda ps:ps.index((0.5,))),"cv-model")
    for k in ("OR","AND","MAX","MIN","TWO","SWITCH"):
        r=dflt[k][0]; S[f"{names[k]} tuned (CV)"]=crossfit(static(r,h),"f1")[0],"cv"
    # mean: separate weights are in the same candidate set
    S[f"{names['MEAN']} tuned w,thr (CV)"]=crossfit(static("mean",h),"f1")[0],"cv"
    for kind,nm in KL:
        S[f"{nm} tuned thr (CV)"]=crossfit(lrget(h,kind),"f1")[0],"cv"
    S["7o oracle noise-switch (true SNR) tuned (CV)"]=crossfit(static("oracle",h),"f1")[0],"cv"
    return S
def table(S,title):
    P_(f"### {title}"); P_()
    P_("pooled over 342 speech clips; point [95% bootstrap CI over clips], percent. Per-level columns: F1 with CI (white+pink pooled).")
    P_()
    P_("| system | P | R | F1 | F1 clean | F1 20 dB | F1 10 dB | F1 5 dB | F1 0 dB |"); P_("|---|---|---|---|---|---|---|---|---|")
    for n,(rows,_) in S.items():
        s=stats(rows); a=s["all"]
        P_(f"| {n} | {fmt(a['P'],*a['Pci'])} | {fmt(a['R'],*a['Rci'])} | {fmt(a['F'],*a['Fci'])} | "+" | ".join(fmt(s[l]['F'],*s[l]['Fci']) for l,_ in LEVELS)+" |")
    P_()
def table_pr_levels(S,title,names):
    P_(f"### {title}"); P_(); P_("| system | "+" | ".join(f"P/R {l}" for l,_ in LEVELS)+" |"); P_("|---|"+"---|"*len(LEVELS))
    for n in names:
        s=stats(S[n][0]); P_(f"| {n} | "+" | ".join(f"{100*s[l]['P']:.1f}/{100*s[l]['R']:.1f}" for l,_ in LEVELS)+" |")
    P_()
def fa_table(S,names,title):
    P_(f"### {title}"); P_(); P_("noise-only clips (12 x 30 s per level, white+pink pooled over the levels shown). False-alarm segments per hour (frame false-positive rate in brackets).")
    P_(); P_("| system | "+" | ".join(l for l,_ in LEVELS[1:])+" | all |"); P_("|---|"+"---|"*5)
    for n in names:
        f=fa(S[n][0]); P_(f"| {n} | "+" | ".join(f"{f[l][0]:.0f} ({100*f[l][1]:.2f}%)" for l,_ in LEVELS[1:])+f" | {f['all'][0]:.0f} ({100*f['all'][1]:.2f}%) |")
    P_()
ALL={}
for h in HEADS:
    S=build(h); ALL[h]=S
    P_(f"## Head = {h}"); P_()
    table(S,f"Cross-validated results ({h} head + Silero; speaker-disjoint 4-fold; 'tuned' = parameters chosen on the 3 training folds, scored on the held-out fold)")
    tuned=[n for n in S if "tuned" in n or "@" in n or "head thr" in n or "Silero thr" in n]
    fa_table(S,[n for n in S],"False alarms on noise-only clips")
    # best fusion vs best single
    singles=["Silero tuned thr (CV)",f"{h} head tuned thr (CV)"]
    fus=[n for n in S if n not in singles and "thr .5 (" not in n and "head thr" not in n and "native" not in n and "oracle" not in n]
    bs=max(singles,key=lambda n:stats(S[n][0])["all"]["F"]); 
    P_(f"**Best single (CV-tuned):** {bs}."); P_()
    P_("| fusion (CV) | dF1 vs best single, pooled | dF1 5 dB | dF1 0 dB | dR pooled | dP pooled |"); P_("|---|---|---|---|---|---|")
    b0=stats(S[bs][0])
    for n in fus:
        s=stats(S[n][0]); d=s["all"]["F_b"]-b0["all"]["F_b"]
        # recall / precision diff, paired
        idx,bi=bidx("all",np.flatnonzero(sp)); Pb,Rb,_=prf_b(S[n][0],idx,bi); Pb0,Rb0,_=prf_b(S[bs][0],idx,bi)
        l5=s["5 dB"]["F_b"]-b0["5 dB"]["F_b"]; l0=s["0 dB"]["F_b"]-b0["0 dB"]["F_b"]
        f=lambda x,y: f"{100*x:+.2f} [{100*y[0]:+.2f},{100*y[1]:+.2f}]"
        P_(f"| {n} | {f(s['all']['F']-b0['all']['F'],ci(d))} | {f(s['5 dB']['F']-b0['5 dB']['F'],ci(l5))} | {f(s['0 dB']['F']-b0['0 dB']['F'],ci(l0))} | {f(s['all']['R']-b0['all']['R'],ci(Rb-Rb0))} | {f(s['all']['P']-b0['all']['P'],ci(Pb-Pb0))} |")
    P_()
    # shift generalisation
    P_(f"### Noise-shift split ({h}): parameters chosen on clean/20/10 dB clips only, scored on 5 and 0 dB clips"); P_()
    P_("| system | P | R | F1 (5+0 dB clips) |"); P_("|---|---|---|---|")
    def shiftrow(nm,getarr):
        rows,ch=shiftfit(getarr,"f1"); idx=np.flatnonzero(sp&(snr<10)); ii=rng.integers(0,len(idx),(B,len(idx))); Pb,Rb,Fb=prf_b(rows,idx,ii); p,r,f=pooled(rows,idx)
        P_(f"| {nm} {ch['shift']} | {fmt(p,*ci(Pb))} | {fmt(r,*ci(Rb))} | {fmt(f,*ci(Fb))} |")
    for r_,nm in (("silero","Silero"),("head",h+" head"),("or","OR"),("and","AND"),("mean","mean"),("max","max"),("min","min"),("two","two-stage"),("switch","noise-switch")):
        shiftrow(nm,static(r_,h))
    for kind in KINDS: shiftrow(kind,lrget(h,kind))
    P_()
    # precision-constrained
    P_(f"### Operating points with precision >= 99% ({h}): CV-selected maximum recall")
    P_(); P_("| system | P | R | F1 | chosen params per fold |"); P_("|---|---|---|---|---|")
    for r_,nm in (("silero","Silero"),("head",h+" head"),("or","OR"),("and","AND"),("mean","mean"),("max","max"),("min","min"),("two","two-stage"),("switch","noise-switch")):
        rows,ch=crossfit(static(r_,h),0.99); s=stats(rows)["all"]; P_(f"| {nm} | {fmt(s['P'],*s['Pci'])} | {fmt(s['R'],*s['Rci'])} | {fmt(s['F'],*s['Fci'])} | {list(ch.values())} |")
    for kind in KINDS:
        rows,ch=crossfit(lrget(h,kind),0.99); s=stats(rows)["all"]; P_(f"| {kind} | {fmt(s['P'],*s['Pci'])} | {fmt(s['R'],*s['Rci'])} | {fmt(s['F'],*s['Fci'])} | thr {[c[0] for c in ch.values()]} |")
    P_()
    # in-sample frontier (descriptive)
    P_(f"### In-sample best operating points ({h}), descriptive only (selected on all clips, optimistic)"); P_()
    P_("| rule | best F1 params | P | R | F1 | max R at P>=99% params | P | R |"); P_("|---|---|---|---|---|---|---|---|")
    allidx=np.flatnonzero(sp)
    for r_ in("silero","head","or","and","mean","max","min","two","switch","oracle"):
        ps,arr=cand(r_,h); i=score(arr,allidx,"f1"); j=score(arr,allidx,0.99)
        a=prf(arr[i][allidx,:4].sum(0)); b=prf(arr[j][allidx,:4].sum(0))
        P_(f"| {r_} | {ps[i]} | {100*a[0]:.1f} | {100*a[1]:.1f} | {100*a[2]:.1f} | {ps[j]} | {100*b[0]:.1f} | {100*b[1]:.1f} |")
    P_()
    # LR coefs
    P_(f"LR coefficients ({h}, fit on folds != 0; features lg(silero), lg(head)[, lg(silero prev/next), lg(head prev/next)]):")
    for kind in("lr2","lr6","lr6nz"): P_(f"- {kind}: {LR[(h,kind,'f0')][1]}")
    P_()
open("results.md","w").write("\n".join(out))
rows_json = {h: {n: v[0].tolist() for n, v in S.items()} for h, S in ALL.items()}
json.dump(rows_json, open("rows.json", "w"))
print("\n".join(out)[:200])
