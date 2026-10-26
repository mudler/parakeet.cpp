import json,sys,numpy as np
from fl import prf
R=json.load(open("results.json","r"))
CFG={(item["rule"],tuple(item["p"]),item["h"],item["ms"]):np.array(item["arr"]) for item in R["cfg"]}
LR={(item["h"],item["kind"],item["split"]):(np.array(item["out"]),item["coef"]) for item in R["lr"]}
M=R["meta"]
N=len(M["names"]); fold=np.array(M["fold"]); noz=np.array(M["noise_only"]); snr=np.array(M["snr_true"]); dur=np.array(M["dur"])
sp=~noz
LEVELS=[("clean",99),("20 dB",20),("10 dB",10),("5 dB",5),("0 dB",0)]
rng=np.random.default_rng(0); B=2000
def pooled(A,idx): c=A[idx,:4].sum(0); return prf(c)
def key(rule,p,h): return (rule,tuple(p),h,0.1)
def A_of(rule,p,h): return CFG[key(rule,p,"ultra" if rule=="silero" else h)]
HEADS=("ultra","redux")
# ---- candidate sets: dict name -> (list of params, array [ncfg,N,5]) ----
def cand(rule,h):
    hh="ultra" if rule=="silero" else h
    ps=[k[1] for k in CFG if k[0]==rule and k[2]==hh and k[3]==0.1]
    # keep insertion order
    ps=[tuple(float(x) for x in p) for p in dict.fromkeys(ps)]; return ps,np.stack([CFG[key(rule,p,hh)] for p in ps])
def lrcand(h,kind,split): o,c=LR[(h,kind,split)]; return [(float(t),) for t in __import__("rules").THR],o
def score(arr,idx,crit):
    """arr [ncfg,N,5]; returns best cfg index on clips idx"""
    c=arr[:,idx,:4].sum(1); tp,fp,fn=c[:,0],c[:,1],c[:,2]
    P=tp/np.maximum(1,tp+fp);Rr=tp/np.maximum(1,tp+fn);F=2*P*Rr/np.maximum(1e-12,P+Rr)
    if crit=="f1": return int(np.argmax(F))
    ok=P>=crit
    return int(np.argmax(np.where(ok,Rr,-1+P*0)) if ok.any() else np.argmax(P))
def crossfit(getarr,crit,mode="speaker"):
    """returns (rows [N,5] with each clip from the cfg chosen without its own fold/shift side, chosen params per split)"""
    rows=np.zeros((N,5)); chosen={}
    if mode=="speaker":
        for f in range(4):
            ps,arr=getarr("f%d"%f); tr=np.flatnonzero(sp&(fold!=f)); te=np.flatnonzero(fold==f)
            i=score(arr,tr,crit); rows[te]=arr[i][te]; chosen[f]=ps[i]
    else:
        ps,arr=getarr("shift"); tr=np.flatnonzero(sp&(snr>=10)); te=np.flatnonzero(snr<10)
        i=score(arr,tr,crit); rows[te]=arr[i][te]; chosen["shift"]=ps[i]
    return rows,chosen
def fixed(arr_ps,p): ps,arr=arr_ps; return arr[ps.index(tuple(p))]
def static(rule,h):
    ps,arr=cand(rule,h); return lambda s:(ps,arr)
def lrget(h,kind): return lambda s:lrcand(h,kind,s)
# ---- bootstrap ----
def boot_idx(idx): return rng.integers(0,len(idx),(B,len(idx)))
def ci(vals): return np.percentile(vals,[2.5,97.5])
def prf_b(rows,idx,bi):
    c=rows[idx][:,:4][bi].sum(1).astype(float)   # B,4
    P=c[:,0]/np.maximum(1,c[:,0]+c[:,1]);Rr=c[:,0]/np.maximum(1,c[:,0]+c[:,2]);F=2*P*Rr/np.maximum(1e-12,P+Rr)
    return P,Rr,F
def fmt(x,lo,hi): return f"{100*x:.1f} [{100*lo:.1f},{100*hi:.1f}]"
BI={}
def bidx(name,idx):
    if name not in BI: BI[name]=(idx,rng.integers(0,len(idx),(B,len(idx))))
    return BI[name]
def level_idx(lv): return np.flatnonzero(sp&(snr==lv))
def stats(rows):
    """pooled + per level point estimates and CIs (paired resample indices fixed per index set)"""
    out={}
    for nm,idx in [("all",np.flatnonzero(sp))]+[(l,level_idx(v)) for l,v in LEVELS]:
        idx,bi=bidx(nm,idx); P,Rr,F=prf_b(rows,idx,bi); p,r,f=pooled(rows,idx)
        out[nm]=dict(P=p,R=r,F=f,Pci=ci(P),Rci=ci(Rr),Fci=ci(F),F_b=F)
    return out
def fa(rows):
    """false-alarm segments per hour and speech-frame FP rate on noise-only clips, per noise level"""
    o={}
    for l,v in LEVELS[1:]:
        idx=np.flatnonzero(noz&(snr==v)); o[l]=(rows[idx,4].sum()/(dur[idx].sum()/3600), rows[idx,1].sum()/max(1,rows[idx,:4].sum()))
    idx=np.flatnonzero(noz); o["all"]=(rows[idx,4].sum()/(dur[idx].sum()/3600), rows[idx,1].sum()/rows[idx,:4].sum())
    return o
if __name__=="__main__":
    pass
