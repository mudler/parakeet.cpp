import sys,json,time,numpy as np
from multiprocessing import Pool
from fl import *
from rules import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
D=load()
for i,d in enumerate(D):
    c=d["cond"].replace("nz-",""); d["snr_true"]=99 if c=="clean" else int(c.lstrip("whitepink"))
    d["noise_only"]=d["name"].startswith("nz-")
    k=int(d["name"].split("_")[1]); d["fold"]=k%4 if d["noise_only"] else d["fold"]
N=len(D)
def ev(mask,d,pk):
    m,regs=post(mask,**pk)
    c=counts(m,d["ref_mask"]); return np.append(c,len(regs))
def eval_cfg(a):
    rule,p,h,pk=a
    return np.array([ev(raw_mask(rule,p,d,h),d,pk) for d in D])
def lg(x): x=np.clip(x,1e-4,1-1e-4); return np.log(x/(1-x))
def feats(d,h,kind):
    s=d["silero"];hh=d[h]
    if kind=="lr2": return np.stack([lg(s),lg(hh)],1)
    def sh(x,k): return np.concatenate([np.full(k,x[0]),x[:-k]]) if k>0 else np.concatenate([x[-k:],np.full(-k,x[-1])])
    return np.stack([lg(s),lg(hh),lg(sh(s,3)),lg(sh(s,-3)),lg(sh(hh,8)),lg(sh(hh,-8))],1)
def lr_split(a):
    h,kind,sname,train=a
    nz=kind.endswith("nz"); kind=kind[:-2] if nz else kind
    if not nz: train=[i for i in train if not D[i]["noise_only"]]
    X=np.concatenate([feats(D[i],h,kind)[::10] for i in train]); y=np.concatenate([D[i]["ref_mask"][::10] for i in train])
    mdl=HistGradientBoostingClassifier(max_iter=120,learning_rate=0.1,random_state=0) if kind=="hgb6" else LogisticRegression(C=1.0,max_iter=300)
    mdl.fit(X,y)
    P=[mdl.predict_proba(feats(d,h,kind))[:,1] for d in D]
    out=np.zeros((len(THR),N,5))
    for ti,t in enumerate(THR):
        for i,d in enumerate(D): out[ti,i]=ev(P[i]>=t,d,POST_HEAD)
    coef=getattr(mdl,"coef_",None)
    return (h,kind+("nz" if nz else ""),sname),out,(None if coef is None else (coef[0].round(3).tolist(),float(mdl.intercept_[0])))
if __name__=="__main__":
    t0=time.time(); res={}
    jobs=[]
    for h in("ultra","redux"):
        for rule in("head","or","and","mean","max","min","two","switch","oracle"):
            for p in candidates(rule): jobs.append((rule,p,h,POST_HEAD))
    for p in candidates("silero"): jobs.append(("silero",p,"ultra",POST_HEAD))
    # native default post for singles at thr 0.5
    jobs.append(("silero",(0.5,),"ultra",POST_SIL))
    print(len(jobs),"jobs",flush=True)
    with Pool(5) as pool:
        outs=pool.map(eval_cfg,jobs,chunksize=4)
    res["cfg"]={ (j[0],j[1],j[2],j[3]["min_speech"]):o for j,o in zip(jobs,outs)}
    print("rules done",time.time()-t0,flush=True)
    folds=np.array([d["fold"] for d in D]); easy=np.array([d["snr_true"]>=10 for d in D])
    splits=[(f"f{f}",[i for i in range(N) if folds[i]!=f]) for f in range(4)]+[("shift",[i for i in range(N) if easy[i]])]
    lj=[(h,k,sn,tr) for h in("ultra","redux") for k in("lr2","lr6","hgb6","lr6nz","hgb6nz") for sn,tr in splits]
    with Pool(5) as pool: lo=pool.map(lr_split,lj,chunksize=1)
    res["lr"]={k:(o,c) for k,o,c in lo}
    res["meta"]=dict(names=[d["name"] for d in D],cond=[d["cond"] for d in D],fold=folds.tolist(),snr_true=[d["snr_true"] for d in D],noise_only=[d["noise_only"] for d in D],snr_est=[d["snr_est"] for d in D],dur=[d["dur"] for d in D])
    res_json = {
        "cfg": [{"rule": k[0], "p": list(k[1]), "h": k[2], "ms": k[3], "arr": v.tolist()} for k, v in res["cfg"].items()],
        "lr": [{"h": k[0], "kind": k[1], "split": k[2], "out": v[0].tolist(), "coef": v[1]} for k, v in res["lr"].items()],
        "meta": res["meta"]
    }
    json.dump(res_json, open("results.json", "w")); print("done", time.time() - t0)
