import os
FUSION=os.environ.get('FUSION_DIR','../fusion')  # scripts/vad_bench/fusion with data/ built by prep_libri.py
import numpy as np,json,sys
sys.path.insert(0,FUSION)
import fl, rules
from vadlib import sig
GR=fl.GR
clips=json.load(open('e_set.json','r'))
for c in clips:
    c['z']={m:np.array(v,dtype=np.float32) for m,v in c['z'].items()}
    c['sil']=np.array(c['sil'],dtype=np.float32)
    c['e10']=np.array(c['e10'],dtype=np.float32)
def grid(c):
    n=int(round(len(c['e10'])))   # 10 ms cells
    return n
def refmask(c,n):
    m=np.zeros(n,bool)
    for s,e in c['spans']: m[int(round(s/GR)):int(round(e/GR))]=True
    return m
def stretch_mask(c,n):
    m=np.zeros(n,bool)
    if 'stretch' in c: a,b=c['stretch']; m[int(round(a/GR)):int(round(b/GR))]=True
    return m
def e80(c):
    p=10**(c['e10']/10); k=np.ones(8)/8; return 10*np.log10(np.convolve(p,k,'same')+1e-12)
def run_rule(rule,c,model):
    n=len(c['e10']); H=fl.hold(sig(c['z'][model]),0.08,n); S=fl.hold(c['sil'],0.032,n)
    kind=rule[0]
    if kind=='head': m=H>=rule[1]; post=fl.POST_HEAD
    elif kind=='head_ms': m=H>=0.5; post=dict(fl.POST_HEAD,min_speech=rule[1])
    elif kind=='gate':   # head thr + relative energy gate
        m=H>=rule[1]; e=e80(c); ref_lvl=np.percentile(c['e10'],95); m=m&(e>=ref_lvl-rule[2]); post=fl.POST_HEAD
    elif kind=='runconf':   # keep a p>=0.5 run only when its median logit is >= Q (noise plateau sits near +1.5, speech runs near +7)
        z=c['z'][model]; keep=np.zeros(len(z),bool); a=fl.runs(z>=0)
        for i,j in zip(*a):
            if np.median(z[i:j])>=rule[1]: keep[i:j]=True
        m=fl.hold(keep.astype(np.float32),0.08,n)>=0.5; post=fl.POST_HEAD
    elif kind=='silero': m=S>=0.5; post=fl.POST_SIL
    elif kind=='and': m=(S>=0.5)&(H>=rule[1]); post=fl.POST_HEAD
    elif kind=='two': m=rules.two_stage(S>=0.5,H>=0.5,16,16,30) if False else rules.two_stage(S>=0.5,H>=0.5,rule[1],rule[2],rule[3]); post=fl.POST_HEAD
    return fl.post(m,**post)
R=[('head',0.5),('head',0.7),('head',0.9),('head',0.97),('head_ms',0.25),('head_ms',0.5),('head_ms',1.0),
   ('gate',0.5,15),('gate',0.5,25),('gate',0.5,35),('gate',0.9,25),
   ('runconf',2.5),('runconf',3.5),('runconf',5.0),('and',0.5),('and',0.9),('two',16,16,30),('silero',)]
def name(r):
    k=r[0]; g=lambda i: r[i] if len(r)>i else ""
    return {'head':lambda:f'head thr {g(1)}','head_ms':lambda:f'head thr 0.5, min_speech {g(1)} s','gate':lambda:f'head thr {g(1)} + energy gate (frame >= file P95 - {g(2)} dB)','and':lambda:f'Silero 0.5 AND head {g(1)}','two':lambda:'two-stage (Silero + head extend/fill, fusion rule 6)','silero':lambda:'Silero 0.5 alone (reference, its own post)','runconf':lambda:f'head thr 0.5, keep run only if median logit >= {g(1)}'}[k]()
out={}
for model in ['redux','ultra']:
    print(f'\n### {model}\n| rule | speech F1 clean | speech F1 noisy (white+pink 20..0 dB) | noise-only FA frames % | embedded-stretch FA frames % | noise-only FA regions per clip |\n|---|---|---|---|---|---|')
    for r in R:
        cc={'clean':np.zeros(4),'noisy':np.zeros(4)}; fa_nz=[0,0]; fa_st=[0,0]; regs=[]
        for c in clips:
            n=len(c['e10']); m,rg=run_rule(r,c,model)
            if c['kind']=='speech':
                key='clean' if c['cond']=='clean' else 'noisy'; cc[key]+=fl.counts(m,refmask(c,n))
            elif c['kind']=='noise': fa_nz[0]+=m.sum(); fa_nz[1]+=n; regs.append(len(rg))
            else:
                st=stretch_mask(c,n); fa_st[0]+=(m&st).sum(); fa_st[1]+=st.sum()
        f=lambda k: 100*fl.prf(cc[k])[2]; P=lambda k:100*fl.prf(cc[k])[0]; Rc=lambda k:100*fl.prf(cc[k])[1]
        out[f'{model}|{name(r)}']=dict(f1_clean=f('clean'),f1_noisy=f('noisy'),R_clean=Rc('clean'),P_clean=P('clean'),fa_noise=100*fa_nz[0]/fa_nz[1],fa_stretch=100*fa_st[0]/fa_st[1],regions=float(np.mean(regs)))
        o=out[f'{model}|{name(r)}']
        print(f'| {name(r)} | {o["f1_clean"]:.1f} (P {o["P_clean"]:.1f}, R {o["R_clean"]:.1f}) | {o["f1_noisy"]:.1f} | {o["fa_noise"]:.1f} | {o["fa_stretch"]:.1f} | {o["regions"]:.1f} |',flush=True)
json.dump(out,open('res_E.json','w'),indent=1)
