import os
FUSION=os.environ.get('FUSION_DIR','../fusion')  # scripts/vad_bench/fusion with data/ built by prep_libri.py
"""Rebuild the fusion synthetic set (same generators and seeds as fusion/mkcorpus.py + mkgap.py) and compute per-clip:
head logits (redux, ultra F16; CPU reference validated against parakeet-cli), Silero probabilities (ONNX), and 10 ms frame energy."""
import numpy as np,json,sys,os

class ClipEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
from vadlib import *; import corpus; import onnxruntime as ort
FL=corpus.FL; libri=corpus.libri; spk=corpus.spk
sets=corpus.speech_sets()
conds=[("clean",None)]+[(k,s) for k in("white","pink") for s in(20,10,5,0)]
Ps=[]; clips=[]
q=lambda y: (np.round(np.clip(y,-1,1)*32767)/32767).astype(np.float32)   # PCM_16 round trip as in the original wavs
for ci,(kind,snr) in enumerate(conds):
    for k,(f,us) in enumerate(sets):
        y,spans=FL.build(us,np.random.default_rng(1000+k))
        if ci==0: Ps.append(float(np.mean(np.concatenate([y[int(s*16000):int(e*16000)] for s,e in spans])**2)))
        y=FL.add_noise(y,spans,kind,snr,np.random.default_rng(5000+ci*100+k))
        clips.append(dict(name=f"{kind}{'' if snr is None else snr}_{k:02d}",cond=f"{kind}{'' if snr is None else snr}",kind='speech',spans=spans,y=q(y)))
for ci,(kind,snr) in enumerate(conds[1:]):
    for k in range(12):
        r=np.random.default_rng(9000+ci*100+k); P=Ps[(k*3)%len(Ps)]; n=30*16000
        z=r.standard_normal(n) if kind=="white" else FL.pink(n,r)
        z=np.clip(z*np.sqrt(P/10**(snr/10)),-1,1).astype(np.float32)
        clips.append(dict(name=f"nz-{kind}{snr}_{k:02d}",cond=f"nz-{kind}{snr}",kind='noise',spans=[],y=q(z)))
for g in corpus.gap_clips():
    clips.append(dict(name=g['name'],cond=f"gap-{g['kind']}{g['snr']}",kind='gap',spans=g['spans'],stretch=g['stretch'],y=q(g['y'])))
print(len(clips),'clips',sum(len(c['y']) for c in clips)/16000,'s',flush=True)
so=ort.SessionOptions(); so.intra_op_num_threads=2; so.inter_op_num_threads=1
sess=ort.InferenceSession(os.environ['SILERO_ONNX'],so,providers=["CPUExecutionProvider"])
def silero(y):
    n=(len(y)+511)//512; y=np.pad(y,(0,n*512-len(y))); st=np.zeros((2,1,128),np.float32); ctx=np.zeros(64,np.float32); out=[]
    for i in range(n):
        x=np.concatenate([ctx,y[i*512:(i+1)*512]])[None].astype(np.float32)
        o,st=sess.run(None,{"input":x,"state":st,"sr":np.array(16000,np.int64)}); out.append(float(o[0,0])); ctx=x[0,-64:]
    return np.array(out,np.float32)
for i,c in enumerate(clips):
    y=c['y']; c['z']={m:logits_blocks(m,y).astype(np.float32) for m in['redux','ultra']}
    c['sil']=silero(y)
    fr=len(y)//160; c['e10']=10*np.log10((y[:fr*160].reshape(fr,160).astype(np.float64)**2).mean(1)+1e-12)   # dBFS^2 per 10 ms
    del c['y']
    if i%40==0: print(i,flush=True)
json.dump(clips,open('e_set.json','w'),cls=ClipEncoder)
