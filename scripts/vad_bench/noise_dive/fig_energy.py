import os
FUSION=os.environ.get('FUSION_DIR','../fusion')  # scripts/vad_bench/fusion with data/ built by prep_libri.py
import numpy as np,json,sys
sys.path.insert(0,FUSION)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from vadlib import frame_mask
clips=json.load(open('e_set.json','r'))
for c in clips:
    c['z']={m:np.array(v,dtype=np.float32) for m,v in c['z'].items()}
    c['sil']=np.array(c['sil'],dtype=np.float32)
    c['e10']=np.array(c['e10'],dtype=np.float32)
fig,axs=plt.subplots(1,2,figsize=(13,4.6),sharey=True)
stats={}
for ax,m in zip(axs,['redux','ultra']):
    X={'speech frames (clean+noisy speech clips)':[],'non-speech frames in speech clips (gaps)':[],'noise-only files':[],'embedded noise stretch':[]}; Y={k:[] for k in X}
    for c in clips:
        z=c['z'][m]; n=len(z); e=c['e10'][:n*8].reshape(n,8) if len(c['e10'])>=n*8 else None
        if e is None: n=len(c['e10'])//8; z=z[:n]; e=c['e10'][:n*8].reshape(n,8)
        e80=10*np.log10((10**(e/10)).mean(1)+1e-12); rel=e80-np.percentile(c['e10'],95)
        t=(np.arange(n)+.5)*.08
        if c['kind']=='speech':
            sp=frame_mask(c['spans'],n); X['speech frames (clean+noisy speech clips)'].append(rel[sp]); Y['speech frames (clean+noisy speech clips)'].append(z[sp])
            X['non-speech frames in speech clips (gaps)'].append(rel[~sp]); Y['non-speech frames in speech clips (gaps)'].append(z[~sp])
        elif c['kind']=='noise': X['noise-only files'].append(rel); Y['noise-only files'].append(z)
        else:
            a,b=c['stretch']; st=(t>=a)&(t<b); X['embedded noise stretch'].append(rel[st]); Y['embedded noise stretch'].append(z[st])
    for (k,col) in zip(X,['C0','C1','C3','C4']):
        x=np.concatenate(X[k]); y=np.clip(np.concatenate(Y[k]),-15,20); idx=np.random.default_rng(0).choice(len(x),min(len(x),4000),replace=False)
        ax.scatter(x[idx],y[idx],s=3,alpha=.35,c=col,label=k)
        stats[(m,k)]=(np.corrcoef(x,y)[0,1],len(x))
    ax.axhline(0,color='k',ls=':'); ax.set_xlabel('80 ms frame energy relative to the file P95 (dB)'); ax.set_title(f'{m}: logit vs relative frame energy'); ax.set_xlim(-60,8)
axs[0].set_ylabel('pre-sigmoid logit'); axs[0].legend(fontsize=7,markerscale=4,loc='lower right'); plt.tight_layout(); plt.savefig('fig_logit_vs_energy.png',dpi=110)
for k,v in stats.items(): print(k,'corr(rel energy, logit) %.2f over %d frames'%v)
