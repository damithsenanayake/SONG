"""Three-seed MNIST comparison of prototype KL, snapshot restoration, and replay."""
from copy import deepcopy
from pathlib import Path
import sys,time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from song.song import SONG
with np.load(ROOT/'data'/'mnist.npz') as z:
    X=z['x_train'].reshape(-1,784).astype(np.float32)/255
    y=z['y_train']
old=y<5
X_old,X_new=X[old],X[~old]
y_old,y_new=y[old],y[~old]
old_tr,old_te=train_test_split(np.arange(len(X_old)),test_size=.2,random_state=42,stratify=y_old)
new_tr,new_te=train_test_split(np.arange(len(X_new)),test_size=.2,random_state=42,stratify=y_new)
SONG(so_steps=2,final_vector_count=16,dispersion_method='none').fit_transform(X_old[:80])

def probe(coords,labels,tr,te):
    clf=KNeighborsClassifier(n_neighbors=5,n_jobs=1).fit(coords[tr],labels[tr])
    return float(accuracy_score(labels[te],clf.predict(coords[te])))

rows=[]
for seed in [42,43,44]:
    base=SONG(so_steps=30,random_seed=seed,dispersion_method='none',
              retention_source='prototypes',retention_kl_weight=1.,
              retention_position_weight=1.,retention_maxiter=30)
    base.fit(X_old)
    W0=base.W.copy();proto_Y0=base.Y.copy()
    Y0=base.transform(X_old).copy()
    initial=probe(Y0,y_old,old_tr,old_te)
    rng=np.random.default_rng(seed)
    idx=rng.choice(len(X_old),1000,replace=False)
    replay_order=rng.permutation(len(X_old))
    A=Y0[idx].astype(np.float64);Ac=A.mean(axis=0)
    centers0=np.stack([Y0[y_old==c].mean(axis=0) for c in range(5)])
    spread=np.sqrt(np.mean(np.sum((Y0-Y0.mean(axis=0))**2,axis=1)))
    configs=[
      ('new only', 'samples',0.,0.,0.,False),
      ('prototype neighborhood KL','prototypes',1.,0.,0.,False),
      ('prototype position KL','prototypes',0.,1.,0.,False),
      ('prototype both KL','prototypes',1.,1.,0.,False),
      ('prototype snapshot restore','prototypes',0.,0.,0.,True),
      ('raw replay 35%','samples',0.,0.,.35,False),
    ]
    for name,source,w_kl,w_pos,replay,restore in configs:
        model=deepcopy(base)
        model.retention_source=source
        model.retention_kl_weight=w_kl
        model.retention_position_weight=w_pos
        count=round(replay*len(X_old))
        X_update=np.concatenate([X_new,X_old[replay_order[:count]]],axis=0)
        start=time.perf_counter();model.fit(X_update)
        if restore:
            model.W[:len(W0)]=W0
            model.Y[:len(proto_Y0)]=proto_Y0
        elapsed=time.perf_counter()-start
        Y_old=model.transform(X_old)
        Y_new=model.transform(X_new)
        B=Y_old[idx].astype(np.float64);Bc=B.mean(axis=0)
        u,s,vt=np.linalg.svd((B-Bc).T@(A-Ac),full_matrices=False)
        aligned=(Y_old.astype(np.float64)-Bc)@(u@vt)*(s.sum()/np.sum((B-Bc)**2))+Ac
        centers1=np.stack([aligned[y_old==c].mean(axis=0) for c in range(5)])
        drift=float(np.sqrt(np.mean(np.sum((centers1-centers0)**2,axis=1)))/spread)
        old_acc=probe(Y_old,y_old,old_tr,old_te)
        row={'seed':seed,'condition':name,'old_replay_n':count,
             'initial_old_accuracy':initial,'old_accuracy':old_acc,
             'old_accuracy_drop_pp':100*(initial-old_acc),
             'new_accuracy':probe(Y_new,y_new,new_tr,new_te),
             'old_centroid_drift':drift,'fit_s':elapsed,
             'meets_example_target':bool(old_acc>=initial-.01 and drift<=.2)}
        rows.append(row)
        print(row,flush=True)
result=pd.DataFrame(rows)
result.to_csv(ROOT/'reports'/'song_prototype_kl_seeds.csv',index=False)
print(result.groupby('condition')[['old_accuracy_drop_pp','new_accuracy','old_centroid_drift','fit_s','meets_example_target']].agg(['mean','std']),flush=True)
