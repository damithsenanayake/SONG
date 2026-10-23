"""Three-seed check of the implemented KL retention and replay tradeoff."""
from copy import deepcopy
from pathlib import Path
import sys, time
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
y_old=y[old]
probe_train,probe_test=train_test_split(np.arange(len(X_old)),test_size=.2,random_state=42,stratify=y_old)
SONG(so_steps=2,final_vector_count=16,dispersion_method='none').fit_transform(X_old[:80])

def probe(coords):
    clf=KNeighborsClassifier(n_neighbors=5,n_jobs=1).fit(coords[probe_train],y_old[probe_train])
    return float(accuracy_score(y_old[probe_test],clf.predict(coords[probe_test])))

rows=[]
for seed in [42,43,44]:
    base=SONG(so_steps=30,random_seed=seed,dispersion_method='none',
              retention_kl_weight=1.,retention_position_weight=1.,
              retention_anchor_count=512,retention_maxiter=40)
    base.fit(X_old)
    Y0=base.transform(X_old).copy()
    initial=probe(Y0)
    rng=np.random.default_rng(seed)
    order=rng.permutation(len(X_old))
    anchor_idx=rng.choice(len(X_old),1000,replace=False)
    A=Y0[anchor_idx].astype(np.float64)
    Ac=A.mean(axis=0)
    centers0=np.stack([Y0[y_old==c].mean(axis=0) for c in range(5)])
    spread=np.sqrt(np.mean(np.sum((Y0-Y0.mean(axis=0))**2,axis=1)))
    for fraction,use_kl in [(.35,False),(.35,True),(.5,False),(.5,True)]:
        model=deepcopy(base)
        if not use_kl:
            model.retention_kl_weight=0.
            model.retention_position_weight=0.
        idx=order[:round(fraction*len(X_old))]
        start=time.perf_counter()
        model.fit(np.concatenate([X_new,X_old[idx]],axis=0))
        elapsed=time.perf_counter()-start
        after=model.transform(X_old)
        B=after[anchor_idx].astype(np.float64)
        Bc=B.mean(axis=0)
        u,s,vt=np.linalg.svd((B-Bc).T@(A-Ac),full_matrices=False)
        aligned=(after.astype(np.float64)-Bc)@(u@vt)*(s.sum()/np.sum((B-Bc)**2))+Ac
        centers1=np.stack([aligned[y_old==c].mean(axis=0) for c in range(5)])
        drift=float(np.sqrt(np.mean(np.sum((centers1-centers0)**2,axis=1)))/spread)
        old_accuracy=probe(after)
        row={'seed':seed,'rehearsal_fraction':fraction,'kl_enabled':use_kl,
             'initial_old_accuracy':initial,'old_accuracy':old_accuracy,
             'old_accuracy_drop_pp':100*(initial-old_accuracy),
             'old_centroid_drift':drift,'fit_s':elapsed,
             'meets_example_target':bool(old_accuracy>=initial-.01 and drift<=.2)}
        rows.append(row)
        print(row,flush=True)
result=pd.DataFrame(rows)
result.to_csv(ROOT/'reports'/'song_retention_kl_seeds.csv',index=False)
print(result.groupby(['rehearsal_fraction','kl_enabled'])[['old_accuracy_drop_pp','old_centroid_drift','fit_s','meets_example_target']].agg(['mean','std']),flush=True)
