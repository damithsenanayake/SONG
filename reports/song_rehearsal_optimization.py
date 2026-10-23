"""Measure rehearsal frontier, layout KL, and pre-update novelty on MNIST."""
from copy import deepcopy
from pathlib import Path
import json, sys, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from song.song import SONG
with np.load(ROOT/'data'/'mnist.npz') as z:
    X=z['x_train'].reshape(-1,784).astype(np.float32)/255
    y=z['y_train']
    X_test=z['x_test'].reshape(-1,784).astype(np.float32)/255
    y_test=z['y_test']
old=y<5
X_old,X_new=X[old],X[~old]
y_old=y[old]
SONG(so_steps=2,final_vector_count=16,dispersion_method='none').fit_transform(X_old[:80])
model=SONG(so_steps=30,random_seed=42,dispersion_method='none')
model.fit(X_old)
Y0=model.transform(X_old).copy()
rng=np.random.default_rng(42)
anchor_idx=rng.choice(len(X_old),512,replace=False)
probe_train,probe_test=train_test_split(np.arange(len(X_old)),test_size=.2,random_state=42,stratify=y_old)

def probe(coords):
    knn=KNeighborsClassifier(n_neighbors=5,n_jobs=1).fit(coords[probe_train],y_old[probe_train])
    return float(accuracy_score(y_old[probe_test],knn.predict(coords[probe_test])))
initial_accuracy=probe(Y0)
A=Y0[anchor_idx].astype(np.float64)
Ac=A.mean(axis=0)
D0=np.sum((A[:,None,:]-A[None,:,:])**2,axis=2)
np.fill_diagonal(D0,np.inf)
# Fixed broad spatial scale from non-identical old-anchor pairs.
tau=float(np.median(D0[np.isfinite(D0) & (D0>0)]))
L0=-D0/tau
logP0=L0-logsumexp(L0,axis=1,keepdims=True)
P0=np.exp(logP0)


def metrics(after):
    B=after[anchor_idx].astype(np.float64)
    Bc=B.mean(axis=0)
    u,s,vt=np.linalg.svd((B-Bc).T@(A-Ac),full_matrices=False)
    R=u@vt
    scale=s.sum()/np.sum((B-Bc)**2)
    aligned=(after.astype(np.float64)-Bc)@R*scale+Ac
    B_aligned=aligned[anchor_idx]
    D1=np.sum((B_aligned[:,None,:]-B_aligned[None,:,:])**2,axis=2)
    np.fill_diagonal(D1,np.inf)
    L1=-D1/tau
    logP1=L1-logsumexp(L1,axis=1,keepdims=True)
    mask=P0>0
    kl=float(np.sum(P0[mask]*(logP0[mask]-logP1[mask]))/len(P0))
    centers0=np.stack([Y0[y_old==c].mean(axis=0) for c in range(5)])
    centers1=np.stack([aligned[y_old==c].mean(axis=0) for c in range(5)])
    spread=np.sqrt(np.mean(np.sum((Y0-Y0.mean(axis=0))**2,axis=1)))
    drift=float(np.sqrt(np.mean(np.sum((centers1-centers0)**2,axis=1)))/spread)
    return {'old_class_5nn_accuracy':probe(after),
            'normalized_old_class_centroid_rms_drift':drift,
            'old_anchor_neighborhood_kl_nats':kl}

fractions=[0.,.05,.1,.2,.35,.5,.75,1.]
replay_order=rng.permutation(len(X_old))
rows=[]
for fraction in fractions:
    trial=deepcopy(model)
    idx=replay_order[:round(fraction*len(X_old))]
    t=time.perf_counter()
    trial.fit(np.concatenate((X_new,X_old[idx]),axis=0))
    elapsed=time.perf_counter()-t
    row={'rehearsal_fraction_of_old':fraction,'rehearsal_old_n':len(idx),
         'increment_fit_s':elapsed,**metrics(trial.transform(X_old))}
    rows.append(row)
    print(row,flush=True)
# A label-based target is used only as an illustrative research constraint.
for row in rows:
    row['meets_example_target']=(row['old_class_5nn_accuracy']>=initial_accuracy-.01 and
                                 row['normalized_old_class_centroid_rms_drift']<=.2)
selected=next((r['rehearsal_fraction_of_old'] for r in rows if r['meets_example_target']),None)

# Novelty is scored before any new-data update, using the frozen old map.
# Quantization distance uses exact float distances; SONG's current returned min-distance is integer-truncated.
W=model.pca.transform(model.W).astype(np.float32)
Wn=np.sum(W*W,axis=1)

def distances(Q):
    out=[]
    for st in range(0,len(Q),1000):
        q=model._get_XPCA(Q[st:st+1000]).astype(np.float32)
        d=np.maximum(0,np.sum(q*q,axis=1)[:,None]+Wn[None,:]-2*q@W.T)
        out.append(d)
    return np.vstack(out)
calibration_idx=rng.choice(len(X_old),min(10000,len(X_old)),replace=False)
D_cal=distances(X_old[calibration_idx])
nearest_cal=np.min(D_cal,axis=1)
# Temperature from positive nearest-vs-second-nearest margins.
margin=np.partition(D_cal,1,axis=1)[:,1]-nearest_cal
temp=float(np.median(margin[margin>0]))
assign_cal=np.exp(-D_cal/temp-logsumexp(-D_cal/temp,axis=1,keepdims=True))
prior=(assign_cal.sum(axis=0)+1)/(len(D_cal)+len(W))
D_test=distances(X_test)
quant=np.min(D_test,axis=1)
logpost=-D_test/temp-logsumexp(-D_test/temp,axis=1,keepdims=True)
post=np.exp(logpost)
assignment_kl=np.sum(post*(logpost-np.log(prior)[None,:]),axis=1)
novel=(y_test>=5).astype(int)
novelty={'old_test_n':int(np.sum(novel==0)),'new_test_n':int(np.sum(novel==1)),
         'nearest_prototype_squared_distance_auroc':float(roc_auc_score(novel,quant)),
         'assignment_to_old_occupancy_kl_auroc':float(roc_auc_score(novel,assignment_kl)),
         'temperature':temp,
         'distance_old_median':float(np.median(quant[novel==0])),
         'distance_new_median':float(np.median(quant[novel==1]))}
# Position KL is available only after an updated map exists. Compare isotropic
# Gaussian location distributions around each pre/post-update 2D coordinate.
Y_after_anchor=trial.transform(X_old[anchor_idx]).astype(np.float64)
Bc=Y_after_anchor.mean(axis=0)
u,s,vt=np.linalg.svd((Y_after_anchor-Bc).T@(A-Ac),full_matrices=False)
R=u@vt
scale=s.sum()/np.sum((Y_after_anchor-Bc)**2)
Y_test_before=model.transform(X_test).astype(np.float64)
Y_test_after=(trial.transform(X_test).astype(np.float64)-Bc)@R*scale+Ac
position_kl=np.sum((Y_test_after-Y_test_before)**2,axis=1)/(2*tau)
novelty['postupdate_position_gaussian_kl_auroc']=float(roc_auc_score(novel,position_kl))
novelty['postupdate_position_kl_old_median']=float(np.median(position_kl[novel==0]))
novelty['postupdate_position_kl_new_median']=float(np.median(position_kl[novel==1]))
novelty['postupdate_position_kl_condition']='Full old-data rehearsal; requires fitting an updated map first'

results={'seed':42,'initial_old_class_5nn_accuracy':initial_accuracy,
         'old_anchor_n':len(anchor_idx),'old_anchor_kl_temperature':tau,
         'example_target':'old 5NN accuracy no more than 1 percentage point below initial and normalized class-centroid drift <= 0.2',
         'selected_rehearsal_fraction':selected,'conditions':rows,'preupdate_novelty':novelty}
(ROOT/'reports'/'song_rehearsal_optimization.json').write_text(json.dumps(results,indent=2)+'\n')
fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
xs=[100*r['rehearsal_fraction_of_old'] for r in rows]
for ax,key,ylabel in zip(axes.flat,
 ['old_class_5nn_accuracy','normalized_old_class_centroid_rms_drift','old_anchor_neighborhood_kl_nats','increment_fit_s'],
 ['Old-digit 5-NN accuracy','Normalized old class-centroid drift','Old-anchor neighborhood KL (nats)','Increment fit time (s)']):
    ax.plot(xs,[r[key] for r in rows],marker='o')
    ax.set(xlabel='Old data rehearsed (%)',ylabel=ylabel)
    ax.grid(alpha=.25)
    if selected is not None:ax.axvline(100*selected,color='tab:green',linestyle='--',alpha=.6)
axes[0,0].axhline(initial_accuracy-.01,color='tab:red',linestyle=':',alpha=.6)
axes[0,1].axhline(.2,color='tab:red',linestyle=':',alpha=.6)
fig.suptitle('SONG rehearsal frontier, MNIST old digits 0-4 then new 5-9')
fig.savefig(ROOT/'reports'/'figures'/'song_rehearsal_tradeoff.png',dpi=160)
plt.close(fig)
print('SELECTED',selected,'NOVELTY',novelty,flush=True)
