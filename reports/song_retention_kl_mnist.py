"""Matched MNIST ablation of the implemented opt-in KL retention optimizer."""
from copy import deepcopy
from pathlib import Path
import json, sys, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import procrustes
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from song.song import SONG
from song.retention import neighborhood_temperature, position_kl_loss_and_grad
with np.load(ROOT/'data'/'mnist.npz') as z:
    X=z['x_train'].reshape(-1,784).astype(np.float32)/255
    labels=z['y_train']
old=labels<5
X_old,X_new=X[old],X[~old]
y_old,y_new=labels[old],labels[~old]
SONG(so_steps=2,final_vector_count=16,dispersion_method='none').fit_transform(X_old[:80])
base=SONG(so_steps=30,random_seed=42,dispersion_method='none',
          retention_kl_weight=1.,retention_position_weight=1.,
          retention_anchor_count=512,retention_maxiter=40)
base.fit(X_old)
Y0=base.transform(X_old).copy()
rng=np.random.default_rng(42)
replay_order=rng.permutation(len(X_old))
probe_old_train,probe_old_test=train_test_split(np.arange(len(X_old)),test_size=.2,random_state=42,stratify=y_old)
probe_new_train,probe_new_test=train_test_split(np.arange(len(X_new)),test_size=.2,random_state=42,stratify=y_new)
metric_idx=rng.choice(len(X_old),1000,replace=False)
plot_old=rng.choice(len(X_old),5000,replace=False)
plot_new=rng.choice(len(X_new),5000,replace=False)
A=Y0[metric_idx].astype(np.float64)
Ac=A.mean(axis=0)

def probe(coords,labels,train,test):
    clf=KNeighborsClassifier(n_neighbors=5,n_jobs=1).fit(coords[train],labels[train])
    return float(accuracy_score(labels[test],clf.predict(coords[test])))
initial_old_accuracy=probe(Y0,y_old,probe_old_train,probe_old_test)

def align(after):
    B=after[metric_idx].astype(np.float64)
    Bc=B.mean(axis=0)
    u,s,vt=np.linalg.svd((B-Bc).T@(A-Ac),full_matrices=False)
    R=u@vt;scale=s.sum()/np.sum((B-Bc)**2)
    aligned=(after.astype(np.float64)-Bc)@R*scale+Ac
    centers0=np.stack([Y0[y_old==c].mean(axis=0) for c in range(5)])
    centers1=np.stack([aligned[y_old==c].mean(axis=0) for c in range(5)])
    spread=np.sqrt(np.mean(np.sum((Y0-Y0.mean(axis=0))**2,axis=1)))
    drift=float(np.sqrt(np.mean(np.sum((centers1-centers0)**2,axis=1)))/spread)
    _,_,disparity=procrustes(A,B)
    return aligned,Bc,R,scale,drift,float(disparity)

conditions=[
 ('No replay; no KL',0.,0.,0.),
 ('No replay; neighborhood KL',0.,1.,0.),
 ('No replay; neighborhood + position KL',0.,1.,1.),
 ('10% replay; no KL',.1,0.,0.),
 ('10% replay; neighborhood + position KL',.1,1.,1.),
 ('20% replay; neighborhood + position KL',.2,1.,1.),
 ('35% replay; no KL',.35,0.,0.),
 ('35% replay; neighborhood + position KL',.35,1.,1.),
 ('50% replay; no KL',.5,0.,0.),
 ('50% replay; neighborhood + position KL',.5,1.,1.),
 ('75% replay; neighborhood + position KL',.75,1.,1.),
 ('75% replay; no KL',.75,0.,0.),
]
rows=[];embeddings={}
for name,fraction,w_kl,w_pos in conditions:
    model=deepcopy(base)
    model.retention_kl_weight=w_kl
    model.retention_position_weight=w_pos
    count=round(fraction*len(X_old))
    X_update=np.concatenate([X_new,X_old[replay_order[:count]]],axis=0)
    start=time.perf_counter();model.fit(X_update);elapsed=time.perf_counter()-start
    Y_old=model.transform(X_old).copy()
    Y_new=model.transform(X_new).copy()
    aligned,Bc,R,scale,drift,disparity=align(Y_old)
    aligned_new=(Y_new.astype(np.float64)-Bc)@R*scale+Ac
    anchor_current=model.Y[model._retention_indices(model._retention_anchor_X)]
    reference=model._retention_anchor_Y
    tau=neighborhood_temperature(reference)
    _,_,anchor_kl,anchor_pos_kl=position_kl_loss_and_grad(anchor_current,reference,tau,1.,1.)
    row={'condition':name,'rehearsal_fraction':fraction,'kl_weight':w_kl,'position_weight':w_pos,
         'update_fit_s':elapsed,'old_class_5nn_accuracy':probe(Y_old,y_old,probe_old_train,probe_old_test),
         'new_class_5nn_accuracy':probe(Y_new,y_new,probe_new_train,probe_new_test),
         'normalized_old_class_centroid_rms_drift':drift,
         'old_procrustes_disparity_1000':disparity,
         'anchor_neighborhood_kl_nats':anchor_kl,'anchor_gaussian_position_kl_nats':anchor_pos_kl,
         'optimizer':model.retention_result_}
    rows.append(row);embeddings[name]=(aligned,aligned_new)
    print(row,flush=True)

results={'seed':42,'old_n':len(X_old),'new_n':len(X_new),'song_steps_per_fit':30,
         'retention_anchor_count':512,'retention_maxiter':40,
         'initial_old_class_5nn_accuracy':initial_old_accuracy,'conditions':rows}
for row in rows:
    row['meets_example_target']=(row['old_class_5nn_accuracy']>=initial_old_accuracy-.01 and
                                 row['normalized_old_class_centroid_rms_drift']<=.2)
results['smallest_measured_replay_fraction_with_kl']=next((r['rehearsal_fraction'] for r in rows if r['kl_weight']>0 and r['position_weight']>0 and r['meets_example_target']),None)
(ROOT/'reports'/'song_retention_kl_mnist.json').write_text(json.dumps(results,indent=2)+'\n')
names=['No replay; no KL','No replay; neighborhood + position KL','35% replay; no KL','35% replay; neighborhood + position KL','75% replay; no KL']
fig,axes=plt.subplots(2,len(names),figsize=(20,8),constrained_layout=True)
short_titles=['0% replay, no KL','0% replay, KL','35% replay, no KL',
              '35% replay, KL','75% replay, no KL']
common=np.concatenate([embeddings[name][0][plot_old] for name in names]+
                      [embeddings[name][1][plot_new] for name in names])
limits=np.quantile(common,[.005,.995],axis=0)
padding=(limits[1]-limits[0])*.04
for j,name in enumerate(names):
    old_coords,new_coords=embeddings[name]
    for i,ax in enumerate(axes[:,j]):
        ax.scatter(old_coords[plot_old,0],old_coords[plot_old,1],c=y_old[plot_old],cmap='tab10',vmin=0,vmax=9,s=3,alpha=.65 if i==0 else .35,rasterized=True)
        if i:
            ax.scatter(new_coords[plot_new,0],new_coords[plot_new,1],c=y_new[plot_new],cmap='tab10',vmin=0,vmax=9,s=3,alpha=.35,rasterized=True)
        ax.set_xlim(limits[0,0]-padding[0],limits[1,0]+padding[0])
        ax.set_ylim(limits[0,1]-padding[1],limits[1,1]+padding[1])
        ax.set_title(short_titles[j] if i==0 else 'Old + new',fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
fig.suptitle('SONG KL retention: old digits 0-4, then new digits 5-9 (aligned old positions)')
fig.savefig(ROOT/'reports'/'figures'/'song_retention_kl_mnist.png',dpi=150)
plt.close(fig)
