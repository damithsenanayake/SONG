"""Matched-state SONG rehearsal test: digits 0-4 then 5-9 with old examples replayed."""
from copy import deepcopy
from pathlib import Path
import json, sys, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import procrustes
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from song.song import SONG

with np.load(ROOT/'data'/'mnist.npz') as z:
    X=z['x_train'].reshape(-1,784).astype(np.float32)/255
    labels=z['y_train']
old=labels<5
X_old,X_new=X[old],X[~old]
y_old,y_new=labels[old],labels[~old]
SONG(so_steps=2,final_vector_count=16,dispersion_method='none').fit_transform(X_old[:80])
base=SONG(so_steps=30,random_seed=42,dispersion_method='none')
t=time.perf_counter();base.fit(X_old);initial_fit_s=time.perf_counter()-t
Y_before=base.transform(X_old).copy()
rng=np.random.default_rng(42)
metric_idx=rng.choice(len(X_old),1000,replace=False)
plot_old_idx=rng.choice(len(X_old),6000,replace=False)
plot_new_idx=rng.choice(len(X_new),6000,replace=False)
probe_train,probe_test=train_test_split(np.arange(len(X_old)),test_size=.2,random_state=42,stratify=y_old)
def old_class_probe(coords):
    clf=KNeighborsClassifier(n_neighbors=5,n_jobs=1).fit(coords[probe_train],y_old[probe_train])
    return float(accuracy_score(y_old[probe_test],clf.predict(coords[probe_test])))
initial_old_probe=old_class_probe(Y_before)

def align_and_measure(after):
    a=Y_before[metric_idx].astype(np.float64)
    b=after[metric_idx].astype(np.float64)
    ac=a.mean(axis=0);bc=b.mean(axis=0)
    u,s,vt=np.linalg.svd((b-bc).T@(a-ac),full_matrices=False)
    rotation=u@vt
    scale=s.sum()/np.sum((b-bc)**2)
    aligned=(after.astype(np.float64)-bc)@rotation*scale+ac
    _,_,disparity=procrustes(a,b)
    before_neighbors=NearestNeighbors(n_neighbors=16).fit(Y_before).kneighbors(Y_before[metric_idx],return_distance=False)[:,1:]
    after_neighbors=NearestNeighbors(n_neighbors=16).fit(after).kneighbors(after[metric_idx],return_distance=False)[:,1:]
    overlap=np.mean([len(set(x)&set(y))/15 for x,y in zip(before_neighbors,after_neighbors)])
    centers_before=np.stack([Y_before[y_old==c].mean(axis=0) for c in range(5)])
    centers_after=np.stack([aligned[y_old==c].mean(axis=0) for c in range(5)])
    rms_spread=np.sqrt(np.mean(np.sum((Y_before-Y_before.mean(axis=0))**2,axis=1)))
    centroid_drift=np.sqrt(np.mean(np.sum((centers_after-centers_before)**2,axis=1)))/rms_spread
    return aligned,rotation,scale,bc,ac,{'aligned_procrustes_disparity_1000':float(disparity),
     'old_neighbor_overlap_at_15':float(overlap),
     'normalized_old_class_centroid_rms_drift':float(centroid_drift),
     'old_class_5nn_accuracy':old_class_probe(after)}

fractions=[0.,.1,.5,1.]
rows=[];embeddings=[]
for fraction in fractions:
    model=deepcopy(base)
    sample_n=round(fraction*len(X_old))
    old_idx=rng.choice(len(X_old),sample_n,replace=False)
    X_increment=np.concatenate([X_new,X_old[old_idx]],axis=0)
    t=time.perf_counter();model.fit(X_increment);increment_s=time.perf_counter()-t
    Y_after=model.transform(X_old).copy()
    Y_new=model.transform(X_new).copy()
    aligned,rotation,scale,bc,ac,metrics=align_and_measure(Y_after)
    aligned_new=(Y_new.astype(np.float64)-bc)@rotation*scale+ac
    row={'rehearsal_fraction_of_old':fraction,'rehearsal_old_n':sample_n,'new_n':len(X_new),
         'increment_fit_s':increment_s,'prototypes_after':len(model.W),**metrics}
    rows.append(row);embeddings.append((aligned,aligned_new))
    print(row,flush=True)

results={'dataset':'Keras MNIST train; old digits 0-4, new digits 5-9',
 'seed':42,'song_steps_per_fit':30,'initial_fit_s':initial_fit_s,
 'initial_prototypes':len(base.W),'initial_old_class_5nn_accuracy':initial_old_probe,'old_n':len(X_old),'new_n':len(X_new),
 'metric_old_sample_n':len(metric_idx),'conditions':rows}
(ROOT/'reports'/'song_rehearsal_mnist.json').write_text(json.dumps(results,indent=2)+'\n')
fig,axes=plt.subplots(2,5,figsize=(20,8),constrained_layout=True)
all_old=[Y_before]+[pair[0] for pair in embeddings]
coords=np.concatenate([Y_before[plot_old_idx]]+[e[0][plot_old_idx] for e in embeddings],axis=0)
xlim=np.quantile(coords[:,0],[.005,.995]);ylim=np.quantile(coords[:,1],[.005,.995])
for j,(ax,old_coords) in enumerate(zip(axes[0],all_old)):
    ax.scatter(old_coords[plot_old_idx,0],old_coords[plot_old_idx,1],c=y_old[plot_old_idx],cmap='tab10',vmin=0,vmax=9,s=3,alpha=.7,rasterized=True)
    ax.set(xlim=xlim,ylim=ylim,title='Before' if j==0 else f'Replay {round(fractions[j-1]*100)}% of old')
for j,ax in enumerate(axes[1]):
    if j==0:
        ax.axis('off');continue
    old_coords,new_coords=embeddings[j-1]
    ax.scatter(old_coords[plot_old_idx,0],old_coords[plot_old_idx,1],c=y_old[plot_old_idx],cmap='tab10',vmin=0,vmax=9,s=3,alpha=.35,rasterized=True)
    ax.scatter(new_coords[plot_new_idx,0],new_coords[plot_new_idx,1],c=y_new[plot_new_idx],cmap='tab10',vmin=0,vmax=9,s=3,alpha=.35,rasterized=True)
    ax.set(xlim=xlim,ylim=ylim,title='Old + new')
for ax in axes.flat:
    ax.set_xticks([]);ax.set_yticks([])
fig.suptitle('SONG MNIST rehearsal: old digits 0-4, then new digits 5-9 (aligned to original old map)')
fig.savefig(ROOT/'reports'/'figures'/'song_rehearsal_mnist.png',dpi=150)
plt.close(fig)
