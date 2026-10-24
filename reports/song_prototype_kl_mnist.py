"""Prototype-KL incremental SONG experiment with no old input replay."""
from copy import deepcopy
from pathlib import Path
import json,sys,time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from song.song import SONG
from song.retention import neighborhood_temperature,position_kl_loss_and_grad
with np.load(ROOT/'data'/'mnist.npz') as z:
    X=z['x_train'].reshape(-1,784).astype(np.float32)/255
    y=z['y_train']
old=y<5
X_old,X_new=X[old],X[~old]
y_old,y_new=y[old],y[~old]
SONG(so_steps=2,final_vector_count=16,dispersion_method='none').fit_transform(X_old[:80])
base=SONG(so_steps=30,random_seed=42,dispersion_method='none',
          retention_source='prototypes',retention_kl_weight=1.,
          retention_position_weight=1.,retention_maxiter=30)
base.fit(X_old)
Y0=base.transform(X_old).copy()
W0=base.W.copy();proto_Y0=base.Y.copy()
rng=np.random.default_rng(42)
old_probe_tr,old_probe_te=train_test_split(np.arange(len(X_old)),test_size=.2,random_state=42,stratify=y_old)
new_probe_tr,new_probe_te=train_test_split(np.arange(len(X_new)),test_size=.2,random_state=42,stratify=y_new)
metric_idx=rng.choice(len(X_old),1000,replace=False)
plot_old=rng.choice(len(X_old),5000,replace=False)
plot_new=rng.choice(len(X_new),5000,replace=False)
replay_order=rng.permutation(len(X_old))
A=Y0[metric_idx].astype(np.float64);Ac=A.mean(axis=0)
centers0=np.stack([Y0[y_old==c].mean(axis=0) for c in range(5)])
spread=np.sqrt(np.mean(np.sum((Y0-Y0.mean(axis=0))**2,axis=1)))

def probe(coords,labels,tr,te):
    classifier=KNeighborsClassifier(n_neighbors=5,n_jobs=1).fit(coords[tr],labels[tr])
    return float(accuracy_score(labels[te],classifier.predict(coords[te])))

def align(coords):
    B=coords[metric_idx].astype(np.float64);Bc=B.mean(axis=0)
    u,s,vt=np.linalg.svd((B-Bc).T@(A-Ac),full_matrices=False)
    R=u@vt;scale=s.sum()/np.sum((B-Bc)**2)
    aligned=(coords.astype(np.float64)-Bc)@R*scale+Ac
    centers1=np.stack([aligned[y_old==c].mean(axis=0) for c in range(5)])
    drift=float(np.sqrt(np.mean(np.sum((centers1-centers0)**2,axis=1)))/spread)
    return aligned,Bc,R,scale,drift

configs=[
 ('Standard: new only','samples',0.,0.,0.),
 ('Protected indices; no KL','prototypes',0.,0.,0.),
 ('Prototype neighborhood KL; new only','prototypes',1.,0.,0.),
 ('Prototype position KL; new only','prototypes',0.,1.,0.),
 ('Prototype both KL; new only','prototypes',1.,1.,0.),
 ('Restore trained prototypes; new only','prototypes',0.,0.,0.),
 ('Raw replay 35%; no KL','samples',0.,0.,.35),
]
rows=[];embeddings={}
for name,source,w_kl,w_pos,replay in configs:
    model=deepcopy(base)
    model.retention_source=source
    model.retention_kl_weight=w_kl
    model.retention_position_weight=w_pos
    count=round(replay*len(X_old))
    X_update=np.concatenate([X_new,X_old[replay_order[:count]]],axis=0)
    start=time.perf_counter();model.fit(X_update)
    if name.startswith('Restore trained prototypes'):
        model.W[:len(W0)] = W0
        model.Y[:len(proto_Y0)] = proto_Y0
    elapsed=time.perf_counter()-start
    Y_old=model.transform(X_old).copy()
    Y_new=model.transform(X_new).copy()
    aligned,Bc,R,scale,drift=align(Y_old)
    aligned_new=(Y_new.astype(np.float64)-Bc)@R*scale+Ac
    old_Wpc=base.pca.transform(W0).astype(np.float64)
    now_Wpc=model.pca.transform(model.W[:len(W0)]).astype(np.float64)
    _,_,input_kl,input_position_kl=position_kl_loss_and_grad(
        now_Wpc,old_Wpc,neighborhood_temperature(old_Wpc),1.,1.)
    _,_,output_kl,output_position_kl=position_kl_loss_and_grad(
        model.Y[:len(W0)],proto_Y0,neighborhood_temperature(proto_Y0),1.,1.)
    row={'condition':name,'old_replay_n':count,'old_prototypes':len(W0),
         'prototypes_after':len(model.W),'fit_s':elapsed,
         'old_class_5nn_accuracy':probe(Y_old,y_old,old_probe_tr,old_probe_te),
         'new_class_5nn_accuracy':probe(Y_new,y_new,new_probe_tr,new_probe_te),
         'normalized_old_class_centroid_rms_drift':drift,
         'old_prototype_input_kl_nats':input_kl,
         'old_prototype_input_position_kl_nats':input_position_kl,
         'old_prototype_output_kl_nats':output_kl,
         'old_prototype_output_position_kl_nats':output_position_kl,
         'optimizer':model.retention_result_}
    rows.append(row);embeddings[name]=(aligned,aligned_new)
    print(row,flush=True)
results={'seed':42,'old_n':len(X_old),'new_n':len(X_new),'song_steps_per_fit':30,
         'initial_old_class_5nn_accuracy':probe(Y0,y_old,old_probe_tr,old_probe_te),
         'conditions':rows}
(ROOT/'reports'/'song_prototype_kl_mnist.json').write_text(json.dumps(results,indent=2)+'\n')
plot_names=['Standard: new only','Prototype neighborhood KL; new only',
            'Prototype position KL; new only','Prototype both KL; new only',
            'Raw replay 35%; no KL']
plot_titles=['New only','Prototype neighbor KL','Prototype position KL',
             'Prototype both KL','Raw replay 35%']
fig,axes=plt.subplots(2,len(plot_names),figsize=(20,8),constrained_layout=True)
common=np.concatenate([embeddings[name][0][plot_old] for name in plot_names]+[embeddings[name][1][plot_new] for name in plot_names])
limits=np.quantile(common,[.005,.995],axis=0);pad=(limits[1]-limits[0])*.04
for j,name in enumerate(plot_names):
    old_coords,new_coords=embeddings[name]
    for i,ax in enumerate(axes[:,j]):
        ax.scatter(old_coords[plot_old,0],old_coords[plot_old,1],c=y_old[plot_old],cmap='tab10',vmin=0,vmax=9,s=3,alpha=.65 if i==0 else .35,rasterized=True)
        if i:
            ax.scatter(new_coords[plot_new,0],new_coords[plot_new,1],c=y_new[plot_new],cmap='tab10',vmin=0,vmax=9,s=3,alpha=.35,rasterized=True)
        ax.set_xlim(limits[0,0]-pad[0],limits[1,0]+pad[0]);ax.set_ylim(limits[0,1]-pad[1],limits[1,1]+pad[1])
        ax.set_title(plot_titles[j] if i==0 else 'Old + new',fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
fig.suptitle('SONG trained-prototype KL versus input replay; MNIST digits 0-4 then 5-9')
fig.savefig(ROOT/'reports'/'figures'/'song_prototype_kl_mnist.png',dpi=150)
plt.close(fig)
