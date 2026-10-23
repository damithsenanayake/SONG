"""Digits 0-4 followed by digits 5-9: structure retention in SONG and UMAP."""
from pathlib import Path
import json, sys, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import procrustes
from sklearn.neighbors import NearestNeighbors
from umap import UMAP
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
UMAP(n_epochs=11,random_state=42).fit_transform(X_old[:80,:50])

def retention(before,after):
    rng=np.random.default_rng(42)
    idx=rng.choice(len(before),1000,replace=False)
    a=np.asarray(before[idx],dtype=np.float64)
    b=np.asarray(after[idx],dtype=np.float64)
    aa,bb,disparity=procrustes(a,b)
    neigh_a=NearestNeighbors(n_neighbors=16).fit(before).kneighbors(before[idx],return_distance=False)[:,1:]
    neigh_b=NearestNeighbors(n_neighbors=16).fit(after).kneighbors(after[idx],return_distance=False)[:,1:]
    overlap=np.mean([len(set(u)&set(v))/15 for u,v in zip(neigh_a,neigh_b)])
    return {'aligned_procrustes_disparity_1000':float(disparity),'old_neighbor_overlap_at_15':float(overlap)}

song=SONG(so_steps=30,random_seed=42,dispersion_method='none')
t=time.perf_counter();song.fit(X_old);song_old_fit=time.perf_counter()-t
Y_song_before=song.transform(X_old).copy()
t=time.perf_counter();song.fit(X_new);song_new_fit=time.perf_counter()-t
Y_song_after=song.transform(X_old).copy()
Y_song_new=song.transform(X_new).copy()
X_old_pc=song._get_XPCA(X_old).astype(np.float32)
X_new_pc=song._get_XPCA(X_new).astype(np.float32)
print('SONG complete',flush=True)
reducer=UMAP(n_neighbors=15,min_dist=.1,n_components=2,random_state=42,transform_seed=42,n_epochs=50)
t=time.perf_counter();Y_umap_before=reducer.fit_transform(X_old_pc).copy();umap_old_fit=time.perf_counter()-t
print('UMAP old fit complete',flush=True)
t=time.perf_counter();Y_umap_new=reducer.transform(X_new_pc);umap_transform=time.perf_counter()-t
Y_umap_after=Y_umap_before.copy()
try:
    t=time.perf_counter();reducer.update(X_new_pc);umap_update=time.perf_counter()-t
    update_error=None
except Exception as exc:
    umap_update=None
    update_error=f'{type(exc).__name__}: {exc}'
print('UMAP transform/update complete',update_error,flush=True)
results={'dataset':'Keras MNIST train; old digits 0-4, new digits 5-9','old_n':len(X_old),'new_n':len(X_new),'song_steps_per_fit':30,'umap_epochs':50,'seed':42,
 'song':{'old_fit_s':song_old_fit,'new_fit_s':song_new_fit,'prototypes_after':len(song.W),**retention(Y_song_before,Y_song_after)},
 'umap':{'old_fit_s':umap_old_fit,'transform_new_s':umap_transform,'update_s':umap_update,'update_error':update_error,**retention(Y_umap_before,Y_umap_after)}}
(ROOT/'reports'/'song_increment_mnist.json').write_text(json.dumps(results,indent=2)+'\n')
fig,axes=plt.subplots(2,2,figsize=(12,11),constrained_layout=True)
rng=np.random.default_rng(42)
for ax,name,old_coords,new_coords,old_y,new_y in [
 (axes[0,0],'SONG before',Y_song_before,None,y_old,None),
 (axes[0,1],'SONG after',Y_song_after,Y_song_new,y_old,y_new),
 (axes[1,0],'UMAP before',Y_umap_before,None,y_old,None),
 (axes[1,1],'UMAP transform (old points fixed)',Y_umap_after,Y_umap_new,y_old,y_new)]:
    oi=rng.choice(len(old_coords),min(6000,len(old_coords)),replace=False)
    ax.scatter(old_coords[oi,0],old_coords[oi,1],c=old_y[oi],cmap='tab10',vmin=0,vmax=9,s=3,alpha=.5,rasterized=True)
    if new_coords is not None:
        ni=rng.choice(len(new_coords),min(6000,len(new_coords)),replace=False)
        ax.scatter(new_coords[ni,0],new_coords[ni,1],c=new_y[ni],cmap='tab10',vmin=0,vmax=9,s=3,alpha=.5,rasterized=True)
    ax.set_title(name);ax.set_xticks([]);ax.set_yticks([])
fig.suptitle('MNIST incremental addition: digits 0-4 followed by 5-9')
fig.savefig(ROOT/'reports'/'figures'/'song_increment_mnist.png',dpi=160)
print(json.dumps(results,indent=2),flush=True)
