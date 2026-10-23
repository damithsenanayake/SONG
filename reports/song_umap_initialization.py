"""Paired MNIST comparison for a draft preprint on SONG-initialized UMAP."""
from pathlib import Path
import sys
import time
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from umap import UMAP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from song.song import SONG

with np.load(ROOT / 'data' / 'mnist.npz') as archive:
    X_train = archive['x_train'].reshape(-1, 784).astype(np.float32) / 255
    X_test = archive['x_test'].reshape(-1, 784).astype(np.float32) / 255
    y_train = archive['y_train']
    y_test = archive['y_test']

SONG(so_steps=2, final_vector_count=16, dispersion_method='none').fit_transform(X_train[:80])
UMAP(n_epochs=11, random_state=42).fit_transform(X_train[:80, :50])

start = time.perf_counter()
song = SONG(so_steps=100, random_seed=42, dispersion_method='none')
song.fit(X_train)
song_fit_s = time.perf_counter() - start
start = time.perf_counter()
Y_core = song.transform(X_train)
core_transform_s = time.perf_counter() - start
start = time.perf_counter()
X_train_pc = song._get_XPCA(X_train).astype(np.float32)
X_test_pc = song._get_XPCA(X_test).astype(np.float32)
pca_transform_s = time.perf_counter() - start
print('SONG fit/core/PCA:', song_fit_s, core_transform_s, pca_transform_s, flush=True)
# Warm the full-size NNDescent and out-of-sample kernels before comparing fits.
warm_umap = UMAP(n_neighbors=15, min_dist=.1, n_components=2, spread=1.,
                 init='spectral', n_epochs=11, learning_rate=1.,
                 random_state=42, transform_seed=42)
warm_umap.fit_transform(X_train_pc)
warm_umap.transform(X_test_pc)
print('Full-size UMAP JIT warm-up complete', flush=True)

configs = [
    ('UMAP spectral, 11 epochs', 'spectral', 11, 1.0),
    ('SONG init, 11 epochs', Y_core, 11, 1.0),
    ('UMAP spectral, 50 epochs', 'spectral', 50, 1.0),
    ('SONG init, 50 epochs', Y_core, 50, 1.0),
    ('UMAP standard defaults', 'spectral', None, 1.0),
]
rows=[]
embeddings={}
for name, init, epochs, lr in configs:
    reducer=UMAP(n_neighbors=15, min_dist=.1, n_components=2, spread=1.,
                 init=init, n_epochs=epochs, learning_rate=lr,
                 random_state=42, transform_seed=42)
    start=time.perf_counter()
    Y_train=reducer.fit_transform(X_train_pc)
    fit_s=time.perf_counter()-start
    start=time.perf_counter()
    Y_test=reducer.transform(X_test_pc)
    transform_s=time.perf_counter()-start
    classifier=KNeighborsClassifier(n_neighbors=5, n_jobs=1)
    classifier.fit(Y_train,y_train)
    predictions=classifier.predict(Y_test)
    row={'method':name,'epochs':epochs,'learning_rate':lr,
         'umap_fit_s':fit_s,'umap_test_transform_s':transform_s,
         'song_fit_plus_core_s':song_fit_s+core_transform_s if name.startswith('SONG') else 0.,
         'total_embedding_fit_s':fit_s+(song_fit_s+core_transform_s if name.startswith('SONG') else 0.),
         'heldout_5nn_accuracy':accuracy_score(y_test,predictions),
         'heldout_5nn_balanced_accuracy':balanced_accuracy_score(y_test,predictions)}
    rows.append(row)
    embeddings[name]=(Y_train,Y_test)
    print(row,flush=True)

results=pd.DataFrame(rows)
results.to_csv(ROOT/'reports'/'song_umap_initialization.csv',index=False)
(ROOT/'reports'/'song_umap_initialization_meta.json').write_text(json.dumps({
 'train_size':60000,'test_size':10000,'features':784,'pca_components':50,
 'song_steps':100,'song_fit_s':song_fit_s,'core_transform_s':core_transform_s,
 'pca_transform_s':pca_transform_s,'seed':42},indent=2)+'\n')

names=['UMAP spectral, 11 epochs','SONG init, 11 epochs','UMAP standard defaults']
rng=np.random.default_rng(42)
train_idx=rng.choice(len(X_train),12000,replace=False)
test_idx=rng.choice(len(X_test),5000,replace=False)
fig,axes=plt.subplots(2,3,figsize=(18,10),constrained_layout=True)
for col,name in enumerate(names):
    Y_train,Y_test=embeddings[name]
    row=results[results.method==name].iloc[0]
    for r,coords,labels,idx in [(0,Y_train,y_train,train_idx),(1,Y_test,y_test,test_idx)]:
        axes[r,col].scatter(coords[idx,0],coords[idx,1],c=labels[idx],cmap='tab10',
                            vmin=0,vmax=9,s=3,alpha=.55,rasterized=True)
        axes[r,col].set_xticks([]);axes[r,col].set_yticks([])
        axes[r,col].set_title(f'{name} — '+('train' if r==0 else f'test; accuracy {row.heldout_5nn_accuracy:.3f}'))
fig.suptitle('MNIST 60k train / 10k test: UMAP initialization comparison')
fig.savefig(ROOT/'reports'/'figures'/'song_umap_initialization_mnist.png',dpi=160)
plt.close(fig)
