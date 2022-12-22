from song.duplex_umap_song import SONG
from sklearn.datasets import make_blobs, load_digits
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import adjusted_mutual_info_score
from umap import UMAP
from scipy.sparse import csr_matrix
import numpy as np
from ott.geometry import pointcloud
from ott.core import gromov_wasserstein as gw
import jax
from jax import numpy as jnp
from jax import random
import timeit
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def plotgraph2d(ax, W, G):
    ax.scatter(W[:, 0], W[:, 1], c = 'black', s = 4)
    for i in range(len(G)):
        for j in range(len(G)):
            if G[i][j]:
                ax.plot([W[i, 0], W[j, 0]], [W[i, 1], W[j, 1]], c='black', linewidth =G[i][j], alpha=0.2)
    return ax

samplesize = 30000
X = np.array(pd.read_csv('~/data/mnist/mnist_train.csv')).astype(float)
c1 = X[: samplesize, 0].astype(float)
X_tr1 = PCA(n_components=50).fit_transform((X[:samplesize, 1:].astype(np.float32)))
X_tr2 = PCA(n_components=120).fit_transform(X[samplesize:, 1:].astype(np.float32))
c2 = X[samplesize:, 0].astype(float)
knn1 = KNeighborsClassifier(n_neighbors=5)
knn2 = KNeighborsClassifier(n_neighbors=5)
knn1.fit(X_tr1, c1)
knn2.fit(X_tr2, c2)

print(X_tr1.shape[0], X_tr2.shape[0])

#%% ----- EARLY MODEL TRAINING ------
model = SONG(verbose = 1, final_vector_count=250, n_neighbors=5, max_age=1,  so_steps=200, pow_err=1 )
model.fit([X_tr1, X_tr2])

#%% ----- GROMOV WASSERSTEIN REALIGNMENT OF COARSE GRAINED CODING SET DONE IN THIS BLOCK
geom_xx = pointcloud.PointCloud(x = model.W[0], y = model.W[0])
geom_yy = pointcloud.PointCloud(x = model.W[1], y = model.W[1])

out = gw.gromov_wasserstein(geom_xx=geom_xx, geom_yy=geom_yy, epsilon = 100, max_iterations = 20, jit = True)

n_outer_iterations = jnp.sum(out.costs != -1)
has_converged = bool(out.linear_convergence[n_outer_iterations - 1])
print(f'{n_outer_iterations} outer iterations were needed.')
print(f'The last Sinkhorn iteration has converged: {has_converged}')
print(f'The outer loop of Gromov Wasserstein has converged: {out.convergence}')
print(f'The final regularised GW cost is: {out.reg_gw_cost:.3f}')
transport = out.matrix

second_manifold_shift_order = jnp.array(np.argmax(transport, axis=1))
model.W[1] = model.W[1][second_manifold_shift_order]

#%% ------------------------------------------------------------------------------


model.ss = 500
model.lrst=.5
model.prototypes = 1200

tic = timeit.default_timer()
#%% ---- VISUALISATION ------------
Y1, Y2 = model.transform([X_tr1, X_tr2])


fig, axes = plt.subplots(2, 3, figsize = (16, 16))
ax = axes.flatten()[2]
plotgraph2d(ax, model.Y, model.G)
ax = axes.flatten()[0]

ax = axes.flatten()[0]
ax.scatter(Y1.T[0], Y1.T[1], c = c1, cmap = plt.cm.tab10, s= 2)


ax = axes.flatten()[1]
ax.scatter(Y2.T[0], Y2.T[1], c = c2, cmap = plt.cm.tab10, s= 2)
Y1, Y2 = model.fit_transform([X_tr1, X_tr2])
toc = timeit.default_timer()


fig, axes = plt.subplots(2, 3, figsize = (16, 16))
ax = axes.flatten()[2]
plotgraph2d(ax, model.Y, model.G)
ax = axes.flatten()[0]

ax = axes.flatten()[0]
ax.scatter(Y1.T[0], Y1.T[1], c = c1, cmap = plt.cm.tab10, s= 2)


ax = axes.flatten()[1]
ax.scatter(Y2.T[0], Y2.T[1], c = c2, cmap = plt.cm.tab10, s= 2)
plt.show()

