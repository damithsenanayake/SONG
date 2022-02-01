from song.duplex_song import SONG
from sklearn.datasets import make_blobs, load_digits
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import adjusted_mutual_info_score
from umap import UMAP
from scipy.sparse import csr_matrix
import numpy as np

import timeit
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def plot_graph(G, Y):
    edges = np.array(np.where(G > 0)).T
    for edge in edges:
        plt.plot([Y[edge[0]][0], Y[edge[1]][0]], [Y[edge[0]][1], Y[edge[1]][1]], c='black', alpha=0.2, linewidth=G[edge[0], edge[1]])

model = SONG(verbose = 1, final_vector_count=200, so_steps=1000, n_neighbors=6, max_age=6)

samplesize = 60000
X = np.array(pd.read_csv('~/data/mnist/mnist_train.csv')).astype(float)
c1 = X[: samplesize, 0].astype(float)
X_tr1 = ((X[:samplesize, 1:].astype(np.float32)))
# X_tr2, c2 = load_digits(10, return_X_y=True)

X_tr2 = X_tr1
c2 = c1

def getImage(x):
    return OffsetImage(255-x, cmap='gray')
p = 1

tic = timeit.default_timer()

Y2, Y1 = model.fit_transform([X_tr2, csr_matrix(X_tr1)])
toc = timeit.default_timer()
print(toc - tic)

fig, axes = plt.subplots(1, 2, figsize = (16, 8))

ax = axes.flatten()[0]

ax.scatter(Y1.T[0], Y1.T[1], c = c1/10., cmap = plt.cm.Spectral, s= 2)


ax = axes.flatten()[1]

ax.scatter(Y2.T[0], Y2.T[1], c = c2/10., cmap = plt.cm.Spectral, s = 2)

plt.show()

