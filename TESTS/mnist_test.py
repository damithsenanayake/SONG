from song import SONG
from sklearn.datasets import make_blobs, load_digits
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import adjusted_mutual_info_score
from umap import UMAP
import numpy as np
import timeit
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

models = [SONG(spread_factor=0.85, n_neighbors = 1,  final_vector_count=2500, epsilon=1e-10, verbose=1)]

samplesize = 60000
X = np.array(pd.read_csv('~/data/mnist/mnist_train.csv')).astype(float)
c = X[: samplesize, 0].astype(float)
pca = PCA(60)
X_tr =   pca.fit_transform(X[:samplesize, 1:].astype(float))
print (pca.explained_variance_ratio_.sum())


def getImage(x):
    return OffsetImage(255-x, cmap='gray')
p = 1

for model in models:
    tic = timeit.default_timer()

    Y = model.fit_transform(X_tr) * 100
    toc = timeit.default_timer()
    print (toc - tic)

    clusterer = KMeans(10)
    clusterer.fit(Y)

    clusters =  clusterer.labels_


    print ('AMIS : ', str(adjusted_mutual_info_score(c, clusters)))
    x, y = Y.T
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(x, y, c = c)


    plt.savefig('/home/student.unimelb.edu.au/senanayaked/mnist_umap_ims.png', bbox_inches='tight')
    plt.show()
    p += 1
    plt.figure(figsize=(10, 10))
    plt.scatter(Y.T[0], Y.T[1], c=plt.cm.Spectral(c/10.), alpha=1., s=3)
    plt.savefig('/home/student.unimelb.edu.au/senanayaked/mnist_umap_labels.png', bbox_inches='tight')

