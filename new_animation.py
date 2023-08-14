from song.umap_song import SONG
from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784')
samplesize = 60000
X = mnist.data.values.astype(float)
c = mnist.target.values.astype(float)
X_tr = X

model = SONG(so_steps=2 , agility=0.99999)
train_ixs = []
i = 0
last_count = 1000
for classix in np.arange(10):
    train_ixs.extend(np.where(c==classix)[0])

    increments = np.linspace(last_count, len(train_ixs), 20)

    for inc in increments:

        X_tr = X[train_ixs[:int(inc)]]
        C_tr = c[train_ixs[:int(inc)]]

        C_tr /= 10.

        C_tr = plt.cm.tab10(C_tr)
        Y = model.fit_transform(X_tr)
        plt.figure()
        plt.scatter(Y.T[0], Y.T[1], s = 2, c= C_tr)
        plt.savefig(f"mnist_animation/{i:04d}.png", bbox_inches = 'tight')
        plt.close()
        i+= 1

        print(f"iteration {i}")

    last_count = len(train_ixs)

