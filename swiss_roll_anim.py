import numpy as np
from sklearn.datasets import make_swiss_roll
from mpl_toolkits.mplot3d import Axes3D
from song.song import SONG
import matplotlib.pyplot as plt

'''Helper Functions'''
def plotgraph3d(ax, W, G):
    ax.scatter(W[:, 0], W[:, 1], W[:, 2], c = 'black')
    for i in range(len(G)):
        for j in range(len(G)):
            if G[i][j]:
                ax.plot([W[i, 0], W[j, 0]], [W[i, 1], W[j, 1]], [W[i, 2], W[j, 2]], c='black',linewidth = G[i][j])
    return ax

def plotgraph2d(ax, W, G):
    ax.scatter(W[:, 0], W[:, 1], c = 'black', s= 3)
    for i in range(len(G)):
        for j in range(len(G)):
            if G[i][j]:
                ax.plot([W[i, 0], W[j, 0]], [W[i, 1], W[j, 1]], c='black', linewidth = G[i][j])
    return ax
def scatter3d(ax, W):
        ax.scatter(W[:, 0], W[:, 1], W[:, 2], c = 'black')

def scatter3drainbow(ax, W, t):
        ax.scatter(W[:, 0], W[:, 1], W[:, 2], c = t, s= 2, cmap=plt.cm.gist_rainbow)


def scatter2d(ax, W):
    ax.scatter(W[:, 0], W[:, 1], c='black', s= 2.5)


def scatter2drainbow(ax, W, t):
    ax.scatter(W[:, 0], W[:, 1], c=t, s = 2, cmap=plt.cm.gist_rainbow)


X, t = make_swiss_roll(n_samples=1000)

song = SONG(so_steps=5, agility=1., verbose=1, final_vector_count=3, fvc_growth=0., min_dist=0.5, um_min_dist=0.5)
for i in range(50):
    Y = song.fit_transform(X)
    W = song.W
    prot = song.Y
    G = song.G
    # if i>8:
    #     song.mutable_graph = False
    if i < 15:
        song.prototypes += 10
    elif i%5==0:
        song.prototypes += 4
    # song.lrst -= 0.005
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(122, projection = '3d')
    ax.view_init(azim=-100, elev=5.)
    scatter3drainbow(ax, X, t)
    plotgraph3d(ax, W, song.G)
    ax = fig.add_subplot(121)
    scatter2drainbow(ax, Y, t)
    plotgraph2d(ax, prot, G)

    plt.savefig(f'figs/{i+1:2d}.png')



