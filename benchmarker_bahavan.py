
import sys
import numpy as np
from song.duplex_umap_song import SONG
from sklearn.datasets import make_blobs, load_digits
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import adjusted_mutual_info_score
from umap import UMAP
from scipy.sparse import csr_matrix
from ott.geometry import pointcloud
from ott.core import gromov_wasserstein as gw
import jax
from evals import *
from jax import numpy as jnp
from jax import random
import timeit
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
#
# sys.path.insert(0, 'METHODS/SCOOTR/')
# sys.path.insert(0, 'METHODS/UnionCOM/')
# sys.path.insert(0, 'METHODS/SCOT/')
# sys.path.insert(0, 'METHODS/PAMONA/')
# sys.path.insert(0, 'METHODS/MMDMA/')
# sys.path.insert(0, 'METHODS/DUPLEX_SONG/')

# from scootr import *
# from coot_torch import *
# from scot import *
# from scot_version_two import *
# from evals import *
# from Pamona import *
# from mmdma import *
# from UnionCom import *
# from GW_early_registration import GASSO_D  # GASSO_D_disected

from sklearn.neighbors import KNeighborsClassifier



def GASSO_D(X_tr1,X_tr2, prototypes = 500,final_vector_count=100, n_neighbors=4, max_age=3,  so_steps=200, pow_err=1):


    print(X_tr1.shape[0], X_tr2.shape[0])

    model = SONG(verbose = 1, final_vector_count=final_vector_count, n_neighbors=n_neighbors, max_age=max_age,  vis_neis= n_neighbors + 15, so_steps=so_steps, pow_err=pow_err)
    print ("Model trained")
    model.fit([X_tr1, X_tr2])
    print ("Model fitted")

    geom_xx = pointcloud.PointCloud(x = model.W[0], y = model.W[0])
    geom_yy = pointcloud.PointCloud(x = model.W[1], y = model.W[1])

    out = gw.gromov_wasserstein(geom_xx=geom_xx, geom_yy=geom_yy, epsilon = 0.01, max_iterations = 50, jit = True)

    n_outer_iterations = jnp.sum(out.costs != -1)
    has_converged = bool(out.linear_convergence[n_outer_iterations - 1])
    print(f'{n_outer_iterations} outer iterations were needed.')
    print(f'The last Sinkhorn iteration has converged: {has_converged}')
    # print(f'The outer loop of Gromov Wasserstein has converged: {out.converged}')
    print(f'The final regularised GW cost is: {out.reg_gw_cost:.3f}')
    transport = out.matrix

    second_manifold_shift_order = jnp.array(np.argmax(transport, axis=1))
    model.W[1] = model.W[1][second_manifold_shift_order]



    model.ss = 1000

    print(f"model ss = {model.ss}")
    model.lrst=.5
    model.prototypes = prototypes

    Y1, Y2 = model.transform([X_tr1, X_tr2])
    return model,Y1,Y2



def transfer_accuracy_test(data1, data2, type1, type2):
    """
    Metric from UnionCom: "Label Transfer Accuracy"
    """
    Min = np.minimum(len(data1), len(data2))
    k = np.maximum(10, (len(data1) + len(data2)) * 0.01)
    k = k.astype(np.int)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(data2, type2)
    type1_predict = knn.predict(data1)
    # np.savetxt("type1_predict.txt", type1_predict)
    count = 0
    for label1, label2 in zip(type1_predict, type1):
        if label1 == label2:
            count += 1
    return count / len(type1)


# DATASET LOADER

# scGEM
def scGEM():
    X = np.genfromtxt("DATASETS/SCOT_data/scGEM_expression.csv", delimiter=",")
    y = np.genfromtxt("DATASETS/SCOT_data/scGEM_methylation.csv", delimiter=",")
    X_label = np.loadtxt("DATASETS/SCOT_data/scGEM_typeExpression.txt")
    y_label = np.loadtxt("DATASETS/SCOT_data/scGEM_typeMethylation.txt")
    X_label = X_label.astype(np.int)
    y_label = y_label.astype(np.int)

    hyperparameters = {"SCOT": {"k": 35, "e": 5e-3, "normalize": True, "norm": "l2"},
                       "SCOTv2": {},
                       "SCOOTR": {},
                       "PAMONA": {"n_shared": [138], "Lambda": 10, "output_dim": 5},
                       "UnionCOM": {},
                       "MMDMA": {},
                       "VQMA": {}}

    viz_data = None  # {"shared1":shared1,"specific1":specific1,"shared2":shared2,"specific2":specific2}

    return X, y, X_label, y_label, hyperparameters, viz_data


# SNAREseq
def SNAREseq():
    X = np.load("DATASETS/SCOT_data/scatac_feat.npy")
    y = np.load("DATASETS/SCOT_data/scrna_feat.npy")
    X_label = np.loadtxt("DATASETS/SCOT_data/SNAREseq_atac_types.txt")
    y_label = np.loadtxt("DATASETS/SCOT_data/SNAREseq_rna_types.txt")
    X_label = X_label.astype(np.int)  # k=110, e=1e-3
    y_label = y_label.astype(np.int)
    hyperparameters = {"SCOT": {"k": 110, "e": 1e-3, "normalize": True, "norm": "l2"},
                       "SCOTv2": {"selfTune": True},
                       "SCOOTR": {"selfTune": True},
                       "PAMONA": {"n_shared": [138], "Lambda": 10, "output_dim": 5},
                       "UnionCOM": {},
                       "MMDMA": {},
                       "VQMA": {}}

    viz_data = None  # {"shared1":shared1,"specific1":specific1,"shared2":shared2,"specific2":specific2}

    return X, y, X_label, y_label, hyperparameters, viz_data


def HSC():
    X = np.loadtxt("METHODS/UnionCOM/hsc/domain1.txt")
    y = np.loadtxt("METHODS/UnionCOM/hsc/domain2.txt")
    X_label = np.loadtxt("METHODS/UnionCOM/hsc/type1.txt")
    y_label = np.loadtxt("METHODS/UnionCOM/hsc/type2.txt")
    X_label = X_label.astype(np.int)  # k=110, e=1e-3
    y_label = y_label.astype(np.int)

    hyperparameters = {"SCOT": {},
                       "SCOTv2": {},
                       "SCOOTR": {},
                       "PAMONA": {},
                       "UnionCOM": {"integration_type": 'BatchCorrect', "distance_mode": 'cosine'},
                       "MMDMA": {},
                       "VQMA": {}}
    viz_data = None  # {"shared1":shared1,"specific1":specific1,"shared2":shared2,"specific2":specific2}

    return X, y, X_label, y_label, hyperparameters, viz_data


def PBMC():
    from sklearn.preprocessing import StandardScaler
    def zscore_standardize(data):
        """
        From SCOT code: https://github.com/rsinghlab/SCOT
        """
        scaler = StandardScaler()
        scaledData = scaler.fit_transform(data)
        return scaledData

    data1 = np.loadtxt("DATASETS/PAMONA_data/PBMC/ATAC_scaledata.txt")
    data2 = np.loadtxt("DATASETS/PAMONA_data/PBMC/RNA_scaledata.txt")
    type1 = np.loadtxt("DATASETS/PAMONA_data/PBMC/ATAC_type.txt")
    type2 = np.loadtxt("DATASETS/PAMONA_data/PBMC/RNA_type.txt")
    data1 = zscore_standardize(np.asarray(data1))
    data2 = zscore_standardize(np.asarray(data2))
    type1 = type1.astype(np.int)
    type2 = type2.astype(np.int)

    index1 = np.argwhere(type1 == 0).reshape(1, -1).flatten()
    index2 = np.argwhere(type1 == 1).reshape(1, -1).flatten()
    index3 = np.argwhere(type1 == 2).reshape(1, -1).flatten()
    index4 = np.argwhere(type1 == 3).reshape(1, -1).flatten()
    shared1 = np.hstack((index1, index2))
    shared1 = np.hstack((shared1, index3))
    shared1 = np.hstack((shared1, index4))

    index1 = np.argwhere(type1 == 4).reshape(1, -1).flatten()
    index2 = np.argwhere(type1 == 5).reshape(1, -1).flatten()
    specific1 = np.hstack((index1, index2))

    index1 = np.argwhere(type2 == 0).reshape(1, -1).flatten()
    index2 = np.argwhere(type2 == 1).reshape(1, -1).flatten()
    index3 = np.argwhere(type2 == 2).reshape(1, -1).flatten()
    index4 = np.argwhere(type2 == 3).reshape(1, -1).flatten()
    shared2 = np.hstack((index1, index2))
    shared2 = np.hstack((shared2, index3))
    shared2 = np.hstack((shared2, index4))

    index1 = np.argwhere(type2 == 6).reshape(1, -1).flatten()
    index2 = np.argwhere(type2 == 7).reshape(1, -1).flatten()
    index3 = np.argwhere(type2 == 8).reshape(1, -1).flatten()
    index4 = np.argwhere(type2 == 9).reshape(1, -1).flatten()
    specific2 = np.hstack((index1, index2))
    specific2 = np.hstack((specific2, index3))
    specific2 = np.hstack((specific2, index4))
    viz_data = {"shared1": shared1, "specific1": specific1, "shared2": shared2, "specific2": specific2}

    hyperparameters = {"SCOT": {"e": 5e-3, "normalize": True, "norm": "l2"},
                       "SCOTv2": {"e": 1e-2, "normalize": True, "norm": "l2"},
                       "SCOOTR": {"reg": 1},
                       "PAMONA": {"n_shared": [1649], "n_neighbors": 30},
                       "UnionCOM": {},
                       "MMDMA": {},
                       "VQMA": {}}

    return data1, data2, type1, type2, hyperparameters, viz_data


def scNMT():
    mode = 0
    X = np.loadtxt("DATASETS/PAMONA_data/scNMT/acc_30.txt")

    if (mode):
        y = np.loadtxt("DATASETS/PAMONA_data/scNMT/rna_30.txt")
    else:
        y = np.loadtxt("DATASETS/PAMONA_data/scNMT/met_30.txt")

    X_label = np.loadtxt("DATASETS/PAMONA_data/scNMT/acc_stage.txt")

    if (mode):
        y_label = np.loadtxt("DATASETS/PAMONA_data/scNMT/rna_stage.txt")
    else:
        y_label = np.loadtxt("DATASETS/PAMONA_data/scNMT/met_stage.txt")

    X_label = X_label.astype(np.int)
    y_label = y_label.astype(np.int)

    hyperparameters = {"SCOT": {"k": 35, "e": 5e-3, "normalize": True, "norm": "l2"},
                       "SCOTv2": {},
                       "SCOOTR": {},
                       "PAMONA": {"Lambda": 10, "n_neighbors": 40},
                       "UnionCOM": {},
                       "MMDMA": {},
                       "VQMA": {}}

    viz_data = None

    return X, y, X_label, y_label, hyperparameters, viz_data


from sklearn.datasets import make_swiss_roll


def SWISS_ROLL_GENERATOR(n_samples=500, label_discrete=100):
    Latent, y = make_swiss_roll(n_samples=n_samples, noise=0.2, random_state=32, hole=False)
    np.random.seed(0)
    data1_gen = np.random.rand(3, 1000)
    np.random.seed(2)
    data2_gen = np.random.rand(3, 800)

    def sigmoid(x, alpha):
        return 1 / (1 + np.exp(-x * alpha))

    y = np.int8(10 * y)
    # if (label_discrete!=0):
    # 	sorted_y = np.sort(y)
    # 	bins = [(int(np.min(y)+ele*256/label_discrete)) for ele in range(1,label_discrete)]
    # 	y = np.digitize(y, bins=bins)

    data1 = 10000 * sigmoid(np.matmul(Latent, data1_gen), 1)
    data2 = 10000 * sigmoid(np.square(np.matmul(Latent, data2_gen)), 1)

    hyperparameters = {"SCOT": {},
                       "SCOTv2": {},
                       "SCOOTR": {},
                       "PAMONA": {},
                       "UnionCOM": {},
                       "MMDMA": {},
                       "VQMA": {}}
    viz_data = None
    return data1.astype(np.float32), data2.astype(np.float32), y, y, hyperparameters, viz_data



from sklearn import preprocessing
import matplotlib.pyplot as plt


def VQMA(X, y, X_label, y_label, hyperparameters):
    print("Starting with " + str(X.shape[0]) + " " + str(y.shape[0]) + " prototypes")
    X = preprocessing.StandardScaler().fit_transform(X)
    y = preprocessing.StandardScaler().fit_transform(y)
    model, X_new, y_new = GASSO_D(X, y)

    Y1, Y2 = model.transform([X, y])
    Ycombined = np.concatenate([Y1, Y2], axis=0)
    Ylabels_special = np.concatenate([5 * np.ones(Y1.shape[0]), 3 * np.ones(Y2.shape[0])], axis=0)

    fig, ax = plt.subplots(figsize=(12, 10))
    plt.scatter(Ycombined[:, 0], Ycombined[:, 1], c=Ylabels_special, cmap="Spectral", s=10)
    plt.title("SONG Transform: Data1 and Data2 visualized on top of each other", fontsize=18)
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(Y1[:, 0], Y1[:, 1], c=X_label, cmap="Spectral", s=10)
    ax1.set_title("System 01")
    ax2.scatter(Y2[:, 0], Y2[:, 1], c=y_label, cmap="Spectral", s=10)
    ax2.set_title("system 02")
    plt.legend()
    plt.title("Side by Side", fontsize=18)
    plt.show()

    return [X_new, y_new]


def Optimize(dataset, method):
    X, y, X_label, y_label, hyperparameters, viz_data = dataset_functions[dataset]
    import time
    timestart = time.process_time()
    integrated_data = method_functions[method](X, y, X_label, y_label, hyperparameters[method])
    timetaken = time.process_time() - timestart
    X_new = integrated_data[0]
    y_new = integrated_data[1]

    if viz_data == None:
        fracs = calc_domainAveraged_FOSCTTM(X_new, y_new)
        transfer_acc = transfer_accuracy_test(X_new, y_new, X_label, y_label)
        alignmentscore = test_alignment_score(X_new, y_new)
        return np.mean(fracs), alignmentscore, np.mean(transfer_acc), timetaken
    else:
        print("Optimized with visuals")
        fracs = calc_domainAveraged_FOSCTTM(X_new, y_new)
        transfer_acc = transfer_accuracy_test(X_new[viz_data["shared1"]], y_new, X_label[viz_data["shared1"]], y_label)
        alignmentscore = test_alignment_score(X_new[viz_data["shared1"]], y_new[viz_data["shared2"]],
                                              X_new[viz_data["specific1"]], y_new[viz_data["specific2"]])
        return np.mean(fracs), alignmentscore, np.mean(transfer_acc), timetaken


if __name__ == "__main__":
    dataset_functions = {
        # "scGEM": scGEM(),
        # "SNAREseq": SNAREseq(),
        # "HSC":HSC(),
        # "PBMC":PBMC(),
        # "scNMT":scNMT(),
        # "SWISSROLL_100":SWISS_ROLL_GENERATOR(n_samples=100,label_discrete=50),
        # "SWISSROLL_200":SWISS_ROLL_GENERATOR(n_samples=100, label_discrete=100),
        # "SWISSROLL_300":SWISS_ROLL_GENERATOR(n_samples=100, label_discrete=100),
        # "SWISSROLL_100_50":SWISS_ROLL_GENERATOR(n_samples=100, label_discrete=50),
        # "SWISSROLL_100_OPEN":SWISS_ROLL_GENERATOR(n_samples=100),
        # "SWISSROLL_500_50":SWISS_ROLL_GENERATOR(n_samples=500, label_discrete=50),
        "SWISSROLL_500_OPEN": SWISS_ROLL_GENERATOR(n_samples=5000),
        # "SWISSROLL_1000_50":SWISS_ROLL_GENERATOR(n_samples=1000, label_discrete=100),
        # "SWISSROLL_1000_OPEN":SWISS_ROLL_GENERATOR(n_samples=1000),
        # "#SWISSROLL_5000_50":SWISS_ROLL_GENERATOR(n_samples=5000, label_discrete=500),
        # "SWISSROLL_5000_OPEN":SWISS_ROLL_GENERATOR(n_samples=5000),
        # "SWISSROLL_10000_50":SWISS_ROLL_GENERATOR(n_samples=10000, label_discrete=500),
        # "SWISSROLL_10000_OPEN":SWISS_ROLL_GENERATOR(n_samples=10000),
        # "SWISSROLL_50000_50":SWISS_ROLL_GENERATOR(n_samples=50000, label_discrete=500),
        # "SWISSROLL_50000_OPEN":SWISS_ROLL_GENERATOR(n_samples=50000),
        # "SWISSROLL_100000_50":SWISS_ROLL_GENERATOR(n_samples=100000, label_discrete=500),
        # "SWISSROLL_100000_OPEN":SWISS_ROLL_GENERATOR(n_samples=100000)
    }

    method_functions = {
        # "SCOT": SCOT_holder,
        # "SCOTv2": SCOTv2_holder,
        # "SCOOTR": SCOOTR_holder,
        # "PAMONA": PAMONA_holder,
        # "UnionCOM": UnionCOM_holder,
        # "MMDMA": MMD_MA_holder,
        "VQMA": VQMA
    }

    methods = method_functions.keys()
    datasets = dataset_functions.keys()
    # dataset = "SWISSROLL"

    from datetime import datetime

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

    for dataset in datasets:
        i = 0
        results = {}
        for method in methods:
            print("Evaluating method : " + method)
            FOSCTTM, ALIGNMENTSCORE, LABEL_TRANSFER, timetaken = Optimize(dataset, method)
            results[i] = {'METHOD': method, 'DATASET': dataset, "FOSCTTM": FOSCTTM, "Alignment Score": ALIGNMENTSCORE,
                          "Label Transfer Accuracy": LABEL_TRANSFER, "Time Taken": timetaken}
            print(results[i])
            i = i + 1

    # import json
    # with open("REPORTS/JSONS/RUN_"+dt_string+" "+str(dataset)+".json", "w") as fp:
    # 	json.dump(results, fp)

