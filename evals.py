"""
Author: Ritambhara Singh, Pinar Demetci, Rebecca Santorella
19 February 2020
"""
import numpy as np
import random, math, os, sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


def test_alignment_score(data1_shared, data2_shared, data1_specific=None, data2_specific=None):
    N = 2

    if len(data1_shared) < len(data2_shared):
        data1 = data1_shared
        data2 = data2_shared
    else:
        data2 = data1_shared
        data1 = data2_shared
    data2 = data2[random.sample(range(len(data2)), len(data1))]
    k = np.maximum(10, (len(data1) + len(data2)) * 0.01)
    k = k.astype(np.int)

    data = np.vstack((data1, data2))

    bar_x1 = 0
    for i in range(len(data1)):
        diffMat = data1[i] - data
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)
        NearestN = np.argsort(sqDistances)[1:k + 1]
        for j in NearestN:
            if j < len(data1):
                bar_x1 += 1
    bar_x1 = bar_x1 / len(data1)

    bar_x2 = 0
    for i in range(len(data2)):
        diffMat = data2[i] - data
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)
        NearestN = np.argsort(sqDistances)[1:k + 1]
        for j in NearestN:
            if j >= len(data1):
                bar_x2 += 1
    bar_x2 = bar_x2 / len(data2)

    bar_x = (bar_x1 + bar_x2) / 2

    score = 0
    score += 1 - (bar_x - k / N) / (k - k / N)

    data_specific = None
    flag = 0
    if data1_specific is not None:
        data_specific = data1_specific
        if data2_specific is not None:
            data_specific = np.vstack((data_specific, data2_specific))
            flag = 1
    else:
        if data2_specific is not None:
            data_specific = data2_specific

    if data_specific is None:
        return score
    else:
        bar_specific1 = 0
        bar_specific2 = 0
        data = np.vstack((data, data_specific))
        if flag == 0:  # only one of data1_specific and data2_specific is not None
            for i in range(len(data_specific)):
                diffMat = data_specific[i] - data
                sqDiffMat = diffMat ** 2
                sqDistances = sqDiffMat.sum(axis=1)
                NearestN = np.argsort(sqDistances)[1:k + 1]
                for j in NearestN:
                    if j > (len(data1) + len(data2)):
                        bar_specific1 += 1
            bar_specific = bar_specific1

        else:  # both data1_specific and data2_specific are not None
            for i in range(len(data1_specific)):
                diffMat = data1_specific[i] - data
                sqDiffMat = diffMat ** 2
                sqDistances = sqDiffMat.sum(axis=1)
                NearestN = np.argsort(sqDistances)[1:k + 1]
                for j in NearestN:
                    if j > (len(data1) + len(data2)) and j < (len(data1) + len(data2) + len(data1_specific)):
                        bar_specific1 += 1

            for i in range(len(data2_specific)):
                diffMat = data2_specific[i] - data
                sqDiffMat = diffMat ** 2
                sqDistances = sqDiffMat.sum(axis=1)
                NearestN = np.argsort(sqDistances)[1:k + 1]
                for j in NearestN:
                    if j > (len(data1) + len(data2) + len(data1_specific)):
                        bar_specific2 += 1

            bar_specific = bar_specific1 + bar_specific2

        bar_specific = bar_specific / len(data_specific)

        score += (bar_specific - k / N) / (k - k / N)

        return score / 2


def calc_frac_idx(x1_mat, x2_mat):
    """
    Returns fraction closer than true match for each sample (as an array)
    """
    fracs = []
    x = []
    nsamp = x1_mat.shape[0] if (x1_mat.shape[0] == x2_mat.shape[0]) else (min(x1_mat.shape[0], x2_mat.shape[0]))
    rank = 0
    for row_idx in range(nsamp):
        euc_dist = np.sqrt(np.sum(np.square(np.subtract(x1_mat[row_idx, :], x2_mat)), axis=1))
        true_nbr = euc_dist[row_idx]
        sort_euc_dist = sorted(euc_dist)
        rank = sort_euc_dist.index(true_nbr)
        frac = float(rank) / (nsamp - 1)

        fracs.append(frac)
        x.append(row_idx + 1)

    return fracs, x


def calc_domainAveraged_FOSCTTM(x1_mat, x2_mat):
    """
    Outputs average FOSCTTM measure (averaged over both domains)
    Get the fraction matched for all data points in both directions
    Averages the fractions in both directions for each data point
    """
    fracs1, xs = calc_frac_idx(x1_mat, x2_mat)
    fracs2, xs = calc_frac_idx(x2_mat, x1_mat)
    fracs = []
    for i in range(len(fracs1)):
        fracs.append((fracs1[i] + fracs2[i]) / 2)
    return fracs


def calc_sil(x1_mat, x2_mat, x1_lab, x2_lab):
    """
    Returns silhouette score for datasets with cell clusters
    """
    sil = []
    sil_d0 = []
    sil_d3 = []
    sil_d7 = []
    sil_d11 = []
    sil_npc = []

    x = np.concatenate((x1_mat, x2_mat))
    lab = np.concatenate((x1_lab, x2_lab))

    sil_score = silhouette_samples(x, lab)

    nsamp = x.shape[0]
    for i in range(nsamp):
        if (lab[i] == 1):
            sil_d0.append(sil_score[i])
        elif (lab[i] == 2):
            sil_d3.append(sil_score[i])
        elif (lab[i] == 3):
            sil_d7.append(sil_score[i])
        elif (lab[i] == 4):
            sil_d11.append(sil_score[i])
        elif (lab[i] == 5):
            sil_npc.append(sil_score[i])

    avg = np.mean(sil_score)
    d0 = sum(sil_d0) / len(sil_d0)
    d3 = sum(sil_d3) / len(sil_d3)
    d7 = sum(sil_d7) / len(sil_d7)
    d11 = sum(sil_d11) / len(sil_d11)
    npc = sum(sil_npc) / len(sil_npc)

    return avg, d0, d3, d7, d11, npc


def binarize_labels(label, x):
    """
    Helper function for calc_auc
    """
    bin_lab = np.array([1] * len(x))
    idx = np.where(x == label)

    bin_lab[idx] = 0
    return bin_lab


def calc_auc(x1_mat, x2_mat, x1_lab, x2_lab):
    """
    calculate avg. ROC AUC scores for transformed data when there are >=2 number of clusters.
    """
    nsamp = x1_mat.shape[0]

    auc = []
    auc_d0 = []
    auc_d3 = []
    auc_d7 = []
    auc_d11 = []
    auc_npc = []

    for row_idx in range(nsamp):
        euc_dist = np.sqrt(np.sum(np.square(np.subtract(x1_mat[row_idx, :], x2_mat)), axis=1))
        y_scores = euc_dist
        y_true = binarize_labels(x1_lab[row_idx], x2_lab)

        auc_score = roc_auc_score(y_true, y_scores)
        auc.append(auc_score)

        if (x1_lab[row_idx] == 0):
            auc_d0.append(auc_score)
        elif (x1_lab[row_idx] == 1):
            auc_d3.append(auc_score)
        elif (x1_lab[row_idx] == 2):
            auc_d7.append(auc_score)
        elif (x1_lab[row_idx] == 3):
            auc_d11.append(auc_score)
        elif (x1_lab[row_idx] == 4):
            auc_npc.append(auc_score)

    avg = sum(auc) / len(auc)
    d0 = sum(auc_d0) / len(auc_d0)
    d3 = sum(auc_d3) / len(auc_d3)
    d7 = sum(auc_d7) / len(auc_d7)
    d11 = sum(auc_d11) / len(auc_d11)
    npc = sum(auc_npc) / len(auc_npc)

    return avg, d0, d3, d7, d11, npc


def transfer_accuracy(domain1, domain2, type1, type2, n):
    """
    Metric from UnionCom: "Label Transfer Accuracy"
    """
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(domain2, type2)
    type1_predict = knn.predict(domain1)
    np.savetxt("type1_predict.txt", type1_predict)
    count = 0
    for label1, label2 in zip(type1_predict, type1):
        if label1 == label2:
            count += 1
    return count / len(type1)
