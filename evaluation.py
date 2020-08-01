import numpy as np
import scipy.io as scio
from sklearn.metrics import precision_recall_curve
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from utils import *


def hamming_precision(params):
    database_code = np.array(params['database_code'])
    validation_code = np.array(params['validation_code'])
    database_labels = np.array(params['database_labels'])
    validation_labels = np.array(params['validation_labels'])
    R = params['R']
    query_num = validation_code.shape[0]
    database_num = database_code.shape[0]

    database_code = np.sign(database_code)
    database_code[database_code == -1] = 0
    database_code = database_code.astype(int)

    validation_code = np.sign(validation_code)
    validation_code[validation_code == -1] = 0
    validation_code = validation_code.astype(int)

    APx = []

    for i in range(query_num):
        query = validation_code[i, :]
        query_matrix = np.tile(query, (database_num, 1))

        label = validation_labels[i, :]
        label[label == 0] = -1
        label_matrix = np.tile(label, (database_num, 1))

        distance = np.sum(np.absolute(query_matrix - database_code), axis=1)
        similarity = np.sum(database_labels == label_matrix, axis=1)
        similarity[similarity > 1] = 1

        total_rel_num = np.sum(distance <= R)
        true_positive = np.sum((distance <= R) * similarity)

        # print('--------')
        # print(i)
        # print(true_positive)
        # print(total_rel_num)
        # print('--------')
        if total_rel_num != 0:
            APx.append(float(true_positive) / total_rel_num)
        else:
            APx.append(float(0))

    # print(np.sum(np.array(APx) != 0))
    return np.mean(np.array(APx))


def precision_curve(params):
    database_code = np.array(params['database_code'])
    validation_code = np.array(params['validation_code'])
    database_labels = np.array(params['database_labels'])
    validation_labels = np.array(params['validation_labels'])
    query_num = validation_code.shape[0]
    database_code = np.sign(database_code)
    validation_code = np.sign(validation_code)

    sim = np.dot(database_code, validation_code.T)
    ids = np.argsort(-sim, axis=0)
    arr = []

    for iter in range(10):
        R = (iter + 1) * params["step"]
        APx = []
        for i in range(query_num):
            label = validation_labels[i, :]
            label[label == 0] = -1
            idx = ids[:, i]
            imatch = np.sum(database_labels[idx[0:R], :] == label, axis=1) > 0
            relevant_num = np.sum(imatch)
            APx.append(float(relevant_num) / R)
        arr.append([R, np.mean(np.array(APx))])
    return np.array(arr)


def precision(params):
    database_code = np.array(params['database_code'])
    validation_code = np.array(params['validation_code'])
    database_labels = np.array(params['database_labels'])
    validation_labels = np.array(params['validation_labels'])
    R = params['R']
    query_num = validation_code.shape[0]
    database_code = np.sign(database_code)
    validation_code = np.sign(validation_code)

    sim = np.dot(database_code, validation_code.T)
    ids = np.argsort(-sim, axis=0)
    APx = []

    for i in range(query_num):
        label = validation_labels[i, :]
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_labels[idx[0:R], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)
        APx.append(float(relevant_num) / R)

    return np.mean(np.array(APx))


def mean_average_precision(params):
    database_code = np.array(params['database_code'])
    validation_code = np.array(params['validation_code'])
    database_labels = np.array(params['database_labels'])
    validation_labels = np.array(params['validation_labels'])
    R = params['R']
    query_num = validation_code.shape[0]
    database_code = np.sign(database_code)
    validation_code = np.sign(validation_code)

    sim = np.dot(database_code, validation_code.T)
    ids = np.argsort(-sim, axis=0)
    APx = []

    for i in range(query_num):
        label = validation_labels[i, :]
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_labels[idx[0:R], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R+1, 1)
        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)

    return np.mean(np.array(APx))


def statistic_prob(params):
    database_code = np.array(params['database_code'])
    validation_code = np.array(params['validation_code'])
    sim = np.dot(database_code, validation_code.T)
    query_num = validation_code.shape[0]
    database_num = database_code.shape[0]
    ones = np.ones((database_num, query_num))
    exp_sim = np.exp(sim)
    prob = ones / (1 + 1 / exp_sim)
    useless = np.sum(prob >= 0.95) + np.sum(prob <= 0.05)
    useful = query_num * database_num - useless
    print("useful")
    print(useful)
    print("useless")
    print(useless)


def pr_curve(params):
    qF = params['validation_code']
    rF = params['database_code']
    qL = params['validation_labels']
    rL = params['database_labels']
    topK = params['R']

    # print(np.shape(qF), np.shape(rF))
    qF, rF, qL, rL = np.array(qF), np.array(rF), np.array(qL), np.array(rL)
    n_query = qF.shape[0]
    if topK == -1 or topK > rF.shape[0]:  # top-K 之 K 的上限
        topK = rF.shape[0]

    Gnd = (np.dot(qL, rL.transpose()) > 0).astype(np.float32)

    # print(qF.shape, rF.shape)
    Rank = np.argsort(cdist(qF, rF, 'hamming'))

    P, R = [], []
    for k in range(1, topK + 1):  # 枚举 top-K 之 K
        # ground-truth: 1 vs all
        p = np.zeros(n_query)  # 各 query sample 的 Precision@R
        r = np.zeros(n_query)  # 各 query sample 的 Recall@R
        for it in range(n_query):  # 枚举 query sample
            gnd = Gnd[it]
            gnd_all = np.sum(gnd)  # 整个被检索数据库中的相关样本数
            if gnd_all == 0:
                continue
            asc_id = Rank[it][:k]

            gnd = gnd[asc_id]
            gnd_r = np.sum(gnd)  # top-K 中的相关样本数

            p[it] = gnd_r / k
            r[it] = gnd_r / gnd_all

        P.append(np.mean(p))
        R.append(np.mean(r))
    # print(P)
    # print(R)
    arr = []
    for p, r in zip(P, R):
        arr.append([p, r])
    return np.array(arr)


def gen_params(database_hash, database_labels, clean_query_hash, query_labels, R=-1):
    params = {"validation_labels": query_labels, "validation_code": clean_query_hash,
              "database_labels": database_labels, "database_code": database_hash,
              "R": R, "step": 100}

    return params
