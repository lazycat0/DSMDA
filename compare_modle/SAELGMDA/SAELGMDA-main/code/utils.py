import numpy.matlib
import numpy as np
import sklearn.metrics
import csv
from model import *
from utils import *
from sklearn.metrics import *
import math
import xlsxwriter as xw
import os
import dataset_fn
import pickle
import matplotlib
import matplotlib.pyplot as plt


def prepare_data(dataset_name):
    print("loading data")
    a, SD, SM, disease_info, microbe_info = dataset_fn.get_load_fn(dataset_name)
    Nd = SD.shape[0]
    Nm = SM.shape[0]
    A = a.transpose()

    A_ = A
    KD = getSimilarMatrix(A_.transpose(), 1)
    KM = getSimilarMatrix(A_, 1)
    for i in range(Nd):
        for j in range(Nd):
            if SD[i][j] == 0:
                SD[i][j] = KD[i][j]
            else:
                SD[i][j] = (SD[i][j] + KD[i][j]) / 2

    for i in range(Nm):
        for j in range(Nm):
            if SM[i][j] == 0:
                SM[i][j] = KM[i][j]
            else:
                SM[i][j] = (SM[i][j] + KM[i][j]) / 2

    sm = np.repeat(SM, repeats=Nd, axis=0)  # (11388,292)

    sd = np.matlib.repmat(SD, Nm, 1)  # (11388,39)
    train = np.concatenate((sm, sd), axis=1)  # (11388,78)
    label = A.reshape((-1, 1))

    return train, label


def Cross_Validation(data, label,dataset_name, k=5, cv=3):
    all_performance = []

    train_index, test_index = kfold(data=data, k=k, row=292, col=39, cv=cv)
    FPR = []
    TPR = []

    PRE = []
    REC = []

    fold_metrics_file = 'fold_metrics.csv'
    with open(fold_metrics_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Fold', 'AUC', 'AUPR', 'F1', 'Accuracy'])

    for i in range(k):
        train = data[train_index[i],]
        train_label = label[train_index[i]]
        test = data[test_index[i],]

        test_pred = classifier(train, train_label, test)
        test_label = transfer_label_from_prob(test_pred, 0.5)

        real_labels = label[test_index[i]]
        acc, precision, sensitivity, specificity, MCC, f1_score = calculate_performace(len(real_labels), test_label,
                                                                                       real_labels)


        fpr, tpr, auc_thresholds = roc_curve(real_labels, test_pred)
        auc_score = sklearn.metrics.auc(fpr, tpr)

        precision1, recall1, pr_threshods = precision_recall_curve(real_labels, test_pred)
        aupr_score = sklearn.metrics.auc(recall1, precision1)

        FPR.append(fpr)
        TPR.append(tpr)
        PRE.append(precision1)
        REC.append(recall1)

        all_performance.append([acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score, f1_score])
        method = "mine_%s" % dataset_name
        path = "../test_result/cv%d/%s/" % (cv, method)
        if not os.path.exists(path):
            os.makedirs(path)
        auc_path = path + "AUC_%s_cv%d_%.3f.xlsx" % (method, cv, auc_score)
        xw_toExcel(fpr, tpr, auc_path, sheet=cv)
        aupr_path = path + "AUPR_%s_cv%d_%.3f.xlsx" % (method, cv, aupr_score)
        xw_toExcel(recall1, precision1, aupr_path, sheet=cv)


        with open(fold_metrics_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i + 1, auc_score, aupr_score, f1_score,
                             acc])

    print('---' * 20)

    tprs = []
    mean_fpr = np.linspace(0, 1, 1000)
    for i in range(len(FPR)):
        tprs.append(np.interp(mean_fpr, FPR[i], TPR[i]))
        tprs[-1][0] = 0.0

    # Save FPR, TPR, and tprs
    with open('roc_data.pkl', 'wb') as f:
        pickle.dump({'FPR': FPR, 'TPR': TPR, 'tprs': tprs, 'mean_fpr': mean_fpr}, f)
    # Save REC, PRE
    with open('pr_data.pkl', 'wb') as f:
        pickle.dump({'REC': REC, 'PRE': PRE}, f)
    with open('roc_data.pkl', 'rb') as f:
        roc_data = pickle.load(f)


    tprs = roc_data['tprs']
    mean_fpr = roc_data['mean_fpr']

    # Plot Mean ROC curve only
    plt.figure(figsize=(10, 8))
    plt.plot(mean_fpr, np.mean(tprs, axis=0), color="#D81C38", lw=2,
             label='Mean ROC (auc=%0.4f)' % auc(mean_fpr, np.mean(tprs, axis=0)))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Average Receiver Operating Characteristic Curve")
    plt.legend(loc='best')
    plt.show()

    with open('pr_data.pkl', 'rb') as f:
        pr_data = pickle.load(f)

    REC = pr_data['REC']
    PRE = pr_data['PRE']

    # Interpolate precision-recall curves
    prs = []
    mean_recall = np.linspace(0, 1, 1000)
    for i in range(len(REC)):
        prs.append(np.interp(mean_recall, REC[i][::-1], PRE[i][::-1])[::-1])

    # Plot Mean PR curve
    plt.figure(figsize=(10, 8))
    mean_precision = np.mean(prs, axis=0)
    mean_aupr = auc(mean_recall, mean_precision)
    plt.plot(mean_recall, mean_precision, label='Mean PR (AUPR=%0.4f)' % mean_aupr, lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Average Precision-Recall Curve")
    plt.legend(loc='best')
    plt.show()




    return all_performance


def xw_toExcel(x, y, fileName, sheet):  # xlsxwriter库储存数据到excel
    workbook = xw.Workbook(fileName)  # 创建工作簿
    worksheet1 = workbook.add_worksheet("sheet_cv%d" % sheet)  # 创建子表
    worksheet1.activate()
    for i in range(len(x)):
        insertData = [x[i], y[i]]
        row = 'A' + str(i+1)
        worksheet1.write_row(row, insertData)
    workbook.close()




def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = preprocessing.LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = utils.np_utils.to_categorical(y)
    return y, encoder


def calculate_performace(test_num, pred_y, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1
    acc = float(tp + tn) / test_num

    if tp == 0 and fp == 0:
        precision = 0
        MCC = 0
        f1_score = 0
        sensitivity = float(tp) / (tp + fn)
        specificity = float(tn) / (tn + fp)
    else:
        precision = float(tp) / (tp + fp)
        sensitivity = float(tp) / (tp + fn)
        specificity = float(tn) / (tn + fp)
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        f1_score = float(2 * tp) / ((2 * tp) + fp + fn)
    return acc, precision, sensitivity, specificity, MCC, f1_score


def transfer_label_from_prob(proba, threshold):
    proba = (proba - proba.min()) / (
            proba.max() - proba.min())
    label = [1 if val >= threshold else 0 for val in proba]
    return label


def getSimilarMatrix(IP, γ_):
    dimensional = IP.shape[0]
    sd = np.zeros(dimensional)
    K = np.zeros((dimensional, dimensional))
    for i in range(dimensional):
        sd[i] = np.linalg.norm(IP[i]) ** 2
    gamad = γ_ * dimensional / np.sum(sd.transpose())
    for i in range(dimensional):
        for j in range(dimensional):
            K[i][j] = math.exp(-gamad * (np.linalg.norm(IP[i] - IP[j])) ** 2)
    return K


def kfold(data, k, row=0, col=0, cv=3):
    dlen = len(data)
    if cv == 2:
        lens = row
        split = col
    elif cv == 1:
        lens = col
        split = row
    else:
        lens = dlen
    d = list(range(lens))
    np.random.shuffle(d)
    test_n = lens // k
    n = lens % k
    test_res = []
    train_res = []
    for i in range(n):
        test = d[i * (test_n + 1):(i + 1) * (test_n + 1)]
        train = list(set(d) - set(test))
        test_res.append(test)
        train_res.append(train)
    for i in range(n, k):
        test = d[i * test_n + n:(i + 1) * test_n + n]
        train = list(set(d) - set(test))
        test_res.append(test)
        train_res.append(train)
    if cv == 3:
        return train_res, test_res

    train_s = []
    test_s = []
    d = range(dlen)
    for k in range(len(test_res)):
        test = []
        for i in range(len(test_res[k])):
            if cv == 2:
                tmp = np.full(split, test_res[k][i] * split) + range(split)
            elif cv == 1:
                tmp = np.full(split, test_res[k][i]) + [i * lens for i in range(split)]
            test = np.concatenate((test, tmp), axis=0)
        test = test.astype(int)
        test_s.append(test)
        train = np.array(list(set(d) - set(test)), dtype=int)
        train_s.append(train)
    return train_s, test_s
