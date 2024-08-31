import numpy as np
import torch
import scipy.sparse as sp
import pandas as pd
import math
import random
from sklearn.preprocessing import minmax_scale, scale
import matplotlib.pyplot as plt
from sklearn import metrics
from itertools import cycle
import heapq
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F


def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def set_seed_2(seed=123):
    random.seed(seed)  # Python内置随机库
    np.random.seed(seed)  # NumPy随机库
    torch.manual_seed(seed)  # PyTorch随机库
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch CUDA随机库
        torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
        torch.backends.cudnn.deterministic = True  # 确定性算法，可能会影响性能
        torch.backends.cudnn.benchmark = False



def calculate_combined_similarity(disease_matrices, microbe_matrices):
    """
    计算疾病和微生物的综合相似性矩阵。

    参数：
    - disease_matrices：包含5个疾病相似性矩阵的列表。
    - microbe_matrices：包含5个微生物相似性矩阵的列表。

    返回值：
    - sim_disease：疾病的综合相似性矩阵。
    - sim_microbe：微生物的综合相似性矩阵。
    """

    # 计算疾病的综合相似性矩阵
    sum_disease_similarity = np.zeros_like(disease_matrices[0], dtype=np.float64)
    count_disease_nonzero = np.zeros_like(disease_matrices[0])

    for sim_matrix in disease_matrices:
        sum_disease_similarity += sim_matrix.astype(float)
        count_disease_nonzero += (sim_matrix != 0).astype(int)

    count_disease_nonzero[count_disease_nonzero == 0] = 1
    sim_disease = sum_disease_similarity / count_disease_nonzero

    # 计算微生物的综合相似性矩阵
    sum_microbe_similarity = np.zeros_like(microbe_matrices[0])
    count_microbe_nonzero = np.zeros_like(microbe_matrices[0])

    for sim_matrix in microbe_matrices:
        sum_microbe_similarity += sim_matrix.astype(float)
        count_microbe_nonzero += (sim_matrix != 0).astype(int)

    count_microbe_nonzero[count_microbe_nonzero == 0] = 1
    sim_microbe = sum_microbe_similarity / count_microbe_nonzero

    return sim_disease, sim_microbe



def get_all_the_samples(A, deep_A, h , top , kbin):
    m, n = A.shape
    pos = []
    neg = []
    for i in range(m):
        for j in range(n):
            if A[i, j] == 1:
                pos.append([i, j, 1, deep_A[i, j]])
            else:
                neg.append([i, j, 0, deep_A[i, j]])
    n = len(pos)
    k = h * n
    neg_smallest = heapq.nsmallest(k, neg, key=lambda x: x[-1])

    neg_new = random.sample(neg_smallest, n)  # 从neg中随机抽取n个样本，使得正负样本数量相等，存入neg_new中
    tep_samples = pos + neg_new
    tep_samples = np.array(tep_samples)
    tep_samples = get_weight_modified(tep_samples, top, kbin)  # 计算的新的权重为（1~top 分为 kbin 个箱子）

    np.random.shuffle(tep_samples)  # 直接在原数组上打乱顺序

    weight = tep_samples[:, -1]  # 提取权重
    samples = tep_samples[:, :3].astype(int)  # 提取样本信息并转换为整型
    return samples, weight


def get_all_pairs(A_in, deep_A):
    A = A_in.copy()
    m, n = A.shape
    pairs = []
    for i in range(m):
        for j in range(n):
            if A[i, j] == 1:
                j_hat = np.argmin(np.where(A[i] == 0, deep_A[i], np.inf))
                # 将找到的位置在A中置为-1
                if j_hat < n:  # 确保找到的索引在范围内
                    A[i, j_hat] = -1

                # 在deep_A的第j_hat列中找到最小值且A中对应位置为0的元素的行索引i_hat
                #i_hat = np.argmin(np.where(A[:, j_hat] == 0, deep_A[:, j_hat], np.inf))
                i_hat = np.argmin(np.where(A[:, j] == 0, deep_A[:, j], np.inf))
                # 将找到的位置在A中置为-1
                if i_hat < m:  # 确保找到的索引在范围内
                    #A[i_hat, j] = -1
                    pass

                pairs.append([i, j, i_hat, j_hat])
    return pairs

def get_all_pairs_random(A_in):

    A = A_in.copy()
    m, n = A.shape
    pairs = []
    deep_A = A_in
    for i in range(m):
        for j in range(n):
            if A[i, j] == 1:
                j_hat = np.argmin(np.where(A[i] == 0, deep_A[i], np.inf))
                # 将找到的位置在A中置为-1
                if j_hat < n:  # 确保找到的索引在范围内
                    A[i, j_hat] = -1

                # 在deep_A的第j_hat列中找到最小值且A中对应位置为0的元素的行索引i_hat
                #i_hat = np.argmin(np.where(A[:, j_hat] == 0, deep_A[:, j_hat], np.inf))
                i_hat = np.argmin(np.where(A[:, j] == 0, deep_A[:, j], np.inf))
                # 将找到的位置在A中置为-1
                if i_hat < m:  # 确保找到的索引在范围内
                    A[i_hat, j] = -1
                pairs.append([i, j, i_hat, j_hat])
    return pairs




def get_all_pairs_v2(A_in, deep_A, K=25):
    A = A_in.copy()
    m, n = A.shape
    pairs = []
    for i in range(m):
        for j in range(n):
            if A[i, j] == 1:
                # 处理行
                zero_indices_row = np.where(A[i] == 0)[0]
                if len(zero_indices_row) > 0:
                    values_row = deep_A[i, zero_indices_row]
                    # 计算分箱边界
                    quantiles_row = np.quantile(values_row, np.linspace(0, 1, K+1))
                    # 分箱
                    bins_row = np.digitize(values_row, quantiles_row) - 1
                    # 选择倒数第二个箱子中的索引
                    if len(np.where(bins_row == K-2)[0]) > 0:
                        selected_row = zero_indices_row[np.random.choice(np.where(bins_row == K-2)[0])]
                        #A[i, selected_row] = -1
                    else:
                        selected_row = None  # 没有倒数第二个箱子时的处理

                # 处理列
                zero_indices_col = np.where(A[:, j] == 0)[0]
                if len(zero_indices_col) > 0:
                    values_col = deep_A[zero_indices_col, j]
                    # 计算分箱边界
                    quantiles_col = np.quantile(values_col, np.linspace(0, 1, K+1))
                    # 分箱
                    bins_col = np.digitize(values_col, quantiles_col) - 1
                    # 选择倒数第二个箱子中的索引
                    if len(np.where(bins_col == K-2)[0]) > 0:
                        selected_col = zero_indices_col[np.random.choice(np.where(bins_col == K-2)[0])]
                        #A[selected_col, j] = -1
                    else:
                        selected_col = None  # 没有倒数第二个箱子时的处理

                pairs.append([i, j, selected_col, selected_row])
    return pairs

def get_all_the_samples_old(A):
    m, n = A.shape
    pos = []
    neg = []
    for i in range(m):
        for j in range(n):
            if A[i, j] == 1:
                pos.append([i, j, 1])
            else:
                neg.append([i, j, 0 ])
    n = len(pos)

    neg_new = random.sample(neg, 2*n)                                     # 从neg中随机抽取n个样本，使得正负样本数量相等，存入neg_new中
    tep_samples = pos + neg_new
    samples = random.sample(tep_samples, len(tep_samples))              # 从tep_samples中随机抽取所有样本，打乱顺序
    samples = random.sample(samples, len(samples))
    samples = np.array(samples)
    return samples



def matrixPow(Matrix, n):                           # 计算矩阵的n次幂
    """
    计算矩阵的n次幂。

    参数:
    - Matrix: 输入矩阵
    - n: 幂次

    返回:
    - 矩阵的n次幂
    """
    if(type(Matrix) == list):
        Matrix = np.array(Matrix)
    if(n == 1):
        return Matrix
    else:
        return np.matmul(Matrix, matrixPow(Matrix, n - 1))

def calculate_metapath_optimized(mm, dd, md, n):        #计算n层元路径
    """
    优化版：计算微生物-疾病第n层元路径。

    参数:
    - mm: 微生物相似度矩阵
    - dd: 疾病相似度矩阵
    - md: 微生物疾病关联矩阵
    - n: 元路径的层数

    返回:
    - n层元路径矩阵
    """
    # 基本情况，如果n为1，直接计算并返回第一层元路径矩阵
    dm = md.T
    MM = md @ dd @ dm @ mm
    MD = md @ dd
    if n == 1:
        return mm @ md @ dd
    else:
        #k = n / 2
        k = n
        k = int(k)
        MK = matrixPow(MM, k)
        deep_A = mm @ MK @ MD

        #if n % 2 ==0:
        #    deep_A = mm @ MK @ MD
        #else:
        #    deep_A = mm @ MK

    return deep_A

def get_metrics(real_score, predict_score):
    real_score = np.array(real_score)
    predict_score = np.array(predict_score)

    # 使用分位数作为阈值
    thresholds = np.percentile(predict_score, np.arange(1, 100, 0.1))

    # 初始化
    TP = np.zeros_like(thresholds)
    FP = np.zeros_like(thresholds)
    FN = np.zeros_like(thresholds)
    TN = np.zeros_like(thresholds)

    # 计算TP, FP, FN, TN
    for i, threshold in enumerate(thresholds):
        predicted_positive = predict_score >= threshold
        predicted_negative = predict_score < threshold

        TP[i] = np.sum((real_score == 1) & predicted_positive)
        FP[i] = np.sum((real_score == 0) & predicted_positive)
        TN[i] = np.sum((real_score == 0) & predicted_negative)
        FN[i] = np.sum((real_score == 1) & predicted_negative)
    threshold_indices = np.where(thresholds >= 0.5)[0]
    if len(threshold_indices) > 0:
        threshold_05_index = threshold_indices[0]  # 使用第一个大于等于0.5的阈值
        accuracy_05 = (TP[threshold_05_index] + TN[threshold_05_index]) / (
                    TP[threshold_05_index] + FP[threshold_05_index] + TN[threshold_05_index] + FN[threshold_05_index])
    else:

        accuracy_05 = (TP[98] + TN[98]) / (
                TP[98] + FP[98] + TN[98] + FN[98])

    # 确保fpr和recall是递增的
    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    # 确保fpr和recall是递增的
    fpr, tpr = np.array(sorted(zip(fpr, tpr))).T
    recall, precision = np.array(sorted(zip(recall, precision))).T

    # 计算AUC
    auc = np.trapz(tpr, fpr)

    # 计算AUPR
    aupr = np.trapz(precision, recall)

    # 计算最佳阈值下的指标
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)  # 添加小的正则项避免分母为零
    best_threshold_index = np.argmax(f1_scores)
    best_f1_score = f1_scores[best_threshold_index]
    best_accuracy = (TP[best_threshold_index] + TN[best_threshold_index]) / (TP[best_threshold_index] + FP[best_threshold_index] + TN[best_threshold_index] + FN[best_threshold_index])
    best_specificity = TN[best_threshold_index] / (TN[best_threshold_index] + FP[best_threshold_index])
    best_recall = recall[best_threshold_index]
    best_precision = precision[best_threshold_index]

    return [aupr, auc, best_f1_score, best_accuracy, best_recall, best_specificity,best_precision,accuracy_05],precision,recall,tpr,fpr


def contrastive_loss(mic_latent_tensor, dis_latent_tensor, train_samples, margin=0.5):   # 自定义的对比损失函数
    # 初始化损失值
    loss = 0.0

    # 遍历所有样本
    for sample in train_samples:
        # 获取对应的潜在特征向量
        i, j, i_hat, j_hat = map(int, sample)
        mic_i = mic_latent_tensor[i]
        dis_j = dis_latent_tensor[j]
        mic_i_hat = mic_latent_tensor[i_hat]
        dis_j_hat = dis_latent_tensor[j_hat]

        # 计算正相关对的余弦相似度
        pos_sim = cosine_similarity(mic_i.unsqueeze(0), dis_j.unsqueeze(0))

        # 计算负相关对的余弦相似度
        neg_sim_i_j_hat = cosine_similarity(mic_i.unsqueeze(0), dis_j_hat.unsqueeze(0))
        neg_sim_i_hat_j = cosine_similarity(mic_i_hat.unsqueeze(0), dis_j.unsqueeze(0))

        # 对比损失：最大化正相关对与负相关对之间的相似度差异
        loss += torch.relu(margin - pos_sim + neg_sim_i_j_hat) + torch.relu(margin - pos_sim + neg_sim_i_hat_j)

    # 返回平均损失
    return loss / len(train_samples)


def contrastive_loss_v2(mic_latent_tensor, dis_latent_tensor, train_samples, tau=0.5):
    loss = 0.0

    # 定义相似度计算函数
    def sim(z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        return torch.mm(z1, z2.t())

    # 遍历所有样本
    for sample in train_samples:
        i, j, i_hat, j_hat = map(int, sample)
        mic_i = mic_latent_tensor[i].unsqueeze(0)
        dis_j = dis_latent_tensor[j].unsqueeze(0)
        mic_i_hat = mic_latent_tensor[i_hat].unsqueeze(0)
        dis_j_hat = dis_latent_tensor[j_hat].unsqueeze(0)

        # 计算正相关对和负相关对的余弦相似度
        pos_sim = torch.exp(sim(mic_i, dis_j) / tau)
        neg_sim_i_j_hat = torch.exp(sim(mic_i, dis_j_hat) / tau)
        neg_sim_i_hat_j = torch.exp(sim(mic_i_hat, dis_j) / tau)

        # 计算当前样本的损失
        sample_loss = (pos_sim / (pos_sim + neg_sim_i_j_hat + neg_sim_i_hat_j))
        loss += sample_loss

    # 返回平均损失
    return loss / len(train_samples)