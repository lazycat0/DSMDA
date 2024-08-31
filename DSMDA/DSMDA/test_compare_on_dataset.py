import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn import metrics
from utils import *
from sklearn.model_selection import KFold
from utils import *
import random
import matplotlib
import torch.optim as optim
from model3 import MicroDiseaseModel_v3,MicroDiseaseModel_v3_No_VAE
from get_sim import *
import torch.nn.functional as F
from sklearn.metrics import auc
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, average_precision_score,
                             f1_score, accuracy_score, recall_score, precision_score, confusion_matrix)
from scipy import interp

matplotlib.use('TkAgg')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',  # 使用颜色编码定义颜色
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# paprameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#input_dim = 1396+43  # phendb  1396+43 peryton    1396+43     disbiome 1622+374 疾病特征维度     HMDAD 292+39
en_hidden_dim = 64    # 隐藏层维度
hidden_dims = [512, 256, 64]
output_dim = 32    # 低维输出维度
epochs = 1000
k_split = 5
P = 0.3   # 调整两项损失之前的权重


sk_tprs_HMDAD = []
sk_aucs_HMDAD = []
sk_precisions_HMDAD = []
sk_recalls_HMDAD = []
sk_average_precisions_HMDAD = []
sk_fpr_HMDAD = []


sk_tprs_Disbiome = []
sk_aucs_Disbiome = []
sk_precisions_Disbiome = []
sk_recalls_Disbiome = []
sk_average_precisions_Disbiome = []
sk_fpr_Disbiome = []


sk_tprs_peryton = []
sk_aucs_peryton = []
sk_precisions_peryton = []
sk_recalls_peryton = []
sk_average_precisions_peryton = []
sk_fpr_peryton = []

#  -------------------------------------------------------------------HMDAD----------------------------------------------------------------------

#HMDAD 94 97

A = pd.read_excel('./优化数据集/HMDAD/adj_mat.xlsx')

disease_Semantics = pd.read_csv('./优化数据集/HMDAD/疾病-语义/similarity_matrix_model2.csv', header=None)
disease_gip = pd.read_csv('./优化数据集/HMDAD/基于关联矩阵的疾病相似性/GIP_Sim.csv', header=None)
disease_symptoms = pd.read_csv('./优化数据集/HMDAD/疾病-症状/complete_disease_similarity_matrix.csv', header=None)

micro_gip = pd.read_csv('./优化数据集/HMDAD/基于关联矩阵的微生物功能/GIP_Sim.csv', header=None)
micro_sem = pd.read_csv('./优化数据集/HMDAD/基于疾病语义的微生物功能/functional_similarity2_matrix.csv')
micro_fun2 = pd.read_csv('./优化数据集/HMDAD/微生物-功能/complete_microbe_similarities_ds2_matrix.csv', header=None)


A = A.iloc[1:, 1:]
disease_gip=disease_gip.iloc[1:, 1:]
micro_gip=micro_gip.iloc[1:, 1:]
micro_fun2=micro_fun2.iloc[1:, 1:]
disease_symptoms=disease_symptoms.iloc[1:, 1:]

microbiome_matrices = [micro_gip.values, micro_sem.values, micro_fun2.values]
disease_matrices = [disease_Semantics.values, disease_gip.values, disease_symptoms.values]
sim_d, sim_m = calculate_combined_similarity(disease_matrices, microbiome_matrices)
input_dim = sim_m.shape[0] + sim_d.shape[0]
A = A.T.to_numpy()

#A = A.astype(float).astype(int).to_numpy()
print("the number of miRNAs and diseases", A.shape)
print("the number of associations", sum(sum(A)))

x, y = A.shape
score_matrix = np.zeros([x, y])                                 # 初始化评分矩阵


md=A
mm=sim_m
dd=sim_d
n=4
deep_A=calculate_metapath_optimized(mm, dd, md, n)


print("the number of microbes and diseases", A.shape)
print("the number of associations", sum(sum(A)))
set_seed_2(123)
samples = get_all_pairs(A , deep_A )   # 返回[i, j, i_hat, j_hat] i 微生物 j疾病
samples = np.array(samples)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()

# cross validation
kf = KFold(n_splits=k_split, shuffle=True, random_state=123)                         # 定义10折交叉验证
iter_ = 0  # control each iterator  控制迭代次数
sum_score = 0               # 总分数初始化为0
out = []                    # 用于存储每一折的训练集和测试集索引
test_label_score = {}       # 存储测试标签和预测得分的字典
criterion = torch.nn.MSELoss()

prob_matrix_avg = np.zeros((A.shape[0], A.shape[1]))

lambda_mse = 4
lambda_l2 = 2e-3
lambda_constrate = 3


prob_matrix_avg = np.zeros((A.shape[0], A.shape[1]))
iter_ = 0
out = []                    # 用于存储每一折的训练集和测试集索引
test_label_score = {}       # 存储测试标签和预测得分的字典



for cl, (train_index, test_index) in enumerate(kf.split(samples)):               # 循环每一折的训练和测试集
    print('############ {} fold #############'.format(cl))                        # 打印当前折数
    out.append([train_index, test_index])                                        # 将训练和测试集索引存入列表中
    iter_ = iter_ + 1                                                           # 迭代次数加1

    train_samples = samples[train_index, :]                                     # 获取当前折的训练集样本
    test_samples = samples[test_index, :]                                       # 获取当前折的测试集样本
    mic_len = sim_m.shape[1]                                                # 计算微生物潜在表示的向量长度
    dis_len = sim_d.shape[1]                                                # 计算疾病潜在表示的向量长度

    train_n = train_samples.shape[0]                                            # 获取训练集样本数量
    test_N = test_samples.shape[0]                                              # 获取测试集样本数量

    mic_feature = np.zeros([mic_len, input_dim])
    dis_feature = np.zeros([dis_len, input_dim])
    mic_feature = np.concatenate( [sim_m , A] , axis=1)
    dis_feature = np.concatenate( [sim_d , A.T] , axis=1)
    train_i_mic_feature = np.zeros([train_n,input_dim])
    train_i_hat_mic_feature = np.zeros([train_n,input_dim])
    train_j_disease_feature = np.zeros([train_n, input_dim])
    train_j_hat_disease_feature = np.zeros([train_n, input_dim])

    test_i_mic_feature = np.zeros([test_N, input_dim])
    test_i_hat_mic_feature = np.zeros([test_N, input_dim])
    test_j_disease_feature = np.zeros([test_N, input_dim])
    test_j_hat_disease_feature = np.zeros([test_N, input_dim])

    model = MicroDiseaseModel_v3(mic_input_dim=input_dim, dis_input_dim=input_dim, latent_dim=output_dim)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001 , weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.8, last_epoch=-1)

    model.train()

    test_list=[]
    train_list=[]
    num = 0

    for sample in train_samples:                                    # epoch ==1000
        # 正相关[i, j]
        i, j, i_hat, j_hat = map(int, sample)
        train_list.append([i, j, 1])
        train_list.append([i, j, 1])
        train_list.append([i, j_hat, 0])
        train_list.append([i_hat, j, 0])

        train_i_mic_feature[num,:] = mic_feature[i,:]
        train_i_hat_mic_feature[num,:] = mic_feature[i_hat,:]
        train_j_disease_feature[num,:] = dis_feature[j,:]
        train_j_hat_disease_feature[num,:] = dis_feature[j_hat,:]
        num += 1
    num = 0
    for sample in test_samples:
        # 正相关[i, j]
        i, j, i_hat, j_hat = map(int, sample)
        test_list.append([i, j, 1])
        # 负相关[i, j_hat]
        test_list.append([i, j_hat, 0])
        test_list.append([i_hat, j, 0])

        test_i_mic_feature[num,:] = mic_feature[i,:]
        test_i_hat_mic_feature[num,:] = mic_feature[i_hat,:]
        test_j_disease_feature[num,:] = dis_feature[j,:]
        test_j_hat_disease_feature[num,:] = dis_feature[j_hat,:]
        num += 1

    train_list = np.array(train_list)
    test_list = np.array(test_list)
    train_list_tensor = torch.tensor(train_list, dtype=torch.float32).to(device)
    test_list_tensor = torch.tensor(test_list, dtype=torch.float32).to(device)

    # 将数据也移动到指定的设备
    train_samples_tensor = torch.tensor(train_samples, dtype=torch.float32).to(device)
    test_samples_tensor = torch.tensor(test_samples, dtype=torch.float32).to(device)
    mic_feature_tenor = torch.tensor(mic_feature, dtype=torch.float32).to(device)
    dis_feature_tensor = torch.tensor(dis_feature, dtype=torch.float32).to(device)

    train_i_mic_feature_tensor = torch.tensor(train_i_mic_feature, dtype=torch.float32).to(device)
    train_i_hat_mic_feature_tensor = torch.tensor(train_i_hat_mic_feature, dtype=torch.float32).to(device)
    train_j_disease_feature_tensor = torch.tensor(train_j_disease_feature, dtype=torch.float32).to(device)
    train_j_hat_disease_feature_tensor= torch.tensor(train_j_hat_disease_feature, dtype=torch.float32).to(device)
    test_i_mic_feature_tensor = torch.tensor(test_i_mic_feature, dtype=torch.float32).to(device)
    test_i_hat_mic_feature_tensor = torch.tensor(test_i_hat_mic_feature, dtype=torch.float32).to(device)
    test_j_disease_feature_tensor = torch.tensor(test_j_disease_feature, dtype=torch.float32).to(device)
    test_j_hat_disease_feature_tensor = torch.tensor(test_j_hat_disease_feature, dtype=torch.float32).to(device)
    A_tensor = torch.tensor(A, dtype=torch.float32).to(device)
    #-------------------------------------------------------   MY   VAE    -----------------------------------------------------------
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = 0
        prob_matrix, constrate_loss = model(mic_feature_tenor, dis_feature_tensor, train_i_mic_feature_tensor, train_i_hat_mic_feature_tensor,
                            train_j_disease_feature_tensor, train_j_hat_disease_feature_tensor )
        train_labels = train_list_tensor[:, 2]  # 实际标签
        indices = train_list_tensor[:, :2].long()  # 确保索引为整数类型
        train_label = prob_matrix[indices[:, 0], indices[:, 1]]  # 使用张量索引获取预测值

        loss_l2 = lambda_l2 * torch.norm(prob_matrix, p='fro')

        # 现在 train_label 和 train_labels 都是张量，并且可以在计算损失时保持梯度追踪
        #matrix_diff_loss = torch.mean((prob_matrix - A_tensor) ** 2)
        loss = lambda_mse * criterion(train_label,train_labels) + lambda_constrate * constrate_loss + loss_l2

        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        prob_matrix, _ = model(mic_feature_tenor, dis_feature_tensor, test_i_mic_feature_tensor,
                               test_i_hat_mic_feature_tensor,
                               test_j_disease_feature_tensor, test_j_hat_disease_feature_tensor)
        prob_matrix_np = prob_matrix.cpu().numpy()  # 如果你已经确保模型和数据都在 CPU 上，可以省略 .cpu() 调用
        prob_matrix_avg += prob_matrix_np
        result = []
        # for i, j, i_hat, j_hat in train_samples_tensor:
        unique_test_list_tensor = torch.unique(test_list_tensor, dim=0)
        test_labels = unique_test_list_tensor[:, 2].cpu().numpy()  # 实际标签
        indices = unique_test_list_tensor[:, :2].long()  # 确保索引为整数类型
        perdcit_score = prob_matrix[indices[:, 0], indices[:, 1]]  # 使用张量索引获取预测值
        perdcit_score = perdcit_score.cpu().numpy()
        perdcit_label = [1 if prob >= 0.5 else 0 for prob in perdcit_score]

    viz = metrics.RocCurveDisplay.from_predictions(test_labels, perdcit_score,
                                                   name='ROC fold {}'.format(cl),
                                                   color=colors[cl],
                                                   alpha=0.6, lw=2, ax=ax)  # 创建ROC曲线显示对象   绘制了每一折的AUC曲线
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)  # 对TPR进行插值
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)  # 将插值后的TPR添加到列表中
    aucs.append(viz.roc_auc)  # 将每一次交叉验证的ROC AUC值添加到aucs列表中。

    fpr_temp, tpr_temp, _ = roc_curve(test_labels, perdcit_score)
    roc_auc = auc(fpr_temp, tpr_temp)
    sk_fpr_HMDAD.append(fpr_temp)
    sk_tprs_HMDAD.append(tpr_temp)
    sk_aucs_HMDAD.append(roc_auc)

    # 计算Precision-Recall曲线和AUPR
    precision_temp, recall_temp, _ = precision_recall_curve(test_labels, perdcit_score)
    average_precision = average_precision_score(test_labels, perdcit_score)
    sk_precisions_HMDAD.append(precision_temp)
    sk_recalls_HMDAD.append(recall_temp)
    sk_average_precisions_HMDAD.append(average_precision)

prob_matrix_avg = prob_matrix_avg/k_split

mean_fpr_HMDAD = np.linspace(0, 1, 100)
mean_recall_HMDAD  = np.linspace(0, 1, 100)
tprs = []
precisions = []
for fpr_temp, tpr_temp in zip(sk_fpr_HMDAD, sk_tprs_HMDAD):
    interp_tpr = np.interp(mean_fpr_HMDAD, fpr_temp, tpr_temp)
    interp_tpr[0] = 0.0  # 确保曲线从 0 开始
    tprs.append(interp_tpr)
mean_tpr_HMDAD  = np.mean(tprs, axis=0)

mean_tpr_HMDAD [-1] = 1.0  # 确保曲线以 1 结束

for recall_temp, precision_temp in zip(sk_recalls_HMDAD, sk_precisions_HMDAD):
    interp_precision = np.interp(mean_recall_HMDAD, recall_temp[::-1], precision_temp[::-1])
    precisions.append(interp_precision)

mean_precision_HMDAD = np.mean(precisions, axis=0)


#sk_fpr_HMDAD = np.mean(sk_fpr_HMDAD, axis=0)
#sk_tprs_HMDAD = np.mean(sk_tprs_HMDAD, axis=0)

#sk_precisions_HMDAD = np.mean(sk_precisions_HMDAD, axis=0)
#sk_recalls_HMDAD = np.mean(sk_recalls_HMDAD, axis=0)

sk_aucs_HMDAD = np.mean(sk_aucs_HMDAD)
sk_average_precisions_HMDAD = np.mean(sk_average_precisions_HMDAD)
np.savetxt('./result/HMDAD_prob_matrix_avg.csv', prob_matrix_avg, delimiter='\t',fmt='%0.5f')


# ---------------------------------------------------disbiome--------------------------------------------------------------------

iter_ = 0
out = []                    # 用于存储每一折的训练集和测试集索引
test_label_score = {}       # 存储测试标签和预测得分的字典

# disbiome   90  95
A = pd.read_csv('./优化数据集/Disbiome/adj_matrix.csv')
disease_Semantics = pd.read_csv('./优化数据集/Disbiome/疾病-语义/similarity_matrix_model2.csv', header=None)
disease_gip = pd.read_csv('./优化数据集/Disbiome/基于关联矩阵的疾病相似性/GIP_Sim.csv', header=None)
disease_symptoms = pd.read_csv('./优化数据集/Disbiome/疾病-症状/complete_disease_similarity_matrix.csv', header=None)
micro_gip = pd.read_csv('./优化数据集/Disbiome/基于关联矩阵的微生物功能/GIP_Sim.csv', header=None)
micro_sem = pd.read_csv('./优化数据集/Disbiome/基于疾病语义的微生物功能/functional_similarity2_matrix.csv')
micro_fun2 = pd.read_csv('./优化数据集/Disbiome/微生物-功能/complete_microbe_similarities_ds2_matrix.csv', header=None)
A = A.iloc[:, 1:]
disease_gip=disease_gip.iloc[1:, 1:]
micro_gip=micro_gip.iloc[1:, 1:]
micro_fun2=micro_fun2.iloc[1:, 1:]
disease_symptoms=disease_symptoms.iloc[1:, 1:]

microbiome_matrices = [micro_gip.values, micro_sem.values, micro_fun2.values]
disease_matrices = [disease_Semantics.values, disease_gip.values, disease_symptoms.values]
sim_d, sim_m = calculate_combined_similarity(disease_matrices, microbiome_matrices)
A = A.T.to_numpy()


input_dim = sim_m.shape[0] + sim_d.shape[0]
# A = A.astype(float).astype(int).to_numpy()
print("the number of miRNAs and diseases", A.shape)
print("the number of associations", sum(sum(A)))

x, y = A.shape
score_matrix = np.zeros([x, y])  # 初始化评分矩阵

md = A
mm = sim_m
dd = sim_d
deep_A = calculate_metapath_optimized(mm, dd, md, n)

random.seed(1)

print("the number of microbes and diseases", A.shape)
print("the number of associations", sum(sum(A)))
set_seed_2(123)
samples = get_all_pairs(A, deep_A)  # 返回[i, j, i_hat, j_hat] i 微生物 j疾病
samples = np.array(samples)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()

# cross validation
kf = KFold(n_splits=k_split, shuffle=True, random_state=123)  # 定义10折交叉验证
iter_ = 0  # control each iterator  控制迭代次数
sum_score = 0  # 总分数初始化为0
out = []  # 用于存储每一折的训练集和测试集索引
test_label_score = {}  # 存储测试标签和预测得分的字典
criterion = torch.nn.MSELoss()


prob_matrix_avg = np.zeros((A.shape[0], A.shape[1]))
iter_ = 0
out = []  # 用于存储每一折的训练集和测试集索引
test_label_score = {}  # 存储测试标签和预测得分的字典

for cl, (train_index, test_index) in enumerate(kf.split(samples)):  # 循环每一折的训练和测试集
    print('############ {} fold #############'.format(cl))  # 打印当前折数
    out.append([train_index, test_index])  # 将训练和测试集索引存入列表中
    iter_ = iter_ + 1  # 迭代次数加1

    train_samples = samples[train_index, :]  # 获取当前折的训练集样本
    test_samples = samples[test_index, :]  # 获取当前折的测试集样本
    mic_len = sim_m.shape[1]  # 计算微生物潜在表示的向量长度
    dis_len = sim_d.shape[1]  # 计算疾病潜在表示的向量长度

    train_n = train_samples.shape[0]  # 获取训练集样本数量
    test_N = test_samples.shape[0]  # 获取测试集样本数量

    mic_feature = np.zeros([mic_len, input_dim])
    dis_feature = np.zeros([dis_len, input_dim])
    mic_feature = np.concatenate([sim_m, A], axis=1)
    dis_feature = np.concatenate([sim_d, A.T], axis=1)
    train_i_mic_feature = np.zeros([train_n, input_dim])
    train_i_hat_mic_feature = np.zeros([train_n, input_dim])
    train_j_disease_feature = np.zeros([train_n, input_dim])
    train_j_hat_disease_feature = np.zeros([train_n, input_dim])

    test_i_mic_feature = np.zeros([test_N, input_dim])
    test_i_hat_mic_feature = np.zeros([test_N, input_dim])
    test_j_disease_feature = np.zeros([test_N, input_dim])
    test_j_hat_disease_feature = np.zeros([test_N, input_dim])

    model = MicroDiseaseModel_v3(mic_input_dim=input_dim, dis_input_dim=input_dim, latent_dim=output_dim)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.8, last_epoch=-1)

    model.train()

    test_list = []
    train_list = []
    num = 0

    for sample in train_samples:  # epoch ==1000
        # 正相关[i, j]
        i, j, i_hat, j_hat = map(int, sample)
        train_list.append([i, j, 1])
        train_list.append([i, j, 1])
        train_list.append([i, j_hat, 0])
        train_list.append([i_hat, j, 0])

        train_i_mic_feature[num, :] = mic_feature[i, :]
        train_i_hat_mic_feature[num, :] = mic_feature[i_hat, :]
        train_j_disease_feature[num, :] = dis_feature[j, :]
        train_j_hat_disease_feature[num, :] = dis_feature[j_hat, :]
        num += 1
    num = 0
    for sample in test_samples:
        # 正相关[i, j]
        i, j, i_hat, j_hat = map(int, sample)
        test_list.append([i, j, 1])
        # test_list.append([i, j, 1])
        # 负相关[i, j_hat]
        test_list.append([i, j_hat, 0])
        test_list.append([i_hat, j, 0])

        test_i_mic_feature[num, :] = mic_feature[i, :]
        test_i_hat_mic_feature[num, :] = mic_feature[i_hat, :]
        test_j_disease_feature[num, :] = dis_feature[j, :]
        test_j_hat_disease_feature[num, :] = dis_feature[j_hat, :]
        num += 1

    train_list = np.array(train_list)
    test_list = np.array(test_list)
    train_list_tensor = torch.tensor(train_list, dtype=torch.float32).to(device)
    test_list_tensor = torch.tensor(test_list, dtype=torch.float32).to(device)

    # 将数据也移动到指定的设备
    train_samples_tensor = torch.tensor(train_samples, dtype=torch.float32).to(device)
    test_samples_tensor = torch.tensor(test_samples, dtype=torch.float32).to(device)
    mic_feature_tenor = torch.tensor(mic_feature, dtype=torch.float32).to(device)
    dis_feature_tensor = torch.tensor(dis_feature, dtype=torch.float32).to(device)

    train_i_mic_feature_tensor = torch.tensor(train_i_mic_feature, dtype=torch.float32).to(device)
    train_i_hat_mic_feature_tensor = torch.tensor(train_i_hat_mic_feature, dtype=torch.float32).to(device)
    train_j_disease_feature_tensor = torch.tensor(train_j_disease_feature, dtype=torch.float32).to(device)
    train_j_hat_disease_feature_tensor = torch.tensor(train_j_hat_disease_feature, dtype=torch.float32).to(device)
    test_i_mic_feature_tensor = torch.tensor(test_i_mic_feature, dtype=torch.float32).to(device)
    test_i_hat_mic_feature_tensor = torch.tensor(test_i_hat_mic_feature, dtype=torch.float32).to(device)
    test_j_disease_feature_tensor = torch.tensor(test_j_disease_feature, dtype=torch.float32).to(device)
    test_j_hat_disease_feature_tensor = torch.tensor(test_j_hat_disease_feature, dtype=torch.float32).to(device)
    A_tensor = torch.tensor(A, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = 0
        prob_matrix, constrate_loss = model(mic_feature_tenor, dis_feature_tensor, train_i_mic_feature_tensor,
                                            train_i_hat_mic_feature_tensor,
                                            train_j_disease_feature_tensor, train_j_hat_disease_feature_tensor)
        train_labels = train_list_tensor[:, 2]  # 实际标签
        indices = train_list_tensor[:, :2].long()  # 确保索引为整数类型
        train_label = prob_matrix[indices[:, 0], indices[:, 1]]  # 使用张量索引获取预测值

        matrix_diff_loss = torch.mean((prob_matrix - A_tensor) ** 2)
        loss = lambda_mse * criterion(train_label,train_labels) + lambda_constrate * constrate_loss + lambda_l2 * matrix_diff_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        prob_matrix, _ = model(mic_feature_tenor, dis_feature_tensor, test_i_mic_feature_tensor,
                               test_i_hat_mic_feature_tensor,
                               test_j_disease_feature_tensor, test_j_hat_disease_feature_tensor)
        prob_matrix_np = prob_matrix.cpu().numpy()  # 如果你已经确保模型和数据都在 CPU 上，可以省略 .cpu() 调用
        prob_matrix_avg += prob_matrix_np
        result = []
        # for i, j, i_hat, j_hat in train_samples_tensor:
        unique_test_list_tensor = torch.unique(test_list_tensor, dim=0)
        test_labels = unique_test_list_tensor[:, 2].cpu().numpy()  # 实际标签
        indices = unique_test_list_tensor[:, :2].long()  # 确保索引为整数类型
        perdcit_score = prob_matrix[indices[:, 0], indices[:, 1]]  # 使用张量索引获取预测值
        perdcit_score = perdcit_score.cpu().numpy()
        perdcit_label = [1 if prob >= 0.5 else 0 for prob in perdcit_score]

    viz = metrics.RocCurveDisplay.from_predictions(test_labels, perdcit_score,
                                                   name='ROC fold {}'.format(cl),
                                                   color=colors[cl],
                                                   alpha=0.6, lw=2, ax=ax)  # 创建ROC曲线显示对象   绘制了每一折的AUC曲线
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)  # 对TPR进行插值
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)  # 将插值后的TPR添加到列表中
    aucs.append(viz.roc_auc)  # 将每一次交叉验证的ROC AUC值添加到aucs列表中。

    fpr_temp, tpr_temp, _ = roc_curve(test_labels, perdcit_score)
    roc_auc = auc(fpr_temp, tpr_temp)
    sk_fpr_Disbiome.append(fpr_temp)
    sk_tprs_Disbiome.append(tpr_temp)
    sk_aucs_Disbiome.append(roc_auc)

    # 计算Precision-Recall曲线和AUPR
    precision_temp, recall_temp, _ = precision_recall_curve(test_labels, perdcit_score)
    average_precision = average_precision_score(test_labels, perdcit_score)
    sk_precisions_Disbiome.append(precision_temp)
    sk_recalls_Disbiome.append(recall_temp)
    sk_average_precisions_Disbiome.append(average_precision)

prob_matrix_avg = prob_matrix_avg/k_split


mean_fpr_Disbiome = np.linspace(0, 1, 100)
mean_recall_Disbiome  = np.linspace(0, 1, 100)
tprs = []
precisions = []
for fpr_temp, tpr_temp in zip(sk_fpr_Disbiome, sk_tprs_Disbiome):
    interp_tpr = np.interp(mean_fpr_Disbiome, fpr_temp, tpr_temp)
    interp_tpr[0] = 0.0  # 确保曲线从 0 开始
    tprs.append(interp_tpr)
mean_tpr_Disbiome  = np.mean(tprs, axis=0)
mean_tpr_Disbiome [-1] = 1.0  # 确保曲线以 1 结束

for recall_temp, precision_temp in zip(sk_recalls_Disbiome, sk_precisions_Disbiome):
    interp_precision = np.interp(mean_recall_Disbiome, recall_temp[::-1], precision_temp[::-1])
    precisions.append(interp_precision)
mean_precision_Disbiome = np.mean(precisions, axis=0)

sk_aucs_Disbiome = np.mean(sk_aucs_Disbiome)
sk_average_precisions_Disbiome = np.mean(sk_average_precisions_Disbiome)

np.savetxt('./result/Disbiome_prob_matrix_avg.csv', prob_matrix_avg, delimiter='\t',fmt='%0.5f')

# ---------------------------------------------------peryton--------------------------------------------------------------------
iter_ = 0
out = []                    # 用于存储每一折的训练集和测试集索引
test_label_score = {}       # 存储测试标签和预测得分的字典


# peryton 94 97
A = pd.read_csv('./优化数据集/peryton/adjacency_matrix.csv')
disease_Semantics = pd.read_csv('./优化数据集/peryton/疾病-语义/similarity_matrix_model2.csv', header=None)
disease_gip = pd.read_csv('./优化数据集/peryton/基于关联矩阵的疾病相似性/GIP_Sim.csv', header=None)
disease_symptoms = pd.read_csv('./优化数据集/peryton/疾病-症状/complete_disease_similarity_matrix.csv', header=None)
micro_gip = pd.read_csv('./优化数据集/peryton/基于关联矩阵的微生物功能/GIP_Sim.csv', header=None)
micro_sem = pd.read_csv('./优化数据集/peryton/基于疾病语义的微生物功能/functional_similarity2_matrix.csv')
micro_fun2 = pd.read_csv('./优化数据集/peryton/微生物-功能/complete_microbe_similarities_ds2_matrix.csv', header=None)
A = A.iloc[:, 1:]
disease_gip=disease_gip.iloc[1:, 1:]
micro_gip=micro_gip.iloc[1:, 1:]
micro_fun2=micro_fun2.iloc[1:, 1:]
disease_symptoms=disease_symptoms.iloc[1:, 1:]

microbiome_matrices = [micro_gip.values, micro_sem.values, micro_fun2.values]
disease_matrices = [disease_Semantics.values, disease_gip.values, disease_symptoms.values]
sim_d, sim_m = calculate_combined_similarity(disease_matrices, microbiome_matrices)
A = A.T.to_numpy()

input_dim = sim_m.shape[0] + sim_d.shape[0]
# A = A.astype(float).astype(int).to_numpy()
print("the number of miRNAs and diseases", A.shape)
print("the number of associations", sum(sum(A)))

x, y = A.shape
score_matrix = np.zeros([x, y])  # 初始化评分矩阵

md = A
mm = sim_m
dd = sim_d
deep_A = calculate_metapath_optimized(mm, dd, md, n)

random.seed(1)

print("the number of microbes and diseases", A.shape)
print("the number of associations", sum(sum(A)))
set_seed_2(123)
samples = get_all_pairs(A, deep_A)  # 返回[i, j, i_hat, j_hat] i 微生物 j疾病
samples = np.array(samples)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()

# cross validation
kf = KFold(n_splits=k_split, shuffle=True, random_state=123)  # 定义10折交叉验证
iter_ = 0  # control each iterator  控制迭代次数
sum_score = 0  # 总分数初始化为0
out = []  # 用于存储每一折的训练集和测试集索引
test_label_score = {}  # 存储测试标签和预测得分的字典
criterion = torch.nn.MSELoss()

prob_matrix_avg = np.zeros((A.shape[0], A.shape[1]))
iter_ = 0
out = []  # 用于存储每一折的训练集和测试集索引
test_label_score = {}  # 存储测试标签和预测得分的字典

for cl, (train_index, test_index) in enumerate(kf.split(samples)):  # 循环每一折的训练和测试集
    print('############ {} fold #############'.format(cl))  # 打印当前折数
    out.append([train_index, test_index])  # 将训练和测试集索引存入列表中
    iter_ = iter_ + 1  # 迭代次数加1

    train_samples = samples[train_index, :]  # 获取当前折的训练集样本
    test_samples = samples[test_index, :]  # 获取当前折的测试集样本
    mic_len = sim_m.shape[1]  # 计算微生物潜在表示的向量长度
    dis_len = sim_d.shape[1]  # 计算疾病潜在表示的向量长度

    train_n = train_samples.shape[0]  # 获取训练集样本数量
    test_N = test_samples.shape[0]  # 获取测试集样本数量

    mic_feature = np.zeros([mic_len, input_dim])
    dis_feature = np.zeros([dis_len, input_dim])
    mic_feature = np.concatenate([sim_m, A], axis=1)
    dis_feature = np.concatenate([sim_d, A.T], axis=1)
    train_i_mic_feature = np.zeros([train_n, input_dim])
    train_i_hat_mic_feature = np.zeros([train_n, input_dim])
    train_j_disease_feature = np.zeros([train_n, input_dim])
    train_j_hat_disease_feature = np.zeros([train_n, input_dim])

    test_i_mic_feature = np.zeros([test_N, input_dim])
    test_i_hat_mic_feature = np.zeros([test_N, input_dim])
    test_j_disease_feature = np.zeros([test_N, input_dim])
    test_j_hat_disease_feature = np.zeros([test_N, input_dim])

    model = MicroDiseaseModel_v3(mic_input_dim=input_dim, dis_input_dim=input_dim, latent_dim=output_dim)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.8, last_epoch=-1)

    model.train()

    test_list = []
    train_list = []
    num = 0

    for sample in train_samples:  # epoch ==1000
        # 正相关[i, j]
        i, j, i_hat, j_hat = map(int, sample)
        train_list.append([i, j, 1])
        train_list.append([i, j, 1])
        train_list.append([i, j_hat, 0])
        train_list.append([i_hat, j, 0])

        train_i_mic_feature[num, :] = mic_feature[i, :]
        train_i_hat_mic_feature[num, :] = mic_feature[i_hat, :]
        train_j_disease_feature[num, :] = dis_feature[j, :]
        train_j_hat_disease_feature[num, :] = dis_feature[j_hat, :]
        num += 1
    num = 0
    for sample in test_samples:
        # 正相关[i, j]
        i, j, i_hat, j_hat = map(int, sample)
        test_list.append([i, j, 1])
        # test_list.append([i, j, 1])
        # 负相关[i, j_hat]
        test_list.append([i, j_hat, 0])
        test_list.append([i_hat, j, 0])

        test_i_mic_feature[num, :] = mic_feature[i, :]
        test_i_hat_mic_feature[num, :] = mic_feature[i_hat, :]
        test_j_disease_feature[num, :] = dis_feature[j, :]
        test_j_hat_disease_feature[num, :] = dis_feature[j_hat, :]
        num += 1

    train_list = np.array(train_list)
    test_list = np.array(test_list)
    train_list_tensor = torch.tensor(train_list, dtype=torch.float32).to(device)
    test_list_tensor = torch.tensor(test_list, dtype=torch.float32).to(device)

    # 将数据也移动到指定的设备
    train_samples_tensor = torch.tensor(train_samples, dtype=torch.float32).to(device)
    test_samples_tensor = torch.tensor(test_samples, dtype=torch.float32).to(device)
    mic_feature_tenor = torch.tensor(mic_feature, dtype=torch.float32).to(device)
    dis_feature_tensor = torch.tensor(dis_feature, dtype=torch.float32).to(device)

    train_i_mic_feature_tensor = torch.tensor(train_i_mic_feature, dtype=torch.float32).to(device)
    train_i_hat_mic_feature_tensor = torch.tensor(train_i_hat_mic_feature, dtype=torch.float32).to(device)
    train_j_disease_feature_tensor = torch.tensor(train_j_disease_feature, dtype=torch.float32).to(device)
    train_j_hat_disease_feature_tensor = torch.tensor(train_j_hat_disease_feature, dtype=torch.float32).to(device)
    test_i_mic_feature_tensor = torch.tensor(test_i_mic_feature, dtype=torch.float32).to(device)
    test_i_hat_mic_feature_tensor = torch.tensor(test_i_hat_mic_feature, dtype=torch.float32).to(device)
    test_j_disease_feature_tensor = torch.tensor(test_j_disease_feature, dtype=torch.float32).to(device)
    test_j_hat_disease_feature_tensor = torch.tensor(test_j_hat_disease_feature, dtype=torch.float32).to(device)
    A_tensor = torch.tensor(A, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = 0
        prob_matrix, constrate_loss = model(mic_feature_tenor, dis_feature_tensor, train_i_mic_feature_tensor,
                                            train_i_hat_mic_feature_tensor,
                                            train_j_disease_feature_tensor, train_j_hat_disease_feature_tensor)
        train_labels = train_list_tensor[:, 2]  # 实际标签
        indices = train_list_tensor[:, :2].long()  # 确保索引为整数类型
        train_label = prob_matrix[indices[:, 0], indices[:, 1]]  # 使用张量索引获取预测值

        matrix_diff_loss = torch.mean((prob_matrix - A_tensor) ** 2)
        loss = lambda_mse * criterion(train_label,train_labels) + lambda_constrate * constrate_loss + lambda_l2 * matrix_diff_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        prob_matrix, _ = model(mic_feature_tenor, dis_feature_tensor, test_i_mic_feature_tensor,
                               test_i_hat_mic_feature_tensor,
                               test_j_disease_feature_tensor, test_j_hat_disease_feature_tensor)
        prob_matrix_np = prob_matrix.cpu().numpy()  # 如果你已经确保模型和数据都在 CPU 上，可以省略 .cpu() 调用
        prob_matrix_avg += prob_matrix_np
        result = []
        # for i, j, i_hat, j_hat in train_samples_tensor:
        unique_test_list_tensor = torch.unique(test_list_tensor, dim=0)
        test_labels = unique_test_list_tensor[:, 2].cpu().numpy()  # 实际标签
        indices = unique_test_list_tensor[:, :2].long()  # 确保索引为整数类型
        perdcit_score = prob_matrix[indices[:, 0], indices[:, 1]]  # 使用张量索引获取预测值
        perdcit_score = perdcit_score.cpu().numpy()
        perdcit_label = [1 if prob >= 0.5 else 0 for prob in perdcit_score]

    prob_matrix_avg = prob_matrix_avg / k_split

    viz = metrics.RocCurveDisplay.from_predictions(test_labels, perdcit_score,
                                                   name='ROC fold {}'.format(cl),
                                                   color=colors[cl],
                                                   alpha=0.6, lw=2, ax=ax)  # 创建ROC曲线显示对象   绘制了每一折的AUC曲线
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)  # 对TPR进行插值
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)  # 将插值后的TPR添加到列表中
    aucs.append(viz.roc_auc)  # 将每一次交叉验证的ROC AUC值添加到aucs列表中。


    fpr_temp, tpr_temp, _ = roc_curve(test_labels, perdcit_score)
    roc_auc = auc(fpr_temp, tpr_temp)
    sk_fpr_peryton.append(fpr_temp)
    sk_tprs_peryton.append(tpr_temp)
    sk_aucs_peryton.append(roc_auc)

    # 计算Precision-Recall曲线和AUPR
    precision_temp, recall_temp, _ = precision_recall_curve(test_labels, perdcit_score)
    average_precision = average_precision_score(test_labels, perdcit_score)
    sk_precisions_peryton.append(precision_temp)
    sk_recalls_peryton.append(recall_temp)
    sk_average_precisions_peryton.append(average_precision)

mean_fpr_peryton = np.linspace(0, 1, 100)
mean_recall_peryton  = np.linspace(0, 1, 100)
tprs = []
precisions = []
for fpr_temp, tpr_temp in zip(sk_fpr_peryton, sk_tprs_peryton):
    interp_tpr = np.interp(mean_fpr_peryton, fpr_temp, tpr_temp)
    interp_tpr[0] = 0.0  # 确保曲线从 0 开始
    tprs.append(interp_tpr)
mean_tpr_peryton  = np.mean(tprs, axis=0)
mean_tpr_peryton [-1] = 1.0  # 确保曲线以 1 结束

for recall_temp, precision_temp in zip(sk_recalls_peryton, sk_precisions_peryton):
    interp_precision = np.interp(mean_recall_peryton, recall_temp[::-1], precision_temp[::-1])
    precisions.append(interp_precision)
mean_precision_peryton = np.mean(precisions, axis=0)

sk_aucs_peryton = np.mean(sk_aucs_peryton)
sk_average_precisions_peryton = np.mean(sk_average_precisions_peryton)

np.savetxt('./result/peryton_prob_matrix_avg.csv', prob_matrix_avg, delimiter='\t',fmt='%0.5f')

#   ----------------------------------------------------    start plt          ----------------------------------------------------

def compute_mean(values):
    return np.mean(values, axis=0)

# 绘制平均PR曲线
fig2, axs2 = plt.subplots(1, 1, figsize=(5, 5))
#model_labels = ['DSMDA', 'DSMDA_enconder', 'DSMDA_no_constrast', 'DSMDA_random']
model_labels = ['HMDAD', 'Disbiome', 'petyon']
model_precisions = [mean_precision_HMDAD, mean_precision_Disbiome, mean_precision_peryton]
model_recalls = [mean_recall_HMDAD, mean_recall_Disbiome, mean_recall_peryton]
model_auprs = [sk_average_precisions_HMDAD, sk_average_precisions_Disbiome, sk_average_precisions_peryton]

for precisions, recalls, auprs, label in zip(model_precisions, model_recalls, model_auprs, model_labels):

    axs2.step(recalls, precisions, where='post', label=f'{label} AUPR={auprs:.2f}')

axs2.plot([0, 1], [1, 0], '--', color='r', label='Random')
axs2.set_xlabel('Recall')
axs2.set_ylabel('Precision')
axs2.set_ylim([-0.05, 1.05])
axs2.set_xlim([-0.05, 1.05])
axs2.set_title('Precision-Recall curve')
axs2.legend(loc="best")
plt.show()

# 绘制平均ROC曲线
fig3, axs3 = plt.subplots(1, 1, figsize=(5, 5))
model_tprs = [mean_tpr_HMDAD, mean_tpr_Disbiome, mean_tpr_peryton]
model_fprs = [mean_fpr_HMDAD, mean_fpr_Disbiome, mean_fpr_peryton]
model_aucs = [sk_aucs_HMDAD, sk_aucs_Disbiome, sk_aucs_peryton]

for tprs, fprs, aucs, label in zip(model_tprs, model_fprs, model_aucs, model_labels):
    axs3.step(fprs, tprs, where='post', label=f'{label} AUC={aucs:.2f}')

axs3.plot([0, 1], [0, 1], '--', color='r', label='Random')
axs3.set_xlabel('FPR')
axs3.set_ylabel('TPR')
axs3.set_ylim([-0.05, 1.05])
axs3.set_xlim([-0.05, 1.05])
axs3.set_title('ROC curve')
axs3.legend(loc="best")
plt.show()
