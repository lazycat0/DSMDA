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
from model3 import MicroDiseaseModel_v3 , MicroDiseaseModel_v3_No_VAE , MicroDiseaseModel_v2
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, average_precision_score,
                             f1_score, accuracy_score, recall_score, precision_score, confusion_matrix)
from get_sim import *
from sklearn.metrics import auc
import torch.nn.functional as F

matplotlib.use('TkAgg')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',  # 使用颜色编码定义颜色
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# paprameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
en_hidden_dim = 64    # 隐藏层维度
hidden_dims = [512, 256, 64]
output_dim = 32    # 低维输出维度
epochs = 1000
k_split = 5
P = 0.3   # 调整两项损失之前的权重
random.seed(1)
set_seed_2(123)


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
A = A.T.to_numpy()


input_dim = sim_m.shape[0]+sim_d.shape[0]
print("the number of miRNAs and diseases", A.shape)
print("the number of associations", sum(sum(A)))

x, y = A.shape
score_matrix = np.zeros([x, y])                                 # 初始化评分矩阵


md=A
mm=sim_m
dd=sim_d
n=4
deep_A=calculate_metapath_optimized(mm, dd, md, n)
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



precisions = []
precisions_No_VAE = []
precisions_random = []
precisions_no_constrate = []

sk_tprs = []
sk_aucs = []
sk_precisions = []
sk_recalls = []
sk_average_precisions = []
sk_fpr= []


sk_tprs_No_VAE = []
sk_aucs_No_VAE = []
sk_precisions_No_VAE = []
sk_recalls_No_VAE = []
sk_average_precisions_No_VAE = []
sk_fpr_No_VAE = []


sk_tprs_random = []
sk_aucs_random = []
sk_precisions_random = []
sk_recalls_random = []
sk_average_precisions_random = []
sk_fpr_random = []


sk_tprs_no_constrate = []
sk_aucs_no_constrate = []
sk_precisions_no_constrate = []
sk_recalls_no_constrate = []
sk_average_precisions_no_constrate = []
sk_fpr_no_constrate = []






lambda_mse = 4
lambda_l2 = 2e-3
lambda_constrate = 3



# ----------------------------------------------------------------------------No VAE---------------------------------------------------------------

metric_tmp_sum = np.zeros(8)
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

    model = MicroDiseaseModel_v3_No_VAE(mic_input_dim=input_dim, dis_input_dim=input_dim, latent_dim=output_dim)
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

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = 0
        prob_matrix, constrate_loss = model(mic_feature_tenor, dis_feature_tensor, train_i_mic_feature_tensor, train_i_hat_mic_feature_tensor,
                            train_j_disease_feature_tensor, train_j_hat_disease_feature_tensor )
        #constrate_loss = 0
        train_labels = train_list_tensor[:, 2]  # 实际标签
        indices = train_list_tensor[:, :2].long()  # 确保索引为整数类型
        train_label = prob_matrix[indices[:, 0], indices[:, 1]]  # 使用张量索引获取预测值

        loss_l2 = lambda_l2 * torch.norm(prob_matrix, p='fro')
        loss = lambda_mse*criterion(train_label, train_labels) +lambda_constrate*constrate_loss+ loss_l2

        if (epoch % 500 ) == 0:
            print('loss=' , loss)
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
    sk_fpr_No_VAE.append(fpr_temp)
    sk_tprs_No_VAE.append(tpr_temp)
    sk_aucs_No_VAE.append(roc_auc)

    # 计算Precision-Recall曲线和AUPR
    precision_temp, recall_temp, _ = precision_recall_curve(test_labels, perdcit_score)
    average_precision = average_precision_score(test_labels, perdcit_score)
    sk_precisions_No_VAE.append(precision_temp)
    sk_recalls_No_VAE.append(recall_temp)
    sk_average_precisions_No_VAE.append(average_precision)

prob_matrix_avg = prob_matrix_avg / k_split

mean_fpr_No_VAE = np.linspace(0, 1, 100)
mean_recall_No_VAE = np.linspace(0, 1, 100)
tprs = []
precisions = []
for fpr_temp, tpr_temp in zip(sk_fpr_No_VAE, sk_tprs_No_VAE):
    interp_tpr = np.interp(mean_fpr_No_VAE, fpr_temp, tpr_temp)
    interp_tpr[0] = 0.0  # 确保曲线从 0 开始
    tprs.append(interp_tpr)
mean_tpr_No_VAE = np.mean(tprs, axis=0)

mean_tpr_No_VAE[-1] = 1.0  # 确保曲线以 1 结束

for recall_temp, precision_temp in zip(sk_recalls_No_VAE, sk_precisions_No_VAE):
    interp_precision = np.interp(mean_recall_No_VAE, recall_temp[::-1], precision_temp[::-1])
    precisions.append(interp_precision)

mean_precision_No_VAE = np.mean(precisions, axis=0)

sk_aucs_No_VAE = np.mean(sk_aucs_No_VAE)
sk_average_precisions_No_VAE = np.mean(sk_average_precisions_No_VAE)



#---------------------------------------------------------   NO_VAE end      ----------------------------------------------------------


prob_matrix_avg = np.zeros((A.shape[0], A.shape[1]))
iter_ = 0
out = []                    # 用于存储每一折的训练集和测试集索引
test_label_score = {}       # 存储测试标签和预测得分的字典

#-------------------------------------------------------   MY   VAE    -----------------------------------------------------------

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
        loss = lambda_mse*criterion(train_label, train_labels) +lambda_constrate*constrate_loss+ loss_l2

        if (epoch % 500 ) == 0:
            print('loss=' , loss)
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
    sk_fpr.append(fpr_temp)
    sk_tprs.append(tpr_temp)
    sk_aucs.append(roc_auc)

    # 计算Precision-Recall曲线和AUPR
    precision_temp, recall_temp, _ = precision_recall_curve(test_labels, perdcit_score)
    average_precision = average_precision_score(test_labels, perdcit_score)
    sk_precisions.append(precision_temp)
    sk_recalls.append(recall_temp)
    sk_average_precisions.append(average_precision)

prob_matrix_avg = prob_matrix_avg/k_split

mean_fpr = np.linspace(0, 1, 100)
mean_recall  = np.linspace(0, 1, 100)
tprs = []
precisions = []
for fpr_temp, tpr_temp in zip(sk_fpr, sk_tprs):
    interp_tpr = np.interp(mean_fpr, fpr_temp, tpr_temp)
    interp_tpr[0] = 0.0  # 确保曲线从 0 开始
    tprs.append(interp_tpr)
mean_tpr  = np.mean(tprs, axis=0)

mean_tpr [-1] = 1.0  # 确保曲线以 1 结束

for recall_temp, precision_temp in zip(sk_recalls, sk_precisions):
    interp_precision = np.interp(mean_recall, recall_temp[::-1], precision_temp[::-1])
    precisions.append(interp_precision)

mean_precision = np.mean(precisions, axis=0)

sk_aucs = np.mean(sk_aucs)
sk_average_precisions = np.mean(sk_average_precisions)




# ---------------------------------------------------------   NO constrate  ----------------------------------------------------------

prob_matrix_avg = np.zeros((A.shape[0], A.shape[1]))
iter_ = 0
out = []                    # 用于存储每一折的训练集和测试集索引
test_label_score = {}       # 存储测试标签和预测得分的字典


samples = get_all_pairs(A , deep_A )   # 返回[i, j, i_hat, j_hat] i 微生物 j疾病
samples = np.array(samples)

sample_list =[]
num = 0
for sample in samples:  # epoch ==1000
    # 正相关[i, j]
    i, j, i_hat, j_hat = map(int, sample)
    sample_list.append([i, j, 1])
    sample_list.append([i, j, 1])
    sample_list.append([i, j_hat, 0])
    sample_list.append([i_hat, j, 0])
    num += 1
num = 0
sample_array = np.array(sample_list)


for cl, (train_index, test_index) in enumerate(kf.split(sample_array)):               # 循环每一折的训练和测试集
    print('############ {} fold #############'.format(cl))                        # 打印当前折数
    out.append([train_index, test_index])                                        # 将训练和测试集索引存入列表中
    iter_ = iter_ + 1                                                           # 迭代次数加1

    train_samples = sample_array[train_index, :]                                     # 获取当前折的训练集样本
    test_samples = sample_array[test_index, :]                                       # 获取当前折的测试集样本
    mic_len = sim_m.shape[1]                                                # 计算微生物潜在表示的向量长度
    dis_len = sim_d.shape[1]                                                # 计算疾病潜在表示的向量长度

    train_n = train_samples.shape[0]                                            # 获取训练集样本数量
    test_N = test_samples.shape[0]                                              # 获取测试集样本数量

    mic_feature = np.zeros([mic_len, input_dim])
    dis_feature = np.zeros([dis_len, input_dim])
    mic_feature = np.concatenate( [sim_m , A] , axis=1)
    dis_feature = np.concatenate( [sim_d , A.T] , axis=1)
    model = MicroDiseaseModel_v2(mic_input_dim=input_dim, dis_input_dim=input_dim, latent_dim=output_dim)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001 , weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.8, last_epoch=-1)

    model.train()

    test_list=[]
    train_list=[]
    train_list = train_samples
    test_list = test_samples

    train_list = np.array(train_list)
    test_list = np.array(test_list)
    train_list_tensor = torch.tensor(train_list, dtype=torch.float32).to(device)
    test_list_tensor = torch.tensor(test_list, dtype=torch.float32).to(device)

    # 将数据也移动到指定的设备
    mic_feature_tenor = torch.tensor(mic_feature, dtype=torch.float32).to(device)
    dis_feature_tensor = torch.tensor(dis_feature, dtype=torch.float32).to(device)
    A_tensor = torch.tensor(A, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = 0
        prob_matrix= model(mic_feature_tenor, dis_feature_tensor)
        train_labels = train_list_tensor[:, 2]  # 实际标签
        indices = train_list_tensor[:, :2].long()  # 确保索引为整数类型
        train_label = prob_matrix[indices[:, 0], indices[:, 1]]  # 使用张量索引获取预测值

        matrix_diff_loss = torch.mean((prob_matrix - A_tensor) ** 2)
        loss = lambda_mse * criterion(train_label,train_labels)+ lambda_l2 * matrix_diff_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        prob_matrix= model(mic_feature_tenor, dis_feature_tensor)
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
    sk_fpr_no_constrate.append(fpr_temp)
    sk_tprs_no_constrate.append(tpr_temp)
    sk_aucs_no_constrate.append(roc_auc)

    # 计算Precision-Recall曲线和AUPR
    precision_temp, recall_temp, _ = precision_recall_curve(test_labels, perdcit_score)
    average_precision = average_precision_score(test_labels, perdcit_score)
    sk_precisions_no_constrate.append(precision_temp)
    sk_recalls_no_constrate.append(recall_temp)
    sk_average_precisions_no_constrate.append(average_precision)

prob_matrix_avg = prob_matrix_avg / k_split

mean_fpr_no_constrate = np.linspace(0, 1, 100)
mean_recall_no_constrate = np.linspace(0, 1, 100)
tprs = []
precisions = []
for fpr_temp, tpr_temp in zip(sk_fpr_no_constrate, sk_tprs_no_constrate):
    interp_tpr = np.interp(mean_fpr_no_constrate, fpr_temp, tpr_temp)
    interp_tpr[0] = 0.0  # 确保曲线从 0 开始
    tprs.append(interp_tpr)
mean_tpr_no_constrate = np.mean(tprs, axis=0)

mean_tpr_no_constrate[-1] = 1.0  # 确保曲线以 1 结束

for recall_temp, precision_temp in zip(sk_recalls_no_constrate, sk_precisions_no_constrate):
    interp_precision = np.interp(mean_recall_no_constrate, recall_temp[::-1], precision_temp[::-1])
    precisions.append(interp_precision)
mean_precision_no_constrate = np.mean(precisions, axis=0)
sk_aucs_no_constrate = np.mean(sk_aucs_no_constrate)
sk_average_precisions_no_constrate = np.mean(sk_average_precisions_no_constrate)


# ---------------------------------------------------------   NO constrate  end      ----------------------------------------------------------


# ---------------------------------------------------------------------------- random select  ---------------------------------------------------------------
metric_tmp_sum = np.zeros(8)
#samples = get_all_pairs_random(A)   # 返回[i, j, 标签] i 微生物 j疾病
# samples = get_all_pairs(A,A)
samples = get_all_pairs_random(A)
samples = np.array(samples)

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

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = 0

        prob_matrix, constrate_loss = model(mic_feature_tenor, dis_feature_tensor, train_i_mic_feature_tensor, train_i_hat_mic_feature_tensor,
                            train_j_disease_feature_tensor, train_j_hat_disease_feature_tensor )
        train_labels = train_list_tensor[:, 2]  # 实际标签
        indices = train_list_tensor[:, :2].long()  # 确保索引为整数类型
        train_label = prob_matrix[indices[:, 0], indices[:, 1]]  # 使用张量索引获取预测值

        loss_l2 = lambda_l2 * torch.norm(prob_matrix, p='fro')
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
    sk_fpr_random.append(fpr_temp)
    sk_tprs_random.append(tpr_temp)
    sk_aucs_random.append(roc_auc)

    # 计算Precision-Recall曲线和AUPR
    precision_temp, recall_temp, _ = precision_recall_curve(test_labels, perdcit_score)
    average_precision = average_precision_score(test_labels, perdcit_score)
    sk_precisions_random.append(precision_temp)
    sk_recalls_random.append(recall_temp)
    sk_average_precisions_random.append(average_precision)

prob_matrix_avg = prob_matrix_avg / k_split

mean_fpr_random = np.linspace(0, 1, 100)
mean_recall_random = np.linspace(0, 1, 100)
tprs = []
precisions = []
for fpr_temp, tpr_temp in zip(sk_fpr_random, sk_tprs_random):
    interp_tpr = np.interp(mean_fpr_random, fpr_temp, tpr_temp)
    interp_tpr[0] = 0.0  # 确保曲线从 0 开始
    tprs.append(interp_tpr)
mean_tpr_random = np.mean(tprs, axis=0)

mean_tpr_random[-1] = 1.0  # 确保曲线以 1 结束

for recall_temp, precision_temp in zip(sk_recalls_random, sk_precisions_random):
    interp_precision = np.interp(mean_recall_random, recall_temp[::-1], precision_temp[::-1])
    precisions.append(interp_precision)

mean_precision_random = np.mean(precisions, axis=0)
sk_aucs_random = np.mean(sk_aucs_random)
sk_average_precisions_random = np.mean(sk_average_precisions_random)
#  ------------------------------------------------------           random end            ----------------------------------------------------------

def compute_mean(values):
    return np.mean(values, axis=0)

fig2, axs2 = plt.subplots(1, 1, figsize=(5, 5))
model_labels = ['DSMDA', 'DSMDA_enconder', 'DSMDA_random']
model_precisions = [mean_precision, mean_precision_No_VAE, mean_precision_random]
model_recalls = [mean_recall, mean_recall_No_VAE, mean_recall_random]
model_auprs = [sk_average_precisions, sk_average_precisions_No_VAE, sk_average_precisions_random]

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
model_tprs = [mean_tpr, mean_tpr_No_VAE, mean_tpr_random]
model_fprs = [mean_fpr, mean_fpr_No_VAE, mean_fpr_random]
model_aucs = [sk_aucs, sk_aucs_No_VAE, sk_aucs_random]

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