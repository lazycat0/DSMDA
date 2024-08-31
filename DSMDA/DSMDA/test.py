# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from sklearn import metrics
# from utils import *
# from sklearn.model_selection import KFold
# import matplotlib
# import torch.optim as optim
# from model3 import MicroDiseaseModel_v3,MicroDiseaseModel_v3_No_VAE
# from get_sim import *
# from sklearn.metrics import (roc_curve, auc, precision_recall_curve, average_precision_score,
#                              f1_score, accuracy_score, recall_score, precision_score, confusion_matrix)
#
# matplotlib.use('TkAgg')
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',  # 使用颜色编码定义颜色
#           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
#
# n=4
# lambda_mse = 4
# lambda_l2 = 2e-3
# lambda_constrate = 3
#
# lambda_l2_list=[1e-3,2e-3,3e-3,4e-3,5e-3,6e-3,7e-3,8e-3,9e-3,1e-2]
# lambda_mse_list=[1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 ,9 , 10]
# lambda_constrate_list=[1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 ,9 , 10]
# #lambda_constrate_list=[10 , 20 , 30 , 40 ,50 , 60,70,80,90,100]
# #lambda_constrate_list=[25,50,75,100,125,150,175,200,225,250,275,300]
# #lambda_constrate_list=[0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 ,0.9 , 1]
# n_list=[1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 ,9 , 10]
#
# set_seed_2(123)
# # paprameters
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# #input_dim = 1396+43  # phendb  1396+43 peryton    1396+43     disbiome 1622+374 疾病特征维度     HMDAD 292+39
# en_hidden_dim = 64    # 隐藏层维度
# hidden_dims = [512, 256, 64]
# output_dim = 32    # 低维输出维度
# epochs = 1000
# k_split = 5
# P = 0.3   # 调整两项损失之前的权重
#
#
# #HMDAD 94 97
#
# A = pd.read_excel('./优化数据集/HMDAD/adj_mat.xlsx')
#
# disease_Semantics = pd.read_csv('./优化数据集/HMDAD/疾病-语义/similarity_matrix_model2.csv', header=None)
# disease_gip = pd.read_csv('./优化数据集/HMDAD/基于关联矩阵的疾病相似性/GIP_Sim.csv', header=None)
# disease_symptoms = pd.read_csv('./优化数据集/HMDAD/疾病-症状/complete_disease_similarity_matrix.csv', header=None)
#
# micro_gip = pd.read_csv('./优化数据集/HMDAD/基于关联矩阵的微生物功能/GIP_Sim.csv', header=None)
# micro_sem = pd.read_csv('./优化数据集/HMDAD/基于疾病语义的微生物功能/functional_similarity2_matrix.csv')
# micro_fun2 = pd.read_csv('./优化数据集/HMDAD/微生物-功能/complete_microbe_similarities_ds2_matrix.csv', header=None)
#
#
# A = A.iloc[1:, 1:]
# disease_gip=disease_gip.iloc[1:, 1:]
# micro_gip=micro_gip.iloc[1:, 1:]
# micro_fun2=micro_fun2.iloc[1:, 1:]
# disease_symptoms=disease_symptoms.iloc[1:, 1:]
#
# microbiome_matrices = [micro_gip.values, micro_sem.values, micro_fun2.values]
# disease_matrices = [disease_Semantics.values, disease_gip.values, disease_symptoms.values]
# sim_d, sim_m = calculate_combined_similarity(disease_matrices, microbiome_matrices)
# A = A.T.to_numpy()
#
#
# input_dim = sim_m.shape[0]+sim_d.shape[0]
# x, y = A.shape
# md=A
# mm=sim_m
# dd=sim_d
#
# '''
# samples = get_all_pairs(A , deep_A )   # 返回[i, j, i_hat, j_hat] i 微生物 j疾病
# samples = np.array(samples)
# kf = KFold(n_splits=k_split, shuffle=True, random_state=123)
# '''
#
# tprs = []
# aucs = []
# mean_fpr = np.linspace(0, 1, 100)
# fig, ax = plt.subplots()
# # cross validation
#
# criterion = torch.nn.MSELoss()
#
# lambda_mse_AUC = []
# lambda_mse_AUPR = []
#
# lambda_constrate_AUC = []
# lambda_constrate_AUPR = []
#
# lambda_l2_AUC = []
# lambda_l2_AUPR = []
#
# n_AUC = []
# n_AUPR = []
#
#
#
# lambda_mse_list=[0,1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 ,9 ]
# lambda_constrate_list=[0,1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 ,9 ]
# n_list=[1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 ,9,10 ]
# results = {}
#
# def test(n_in,lambda_mse_in,lambda_constrate_in):
#
#     sk_tprs = []
#     sk_aucs = []
#     sk_precisions = []
#     sk_recalls = []
#     sk_average_precisions = []
#     sk_fpr = []
#     n = n_in
#     lambda_mse = lambda_mse_in
#     lambda_constrate = lambda_constrate_in
#     deep_A = calculate_metapath_optimized(mm, dd, md, n)
#     samples = get_all_pairs(A, deep_A)  # 返回[i, j, i_hat, j_hat] i 微生物 j疾病
#     samples = np.array(samples)
#     kf = KFold(n_splits=k_split, shuffle=True, random_state=123)
#     #lambda_l2=lambda_l2
#     metric_tmp_sum = np.zeros(8)
#     prob_matrix_avg = np.zeros((A.shape[0], A.shape[1]))
#     iter_ = 0
#     out = []  # 用于存储每一折的训练集和测试集索引
#     test_label_score = {}  # 存储测试标签和预测得分的字典
#
#     for cl, (train_index, test_index) in enumerate(kf.split(samples)):               # 循环每一折的训练和测试集
#         # print('############ {} fold #############'.format(cl))                        # 打印当前折数
#         out.append([train_index, test_index])                                        # 将训练和测试集索引存入列表中
#         iter_ = iter_ + 1                                                           # 迭代次数加1
#
#         train_samples = samples[train_index, :]                                     # 获取当前折的训练集样本
#         test_samples = samples[test_index, :]                                       # 获取当前折的测试集样本
#         mic_len = sim_m.shape[1]                                                # 计算微生物潜在表示的向量长度
#         dis_len = sim_d.shape[1]                                                # 计算疾病潜在表示的向量长度
#
#         train_n = train_samples.shape[0]                                            # 获取训练集样本数量
#         test_N = test_samples.shape[0]                                              # 获取测试集样本数量
#
#         mic_feature = np.zeros([mic_len, input_dim])
#         dis_feature = np.zeros([dis_len, input_dim])
#         mic_feature = np.concatenate( [sim_m , A] , axis=1)
#         dis_feature = np.concatenate( [sim_d , A.T] , axis=1)
#         train_i_mic_feature = np.zeros([train_n,input_dim])
#         train_i_hat_mic_feature = np.zeros([train_n,input_dim])
#         train_j_disease_feature = np.zeros([train_n, input_dim])
#         train_j_hat_disease_feature = np.zeros([train_n, input_dim])
#
#         test_i_mic_feature = np.zeros([test_N, input_dim])
#         test_i_hat_mic_feature = np.zeros([test_N, input_dim])
#         test_j_disease_feature = np.zeros([test_N, input_dim])
#         test_j_hat_disease_feature = np.zeros([test_N, input_dim])
#
#         model = MicroDiseaseModel_v3(mic_input_dim=input_dim, dis_input_dim=input_dim, latent_dim=output_dim)
#         model.to(device)
#
#         optimizer = optim.Adam(model.parameters(), lr=0.001 , weight_decay=1e-4)
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.8, last_epoch=-1)
#
#         model.train()
#
#         test_list=[]
#         train_list=[]
#         num = 0
#         for sample in train_samples:                                    # epoch ==1000
#             # 正相关[i, j]
#             i, j, i_hat, j_hat = map(int, sample)
#             train_list.append([i, j, 1])
#             train_list.append([i, j, 1])
#             train_list.append([i, j_hat, 0])
#             train_list.append([i_hat, j, 0])
#
#             train_i_mic_feature[num,:] = mic_feature[i,:]
#             train_i_hat_mic_feature[num,:] = mic_feature[i_hat,:]
#             train_j_disease_feature[num,:] = dis_feature[j,:]
#             train_j_hat_disease_feature[num,:] = dis_feature[j_hat,:]
#             num += 1
#         num = 0
#         for sample in test_samples:
#             # 正相关[i, j]
#             i, j, i_hat, j_hat = map(int, sample)
#             test_list.append([i, j, 1])
#             test_list.append([i, j, 1])
#             # 负相关[i, j_hat]
#             test_list.append([i, j_hat, 0])
#             test_list.append([i_hat, j, 0])
#
#             test_i_mic_feature[num,:] = mic_feature[i,:]
#             test_i_hat_mic_feature[num,:] = mic_feature[i_hat,:]
#             test_j_disease_feature[num,:] = dis_feature[j,:]
#             test_j_hat_disease_feature[num,:] = dis_feature[j_hat,:]
#             num += 1
#
#         train_list = np.array(train_list)
#         test_list = np.array(test_list)
#         train_list_tensor = torch.tensor(train_list, dtype=torch.float32).to(device)
#         test_list_tensor = torch.tensor(test_list, dtype=torch.float32).to(device)
#
#         # 将数据也移动到指定的设备
#         train_samples_tensor = torch.tensor(train_samples, dtype=torch.float32).to(device)
#         test_samples_tensor = torch.tensor(test_samples, dtype=torch.float32).to(device)
#         mic_feature_tenor = torch.tensor(mic_feature, dtype=torch.float32).to(device)
#         dis_feature_tensor = torch.tensor(dis_feature, dtype=torch.float32).to(device)
#
#         train_i_mic_feature_tensor = torch.tensor(train_i_mic_feature, dtype=torch.float32).to(device)
#         train_i_hat_mic_feature_tensor = torch.tensor(train_i_hat_mic_feature, dtype=torch.float32).to(device)
#         train_j_disease_feature_tensor = torch.tensor(train_j_disease_feature, dtype=torch.float32).to(device)
#         train_j_hat_disease_feature_tensor= torch.tensor(train_j_hat_disease_feature, dtype=torch.float32).to(device)
#         test_i_mic_feature_tensor = torch.tensor(test_i_mic_feature, dtype=torch.float32).to(device)
#         test_i_hat_mic_feature_tensor = torch.tensor(test_i_hat_mic_feature, dtype=torch.float32).to(device)
#         test_j_disease_feature_tensor = torch.tensor(test_j_disease_feature, dtype=torch.float32).to(device)
#         test_j_hat_disease_feature_tensor = torch.tensor(test_j_hat_disease_feature, dtype=torch.float32).to(device)
#         A_tensor = torch.tensor(A, dtype=torch.float32).to(device)
#
#         #-------------------------------------------------------   MY   VAE    -----------------------------------------------------------
#         for epoch in range(epochs):
#             optimizer.zero_grad()
#             loss = 0
#             prob_matrix, constrate_loss = model(mic_feature_tenor, dis_feature_tensor, train_i_mic_feature_tensor, train_i_hat_mic_feature_tensor,
#                                 train_j_disease_feature_tensor, train_j_hat_disease_feature_tensor )
#             #constrate_loss = 0
#             train_labels = train_list_tensor[:, 2]  # 实际标签
#             indices = train_list_tensor[:, :2].long()  # 确保索引为整数类型
#             train_label = prob_matrix[indices[:, 0], indices[:, 1]]  # 使用张量索引获取预测值
#             matrix_diff_loss = torch.mean((prob_matrix - A_tensor) ** 2)
#             loss_l2 = lambda_l2 * torch.norm(prob_matrix, p='fro')
#             loss = lambda_mse*criterion(train_label, train_labels) +lambda_constrate*constrate_loss+ loss_l2
#
#             # if (epoch % 500 ) == 0:
#             #     print('loss=' , loss)
#             loss.backward()
#             optimizer.step()
#             scheduler.step()
#
#         model.eval()
#         with torch.no_grad():
#             prob_matrix, _ = model(mic_feature_tenor, dis_feature_tensor, test_i_mic_feature_tensor,
#                                    test_i_hat_mic_feature_tensor,
#                                    test_j_disease_feature_tensor, test_j_hat_disease_feature_tensor)
#             prob_matrix_np = prob_matrix.cpu().numpy()  # 如果你已经确保模型和数据都在 CPU 上，可以省略 .cpu() 调用
#             prob_matrix_avg += prob_matrix_np
#             result = []
#             # for i, j, i_hat, j_hat in train_samples_tensor:
#             unique_test_list_tensor = torch.unique(test_list_tensor, dim=0)
#             test_labels = unique_test_list_tensor[:, 2].cpu().numpy()  # 实际标签
#             indices = unique_test_list_tensor[:, :2].long()  # 确保索引为整数类型
#             perdcit_score = prob_matrix[indices[:, 0], indices[:, 1]]  # 使用张量索引获取预测值
#             perdcit_score = perdcit_score.cpu().numpy()
#             perdcit_label = [1 if prob >= 0.5 else 0 for prob in perdcit_score]
#
#         viz = metrics.RocCurveDisplay.from_predictions(test_labels, perdcit_score,
#                                                        name='ROC fold {}'.format(cl),
#                                                        color=colors[cl],
#                                                        alpha=0.6, lw=2, ax=ax)  # 创建ROC曲线显示对象   绘制了每一折的AUC曲线
#         interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)  # 对TPR进行插值
#         interp_tpr[0] = 0.0
#         tprs.append(interp_tpr)  # 将插值后的TPR添加到列表中
#         aucs.append(viz.roc_auc)  # 将每一次交叉验证的ROC AUC值添加到aucs列表中。
#         tn, fp, fn, tp = confusion_matrix(test_labels, perdcit_label).ravel()
#         specificity = tn / (tn + fp)
#
#         fpr_temp, tpr_temp, _ = roc_curve(test_labels, perdcit_score)
#         roc_auc = auc(fpr_temp, tpr_temp)
#         sk_fpr.append(fpr_temp)
#         sk_tprs.append(tpr_temp)
#         sk_aucs.append(roc_auc)
#
#         # 计算Precision-Recall曲线和AUPR
#         precision_temp, recall_temp, _ = precision_recall_curve(test_labels, perdcit_score)
#         average_precision = average_precision_score(test_labels, perdcit_score)
#         sk_precisions.append(precision_temp)
#         sk_recalls.append(recall_temp)
#         sk_average_precisions.append(average_precision)
#
#         test_label_score[cl] = [test_labels, perdcit_score]  # 将每次测试的标签和预测概率存储到字典中，以便于后续分析。
#
#     AUPR = np.mean(sk_average_precisions)
#     AUC = np.mean(sk_aucs)
#     return  AUPR,AUC
#
#
# # Iterate over all combinations of the parameter values
# for n in n_list:
#     for lambda_constrate in lambda_constrate_list:
#         for lambda_mse in lambda_mse_list:
#             # Call the test function with the current parameter values
#             print("n:",n)
#             print("lambda_constrate:", lambda_constrate)
#             print("lambda_mse:", lambda_mse)
#             AUPR, AUC = test(n, lambda_mse, lambda_constrate)
#
#             # Store the results in the dictionary
#             if n not in results:
#                 results[n] = {}
#             if lambda_constrate not in results[n]:
#                 results[n][lambda_constrate] = {}
#             results[n][lambda_constrate][lambda_mse] = {
#                 'AUPR': AUPR,
#                 'AUC': AUC
#             }
#
#
# # Save the results to a file (e.g., as a JSON file)
# import json
#
# with open('results.json', 'w') as f:
#     json.dump(results, f, indent=4)
#
# # To load the results later
# with open('results.json', 'r') as f:
#     loaded_results = json.load(f)


#
# def test1(lambda_l2_in):
#
#     sk_tprs = []
#     sk_aucs = []
#     sk_precisions = []
#     sk_recalls = []
#     sk_average_precisions = []
#     sk_fpr = []
#     lambda_l2 = lambda_l2_in
#     n=3
#     lambda_mse = 3
#     lambda_constrate = 2
#     deep_A = calculate_metapath_optimized(mm, dd, md, n)
#     samples = get_all_pairs(A, deep_A)  # 返回[i, j, i_hat, j_hat] i 微生物 j疾病
#     samples = np.array(samples)
#     kf = KFold(n_splits=k_split, shuffle=True, random_state=123)
#     #lambda_l2=lambda_l2
#     metric_tmp_sum = np.zeros(8)
#     prob_matrix_avg = np.zeros((A.shape[0], A.shape[1]))
#     iter_ = 0
#     out = []  # 用于存储每一折的训练集和测试集索引
#     test_label_score = {}  # 存储测试标签和预测得分的字典
#
#     for cl, (train_index, test_index) in enumerate(kf.split(samples)):               # 循环每一折的训练和测试集
#         # print('############ {} fold #############'.format(cl))                        # 打印当前折数
#         out.append([train_index, test_index])                                        # 将训练和测试集索引存入列表中
#         iter_ = iter_ + 1                                                           # 迭代次数加1
#
#         train_samples = samples[train_index, :]                                     # 获取当前折的训练集样本
#         test_samples = samples[test_index, :]                                       # 获取当前折的测试集样本
#         mic_len = sim_m.shape[1]                                                # 计算微生物潜在表示的向量长度
#         dis_len = sim_d.shape[1]                                                # 计算疾病潜在表示的向量长度
#
#         train_n = train_samples.shape[0]                                            # 获取训练集样本数量
#         test_N = test_samples.shape[0]                                              # 获取测试集样本数量
#
#         mic_feature = np.zeros([mic_len, input_dim])
#         dis_feature = np.zeros([dis_len, input_dim])
#         mic_feature = np.concatenate( [sim_m , A] , axis=1)
#         dis_feature = np.concatenate( [sim_d , A.T] , axis=1)
#         train_i_mic_feature = np.zeros([train_n,input_dim])
#         train_i_hat_mic_feature = np.zeros([train_n,input_dim])
#         train_j_disease_feature = np.zeros([train_n, input_dim])
#         train_j_hat_disease_feature = np.zeros([train_n, input_dim])
#
#         test_i_mic_feature = np.zeros([test_N, input_dim])
#         test_i_hat_mic_feature = np.zeros([test_N, input_dim])
#         test_j_disease_feature = np.zeros([test_N, input_dim])
#         test_j_hat_disease_feature = np.zeros([test_N, input_dim])
#
#         model = MicroDiseaseModel_v3(mic_input_dim=input_dim, dis_input_dim=input_dim, latent_dim=output_dim)
#         model.to(device)
#
#         optimizer = optim.Adam(model.parameters(), lr=0.001 , weight_decay=1e-4)
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.8, last_epoch=-1)
#
#         model.train()
#
#         test_list=[]
#         train_list=[]
#         num = 0
#         for sample in train_samples:                                    # epoch ==1000
#             # 正相关[i, j]
#             i, j, i_hat, j_hat = map(int, sample)
#             train_list.append([i, j, 1])
#             train_list.append([i, j, 1])
#             train_list.append([i, j_hat, 0])
#             train_list.append([i_hat, j, 0])
#
#             train_i_mic_feature[num,:] = mic_feature[i,:]
#             train_i_hat_mic_feature[num,:] = mic_feature[i_hat,:]
#             train_j_disease_feature[num,:] = dis_feature[j,:]
#             train_j_hat_disease_feature[num,:] = dis_feature[j_hat,:]
#             num += 1
#         num = 0
#         for sample in test_samples:
#             # 正相关[i, j]
#             i, j, i_hat, j_hat = map(int, sample)
#             test_list.append([i, j, 1])
#             test_list.append([i, j, 1])
#             # 负相关[i, j_hat]
#             test_list.append([i, j_hat, 0])
#             test_list.append([i_hat, j, 0])
#
#             test_i_mic_feature[num,:] = mic_feature[i,:]
#             test_i_hat_mic_feature[num,:] = mic_feature[i_hat,:]
#             test_j_disease_feature[num,:] = dis_feature[j,:]
#             test_j_hat_disease_feature[num,:] = dis_feature[j_hat,:]
#             num += 1
#
#         train_list = np.array(train_list)
#         test_list = np.array(test_list)
#         train_list_tensor = torch.tensor(train_list, dtype=torch.float32).to(device)
#         test_list_tensor = torch.tensor(test_list, dtype=torch.float32).to(device)
#
#         # 将数据也移动到指定的设备
#         train_samples_tensor = torch.tensor(train_samples, dtype=torch.float32).to(device)
#         test_samples_tensor = torch.tensor(test_samples, dtype=torch.float32).to(device)
#         mic_feature_tenor = torch.tensor(mic_feature, dtype=torch.float32).to(device)
#         dis_feature_tensor = torch.tensor(dis_feature, dtype=torch.float32).to(device)
#
#         train_i_mic_feature_tensor = torch.tensor(train_i_mic_feature, dtype=torch.float32).to(device)
#         train_i_hat_mic_feature_tensor = torch.tensor(train_i_hat_mic_feature, dtype=torch.float32).to(device)
#         train_j_disease_feature_tensor = torch.tensor(train_j_disease_feature, dtype=torch.float32).to(device)
#         train_j_hat_disease_feature_tensor= torch.tensor(train_j_hat_disease_feature, dtype=torch.float32).to(device)
#         test_i_mic_feature_tensor = torch.tensor(test_i_mic_feature, dtype=torch.float32).to(device)
#         test_i_hat_mic_feature_tensor = torch.tensor(test_i_hat_mic_feature, dtype=torch.float32).to(device)
#         test_j_disease_feature_tensor = torch.tensor(test_j_disease_feature, dtype=torch.float32).to(device)
#         test_j_hat_disease_feature_tensor = torch.tensor(test_j_hat_disease_feature, dtype=torch.float32).to(device)
#         A_tensor = torch.tensor(A, dtype=torch.float32).to(device)
#
#         #-------------------------------------------------------   MY   VAE    -----------------------------------------------------------
#         for epoch in range(epochs):
#             optimizer.zero_grad()
#             loss = 0
#             prob_matrix, constrate_loss = model(mic_feature_tenor, dis_feature_tensor, train_i_mic_feature_tensor, train_i_hat_mic_feature_tensor,
#                                 train_j_disease_feature_tensor, train_j_hat_disease_feature_tensor )
#             #constrate_loss = 0
#             train_labels = train_list_tensor[:, 2]  # 实际标签
#             indices = train_list_tensor[:, :2].long()  # 确保索引为整数类型
#             train_label = prob_matrix[indices[:, 0], indices[:, 1]]  # 使用张量索引获取预测值
#             matrix_diff_loss = torch.mean((prob_matrix - A_tensor) ** 2)
#             loss_l2 = lambda_l2 * torch.norm(prob_matrix, p='fro')
#             loss = lambda_mse*criterion(train_label, train_labels) +lambda_constrate*constrate_loss+ loss_l2
#
#             # if (epoch % 500 ) == 0:
#             #     print('loss=' , loss)
#             loss.backward()
#             optimizer.step()
#             scheduler.step()
#
#         model.eval()
#         with torch.no_grad():
#             prob_matrix, _ = model(mic_feature_tenor, dis_feature_tensor, test_i_mic_feature_tensor,
#                                    test_i_hat_mic_feature_tensor,
#                                    test_j_disease_feature_tensor, test_j_hat_disease_feature_tensor)
#             prob_matrix_np = prob_matrix.cpu().numpy()  # 如果你已经确保模型和数据都在 CPU 上，可以省略 .cpu() 调用
#             prob_matrix_avg += prob_matrix_np
#             result = []
#             # for i, j, i_hat, j_hat in train_samples_tensor:
#             unique_test_list_tensor = torch.unique(test_list_tensor, dim=0)
#             test_labels = unique_test_list_tensor[:, 2].cpu().numpy()  # 实际标签
#             indices = unique_test_list_tensor[:, :2].long()  # 确保索引为整数类型
#             perdcit_score = prob_matrix[indices[:, 0], indices[:, 1]]  # 使用张量索引获取预测值
#             perdcit_score = perdcit_score.cpu().numpy()
#             perdcit_label = [1 if prob >= 0.5 else 0 for prob in perdcit_score]
#
#         viz = metrics.RocCurveDisplay.from_predictions(test_labels, perdcit_score,
#                                                        name='ROC fold {}'.format(cl),
#                                                        color=colors[cl],
#                                                        alpha=0.6, lw=2, ax=ax)  # 创建ROC曲线显示对象   绘制了每一折的AUC曲线
#         interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)  # 对TPR进行插值
#         interp_tpr[0] = 0.0
#         tprs.append(interp_tpr)  # 将插值后的TPR添加到列表中
#         aucs.append(viz.roc_auc)  # 将每一次交叉验证的ROC AUC值添加到aucs列表中。
#         tn, fp, fn, tp = confusion_matrix(test_labels, perdcit_label).ravel()
#         specificity = tn / (tn + fp)
#
#         fpr_temp, tpr_temp, _ = roc_curve(test_labels, perdcit_score)
#         roc_auc = auc(fpr_temp, tpr_temp)
#         sk_fpr.append(fpr_temp)
#         sk_tprs.append(tpr_temp)
#         sk_aucs.append(roc_auc)
#
#         # 计算Precision-Recall曲线和AUPR
#         precision_temp, recall_temp, _ = precision_recall_curve(test_labels, perdcit_score)
#         average_precision = average_precision_score(test_labels, perdcit_score)
#         sk_precisions.append(precision_temp)
#         sk_recalls.append(recall_temp)
#         sk_average_precisions.append(average_precision)
#
#         test_label_score[cl] = [test_labels, perdcit_score]  # 将每次测试的标签和预测概率存储到字典中，以便于后续分析。
#
#     AUPR = np.mean(sk_average_precisions)
#     AUC = np.mean(sk_aucs)
#     return  AUPR,AUC
#
#
# for lambda_l2 in lambda_l2_list:
#     AUPR,AUC = test1(lambda_l2)
#     lambda_l2_AUPR.append(AUPR)
#     lambda_l2_AUC.append(AUC)
#
#
#
# plt.figure(figsize=(10, 6))
# plt.plot(lambda_l2_list, lambda_l2_AUPR, marker='o', linestyle='-', color='b')
# plt.title('AUC Score vs. lambda_l2')
# plt.xlabel('lambda_l2')
# plt.ylabel('AUC Score')
# plt.grid(True)
# plt.xticks(lambda_l2_list)
# plt.show()
#
#
# plt.figure(figsize=(10, 6))
# plt.plot(lambda_l2_list, lambda_l2_AUC, marker='o', linestyle='-', color='b')
# plt.title('AUPR Score vs. lambda_l2')
# plt.xlabel('lambda_l2')
# plt.ylabel('AUPR Score')
# plt.grid(True)
# plt.xticks(lambda_l2_list)
# plt.show()





################################## plot  ######################################################
# import json
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.colors as mcolors
#
# # 从 JSON 文件中加载结果数据
# with open('results.json', 'r', encoding='utf-8') as f:
#     results = json.load(f)
#
# # 准备绘图所需的列表
# n_values = []
# lambda_mse_values = []
# lambda_constrate_values = []
# AUC_values = []
# AUPR_values = []
#
# # 遍历结果数据并提取值
# for n, n_data in results.items():
#     for lambda_constrate, lambda_constrate_data in n_data.items():
#         for lambda_mse, metrics in lambda_constrate_data.items():
#             n_values.append(int(n))
#             lambda_mse_values.append(int(lambda_mse))
#             lambda_constrate_values.append(int(lambda_constrate))
#             AUC_values.append(metrics['AUC'])
#             AUPR_values.append(metrics['AUPR'])
#
# # 创建一个具有明确区间的自定义颜色映射
# colors = [(0.0, 'blue'), (0.8, 'yellow'), (0.85, 'red'),
#           (0.9, 'purple'), (0.95, 'darkred'), (1.0, 'black')]
#
# # 通过插值创建更多的颜色区间
# cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
#
# # 创建 AUC 的 3D 散点图
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111, projection='3d')
# scatter1 = ax1.scatter(n_values, lambda_mse_values, lambda_constrate_values, c=AUC_values, cmap=cmap, vmin=0.8, vmax=1.0)
# ax1.set_xlabel('n')
# ax1.set_ylabel('lambda_mse')
# ax1.set_zlabel('lambda_constrate')
# ax1.set_title('3D Visualization of AUC Scores')
# ax1.view_init(elev=30, azim=120)
#
# # 创建并调整颜色条的位置
# cbar1 = plt.colorbar(scatter1, ax=ax1, pad=0.1, aspect=10, shrink=0.7)
# cbar1.set_label('AUC')
# cbar1.ax.tick_params(labelsize=8)  # 调整颜色条刻度的字体大小
#
# # 调整布局以避免标签或标题被截断
# plt.tight_layout()
# plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)  # 微调布局以防颜色条遮挡
#
# # 保存为 PDF 格式
# fig1.savefig('AUC_3D_Visualization.pdf', format='pdf')
#
# # 创建 AUPR 的 3D 散点图
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111, projection='3d')
# scatter2 = ax2.scatter(n_values, lambda_mse_values, lambda_constrate_values, c=AUPR_values, cmap=cmap, vmin=0.8, vmax=1.0)
# ax2.set_xlabel('n')
# ax2.set_ylabel('lambda_mse')
# ax2.set_zlabel('lambda_constrate')
# ax2.set_title('3D Visualization of AUPR Scores')
# ax2.view_init(elev=30, azim=120)
#
# # 创建并调整颜色条的位置
# cbar2 = plt.colorbar(scatter2, ax=ax2, pad=0.1, aspect=10, shrink=0.7)
# cbar2.set_label('AUPR')
# cbar2.ax.tick_params(labelsize=8)  # 调整颜色条刻度的字体大小
#
# # 调整布局以避免标签或标题被截断
# plt.tight_layout()
# plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)  # 微调布局以防颜色条遮挡
#
# # 保存为 PDF 格式
# fig2.savefig('AUPR_3D_Visualization.pdf', format='pdf')
#
# # 分别显示图形
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# # 保存所有绘图所需数据
# np.savez('complete_average_plot_data.npz', mean_fpr=mean_fpr, mean_tprs=mean_tprs, mean_auc=mean_auc,
#          mean_recall=mean_recall, mean_precisions=mean_precisions, mean_average_precision=mean_average_precision)

# 加载数据
data = np.load('complete_average_plot_data.npz')
mean_fpr = data['mean_fpr']
mean_tprs = data['mean_tprs']
mean_auc = data['mean_auc']
mean_recall = data['mean_recall']
mean_precisions = data['mean_precisions']
mean_average_precision = data['mean_average_precision']

# 绘制平均ROC曲线
fig4, axs4 = plt.subplots(1, 1, figsize=(5, 5))
axs4.plot(mean_fpr, mean_tprs, label=f'Average AUC = {mean_auc:.2f}')
axs4.plot([0, 1], [0, 1], 'k--', label='Random', color='r')
axs4.set_xlim([-0.05, 1.05])
axs4.set_ylim([0.0, 1.05])
axs4.set_xlabel('False Positive Rate')
axs4.set_ylabel('True Positive Rate')
axs4.set_title('Average ROC Curve')
axs4.legend(loc="lower right")
plt.show()

# 绘制平均Precision-Recall曲线
fig5, axs5 = plt.subplots(1, 1, figsize=(5, 5))
axs5.plot(mean_recall, mean_precisions, label=f'Average AUPR = {mean_average_precision:.2f}')
axs5.plot([0, 1], [1, 0], 'k--', label='Random', color='r')
axs5.set_xlim([-0.05, 1.05])
axs5.set_ylim([0.0, 1.05])
axs5.set_xlabel('Recall')
axs5.set_ylabel('Precision')
axs5.set_title('Average Precision-Recall Curve')
axs5.legend(loc="lower left")
plt.show()
