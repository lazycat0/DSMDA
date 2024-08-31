import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn import metrics
import  matplotlib
matplotlib.use('TkAgg')


# 定义统一的线条粗细和颜色列表  DSMDA
line_width = 2
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# 创建画布
plt.figure(figsize=(10, 8))

# 模型1: DSMDA
data_mvg = np.load('complete_average_plot_data.npz')
plt.plot(data_mvg['mean_fpr'], data_mvg['mean_tprs'], label=f'DSMDA (AUC = {data_mvg["mean_auc"]:.2f})', lw=line_width, color=colors[0])

# 模型2: DSAE_RF
with open('../../compare_modle/compare_data/DSAE_RF/roc_data.pkl', 'rb') as f:
    roc_data = pickle.load(f)
auc = metrics.auc(roc_data['mean_fpr'], np.mean(roc_data['tprs'], axis=0))
plt.plot(roc_data['mean_fpr'], np.mean(roc_data['tprs'], axis=0), label=f'DSAE_RF (AUC = {auc:.3f})', lw=line_width, color=colors[1])


# 模型3: SAELGMDA

with open('../../compare_modle/compare_data/SAELGMDA/roc_data.pkl', 'rb') as f:
    roc_data = pickle.load(f)
tprs = roc_data['tprs']
mean_fpr = roc_data['mean_fpr']
auc = metrics.auc(mean_fpr, np.mean(tprs ,axis=0))
plt.plot(mean_fpr, np.mean(tprs, axis=0), label=f'SAELGMDA (AUC = {auc:.3f})', lw=line_width, color=colors[2] )

# 模型4: GPUMDA
with open('../../compare_modle/compare_data/GPUDMDA/average_roc_data.pkl', 'rb') as f:
    roc_data_gpu = pickle.load(f)
plt.plot(roc_data_gpu['fpr'], roc_data_gpu['tpr'], label=f'GPUMDA (AUC = {roc_data_gpu["auc"]:.3f})', lw=line_width, color=colors[3])

# 模型5: KGNMDA
with open('../../compare_modle/compare_data/KGNMDA/averaged_metrics_data.pkl', 'rb') as f:
    metrics_data_kgn = pickle.load(f)
plt.plot(metrics_data_kgn['fpr'], metrics_data_kgn['tpr'], label=f'KGNMDA (AUC = {metrics_data_kgn["roc_auc"]:.3f})', lw=line_width, color=colors[4])

# 完成绘图设置
plt.plot([0, 1], [0, 1], 'k--', label='Random', color='r', lw=line_width)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Comparison of ROC Curves Across Models')
plt.legend(loc="lower right")
plt.show()


plt.figure(figsize=(10, 8))

# 模型1: DSMDA
plt.plot(data_mvg['mean_recall'], data_mvg['mean_precisions'], label=f'DSMDA (AUPR = {data_mvg["mean_average_precision"]:.2f})', lw=line_width, color=colors[0])

# 模型2: DSAE_RF
with open('../../compare_modle/compare_data/DSAE_RF/pr_data.pkl', 'rb') as f:
    pr_data = pickle.load(f)
for i, (rec, pre) in enumerate(zip(pr_data['REC'], pr_data['PRE'])):
    aupr = metrics.auc(rec, pre)
    if i == 0:
        plt.plot(rec, pre, label=f'DSAE_RF (AUPR = {aupr:.4f})', lw=line_width, color=colors[1])

# 模型3: SAELGMDA
with open('../../compare_modle/compare_data/SAELGMDA/pr_data.pkl', 'rb') as f:
    pr_data = pickle.load(f)

REC = pr_data['REC']
PRE = pr_data['PRE']

prs = []
mean_recall = np.linspace(0, 1, 1000)
for i in range(len(REC)):
    prs.append(np.interp(mean_recall, REC[i][::-1], PRE[i][::-1])[::-1])
mean_precision = np.mean(prs, axis=0)
mean_aupr = metrics.auc(mean_recall, mean_precision)

all_recall = np.concatenate(REC)
all_precision = np.concatenate(PRE)
sort_order = np.argsort(all_recall)
sorted_recall = all_recall[sort_order]
sorted_precision = all_precision[sort_order]
precision_envelope = np.maximum.accumulate(sorted_precision[::-1])[::-1]

plt.plot(sorted_recall, precision_envelope, label=f'SAELGMDA (AUPR = {mean_aupr:.3f})', lw=line_width, color=colors[2])


# 模型4: GPUMDA
with open('../../compare_modle/compare_data/GPUDMDA/average_pr_data.pkl', 'rb') as f:
    pr_data_gpu = pickle.load(f)
plt.plot(pr_data_gpu['recall'], pr_data_gpu['precision'], label=f'GPUMDA (AUPR = {pr_data_gpu["aupr"]:.3f})', lw=line_width, color=colors[3])
# 模型5: KGNMDA
plt.plot(metrics_data_kgn['recall'], metrics_data_kgn['precision'], label=f'KGNMDA (AUPR = {metrics_data_kgn["pr_auc"]:.3f})', lw=line_width, color=colors[4])

# 完成绘图设置
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Comparison of Precision-Recall Curves Across Models')
plt.plot([0, 1], [1, 0], 'k--', label='Random', color='r', lw=line_width)
plt.legend(loc="lower left")
plt.show()