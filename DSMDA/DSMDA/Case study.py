# Case_Study.py
# PyCharm
# -*-coding:utf-8-*-
import pandas as pd
import numpy as np

k = 20

def save_top_microbes(dataset_name, association_df, prob_matrix):
    # 确保排序不是在原矩阵上进行
    sorted_indices = np.argsort(-prob_matrix, axis=0)[:k, :]
    with open(f"./case study/{dataset_name}_top_microbes.txt", "w", encoding='utf-8') as file:
        for col_idx, disease in enumerate(association_df.columns):
            # 为当前疾病获取顶部微生物的索引
            top_microbes_indices = sorted_indices[:, col_idx]
            # 获取微生物名称
            top_microbes = association_df.index[top_microbes_indices].tolist()
            # 获取对应的概率，并排序
            top_probs = prob_matrix[top_microbes_indices, col_idx]
            microbes_with_probs = sorted(zip(top_microbes, top_probs), key=lambda x: x[1], reverse=True)
            # 只取微生物名称
            sorted_microbes = [microbe for microbe, _ in microbes_with_probs]
            # 写入文件
            file.write(f"{disease}: {', '.join(sorted_microbes)}\n")

# 加载数据集
Disbiome_prob = pd.read_csv('./result/Disbiome_prob_matrix_avg.csv', sep="\t",header=None)
HMDAD_prob = pd.read_csv('./result/HMDAD_prob_matrix_avg.csv', sep="\t",header=None)
phendb_prob = pd.read_csv('./result/peryton_prob_matrix_avg.csv', sep="\t",header=None)

Disbiome_mic_dis_association = pd.read_csv('./优化数据集/Disbiome/adj_matrix _with_name.csv', index_col=0)
HMDAD_mic_dis_association = pd.read_excel('./优化数据集/HMDAD/adj_mat_with_names.xlsx', index_col=0)
phendb_mic_dis_association = pd.read_csv('./优化数据集/peryton/adjacency_matrix.csv', index_col=0)



# 保存每个数据集中每个疾病的前20个微生物
save_top_microbes("Disbiome", Disbiome_mic_dis_association.T, Disbiome_prob.values)
save_top_microbes("HMDAD", HMDAD_mic_dis_association.T, HMDAD_prob.values)
save_top_microbes("peryton", phendb_mic_dis_association.T, phendb_prob.values)

k2 = 10  # 这次我们关注的是每个微生物的前10个疾病


def save_top_microbes_2(dataset_name, association_df, prob_matrix):
    # 确保排序不是在原矩阵上进行
    sorted_indices = np.argsort(-prob_matrix, axis=0)[:k, :]
    with open(f"./case study/{dataset_name}_top_disease.txt", "w", encoding='utf-8') as file:
        for col_idx, disease in enumerate(association_df.columns):
            # 为当前疾病获取顶部微生物的索引
            top_microbes_indices = sorted_indices[:, col_idx]
            # 获取微生物名称
            top_microbes = association_df.index[top_microbes_indices].tolist()
            # 获取对应的概率，并排序
            top_probs = prob_matrix[top_microbes_indices, col_idx]
            microbes_with_probs = sorted(zip(top_microbes, top_probs), key=lambda x: x[1], reverse=True)
            # 只取微生物名称
            sorted_microbes = [microbe for microbe, _ in microbes_with_probs]
            # 写入文件
            file.write(f"{disease}: {', '.join(sorted_microbes)}\n")

save_top_microbes_2("HMDAD", HMDAD_mic_dis_association, HMDAD_prob.T.values)