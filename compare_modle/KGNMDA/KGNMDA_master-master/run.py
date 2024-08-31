# -*- coding: utf-8 -*-
import sys
import random
import os
import numpy as np
import csv
from collections import defaultdict
sys.path.append(os.getcwd()) #add the env path
from sklearn.model_selection import train_test_split,StratifiedKFold
from main import train
from config import DISEASE_MICROBE_EXAMPLE, PROCESSED_DATA_DIR, LOG_DIR, MODEL_SAVED_DIR, ENTITY2ID_FILE, KG_FILE, \
    EXAMPLE_FILE, ENTITY_VOCAB_TEMPLATE, RESULT_LOG, MICROBE_SIMILARITY_FILE, DISEASE_SIMILARITY_FILE, \
    RELATION_VOCAB_TEMPLATE, SEPARATOR, TRAIN_DATA_TEMPLATE, \
    TEST_DATA_TEMPLATE, ADJ_ENTITY_TEMPLATE, ADJ_RELATION_TEMPLATE, ModelConfig, NEIGHBOR_SIZE
from utils import pickle_dump, format_filename,write_log,pickle_load
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, average_precision_score,
                             f1_score, accuracy_score, recall_score, precision_score, confusion_matrix)
import pickle
import matplotlib.pyplot as plt



def read_entity2id_file(file_path: str, entity_vocab: dict):
    print(f'Logging Info - Reading entity2id file: {file_path}' )
    assert len(entity_vocab) == 0                                   # 断言确保entity_vocab字典是空的，以便于开始时没有任何实体
    with open(file_path, encoding='utf8') as reader:
        count=0                                                      # 初始化计数器，用于跳过文件的第一行（通常是标题行）
        for line in reader:
            if(count==0):                                            # 如果是文件的第一行
                count+=1                                             # 跳过此行，不进行处理
                continue
            _, entity = line.strip().split('\t')
            entity_vocab[entity] = len(entity_vocab)#entity_vocab:{'0':0,...}        # 使用制表符分割每一行，并解包到_和entity变量，_用

def read_example_file(file_path:str,separator:str,entity_vocab:dict):
    print(f'Logging Info - Reading example file: {file_path}')
    assert len(entity_vocab)>0                                                       # 断言确保entity_vocab字典已经包含了实体，否则无法继续处理
    examples=[]
    with open(file_path,encoding='utf8') as reader:
        for idx,line in enumerate(reader):                                          # 遍历文件的每一行，使用enumerate获取行的索引idx和内容line
            d1,d2,flag=line.strip().split(separator)[:3]                            # 使用指定的分隔符分割每一行，并获取前三个元素
            if d1 not in entity_vocab or d2 not in entity_vocab:                    # 如果实体d1或d2不在entity_vocab中，则跳过当前行
                continue
            if d1 in entity_vocab and d2 in entity_vocab:
                examples.append([entity_vocab[d1],entity_vocab[d2],int(flag)])      # 如果实体d1和d2都在entity_vocab中
    
    examples_matrix=np.array(examples)
    print(f'size of example: {examples_matrix.shape}')

    return examples_matrix

def read_kg(file_path: str, entity_vocab: dict, relation_vocab: dict, neighbor_sample_size: int):
    print(f'Logging Info - Reading kg file: {file_path}')

    kg = defaultdict(list)                                      # 初始化一个默认字典，用于存储知识图谱中的实体及其相邻实体和关系
    with open(file_path, encoding='utf8') as reader:
        count=0
        for line in reader:
            if count==0:
                count+=1
                continue
           # head, tail, relation = line.strip().split(' ') 
            head, relation, tail = line.strip().split('\t')     # 使用制表符分割每一行，并获取头实体、关系和尾实体
            if head not in entity_vocab:                        # 如果头实体不在entity_vocab中 将头实体添加到entity_vocab中，并分配一个唯一编号
                entity_vocab[head] = len(entity_vocab)
            if tail not in entity_vocab:
                entity_vocab[tail] = len(entity_vocab)
            if relation not in relation_vocab:
                relation_vocab[relation] = len(relation_vocab)

            # undirected graph   无向图处理：将关系添加到知识图谱中，包括正向和反向
            kg[entity_vocab[head]].append((entity_vocab[tail], relation_vocab[relation]))   # 将尾实体和关系作为头实体的邻居添加
            kg[entity_vocab[tail]].append((entity_vocab[head], relation_vocab[relation]))   # 将头实体和关系作为尾实体的邻居添加
    print(f'Logging Info - num of entities: {len(entity_vocab)}, '
          f'num of relations: {len(relation_vocab)}')

    print('Logging Info - Constructing adjacency matrix...')
    n_entity = len(entity_vocab)
    adj_entity = np.zeros(shape=(n_entity, neighbor_sample_size), dtype=np.int64)       # 初始化邻接矩阵，用于存储每个实体的邻居实体和邻居关系
    adj_relation = np.zeros(shape=(n_entity, neighbor_sample_size), dtype=np.int64)     # 其中邻接矩阵的形状为实体总数 x 邻居采样大小，数据类型为int64
    random.seed(1)
    for entity_id in range(n_entity):                                                   # 遍历每个实体ID
        all_neighbors = kg[entity_id]                                                   # 获取当前实体的所有邻居（包括邻居实体和对应的关系）
        n_neighbor = len(all_neighbors)                                                 # 获取邻居的数量
        if n_neighbor > 0:                                                              #  如果存在邻居
            sample_indices = np.random.choice(
                n_neighbor,
                neighbor_sample_size,
                replace=False if n_neighbor >= neighbor_sample_size else True
            )       # 从邻居中随机选择一定数量的样本索引

            adj_entity[entity_id] = np.array([all_neighbors[i][0] for i in sample_indices])          # 根据选择的邻居索引获取邻接实体
            adj_relation[entity_id] = np.array([all_neighbors[i][1] for i in sample_indices])        # 根据选择的邻居索引获取邻接关系
    print('adj_entity=',adj_entity)
    return adj_entity, adj_relation
#gaussian similarity
def gaussian_similarity(interaction):
    inter_matrix = interaction
    nd = len(interaction)                   # 疾病数量
    nm = len(interaction[0])                # 微生物数量
    #generate disease similarity
    kd = np.zeros((nd, nd))                                             # 初始化一个疾病相似性矩阵，全零矩阵
    gama_d = nd / (pow(np.linalg.norm(inter_matrix),2))                 # 计算疾病相似性的γ参数
    d_matrix = np.dot(inter_matrix,inter_matrix.T)                      # 计算交互矩阵的转置乘以自身，得到疾病之间的相似度
    for i in range(nd):
        j = i
        while j < nd:
            kd[i,j] = np.exp(-gama_d * (d_matrix[i,i] + d_matrix[j,j] - 2*d_matrix[i,j]))   # 根据高斯核函数计算疾病相似度
            j += 1
    kd = kd + kd.T - np.diag(np.diag(kd))                                                   # 使得疾病相似性矩阵对称化
    #generate microbe similarity   生成微生物相似性
    km = np.zeros((nm,nm))
    gama_m = nm / (pow(np.linalg.norm(inter_matrix),2))
    m_matrix = np.dot(inter_matrix.T, inter_matrix)
    for l in range(nm):
        k = l
        while k < nm:
            km[l,k] = np.exp(-gama_m * (m_matrix[l,l] + m_matrix[k,k] - 2*m_matrix[l,k]))
            k += 1
    km = km + km.T - np.diag(np.diag(km))
    #print('kd=',kd,'km=',km)
    return kd,km
def generate_interaction(pairs_array):
    # first column:disease, second column:microbe, third column:0 or 1                               第一列：疾病，第二列：微生物，第三列：0 或 1
    first_term2id, second_term2id = generate_dict_id(pairs_array)                                    # 调用函数生成疾病和微生物的字典映射
    # interaction = np.zeros((disease_num, microbe_num))                                            交互矩阵 = np.zeros((疾病数量, 微生物数量))
    print('len(first_term2id)=',len(first_term2id),'len(second_term2id)=',len(second_term2id))      # 打印疾病和微生物的数量
    print('first_term2id=',first_term2id,'second_term2id=',second_term2id)                          # 打印疾病和微生物的字典映射
    interaction = np.zeros((len(first_term2id), len(second_term2id)))                               # 初始化交互矩阵，全零矩阵
    for i in range(len(pairs_array)):                                                               # 遍历每一个数据对
        if pairs_array[i,2] == 1:                                                                   # 如果数据对的第三列为1
            interaction[first_term2id[pairs_array[i,0]],second_term2id[pairs_array[i,1]]] = 1       # 根据数据对填充交互矩阵
    return interaction,first_term2id, second_term2id                                                # 返回交互矩阵和疾病微生物的字典映射
#generate dict id for disease and microbe
def generate_dict_id(approved_data):
    #approved_data[:,:1]:disease, approved_data[:,1:2]:microbe, approved_data[:,2:3]:label  生成疾病和微生物的字典映射
    first_term = set()                          # 存储疾病的集合
    second_term = set()                          # 存储微生物的集合
    for i in range(len(approved_data)):         # 遍历所有已批准的数据
        if approved_data[i,2] == 1:                                 # 如果数据的第三列为1，表示已批准
            first_term.add(approved_data[i, 0])                      # 将疾病加入集合中
            second_term.add(approved_data[i, 1])
    first_term2id = {}                                                # 存储疾病到ID的映射
    first_id = 0
    # first_termid = open('disease2id.txt','w')
    for term in first_term:                                         # 遍历疾病集合
        first_term2id[term] = first_id                              # 将疾病与ID进行映射
        # first_termid.write(str(term)+'\t'+str(first_id)+'\n')
        first_id += 1                                               # ID递增
    # first_termid.close()
    second_term2id = {}
    second_id = 0
    # second_termid = open('microbe2id.txt','w')
    for term in second_term:
        second_term2id[term] = second_id
        # second_termid.write(str(term)+'\t'+str(second_id)+'\n')
        second_id += 1
    # second_termid.close()
    return first_term2id, second_term2id

def generate_gaussian_file(all_data,test_data,disease_similarity_file,microbe_similarity_file):              # 生成高斯相似性文件
    interaction, disease_term2id, microbe_term2id = generate_interaction(np.array(all_data))                 # 生成交互矩阵和疾病、微生物的字典映射
    for i in range(len(test_data)):                                                                           # 遍历测试数据
        if test_data[i,2] == 1:                                                                              # 如果测试数据的第三列为1，表示已批准
            interaction[disease_term2id[test_data[i,0]],microbe_term2id[test_data[i,1]]] = 0                 # 将已批准的数据置为0，表示不考虑
    gaussian_d,gaussian_m = gaussian_similarity(interaction)                                                # 计算疾病和微生物的高斯相似性
    disease_id2term = {value:key for key,value in disease_term2id.items()}                                   # 将疾病ID和微生物ID映射回名称
    microbe_id2term = {value:key for key,value in microbe_term2id.items()}
    disease_similarity = open(disease_similarity_file,'w')                                                   # 写入疾病相似性文件
    for i in range(len(gaussian_d)):                                                                         # 遍历疾病相似性矩阵的行数
        disease_similarity.write(str(disease_id2term[i])+':')                                                 # 写入疾病名称
        for j in range(len(gaussian_d[i])):                                                                   # 遍历疾病相似性矩阵的列数
            if j != len(gaussian_d[i])-1:
                disease_similarity.write(str(gaussian_d[i][j])+'\t')                                        # 写入相似性值和制表符
            if j == len(gaussian_d[i])-1:
                disease_similarity.write(str(gaussian_d[i][j])+'\n')
    disease_similarity.close()
    microbe_similarity = open(microbe_similarity_file,'w')                                                   # 写入微生物相似性文件
    for i in range(len(gaussian_m)):                                                                         # 遍历微生物相似性矩阵的行数
        microbe_similarity.write(str(microbe_id2term[i])+':')
        for j in range(len(gaussian_m[i])):
            if j != len(gaussian_m[i])-1:
                microbe_similarity.write(str(gaussian_m[i][j])+'\t')
            if j == len(gaussian_m[i])-1:
                microbe_similarity.write(str(gaussian_m[i][j])+'\n')
    microbe_similarity.close()
    return 0

def process_data(dataset: str, neighbor_sample_size: int,K:int):

    entity_vocab = {}                                                               # 初始化实体 关系词汇表
    relation_vocab = {}

    read_entity2id_file(ENTITY2ID_FILE[dataset], entity_vocab)                     # 读取实体ID文件并更新实体词汇表

    # 将实体词汇表序列化并保存到指定的处理后数据目录
    pickle_dump(format_filename(PROCESSED_DATA_DIR, ENTITY_VOCAB_TEMPLATE, dataset=dataset),entity_vocab)#save entity_vocab

    examples_file=format_filename(PROCESSED_DATA_DIR, DISEASE_MICROBE_EXAMPLE, dataset=dataset) # 构造示例文件路径
    examples = read_example_file(EXAMPLE_FILE[dataset], SEPARATOR[dataset],entity_vocab)         # 读取示例文件
    np.save(examples_file,examples)#save examples  保存示例数据为numpy数组

    # 读取知识图谱文件，并返回邻接实体和邻接关系矩阵
    adj_entity, adj_relation = read_kg(KG_FILE[dataset], entity_vocab, relation_vocab,
                                       neighbor_sample_size)

    # 将实体词汇表和关系词汇表序列化并保存
    pickle_dump(format_filename(PROCESSED_DATA_DIR, ENTITY_VOCAB_TEMPLATE, dataset=dataset),
                entity_vocab)#save entity_vocab
    pickle_dump(format_filename(PROCESSED_DATA_DIR, RELATION_VOCAB_TEMPLATE, dataset=dataset),
                relation_vocab)#save relation_vocab
    # 保存邻接实体矩阵
    adj_entity_file = format_filename(PROCESSED_DATA_DIR, ADJ_ENTITY_TEMPLATE, dataset=dataset)
    np.save(adj_entity_file, adj_entity)#save adj_entity
    print('Logging Info - Saved:', adj_entity_file)
    
    disease_similarity_file = DISEASE_SIMILARITY_FILE[dataset]      # 定义疾病相似性和微生物相似性文件路径
    microbe_similarity_file = MICROBE_SIMILARITY_FILE[dataset]
    # test_disease_similarity_file = TEST_DISEASE_SIMILARITY_FILE[dataset]
    # test_microbe_similarity_file = TEST_MICROBE_SIMILARITY_FILE[dataset]
    adj_relation_file = format_filename(PROCESSED_DATA_DIR, ADJ_RELATION_TEMPLATE, dataset=dataset) # 构造邻接关系文件路径
    np.save(adj_relation_file, adj_relation)#save adj_relation                               # 保存邻接关系矩阵
    print('Logging Info - Saved:', adj_entity_file)                                         # 打印日志信息，标识已保存的文件
    number_train = 10                                                                       # 初始化交叉验证相关的变量
    cv_total_auc = 0                                                                            # 累计AUC值
    cv_total_aupr = 0
    cvs2_total_auc = 0
    cvs2_total_aupr = 0
    cvs3_total_auc = 0
    cvs3_total_aupr = 0
    for i in range(number_train):                                                           # 进行K折交叉验证
        cv_auc, cv_aupr= cross_validation(K,examples,dataset,neighbor_sample_size, \
                                          disease_similarity_file,microbe_similarity_file)

        cv_total_auc += cv_auc
        cv_total_aupr += cv_aupr

    cv_average_auc = cv_total_auc / number_train
    cv_average_aupr = cv_total_aupr / number_train

    print(f'This is {K}_fold cv')
    print('cv_average_auc=',cv_average_auc,'cv_average_aupr=',cv_average_aupr)
    return 0


def cross_validation(K_fold,examples,dataset,neighbor_sample_size,disease_similarity_file,microbe_similarity_file):#self.K_Fold=1 do cross-validation
    subsets=dict()                                                                                          # 用于存储每一折的数据索引
    n_subsets=int(len(examples)/K_fold)                                                                     # 计算每一折应有的样本数
    remain=set(range(0,len(examples)))#examples:drug_vocab[d1] drug_vocab[d2] int(flag)(0 or 1)             # 初始化一个包含所有样本索引的集合
    for i in reversed(range(0,K_fold-1)):                                                                   # 将数据集分成K-1个子集
        subsets[i]=random.sample(remain,n_subsets)                                                          # 随机选取n_subsets个样本作为一个子集
        remain=remain.difference(subsets[i])                                                                # 更新剩余样本集合
    subsets[K_fold-1]=remain                                                                                # 最后一折使用剩余的所有样本
    #aggregator_types = ['sum_concat']
    aggregator_types = ['concat']                                                                           # 本例中使用concat聚合类型
    print('aggregator_types=concat')
    #aggregator_types=['sum','concat','neigh']
    for t in aggregator_types:
        count=1                                                                                            # 初始化计数器，用于标记当前的交叉验证折数
        temp={'dataset':dataset,'aggregator_type':t,'avg_auc':0.0,'avg_acc':0.0,'avg_f1':0.0,'avg_aupr':0.0}    # 初始化用于存储交叉验证结果的字典

        sk_tprs = []
        sk_aucs = []
        sk_precisions = []
        sk_recalls = []
        sk_average_precisions = []
        sk_fpr = []

        metrics_summary = {
            'f1_scores': [],
            'accuracies': [],
            'recalls': [],
            'specificities': [],
            'precisions': []
        }

        fold_metrics = {
            'aucs': [],
            'auprs': [],
            'f1_scores': [],
            'accuracies': []
        }

        fold_metrics_file = 'fold_metrics.csv'
        with open(fold_metrics_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Fold', 'AUC', 'AUPR', 'F1', 'Accuracy'])
        my_count = 0
        for i in reversed(range(0,K_fold)):                                                                 # 对每折数据进行训练和测试
            test_data=examples[list(subsets[i])]
            #val_d,test_data=train_test_split(test_d,test_size=0.5)
            train_d=[]
            for j in range(0,K_fold):
                if i!=j:
                    train_d.extend(examples[list(subsets[j])])
            train_data=np.array(train_d)
            #generate_gaussian_file(train_data,disease_similarity_file,microbe_similarity_file)
            generate_gaussian_file(examples,test_data,disease_similarity_file,microbe_similarity_file)
            print('This is cross-validation S1.')
            print('len(train_data=)',len(train_data))
            print('len(test_data=)',len(test_data))
            train_log,y_true,y_label=train(
            kfold=count,
            dataset=dataset,
            train_d=train_data,
            test_d=test_data,
            neighbor_sample_size=neighbor_sample_size,
            embed_dim=32,
            n_depth=2,
            #n_depth=4,
            #l2_weight=5e-3,
            l2_weight=1e-1,
            lr=1e-3,
            #lr=1e-1,
            optimizer_type='adam',
            batch_size=32,
            aggregator_type=t,
            n_epoch=50,
            callbacks_to_add=['modelcheckpoint', 'earlystopping']
            )#train          # 调用train函数进行训练，并返回训练日志
            count+=1         # 更新交叉验证结果
            temp['avg_auc']=temp['avg_auc']+train_log['test_auc']
            temp['avg_acc']=temp['avg_acc']+train_log['test_acc']
            temp['avg_f1']=temp['avg_f1']+train_log['test_f1']
            temp['avg_aupr']=temp['avg_aupr']+train_log['test_aupr']
        for key in temp:         # 计算每个评估指标的平均值
            if key=='aggregator_type' or key=='dataset':
                continue


            temp[key]=temp[key]/K_fold
            metrics_summary['f1_scores'].append(f1_score(y_true, y_label))
            metrics_summary['accuracies'].append(accuracy_score(y_true, y_label))
            metrics_summary['recalls'].append(recall_score(y_true, y_label))
            metrics_summary['precisions'].append(precision_score(y_true, y_label))
            tn, fp, fn, tp = confusion_matrix(y_true, y_label).ravel()
            specificity = tn / (tn + fp)
            metrics_summary['specificities'].append(specificity)

            fpr_temp, tpr_temp, _ = roc_curve(y_true, y_label)
            roc_auc = auc(fpr_temp, tpr_temp)
            sk_fpr.append(fpr_temp)
            sk_tprs.append(tpr_temp)
            sk_aucs.append(roc_auc)

            # 计算Precision-Recall曲线和AUPR
            precision_temp, recall_temp, _ = precision_recall_curve(y_true, y_label)
            average_precision = average_precision_score(y_true, y_label)
            sk_precisions.append(precision_temp)
            sk_recalls.append(recall_temp)
            sk_average_precisions.append(average_precision)

            precision, recall, _ = precision_recall_curve(y_true, y_label)
            pr_auc = auc(recall, precision)

            with open(fold_metrics_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                my_count = my_count+1
                writer.writerow([my_count, roc_auc, pr_auc, f1_score(y_true, y_label),
                                 accuracy_score(y_true, y_label)])

        print('############ avg score #############')
        for metric, values in metrics_summary.items():
            print(f"{metric}: {np.mean(values):.2f} ± {np.std(values):.2f}")

        #############################################################   保存画图的数据   ########################################################
        print('mean AUC = ', np.mean(sk_aucs))
        print('mean AUPR = ', np.mean(sk_average_precisions))

        mean_fpr = np.linspace(0, 1, 100)
        mean_recall = np.linspace(0, 1, 100)
        tprs = []
        precisions = []
        for fpr_temp, tpr_temp in zip(sk_fpr, sk_tprs):
            interp_tpr = np.interp(mean_fpr, fpr_temp, tpr_temp)
            interp_tpr[0] = 0.0  # 确保曲线从 0 开始
            tprs.append(interp_tpr)
        mean_tpr = np.mean(tprs, axis=0)

        mean_tpr[-1] = 1.0  # 确保曲线以 1 结束

        for recall_temp, precision_temp in zip(sk_recalls, sk_precisions):
            interp_precision = np.interp(mean_recall, recall_temp[::-1], precision_temp[::-1])
            precisions.append(interp_precision)

        mean_precision = np.mean(precisions, axis=0)
        sk_aucs = np.mean(sk_aucs)
        sk_average_precisions = np.mean(sk_average_precisions)

        roc_data = {
            'fpr': mean_fpr,
            'tprs': mean_tpr,
            'aucs': sk_aucs
        }

        # Precision-Recall曲线数据
        pr_data = {
            'recalls': mean_recall,
            'precisions': mean_precision,
            'average_precisions': sk_average_precisions
        }

        # 保存ROC数据
        with open('roc_data.pkl', 'wb') as f:
            pickle.dump(roc_data, f)
        # 保存Precision-Recall数据
        with open('pr_data.pkl', 'wb') as f:
            pickle.dump(pr_data, f)

        # 加载ROC数据
        with open('roc_data.pkl', 'rb') as f:
            roc_data = pickle.load(f)
        # 加载Precision-Recall数据
        with open('pr_data.pkl', 'rb') as f:
            pr_data = pickle.load(f)

        # 绘制ROC曲线
        fig2, axs2 = plt.subplots(1, 1, figsize=(5, 5))

        axs2.plot(roc_data['fpr'], roc_data['tprs'], label=f" AUC = {roc_data['aucs']:.2f}")
        axs2.plot([0, 1], [0, 1], 'k--', label='Random', color='r')
        axs2.set_xlim([-0.05, 1.05])
        axs2.set_ylim([0.0, 1.05])
        axs2.set_xlabel('False Positive Rate')
        axs2.set_ylabel('True Positive Rate')
        axs2.set_title('ROC')
        axs2.legend(loc="lower right")
        plt.show()

        # 绘制Precision-Recall曲线
        fig3, axs3 = plt.subplots(1, 1, figsize=(5, 5))

        axs3.plot(pr_data['recalls'], pr_data['precisions'], label=f" AUPR = {pr_data['average_precisions']:.2f}")
        axs3.plot([0, 1], [1, 0], 'k--', label='Random', color='r')
        axs3.set_xlim([-0.05, 1.05])
        axs3.set_ylim([0.0, 1.05])
        axs3.set_xlabel('Recall')
        axs3.set_ylabel('Precision')
        axs3.set_title('Precision-Recall Curvesin')
        axs3.legend(loc="lower left")
        plt.show()

        print("模型指标和曲线数据已保存。")

        write_log(format_filename(LOG_DIR, RESULT_LOG[dataset]),temp,'a')           # 将交叉验证结果写入日志
        print(f'Logging Info - {K_fold} fold result: avg_auc: {temp["avg_auc"]}, avg_acc: {temp["avg_acc"]}, avg_f1: {temp["avg_f1"]}, avg_aupr: {temp["avg_aupr"]}')
    return temp['avg_auc'], temp['avg_aupr']             # 返回平均AUC和AUPR值

if __name__ == '__main__':
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(MODEL_SAVED_DIR):
        os.makedirs(MODEL_SAVED_DIR)
    model_config = ModelConfig()
    process_data('mdkg_hmdad',NEIGHBOR_SIZE['mdkg_hmdad'],5)



