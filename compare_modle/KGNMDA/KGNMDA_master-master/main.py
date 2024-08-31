# -*- coding: utf-8 -*-

import os
import gc
import time

import numpy as np
from collections import defaultdict
from keras import backend as K
from keras import optimizers

from utils import load_data, pickle_load, format_filename, write_log
from models import KGCN
from config import ModelConfig, PROCESSED_DATA_DIR,  ENTITY_VOCAB_TEMPLATE, \
    RELATION_VOCAB_TEMPLATE, ADJ_ENTITY_TEMPLATE, ADJ_RELATION_TEMPLATE, LOG_DIR, PERFORMANCE_LOG, \
    MICROBE_SIMILARITY_FILE,DISEASE_SIMILARITY_FILE



os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def get_optimizer(op_type, learning_rate):
    if op_type == 'sgd':
        return optimizers.SGD(learning_rate)
    elif op_type == 'rmsprop':
        return optimizers.RMSprop(learning_rate)
    elif op_type == 'adagrad':
        return optimizers.Adagrad(learning_rate)
    elif op_type == 'adadelta':
        return optimizers.Adadelta(learning_rate)
    elif op_type == 'adam':
        return optimizers.Adam(learning_rate, clipnorm=5)
    else:
        raise ValueError('Optimizer Not Understood: {}'.format(op_type))
def loadsimilarity(similarity_file):
    term_id_similarity = {}
    with open(similarity_file,'r') as reader:
        for line in reader:
            term_id, term_similarity = line.strip().split(':')
            term_similarity_array = np.array(term_similarity.strip().split('\t'))
            term_id_similarity[int(term_id)] = term_similarity_array
    return term_id_similarity, len(term_similarity_array)
def generate_pre_embedding(matrix_row,matrix_column,data_dict):
    pre_embed = np.zeros((matrix_row,matrix_column),dtype='float64')
    for key1 in data_dict:
        for i in range(len(data_dict[key1])):
            pre_embed[key1][i] = data_dict[key1][i]
    return pre_embed
def train(train_d,test_d,kfold,dataset, neighbor_sample_size, embed_dim, n_depth, l2_weight, lr, optimizer_type,
          batch_size, aggregator_type, n_epoch, callbacks_to_add=None, overwrite=True):
    # 初始化模型配置
    config = ModelConfig()
    config.neighbor_sample_size = neighbor_sample_size
    config.embed_dim = embed_dim
    config.n_depth = n_depth
    config.l2_weight = l2_weight
    config.dataset=dataset
    config.K_Fold=kfold
    config.lr = lr
    config.optimizer = get_optimizer(optimizer_type, lr)
    config.batch_size = batch_size
    config.aggregator_type = aggregator_type
    config.n_epoch = n_epoch
    config.callbacks_to_add = callbacks_to_add

    # 加载处理后的数据
    config.entity_vocab_size = len(pickle_load(format_filename(PROCESSED_DATA_DIR,
                                                               ENTITY_VOCAB_TEMPLATE,
                                                               dataset=dataset)))#the size of entity_vocab   实体词汇表大小
    config.relation_vocab_size = len(pickle_load(format_filename(PROCESSED_DATA_DIR,
                                                                 RELATION_VOCAB_TEMPLATE,
                                                                 dataset=dataset)))#the size of relation_vocab 关系词汇表大小
    config.adj_entity = np.load(format_filename(PROCESSED_DATA_DIR, ADJ_ENTITY_TEMPLATE,
                                                dataset=dataset))#load adj_entity matrix  邻接实体矩阵
    config.adj_relation = np.load(format_filename(PROCESSED_DATA_DIR, ADJ_RELATION_TEMPLATE,
                                                 dataset=dataset))#load adj_relation matrix 邻接关系矩阵

    # 加载疾病相似性和微生物相似性矩阵，并生成预训练嵌入特征
    config.disease_similarity, disease_feature_dim = loadsimilarity(DISEASE_SIMILARITY_FILE[dataset])
    config.microbe_similarity, microbe_feature_dim = loadsimilarity(MICROBE_SIMILARITY_FILE[dataset])
    config.disease_pre_feature = generate_pre_embedding(config.entity_vocab_size,disease_feature_dim,config.disease_similarity)
    config.microbe_pre_feature = generate_pre_embedding(config.entity_vocab_size,microbe_feature_dim,config.microbe_similarity)
    print('config.disease_pre_feature.shape=',config.disease_pre_feature.shape)
    print('config.microbe_pre_feature.shape=',config.microbe_pre_feature.shape)
    # 构造实验名称
    config.exp_name = f'kgcn_{dataset}_neigh_{neighbor_sample_size}_embed_{embed_dim}_depth_' \
                      f'{n_depth}_agg_{aggregator_type}_optimizer_{optimizer_type}_lr_{lr}_' \
                      f'batch_size_{batch_size}_epoch_{n_epoch}'
    callback_str = '_' + '_'.join(config.callbacks_to_add)
    callback_str = callback_str.replace('_modelcheckpoint', '').replace('_earlystopping', '')
    config.exp_name += callback_str

    # 记录实验配置
    train_log = {'exp_name': config.exp_name, 'batch_size': batch_size, 'optimizer': optimizer_type,
                 'epoch': n_epoch, 'learning_rate': lr}
    print('Logging Info - Experiment: %s' % config.exp_name)
    model_save_path = os.path.join(config.checkpoint_dir, '{}.hdf5'.format(config.exp_name))         # 模型保存路径
    model = KGCN(config)                                                                            # 初始化KGCN模型

    train_data=np.array(train_d)
    test_data=np.array(test_d)
    if not os.path.exists(model_save_path) or overwrite:                                        # 如果模型未保存或选择覆盖现有模型，则训练模型
        start_time = time.time()                                                                # 记录训练开始时间

        model.fit(x_train=[train_data[:,:1],train_data[:,1:2]],y_train=train_data[:,2:3])       # 训练模型
        #model.fit(x_train=[train_data[:, :1], train_data[:, 1:2]], y_train=train_data[:, 2:3],
                 # x_valid=[valid_data[:, :1], valid_data[:, 1:2]], y_valid=valid_data[:, 2:3])
        elapsed_time = time.time() - start_time                                                     # 计算训练耗时
        print('Logging Info - Training time: %s' % time.strftime("%H:%M:%S",
                                                                 time.gmtime(elapsed_time)))
        train_log['train_time'] = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    # 评估模型性能
    print('Logging Info - Evaluate over test data:')
    model.load_best_model()
    # 在测试数据上评估模型，获取性能指标
    auc, acc, f1, aupr ,y_true,y_label= model.score(x=[test_data[:, :1], test_data[:, 1:2]], y=test_data[:, 2:3])
    # gain test_y_label and test_y_predict value
    test_y_label, test_y_prediction = model.gain_ytrue_ypredict(x=[test_data[:, :1], test_data[:, 1:2]], y=test_data[:, 2:3])
    print('test_y_label=',test_y_label)
    print('test_y_prediction=',test_y_prediction)

    label_score = open('Hmda_label_score.txt','w')
    for i in range(len(test_y_label)):
        label_score.write(str(test_y_label[i]))
        label_score.write('\t')
        label_score.write(str(test_y_prediction[i]))
        label_score.write('\n')
    label_score.close()
    
    train_log['test_auc'] = auc
    train_log['test_acc'] = acc
    train_log['test_f1'] = f1
    train_log['test_aupr'] =aupr
    print(f'Logging Info - test_auc: {auc}, test_acc: {acc}, test_f1: {f1}, test_aupr: {aupr}')
    if 'swa' in config.callbacks_to_add:                      # 如果配置了SWA回调，加载SWA模型并评估性能
        model.load_swa_model()
        print('Logging Info - Evaluate over test data based on swa model:')
        auc, acc, f1,aupr,y_true,y_label = model.score(x=[test_data[:, :1], test_data[:, 1:2]], y=test_data[:, 2:3])
        train_log['swa_test_auc'] = auc
        train_log['swa_test_acc'] = acc
        train_log['swa_test_f1'] = f1
        train_log['swa_test_aupr'] = aupr
        print(f'Logging Info - swa_test_auc: {auc}, swa_test_acc: {acc}, swa_test_f1: {f1}, swa_test_aupr: {aupr}')
    train_log['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())   # 记录实验时间戳
    write_log(format_filename(LOG_DIR, PERFORMANCE_LOG), log=train_log, mode='a')    # 将训练日志写入日志文件
    del model                                                                       # 清理模型和相关资源
    gc.collect()
    K.clear_session()
    return train_log,y_true,y_label

