import csv
import numpy as np
import pandas as pd
from data_input import  Neg_DataLoader, Non_Neg_DataLoader
from net import transNet
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import torch
from hyperparams import hyperparams as params
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, average_precision_score,
                             f1_score, accuracy_score, recall_score, precision_score, confusion_matrix)
import pickle
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

class CV():
    def __init__(self, cv, n_repeats, inc_matrix) -> None:      # 类的初始化方法
        self.cv = cv                                             # cv参数指定了交叉验证的类型
        self.inc_matrix = inc_matrix                            # 输入的矩阵，可能代表了特征或者其他数据
        self.n_repeats = n_repeats                              # 重复的次数，用于重复K折交叉验证
        self.i, self.j = inc_matrix.shape                       # 从输入矩阵获取其形状，即行数和列数
        self.trains, self.tests = [], []                        # 初始化训练集和测试集列表

    def bance(self, index, type="train"):                       # 用于平衡数据集的方法，根据索引和类型（训练或测试）进行数据处理
        if self.cv == 1:  # 行                                     如果cv参数为1，处理行
            t = self.inc_matrix.loc[index]  
        elif self.cv == 2:  # 列                                 # 如果cv参数为2，处理列
            t = self.inc_matrix.loc[:, index]
        elif self.cv == 3:  # hl                                # 如果cv参数为3，处理行和列
            inc = self.inc_matrix.stack().reset_index()
            inc = inc.loc[index]

        if self.cv == 1 or self.cv == 2:
            inc = t.stack().reset_index()
        # 分别获取值为1，0的索引

        if(type=="train"):

            # 对于训练数据，获取正负样本的索引
            s1 = inc[inc.loc[:, 0].values == 1].index
            s0 = inc[inc.loc[:, 0].values != 1].index

            # 重构正负样本的索引以形成二维数组
            s1 = np.vstack((inc.loc[s1, 'level_0'].values, (inc.loc[s1, 'level_1'].values))).T
            s0 = np.vstack((inc.loc[s0, 'level_0'].values, (inc.loc[s0, 'level_1'].values))).T
            s = np.vstack((s1, s0))

        if (type == "test"):

            # 对于测试数据，获取正负样本的索引
            s1 = inc[inc.loc[:, 0].values == 1].index
            s0 = inc[inc.loc[:, 0].values == 0].index

            # 重构正负样本的索引以形成二维数组
            s1 = np.vstack((inc.loc[s1, 'level_0'].values, (inc.loc[s1, 'level_1'].values))).T
            s0 = np.vstack((inc.loc[s0, 'level_0'].values, (inc.loc[s0, 'level_1'].values))).T
            s = np.vstack((s1, s0))

        # print(len(s1),len(s0),len(s))
        return s  # 返回[[行号，列号]]的二维ndarry  返回处理后的索引数组

    def cv_1(self):                                                              # 针对行进行的交叉验证方法
        # 行
        print('cv1 {}行'.format(self.i))                                         # 打印行数信息
        lens = self.i
        rkf = RepeatedKFold(n_splits=5, n_repeats=self.n_repeats)               # 使用重复的K折交叉验证
        for train_index, test_index in rkf.split(list(range(lens))):  
            self.trains.append(self.bance(train_index,"train"))                 # 处理训练数据
            self.tests.append(self.bance(test_index,"test"))                    # 处理测试数据
        return self.trains, self.tests

    def cv_2(self):                                                             # 针对列进行的交叉验证方法
        # 列
        print('cv2 {}列'.format(self.j))                                         # 打印列数信息
        lens = self.j
        rkf = RepeatedKFold(n_splits=5, n_repeats=self.n_repeats)                # 使用重复的K折交叉验证
        for train_index, test_index in rkf.split(list(range(lens))):
            self.trains.append(self.bance(train_index,"train"))                 # 处理训练数据
            self.tests.append(self.bance(test_index,"test"))                    # 处理测试数据
        return self.trains, self.tests

    def cv_3(self):                                                             # 针对行和列进行的交叉验证方法
        print('cv3')
        lens = self.i * self.j                                                  # 总的数据点数为行数乘以列数
        rkf = RepeatedKFold(n_splits=5, n_repeats=self.n_repeats)               # 使用重复的K折交叉验证
        for train_index, test_index in rkf.split(list(range(lens))):
            self.trains.append(self.bance(train_index,"train"))                  # 处理训练数据
            self.tests.append(self.bance(test_index,"test"))                    # 处理测试数据
        return self.trains, self.tests


    @classmethod
    def get_cv(cls, cv, n_repeats, inc_matrix):                                 # 类方法，用于获取交叉验证的训练和测试数据集

        if cv == 1:
            trains, tests = cls(cv, n_repeats, inc_matrix).cv_1()  
            trains, tests = cls(cv, n_repeats, inc_matrix).cv_2()
        elif cv == 3:
            trains, tests = cls(cv, n_repeats, inc_matrix).cv_3()

        return trains, tests




def get_data(data,index):                                   # 定义一个函数用来从数据中获取特征和标签
    dataset = []
    for i in range(index.shape[0]):
            # 对于每一个索引，将疾病特征、微生物特征和它们的相互作用拼接起来
            dataset.append(np.hstack((data.disease_feature[index[i,0]], data.microbe_feature[index[i,1]], data.interaction.iloc[index[i,0],index[i,1]])))
    reslut = pd.DataFrame(dataset).values                   # 将获取的数据转换为Pandas DataFrame并返回其值
    return  reslut


if __name__ == '__main__':
    # 加载数据
    data =  Neg_DataLoader("./trans_data/data1")
    # 获取相互作用数据集，并设置列名和索引
    dataset = data.interaction
    dataset.columns=list(range(data.interaction.shape[1]))
    dataset.index=list(range(data.interaction.shape[0]))
    # 初始化各种评价指标的列表
    n_acc = []
    n_precision = []
    n_recall = []
    n_f1 = []
    n_AUC = []
    n_AUPR = []

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

    trains, tests = CV.get_cv(cv=3,n_repeats= params.number ,inc_matrix=dataset)             # 获取交叉验证的训练和测试数据集
    for i in range(len(trains)):                                                             # 对每一组训练和测试数据
        print(len(trains), trains[i].shape, tests[i].shape)                                  # 打印训练和测试数据的数量
        train = get_data(data,trains[i])                                                     # 获取训练和测试数据的特征和标签
        test = get_data(data,tests[i])
        feature_train = train[:,0:-1]                                                       # 分离特征和标签
        target_train = train[:,-1].reshape(-1)
        feature_test = test[:,0:-1]
        target_test = test[:,-1].reshape(-1)
        # 开始训练
        print('begin training:')
        model = transNet( params.col_num, 100, 1).to(params.device)                         # 初始化模型、优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
        loss_fn = torch.nn.MSELoss().to(params.device)
        for epoch in range(params.epoch_num):                                               # 训练模型
            model.train()
            model.type = "train"
            feature_train = torch.FloatTensor(feature_train)
            target_train = torch.FloatTensor(target_train)
            train_x = feature_train.to(params.device)
            train_y = target_train.to(params.device)
            pred = model(train_x)
            loss = loss_fn(pred, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 50 == 0:
                print(loss.item())

        model.eval()                                                                     # 模型评估
        model.type = "test"
        feature_test = torch.FloatTensor(feature_test)
        target_test = torch.LongTensor(target_test)
        test_x = feature_test.to(params.device)
        test_y = target_test.to(params.device)
        pred = model(test_x)
        pred = pred.cuda().data.cpu().numpy()
        KT_y_prob_1 = np.arange(0, dtype=float)
        for i in pred:
            KT_y_prob_1 = np.append(KT_y_prob_1, i)
        light_y = []
        for i in KT_y_prob_1:  # 0 1
            if i > 0.5:
                light_y.append(1)
            else:
                light_y.append(0)
        #n_acc.append(accuracy_score(target_test, light_y))
        # 计算并保存评价指标
        metrics_summary['f1_scores'].append(f1_score(target_test, light_y))
        metrics_summary['accuracies'].append(accuracy_score(target_test, light_y))
        metrics_summary['recalls'].append(recall_score(target_test, light_y))
        metrics_summary['precisions'].append(precision_score(target_test, light_y))
        tn, fp, fn, tp = confusion_matrix(target_test, light_y).ravel()
        specificity = tn / (tn + fp)
        metrics_summary['specificities'].append(specificity)

        fpr_temp, tpr_temp, _ = roc_curve(target_test, light_y)
        roc_auc = auc(fpr_temp, tpr_temp)
        sk_fpr.append(fpr_temp)
        sk_tprs.append(tpr_temp)
        sk_aucs.append(roc_auc)

        precision, recall, _ = precision_recall_curve(target_test, light_y)
        pr_auc = auc(recall, precision)

        # 计算Precision-Recall曲线和AUPR
        precision_temp, recall_temp, _ = precision_recall_curve(target_test, light_y)
        average_precision = average_precision_score(target_test, light_y)
        sk_precisions.append(precision_temp)
        sk_recalls.append(recall_temp)
        sk_average_precisions.append(average_precision)

        with open(fold_metrics_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i + 1, roc_auc, pr_auc, f1_score(target_test, light_y),
                             accuracy_score(target_test, light_y)])
        print("accuracy:%.4f" % accuracy_score(target_test, light_y))
        print("precision:%.4f" % precision_score(target_test, light_y))
        print("recall:%.4f" % recall_score(target_test, light_y))
        print("F1 score:%.4f" % f1_score(target_test, light_y))


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

