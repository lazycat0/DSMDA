#!/usr/bin/env python
# coding: utf-8

# # DATA

# In[1]:

import csv
import numpy as np
import pandas as pd
import math
import torch
import os
import random
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score 
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier


# In[2]:
matplotlib.use('TkAgg')

# functional similarity
def S_fun1(DDsim, T0, T1):
    DDsim = np.array(DDsim)
    T0_T1 = []
    if len(T0) != 0 and len(T1) != 0:
        for ti in T0:
            m_ax = []
            for tj in T1:
                m_ax.append(DDsim[ti][tj])
            T0_T1.append(max(m_ax))
    if len(T0) == 0 or len(T1) == 0:
        T0_T1.append(0)
    T1_T0 = []
    if len(T0) != 0 and len(T1) != 0:
        for tj in T1:
            m_ax = []
            for ti in T0:
                m_ax.append(DDsim[tj][ti])
            T1_T0.append(max(m_ax))
    if len(T0) == 0 or len(T1) == 0:
        T1_T0.append(0)
    return T0_T1, T1_T0

# 计算Fs
def FS_fun1(T0_T1, T1_T0, T0, T1):
    a = len(T1)
    b = len(T0)
    S1 = sum(T0_T1)
    S2 = sum(T1_T0)
    FS = []
    if a != 0 and b != 0:
        Fsim = (S1+S2)/(a+b)
        FS.append(Fsim)
    if a == 0 or b == 0:
        FS.append(0)
    return FS


# In[3]:


# Gaussian interaction profile kernel similarity
def r_func(MD):
    m = MD.shape[0]
    n = MD.shape[1]
    EUC_MD = np.linalg.norm(MD, ord=2, axis=1, keepdims=False)
    EUC_DL = np.linalg.norm(MD.T, ord=2, axis=1, keepdims=False)
    EUC_MD = EUC_MD**2
    EUC_DL = EUC_DL**2
    sum_EUC_MD = np.sum(EUC_MD)
    sum_EUC_DL = np.sum(EUC_DL)
    rl = 1 / ((1 / m) * sum_EUC_MD)
    rt = 1 / ((1 / n) * sum_EUC_DL)
    return rl, rt


def Gau_sim(MD, rl, rt):
    MD = np.mat(MD)
    DL = MD.T
    m = MD.shape[0]
    n = MD.shape[1]
    c = []
    d = []
    for i in range(m):
        for j in range(m):
            b_1 = MD[i] - MD[j]
            b_norm1 = np.linalg.norm(b_1, ord=None, axis=1, keepdims=False)
            b1 = b_norm1**2
            b1 = math.exp(-rl * b1)
            c.append(b1)
    for i in range(n):
        for j in range(n):
            b_2 = DL[i] - DL[j]
            b_norm2 = np.linalg.norm(b_2, ord=None, axis=1, keepdims=False)
            b2 = b_norm2**2
            b2 = math.exp(-rt * b2)
            d.append(b2)
    GMM = np.mat(c).reshape(m, m)
    GDD = np.mat(d).reshape(n, n)
    return GMM, GDD


# In[4]:


#cosine similarity
def cos_sim(MD):
    m = MD.shape[0]
    n = MD.shape[1]
    cos_MS1 = []
    cos_DS1 = []
    for i in range(m):
        for j in range(m):
            a = MD[i,:]
            b = MD[j,:]
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
            if a_norm!=0 and b_norm!=0:
                cos_ms = np.dot(a,b)/(a_norm * b_norm)
                cos_MS1.append(cos_ms)
            else:
                cos_MS1.append(0)
            
    for i in range(n):
        for j in range(n):
            a1 = MD[:,i]
            b1 = MD[:,j]
            a1_norm = np.linalg.norm(a1)
            b1_norm = np.linalg.norm(b1)
            if a1_norm!=0 and b1_norm!=0:
                cos_ds = np.dot(a1,b1)/(a1_norm * b1_norm)
                cos_DS1.append(cos_ds)  
            else:
                cos_DS1.append(0)
        
    cos_MS1 = np.array(cos_MS1).reshape(m, m)
    cos_DS1 = np.array(cos_DS1).reshape(n, n)
    return cos_MS1,cos_DS1


# In[5]:


#sigmoid function kernel similarity
def sig_kr(MD):
    m = MD.shape[0]
    n = MD.shape[1]
    sig_MS1 = []
    sig_DS1 = []
    for i in range(m):
        for j in range(m):
            a = MD[i,:]
            b = MD[j,:]
            z = (1/m)*(np.dot(a,b))
            sig_ms = math.tanh(z)
            sig_MS1.append(sig_ms)
            
    for i in range(n):
        for j in range(n):
            a1 = MD[:,i]
            b1 = MD[:,j]
            z1 = (1/n)*(np.dot(a1,b1))
            sig_ds = math.tanh(z1)
            sig_DS1.append(sig_ds)            
        
    sig_MS1 = np.array(sig_MS1).reshape(m, m)
    sig_DS1 = np.array(sig_DS1).reshape(n, n)
    return sig_MS1,sig_DS1    


# In[6]:


MD = pd.read_csv("../Data/MD_A.csv",index_col=0)
MD


# In[7]:


DS = pd.read_csv("../Data/DS.csv",index_col=0)
DS


# In[8]:


m = MD.shape[0]
T = []
for i in range(m):                                     # 对于每个微生物，找出与之相关的疾病（即 MD 矩阵中值为1的列索引），并将这些索引作为一个元组添加到列表 T 中。
    T.append(np.where(MD.iloc[i] == 1))
Fs = []
for ti in range(m):
    for tj in range(m):
        Ti_Tj, Tj_Ti = S_fun1(DS, T[ti][0], T[tj][0])
        FS_i_j = FS_fun1(Ti_Tj, Tj_Ti, T[ti][0], T[tj][0])   # 计算每对微生物之间的特征分数
        Fs.append(FS_i_j)
Fs = np.array(Fs).reshape(MD.shape[0], MD.shape[0])
Fs=pd.DataFrame(Fs)
for index,rows in Fs.iterrows():
    #for col,rows in Fs.iteritems():
    for col, rows in Fs.items():
        if index==col:
            Fs.loc[index,col]=1                             # 将对应的元素值设置为1。这表示每个微生物与自身的特征分数为1。
Fs


# In[9]:


rm, rt = r_func(MD)
GaM, GaD = Gau_sim(MD, rm, rt)


# In[10]:


GaM = pd.DataFrame(GaM)
GaM


# In[11]:


GaD = pd.DataFrame(GaD)
GaD


# In[12]:


MD_c = MD.copy()
MD_c.columns=range(0,MD.shape[1])
MD_c.index=range(0,MD.shape[0])
MD_c=np.array(MD_c)
MD_c


# In[13]:


cos_MS, cos_DS=cos_sim(MD_c)


# In[14]:


cos_MS=pd.DataFrame(cos_MS)
cos_MS


# In[15]:


cos_DS=pd.DataFrame(cos_DS)
cos_DS


# In[16]:


sig_MS,sig_DS = sig_kr(MD_c)


# In[17]:


sig_MS = pd.DataFrame(sig_MS)
sig_MS


# In[18]:


sig_DS = pd.DataFrame(sig_DS)
sig_DS


# # Multi-source features fusion

# In[19]:


MM = (Fs+GaM+cos_MS+sig_MS)/4
MM


# In[20]:


DS_t=DS.copy()
DS_t.columns=np.arange(DS.shape[0])
DS_t.index=np.arange(DS.shape[0])
DD = (DS_t+GaD+cos_DS+sig_DS)/4
DD


# In[21]:


MM.max().max()


# In[22]:


DD.max().max()


# # feature

# In[23]:


MM = np.array(MM)
DD = np.array(DD)


# In[24]:


EIG = []# feature matrix of total sample
for i in range(DD.shape[0]):
    for j in range(MM.shape[0]):
        eig = np.hstack((DD[i],MM[j]))#feature vector length :DD.shape[0]+MM.shape[0]
        EIG.append(eig)
#  EIG[i][j] The eigenvector of the sample (d, m), and the corresponding label matrix is DM.
EIG_t = np.array(EIG).reshape(DD.shape[0],MM.shape[0],DD.shape[0]+MM.shape[0])
EIG_t


# In[25]:


#Define random number seed
def setup_seed(seed):
    torch.manual_seed(seed)# 
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)#
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
     #cudnn.benchmark = False
     #cudnn.enabled = False


# In[26]:


DM_lable = MD_c.T
DM_lable


# In[27]:


# 
DM_lable = np.array(DM_lable)
lable = DM_lable.reshape(DM_lable.size).copy()  # 
lable


# In[28]:


PS_sub = np.where(lable==1)[0]#
PS_sub


# # Selecting negative samples by k_means clustering

# In[29]:


PS_num = len(np.where(lable==1)[0])# Positive sample number
NS1 = np.where(lable==0)[0]# 
# NS_sub = np.array(random.sample(list(NS1),PS_num))#
# NS_sub
NS1


# In[30]:


NS1.shape


# In[31]:


#Take the labels and feature vectors corresponding to all negative samples.
N_SAMPLE_lable = []#Label  corresponding to the sample
N_CHA=[]#The eigenvector matrix of the sample
for i in NS1:
    N_SAMPLE_lable.append(lable[i])
    N_CHA.append(EIG[i])
N_CHA = np.array(N_CHA)
print("N_SAMPLE_lable",N_SAMPLE_lable)
print("N_CHA",N_CHA)


# In[32]:


np.array(N_CHA).shape


# In[33]:


kmeans=KMeans(n_clusters=23, random_state=36).fit(N_CHA)
kmeans


# In[34]:


center=kmeans.cluster_centers_
center


# In[35]:


labels=kmeans.labels_
labels


# In[36]:


center.shape


# In[37]:


labels.shape


# In[38]:


# center_x=[]
# center_y=[]
# for j in range(len(center)):
#     center_x.append(center[j][0])
#     center_y.append(center[j][1])


# In[39]:


# setup_seed(36)


# In[40]:


type1=[]
type2=[]
type3=[]
type4=[]
type5=[]
type6=[]
type7=[]
type8=[]
type9=[]
type10=[]
type11=[]
type12=[]
type13=[]
type14=[]
type15=[]
type16=[]
type17=[]
type18=[]
type19=[]
type20=[]
type21=[]
type22=[]
type23=[]
for i in range(len(labels)):
    if labels[i]==0:
        type1.append(NS1[i])
    if labels[i]==1:
        type2.append(NS1[i])
    if labels[i]==2:
        type3.append(NS1[i])
    if labels[i]==3:
        type4.append(NS1[i])
    if labels[i]==4:
        type5.append(NS1[i])
    if labels[i]==5:
        type6.append(NS1[i])
    if labels[i]==6:
        type7.append(NS1[i])       
    if labels[i]==7:
        type8.append(NS1[i])        
    if labels[i]==8:
        type9.append(NS1[i])      
    if labels[i]==9:
        type10.append(NS1[i])        
    if labels[i]==10:
        type11.append(NS1[i])        
    if labels[i]==11:
        type12.append(NS1[i])       
    if labels[i]==12:
        type13.append(NS1[i])      
    if labels[i]==13:
        type14.append(NS1[i])        
    if labels[i]==14:
        type15.append(NS1[i])        
    if labels[i]==15:
        type16.append(NS1[i])       
    if labels[i]==16:
        type17.append(NS1[i])       
    if labels[i]==17:
        type18.append(NS1[i])        
    if labels[i]==18:
        type19.append(NS1[i])      
    if labels[i]==19:
        type20.append(NS1[i])     
    if labels[i]==20:
        type21.append(NS1[i])        
    if labels[i]==21:
        type22.append(NS1[i])      
    if labels[i]==22:
        type23.append(NS1[i])


# In[41]:


type23


# In[42]:


len(type23)


# In[43]:


len(type2)


# In[44]:


type=[type1,type2,type3,type4,type5,type6,type7,type8,type9,type10,type11,type12,type13,
      type14,type15,type16,type17,type18,type19,type20,type21,type22,type23] 
type


# In[45]:


setup_seed(36)


# In[46]:


# Select a negative sample equal to the positive sample.
type=[type1,type2,type3,type4,type5,type6,type7,type8,type9,type10,type11,type12,type13,
      type14,type15,type16,type17,type18,type19,type20,type21,type22,type23]                                       
mtype=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]                                                                     
for k in range(23):
    mtype[k]=random.sample(type[k],196) 
mtype


# In[47]:


len(mtype)


# In[48]:


# Negative sample subscript
NS_sub=[]
for i in range(len(lable)):
    for z2 in range(23):
        if i in mtype[z2]: 
            NS_sub.append(i)
NS_sub


# In[49]:


len(NS_sub)


# # Determine the total sample

# In[50]:


#
SAMPLE_sub = np.hstack((PS_sub,NS_sub))#
SAMPLE_sub


# In[51]:


#The label and feature vector corresponding to this sample
SAMPLE_lable = []#Labels 0 and 1 corresponding to samples
CHA=[]#The eigenvector matrix of the sample
for i in SAMPLE_sub:
    SAMPLE_lable.append(lable[i])
    CHA.append(EIG[i])
CHA = np.array(CHA)
print("SAMPLE_lable",SAMPLE_lable)
print("CHA",CHA)


# # DSAE

# In[5]:


# Define some global constants
BETA = math.pow(10,-7)
N_INP = CHA.shape[1]#Input dimension
N_HIDDEN = 1152#Hide layer dimension
N_EPOCHS = 150# Epoch times
use_sparse = True #Sparse or not


# In[6]:


#DSAE
class DSAE(nn.Module):
        def __init__(self):
            super(DSAE,self).__init__()
            #encoder
            self.encoder = nn.Sequential(
                nn.Linear(in_features=N_INP,out_features=N_HIDDEN),
                nn.Sigmoid(),
                nn.Linear(N_HIDDEN, int(N_HIDDEN/2)),
                nn.Sigmoid(),
                nn.Linear(int(N_HIDDEN/2), int(N_HIDDEN/4)),
                nn.Sigmoid(),
                nn.Linear(int(N_HIDDEN/4), int(N_HIDDEN/8)),
                nn.Sigmoid()
            )
        # decoder
            self.decoder = nn.Sequential(
                nn.Linear(int(N_HIDDEN/8), int(N_HIDDEN/4)),
                nn.Sigmoid(),
                nn.Linear(int(N_HIDDEN/4), int(N_HIDDEN/2)),
                nn.Sigmoid(),
                nn.Linear(int(N_HIDDEN/2), N_HIDDEN),
                nn.Sigmoid(),
                nn.Linear(N_HIDDEN, N_INP),
                nn.Sigmoid(),      
            )
            
        def forward(self, x):
            x = x.view(x.size(0),-1)
            encode = self.encoder(x)
            decode = self.decoder(encode)
            return encode, decode
net = DSAE()
net


# In[54]:


# Transform feature vectors into tensors
# T = np.array(T)
CHA_t = CHA.copy()
CHA_t = torch.from_numpy(CHA_t)
# CHA_t = torch.as_tensor(CHA_t)
# CHA = CHA.float()
CHA_t = CHA_t.float()#将数据转化float32
CHA_t


# In[55]:


CHA_t.shape


# In[56]:


# Define and save network functions.
def save(net, path):
    torch.save(net.state_dict(), path)
#     torch.save(net, path)


# In[57]:


# Define training network function, network, loss evaluation, optimizer and training set.
def train(net, trainloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    loss_rate = []
    lr_t = []#存储学习率
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)# Network parameters, learning rate
#     scheduler = lr_scheduler.StepLR(optimizer,step_size=5,gamma = 0.8)
#     scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10,30,50,70,90], gamma=0.5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    for epoch in range(N_EPOCHS):  #
        optimizer.zero_grad()  # 
            
            # forward + backward + optimize
#         encoded = net.encode(trainloader)
        encoded, decoded = net(trainloader)              #
#         print("encoded",encoded)
#         print("decoded",decoded)
        
       # loss
        inputs1 = trainloader.view(trainloader.size(0),-1)
        loss1 = criterion(decoded,inputs1)
#             print(loss1)
         
        if use_sparse:#
            kl_loss = torch.sum(torch.sum(trainloader*(torch.log(trainloader/decoded))+(1-trainloader)*(torch.log((1-trainloader)/(1-decoded)))))
#             p=trainloader
#             q=decoded
#             kl_loss = F.kl_div(q.log(),p,reduction="sum")+F.kl_div((1-q).log(),1-p,reduction="sum")
            loss = loss1+ BETA * kl_loss 
#             print(kl_loss)
#             print(loss1)
        else:
            loss = loss1  
        loss.backward()                   #
        optimizer.step()                #
            
            #
        scheduler.step(loss)
        print("[%d] loss: %.5f" % (epoch + 1, loss))
        lr = optimizer.param_groups[0]['lr']
        lr_t.append(lr)
        print("epoch={}, lr={}".format(epoch + 1, lr_t[-1]))
        loss_t = loss.clone()
        loss_rate.append(loss_t.cpu().detach().numpy())                
    x = list(range(len(lr_t)))
    plt.subplots(figsize=(10,6))
    plt.subplots_adjust(left=None,wspace=0.5)
    plt.subplot(121)
    plt.title('(a)',x=-0.2,y=1)
#     loss_rate = ['{:.5f}'.format(i) for i in loss_rate] 
    plt.plot(x,loss_rate,label='loss change curve',color="#f59164")
    plt.ylabel('loss changes')
    plt.xlabel('epoch')
    plt.legend()
    
    plt.subplot(122)
    plt.title('(b)',x=-0.2,y=1)
    plt.plot(x, lr_t, label = 'lr curve',color="#f59164")
    plt.ylabel('learning rate')
    plt.xlabel('epoch')
    plt.legend()
    #plt.savefig("E:/loss_lr.png",dpi=300)
    plt.show()
    print('Finished Training')


# In[58]:


net.train()#Training mode
train(net,CHA_t)


# In[59]:


path = '../Data/DSAE_RF.pth'
save(net, path)


# In[7]:


net.load_state_dict(torch.load( '../Data/DSAE_RF.pth'))


# In[8]:


torch.load('../Data/DSAE_RF.pth')



# # predict

# In[62]:


net.eval()    #test 
sample_extract=net(CHA_t)
sample_extract


# In[63]:


encoded = sample_extract[0]
decoded = sample_extract[1]

# Reserved features
SAMPLE_feature = encoded.detach().numpy()#
SAMPLE_feature


# In[64]:


SAMPLE_lable = np.array(SAMPLE_lable)# Sample label
SAMPLE_lable 


# In[81]:





# ## final result

# ## Ten-fold cross verification

# In[65]:

AUC = []
ACC = []
RECALL = []
PREECISION = []
AUPR = []
F1 = []

FPR = []
TPR = []
THR1 = []

THR2= []
PRE = []
REC = []

y_real = []
y_proba = []
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

kf = KFold(n_splits=10, shuffle=True, random_state=36)
for train_index, test_index in kf.split(SAMPLE_sub):
    
    train_features = SAMPLE_feature[train_index]
    test_features = SAMPLE_feature[test_index]
    train_labels = SAMPLE_lable[train_index]
    test_labels = SAMPLE_lable[test_index]
        #normalize
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)
#     print(SAMPLE_sub[train_index]) 
    rf = RandomForestClassifier(n_estimators=1100, criterion='entropy', max_depth=23,bootstrap=False,
                                min_samples_leaf=5,min_samples_split=6,
                                max_features='sqrt',random_state = 36).fit(train_features, train_labels)#Enter the training set and the label of the training set.

#     test_score1 = clf1.score(test_features, test_labels)#ACC
    test_predict = rf.predict(test_features)
    pre = rf.predict_proba(test_features)[:, 1]
    tru = test_labels
    pre2 = test_predict

    # auc
    auc = roc_auc_score(tru, pre)
    fpr, tpr, thresholds1 = metrics.roc_curve(tru, pre, pos_label=1)  # The actual value indicated as 1 is a positive sample.
    FPR.append(fpr)
    TPR.append(tpr)
    THR1.append(thresholds1)
    AUC.append(auc)
    print("auc:", auc)
    #ACC
    acc = accuracy_score(tru, pre2)
    ACC.append(acc)
    print("acc:",acc)
    # aupr
    precision, recall, thresholds2 = precision_recall_curve(tru, pre)
    aupr = metrics.auc(recall,precision)# 
    AUPR.append(aupr)
    THR2.append(thresholds2)
    PRE.append(precision)
    REC.append(recall)
    y_real.append(test_labels)
    y_proba.append(pre)    
    print("aupr:",aupr)
    # recall
    recall1 = metrics.recall_score(tru, pre2,average='macro')#
    RECALL.append(recall1)
    print("recall:",recall1)
	# precision
    precision1 = metrics.precision_score(tru, pre2,average='macro')
    print("precision:",precision1)
    PREECISION.append(precision1)
	#f1_score
    f1 = metrics.f1_score(tru, pre2,average='macro')#F1
    F1.append(f1)
    print("f1:",f1)

    with open(fold_metrics_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            [j , auc, aupr, f1_score(test_labels, test_predict), accuracy_score(test_labels, test_predict)])
print("AUC:", np.average(AUC))
print("ACC:", np.average(ACC))
print("AUPR:", np.average(AUPR))
print("RECALL:", np.average(RECALL))
print("PREECISION:", np.average(PREECISION))
print("F1:", np.average(F1))                                     


# ## Draw ROC and PR

# In[66]:


# ROC curve
plt.figure(figsize=(10,8))
tprs=[]
mean_fpr=np.linspace(0,1,1000)
for i in range(len(FPR)):
    tprs.append(np.interp(mean_fpr,FPR[i],TPR[i]))
    tprs[-1][0]=0.0
    auc = metrics.auc(FPR[i], TPR[i])
#     print(auc)
#     plt.xlim(0, 1)  # 
#     plt.ylim(0, 1)  # 
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.plot(FPR[i], TPR[i],lw=0.5,label='ROC fold %d(auc=%0.4f)' % (i, auc))
    plt.legend()  # 
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
# plt.plot([0,1],[0,1],linestyle='--',lw=2,color='r',label='Base Line')  # 
mean_auc = metrics.auc(mean_fpr, mean_tpr)  # 
std_auc = np.std(tprs, axis=0)
plt.plot(mean_fpr,mean_tpr,color="#D81C38",lw=2,label='Mean ROC (mean auc=%0.4f)' % (0.944790693551357))
# 
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
# 
plt.title("Receiver Operating Characteristic Curve")
plt.legend(bbox_to_anchor = (1.05, 0), loc=3, borderaxespad = 0)#
# plt.legend()  

plt.show()


# In[67]:


# PR curve
plt.figure(figsize=(10,8))
for i in range(len(REC)):
    aupr = metrics.auc(REC[i], PRE[i])
#     print(auc)
#     plt.xlim(0, 1)  # 
#     plt.ylim(0, 1)  #
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.plot(REC[i], PRE[i],lw=0.5,label='PR fold %d(aupr=%0.4f)' % (i, aupr))
    plt.legend()  #    

# plt.plot([0,1],[1,0],linestyle='--',lw=2,color='r',label='Base Line')  # 

y_real1 = np.concatenate(y_real)
y_proba1 = np.concatenate(y_proba)
precisions, recalls, _ = precision_recall_curve(y_real1, y_proba1)


plt.plot(recalls, precisions, lw=2,color="#D81C38", label='Mean PR (mean aupr=%0.4f)' % (0.943138100701337))
# 
plt.xlabel("Recall")
plt.ylabel("Precision")
# 
plt.title("Precision-Recall Curve")
plt.legend(bbox_to_anchor = (1.05, 0), loc=3, borderaxespad = 0)#
# plt.legend()  

plt.show()







# # Tune parameters

# ## Random search



# In[98]:


"""

# In[121]:

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV 
from bayes_opt import BayesianOptimization
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score



n_estimators = np.arange(100, 2000, step=100)
max_features = [ "sqrt", "log2"]
max_depth = list(np.arange(1, 100, step=2)) + [None]
min_samples_split = np.arange(2, 100, step=2)
min_samples_leaf = np.arange(1, 100, step=2)
bootstrap = [True, False]
criterion=["gini","entropy"]

param_grid = {
    "n_estimators": n_estimators,
    "max_features": max_features,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "bootstrap": bootstrap,
    "criterion":criterion,
}
param_grid


kf = KFold(n_splits=10, shuffle=True, random_state=36)
clf1 = RandomForestClassifier(random_state = 36)
random_search = RandomizedSearchCV(clf1, param_grid,return_train_score=True,cv=kf,scoring = 'roc_auc',n_iter=200,n_jobs=-1)
# SAMPLE_feature,SAMPLE_lable
scaler = StandardScaler()
train_features_r = scaler.fit_transform(SAMPLE_feature)
random_search.fit(train_features_r,SAMPLE_lable)

print("Best parameters: {}".format(random_search.best_params_))
print("Best cross-validation score: {:.4f}".format(random_search.best_score_))
print("Best estimator:\n{}".format(random_search.best_estimator_))

#HalvingGridSearchCV


from bayes_opt import BayesianOptimization
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

n_estimators = np.arange(800, 1300, step=100)
max_depth = list(np.arange(15, 30, step=2)) 
min_samples_split = np.arange(4, 8, step=1)
min_samples_leaf = np.arange(5, 9, step=1)

param_grid2 = {
    "n_estimators": n_estimators,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
}
param_grid2


kf = KFold(n_splits=10, shuffle=True, random_state=36)
clf2 = RandomForestClassifier(bootstrap=False,
                                max_features='sqrt',criterion='entropy',random_state = 36)
Halving_search = HalvingGridSearchCV(clf2, param_grid2,return_train_score=True,cv=kf,scoring = 'accuracy',n_jobs=-1)
# SAMPLE_feature,SAMPLE_lable
scaler = StandardScaler()
train_features_r = scaler.fit_transform(SAMPLE_feature)
Halving_search.fit(train_features_r,SAMPLE_lable)

print("Best parameters: {}".format(Halving_search.best_params_))
print("Best cross-validation score: {:.4f}".format(Halving_search.best_score_))
print("Best estimator:\n{}".format(Halving_search.best_estimator_))




"""