

# # python处理

# In[1]:


import numpy as np
import pandas as pd


# In[4]:


dd=pd.read_csv("E:/Data/ddsim",index_col=0)
dd


# In[8]:


dname=pd.read_csv("E:/Data/dname.csv",index_col=0)
dname


# In[9]:


dd.columns=dname['Disease']
dd.index=dname['Disease']
# dd.columns=np.arange(217)
# dd.index=np.arange(217)
DS=dd.fillna(0)
DS


# In[10]:


DS.to_csv("E:/Data/DS.csv")#顺序为dname的


