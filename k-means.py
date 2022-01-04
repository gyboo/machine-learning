#!/usr/bin/env python
# coding: utf-8

# In[60]:


from pandas import read_csv
import numpy as np
from sklearn.cluster import KMeans
filename='D:\datasets\mldata\seeds_dataset.csv' #本机地址 
names = ['area','perimeter','compactness','klength','width','asymmetry','kglength','kind']
re = read_csv(filename,names=names)   #读取数据
seed = np.array(re)  #转换为numpy数组
print(seed.shape)
X = seed[::,0:6]   #训练数据
kmeans = KMeans(n_clusters=3, random_state=10).fit(X)  #定义三个质心，并拟合质心
#print(seed[:,7])
print(kmeans.labels_)   #标签 
print(kmeans.cluster_centers_)  #打印质心
print(X[1])
print(kmeans.predict([X[1]]))  #预测样本


# In[ ]:




