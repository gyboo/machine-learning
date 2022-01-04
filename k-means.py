#!/usr/bin/env python
# coding: utf-8

# In[60]:


from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA
filename='D:\datasets\mldata\seeds_dataset.csv' #地址 
names = ['area','perimeter','compactness','klength','width','asymmetry','kglength','kind']
re = read_csv(filename,names=names)   #读取数据
seed = np.array(re)  #转换为numpy数组
#print(seed.shape)
np.random.shuffle(seed)  #打乱顺序
X_train = seed[:180:,0:6]   #训练数据
X_test = seed[180:,0:6]
#print(X_train.shape)
#print(X_test.shape)
pca = PCA(0.95)
pca.fit(X_train)
X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)
#print(X_train_reduction.shape)
'''
#预测性能最好的簇数n_clusters
#第一种SSE 样本距离最近的聚类中心的距离总和 （簇内误差平方和）
#只对单个族中的数据分析，族与族之间的关系没有涉及
distortions = []
for i in range(1, 40):
        km = KMeans(n_clusters=i,
                    init='k-means++',
                    n_init=10,
                    max_iter=300,
                    random_state=0)
        km.fit(X_train_reduction)
        distortions.append(km.inertia_)

plt.plot(range(1,40), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

#第二种，轮廓系数
#考虑了族内族外量方面的因素，系数越大越好
scores = []
for i in range(2, 50):
    km = KMeans(        n_clusters=i,
                        init='k-means++',
                        n_init=10,
                        max_iter=300,
                        random_state=0      )
    km.fit(X_train_reduction)
    scores.append(metrics.silhouette_score(X_train_reduction, km.labels_ , metric='euclidean'))
plt.plot(range(2,50), scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('silhouette_score')
plt.show()


'''



kmeans = KMeans(n_clusters=2, random_state=10)  #定义三个质心，并拟合质心
kmeans.fit(X_train_reduction)
#print(seed[:,7])
#print(kmeans.score(X_test))
print("簇心={}".format(kmeans.cluster_centers_))  #打印簇心
print("测试结果：{}".format(kmeans.predict(X_test_reduction)))  #预测样本
print("CH分数={}".format(metrics.calinski_harabasz_score(X_train_reduction, kmeans.labels_))) 
print("轮廓系数={}".format(metrics.silhouette_score(X_train_reduction,kmeans.labels_,metric='euclidean'))) 

plt.scatter(X_train_reduction[:,0],X_train_reduction[:,1],c=kmeans.labels_)
plt.show()


# In[ ]:




