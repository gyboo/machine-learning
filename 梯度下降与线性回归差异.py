#1、波士顿预测房价 
#比较线性回归算法与随机梯度下降算法的差异
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,classification_report
from sklearn.cluster import KMeans
from time import *
#begin_time = time()
 
boston = load_boston()
#print(boston)
#线性回归算法
from sklearn.model_selection  import train_test_split
import numpy as np
X = boston.data
y = boston.target
# 数据分割


X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=666)

lr=LinearRegression()
#x_train=X_train.reshape(-1,1)
lr.fit(X_train,y_train)
print(lr.score(X_test,y_test))

y_lr_predict = lr.predict(X_test)
#mse_test = np.sum((y_lr_predict - y_test)**2) / len(y_test)
#print(mse_test)
print("lr的均方误差为：", mean_squared_error(y_test, y_lr_predict))
#print(y_lr_predict)

#梯度下降方法一：
#print(X_train[:1,])
scaler=StandardScaler().fit(X_train)
x_train = scaler.transform(X_train)
x_test = scaler.transform(X_test)
#print(x_train[:1,])
print(np.std(x_train)
sgd1_reg = SGDRegressor()
sgd1_reg.fit(x_train,y_train)
print(sgd1_reg.score(x_test, y_test))

'''
end_time = time()
run_time = end_time-begin_time
print(run_time)
'''

#梯度下降方法二：

sgd2_reg = SGDRegressor(n_iter_no_change=50)
sgd2_reg.fit(x_train,y_train)
print(sgd2_reg.score(x_test, y_test))



#梯度下降方法三：
std_x = StandardScaler()
std_y = StandardScaler()
 
x_train = std_x.fit_transform(X_train)
x_test = std_x.transform(X_test)
y_train = std_y.fit_transform(y_train.reshape(-1,1))
y_test = std_y.transform(y_test.reshape(-1,1))


sgd=SGDRegressor()
sgd.fit(x_train,y_train)

print(sgd.score(x_test,y_test))
#plt.scatter(x_train,y_train)
#plt.show()


y_sgd_predict = sgd.predict(x_test)
 
y_sgd_predict = std_y.inverse_transform(y_sgd_predict)

# 两种模型评估结果   数据量小使用lr, 大使用sgd
#mse_test = np.sum((y_lr_predict - y_test)**2) / len(y_test)
#print(mse_test)

 
print("SGD的均方误差为：", mean_squared_error(std_y.inverse_transform(y_test), y_sgd_predict))
 
#print("Ridge的均方误差为：", mean_squared_error(std_y.inverse_transform(y_test), y_rd_predict))'''
