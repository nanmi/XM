import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#线性回归模型
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

import sklearn.datasets as datasets

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score


boston = datasets.load_boston()
train = boston.data
target = boston.target

X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.2)

linear = LinearRegression()
ridge = Ridge()
lasso = Lasso()
elastic_net = ElasticNet()

##训练模型================
linear.fit(X_train, y_train)
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)
elastic_net.fit(X_train, y_train)

##数据预测==============
y_pre_linear = linear.predict(X_test)
y_pre_ridge = ridge.predict(X_test)
y_pre_lasso = lasso.predict(X_test)
y_pre_elasticnet = elastic_net.predict(X_test)

##评分==============
linear_score=r2_score(y_test, y_pre_linear)
ridge_score=r2_score(y_test, y_pre_ridge)
lasso_score=r2_score(y_test, y_pre_lasso)
elasticnet_score=r2_score(y_test, y_pre_elasticnet)
print(linear_score, ridge_score, lasso_score, elasticnet_score)

##绘图===========
plt.plot(y_test, c='r', label='true')

#Linear
plt.plot(y_pre_linear,label='linear')
plt.legend()

#Ridge
plt.plot(y_pre_ridge,label='ridge')
plt.legend()

#lasso
plt.plot(y_pre_lasso,label='lasso')
plt.legend()

#elasticnet
plt.plot(y_pre_elasticnet,label='elasticnet')
plt.legend()

plt.show()