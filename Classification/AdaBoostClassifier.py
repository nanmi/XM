import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import make_gaussian_quantiles

#生成数据集
# 生成2维正态分布，生成的数据按分位数分为两类，500个样本,2个样本特征，协方差系数为2
X1 , y1 = make_gaussian_quantiles(cov=2.0, n_samples=500, n_features=2, \
    n_classes=2, random_state=1)

# 生成2维正态分布，生成的数据按分位数分为两类，400个样本,2个样本特征均值都为3，协方差系数为2
X2 , y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5, n_samples=400, \
    n_features=2, n_classes=2, random_state=1)

X_train = np.concatenate((X1, X2))
y_train = np.concatenate((y1, 1 - y2))


abc = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5), \
    algorithm='SAMME', n_estimators=200, learning_rate=0.8)

abc.fit(X_train, y_train)

#可视化
X_min, X_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

XX, yy = np.meshgrid(np.arange(X_min, X_max, 0.02), \
    np.arange(y_min, y_max, 0.02))

Z_predict = abc.predict(np.c_[XX.ravel(), yy.ravel()]).reshape(XX.shape)

cs = plt.contourf(XX, yy, Z_predict, cmap=plt.cm.Paired)

plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train, edgecolors='black')
plt.show()

print('fit score(r2): ', abc.score(X_train, y_train))