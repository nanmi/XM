import matplotlib.pyplot as plt

from sklearn.datasets import make_gaussian_quantiles

"""
Parameters
----------
mean : array of shape [n_features],可选（默认值=无）
	   多维正态分布的平均值。
	   如果没有，则使用原点（0，0，…）。

cov : float，可选（默认值为1。）
      协方差矩阵就是这个值乘以单位矩阵。
      这个数据集只生成对称正态分布。

n_samples : int，可选（默认值为100）
			平均分配给各个班级的总分。

n_features : int，可选（默认值为2）
             每个样本的特征数。

n_classes : int，可选（默认值为3）
			classes数量
			
shuffle : boolean, 可选(默认为True)
          将样本进行乱序。

random_state : int, RandomState instance or None （默认）
			   确定数据集创建的随机数生成。
			   传递int用于跨多个函数调用的可重复输出。

Returns
-------
X : array of shape [n_samples, n_features]
    生成的样本。

y : array of shape [n_samples]
    每个样本的分位数成员的整数标签。"""



X, y = make_gaussian_quantiles(n_samples=1000, n_features=2, n_classes=4)


plt.scatter(X[:, 0], X[:, 1], c=y, s=40, marker='o', edgecolors='black')
plt.show()