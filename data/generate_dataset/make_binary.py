import matplotlib.pyplot as plt

from sklearn.datasets import make_hastie_10_2

"""
Parameters
----------
n_samples : int，可选（默认值=12000）
		    样本数。

random_state : int，RandomState instance or None（默认）
			   确定数据集创建的随机数生成。
			   为跨多个函数调用的可复制输出传递int。

Returns
-------
X : array of shape [n_samples, 10]
    输入样本。

y : array of shape [n_samples]
    输出值。
"""

X, y = make_hastie_10_2(n_samples=1000)


plt.scatter(X[:, 0], X[:, 1], c=y, s=40, marker='o', edgecolors='black')
plt.show()