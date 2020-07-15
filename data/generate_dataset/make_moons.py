import matplotlib.pyplot as plt

from sklearn.datasets import make_moons

"""
Parameters
----------
n_samples : int，可选（默认值为100）
			生成的点的总数。

shuffle : boolean, 可选（默认值为True）
		  是否将样本乱序。

noise : double or None（默认值为None）
		加在数据中的高斯噪声的标准差。

random_state : int, RandomState instance or None （默认）
			   确定数据集洗牌和噪声的随机数生成。
			   为跨多个函数调用的可复制输出传递int。

Returns
-------
X : array of shape [n_samples, 2]
    生成的样本。

y : array of shape [n_samples]
    每个样本的类成员身份的整数标签（0或1）。"""



X, y = make_moons(n_samples=1500, shuffle=True,
                  noise=0.06, random_state=None)


plt.scatter(X[:, 0], X[:, 1], c=y, s=40, marker='o', edgecolors='black')
plt.show()