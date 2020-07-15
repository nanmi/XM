import matplotlib.pyplot as plt

from sklearn.datasets import make_swiss_roll
from mpl_toolkits.mplot3d import Axes3D
"""
Parameter
---------
n_samples：int, optional (default=100)
		   S曲线上的采样点数。

noise：float, optional (default=0.0)
	   高斯噪声的标准偏差。

random_state：int, RandomState instance, default=None
		      确定用于创建数据集的随机数生成。
		      为多个函数调用传递可重复输出的int值。

Returns
-------
X：array of shape [n_samples, 3]
   数据点。

t：array of shape [n_samples]
   根据流形中点的主要尺寸，样本的单变量位置。

"""

X, y = make_swiss_roll(n_samples=2000, noise=0.1)


fig = plt.figure(figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y,
           cmap=plt.cm.Set1, marker='o', edgecolor='black', s=40)

plt.show()