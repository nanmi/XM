import matplotlib.pyplot as plt

from sklearn.datasets import make_circles

"""
Parameters
----------
n_samples : int，可选（默认值为100）
		    生成的点的总数。
		    如果是奇数，则内圈比外圈多出一个点。
		    
shuffle : boolean，可选（默认值为True）
		  是否打乱样本。

noise : double or None (默认值为无)
        加在数据中的高斯噪声的标准差。

random_state : int，随机状态实例或无（默认）
			   确定数据集洗牌和噪声的随机数生成。
			   为跨多个函数调用的可复制输出传递int。

factor : 0 < double < 1 (默认值为.8)
        内外圆之间的比例因子。

Returns
-------
X : array of shape [n_samples, 2]
    生成的样本。

y : array of shape [n_samples]
    每个样本的类成员身份的整数标签（0或1）。"""

X, y = make_circles(n_samples=15000, shuffle=True,
                    noise=0.03, random_state=None, factor=0.6)


plt.scatter(X[:, 0], X[:, 1], c=y, s=40, marker='o', edgecolors='black')
plt.show()