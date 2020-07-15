import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

"""
Parameters
----------
n_samples :int或类似数组，可选（默认值为100）如果是int，则为平均除以集群。
		   如果数组相似，则序列的每个元素指示每个群集的样本数。

n_features : int，可选（默认值为2）
			 每个样本的特征数。

centers : 形状的整数或数组[n_中心，n_特征]，可选（默认设置为无）
		  要生成的中心数或固定中心位置。
		  如果n_samples为int，centers为None，则生成3个中心。    			
		  如果n_样本是数组形式，则中心必须是无或长度等于n个样本长度的数组。

cluster_std : 浮动或浮动序列，可选（默认值=1.0）
		      簇的标准差。

center_box : 一对浮动（最小，最大），可选（默认值=（-10.0，10.0））
		     随机生成中心时每个群集中心的边界框。

shuffle : boolean，可选（默认值为True）
		  洗牌样本（应该是将样本乱序的意思）。

random_state : int，随机状态实例或无（默认）
			   确定数据集创建的随机数生成。
			   为跨多个函数调用的可复制输出传递int。

Returns
-------
X : array of shape [n_samples, n_features]
    生成的样本。

y : array of shape [n_samples]
    每个样本的群集成员身份的整数标签。"""
#生成符合正态分布的聚类数据
X, y = make_blobs(n_samples=1500, n_features=2, centers=5)


plt.scatter(X[:, 0], X[:, 1], c=y, s=40, marker='o', edgecolors='black')
plt.show()