import matplotlib.pyplot as plt

from sklearn.datasets import make_classification

"""
Parameters
----------
n_samples :int，可选（默认值为100）
 		   样本数。

n_features : int，可选（默认值为20）
		     功能的总数。其中包括
		     ``n_informative``信息功能
	         ``n_redundant`` 冗余功能,
        	 ``n_repeated`` 重复特征
        	 ``n_features-n_informative-n_redundant-n_repeated`` 随机绘制的无用特征.

n_informative : int，可选（默认值=2）
			    信息功能的数量。
			    每一类都由若干个高斯簇组成，
			    每个高斯簇都位于多维信息子空间中的超立方体顶点周围。
				对于每个聚类，信息特征独立于N（0，1）提取，
				然后在每个聚类内随机线性组合以添加协方差。
				然后将这些簇放置在超立方体的顶点上。
				
n_redundant : int，可选（默认值为2）
			  冗余功能的数量。
			  这些功能生成为信息特征的随机线性组合。

n_repeated : int，可选（默认值为0）
			 从信息性特征和冗余特征中随机抽取的重复特征数。

n_classes : int，可选（默认值为2）
		    分类问题的类（或标签）数。

n_clusters_per_class : int，可选（默认值为2）
				 	   每个类的群集数。

weights : list of floats or None（默认值为无）
		  分配给每个类的样本比例。
		  如果没有，则类是平衡的。
		  注意，如果“len（weights）==n_classes-1``，
		  则自动推断最后一个类的权重。
		  如果“权重”之和超过1，则可能返回超过“n个样本”的样本。

flip_y : float，可选（默认值=0.01）
		 其类是随机交换的样本的分数。
		 较大的值会在标签中引入噪声，并使分类任务更加困难。

class_sep : 浮点，可选（默认值为1.0）
			乘以超立方体大小的因子。
			更大的值分布、分布集群/类，使分类任务更容易。

hypercube : boolean, 可选（默认值为True）
			如果为True，则将簇放置在超立方体的顶点上。
			如果为False，则将簇放置在随机多面体的顶点上。

shift : float, 可选（默认值为1.0）
		乘以超立方体大小的因子。
		更大的值分布、分布集群/类，使分类任务更容易。

scale : float, array of shape [n_features] or None, 可选（默认值为1.0）
		将特征乘以指定值。
		如果没有，则特征按[1100]中的随机值缩放。
		注意，缩放发生在移位之后。

shuffle : boolean, 可选（默认值为True）
		  将样本和标签进行乱序。

random_state : int, RandomState instance or None (default)
        	   确定数据集创建的随机数生成。
       		   为跨多个函数调用的可复制输出传递int。

Returns
-------
X : array of shape [n_samples, n_features]
    生成的样本。

y : array of shape [n_samples]
    每个samp类成员的整数标签。"""


X, y = make_classification(n_samples=500, n_features=20, n_informative=2,
                           n_redundant=2, n_repeated=0, n_classes=2,
                           n_clusters_per_class=2, weights=None,
                           flip_y=0.01, class_sep=1.0, hypercube=True,
                           shift=0.0, scale=1.0, shuffle=True, random_state=None)


plt.scatter(X[:, 0], X[:, 1], c=y, s=40, marker='o', edgecolors='black')
plt.show()