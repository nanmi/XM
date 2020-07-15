'''流形学习Isomap 
原理：将非欧几里德空间转换从欧几里德空间，将非欧几里得空间拆解成一个一个的欧几里得空间
MDS和Isomap都是保留全局特征的非线性数据降维算法，且出发点都是基于距离保持。
不同的是MDS是基于欧式距离，Isomap则是测地线距离
测地线距离：地球两个城市的距离无法使用两点之间直线最短的距离，只能依附地球表面的弧形来计算距离
根据邻近的点计算，超参数n_neighbors来设置邻近点的个数
'''
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import Isomap

from mpl_toolkits.mplot3d import Axes3D

# Load iris data
iris = datasets.load_iris()
X_train = iris.data
y_train = iris.target

# Create Isomap model
n_neighbors = 90
isomap = Isomap(n_components=3, n_neighbors=n_neighbors)
y_transform = isomap.fit_transform(X_train)


# Visualization
fig = plt.figure(figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)

ax.scatter(y_transform[:, 0], y_transform[:, 1], y_transform[:, 2], c=y_train,
           cmap=plt.cm.Set1, marker='o', edgecolor='black', s=40)

ax.set_title("Isomap Algorithm")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()