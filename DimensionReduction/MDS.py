'''MDS降维（多维标度法）基于欧氏距离
MDS的原理就是保持新空间与原空间的相对位置关系不变
常用于市场调研、心理学数据分析
'''
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import MDS

from mpl_toolkits.mplot3d import Axes3D

# Load iris data
iris = datasets.load_iris()
X_train = iris.data
y_train = iris.target

# Create MDS model
mds = MDS(n_components=3, metric=True)
y_transform = mds.fit_transform(X_train)


# Visualization
fig = plt.figure(figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)

ax.scatter(y_transform[:, 0], y_transform[:, 1], y_transform[:, 2], c=y_train,
           cmap=plt.cm.Set1, marker='o', edgecolor='black', s=40)

ax.set_title("MDS Algorithm")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()