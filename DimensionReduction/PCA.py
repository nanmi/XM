import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D

# Load iris data
iris = datasets.load_iris()
X_train = iris.data
y_train = iris.target

# Create PCA model
pca = PCA(n_components=3)
#下面等价于y_predict = pca.fit(X_train).transform(X_train)
y_transform = pca.fit_transform(X_train)


# Visualization
# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)

ax.scatter(y_transform[:, 0], y_transform[:, 1], y_transform[:, 2], c=y_train,
           cmap=plt.cm.Set1, marker='o', edgecolor='black', s=40)

ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()
