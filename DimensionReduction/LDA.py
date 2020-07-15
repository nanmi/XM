'''线性判别分析LDA(Linear Discriminant Analysis)
核心思想就是投影后类内方差(Qe)最小，类间方差(Qa)最大，
它广泛用于模式识别领域(人脸识别)，这个LDA要区别
与自然语言处理领域的LDA(Linear DIrihlet Allocation)
'''
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from mpl_toolkits.mplot3d import Axes3D

# Load iris data
iris = datasets.load_iris()
X_train = iris.data
y_train = iris.target

# Create LDA model
lda = LinearDiscriminantAnalysis(n_components=3)
y_transform = lda.fit(X_train, y_train).transform(X_train)

print("降维后主成分数：",y_transform.shape)

'''我们可以发现并不是按照我设置的维度降维的，而是直接降到了2维。
这是因为LDA1的n_component需满足1≤n_components≤n_classes-1的情况，而这里n_classes为3，
因此n_component无论取3还是4都是降维到2。
'''
# Visualization
# fig = plt.figure(figsize=(8, 6))
# ax = Axes3D(fig, elev=-150, azim=110)

# ax.scatter(y_transform[:, 0], y_transform[:, 1], y_transform[:, 2], c=y_train,
#            cmap=plt.cm.Set1, marker='o', edgecolor='black', s=40)

# ax.set_title("First three LDA directions")
# ax.set_xlabel("1st eigenvector")
# ax.w_xaxis.set_ticklabels([])
# ax.set_ylabel("2nd eigenvector")
# ax.w_yaxis.set_ticklabels([])
# ax.set_zlabel("3rd eigenvector")
# ax.w_zaxis.set_ticklabels([])

plt.scatter(y_transform[:, 0], y_transform[:, 1], marker='o', c=y_train,edgecolor='black', s=40)
plt.title('LDA Algorithm')
plt.show()