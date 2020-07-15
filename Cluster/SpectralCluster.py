'''谱聚类算法
比起传统的K-Means算法，谱聚类对数据分布的适应性更强，聚类效果也很优秀，同时聚类的计算量也小很多
谱聚类算法的主要优点有：

　　　　1）谱聚类只需要数据之间的相似度矩阵，因此对于处理稀疏数据的聚类很有效。这点传统聚类算法比如K-Means                       很难做到

　　　　2）由于使用了降维，因此在处理高维数据聚类时的复杂度比传统聚类算法好。

谱聚类算法的主要缺点有：

　　　　1）如果最终聚类的维度非常高，则由于降维的幅度不够，谱聚类的运行速度和最后的聚类效果均不好。

　　　　2) 聚类效果依赖于相似矩阵，不同的相似矩阵得到的最终聚类效果可能很不同。
参考:
https://www.cnblogs.com/pinard/p/6221564.html
https://www.cnblogs.com/pinard/p/6235920.html
'''
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

from sklearn.cluster import SpectralClustering

from sklearn.metrics import calinski_harabaz_score

# Create cluster datasets
clsuter_std = [0.4, 0.4, 0.3, 0.3, 0.4]
X, y = make_blobs(n_samples=500, n_features=6, centers=5, \
                cluster_std=clsuter_std, random_state=666)
print(X.shape, y.shape)
# Create cluster model
n_clusters = 5
gamma = 0.1
spectral = SpectralClustering(n_clusters=n_clusters, gamma=gamma)

y_predict = spectral.fit_predict(X)

print('Calinski-Harabasz Index Score: ', calinski_harabaz_score(X, y_predict))

# Visualization
plt.scatter(X[:,0], X[:,1], c=y_predict, marker='o', edgecolors='black')

plt.xlabel('X')
plt.ylabel('y')
plt.title('Spectral Cluster Algorithm')

plt.show()