'''BIRCH 聚类是一种基于层次划分的算法， 算法详细原理可以参照：
https://www.cnblogs.com/pinard/p/6179132.html
'''
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets.samples_generator import make_blobs

from sklearn.cluster import Birch

from sklearn.metrics import calinski_harabaz_score

# Create cluster dataset
centers = [[-1,-1],[0,0], [1,1], [2,2]]
cluster_std = [0.4, 0.3, 0.4, 0.2]
X_train, y_train = make_blobs(n_samples=1000, n_features=2, centers = centers, \
                                cluster_std = cluster_std, random_state = 666)

# Create cluster model
n_clusters = 4
birch = Birch(n_clusters=n_clusters)

y_predict = birch.fit_predict(X_train)

print('Calinski-Harabasz Index Score: ', calinski_harabaz_score(X_train, y_predict))

# Visualization
plt.scatter(X_train[:,0], X_train[:,1], c=y_predict, marker='o', edgecolors='black')

plt.xlabel('X')
plt.ylabel('y')
plt.title('BIRCH Cluster Algorithm')

plt.show()