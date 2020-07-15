import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets.samples_generator import make_blobs

from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabaz_score

# Create the datasets
centers = [[-1, -1], [0, 0], [1, 1], [2, 2]]
cluster_std = [0.4, 0.2, 0.2, 0.2]

X_train, y_train = make_blobs(n_samples=1000, n_features=2, centers=centers, cluster_std=cluster_std, \
                    random_state=666)


# Create cluster model
except_clusters = 4
k_means = KMeans(n_clusters=except_clusters, random_state=666)

y_predict = k_means.fit_predict(X_train)

# Output CH score 分数越高，表示聚类的效果越好
print('Calinski-Harabasz Index Score: ', calinski_harabaz_score(X_train, y_predict))

# Visualization
plt.scatter(X_train[:,0], X_train[:,1], c=y_predict, marker='o', edgecolors='black')

plt.xlabel('X train')
plt.ylabel('y train')
plt.title('K-Means Cluster')

plt.show()