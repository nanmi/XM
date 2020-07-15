import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_circles, make_blobs

from sklearn.cluster import DBSCAN

from sklearn.metrics import calinski_harabaz_score

# Create cluster dataset
X1, y1 = make_circles(n_samples=5000, factor=0.6, noise=0.05, random_state=666)
X2, y2 = make_blobs(n_samples=1000, n_features=2, centers=[[1.2, 1.2]], \
                    cluster_std=[[0.1]], random_state=666)

X = np.concatenate((X1, X2))
y = np.concatenate((y1, y2))

# Create cluster model
dbscan = DBSCAN(eps=0.1, min_samples=5)

y_predict = dbscan.fit_predict(X)

print('Calinski-Harabasz Index Score: ', calinski_harabaz_score(X, y_predict))

# Visualization
plt.scatter(X[:,0], X[:,1], c=y_predict, marker='o', edgecolors='black')

plt.xlabel('X')
plt.ylabel('y')
plt.title('DBSCAN Cluster Algorithm')

plt.show()