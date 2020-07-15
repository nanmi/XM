'''截断奇异值分解（Truncated singular value decomposition，TSVD）是一种矩阵因式分解技术，
将矩阵 M 分解成 U ， Σ 和 V 。它与PCA很像，只是SVD分解是在数据矩阵上进行，而PCA是在数据的
协方差矩阵上进行。通常，SVD用于发现矩阵的主成份。对于病态矩阵，目前主要的处理办法有预调节矩
阵方法、区域分解法、正则化方法等，截断奇异值分解技术TSVD就是一种正则化方法，它牺牲部分精度换
去解的稳定性，使得结果具有更高的泛化能力。
'''
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib import offsetbox

from sklearn.datasets import load_digits

from sklearn.decomposition import TruncatedSVD

# Load digits data
digits = load_digits(n_class=7)
X = digits.data
y = digits.target

n_samples, n_features = X.shape

# Create Display function
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)#正则化
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 12})
    #打印彩色字体
    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)#输出图上输出图片
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


# Create Dimension Reduction model
T_SVD = TruncatedSVD(n_components=2)

s_t = time.time()

y_transform = T_SVD.fit_transform(X)

# Visualization
plot_embedding(y_transform, \
    title='TruncatedSVD Dimension Reduction of the digits ({:.3f}s)'.format(time.time() - s_t))

plt.show()