'''t-SNE(t-分布邻域嵌入算法)
流形学习方法(Manifold Learning)，简称流形学习,可以将流形学习方法分为线性的和非线性的两种，
线性的流形学习方法如我们熟知的主成份分析（PCA），非线性的流形学习方法如等距映射（Isomap）、
拉普拉斯特征映射（Laplacian eigenmaps，LE）、局部线性嵌入(Locally-linear embedding，LLE)
参考：
http://lvdmaaten.github.io/tsne/
'''
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib import offsetbox

from sklearn.datasets import load_digits

from sklearn.manifold import TSNE

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


# Create model
tsne = TSNE(n_components=2, init='pca', random_state=0)

s_t = time.time()

y_transform = tsne.fit_transform(X)

# Visualization
plot_embedding(y_transform, \
    title='t-SNE Dimension Reduction of the digits ({:.3f})s'.format(time.time() - s_t))

plt.show()