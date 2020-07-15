'''在scikit-learn中，一共有3个朴素贝叶斯的分类算法类。
分别是GaussianNB，MultinomialNB和BernoulliNB。
其中:
GaussianNB就是先验为高斯分布的朴素贝叶斯，
MultinomialNB就是先验为多项式分布的朴素贝叶斯，
而BernoulliNB就是先验为伯努利分布的朴素贝叶斯'''

'''在使用GaussianNB的fit方法拟合数据后，我们可以进行预测。此时预测有三种方法
predict，predict_log_proba和predict_proba
* predict方法就是我们最常用的预测方法，直接给出测试集的预测类别输出。

* predict_proba则不同，它会给出测试集样本在各个类别上预测的概率。
容易理解，predict_proba预测出的各个类别概率里的最大值对应的类别，也就是predict方法得到类别。

* predict_log_proba和predict_proba类似，它会给出测试集样本在各个类别上预测的概率的一个对数转化。
转化后predict_log_proba预测出的各个类别对数概率里的最大值对应的类别，也就是predict方法得到类别。
'''
import numpy as np

from sklearn.naive_bayes import GaussianNB

#生成数据集
X_train = np.array([-1, -1, -2, -1, -3, -2, 1, 1, 2, 1, 3, 2]).reshape(6,2)
y_train = np.array([1, 1, 1, 2, 2, 2])

X_test = np.array([-0.8, -1]).reshape(1,2)


#构建模型
naive_bayes_Gaussian = GaussianNB()

naive_bayes_Gaussian.fit(X_train, y_train)

y_predict = naive_bayes_Gaussian.predict(X_test)
y_predict_proba = naive_bayes_Gaussian.predict_proba(X_test)
y_predict_log_proba = naive_bayes_Gaussian.predict_log_proba(X_test)

print('predict: \n', y_predict)
print('predict_proba: \n', y_predict_proba)
print('predict_log_proba: \n', y_predict_log_proba)