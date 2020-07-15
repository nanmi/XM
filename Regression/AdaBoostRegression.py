'''集成学习中算法的原理都是利用基础分类回归算法作为弱分类回归器完成的
AdaBoost回归实际上是利用决策树回归'''
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

# Create the datasets
rng = np.random.RandomState(1)
X_train = np.linspace(0, 6, 100)[:, np.newaxis]
y_train = np.sin(X_train).ravel() + np.sin(6*X_train).ravel() + rng.normal(0, 0.1, X_train.shape[0])

# Create 2 regression model
dtr = DecisionTreeRegressor(max_depth=4)

n_estimators=300
abr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), \
                        n_estimators=n_estimators, random_state=rng)

dtr.fit(X_train, y_train)
abr.fit(X_train, y_train)

# Predict
y_predict_dtr = dtr.predict(X_train)
y_predict_abr = abr.predict(X_train)


# Plot results
plt.scatter(X_train, y_train, c='k', label='data', marker='o', edgecolors='black')
plt.plot(X_train, y_predict_dtr, c='g', label='DecisionTreeRegression 1 tree')
plt.plot(X_train, y_predict_abr, c='r', label='AdaBoostRegression {} trees'.format(n_estimators))

plt.xlabel('data')
plt.ylabel('target')
plt.title('AdaBoost and DecisionTree Regression')
plt.legend()
plt.show()