from sklearn.svm import SVR

#save model and load model
from sklearn.externals import joblib

#model params auto search optimal params
from sklearn.model_selection import GridSearchCV

import numpy as np
import matplotlib.pyplot as plt

rng = np.random
# svr = joblib.load('svr.pkl')        # 读取模型

#生成数据集
X_train = rng.uniform(1, 100, (100, 1))
#生成y_train并将其变成一维，因为X_train是多维生成的y_train是多维的，变成一维
y_train = (5 * X_train + np.sin(X_train) * 5000 + 2 + np.square(X_train) + rng.rand(100, 1) * 5000).ravel()

X_test = np.linspace(0, 100, 100)[:, None]


# 自动选择合适的参数,
svr = GridSearchCV(SVR(), param_grid={"kernel": ("linear", 'rbf'), \
    "C": np.logspace(-3, 3, 7), "gamma": np.logspace(-3, 3, 7)})

svr.fit(X_train, y_train)

# joblib.dump(svr, 'svr.pkl')        # 保存模型

y_predict = svr.predict(X_test)

# 对结果进行可视化：
plt.scatter(X_train, y_train, c='k', label='data', zorder=1)
# plt.hold(True)
plt.plot(X_test, y_predict, c='r', label='SVR_fit')
plt.xlabel('data')
plt.ylabel('target')
plt.title('SVR versus Kernel Ridge')
plt.legend()
plt.show()

print(svr.best_params_)