import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor

#创建数据
#train_data
rng = np.random.RandomState(1)
X_train = np.sort(5*rng.rand(80, 1), axis=0)
y_train = np.sin(X_train).ravel()
y_train[::5] += 3*(0.5 - rng.rand(16))

#test_data
X_test = np.arange(0.0, 5.0, 0.01)[:np.newaxis].reshape(-1,1)


#实例化2个模型
dt_r1 = DecisionTreeRegressor(max_depth = 2)
dt_r2 = DecisionTreeRegressor(max_depth = 5)
dt_r1.fit(X_train, y_train)
dt_r2.fit(X_train, y_train)


#测试预测结果
dt_r1_predict = dt_r1.predict(X_test)
dt_r2_predict = dt_r2.predict(X_test)


#可视化
plt.scatter(X_train, y_train, edgecolors='black', c='orange', label='data')

plt.plot(X_test, dt_r1_predict, c='blue', label='max_depth=2', linewidth=2)
plt.plot(X_test, dt_r2_predict, c='green', label='max_depth=5', linewidth=2)

plt.xlabel('data')
plt.ylabel('target')
plt.title('Decision Tree Regression')
plt.legend()

plt.show()
