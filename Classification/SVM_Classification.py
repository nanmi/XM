import numpy as np

#model params auto search optimal params
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score

#生成数据集
X_train = np.array([-1, -1, -2, -1, 1, 1, 2, 1]).reshape(4, 2)
y_train = np.array([1, 1, 2, 2])

X_test = np.array([-0.8, -1]).reshape(1,2)

# svc = GridSearchCV(SVC(), param_grid={"kernel": ("linear", 'rbf'), \
#     "C": np.logspace(-3, 3, 7), "gamma": np.logspace(-3, 3, 7)})

svc = SVC()
svc.fit(X_train, y_train)

y_predict = svc.predict(X_test)

print('support vectors: \n', svc.support_vectors_)
print('support vectors index: \n', svc.support_)
# print(svc.best_params_)