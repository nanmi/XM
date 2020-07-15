import numpy as np

from sklearn.linear_model import LogisticRegression

import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#load iris data
iris = datasets.load_iris()
data = iris.data[:, [2,3]]
target = iris.target

#split train test datasets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

#standard x'data
ss = StandardScaler()
ss.fit(X_train)

X_train_std = ss.transform(X_train)
X_test_std = ss.transform(X_test)

#establish model
lr = LogisticRegression(C=1000, random_state=123) 
#C：正则化系数λ的倒数，float类型，默认为1.0。必须是正浮点型数。像SVM一样，越小的数值表示越强的正则化。
#random_state：随机数种子，int类型，可选参数，默认为无，仅在正则化优化算法为sag,liblinear时有用。
lr.fit(X_train_std, y_train)

print('Precision: ', lr.score(X_test_std, y_test))

predict_y = lr.predict(X_test_std)

print('confusion_matrix: \n', confusion_matrix(y_test, predict_y))
print('accuracy_score: \n', accuracy_score(y_test, predict_y))
print('coefficience: \n', lr.coef_)