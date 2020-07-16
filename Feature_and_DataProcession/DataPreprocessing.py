'''特征工程主要分为三部分：
1. 数据预处理
2. 特征选择
3. 降维
'''
from sklearn.datasets import load_iris

# Load iris data
iris = load_iris()
X_train = iris.data
y_train = iris.target


# Data preprocession
""" 1.无量纲化
    2.二值化.例如学习成绩，假若只关心“及格”或不“及格”
    3.哑编码
    4.填充缺失值NAN
    5.数据变换
"""
#1 无量纲化
#1.1 标准化（也叫Z-score standardization）（对列向量处理）
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_train_std = ss.fit_transform(X_train)

#1.2 区间缩放（对列向量处理）
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
X_train_std = mms.fit_transform(X_train)

#1.3 归一化（对行向量处理）
from sklearn.preprocessing import Normalizer

normal = Normalizer()
X_train_std = normal.fit_transform(X_train)

#2 二值化（对列向量处理）
from sklearn.preprocessing import Binarizer

binarizer = Binarizer(threshold=3)
X_train_binarizered = binarizer.fit_transform(X_train)

#3 哑编码==独热编码（对列向量处理）
from sklearn.preprocessing import OneHotEncoder

one_hot_encoder = OneHotEncoder()
X_train_one_hot_encoder = one_hot_encoder.fit_transform(X_train.reshape(-1,1))

#4 填充缺失值NAN（对列向量处理）
#因为iris数据集没有缺失值，首先构造数据
import numpy as np
from sklearn.preprocessing import Imputer

X = np.vstack(np.array([np.nan]*4), X_train)

imputer = Imputer()
X_NAN = imputer.fit_transform(X)

#5 数据变换
#5.1 多项式变换（对行向量处理）
from sklearn.preprocessing import PolynomialFeatures

poly_feature = PolynomialFeatures()
X_train_trans = poly_feature.fit_transform(X_train)

#5.2 自定义变换
import numpy as np
from sklearn.preprocessing import FunctionTransformer

func_trans = FunctionTransformer(np.log1p)
X_train_trans = func_trans.fit_transform(X_train)