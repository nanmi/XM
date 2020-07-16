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


# 特征选择
"""通常来说，从两个方面考虑来选择特征：
a.特征是否发散：如果一个特征不发散，例如方差接近于0，
也就是说样本在这个特征上基本上没有差异，这个特征对于
样本的区分并没有什么用。
b.特征与目标的相关性：这点比较显见，与目标相关性高的
特征，应当优选选择。除方差法外，本文介绍的其他方法均
从相关性考虑。

根据特征选择的形式又可以将特征选择方法分为3种：
1.Filter：过滤法
2.Wrapper：包装法
3.Embedded：嵌入法
"""
#1 Filter：过滤法
#1.1 方差选择法
'''计算各个特征的方差，然后根据阈值，选择方差大于阈值的特征。
'''
from sklearn.feature_selection import VarianceThreshold

vt = VarianceThreshold(threshold=3)
X_train_trans = vt.fit_trandform(X_train)

#1.2 卡方检验
'''检验特征对标签的相关性，选择其中K个与标签最相关的特征。
'''
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

select_k_best = SelectKBest(chi2, k=2)
#返回选择特征后的数据
data_trans = select_k_best.fit_trandform(X_train, y_train)

#2 Wrapper：包装法
#2.1 递归特征消除法
'''递归消除特征法使用一个基模型来进行多轮训练，每轮训练后，
消除若干权值系数的特征，再基于新的特征集进行下一轮训练。
'''
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

base_model = LogisticRegression()
rfe = RFE(estimator=base_model, n_features_to_select=2)
data_trans = rfe.fit_trandform(X_train, y_train)

#3 Embedded：嵌入法
#3.1 基于惩罚项的特征选择法
'''使用带惩罚项的基模型，除了筛选出特征外，同时也进行了降维。
'''
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

base_model = LogisticRegression
select_from_model = SelectFromModel(base_model(penalty='l1', C=0.1))
data_trans = select_from_model.fit_trandform(X_train, y_train)

#3.2 基于树模型的特征选择法
'''树模型中GBDT可用来作为基模型进行特征选择
'''
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier

base_model = GradientBoostingClassifier
select_from_model = SelectFromModel(base_model())
data_trans = select_from_model.fit_trandform(X_train, y_train)