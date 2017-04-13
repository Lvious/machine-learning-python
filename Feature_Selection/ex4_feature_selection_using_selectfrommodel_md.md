##特徵选择/范例四: Feature selection using SelectFromModel and LassoCV

http://scikit-learn.org/stable/auto_examples/feature_selection/plot_select_from_model_boston.html

此范例是示范以`LassoCV`来挑选特徵，Lasso是一种用来计算稀疏矩阵的线性模形。在某些情况下是非常有用的，因为在此演算过程中会以较少数的特徵来找最佳解，基于参数有相依性的情况下，使变数的数目有效的缩减。因此，Lasso法以及它的变形式可算是压缩参数关系基本方法。在某些情况下，此方法可以准确的侦测非零权重的值。

Lasso最佳化的目标函数:

![](http://scikit-learn.org/stable/_images/math/5ff15825a85204658e3e5aa6e3b5952b8f709c27.png)

1. 以`LassoCV`法来计算目标资讯性特徵数目较少的资料
2. 用`SelectFromModel`设定特徵重要性的门槛值来选择特徵
3. 提高`SelectFromModel`的`.threshold`使目标资讯性特徵数逼近预期的数目



### (一)取得波士顿房产资料
```
from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

# Load the boston dataset.
boston = load_boston()
X, y = boston['data'], boston['target']
```

### (二)使用LassoCV功能来筛选具有影响力的特徵
1. 由于资料的类型为连续数字，选用LassoCV来做最具有代表性的特徵选取。
2. 当设定好门槛值，并做训练后，可以用transform(X)取得计算过后，被认为是具有影响力的特徵以及对应的样本，可以由其列的数目知道总影响力特徵有几个。
3. 后面使用了增加门槛值来达到限制最后特徵数目的
4. 使用门槛值来决定后来选取的参数，其说明在下一个标题。
5. 需要用后设转换

### (三)设定选取参数的门槛值
```
while n_features > 2:
    sfm.threshold += 0.1
    X_transform = sfm.transform(X)
    n_features = X_transform.shape[1]
```

### (四)原始码之出处
Python source code: [plot_select_from_model_boston.py](http://scikit-learn.org/stable/_downloads/plot_select_from_model_boston.py)

```Python
# Author: Manoj Kumar <mks542@nyu.edu>
# License: BSD 3 clause

print(__doc__)

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

# Load the boston dataset.
boston = load_boston()
X, y = boston['data'], boston['target']

# We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
clf = LassoCV()

# Set a minimum threshold of 0.25
sfm = SelectFromModel(clf, threshold=0.25)
sfm.fit(X, y)
n_features = sfm.transform(X).shape[1]

# Reset the threshold till the number of features equals two.
# Note that the attribute can be set directly instead of repeatedly
# fitting the metatransformer.
while n_features > 2:
    sfm.threshold += 0.1
    X_transform = sfm.transform(X)
    n_features = X_transform.shape[1]

# Plot the selected two features from X.
plt.title(
    "Features selected from Boston using SelectFromModel with "
    "threshold %0.3f." % sfm.threshold)
feature1 = X_transform[:, 0]
feature2 = X_transform[:, 1]
plt.plot(feature1, feature2, 'r.')
plt.xlabel("Feature number 1")
plt.ylabel("Feature number 2")
plt.ylim([np.min(feature2), np.max(feature2)])
plt.show()
```
