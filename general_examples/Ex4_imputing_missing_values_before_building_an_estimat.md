# 通用范例/范例四: Imputing missing values before building an estimator

http://scikit-learn.org/stable/auto_examples/missing_values.htm

在这范例说明有时补充缺少的数据(missing values)，可以得到更好的结果。但仍然需要进行交叉验证。来验证填充是否合适<br />。而missing values可以用均值、中位值，或者频繁出现的值代替。中位值对大数据之机器学习来说是比较稳定的估计值。

## (一)引入函式库及内建测试资料库

引入之函式库如下

1. `sklearn.ensemble.RandomForestRegressor`: 随机森林回归
2. `sklearn.pipeline.Pipeline`: 串联估计器
3. `sklearn.preprocessing.Imputer`: 缺失值填充
4. `sklearn.cross_validation import cross_val_score`:交叉验证

## (二)引入内建测试资料库(boston房产资料)
使用 `datasets.load_boston()` 将资料存入， `boston` 为一个dict型别资料，我们看一下资料的内容。<br />
n_samples 为样本数<br />
n_features 为特征数

```python
dataset = load_boston()
X_full, y_full = dataset.data, dataset.target
n_samples = X_full.shape[0]
n_features = X_full.shape[1]
```

| 显示 | 说明 |
| -- | -- |
| ('data', (506, 13))| 机器学习数据 |
| ('feature_names', (13,)) | 房地产相关特征 |
| ('target', (506,)) | 回归目标 |
| DESCR | 资料之描述 |

共有506笔资料及13个特征('CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT')用来描述房地产的週边状况，如CRIM (per capita crime rate by town)跟该区域之犯罪率有关。而迴归目标为房地产的价格，以1000美元为单位。也就是说这个范例希望以房地产的週遭客观数据来预测房地产的价格。

## (三)利用整个数据集来预测
全部的资料使用随机森林回归函数进行交叉验证，得到一个分数。<br />

Score with the entire dataset = 0.56
```python
estimator = RandomForestRegressor(random_state=0, n_estimators=100)
score = cross_val_score(estimator, X_full, y_full).mean()
print("Score with the entire dataset = %.2f" % score)
```

## (四)模拟资料损失时之预测情形
设定损失比例，并估计移除missing values后的得分
损失比例75%，损失样本数为379笔，剩馀样本为127笔。<br />
将127笔资料进行随机森林回归函数进行交叉验证，并得到一个分数。<br />

Score without the samples containing missing values = 0.49
```python
missing_rate = 0.75
n_missing_samples = np.floor(n_samples * missing_rate)
missing_samples = np.hstack((np.zeros(n_samples - n_missing_samples,
                                      dtype=np.bool),
                             np.ones(n_missing_samples,
                                     dtype=np.bool)))
rng.shuffle(missing_samples)
missing_features = rng.randint(0, n_features, n_missing_samples)

X_filtered = X_full[~missing_samples, :]
y_filtered = y_full[~missing_samples]
estimator = RandomForestRegressor(random_state=0, n_estimators=100)
score = cross_val_score(estimator, X_filtered, y_filtered).mean()
print("Score without the samples containing missing values = %.2f" % score)
```

## (五)填充missing values，估计填充后的得分
每一笔样本资料都在13个特征中随机遗失一个特征资料，<br />
使用`sklearn.preprocessing.Imputer`进行missing values的填充。<br />

```
class sklearn.preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)
```

填充后进行随机森林回归函数进行交叉验证，获得填充后分数。


```python
X_missing = X_full.copy()
X_missing[np.where(missing_samples)[0], missing_features] = 0
y_missing = y_full.copy()
estimator = Pipeline([("imputer", Imputer(missing_values=0,
                                          strategy="mean",
                                          axis=0)),
                      ("forest", RandomForestRegressor(random_state=0,
                                                       n_estimators=100))])
score = cross_val_score(estimator, X_missing, y_missing).mean()
print("Score after imputation of the missing values = %.2f" % score)
```

利用数据填充后的迴归函数，去测试填充前的资料，预测的准确率获得提升。<br/>

Score after imputation of the missing values = 0.57

## (六)完整程式码
Python source code: missing_values.py<br />
http://scikit-learn.org/stable/auto_examples/missing_values.html#example-missing-values-py
```python
import numpy as np

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import cross_val_score

rng = np.random.RandomState(0)

dataset = load_boston()
X_full, y_full = dataset.data, dataset.target
n_samples = X_full.shape[0]
n_features = X_full.shape[1]

# Estimate the score on the entire dataset, with no missing values
estimator = RandomForestRegressor(random_state=0, n_estimators=100)
score = cross_val_score(estimator, X_full, y_full).mean()
print("Score with the entire dataset = %.2f" % score)

# Add missing values in 75% of the lines
missing_rate = 0.75
n_missing_samples = np.floor(n_samples * missing_rate)
missing_samples = np.hstack((np.zeros(n_samples - n_missing_samples,
                                      dtype=np.bool),
                             np.ones(n_missing_samples,
                                     dtype=np.bool)))
rng.shuffle(missing_samples)
missing_features = rng.randint(0, n_features, n_missing_samples)

# Estimate the score without the lines containing missing values
X_filtered = X_full[~missing_samples, :]
y_filtered = y_full[~missing_samples]
estimator = RandomForestRegressor(random_state=0, n_estimators=100)
score = cross_val_score(estimator, X_filtered, y_filtered).mean()
print("Score without the samples containing missing values = %.2f" % score)

# Estimate the score after imputation of the missing values
X_missing = X_full.copy()
X_missing[np.where(missing_samples)[0], missing_features] = 0
y_missing = y_full.copy()
estimator = Pipeline([("imputer", Imputer(missing_values=0,
                                          strategy="mean",
                                          axis=0)),
                      ("forest", RandomForestRegressor(random_state=0,
                                                       n_estimators=100))])
score = cross_val_score(estimator, X_missing, y_missing).mean()
print("Score after imputation of the missing values = %.2f" % score)
```
```
results:
Score with the entire dataset = 0.56
Score without the samples containing missing values = 0.48
Score after imputation of the missing values = 0.55
```
