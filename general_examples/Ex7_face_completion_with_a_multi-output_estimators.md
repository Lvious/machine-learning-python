##通用范例/范例七: Face completion with a multi-output estimators

http://scikit-learn.org/stable/auto_examples/plot_multioutput_face_completion.html

这个范例用来展示scikit-learn如何用 `extremely randomized trees`, `k nearest neighbors`, `linear regression` 和 `ridge regression` 演算法来完成人脸估测。


## (一)引入函式库及内建影像资料库

引入之函式库如下

1. `sklearn.datasets`: 用来绘入内建之影像资料库
2. `sklearn.utils.validation`: 用来取乱数
3. `sklearn.ensemble`
4. `sklearn.neighbors`
5. `sklearn.linear_model`

使用 `datasets.load_digits()` 将资料存入， `data` 为一个dict型别资料，我们看一下资料的内容。

```python
from sklearn.datasets import fetch_olivetti_faces
data = fetch_olivetti_faces()
targets = data.target
data = data.images.reshape((len(data.images), -1))
```

| 显示 | 说明 |
| -- | -- |
| ('images', (400, 64, 64))| 共有40个人，每个人各有10张影像，共有 400 张影像，影像大小为 64x64 |
| ('data', (400, 4096)) | data 则是将64x64的矩阵摊平成4096个元素之一维向量 |
| ('targets', (400,)) | 说明400张图与40个人之分类对应 0-39，记录每张影像是哪一个人 |
| DESCR | 资料之描述 |


前面30个人当训练资料，之后当测试资料
```python
train = data[targets < 30]
test = data[targets >= 30]
```
测试影像从100张乱数选5张出来，变数`test`的大小变成(5,4096)
```python
# Test on a subset of people
n_faces = 5
rng = check_random_state(4)
face_ids = rng.randint(test.shape[0], size=(n_faces, ))
test = test[face_ids, :]
```

把每张训练影像和测试影像都切割成上下两部分:

X_人脸上半部分，
Y_人脸下半部分。
```python
n_pixels = data.shape[1]
X_train = train[:, :np.ceil(0.5 * n_pixels)]  
y_train = train[:, np.floor(0.5 * n_pixels):]  
X_test = test[:, :np.ceil(0.5 * n_pixels)]
y_test = test[:, np.floor(0.5 * n_pixels):]
```

## (二)资料训练
分别用以下四种演算法来完成人脸下半部估测

1. `extremely randomized trees` (绝对随机森林演算法)
2. `k nearest neighbors` (K-邻近演算法)
3. `linear regression` (线性回归演算法)
4. `ridge regression` (脊回归演算法)


```python
ESTIMATORS = {
    "Extra trees": ExtraTreesRegressor(n_estimators=10, max_features=32,random_state=0),
    "K-nn": KNeighborsRegressor(),
    "Linear regression": LinearRegression(),
    "Ridge": RidgeCV(),
}
```

分别把训练资料人脸上、下部分放入`estimator.fit()`中进行训练。上半部分人脸为条件影像，下半部人脸为目标影像。

`y_test_predict`为一个dict型别资料，存放5位测试者分别用四种演算法得到的人脸下半部估计结果。

```python
y_test_predict = dict()
for name, estimator in ESTIMATORS.items():
    estimator.fit(X_train, y_train)
    y_test_predict[name] = estimator.predict(X_test)
```

## (三)`matplotlib.pyplot`画出结果

每张影像都是64*64，总共有5位测试者，每位测试者分别有1张原图，加上使用4种演算法得到的估测结果。

```python
image_shape = (64, 64)
n_cols = 1 + len(ESTIMATORS)
plt.figure(figsize=(2. * n_cols, 2.26 * n_faces))
plt.suptitle("Face completion with multi-output estimators", size=16)

for i in range(n_faces):
    true_face = np.hstack((X_test[i], y_test[i]))

    if i:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1)
    else:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1,
                          title="true faces")


    sub.axis("off")
    sub.imshow(true_face.reshape(image_shape),
               cmap=plt.cm.gray,
               interpolation="nearest")

    for j, est in enumerate(sorted(ESTIMATORS)):
        completed_face = np.hstack((X_test[i], y_test_predict[est][i]))

        if i:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j)

        else:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j,
                              title=est)

        sub.axis("off")
        sub.imshow(completed_face.reshape(image_shape),
                   cmap=plt.cm.gray,
                   interpolation="nearest")

plt.show()
```

![](images/multioutput_face_completion_figure_1.png)
