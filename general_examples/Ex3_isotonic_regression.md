# 通用范例/范例三: Isotonic Regression

http://scikit-learn.org/stable/auto_examples/plot_isotonic_regression.html

迴归函数採用递增函数。

* y[] are inputs (real numbers)
* y_[] are fitted


这个范例的主要目的：

比较

* Isotonic Fit
* Linear Fit


# (一) Regression「迴归」
「迴归」就是找一个函数，尽量符合手边的一堆数据。此函数称作「迴归函数」。

# (二) Linear Regression「线性迴归」
迴归函数採用线性函数。误差採用平方误差。

`class sklearn.linear_model.LinearRegression`

二维数据，迴归函数是直线。

![](images/Isotonic Regression_figure_1.png)


# (三) Isotonic Regression「保序迴归」
具有分段迴归的效果。迴归函数採用递增函数。

`class sklearn.isotonic.IsotonicRegression`

採用平方误差，时间複杂度 O(N) 。

![](images/Isotonic Regression_figure_2.png)


# (四) 完整程式码

Python source code: plot_isotonic_regression.py

http://scikit-learn.org/stable/auto_examples/plot_isotonic_regression.html

```
print(__doc__)

# Author: Nelle Varoquaux <nelle.varoquaux@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# Licence: BSD

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state

n = 100
x = np.arange(n)
rs = check_random_state(0)
y = rs.randint(-50, 50, size=(n,)) + 50. * np.log(1 + np.arange(n))

###############################################################################
# Fit IsotonicRegression and LinearRegression models

ir = IsotonicRegression()

y_ = ir.fit_transform(x, y)

lr = LinearRegression()
lr.fit(x[:, np.newaxis], y)  # x needs to be 2d for LinearRegression

###############################################################################
# plot result

segments = [[[i, y[i]], [i, y_[i]]] for i in range(n)]
lc = LineCollection(segments, zorder=0)
lc.set_array(np.ones(len(y)))
lc.set_linewidths(0.5 * np.ones(n))

fig = plt.figure()
plt.plot(x, y, 'r.', markersize=12)
plt.plot(x, y_, 'g.-', markersize=12)
plt.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
plt.gca().add_collection(lc)
plt.legend(('Data', 'Isotonic Fit', 'Linear Fit'), loc='lower right')
plt.title('Isotonic regression')
plt.show()

```
![](images/Isotonic Regression_figure_3.png)
