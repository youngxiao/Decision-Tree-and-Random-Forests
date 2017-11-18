# Decision Tree and Random Forests demo with treeinterpreter

# 示例说明
这个博客深入到决策树和随机森林的基础上，以便更好地解释它们。
在过去的几年中，随机森林是一种新兴的机器学习技术。它是一种基于非线性树的模型，可以提供精确的结果。然而，大多是黑箱，通常很难解释和充分理解。在本文中，我们将深入了解随机森林的基本知识，以便更好地掌握它们。首先从决策树开始。
编译环境是 jupyter notebook， 可以通过安装  Anaconda，导入 scikit-learn 库([sklearn 安装](http://scikit-learn.org/stable/install.html#install-bleeding-edge。))可以很容易实现，另外要用到 [treeinterpreter](https://pypi.python.org/pypi/treeinterpreter/0.1.0) ，用到的数据集为 [abalone](https://archive.ics.uci.edu/ml/datasets/abalone) 数据集，本文的 [github示例代码](https://github.com/youngxiao/Decision-Tree-and-Random-Forests)。其中代码为 `Decision_Tree_and_Random_Forest.ipynb`，在 `tree_interp_functions.py` 中有很多 plot 函数。
## 概述
`Decision_Tree_and_Random_Forest.ipynb`代码中主要分为两个部分
* 决策树
* 随机森林

用到的数据集 [abalone](https://archive.ics.uci.edu/ml/datasets/abalone)，其中 `Rings` 是预测值：要么作为连续值，要么作为分类值。具体为：

名称 | 数据类型 | 单位 | 描述 
----|------|------|------
Sex | nominal | -- | M, F, and I (infant) 
Length | continuous | mm | Longest shell measurement 
Diameter(直径)	| continuous | mm | perpendicular to length 
Height | continuous | mm | with meat in shell 
Whole weight | continuous | grams | whole abalone 
Shucked weight(去壳重) | continuous	| grams | weight of meat 
Viscera weight(内脏重) | continuous | grams | gut weight (after bleeding) 
Shell weight(壳重) | continuous | grams | after being dried 
Rings(预测值) | integer | -- | +1.5 gives the age in years 

首先 import 各种库
```
from __future__ import division

from IPython.display import Image, display
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor,\
                         export_graphviz
from treeinterpreter import treeinterpreter as ti
import pydotplus

from tree_interp_functions import *
```
设置 matplotlib，设置 seaborn 颜色
```
# Set default matplotlib settings
plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['figure.titlesize'] = 26
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 16

# Set seaborn colours
sns.set_style('darkgrid')
sns.set_palette('colorblind')
blue, green, red, purple, yellow, cyan = sns.color_palette('colorblind')
```



## Section 1: 决策树
决策树通过以贪婪的方式迭代地将数据分成不同的子集。对于回归树，是最小化所有子集中的 MSE（均方误差）或 MAE（平均绝对误差）来选择。对于分类树，通过最小化生成子集中的熵或 Gini 来决定分类。由此产生的分类器将特征空间分隔成不同的子集。分别设置深度为 1，2，5，10.
```
light_blue, dark_blue, light_green, dark_green, light_red, dark_red = sns.color_palette('Paired')
x, z = make_moons(noise=0.20, random_state=5)
df = pd.DataFrame({'z': z,
                   'x': x[:, 0],
                   'y': x[:, 1]
                  })

md_list = [1, 2, 5, 10]
fig, ax = plt.subplots(2, 2)

fig.set_figheight(10)
fig.set_figwidth(10)
for i in xrange(len(md_list)):
    md = md_list[i]
    ix_0 = int(np.floor(i/2))
    ix_1 = i%2
    
    circle_dt_clf = DecisionTreeClassifier(max_depth=md, random_state=0)
    circle_dt_clf.fit(df[['x', 'y']], df['z'])

    xx, yy = np.meshgrid(np.linspace(df.x.min() - 0.5, df.x.max() + 0.5, 50),
                         np.linspace(df.y.min() - 0.5, df.y.max() + 0.5, 50))
    z_pred = circle_dt_clf.predict(zip(xx.reshape(-1), yy.reshape(-1)))
    z_pred = np.array([int(j) for j in z_pred.reshape(-1)])\
        .reshape(len(xx), len(yy))

    ax[ix_0, ix_1].contourf(xx, yy, z_pred, cmap=plt.get_cmap('GnBu'))

    df.query('z == 0').plot('x', 'y', kind='scatter',
                            s=40, c=green, ax=ax[ix_0, ix_1])
    df.query('z == 1').plot('x', 'y', kind='scatter',
                            s=40, c=light_blue, ax=ax[ix_0, ix_1])
    
    ax[ix_0, ix_1].set_title('Max Depth: {}'.format(md))
    ax[ix_0, ix_1].set_xticks([], [])
    ax[ix_0, ix_1].set_yticks([], [])
    ax[ix_0, ix_1].set_xlabel('')
    ax[ix_0, ix_1].set_ylabel('')

plt.tight_layout()
plt.savefig('plots/dt_iterations.png')
```
<div align=center><img height="400" src="https://github.com/youngxiao/Decision-Tree-and-Random-Forests/raw/master/results/dt_iterations.png"/></div>


导入 abalone 数据。 展示决策树和随机森林回归以及分类如何工作的。我们使用 `Rings` 变量作为连续变量，并从中创建一个二进制变量来表示 `Rings`。
```
column_names = ["sex", "length", "diameter", "height", "whole weight", 
                "shucked weight", "viscera weight", "shell weight", "rings"]
abalone_df = pd.read_csv('abalone.csv', names=column_names)
abalone_df['sex'] = abalone_df['sex'].map({'F': 0, 'M': 1, 'I': 2})
abalone_df['y'] = abalone_df.rings.map(lambda x: 1 if x > 9 else 0)
abalone_df.head()
```
将数据集分为 train 和 test。
```
abalone_train, abalone_test = train_test_split(abalone_df, test_size=0.2,
                                               random_state=0)

X_train = abalone_train.drop(['sex', 'rings', 'y'], axis=1)
y_train_bin_clf = abalone_train.y
y_train_multi_clf = abalone_train.sex
y_train_reg = abalone_train.rings

X_test = abalone_test.drop(['sex', 'rings', 'y'], axis=1)
y_test_bin_clf = abalone_test.y
y_test_multi_clf = abalone_test.sex
y_test_reg = abalone_test.rings

X_train = X_train.copy().reset_index(drop=True)
y_train_bin_clf = y_train_bin_clf.copy().reset_index(drop=True)
y_train_multi_clf = y_train_multi_clf.copy().reset_index(drop=True)
y_train_reg = y_train_reg.copy().reset_index(drop=True)

X_test = X_test.copy().reset_index(drop=True)
y_test_bin_clf = y_test_bin_clf.copy().reset_index(drop=True)
y_test_multi_clf = y_test_multi_clf.copy().reset_index(drop=True)
y_test_reg = y_test_reg.copy().reset_index(drop=True)
```
创建简单的决策树和随机森林模型，可以设置决策树的深度来看 interpretation 的用法。
```
dt_bin_clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
dt_bin_clf.fit(X_train, y_train_bin_clf)

dt_multi_clf = DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=0)
dt_multi_clf.fit(X_train, y_train_multi_clf)

dt_reg = DecisionTreeRegressor(criterion='mse', max_depth=3, random_state=0)
dt_reg.fit(X_train, y_train_reg)

rf_bin_clf = RandomForestClassifier(criterion='entropy', max_depth=10, n_estimators=100, random_state=0)
rf_bin_clf.fit(X_train, y_train_bin_clf)

rf_multi_clf = RandomForestClassifier(criterion='entropy', max_depth=10,  n_estimators=100, random_state=0)
rf_multi_clf.fit(X_train, y_train_multi_clf)

rf_reg = RandomForestRegressor(criterion='mse', max_depth=10,  n_estimators=100, random_state=0)
rf_reg.fit(X_train, y_train_reg)
```
创建特征贡献值，用 `ti.predict` 可以得到预测值，偏差项和贡献值. 贡献值矩阵是一个 3D 数组，由每个样本的贡献值，特征和分类标签组成。
```
dt_bin_clf_pred, dt_bin_clf_bias, dt_bin_clf_contrib = ti.predict(dt_bin_clf, X_test)
rf_bin_clf_pred, rf_bin_clf_bias, rf_bin_clf_contrib = ti.predict(rf_bin_clf, X_test)

dt_multi_clf_pred, dt_multi_clf_bias, dt_multi_clf_contrib = ti.predict(dt_multi_clf, X_test)
rf_multi_clf_pred, rf_multi_clf_bias, rf_multi_clf_contrib = ti.predict(rf_multi_clf, X_test)

dt_reg_pred, dt_reg_bias, dt_reg_contrib = ti.predict(dt_reg, X_test)
rf_reg_pred, rf_reg_bias, rf_reg_contrib = ti.predict(rf_reg, X_test)
```
可视化决策树，利用 `graphviz` 可视化决策树。可以显示到每个叶子节点的路径以及每个节点分类的比例。
```
reg_dot_data = export_graphviz(dt_reg,
                               out_file=None,
                               feature_names=X_train.columns
                              )
reg_graph = pydotplus.graph_from_dot_data(reg_dot_data)
reg_graph.write_png('plots/reg_dt_path.png')
Image(reg_graph.create_png())
```
<div align=center><img height="300" src="https://github.com/youngxiao/Decision-Tree-and-Random-Forests/raw/master/results/reg_dt_path.png"/></div>

```
bin_clf_dot_data = export_graphviz(dt_bin_clf,
                                   out_file=None,
                                   feature_names=X_train.columns
                                  )
bin_clf_graph = pydotplus.graph_from_dot_data(bin_clf_dot_data)
reg_graph.write_png('plots/bin_clf_dt_path.png')
Image(bin_clf_graph.create_png())
```
<div align=center><img height="300" src="https://github.com/youngxiao/Decision-Tree-and-Random-Forests/raw/master/results/bin_clf_dt_path.png"/></div>



* 分隔超平面：上述将数据集分割开来的直线叫做分隔超平面。
* 超平面：如果数据集是N维的，那么就需要N-1维的某对象来对数据进行分割。该对象叫做超平面，也就是分类的决策边界。
* 间隔：一个点到分割面的距离，称为点相对于分割面的距离。数据集中所有的点到分割面的最小间隔的2倍，称为分类器或数据集的间隔。
* 最大间隔：SVM分类器是要找最大的数据集间隔。
* 支持向量：离分割超平面最近的那些点

sklearn的SVM里面会有一个属性support_vectors_，标示“支持向量”，也就是样本点里离超平面最近的点，组成的。
咱们来画个图，把超平面和支持向量都画出来。

```
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='spring')
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=200, facecolors='none');
```
<div align=center><img height="320" src="https://github.com/youngxiao/SVM-demo/raw/master/reasult/svm2.png"/></div>

可以用IPython的 `interact` 函数来看看样本点的分布，会怎么样影响超平面:
```
from IPython.html.widgets import interact

def plot_svm(N=100):
    X, y = make_blobs(n_samples=200, centers=2,
                      random_state=0, cluster_std=0.60)
    X = X[:N]
    y = y[:N]
    clf = SVC(kernel='linear')
    clf.fit(X, y)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='spring')
    plt.xlim(-1, 4)
    plt.ylim(-1, 6)
    plot_svc_decision_function(clf, plt.gca())
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=200, facecolors='none')
    
interact(plot_svm, N=[10, 200], kernel='linear');
```
<div align=center><img height="320" src="https://github.com/youngxiao/SVM-demo/raw/master/reasult/svm3.png"/></div>



## Section 2: SVM 与 核函数
对于非线性可切分的数据集，要做分割，就要借助于核函数了简单一点说呢，核函数可以看做对原始特征的一个映射函数，
不过SVM不会傻乎乎对原始样本点做映射，它有更巧妙的方式来保证这个过程的高效性。
下面有一个例子，你可以看到，线性的kernel(线性的SVM)对于这种非线性可切分的数据集，是无能为力的。
```
from sklearn.datasets.samples_generator import make_circles
X, y = make_circles(100, factor=.1, noise=.1)

clf = SVC(kernel='linear').fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='spring')
plot_svc_decision_function(clf);
```
<div align=center><img height="320" src="https://github.com/youngxiao/SVM-demo/raw/master/reasult/svm4.png"/></div>

然后强大的高斯核/radial basis function就可以大显身手了:
```
r = np.exp(-(X[:, 0] ** 2 + X[:, 1] ** 2))

from mpl_toolkits import mplot3d

def plot_3D(elev=30, azim=30):
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='spring')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')

interact(plot_3D, elev=[-90, 90], azip=(-180, 180));
```
<div align=center><img height="320" src="https://github.com/youngxiao/SVM-demo/raw/master/reasult/svm5.png"/></div>

你在上面的图上也可以看到，原本在2维空间无法切分的2类点，映射到3维空间以后，可以由一个平面轻松地切开了。
而带rbf核的SVM就能帮你做到这一点:
```
clf = SVC(kernel='rbf')
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='spring')
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=200, facecolors='none');
```
<div align=center><img height="320" src="https://github.com/youngxiao/SVM-demo/raw/master/reasult/svm6.png"/></div>

## 关于SVM的总结:
* 非线性映射是SVM方法的理论基础，SVM利用内积核函数代替向高维空间的非线性映射；
* 对特征空间划分的最优超平面是SVM的目标，最大化分类边际的思想是SVM方法的核心；
* 支持向量是SVM的训练结果,在SVM分类决策中起决定作用的是支持向量。因此，模型需要存储空间小，算法鲁棒性强；
* 无任何前提假设，不涉及概率测度；
* SVM算法对大规模训练样本难以实施
* 用SVM解决多分类问题存在困难，经典的支持向量机算法只给出了二类分类的算法，而在数据挖掘的实际应用中，一般要解决多类的分类问题。可以通过多个二类支持向量机的组合来解决。主要有一对多组合模式、一对一组合模式和SVM决策树；再就是通过构造多个分类器的组合来解决。主要原理是克服SVM固有的缺点，结合其他算法的优势，解决多类问题的分类精度。如：与粗集理论结合，形成一种优势互补的多类问题的组合分类器。
* SVM是O(n^3)的时间复杂度。在sklearn里，LinearSVC是可扩展的(也就是对海量数据也可以支持得不错), 对特别大的数据集SVC就略微有点尴尬了。不过对于特别大的数据集，你倒是可以试试采样一些样本出来，然后用rbf核的SVC来做做分类。

## 依赖的 packages
* matplotlib
* pylab
* numpy
* seaborn

## 欢迎关注
* Github：https://github.com/youngxiao
* Email： yxiao2048@gmail.com  or yxiao2017@163.com
* Blog:   https://youngxiao.github.io
