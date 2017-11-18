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

为了预测鲍鱼的年轮数，决策树将沿着树向下移动，直到它到达一片树叶。每一步将当前子集拆分为两个子集。对于每次分裂，`Rings` 均值变化定义为变量的贡献值，决定怎么分裂。

变量 `dt_reg` 是 sklearn 分类器目标值，`x_test` 表示 Pandas 或 NumPy 数组，包含我们希望得到的预测和贡献值的特征变量。贡献值变量 
`dt_reg_contrib` 是一个 2D NumPy数组（n_obs，n_features），其中 `n_obs` 观测数，`n_features` 是特征的数量。绘制一个给定鲍鱼的各特征的贡献值，看看哪些特征最影响其预测值。从下面的图中可以看出，这种特定的鲍鱼的重量和长度值对其预测的 `Rings` 有负面影响。
```
# Find abalones that are in the left-most leaf
X_test[(X_test['shell weight'] <= 0.0587) & (X_test['length'] <= 0.2625)].head()
df, true_label, pred = plot_obs_feature_contrib(dt_reg,
                                                dt_reg_contrib,
                                                X_test,
                                                y_test_reg,
                                                3,
                                                order_by='contribution'
                                               )
plt.tight_layout()
plt.savefig('plots/contribution_plot_dt_reg.png')
```
<div align=center><img height="300" src="https://github.com/youngxiao/Decision-Tree-and-Random-Forests/raw/master/results/contribution_plot_dt_reg.png"/></div>

可以用 violin 画出这个特定鲍鱼与整个种群的各变量贡献值比较，下图中，可以看出这个特定鲍鱼壳重相较其他相比异常低，事实上，大部分鲍鱼的壳重对应一个正的贡献值。
```
df, true_label, score = plot_obs_feature_contrib(dt_reg,
                                                 dt_reg_contrib,
                                                 X_test,
                                                 y_test_reg,
                                                 3,
                                                 order_by='contribution',
                                                 violin=True
                                                )
plt.tight_layout()
plt.savefig('plots/contribution_plot_violin_dt_reg.png')
```
<div align=center><img height="300" src="https://github.com/youngxiao/Decision-Tree-and-Random-Forests/raw/master/results/contribution_plot_violin_dt_reg.png"/></div>

以上的描述并没有对一个特定的变量如何影响鲍鱼的 `Rings` 有一个全面的解释。因此，我们可以根据一个特定特征的值绘制给它的贡献值。如果把壳重与它的贡献值进行比较，我们就可以看出随着壳重增加其贡献值增加。
```
plot_single_feat_contrib('shell weight', dt_reg_contrib, X_test, class_index=1)
plt.savefig('plots/shell_weight_contribution_dt.png')
```
<div align=center><img height="300" src="https://github.com/youngxiao/Decision-Tree-and-Random-Forests/raw/master/results/shell_weight_contribution_dt.png"/></div>

再来看看去壳重这个变量的贡献值具有非线性、非单调的特点。低的去壳重没有贡献，高的去壳重具有负的的贡献，而在低和高之间具有正的贡献。
```
plot_single_feat_contrib('shucked weight', dt_reg_contrib, X_test, class_index=1)
plt.savefig('plots/shucked_weight_contribution_dt.png')
```
<div align=center><img height="300" src="https://github.com/youngxiao/Decision-Tree-and-Random-Forests/raw/master/results/shucked_weight_contribution_dt.png"/></div>


## Section 2: 扩展到随机森林
以上的过程都可以扩展到随机森林，看看变量在森林中所有树的平均贡献。
```
df, true_label, pred = plot_obs_feature_contrib(rf_reg,
                                                rf_reg_contrib,
                                                X_test,
                                                y_test_reg,
                                                3,
                                                order_by='contribution'
                                               )
plt.tight_layout()
plt.savefig('plots/contribution_plot_rf.png')
```
<div align=center><img height="300" src="https://github.com/youngxiao/Decision-Tree-and-Random-Forests/raw/master/results/contribution_plot_rf.png"/></div>

```
df = plot_obs_feature_contrib(rf_reg,
                              rf_reg_contrib,
                              X_test,
                              y_test_reg,
                              3,
                              order_by='contribution',
                              violin=True
                             )
plt.tight_layout()
plt.savefig('plots/contribution_plot_violin_rf.png')
```
<div align=center><img height="300" src="https://github.com/youngxiao/Decision-Tree-and-Random-Forests/raw/master/results/contribution_plot_violin_rf.png"/></div>

由于随机森林本质上是随机的，对于给定的壳重下的贡献具有可变性。然而，平滑的黑色趋势线仍显示出增长的趋势。与决策树一样，我们看到壳重增加对应于较高的贡献
```
plot_single_feat_contrib('shell weight', rf_reg_contrib, X_test,
                         class_index=1, add_smooth=True, frac=0.3)
plt.savefig('plots/shell_weight_contribution_rf.png')
```
<div align=center><img height="300" src="https://github.com/youngxiao/Decision-Tree-and-Random-Forests/raw/master/results/shell_weight_contribution_rf.png"/></div>

再看看 直径 这个变量，我们具有复杂的、非单调的特点。直径似乎在贡献约0.45下降，在0.3和0.6左右的贡献高峰。除此之外，直径与目标变量 `Rings` 似乎具有普遍的正相关关系。
```
plot_single_feat_contrib('diameter', rf_reg_contrib, X_test,
                         class_index=1, add_smooth=True, frac=0.3)
plt.savefig('plots/diameter_contribution_rf.png')
```
<div align=center><img height="300" src="https://github.com/youngxiao/Decision-Tree-and-Random-Forests/raw/master/results/diameter_contribution_rf.png"/></div>

再看看其他变量
```
plot_single_feat_contrib('shucked weight', rf_reg_contrib, X_test,
                         class_index=1, add_smooth=True, frac=0.3)
plt.savefig('plots/shucked_weight_contribution_rf.png')
```
<div align=center><img height="300" src="https://github.com/youngxiao/Decision-Tree-and-Random-Forests/raw/master/results/shucked_weight_contribution_rf.png"/></div>

如果按照性别来分类 分为 male，female，infant，可以画出每一类的贡献，比如说 infant 这一类
```
df, true_label, scores = plot_obs_feature_contrib(dt_multi_clf,
                                                  dt_multi_clf_contrib,
                                                  X_test,
                                                  y_test_multi_clf,
                                                  3,
                                                  class_index=2,
                                                  order_by='contribution',
                                                  violin=True
                                                 )
true_value_list = ['Female', 'Male', 'Infant']
score_dict = zip(true_value_list, scores)
title = 'Contributions for Infant Class\nTrue Value: {}\nScores: {}'.format(true_value_list[true_label],
                                            ', '.join(['{} - {}'.format(i, j) for i, j in score_dict]))
plt.title(title)
plt.tight_layout()
plt.savefig('plots/contribution_plot_violin_multi_clf_dt.png')
```
<div align=center><img height="300" src="https://github.com/youngxiao/Decision-Tree-and-Random-Forests/raw/master/results/contribution_plot_violin_multi_clf_dt.png"/></div>

像之前的一样，我们还可以为每个类特征和贡献值的对应图。对于是雌性的鲍鱼，贡献随壳重的增加而增加，而对 infant 的鲍鱼，其贡献随壳重增加而减小。对雄性来说，当壳重超过0.5时，贡献开始增加，然后减少。
```
fig, ax = plt.subplots(1, 3, sharey=True)
fig.set_figwidth(20)

for i in xrange(3):
    plot_single_feat_contrib('shell weight', rf_multi_clf_contrib, X_test,
                             class_index=i, class_name=class_names[i],
                             add_smooth=True, c=colours[i], ax=ax[i])
    
plt.tight_layout()
plt.savefig('plots/shell_weight_contribution_by_sex_rf.png')
```
<div align=center><img height="300" src="https://github.com/youngxiao/Decision-Tree-and-Random-Forests/raw/master/results/shell_weight_contribution_by_sex_rf.png"/></div>


## 关于随机森林的总结:
随机森林是一个并行的，典型的高性能的机器学习模型。为了满足的客户的业务需求，我们不仅要提供一个高度预测模型，而且要模型也可以解释。也就是说，不是给他们一个黑盒子，不管模型表现得有多好。特别是对于政府或者金融界的客户。

## 依赖的 packages
* matplotlib
* pandas
* numpy
* seaborn
* treeinterpreter

## 欢迎关注
* Github：https://github.com/youngxiao
* Email： yxiao2048@gmail.com  or yxiao2017@163.com
* Blog:   https://youngxiao.github.io
