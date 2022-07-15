import pandas as pd
import itertools
import seaborn as sns
from k_cross import K_Flod_spilt
import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
# noinspection PyUnresolvedReferences
from IPython.display import Image, display
# noinspection PyUnresolvedReferences
import pydotplus
import numpy as np
import warnings
warnings.filterwarnings("ignore")
# noinspection PyUnresolvedReferences
# import fitz
# import os
# os.environ["PATH"] += os.pathsep+'C:/Program Files/Graphviz/bin/'


# 导入数据
df = pd.read_table("iris.txt", sep=",", header=None, encoding=None)

X = df[[0, 1, 2, 3]]
Y = df[[4]]
feature = ['sepal length(cm)', 'sepal width(cm)', 'petal length(cm)', 'petal width(cm)']

# 交叉验证法
x_train, x_test, y_train, y_test = K_Flod_spilt(10, 1, X, Y)
train_x = np.array(x_train)
train_y = np.array(y_train)
test_x = np.array(x_test)
test_y = np.array(y_test)

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
# dot_data = export_graphviz(clf, out_file=None, feature_names=feature, class_names=True, filled=True, rounded=True)

# 在训练集和测试集上分布利用训练好的模型进行预测
train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)
train_result = clf.predict(train_x)
test_result = clf.predict(test_x)

# 利用精确率评估模型效果
print('The accuracy of train set :', metrics.accuracy_score(y_train, train_predict))
print('The accuracy of test set:', metrics.accuracy_score(y_test, test_predict))

# 绘制散点图
subplot_start1 = 321# 绘制一个3行2列的图
subplot_start2 = 321# 绘制一个3行2列的图
col_numbers = range(0, 4)
col_pairs1 = itertools.combinations(col_numbers, 2)
col_pairs2 = itertools.combinations(col_numbers, 2)
col_name = ['sepal length(cm)', 'sepal width(cm)', 'petal length(cm)', 'petal width(cm)']
# cValue = train_y.copy()
cValue1 = []
for index in range(0, len(train_result)):
    if train_result[index] == 'Iris-setosa':
        cValue1.append('r')
    elif train_result[index] == 'Iris-versicolor':
        cValue1.append('b')
    else:
        cValue1.append('g')
cValue2 = []
for index in range(0, len(test_result)):
    if test_result[index] == 'Iris-setosa':
        cValue2.append('r')
    elif test_result[index] == 'Iris-versicolor':
        cValue2.append('b')
    else:
        cValue2.append('g')
plt.figure(figsize=(12, 12))
for i in col_pairs1:
    plt.subplot(subplot_start1)
    plt.scatter(train_x[:, i[0]], train_x[:, i[1]], c=cValue1)
    plt.xlabel(col_name[i[0]])
    plt.ylabel(col_name[i[1]])

    subplot_start1 += 1
plt.savefig(r'decisionTree-训练集.png')

plt.figure(figsize=(12, 12))
for j in col_pairs2:
    plt.subplot(subplot_start2)
    plt.scatter(test_x[:, j[0]], test_x[:, j[1]], c=cValue2)
    plt.xlabel(col_name[j[0]])
    plt.ylabel(col_name[j[1]])

    subplot_start2 += 1

plt.savefig(r'decisionTree-测试集.png')

# 混淆矩阵
confusion_matrix_result = metrics.confusion_matrix(test_predict, y_test)
print('The confusion matrix result:\n', confusion_matrix_result)

# 利用热力图对于混淆矩阵进行可视化
# ticks = ['setosa', 'versicolor', 'virginica']
ticks = np.unique(train_y)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_result, xticklabels=ticks, yticklabels=ticks, annot=True, cmap='YlGnBu')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()
plt.savefig(r'decisionTree-混淆矩阵.png')
# 决策树可视化
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_png("out.png")