import pandas as pd
from k_cross import K_Flod_spilt
# noinspection PyUnresolvedReferences
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
# noinspection PyUnresolvedReferences
from IPython.display import Image, display
# noinspection PyUnresolvedReferences
import pydotplus
import warnings
warnings.filterwarnings("ignore")
# noinspection PyUnresolvedReferences
import fitz
import os
os.environ["PATH"] += os.pathsep+'C:/Program Files/Graphviz/bin/'


# 导入数据
df = pd.read_table("iris.txt", sep=",", header=None, encoding=None)

X = df[[0, 1, 2, 3]]
Y = df[[4]]
feature = ['sepal length(cm)', 'sepal width(cm)', 'petal length(cm)', 'petal width(cm)']

# 交叉验证法
x_train, x_test, y_train, y_test = K_Flod_spilt(10, 1, X, Y)


clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
dot_data = export_graphviz(clf, out_file=None, feature_names=feature, class_names=True, filled=True, rounded=True)

# 在训练集和测试集上分布利用训练好的模型进行预测
train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)

# 利用精确率评估模型效果
print('The accuracy of train set :', metrics.accuracy_score(y_train, train_predict))
print('The accuracy of test set:', metrics.accuracy_score(y_test, test_predict))

# 决策树可视化
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("out.png")