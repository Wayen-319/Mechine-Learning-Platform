# noinspection PyUnresolvedReferences
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from k_cross import K_Flod_spilt
import pandas as pd
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

# 导入数据
df = pd.read_table("iris.txt", sep=",", header=None, encoding=None)

X = df[[0, 1, 2, 3]]
Y = df[[4]]

# 交叉验证法
x_train, x_test, y_train, y_test = K_Flod_spilt(10, 1, X, Y)

# 定义逻辑回归模型
clf = LogisticRegression(random_state=0, solver='lbfgs')

# 在训练集上训练逻辑回归模型
clf.fit(x_train, y_train)

# 在训练集和测试集上分布利用训练好的模型进行预测
train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)

# 利用predict_proba函数预测其概率
train_predict_proba = clf.predict_proba(x_train)
test_predict_proba = clf.predict_proba(x_test)

# 利用精确率评估模型效果
print('The accuracy of train set :', metrics.accuracy_score(y_train, train_predict))
print('The accuracy of test set:', metrics.accuracy_score(y_test, test_predict))

# 混淆矩阵
confusion_matrix_result = metrics.confusion_matrix(test_predict, y_test)
print('The confusion matrix result:\n', confusion_matrix_result)

# 利用热力图对于混淆矩阵进行可视化
ticks = ['setosa', 'versicolor', 'virginica']
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_result, xticklabels=ticks, yticklabels=ticks, annot=True, cmap='YlGnBu')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()