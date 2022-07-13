import numpy as np  # numpy库
from sklearn import svm
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
import pandas as pd  # 导入pandas
import matplotlib.pyplot as plt  # 导入图形展示库
from scipy import io as spio
from sklearn.metrics import r2_score as r2, mean_squared_error as mse, mean_absolute_error as mae
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.datasets import load_boston
# import seaborn as sns

# dir(load_boston())
# X = load_boston().data
# y = load_boston().target
# df = pd.DataFrame(X, columns=load_boston().feature_names)
# df.head()

# wine数据集
# train_x = spio.loadmat('D:\VSProject\data\机器学习平台\house_x.mat')
# train_Y = spio.loadmat('D:\VSProject\data\机器学习平台\house_y.mat')
# train = spio.loadmat('D:\VSProject\data\机器学习平台\house.mat')

# train_x = train_x['housing'] #  自变量x
# train_Y = train_Y['housing'] #  因变量y
# train = train['housing']

# train_y = np.ravel(train_Y)

# 用于归一化
x_scaler = MinMaxScaler(feature_range=(-1, 1))
y_scaler = MinMaxScaler(feature_range=(-1, 1))

# sklearn库里的数据
boston = load_boston()

X,y = boston.data,boston.target

X = x_scaler.fit_transform(X)
y = y_scaler.fit_transform(y.reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = svm.SVR(kernel='rbf',C=10, gamma = 0.01)

clf.fit(X_train,y_train) # (全体自变量，某个自变量)

train_result = clf.predict(X_train)
test_result = clf.predict(X_test)

test_predict = test_result.reshape(-1, 1)
train_result = train_result.reshape(-1, 1)

# test_predict = y_scaler.inverse_transform(test_predict)
# train_result = y_scaler.inverse_transform(train_result)

# y_test = x_scaler.inverse_transform(y_test)
# y_train = x_scaler.inverse_transform(y_train)
# 解决中文无法显示的问题
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(12, 12))
plt.plot(train_result, label='pred')
plt.plot(y_train, label='actual')
plt.title('波士顿房价-训练集')
plt.legend()
plt.savefig('D:\VSProject\image\波士顿房价-训练集SVR.png')
plt.show()

plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(12, 12))
plt.plot(test_result, label='pred')
plt.plot(y_test, label='actual')
plt.title('波士顿房价-测试集')
plt.legend()
plt.savefig('D:\VSProject\image\波士顿房价-测试集SVR.png')
plt.show()

# print(' SVR的默认衡量评估值值为:', clf.score(X_test,y_test))
# print(' SVR的R-squared值为:', r2(y_test, predict_y))
print(' 测试集-SVR的均方误差(mean squared error)为:',mse(y_test, test_result))
print(' 训练集-SVR的均方误差(mean squared error)为:',mse(y_train, train_result))
# print(' SVR的平均绝对误差(mean absolute error)为:',mae(y_test, predict_y))

# predictions = np.ravel(predictions)


# plt.scatter(predict_x[:,0], predict_y[:], s=8, c='blue',label ='prediction point')
# plt.scatter(train_x[:,0],train_y, s=8, c='red',label ='train point')
# plt.legend(loc='upper right')
# plt.show()

