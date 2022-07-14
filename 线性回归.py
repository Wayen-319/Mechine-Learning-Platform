from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as mse

# 用于归一化
x_scaler = MinMaxScaler(feature_range=(-1, 1))
y_scaler = MinMaxScaler(feature_range=(-1, 1))

# Read dataset to pandas dataframe
# sklearn库里的数据
iris = load_boston()
col_name = iris['feature_names']#列名
X,y = iris.data,iris.target

X = x_scaler.fit_transform(X)
y = y_scaler.fit_transform(y.reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

#定义
model = LinearRegression()
#训练
model.fit(X_train, y_train)

prediction = model.predict(X_test)

print(mse(prediction, y_test))

#画图
plt.figure(figsize=(12, 12))
plt.plot(y_test, label='actual')

w = model.coef_
b = model.intercept_
xp = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
yp = -(w[0, 0] * xp + b) / w[0, 1]
plt.plot(xp, yp, 'b-', linewidth=2.0)

plt.title('波士顿房价-测试集')
plt.legend()
plt.savefig('D:\VSProject\image\波士顿房价LR.png')
# # 解决中文无法显示的问题
# plt.rcParams['font.sans-serif'] = [u'SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# plt.figure(figsize=(8, 6))
# plt.plot(prediction, label='pred')
# plt.plot(y_test, label='actual')
# plt.title('波士顿房价-测试集')
# plt.legend()
# plt.savefig('D:\VSProject\image\波士顿房价LR.png')
# plt.show()
