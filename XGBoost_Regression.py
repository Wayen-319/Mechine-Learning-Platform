# ================基于Scikit-learn接口的回归================
import xgboost as xgb
from xgboost import plot_importance
from scipy import io as spio
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 自己的数据
# x_scaler = MinMaxScaler(feature_range=(-1, 1))
# y_scaler = MinMaxScaler(feature_range=(-1, 1))
# df = spio.loadmat('D:\VSProject\data\机器学习平台\house.mat')
# df = df['housing']
# x = df[:,0:12]
# y = df[:,13]
# y = y.reshape(-1, 1)  # 在sklearn中，所有的数据都应该是二维矩阵,所以需要使用.reshape(1,-1)进行转换
# # 对数据进行最大最小值归一化
# x = x_scaler.fit_transform(x)
# y = y_scaler.fit_transform(y)
# # 训练集
# x_train = x[0:399, :]  # 二维
# y_train = y[0:399]
# # 测试集
# x_test = x[400:505, :]
# y_test = y[400:505]

# sklearn库里的数据
boston = load_boston()
X,y = boston.data,boston.target

# XGBoost训练过程
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True)
model.fit(X_train, y_train)

# 对测试集进行预测
train_result = model.predict(X_train)
test_result = model.predict(X_test)

# 解决中文无法显示的问题
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(12, 12))
plt.plot(train_result.reshape(-1,1), label='pred')
plt.plot(y_train, label='actual')
plt.title('波士顿房价-训练集')
plt.legend()
plt.savefig('D:\VSProject\image\波士顿房价-训练集XGB.png')
plt.show()

# 解决中文无法显示的问题
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(12, 12))
plt.plot(test_result.reshape(-1,1), label='pred')
plt.plot(y_test, label='actual')
plt.title('波士顿房价-测试集')
plt.legend()
plt.savefig('D:\VSProject\image\波士顿房价-测试集XGB.png')
plt.show()

# 显示重要特征
plot_importance(model)
plt.savefig('D:\VSProject\image\波士顿房价XGB重要特征.png')
plt.show()

