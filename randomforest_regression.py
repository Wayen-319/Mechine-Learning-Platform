# noinspection PyUnresolvedReferences
import pandas as pd
import numpy as np
from k_cross import K_Flod_spilt
import warnings
from sklearn import metrics
warnings.filterwarnings("ignore")

# 导入波士顿数据集
df = pd.read_table("housing.csv", sep=",", header=None, encoding=None)
# 数据归一化
df_norm = (df - df.min()) / (df.max() - df.min())
X = df_norm[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
Y = df_norm[[13]]

# 交叉验证法
x_train, x_test, y_train, y_test = K_Flod_spilt(10, 1, X, Y)

train_x = np.array(x_train)
train_y = np.array(y_train)
test_x = np.array(x_test)
test_y = np.array(y_test)

# 训练随机森林解决回归问题
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=200, random_state=0)
# 模型训练
regressor.fit(x_train, y_train)
# 预测
y_pred = regressor.predict(x_test)

# 评估回归性能
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, y_pred)))