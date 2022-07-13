# noinspection PyUnresolvedReferences
import pandas as pd
# noinspection PyUnresolvedReferences
from testSplit import train_test_split
# noinspection PyUnresolvedReferences
from k_cross import K_Flod_spilt
# noinspection PyUnresolvedReferences
import numpy as np  # 导入numpy包

# 无省略显示1000行数据
pd.set_option("display.max_rows", 1000)

# 导入数据
df = pd.read_table("iris.txt", sep=",", header=None, encoding=None)


# 去重
df.drop_duplicates()

# 去除含空值的行
# df.dropna(inplace=True)
# 去除全为空值的行
# df.dropna(axis=0, how="any", inplace=True)
# 把空值填为0
# df.fillna(value=0, inplace=True)

X = df[[0, 1, 2, 3]]
Y = df[[4]]

# 留出法
# train_X, test_X, train_Y, test_Y = train_test_split(X, Y, 0.2)

# 交叉验证法
train_X, test_X, train_Y, test_Y = K_Flod_spilt(5, 1, X, Y)
print(train_X)
print(train_Y)

