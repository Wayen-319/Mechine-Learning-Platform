import numpy as np
import pandas as pd
import matplotlib as mpl
from scipy import io as spio
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# KNN核心算法
def classify(inX, dataSet, y_train, k):
    m, n = dataSet.shape  # shape（m, n）测试集中有m个个体和n个特征
    # 计算测试数据到每个点的欧式距离
    distances = []
    for i in range(m):
        sum = 0
        for j in range(n):
            sum += (inX[j] - dataSet[i][j]) ** 2
        distances.append(sum ** 0.5)
    sortDist = sorted(distances)  # 得到的是按照distance排序好的
    # 求k个最近的值的平均值
    sum = 0  
    for i in range(k):
        sum += y_train[distances.index(sortDist[i])]
    return sum/k


# 用于归一化
x_scaler = MinMaxScaler(feature_range=(-1, 1))
y_scaler = MinMaxScaler(feature_range=(-1, 1))


# 数据的读取和归一化
def dataTransform():
    df = spio.loadmat('D:\VSProject\data\机器学习平台\house.mat')
    df = df['housing']
    x = df[:,0:12]
    y = df[:,13]
    y = y.reshape(-1, 1)  # 在sklearn中，所有的数据都应该是二维矩阵,所以需要使用.reshape(1,-1)进行转换
    # 对数据进行最大最小值归一化
    x = x_scaler.fit_transform(x)
    y = y_scaler.fit_transform(y)
    # 训练集
    x_train = x[0:399, :]  # 二维
    y_train = y[0:399]
    # 测试集
    x_test = x[400:505, :]
    y_test = y[400:505]
    return x_train, y_train, x_test, y_test


# 测试算法
def Test():
    x_train, y_train, x_test, y_test = dataTransform()
    predict = []  # 记录预测值
    err = 0
    for i in range(len(x_test)):  # 对每一个测试数据
        predict.append(classify(x_test[i], x_train, y_train, 5))  # 返回平均值
        # print(predict[i], y_test[i])
        err += np.square(y_test[i]-predict[i])  # 计算误差和
    mse_err = err/len(x_test)
    print("the total mse error is: ", mse_err)
    predict = np.array(predict)  # 转成array
    draw(predict, y_test)


# 画图函数
def draw(predict, y_test):
    # 先转化为实际值
    predict = predict.reshape(-1, 1)
    predict = y_scaler.inverse_transform(predict)
    y_test = y_scaler.inverse_transform(y_test)
    # 解决中文无法显示的问题
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(8, 6))
    plt.plot(predict, label='pred')
    plt.plot(y_test, label='actual')
    plt.title('波士顿房价-测试集', )
    plt.legend()
    plt.savefig('D:\VSProject\image\波士顿房价KNN.png')
    plt.show()


if __name__ == '__main__':
    Test()

