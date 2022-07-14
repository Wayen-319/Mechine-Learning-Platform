# # # -*- coding: UTF-8 -*-
# from sklearn.datasets import load_iris
# from matplotlib.font_manager import FontProperties
# from matplotlib.colors import ListedColormap
# import matplotlib.lines as mlines
# import matplotlib.pyplot as plt
# import numpy as np
# import operator
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler

# def classify0(test, train, labels, k):
#     #numpy函数shape[0]返回train的行数
#     trainSize = train.shape[0]
#     #在列向量方向上重复test共1次(横向),行向量方向上重复test共trainSize次(纵向)
#     diffMat = np.tile(test, (trainSize, 1)) - train
#     #二维特征相减后平方
#     sqDiffMat = diffMat**2
#     #sum()所有元素相加,sum(0)列相加,sum(1)行相加
#     sqDistances = sqDiffMat.sum(axis=1)
#     #开方,计算出距离
#     distances = sqDistances**0.5
#     #返回distances中元素从小到大排序后的索引值
#     sortedDistIndices = distances.argsort()
#     #定一个记录类别次数的字典
#     classCount = {}
#     for i in range(k):
#         #取出前k个元素的类别
#         voteIlabel = labels[sortedDistIndices[i]]
#         #dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
#         #计算类别次数
#         classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
#     #python3中用items()替换python2中的iteritems()
#     #key=operator.itemgetter(1)根据字典的值进行排序
#     #key=operator.itemgetter(0)根据字典的键进行排序
#     #reverse降序排序字典
#     sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
#     print(sortedClassCount)
#     #返回次数最多的类别,即所要分类的类别
#     return sortedClassCount[0][0]

# def file2matrix(X, y):

#     numberOfLines = len(X)
#     returnMat = np.zeros((numberOfLines,4))

#     classLabelVector = []

#     returnMat[:,:] = X[0:4]
#     classLabelVector = y 
#     return returnMat, classLabelVector


# def autoNorm(train):
#     #获得数据的最小值
#     minVals = train.min(0)
#     maxVals = train.max(0)
#     #最大值和最小值的范围
#     ranges = maxVals - minVals
#     #shape(train)返回train的矩阵行列数
#     normtrain = np.zeros(np.shape(train))
#     #返回train的行数
#     m = train.shape[0]
#     #原始值减去最小值
#     normtrain = train - np.tile(minVals, (m, 1))
#     #除以最大和最小值的差,得到归一化数据
#     normtrain = normtrain / np.tile(ranges, (m, 1))
#     #返回归一化数据结果,数据范围,最小值
#     return normtrain, ranges, minVals


# def datingClassTest():
#     # 用于归一化
#     x_scaler = MinMaxScaler(feature_range=(-1, 1))
#     y_scaler = MinMaxScaler(feature_range=(-1, 1))
#     # sklearn库里的数据
#     iris = load_iris()
#     X,y = iris.data,iris.target

#     X = x_scaler.fit_transform(X)
#     y = y_scaler.fit_transform(y.reshape(-1,1))
#     #将返回的特征矩阵和分类向量分别存储到datingDataMat和datingLabels中
#     datingDataMat, datingLabels = file2matrix(X, y)
#     #取所有数据的百分之十
#     hoRatio = 0.1
#     #数据归一化,返回归一化后的矩阵,数据范围,数据最小值
#     normMat, ranges, minVals = autoNorm(datingDataMat)
#     #获得normMat的行数
#     m = normMat.shape[0]
#     #百分之十的测试数据的个数
#     numTestVecs = int(m * hoRatio)
#     #分类错误计数
#     errorCount = 0.0

#     for i in range(numTestVecs):
#         #前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
#         classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 5)
#         # print("分类结果:%s\t真实类别:%d" % (classifierResult, datingLabels[i]))
#         if classifierResult != datingLabels[i]:
#             errorCount += 1.0
#     print("错误率:%f%%" %(errorCount/float(numTestVecs)*100))
#     # plot_decisionBoundary(normMat[0:numTestVecs,:], datingLabels[0:numTestVecs], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 'KNN分类')
#     # showdatas(datingDataMat, datingLabels)

# if __name__ == '__main__':
#     datingClassTest()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import itertools
from sklearn.preprocessing import MinMaxScaler

# 用于归一化
x_scaler = MinMaxScaler(feature_range=(-1, 1))
y_scaler = MinMaxScaler(feature_range=(-1, 1))

# Read dataset to pandas dataframe
# sklearn库里的数据
iris = load_iris()
col_name = iris['feature_names']#列名
X,y = iris.data,iris.target

X = x_scaler.fit_transform(X)
y = y_scaler.fit_transform(y.reshape(-1,1))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

#训练集
train_result = classifier.predict(X_train)
#测试集
test_result = classifier.predict(X_test)

train_total = 0
for i in range(train_result.shape[0]):
    if train_result[i]==y_train[i]:
        train_total = train_total + 1

test_total = 0
for i in range(test_result.shape[0]):
    if test_result[i]==y_test[i]:
        test_total = test_total + 1

train_precision = train_total/y_train.shape[0] #precision是准确率
test_precision = test_total/y_test.shape[0] #precision是准确率

print('Training precision: ', train_precision)
print('Testing precision: ', test_precision)

subplot_start1 = 321#绘制一个3行2列的图
subplot_start2 = 321#绘制一个3行2列的图
col_numbers = range(0, 4)
col_pairs1 = itertools.combinations(col_numbers, 2)
col_pairs2 = itertools.combinations(col_numbers, 2)
#plt.subplots_adjust(wspace=0.5)

plt.figure(figsize=(12, 12))
for i in col_pairs1:
    plt.subplot(subplot_start1)
    plt.scatter(X_train[:,i[0]], X_train[:,i[1]], c=train_result)
    plt.xlabel(col_name[i[0]])
    plt.ylabel(col_name[i[1]])

    subplot_start1 += 1
plt.savefig(r'D:\VSProject\image\train_iris_KNN.png')

plt.figure(figsize=(12, 12))
for j in col_pairs2:
    plt.subplot(subplot_start2)
    plt.scatter(X_test[:,j[0]], X_test[:,j[1]], c=test_result)
    plt.xlabel(col_name[j[0]])
    plt.ylabel(col_name[j[1]])

    subplot_start2 += 1

plt.savefig(r'D:\VSProject\image\test_iris_KNN.png')