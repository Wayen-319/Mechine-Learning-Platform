#Import Library
from sklearn.ensemble import GradientBoostingClassifier
from scipy import io as spio
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import itertools

# # 作图
# def plot_data(X, y, name):    
#     plt.figure(name, figsize=(10, 8))
#     pos = np.where(y == 1)  # 找到y=1的位置
#     neg = np.where(y == 0)  # 找到y=0的位置
#     p1, = plt.plot(np.ravel(X[pos, 0]), np.ravel(X[pos, 1]), 'ro', markersize=8)
#     p2, = plt.plot(np.ravel(X[neg, 0]), np.ravel(X[neg, 1]), 'g^', markersize=8)
#     plt.xlabel("X1")
#     plt.ylabel("X2")
#     plt.legend([p1, p2], ["y==1", "y==0"])
#     # plt.show()
#     return plt


# # 画决策边界
# def plot_decisionBoundary(X, y, model, name):
#     plt = plot_data(X, y, name)
#     # 非线性边界
#     x_1 = np.transpose(np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 213).reshape(1, -1))
#     x_2 = np.transpose(np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 213).reshape(1, -1))
#     X1, X2 = np.meshgrid(x_1, x_2)
#     vals = np.zeros(X1.shape)
#     for i in range(X1.shape[1]):
#         this_X = np.hstack((X1[:, i].reshape(-1, 1), X2[:, i].reshape(-1, 1)))
#         vals[:, i] = model.predict(this_X)

#     plt.contour(X1, X2, vals, [0, 1], color='blue')
    
#     # plt.show()
#     plt.savefig('D:\VSProject\image\\'+name+'.png')

#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create Gradient Boosting Classifier object
model= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
# Train the model using the training sets and check score

# #训练集
# train_x = spio.loadmat('D:\VSProject\data\机器学习平台\\train_x.mat')
# train_y = spio.loadmat('D:\VSProject\data\机器学习平台\\train_y.mat')
# train_x = train_x['X']
# train_y = train_y['y']

# #测试集
# test_x = spio.loadmat('D:\VSProject\data\机器学习平台\\test_x.mat')
# test_y = spio.loadmat('D:\VSProject\data\机器学习平台\\test_y.mat')
# test_x = test_x['X']
# test_y = test_y['y']

# 用于归一化
x_scaler = MinMaxScaler(feature_range=(-1, 1))
y_scaler = MinMaxScaler(feature_range=(-1, 1))

# sklearn库里的数据
iris = load_iris()
col_name = iris['feature_names']#列名
X,y = iris.data,iris.target

X = x_scaler.fit_transform(X)
y = y_scaler.fit_transform(y.reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


train_x = np.array(X_train)
train_y = np.array(y_train)
test_x = np.array(X_test)
test_y = np.array(y_test)


model.fit(train_x, train_y)
#Predict Output
train_result= model.predict(train_x)
test_result= model.predict(test_x)

train_total = 0
for i in range(train_result.shape[0]):
    if train_result[i]==train_y[i]:
        train_total = train_total + 1

test_total = 0
for i in range(test_result.shape[0]):
    if test_result[i]==test_y[i]:
        test_total = test_total + 1

train_precision = train_total/train_y.shape[0] #precision是准确率
test_precision = test_total/test_y.shape[0] #precision是准确率

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
    plt.scatter(train_x[:,i[0]], train_x[:,i[1]], c=train_result)
    plt.xlabel(col_name[i[0]])
    plt.ylabel(col_name[i[1]])

    subplot_start1 += 1
plt.savefig(r'D:\VSProject\image\鸢尾花-训练集GBM.png')

plt.figure(figsize=(12, 12))
for j in col_pairs2:
    plt.subplot(subplot_start2)
    plt.scatter(test_x[:,j[0]], test_x[:,j[1]], c=test_result)
    plt.xlabel(col_name[j[0]])
    plt.ylabel(col_name[j[1]])

    subplot_start2 += 1

plt.savefig(r'D:\VSProject\image\鸢尾花-测试集GBM.png')
# #实际值
# plt.figure(figsize=(10, 8))
# pos = np.where(test_y == 0)  # 找到y=0的位置
# neg = np.where(test_y == 1)  # 找到y=1的位置
# # tem = np.where(test_y == 3)  # 找到y=3的位置
# p1, = plt.plot(np.ravel(test_x[pos, 0]), np.ravel(test_x[pos, 1]), 'ro', markersize=10)
# p2, = plt.plot(np.ravel(test_x[neg, 0]), np.ravel(test_x[neg, 1]), 'bo', markersize=10)
# # p3, = plt.plot(np.ravel(test_x[tem, 0]), np.ravel(test_x[tem, 1]), 'r*', markersize=8)


# #测试值
# pos = np.where(predicted == 0)  # 找到y=0的位置
# neg = np.where(predicted == 1)  # 找到y=1的位置
# # tem = np.where(test_y == 3)  # 找到y=3的位置
# p3, = plt.plot(np.ravel(test_x[pos, 0]), np.ravel(test_x[pos, 1]), 'r^', markersize=8)
# p4, = plt.plot(np.ravel(test_x[neg, 0]), np.ravel(test_x[neg, 1]), 'b^', markersize=8)
# # p3, = plt.plot(np.ravel(test_x[tem, 0]), np.ravel(test_x[tem, 1]), 'r*', markersize=8)
# plt.xlabel("X1")
# plt.ylabel("X2")
# plt.legend([p1, p2, p3, p4], ["y==0", "y==1", "predicted_y==0", "predicted_y==1"])
# plt.title('test GBM')
# plt.savefig('D:\VSProject\image\GBM.png')

# total = 0
# for i in range(predicted.shape[0]):
#     if predicted[i]==test_y[i]:
#         total = total + 1

# precision = total/test_y.shape[0] #precision是准确率
# print('Training precision: ', precision)
# plot_decisionBoundary(test_x, test_y, model, 'GBM') # 默认非线性

# print(predicted)
# print(test_y)