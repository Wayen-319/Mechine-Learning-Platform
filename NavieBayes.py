import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
from scipy import io as spio
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
import itertools

# 作图
def plot_data(X, y, name):    
    plt.figure(name, figsize=(10, 8))
    pos = np.where(y == 1)  # 找到y=1的位置
    neg = np.where(y == 0)  # 找到y=0的位置
    p1, = plt.plot(np.ravel(X[pos, 0]), np.ravel(X[pos, 1]), 'ro', markersize=8)
    p2, = plt.plot(np.ravel(X[neg, 0]), np.ravel(X[neg, 1]), 'g^', markersize=8)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend([p1, p2], ["y==1", "y==0"])
    return plt


# 画决策边界
def plot_decisionBoundary(X, y, model, name):
    plt = plot_data(X, y, name)

    x_1 = np.transpose(np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100).reshape(1, -1))
    x_2 = np.transpose(np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100).reshape(1, -1))
    X1, X2 = np.meshgrid(x_1, x_2)
    vals = np.zeros(X1.shape)
    for i in range(X1.shape[1]):
        this_X = np.hstack((X1[:, i].reshape(-1, 1), X2[:, i].reshape(-1, 1)))
        vals[:, i] = model.predict(this_X)

    plt.contour(X1, X2, vals, [0, 1], color='blue')
    plt.show()

# train_x = spio.loadmat(r'D:\VSProject\data\机器学习平台\train_x.mat')
# train_y = spio.loadmat(r'D:\VSProject\data\机器学习平台\train_y.mat')
# test_x = spio.loadmat(r'D:\VSProject\data\机器学习平台\test_x.mat')
# test_y = spio.loadmat(r'D:\VSProject\data\机器学习平台\test_y.mat')
# train_x = train_x['X']
# train_y = train_y['y']
# test_x = test_x['X']
# test_y = test_y['y']

# train_x = np.array(train_x)
# train_y = np.array(train_y) 
# test_x = np.array(test_x)
# test_y = np.array(test_y) 

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


train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.20)

# 调用GaussianNB分类器，假定数据服从正太分布
clf=GaussianNB().fit(train_x,train_y)      #训练

train_result = clf.predict(train_x)  #训练集
test_result = clf.predict(test_x)    #测试集

test_result = test_result.reshape(-1, 1)
train_result = train_result.reshape(-1, 1)

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
plt.savefig(r'D:\VSProject\image\鸢尾花-训练集NB.png')

plt.figure(figsize=(12, 12))
for j in col_pairs2:
    plt.subplot(subplot_start2)
    plt.scatter(test_x[:,j[0]], test_x[:,j[1]], c=test_result)
    plt.xlabel(col_name[j[0]])
    plt.ylabel(col_name[j[1]])

    subplot_start2 += 1

plt.savefig(r'D:\VSProject\image\鸢尾花-测试集NB.png')
# plt.rcParams['font.sans-serif'] = [u'SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# plt.figure(figsize=(12, 12))
# plt.plot(train_y, label='actual')
# plt.plot(train_result, label='pred')
# plt.title('鸢尾花-训练集')
# plt.legend()
# plt.savefig('D:\VSProject\image\鸢尾花-训练集NB.png')
# plt.show()

# plt.rcParams['font.sans-serif'] = [u'SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# plt.figure(figsize=(12, 12))
# plt.plot(test_y, label='actual')
# plt.plot(test_result, label='pred')
# plt.title('鸢尾花-测试集')
# plt.legend()
# plt.savefig('D:\VSProject\image\鸢尾花-测试集nb.png')
# plt.show()


# print(' 训练集-SVR的均方误差(mean squared error)为:',mse(train_y, train_result))
# print(' 测试集-SVR的均方误差(mean squared error)为:',mse(test_y, test_result))

# print(test_y)                  #输出实际结果
# print(doc_class_predicted)     #输出测试结果
#结果报告输出
# plot_decisionBoundary(test_x, test_y, clf, "navie bayes")
# print(metrics.classification_report(test_y, test_result))    #输出结果，精确度、召回率、f-1分数
# print(metrics.confusion_matrix(test_y, test_result))         #混淆矩阵

