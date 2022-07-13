from xgboost.sklearn import XGBClassifier
from sklearn.datasets import load_iris
from scipy import io as spio
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import itertools
import numpy as np

model = XGBClassifier(
silent=0 ,#设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
#nthread=4,# cpu 线程数 默认最大
learning_rate= 0.05, # 如同学习率
min_child_weight=1, 
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
max_depth=6, # 构建树的深度，越大越容易过拟合
gamma=0.1,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
subsample=1, # 随机采样训练样本 训练实例的子采样比
max_delta_step=0,#最大增量步长，我们允许每个树的权重估计。
colsample_bytree=1, # 生成树时进行的列采样 
reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
#reg_alpha=0, # L1 正则项参数
#scale_pos_weight=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
#objective= 'multi:softmax', #多分类的问题 指定学习任务和相应的学习目标
#num_class=10, # 类别数，多分类与 multisoftmax 并用
n_estimators=100, #树的个数
seed=1000 #随机种子
#eval_metric= 'auc'
)

# 加载样本数据集
# iris = load_iris()
# X,y = iris.data,iris.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234565) # 数据集分割

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
x_scaler = MinMaxScaler(feature_range=(0, 2))
y_scaler = MinMaxScaler(feature_range=(0, 2))

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

# 训练模型
model.fit(train_x, train_y)

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
plt.savefig(r'D:\VSProject\image\鸢尾花-训练集XGB.png')

plt.figure(figsize=(12, 12))
for j in col_pairs2:
    plt.subplot(subplot_start2)
    plt.scatter(test_x[:,j[0]], test_x[:,j[1]], c=test_result)
    plt.xlabel(col_name[j[0]])
    plt.ylabel(col_name[j[1]])

    subplot_start2 += 1

plt.savefig(r'D:\VSProject\image\鸢尾花-测试集XGB.png')
# # 对测试集进行预测
# y_pred = clf.predict(test_x)

# # 计算准确率
# accuracy = accuracy_score(test_y,y_pred)
# print("accuarcy: %.2f%%" % (accuracy*100.0))

# 显示重要特征
plot_importance(model)
plt.savefig(r'D:\VSProject\image\鸢尾花XGB重要特征.png')
plt.show()