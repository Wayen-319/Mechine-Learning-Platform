from sklearn import svm
import numpy
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import itertools

if __name__=="__main__":
    # train_x = spio.loadmat('D:\VSProject\data\机器学习平台\wine_x.mat')
    # train_y = spio.loadmat('D:\VSProject\data\机器学习平台\wine_y.mat')
    # train_x = train_x['X']
    # train_y = train_y['y']

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


    train_x = numpy.array(X_train)
    train_y = numpy.array(y_train)
    test_x = numpy.array(X_test)
    test_y = numpy.array(y_test)

    clf = svm.SVC(C=5, gamma=0.05,max_iter=200)
    clf.fit(train_x, train_y)


    #Test on Training data
    train_result = clf.predict(train_x[:])
    #Test on Test data
    test_result = clf.predict(test_x[:])

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
    plt.savefig(r'D:\VSProject\image\train_iris_SVM.png')

    plt.figure(figsize=(12, 12))
    for j in col_pairs2:
        plt.subplot(subplot_start2)
        plt.scatter(test_x[:,j[0]], test_x[:,j[1]], c=test_result)
        plt.xlabel(col_name[j[0]])
        plt.ylabel(col_name[j[1]])

        subplot_start2 += 1
    
    plt.savefig(r'D:\VSProject\image\test_iris_SVM.png')