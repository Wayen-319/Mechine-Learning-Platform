import os
import sys
sys.path.append(os.path.abspath(
    os.path.join(
        os.path.abspath(__file__), '..', '..'
    )
))
from matplotlib import pyplot as plt
import scipy.io as scio
import numpy as py
from typing import Any, Tuple
import math
import pandas as pd
import itertools
import seaborn as sns
from sklearn import metrics,svm
from sklearn.metrics import r2_score as r2, mean_squared_error as mse, mean_absolute_error as mae
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_boston,load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from dlframe import DataSet, Splitter, Model, Judger, WebManager,CmdManager
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from k_cross import K_Flod_spilt
import xgboost as xgb
from xgboost import plot_importance
from xgboost.sklearn import XGBClassifier
from IPython.display import Image, display
# noinspection PyUnresolvedReferences
import pydotplus

X_scaler = MinMaxScaler(feature_range=(-1, 1))
y_scaler = MinMaxScaler(feature_range=(-1, 1))
global PH
PH=1

class FinalDataset(DataSet):
    X:py.ndarray
    y:py.ndarray
    X_all:py.ndarray
    col_name:py.ndarray
    def __init__(self, fname) -> None:
        super().__init__()
        if fname=="Boston":
            boston = load_boston()
            X,y = boston.data,boston.target
        elif fname=="Iris":
            iris = load_iris()
            self.col_name = iris['feature_names']#列名
            X,y = iris.data,iris.target
        self.X = X_scaler.fit_transform(X)
        self.y = y_scaler.fit_transform(y.reshape(-1,1))
        self.logger.print("get"+fname)
    
        
    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Any:
        return self.X[idx,:]
    
class TTDataset(DataSet):
    X:py.ndarray
    y:py.ndarray
    
    def __init__(self,z:py.ndarray,z_t:py.ndarray)->None:
        super().__init__()
        self.X=z
        self.y=z_t


    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Any:
        return self.X[idx,:]
    
class FinalSplitter(Splitter):
    # X_train:py.array
    # X_test:py.array
    # y_train:py.array
    # y_test:py.array
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio
        self.logger.print("I'm ratio:{}".format(self.ratio))

    def split(self, dataset: DataSet) -> Tuple[DataSet, DataSet]:
        X_train, X_test, y_train, y_test = train_test_split(dataset.X, dataset.y, test_size=(1-self.ratio))
        trainingSet = TTDataset(X_train,y_train)
        testingSet = TTDataset(X_test,y_test)
        return trainingSet, testingSet

class KflodSplitter(Splitter):
    
    def __init__(self, k) -> None:
        super().__init__()
        self.k = k
        self.logger.print("I'm Kflod:{}".format(self.k))
        
    def split(self, dataset: DataSet) -> Tuple[DataSet, DataSet]:
        df = pd.read_table("iris.txt", sep=",", header=None, encoding=None)
        X = df[[0, 1, 2, 3]]
        Y = df[[4]]
        x_train, x_test, y_train, y_test = K_Flod_spilt(10, 1, X, Y)
        train_x = np.array(x_train)
        train_y = np.array(y_train)
        test_x = np.array(x_test)
        test_y = np.array(y_test)
        trainingSet = TTDataset(train_x,train_y)
        testingSet = TTDataset(test_x,test_y)
        return trainingSet, testingSet
    

class KNN_2(Model):
    test_result:py.ndarray
    classifier = KNeighborsClassifier(n_neighbors=5)
    def __init__(self) -> None:
        super().__init__()


    def train(self, traindata:DataSet) -> None:
        
        self.classifier.fit(traindata.X, traindata.y)
        # self.train_result = classifier.predict(X_train)
        self.logger.print("trainging KNN")
        return super().train(traindata)

    def test(self, testdata: DataSet) -> Any:
        self.test_result = self.classifier.predict(testdata.X)
        iris = load_iris()
        col_name = iris['feature_names']
        global PH
        subplot_start2 = 321#绘制一个3行2列的图
        col_numbers = range(0, 4)
  
        col_pairs2 = itertools.combinations(col_numbers, 2)
        #plt.subplots_adjust(wspace=0.5)

        plt.figure(figsize=(12, 12))
        for j in col_pairs2:
            plt.subplot(subplot_start2)
            plt.scatter(testdata.X[:,j[0]], testdata.X[:,j[1]], c=self.test_result)
            plt.xlabel(col_name[j[0]])
            plt.ylabel(col_name[j[1]])

            subplot_start2 += 1

        plt.savefig(r'E:\XXQ\image\test_iris_KNN'+str(PH)+'.png')
        PH+=1
        self.logger.print("testing")
        return self.test_result

class GBM(Model):
    test_result:py.ndarray
    model= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    def __init__(self) -> None:
        super().__init__()


    def train(self, traindata:DataSet) -> None:
        
        self.model.fit(traindata.X, traindata.y)
        # self.train_result = classifier.predict(X_train)
        self.logger.print("trainging GBM")
        return super().train(traindata)

    def test(self, testdata: DataSet) -> Any:
        iris = load_iris()
        col_name = iris['feature_names']
        self.test_result = self.model.predict(testdata.X)
        global PH
        subplot_start2 = 321#绘制一个3行2列的图
        col_numbers = range(0, 4)
        col_pairs2 = itertools.combinations(col_numbers, 2)
        #plt.subplots_adjust(wspace=0.5)


        plt.figure(figsize=(12, 12))
        for j in col_pairs2:
            plt.subplot(subplot_start2)
            plt.scatter(testdata.X[:,j[0]], testdata.X[:,j[1]], c=self.test_result)
            plt.xlabel(col_name[j[0]])
            plt.ylabel(col_name[j[1]])
            subplot_start2 += 1

        plt.savefig(r'E:\XXQ\image\鸢尾花-测试集GBM'+str(PH)+'.png')
        
        self.logger.print("testing")
        return self.test_result

class Logistic(Model):
    # clf = LogisticRegression(random_state=0, solver='lbfgs')
    # df = pd.read_table("iris.txt", sep=",", header=None, encoding=None)

    # X = df[[0, 1, 2, 3]]
    # Y = df[[4]]

    # # 交叉验证法
    # x_train, x_test, y_train, y_test = K_Flod_spilt(10, 1, X, Y)

    # train_x = np.array(x_train)
    # train_y = np.array(y_train)
    # test_x = np.array(x_test)
    # test_y = np.array(y_test)
     def __init__(self) -> None:
         super().__init__()
  

    # def train(self, trainDataset: DataSet) -> None:
    #     self.clf.fit(trainDataset.X, trainDataset.y)
    #     self.logger.print("trainging Logistic")
    #     return super().train(trainDataset)

    # def test(self, testDataset: DataSet) -> Any:       
        
    #     test_result = clf.predict(testDataset.X)   
    #     self.logger.print('The accuracy of test set:', metrics.accuracy_score(testDataset.y,test_result))
    #     global PH
    #     subplot_start2 = 321# 绘制一个3行2列的图
  
    #     col_pairs2 = itertools.combinations(col_numbers, 2)
    #     col_name = ['sepal length(cm)', 'sepal width(cm)', 'petal length(cm)', 'petal width(cm)']
    #     # cValue = train_y.copy()
    #     cValue2 = []
        
    #     for index in range(0, len(self.test_result)):
    #         if self.test_result[index] == 'Iris-setosa':
    #             cValue2.append('r')
    #         elif self.test_result[index] == 'Iris-versicolor':
    #             cValue2.append('b')
    #         else:
    #             cValue2.append('g')

    #     plt.figure(figsize=(12, 12))
    #     for j in col_pairs2:
    #         plt.subplot(subplot_start2)
    #         plt.scatter(self.test_x[:, j[0]], self.test_x[:, j[1]], c=cValue2)
    #         plt.xlabel(col_name[j[0]])
    #         plt.ylabel(col_name[j[1]])
    #         subplot_start2 += 1

    #     plt.savefig(r'E:\XXQ\image\logistic'+str(PH)+'.png')

    #     # 混淆矩阵
    #     confusion_matrix_result = metrics.confusion_matrix(test_result, self.y_test)
    #     print('The confusion matrix result:\n', confusion_matrix_result)

    #     # 利用热力图对于混淆矩阵进行可视化
    #     ticks = np.unique(train_y)
    #     plt.figure(figsize=(8, 6))
    #     sns.heatmap(confusion_matrix_result, xticklabels=ticks, yticklabels=ticks, annot=True, cmap='YlGnBu')
    #     plt.xlabel('Predicted labels')
    #     plt.ylabel('True labels')
    #     # plt.show()
    #     plt.savefig(r'E:\XXQ\image\logistic-混淆矩阵'+str(PH)+'.png')

    #     return test_result
   
class Decision_Tree(Model):
    # clf = DecisionTreeClassifier()

    # df = pd.read_table("iris.txt", sep=",", header=None, encoding=None)

    # X = df[[0, 1, 2, 3]]
    # Y = df[[4]]
    # feature = ['sepal length(cm)', 'sepal width(cm)', 'petal length(cm)', 'petal width(cm)']

    # # 交叉验证法
    # x_train, x_test, y_train, y_test = K_Flod_spilt(10, 1, X, Y)
     def __init__(self) -> None:
         super().__init__()
    
    # def train(self, traindata:DataSet) -> None:
        
    #     self.clf.fit(self.x_train, self.y_train)
    #     # self.train_result = classifier.predict(X_train)
    #     self.logger.print("trainging DecisionTree")
    #     return super().train(traindata)
    
    # def test(self, testdata: DataSet) -> Any:
    #     global PH
    #     test_predict = clf.predict(self.x_test)
    #     subplot_start2 = 321# 绘制一个3行2列的图
  
    #     col_pairs2 = itertools.combinations(col_numbers, 2)
    #     col_name = ['sepal length(cm)', 'sepal width(cm)', 'petal length(cm)', 'petal width(cm)']
    #     # cValue = train_y.copy()
    #     cValue2 = []
        
    #     for index in range(0, len(self.test_result)):
    #         if self.test_result[index] == 'Iris-setosa':
    #             cValue2.append('r')
    #         elif self.test_result[index] == 'Iris-versicolor':
    #             cValue2.append('b')
    #         else:
    #             cValue2.append('g')

    #     plt.figure(figsize=(12, 12))
    #     for j in col_pairs2:
    #         plt.subplot(subplot_start2)
    #         plt.scatter(test_x[:, j[0]], test_x[:, j[1]], c=cValue2)
    #         plt.xlabel(col_name[j[0]])
    #         plt.ylabel(col_name[j[1]])
    #         subplot_start2 += 1
    #     self.logger.print("testing")
    #     self.logger.print('The accuracy of test set:', metrics.accuracy_score(self.y_test, self.test_predict))
        
    #     plt.savefig(r'E:\XXQ\image\DecisionTree'+str(PH)+'.png')
    #     return test_predict
        
class NavieBayes(Model):
    test_result:py.ndarray
    clf:GaussianNB
    def __init__(self) -> None:
        super().__init__()


    def train(self, traindata:DataSet) -> None:
        self.clf=GaussianNB().fit().fit(traindata.X,traindata.y)  
        # self.train_result = classifier.predict(X_train)
        self.logger.print("trainging NavieBayes")
        return super().train(traindata)

    def test(self, testdata: DataSet) -> Any:
        iris = load_iris()
        col_name = iris['feature_names']
        self.test_result = self.clf.predict(testdata.X)
        self.test_result = self.test_result.reshape(-1, 1)
        global PH

        subplot_start2 = 321#绘制一个3行2列的图
        col_numbers = range(0, 4)

        col_pairs2 = itertools.combinations(col_numbers, 2)
        #plt.subplots_adjust(wspace=0.5)

        plt.figure(figsize=(12, 12))
        for j in col_pairs2:
            plt.subplot(subplot_start2)
            plt.scatter(testdata.X[:,j[0]], testdata.X[:,j[1]], c=self.test_result)
            plt.xlabel(col_name[j[0]])
            plt.ylabel(col_name[j[1]])

            subplot_start2 += 1

        plt.savefig(r'E:\XXQ\image\鸢尾花-测试集NB'+str(PH)+'.png')
        
        self.logger.print("testing NavieBayes")
        return self.test_result        

class SVM(Model):
    test_result:py.ndarray
    clf = svm.SVC(C=5, gamma=0.05,max_iter=200)
    def __init__(self) -> None:
        super().__init__()


    def train(self, traindata:DataSet) -> None:
        self.clf.fit(traindata.X,traindata.y)  
        # self.train_result = classifier.predict(X_train)
        self.logger.print("trainging SVM")
        return super().train(traindata)

    def test(self, testdata: DataSet) -> Any:
        iris = load_iris()
        col_name = iris['feature_names']
        self.test_result = self.clf.predict(testdata.X)
        
        global PH

        subplot_start2 = 321#绘制一个3行2列的图
        col_numbers = range(0, 4)
        col_pairs2 = itertools.combinations(col_numbers, 2)
        #plt.subplots_adjust(wspace=0.5)

        plt.figure(figsize=(12, 12))
        for j in col_pairs2:
            plt.subplot(subplot_start2)
            plt.scatter(testdata.X[:,j[0]], testdata.y[:,j[1]], c=self.test_result)
            plt.xlabel(col_name[j[0]])
            plt.ylabel(col_name[j[1]])

            subplot_start2 += 1
        
        plt.savefig(r'E:\XXQ\image\test_iris_SVM'+str(PH)+'.png')
        
        self.logger.print("testing SVM")
        return self.test_result        

class SVR(Model):
    test_result:py.ndarray
    clf = svm.SVR(kernel='rbf',C=10, gamma = 0.01)
    def __init__(self) -> None:
        super().__init__()


    def train(self, traindata:DataSet) -> None:
        self.clf.fit(traindata.X,traindata.y)  
        # self.train_result = classifier.predict(X_train)
        self.logger.print("trainging SVR")
        return super().train(traindata)

    def test(self, testdata: DataSet) -> Any:
       
        self.test_result = self.clf.predict(testdata.X)
        
        global PH

        plt.rcParams['font.sans-serif'] = [u'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(12, 12))
        plt.plot(self.test_result, label='pred')
        plt.plot(testdata.y, label='actual')
        plt.title('波士顿房价-测试集')
        plt.legend()
        plt.savefig('E:\XXQ\image\波士顿房价-测试集SVR'+str(PH)+'.png')
  
        
        self.logger.print("testing SVR")
        return self.test_result    

class TreeJudger(Judger):
    def __init__(self) -> None:
        super().__init__()
        
    def judge(self, test_result, test_dataset: DataSet) -> None:    
        
        return super().judge(test_result, test_dataset)   
       

class AccuracyJudger(Judger):
    def __init__(self) -> None:
        super().__init__()

    def judge(self, test_result, test_dataset: DataSet) -> None:
        test_total = 0
        for i in range(test_result.shape[0]):
            if test_result[i]==test_dataset.y[i]:
                test_total = test_total + 1
        test_precision = test_total/test_dataset.y.shape[0]
        self.logger.print("test_result =",test_result)
        self.logger.print("gt = ",test_precision)
        return super().judge(test_result, test_dataset)

class MSEJudger(Judger):
    def __init__(self) -> None:
        super().__init__()

    def judge(self, test_result, test_dataset: DataSet) -> None:
        self.logger.print('均方误差(mean squared error)为:',mse(test_dataset.y, test_result))
        return super().judge(test_result, test_dataset)

class XGBoost_C(Model):
    test_result:py.ndarray
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
    def __init__(self) -> None:
        super().__init__()


    def train(self, traindata:DataSet) -> None:
        self.model.fit(traindata.X,traindata.y)  
        # self.train_result = classifier.predict(X_train)
        self.logger.print("trainging XGBoost_C")
        return super().train(traindata)

    def test(self, testdata: DataSet) -> Any:
        iris = load_iris()
        col_name = iris['feature_names']
        self.test_result= self.model.predict(testdata.X)
        
        global PH

        subplot_start2 = 321#绘制一个3行2列的图
        col_numbers = range(0, 4)
        col_pairs2 = itertools.combinations(col_numbers, 2)
        #plt.subplots_adjust(wspace=0.5)

        plt.figure(figsize=(12, 12))
        for j in col_pairs2:
            plt.subplot(subplot_start2)
            plt.scatter(testdata.X[:,j[0]], testdata.y[:,j[1]], c=self.test_result)
            plt.xlabel(col_name[j[0]])
            plt.ylabel(col_name[j[1]])

            subplot_start2 += 1
        
        plt.savefig(r'E:\XXQ\image\鸢尾花-测试集XGB'+str(PH)+'.png')
        
        self.logger.print("testing XGBoost_C")
        return self.test_result        

class XGBoost_R(Model):
    test_result:py.ndarray
    model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True)
    def __init__(self) -> None:
        super().__init__()


    def train(self, traindata:DataSet) -> None:
        self.model.fit(traindata.X,traindata.y)  
        # self.train_result = classifier.predict(X_train)
        self.logger.print("trainging XGBoost_R")
        return super().train(traindata)

    def test(self, testdata: DataSet) -> Any:
        iris = load_iris()
        col_name = iris['feature_names']
        self.test_result= self.model.predict(testdata.X)
        
        global PH

        plt.rcParams['font.sans-serif'] = [u'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(12, 12))
        plt.plot(self.test_result.reshape(-1,1), label='pred')
        plt.plot(testdata.y, label='actual')
        plt.title('波士顿房价-测试集')
        plt.legend()
        plt.savefig('E:\XXQ\image\波士顿房价-测试集XGB'+str(PH)+'.png')
        plot_importance(self.model)
        plt.savefig('E:\XXQ\image\波士顿房价XGB重要特征'+str(PH)+'.png')
        
        self.logger.print("testing XGBoost_R")
        return self.test_result    

class RegressionJudger(Judger):
    def __init__(self) -> None:
        super().__init__()

    def judge(self, y_hat, test_dataset: DataSet) -> None:
        
        return super().judge(y_hat, test_dataset)

if __name__ == '__main__':
    WebManager().register_dataset(
        FinalDataset("Boston"), '波士顿房价'
    ).register_dataset(
        FinalDataset("Iris"), '鸢尾花'
    ).register_splitter(
        FinalSplitter(0.8), 'ratio:0.8'
    ).register_splitter(
        KflodSplitter(10), 'Kflod:10'
    ).register_model(
        KNN_2(),"KNN"
    ).register_model(
        GBM(),"GBM"
    ).register_model(
        Logistic(),"Logistic"
    ).register_model(
        NavieBayes(),"NavieBayes"
    ).register_model(
        SVR(),"SVR"
    ).register_model(
        SVM(),"SVM"
    ).register_model(
        XGBoost_C(),"XGBoost_Classifier"
    ).register_model(
        XGBoost_R(),"XGBoost_Regression"
    ).register_model(
        Decision_Tree(),"DecisionTree"
    ).register_judger(
        AccuracyJudger()
    ).register_judger(
        MSEJudger()
    ).register_judger(
        RegressionJudger()
    ).register_judger(
        TreeJudger()
    ).start()
