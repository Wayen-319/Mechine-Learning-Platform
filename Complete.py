import os
import sys
sys.path.append(os.path.abspath(
    os.path.join(
        os.path.abspath(__file__), '..', '..'
    )
))

import scipy.io as scio
import numpy as py
from typing import Any, Tuple
import math
import itertools
from sklearn.metrics import r2_score as r2, mean_squared_error as mse, mean_absolute_error as mae
from sklearn.datasets import load_boston,load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from dlframe import DataSet, Splitter, Model, Judger, WebManager,CmdManager
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import GradientBoostingClassifier

X_scaler = MinMaxScaler(feature_range=(-1, 1))
y_scaler = MinMaxScaler(feature_range=(-1, 1))

class FinalDataset(DataSet):
    X:py.ndarray
    y:py.ndarray
    X_all:py.ndarray
    def __init__(self, fname) -> None:
        super().__init__()
        if fname=="Boston":
            boston = load_boston()
            X,y = boston.data,boston.target
        elif fname=="Iris":
            iris = load_iris()
            col_name = iris['feature_names']#列名
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
        X_train, X_test, y_train, y_test = K_Flod_spilt(k, 1, X, Y)
        trainingSet = TTDataset(X_train,y_train)
        testingSet = TTDataset(X_test,y_test)
        return trainingSet, testingSet
    

class KNNModel(Model):
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
        self.test_result = self.model.predict(testdata.X)
        self.logger.print("testing")
        return self.test_result
    
class Decision_Tree(Model):
    clf = DecisionTreeClassifier()
    test_result:py.ndarray
    def __init__(self) -> None:
        super().__init__()
    
    def train(self, traindata:DataSet) -> None:
        
        self.clf.fit(traindata.X, traindata.y)
        # self.train_result = classifier.predict(X_train)
        self.logger.print("trainging DecisionTree")
        return super().train(traindata)
    
    def test(self, testdata: DataSet) -> Any:
        self.test_result = self.clf.predict(testdata.X)
        self.logger.print("testing")
        return self.test_result
        
class TreeJudger(Judger):
    def __init__(self) -> None:
        super().__init__()
        
    def judge(self, test_result, test_dataset: DataSet) -> None:    
        self.logger.print('The accuracy of test set:', metrics.accuracy_score(test_data.y, test_result))
        return super().judge(test_result, test_dataset)   
       

class ZhunJudger(Judger):
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

class TestSplitter(Splitter):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio
        self.logger.print("I'm ratio:{}".format(self.ratio))

    def split(self, dataset: DataSet) -> Tuple[DataSet, DataSet]:
        trainingSet = [dataset[i] for i in range(math.floor(len(dataset) * self.ratio))]
        trainingSet = TrainTestDataset(trainingSet)

        testingSet = [dataset[i] for i in range(math.floor(len(dataset) * self.ratio), len(dataset))]
        testingSet = TrainTestDataset(testingSet)

        self.logger.print("split!")
        self.logger.print("training_len = {}".format([trainingSet[i] for i in range(len(trainingSet))]))
        return trainingSet, testingSet

class TestModel(Model):
    def __init__(self, learning_rate) -> None:
        super().__init__()
        self.learning_rate = learning_rate

    def train(self, trainDataset: DataSet) -> None:
        self.logger.print("trainging, lr = {}".format(self.learning_rate))
        return super().train(trainDataset)

    def test(self, testDataset: DataSet) -> Any:
        self.logger.print("testing")
        return testDataset

class TestJudger(Judger):
    def __init__(self) -> None:
        super().__init__()

    def judge(self, y_hat, test_dataset: DataSet) -> None:
        self.logger.print("y_hat = {}".format([y_hat[i] for i in range(len(y_hat))]))
        self.logger.print("gt = {}".format([test_dataset[i] for i in range(len(test_dataset))]))
        return super().judge(y_hat, test_dataset)

if __name__ == '__main__':
    WebManager().register_dataset(
        FinalDataset("Boston"), '波士顿放假'
    ).register_dataset(
        FinalDataset("Iris"), '鸢尾花'
    ).register_splitter(
        FinalSplitter(0.8), 'ratio:0.8'
    ).register_splitter(
        KflodSplitter(10), 'Kflod:10'
    ).register_model(
        KNNModel(),"KNN"
    ).register_model(
        GBM(),"GBM"
    ).register_model(
        Decision_Tree(),"DecisionTree"
    ).register_judger(
        ZhunJudger()
    ).register_judger(
        TreeJudger()
    ).start()
