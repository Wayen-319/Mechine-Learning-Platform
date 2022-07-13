# noinspection PyUnresolvedReferences
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
# noinspection PyUnresolvedReferences
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# 导入数据
df = pd.read_table("iris.txt", sep=",", header=None, encoding=None)

X = df[[0, 1, 2, 3]]
Y = df[[4]]

# 准备模型
clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = DecisionTreeClassifier()
clf3 = LogisticRegression()

# 集成模型
sclf = VotingClassifier([('knn', clf1), ('dtree', clf2), ('lr', clf3)], voting='soft')

for clf, label in zip([clf1, clf2, clf3, sclf],
                      ['KNN', 'Decision Tree', 'LogisticRegression', 'VotingClassifier']):
    scores = model_selection.cross_val_score(clf, X, Y, cv=3, scoring='accuracy')
    print("Accuracy: %0.2f [%s]" % (scores.mean(), label))