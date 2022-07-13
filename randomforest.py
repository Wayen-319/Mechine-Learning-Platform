# noinspection PyUnresolvedReferences
from sklearn.ensemble import RandomForestClassifier
# noinspection PyUnresolvedReferences
from sklearn.preprocessing import StandardScaler
# noinspection PyUnresolvedReferences
from k_cross import K_Flod_spilt
# noinspection PyUnresolvedReferences
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# 导入数据
df = pd.read_table("iris.txt", sep=",", header=None, encoding=None)

X = df[[0, 1, 2, 3]]
Y = df[[4]]

# 交叉验证法
x_train, x_test, y_train, y_test = K_Flod_spilt(10, 1, X, Y)

# 建立模型
RF = RandomForestClassifier(n_estimators=100, n_jobs=4, oob_score=True)

# 数据标准化
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# 训练模型
RF.fit(X_train, y_train)
y_pred = RF.predict(X_test)
print("随机森林准确率:", accuracy_score(y_test, y_pred))
print("其他指标：\n", classification_report(y_test, y_pred, target_names=['setosa', 'versicolor', 'virginica']))