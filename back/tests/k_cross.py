import numpy as np  # 导入numpy包
from sklearn.model_selection import KFold  # 从sklearn导入KFold包
# noinspection PyUnresolvedReferences
import pandas as pd

def K_Flod_spilt(K, P, data, label):
    # K为折数，P为取选取的划分数
    split_list = []
    kf = KFold(n_splits=K, shuffle=True)
    for train, test in kf.split(data):
        split_list.append(train.tolist())
        split_list.append(test.tolist())
    train, test = split_list[2 * P], split_list[2 * P + 1]
    return data.iloc[train, :], data.iloc[test, :], label.iloc[train, :], label.iloc[test, :]