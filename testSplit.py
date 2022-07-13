import random

def train_test_split(data, label, test_size):
    length = len(data)
    train_index = list(range(length))
    test_index = random.sample(train_index, int(length*test_size))
    for x in test_index:
        train_index.remove(x)
    if label is not None:
        return data.iloc[train_index], data.iloc[test_index], label.iloc[train_index], label.iloc[test_index]
    else:
        return data.iloc[train_index], data.iloc[test_index]