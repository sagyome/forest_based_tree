import numpy as np
import pandas as pd
from random import shuffle
def divide_to_train_test(data,x_columns,y_column,train_ratio=0.8):
    idx = np.arange(data.shape[0])
    shuffle(idx)
    train_indexes = idx[:int(len(idx)*train_ratio)]
    test_indexes = idx[int(len(idx) * train_ratio):]
    X = data[x_columns].as_matrix()
    Y = data[y_column].values
    return X[train_indexes],Y[train_indexes],X[test_indexes],Y[test_indexes]

