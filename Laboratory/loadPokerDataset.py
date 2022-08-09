import numpy as np
import os
import pickle



scriptpath = os.path.dirname(__file__)
file = os.path.join(scriptpath, 'PokerTestDataSet.pickle')

with open(file, "rb") as fr:
    dataset = pickle.load(fr)

(x_train, t_train), (x_test, t_test) = dataset.loadTestData()

print(x_train[4])
print(t_train[4])