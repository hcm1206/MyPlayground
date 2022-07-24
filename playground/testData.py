import random
import pickle
from inputTestData import TestDataset


def func1(x):
    return x * (x+1) - (x-1)


x_train = []
t_train = []
x_test = []
t_test = []



for i in range(10000):
    x = random.randrange(10000)
    x_train.append(x)
    t_train.append(func1(x))

for i in range(2000):
    x = random.randrange(10000)
    x_test.append(x)
    t_test.append(func1(x))

print(x_test[4])

testData = TestDataset(x_train, t_train, x_test, t_test)

with open("TestDataSet.pickle", "wb") as fw:
    pickle.dump(testData, fw)

with open("TestDataSet.pickle", "rb") as fr:
    dataset = pickle.load(fr)

_x_train, _t_train, _x_test, _t_test = dataset.loadTestData()

print(_x_test[4])