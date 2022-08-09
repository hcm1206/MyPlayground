import random
import pickle
import numpy as np
from PokerDataset import PokerDataset
from Cards import Cards
from CheckJokbo import getScore

CardDeck = Cards()

x_train = []
t_train = []
x_test = []
t_test = []



for i in range(10000):
    allDeck = []
    CardDeck.initDeck()
    allDeck += CardDeck.drawCpuCard()
    allDeck += CardDeck.drawInitCommonCard()
    allDeck += CardDeck.drawMyCard()
    cpuDeck = np.array(allDeck[:7])
    myDeck = np.array(allDeck[2:])

    cpuScore = getScore(cpuDeck)
    myScore = getScore(myDeck)

    if cpuScore < myScore:
        win = np.array([0,1])
    else:
        win = np.array([1,0])

    x_train.append(cpuDeck)
    t_train.append(win)


for i in range(2000):
    allDeck = []
    CardDeck.initDeck()
    allDeck += CardDeck.drawCpuCard()
    allDeck += CardDeck.drawInitCommonCard()
    allDeck += CardDeck.drawMyCard()
    cpuDeck = allDeck[:7]
    myDeck = allDeck[2:]

    cpuScore = getScore(cpuDeck)
    myScore = getScore(myDeck)

    if cpuScore < myScore:
        win = np.array([0,1])
    else:
        win = np.array([1,0])

    x_test.append(cpuDeck)
    t_test.append(win)

testData = PokerDataset(x_train, t_train, x_test, t_test)

with open("PokerTestDataSet.pickle", "wb") as fw:
    pickle.dump(testData, fw)

with open("PokerTestDataSet.pickle", "rb") as fr:
    dataset = pickle.load(fr)

_x_train, _t_train, _x_test, _t_test = dataset.loadTestData()
