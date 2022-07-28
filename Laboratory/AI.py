from TwoLayerNet import TwoLayerNet
from CheckJokbo import getScore
from Cards import Cards
import numpy as np

CardDeck = Cards()

network = TwoLayerNet(input_size = 9, hidden_size=50, output_size=2)
iters_num = 100000
iters_per_epoch = 10000
learning_rate = 0.1

train_loss_list = []
train_acc_list = []

for i in range(iters_num):
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

    inputDeck = []
    inputDeck.append(allDeck)
    inputDeck = np.array([inputDeck])

    grad = network.gradient(inputDeck, win)

    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(inputDeck, win)
    train_loss_list.append(loss)
    if i % iters_per_epoch == 0:
        train_acc = network.accuracy(inputDeck, win)
        train_acc_list.append(train_acc)
        print("===================== " + str(i) + " learning =====================")
        print("predict : " + str(network.predict(inputDeck)))
        print("win" + str(win))
        print("accuracy : " + str(train_acc))