from TwoLayerNet import TwoLayerNet
import numpy as np
import pickle

# with open("PokerTestDataSet.pickle", "rb") as fr:
#     dataset = pickle.load(fr)

datas = np.load("PokerTestDataSet.npz")

x_train = datas['x_train']
t_train = datas['t_train']
x_test = datas['x_test']
t_test = datas['t_test']

# (x_train, t_train), (x_test, t_test) = dataset.loadTestData()
network = TwoLayerNet(input_size = 7, hidden_size=7, output_size=2)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iters_per_epoch = max(train_size/batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)

    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    if i % iters_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        print("===================== " + str(i) + " learning =====================")
        print(train_acc, test_acc)