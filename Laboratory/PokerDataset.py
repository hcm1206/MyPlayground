import numpy as np

class PokerDataset:
    def __init__(self, x_train, t_train, x_test, t_test):
        self.x_train = np.array(x_train)
        self.t_train = np.array(t_train)
        self.x_test = np.array(x_test)
        self.t_test = np.array(t_test)


    
    def loadTestData(self):

        return (self.x_train, self.t_train), (self.x_test, self.t_test)