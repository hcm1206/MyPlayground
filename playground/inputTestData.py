class TestDataset:
    def __init__(self, x_train, t_train, x_test, t_test):
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
    
    def loadTestData(self):
        return self.x_train, self.t_train, self.x_test, self.t_test