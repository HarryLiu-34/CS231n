import numpy as np

#Lec2. Spend much time in testing time, while do nothing(lazy) at training
class NearestNeighbour:
    def train(self, X_train, y):
        '''X is N * D where each row is an example. Y stands for label is 1-dimension of size N'''
        #the nearest neighbour classifier simply remembers all the training data
        self.Xtr = X_train
        self.ytr = y

    def predict(self, X_test):
        '''X is N * D where each row is an example we wish to predict label for'''
        num_test = X_test.shape[0]
        #lets make sure that the output type (type of Ypred) matches the input type (type of ytr)
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

        #look over all test rows
        #xrange produces an iterator rather than an array, different from range
        for i in range(num_test):
            #find the nearest training image to the ith test image
            #using the L1 distance (sum of absolute value diff)
            distances = np.sum(np.abs(self.Xtr - X_test[i, :]), axis = 1)    # axis = 1 stands for suming-up same row values, see sum_demo.py for detail
            min_index = np.argmin(distances)    # get the label with the smallest distance      
            Ypred[i] = self.ytr[min_index]    # predict the label same as its nearest example
        return Ypred

if __name__ == "__main__":
    X_train = np.array([[1,2,5,6], [8,1,3,4], [5,6,6,7]])
    y = np.array([0,1,2,3])
    X_test = np.array([[1,3,4,7], [5,5,5,5]])

    NN = NearestNeighbour()
    NN.train(X_train, y)
    print(NN.predict(X_test))
