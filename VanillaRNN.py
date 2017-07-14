import numpy as np
import re
import math
import random
from Tools import *


class MyRNN:
    # Parameters
    N = T = D = H = eta = n_train = 0

    # Variables
    X = Y = O = S = np.empty((N, T))

    # weights
    U = W = V = np.empty((D, H))

    # Input parameters
    dictionary = train = val = test = []

    # Main RNN constructor
    def __init__(self, dictionary, train, val, test, H, eta):
        self.dictionary, self.train, self.val, self.test = dictionary, train, val, test

        self.D = len(dictionary)
        self.T = max(len(s) for s in (train + val + test)) + 2
        self.H, self.eta = H, eta
        self.n_train = len(train)
        # print('Total number of phrases: ', self.N)

        # Weight assignment with Glorot uniform
        wgtD = self.D ** (-0.5)
        wgtH = self.H ** (-0.5)

        self.U = np.random.uniform(-wgtD, wgtD, (self.H, self.D))  # HxD matrix
        self.W = np.random.uniform(-wgtH, wgtH, (self.H, self.H))  # HxH matrix
        self.V = np.random.uniform(-wgtH, wgtH, (self.D, self.H))  # DxH matrix

    # Main parameter initializator
    def init_mainParam(self, data):
        self.N = len(data)
        # print('N dimension: ', self.N)
        # Preparation of X
        self.X = np.zeros((self.N, self.T, self.D))
        self.X[:, 0, 0] = 1
        #self.X[:, 1:, 2] = 1

        for s in data:
            i = data.index(s) #Index of the phrase
            j, k = 0, 0
            for j in range(len(s)):
                # j = s.index(w) + 1  # Index of the word in the phrase +1 for the <startWD> token
                if s[j] != '':
                    w = s[j]
                    j += 1
                    k = self.dictionary.index(w) #Index of the word in the dictionary
                    self.X[i, j, k] = 1
                    #self.X[i, j, 2] = 0

            #self.X[i, j + 1, 2] = 0
            #self.X[i, j + 1, 1] = 1

        # Preparation of Y
        self.Y = np.zeros((self.N, self.T, self.D))
        self.Y[:, :-1, :] = self.X[:, 1:, :]
        #self.Y[:, -1:, 2] = 1
        self.S = np.zeros((self.N, self.T, self.H))
        self.O = np.zeros((self.N, self.T, self.D))

    # Function that implements the activation of the hidden layer
    def outHL(self, x, s_prev):
        return np.tanh(np.dot(self.U, x) + np.dot(self.W, s_prev))

    # Forward pass of the RNN
    def fwdRnn(self, X, S, O):
        # 1. s = tanh(Ux + Ws_prev)
        # 2. o = sigma(Vs)
        # 3. U,W,V = L(y,o)

        for t in range(self.T):
            S[:, t, :] = self.outHL(X[:, t, :].T, S[:, t - 1, :].T).T
            O[:, t, :] = self.softmax(self.V.dot(S[:, t, :].T)).T
        return S, O

    # Backward pass of the RNN
    def bwRnn(self):
        # Evaluation of dLdV
        dLdO = -self.Y / self.O
        dOdVS = self.O * (1.0 - self.O)
        dLdV = np.tensordot(dLdO*dOdVS, self.S, axes=((0, 1), (0, 1)))
        c = self.eta * (1.0 / (self.n_train * self.T * self.D)) #Constant value including eta and 1/n
        # New matrix V
        Vnew = self.V - c * dLdV

        # Evaluation of dLdU
        S0 = np.zeros(self.S.shape)
        S0[:, 1:, :] = self.S[:, :-1, :]
        dS_dargTanh1 = 1 - self.S # Decomposition of dSdargTanh = tanh' = 1 - tanh^2 = (1 + tanh)(1 - tanh)
        dS_dargTanh2 = 1 + self.S # Decomposition of dSdargTanh = tanh' = 1 - tanh^2 = (1 + tanh)(1 - tanh)
        dLdS = np.tensordot(dLdO * dOdVS, self.V, axes=(2, 0))  # returns an NxTxH matrix
        dL_dargTanh1 = np.tensordot(dLdS, dS_dargTanh1, axes=((0, 1), (0, 1)))  # returns an HxH matrix
        dargTanh2_dU = np.tensordot(dS_dargTanh2, self.X, axes=((0, 1), (0, 1)))  # returns an HxD matrix
        dLdU = dL_dargTanh1.dot(dargTanh2_dU)
        # New matrix U
        Unew = self.U - c * dLdU
        # print('U aggiornato con dimensioni = ',Unew.shape)

        # Evaluation of dLdW
        dargTanh2_dW = np.tensordot(dS_dargTanh2, S0, axes=((0, 1), (0, 1)))  # returns an HxH matrix
        dLdW = dL_dargTanh1.dot(dargTanh2_dW)  # returns an HxH matrix
        Wnew = self.W - c * dLdW
        # print('W aggiornato con dimensione = ',Wnew.shape)

        return (Vnew, Unew, Wnew)

    '''
    # Second version of the Backward step of the RNN
    def weight_update(self):

        # V update

        # U update

        # W update
    '''

    # Function that implements the softmax computation
    def softmax(self,s):
        # Softmax over 2D-matrix if D dimension is on axis = 0
        s -= np.amax(s, axis=0)
        s = np.exp(s)
        return s / np.sum(s, axis=0)

    # Function that implements the loss function computation
    def lossFunction(self):
        '''
        o = o.transpose()
        a = -y*np.log(o)
        return a.sum()
        '''
        O_ = np.log(self.O)
        c = -1 / self.N
        return c * np.tensordot(self.Y, O_, axes=((0, 1, 2), (0, 1, 2)))

    # Function that implements the forward step in the RNN
    def training_step(self, K, mini_batch_size):
        lossTrain, lossVal = [], []
        lossT = 0.0
        idxTrain = list(range(self.n_train))

        for i in range(K):
            print('Epoch ', i, ':')
            # Training set computation
            random.shuffle(idxTrain)
            #self.N = 500
            self.N = mini_batch_size
            print('Train dimension: ', len(self.train))
            print('N dimension: ', self.N)
            print('Iteration range: ', len(self.train) // self.N)

            for j in range(len(self.train) // self.N):
                print('Iteration limits: [', j * self.N, ', ', j * self.N + self.N, ')')
                self.init_mainParam([self.train[idxTrain[k]] for k in range(j * self.N, j * self.N + self.N)])
                self.S, self.O = self.fwdRnn(self.X, self.S, self.O)
                self.V, self.U, self.W = self.bwRnn()
                lossT += self.lossFunction()
            #lossTrain += [lossT / 100]
            lossTrain += [lossT / (self.n_train // self.N)]
            lossT = 0.0

            # Validation set computation
            self.init_mainParam(self.val)
            self.fwdRnn(self.X, self.S, self.O)
            lossVal += [self.lossFunction()]

        return lossTrain, lossVal

    def test_step(self):
        self.init_mainParam(self.test)
        self.fwdRnn(self.X, self.S, self.O)
        print('N = ', self.N)
        lossTest = self.lossFunction()
        accTest = self.accuracy()
        return lossTest, accTest

    def accuracy(self):
        O_ = self.O.argmax(axis=2)
        Y_ = self.Y.argmax(axis=2)
        compRes = Y_ == O_
        return compRes.sum()/(self.N*self.T)
