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

        # self.U = np.random.uniform(-wgtD, wgtD, (self.H, self.D))  # HxD matrix
        # self.W = np.random.uniform(-wgtH, wgtH, (self.H, self.H))  # HxH matrix
        # self.V = np.random.uniform(-wgtH, wgtH, (self.D, self.H))  # DxH matrix
        self.U = np.random.randn(self.H, self.D) * 0.01
        self.W = np.random.randn(self.H, self.H) * 0.01
        self.V = np.random.randn(self.D, self.H) * 0.01

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

    # Function that implements the softmax computation
    def softmax(self, s):
        # Softmax over 2D-matrix if D dimension is on axis = 0
        s -= np.amax(s, axis=0)
        s = np.exp(s)
        return s / np.sum(s, axis=0)

    # Function that implements the forward pass of the RNN
    def fwdRnn(self, X, S, O):
        # 1. s = tanh(Ux + Ws_prev)
        # 2. o = sigma(Vs)
        # 3. U,W,V = L(y,o)

        for t in range(self.T):
            S[:, t, :] = self.outHL(X[:, t, :].T, S[:, t - 1, :].T).T
            O[:, t, :] = self.softmax(self.V.dot(S[:, t, :].T)).T
        return S, O

    # Function that implements the backward pass of the RNN
    def bwRnn(self):

        '''
        #Good version

        # Evaluation of dLdV
        dL_dO = self.Y / self.O
        dO_dVS = self.O * (1.0 - self.O)
        dL_dV = np.tensordot(dL_dO*dO_dVS, self.S, axes=((0, 1), (0, 1)))
        c = (-self.eta) / (self.N * self.T) # Constant value including eta and 1/n
        # New matrix V
        Vnew = self.V - c * dL_dV

        # Verifying the Good and the Third versions
        A = - self.Y + (self.Y * self.O)
        dL_dO = self.Y / self.O
        dO_dVS = self.O * (1.0 - self.O)
        B = dL_dO*dO_dVS
        print('Uguaglianza tra versione semplificata ed espansa: ', A == B)
        '''

        # Third version - As good as the Good version
        
        # Evaluation of dL_dV
        dL_dVS = - self.Y + (self.Y * self.O)
        dL_dV = np.tensordot(dL_dVS, self.S, axes=((0, 1), (0, 1)))
        c = self.eta / (self.n_train * self.T)  # Constant value including eta and 1/n
        #c = self.eta
        # New matrix V
        Vnew = self.V - c * dL_dV


        # Evaluation of dLdU
        S0 = np.zeros(self.S.shape) # S(t-1)
        S0[:, 1:, :] = self.S[:, :-1, :]


        # Second version of the second part - correct
        dS_dargTanh = (1 - np.power(self.S,2))
        dL_dS = np.tensordot(dL_dVS, self.V, axes=(2, 0))
        dL_dU = np.tensordot(dL_dS*dS_dargTanh, self.X, axes=((0, 1), (0, 1))) # returns an HxD matrix
        Unew = self.U - c * dL_dU

        # Evaluation of dLdW
        dL_dW = np.tensordot(dL_dS*dS_dargTanh, S0, axes=((0, 1), (0, 1))) # returns an HxH matrix
        Wnew = self.W - (c * dL_dW)


        '''

        # First version of the second part - not correct
        dS_dargTanh1 = 1 - self.S # Decomposition of dSdargTanh = tanh' = 1 - tanh^2 = (1 + tanh)(1 - tanh)
        dS_dargTanh2 = 1 + self.S # Decomposition of dSdargTanh = tanh' = 1 - tanh^2 = (1 + tanh)(1 - tanh)
        dL_dS = np.tensordot(dL_dVS, self.V, axes=(2, 0))  # returns an NxTxH matrix
        dL_dargTanh1 = np.tensordot(dL_dS, dS_dargTanh1, axes=((0, 1), (0, 1)))  # returns an HxH matrix
        dargTanh2_dU = np.tensordot(dS_dargTanh2, self.X, axes=((0, 1), (0, 1)))  # returns an HxD matrix
        dL_dU = dL_dargTanh1.dot(dargTanh2_dU)
        # New matrix U
        Unew = self.U - c * dL_dU
        # print('U aggiornato con dimensioni = ',Unew.shape)

        # Evaluation of dLdW
        dargTanh2_dW = np.tensordot(dS_dargTanh2, S0, axes=((0, 1), (0, 1)))  # returns an HxH matrix
        dL_dW = dL_dargTanh1.dot(dargTanh2_dW)  # returns an HxH matrix
        Wnew = self.W - c * dL_dW
        # print('W aggiornato con dimensione = ',Wnew.shape)
        '''

        return (Vnew, Unew, Wnew)

    # Function that implements the loss function computation
    def lossFunction(self):
        '''
        o = o.transpose()
        a = -y*np.log(o)
        return a.sum()
        '''
        O_ = np.log(self.O)
        c = (-1) / (self.N * self.T)
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
            lossTrain += [lossT / (len(self.train) // self.N)]
            #lossTrain += [lossT]
            # lossTrain += [lossT / (self.n_train // self.N)]
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
