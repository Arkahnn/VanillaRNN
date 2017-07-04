import numpy as np
import re
import math
import random
from Tools import*

class MyRNN:
    #Parameters
    N = T = D = H = eta = Ntrain = 0
    
    #Variables
    X = Y = O = S = np.empty((N,T))
    
    #weights
    U = W = V = np.empty((D,H))
    
    #Input parameters
    dictionary = train = valid = test = []

    def __init__(self, dictionary, train, valid, test, H, eta):
        self.dictionary, self.train, self.valid, self.test = dictionary, train, valid, test
        
        self.D = len(dictionary)
        #N = len(train) + len(valid) + len(test)
        self.T = max(len(s) for s in (train + valid + test)) + 2
        self.H, self.eta = H, eta
        self.Ntrain = len(train)
        # Weight assignment
        wgtD = self.D ** (-0.5)
        wgtH = self.H ** (-0.5)
        
        self.U = np.random.uniform(-wgtD, wgtD, (H, self.D))  # HxD matrix 
        self.W = np.random.uniform(-wgtH, wgtH, (H, H))  # HxH matrix
        self.V = np.random.uniform(-wgtH, wgtH, (self.D, H))  # DxH matrix
        
    def init_mainParam(self, data):
        self.N = len(data)
        #print('N dimension: ', self.N)
        # Preparation of X
        self.X = np.zeros((self.N, self.T, self.D))
        self.X[:, 0, 0] = 1
        self.X[:, 1:, 2] = 1
        
        for s in data:
            i = data.index(s)
            for w in s:
                j = s.index(w) + 1
                k = self.dictionary.index(w)
                self.X[i, j, k] = 1
                self.X[i, j, 2] = 0
                
            self.X[i, j + 1, 2] = 0
            self.X[i, j + 1, 1] = 1
        
        # Preparation of Y
        self.Y = np.zeros((self.N, self.T, self.D))
        self.Y[:, :-1, :] = self.X[:, 1:, :]
        self.Y[:, -1:, 2] = 1
        self.S = np.zeros((self.N, self.T, self.H))
        self.O = np.zeros((self.N, self.T, self.D))
        
    def fwdRnn(self,X,S,O):
        # 1. s = tanh(Ux + Ws_prev)
        # 2. o = sigma(Vs)
        # 3. U,W,V = L(y,o)
        
        # TODO: add SGD
        for i in range(self.T):
            S[:, i, :] = self.outHL(X[:, i, :].T, S[:, i - 1, :].T).T
            O[:, i, :] = self.softmax(self.V.dot(S[:, i, :].T)).T
        return (S,O)

    def bwRnn(self):
        # prod = eta*(-1/N)*Y*(1-O)
        # Aggiornare V
        # Vnew = V - eta*(-1/N)*np.dot((Y*(1-O)).T,S)
        #print('Inizio backward pass')
        Y_ = self.Y * (1 - self.O)
        dV = np.tensordot(Y_.T, self.S, axes=((1, 2), (1, 0)))
        c = self.eta * (-1 / (self.Ntrain * self.T * self.D))
        Vnew = self.V - c * dV
        # print('Valore vecchio di V: ', self.V)
        # print('Valore di dV: ',dV)
        # print('Valore nuovo di V: ',Vnew)
        
        #print('Inizio ad aggiornare U e W')
        S0 = np.zeros(self.S.shape)
        S0[:, 1:, :] = self.S[:, :-1, :] 
        deTanh1 = 1 - self.S
        deTanh2 = 1 + self.S
        Y_2 = np.tensordot(Y_, self.V, axes=(2, 0))  # returns an NxTxH matrix
        Y_3 = np.tensordot(Y_2, deTanh1, axes=((0, 1), (0, 1)))  # returns an HxH matrix
        S0_ = np.tensordot(deTanh2, self.X, axes=((0, 1), (0, 1)))  # returns an HxD matrix
        dU = Y_3.dot(S0_)
        Unew = self.U - c * dU
        # print('U aggiornato con dimensioni = ',Unew.shape)
        
        
        SS0_ = np.tensordot(deTanh2, S0, axes=((0, 1), (0, 1)))  # returns an HxH matrix
        dW = Y_3.dot(SS0_)  # returns an HxH matrix
        Wnew = self.W - c * dW
        # print('W aggiornato con dimensione = ',Wnew.shape)
        
        return (Vnew, Unew, Wnew)

    # Function that implements the activation of the hidden layer
    def outHL(self, x, s_prev):
        return np.tanh(self.U.dot(x) + self.W.dot(s_prev))

    # Function that implements the softmax computation
    def softmax(self,s):
        # TODO: sum and max over the correct axis
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
    def training_step(self, K):
        lossTrain = lossVal = []
        lossT = 0
        #self.init_mainParam(self.train)
        idxTrain = list(range(len(self.train)))
        
        
        for i in range(K):
            print('Epoch ',i,':')
            #Training set computation
            random.shuffle(idxTrain)
            #print('Train dimension: ', len(self.train))
            print('Iteration range: ', len(self.train)//500)
            
            for j in range(len(self.train)//500):
                #self.S[j*300:j*300+299,t,:],self.O[j*300:j*300+299,t,:] = self.fwdRnn(self.X[j*300:j*300+299,t,:],self.S[j*300:j*300+299,t,:],self.O[j*300:j*300+299,t,:])
                print('Iteration limits: [', j*500, ', ', j*500+500,')')
                self.init_mainParam([self.train[idxTrain[k]] for k in range(j*500,j*500+500)])
                self.S, self.O = self.fwdRnn(self.X, self.S, self.O)
                self.V, self.U, self.W = self.bwRnn()
                lossT += self.lossFunction()
            lossTrain += [lossT]
            lossT = 0
            
            #Validation set computation
            self.init_mainParam(self.valid)
            self.fwdRnn(self.X, self.S, self.O)
            lossVal += [self.lossFunction()]
    
        return (lossTrain, lossVal)
    
    def test_step(self):
        self.init_mainParam(self.test)
        self.fwdRnn(self.X, self.S, self.O)
        lossTest = self.lossFunction()
        accTest = self.accuracy()
        return (lossTest, accTest)
    
    def accuracy(self):
        compRes = O_ = Y_ = np.zeros((self.N,self.T))
        O_ = self.O.argmax(axis = 2)
        Y_ = self.Y.argmax(axis = 2)
        compRes = Y_ == O_
        return compRes.sum()/(self.N*self.T)
                
                
                