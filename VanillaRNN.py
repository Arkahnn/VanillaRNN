import random
import numpy as np
import numpy.matlib as mat
import tools


class RNN:

    def __init__(self, dictionary, train, val, test, H, eta):
        self.dictionary, self.train, self.val, self.test = dictionary, train, val, test

        self.D = len(dictionary)
        self.T = max(len(s) for s in (train + val + test)) + 2
        self.H, self.eta = H, eta
        self.n_train = len(train)
        # print('Total number of phrases: ', self.N)

        # Weight assignment with Glorot uniform
        # wgtD = self.D ** (-0.5)
        # wgtH = self.H ** (-0.5)

        # self.U = np.random.uniform(-wgtD, wgtD, (self.H, self.D))  # HxD matrix
        # self.W = np.random.uniform(-wgtH, wgtH, (self.H, self.H))  # HxH matrix
        # self.V = np.random.uniform(-wgtH, wgtH, (self.D, self.H))  # DxH matrix

        # weight assignment with simple weights
        self.U = np.random.randn(self.H, self.D) * 0.01
        self.W = np.random.randn(self.H, self.H) * 0.01
        self.V = np.random.randn(self.D, self.H) * 0.01

    def init_main_params(self, data):
        # Set X (input)
        self.N = len(data)
        self.X = np.zeros((self.N, self.T, self.D))
        for n, sent in enumerate(data):
            self.X[n, range(len(sent)), [self.dictionary.index(x) for x in sent]] = 1.0

        # Set Y (labels)
        self.Y = np.zeros((self.N, self.T, self.D))
        self.Y[:, :-1, :] = self.X[:, 1:, :]
        # self.Y[:, -1:, 2] = 1

        # Set S and O (hidden output and output)
        self.S = np.zeros((self.N, self.T, self.H))
        self.O = np.zeros((self.N, self.T, self.D))

    # forward step of the RNN
    def forward(self, X, S, O):
        # 1. s = tanh(Ux + Ws_prev)
        # 2. o = sigma(Vs)
        # 3. U,W,V = L(y,o)

        for t in range(self.T):
            S[:, t, :] = self.out_HL(X[:, t, :].T, S[:, t - 1, :].T).T
            # O[:, t, :] = self.softmax(self.V.dot(S[:, t, :].T)).T
            O[:, t, :] = self.softmax(np.dot(S[:, t, :], self.V.T))
        return S, O

    """
    def dLdO(self):
        # #dL_dO = np.tile(self.Y.T, (self.D, 1, 1, 1)).T
        # #dL_dO = self.Y.repeat(self.D).reshape(self.N, self.T, self.D, self.D)
        # dL_dO = np.zeros((self.D, self.D))
        # for i in range(self.N):
        #     for j in range(self.T):
        #         Y = self.Y[i,j,:].repeat(self.D).reshape(self.D, self.D)
        #         dL_dO += np.multiply(Y, 1.0/self.O[i,j,:])
        # #return np.sum(dL_dO, axis=(0,1)) #returns a DxD matrix
        return np.multiply(-self.Y, 1.0/self.O) # returns a NxTxD matrix

    def dOdV(self, dO_dVS):
        #dO_dV = np.tile(dO_dVS.T, (self.H, 1, 1, 1)).T
        dOdVS = dO_dVS.repeat(self.H).reshape(self.N, self.T, self.D, self.H)
        dO_dV = np.zeros((self.D, self.H))
        for i in range(self.N):
            for j in range(self.T):
                #dOdVS = dO_dVS[i, j, :].repeat(self.H).reshape(self.D, self.H)
                dO_dV += np.multiply(dOdVS[i, j, :, :], self.S[i, j, :])
        #return np.sum(dO_dV, axis=(0, 1)) # returns a DxH matrix
        return dO_dV # returns a DxH matrix

    def dOdS(self, dO_dVS):
        #dO_dS = np.tile(dO_dVS.T, (self.H, 1, 1, 1)).T
        #dO_dS = dO_dVS.repeat(self.H).reshape(self.N, self.T, self.D, self.H)
        dO_dS = np.zeros((self.D, self.H))
        for i in range(self.N):
            for j in range(self.T):
                dOdVS = dO_dVS[i, j, :].repeat(self.H).reshape(self.D, self.H)
                dO_dS += np.multiply(dOdVS, self.V[:, :])
        #return np.sum(dO_dS, axis=(0, 1)) # returns a DxH matrix
        return dO_dS # returns a DxH matrix

    def dSdU(self):
        #dS_dU = np.tile((1 - self.S**2).T, (self.D, 1, 1, 1)).T
        #dS_dU = (1 - self.S**2).repeat(self.D).reshape(self.N, self.T, self.H, self.D)
        S = (1 - self.S**2)
        dS_dU = np.zeros((self.H, self.D))
        for i in range(self.N):
            for j in range(self.T):
                dS_dArgT = S[i, j, :].repeat(self.D).reshape(self.H, self.D)
                dS_dU += np.multiply(dS_dArgT, self.V.T)
        #return np.sum(dS_dU, axis=(0, 1))  # returns a HxD matrix
        return dS_dU # returns a HxD matrix

    # Per il prodotto di Hadamard:
    # Se ho una matrice NxTxD(xH) (O) e una NxTxH (S) invece di ciclare su N e su T, ciclare su D e sommare su N e T

    def dSdW(self, S0):
        #dS_dW = np.tile((1 - self.S**2).T, (self.D, self.H, 1, 1, 1)).T
        #dS_dW = (1 - self.S**2).repeat(self.H).reshape(self.N, self.T, self.H, self.H)
        #dS_dW = dS_dW.repeat(self.D).reshape(self.N, self.T, self.H, self.H, self.D)
        S = (1 - self.S**2)
        #dS_dW = np.zeros((self.H, self.H, self.D))
        dS_dArgT1 = S.repeat(self.H).reshape(self.N, self.T, self.H, self.H)
        dS_dW = np.zeros((self.H, self.H))
        for i in range(self.N):
            for j in range(self.T):
                dS_dW += np.multiply(dS_dArgT1[i,j,:,:],S0[i,j,:])
        # for i in range(self.N):
        #     for j in range(self.T):
        #         dS_dArgT1 = S[i,j,:].repeat(self.H).reshape(self.H, self.H)
        #         dS_dW += dS_dArgT1.repeat(self.D).reshape(self.H, self.H, self.D)
        #
        # for k in range(self.D):
        #     dS_dW[:, :, k] = np.multiply(dS_dW[:, :, k], S0[i, j, :].T)
        #return np.sum(dS_dW, axis=(0, 1))  # returns a HxHxD matrix
        return dS_dW # returns an HxD matrix

    def dLdV(self, dL_dO, dO_dV):
        dL_dO1 = dL_dO.repeat(self.H).reshape(self.N, self.T, self.D, self.H)
        dL_dV = np.zeros((self.D, self.H))
        for i, j, k in zip(range(self.N), range(self.T), range(self.D)):
            dL_dV += np.multiply(dL_dO1[i, j, :, :], dO_dV)
        return dL_dV # returns a DxH matrix

    def dLdU(self, dL_dO, dO_dS, dS_dU):
        dO_dU = np.multiply(dO_dS,dS_dU.T) # DxH matrix
        dL_dO1 = dL_dO.repeat(self.H).reshape(self.N, self.T, self.D, self.H)
        dL_dU = np.zeros((self.D, self.H))
        for i, j in zip(range(self.N), range(self.T)):
            dL_dU += np.multiply(dL_dO1[i,j,:,:], dL_dU)
        return dL_dU.T # returns an HxD matrix


    def dLdW(self, dL_dO, dO_dS, dS_dW):
        dO_dW = np.zeros((self.H, self.H))
        for i in range(self.D):
            dO_dW += np.multiply(dO_dS[i,:],dS_dW)

        dL_dO1 = dL_dO.repeat(self.H).reshape(self.N, self.T, self.D, self.H)
        dL_dW = np.zeros((self.H, self.H))
        for i, j, k in zip(range(self.N), range(self.T), range(self.D)):
            dL_dW += np.multiply(dL_dO1[i,j,k,:], dO_dW)
        return dL_dW # returns an HxH matrix
    """

    # New version
    def dLdO(self):
        return (-self.Y)/self.O

    def dOdV(self, dO_dVS):
        return np.einsum('ntd,nth->dh',dO_dVS,self.S)

    def dOdS(self, dO_dVS):
        return np.einsum('ntd,dh->dh', dO_dVS, self.V)

    def dSdU(self):
        S = (1 - self.S ** 2)
        return np.einsum('nth,ntd->hd', S, self.X)

    def dSdW(self, S0):
        S = (1 - self.S ** 2)
        return np.einsum('nth,ntd->hd', S, S0)

    def dLdV(self, dL_dO, dO_dV):
        return np.einsum('ntd,dh->dh', dL_dO, dO_dV)

    def dLdU(self, dL_dO, dO_dS, dS_dU):
        dO_dU = np.einsum('dh,hd->hd', dO_dS, dS_dU)
        return np.einsum('ntd,hd->hd', dL_dO, dO_dU)

    def dLdW(self, dL_dO, dO_dS, dS_dW):
        dL_dS = np.einsum('ntd,dh->dh', dL_dO, dO_dS)
        return np.einsum('dh,hn->hn', dL_dS, dS_dW)

    # backward pass of the RNN
    def backprop(self):

        # Evaluation of dL/dV
        print('Evaluation of dL/dV')
        dL_dO = self.dLdO() # returns a DxD matrix
        dO_dVS = self.O*(1 - self.O)
        dO_dV = self.dOdV(dO_dVS) # returns a DxH matrix
        dL_dV = self.dLdV(dL_dO, dO_dV) # returns the final DxH matrix
        c = (-self.eta) / (self.n_train * self.T)  # Constant value including eta and 1/n
        Vnew = self.V - (c * dL_dV)

        # Evalutation of dL/dU
        print('Evaluation of dL/dU')
        dO_dS = self.dOdS(dO_dVS) # returns a DxH matrix
        # dL_dS = dL_dO.dot(dO_dS) # returns a DxH matrix
        dS_dU = self.dSdU() # returns a HxD matrix
        dL_dU = self.dLdU(dL_dO, dO_dS,dS_dU)
        # dL_dU = dL_dS.T * dS_dU # returns the final HxD matrix
        Unew = self.U - (c * dL_dU)

        # Evaluation of dL/dW
        print('Evaluation of dL/dW')
        S0 = np.zeros(self.S.shape)  # S(t-1)
        S0[:, 1:, :] = self.S[:, :-1, :]
        dS_dW = self.dSdW(S0) # returns a HxHxD matrix
        dL_dW = self.dLdW(dL_dO, dO_dS, dS_dW)
        #dL_dW = np.tensordot(dL_dS, dS_dW, axes=(0, 2))
        #dL_dW = np.tensordot(dL_dS, dS_dW, axes=((0,1),(2,1)))
        #print('dL/dW dimensions: ', dL_dW.shape)
        Wnew = self.W - (c * dL_dW)

        return (Vnew, Unew, Wnew)

    # #Old Version
    # def backprop(self):
    #     '''
    #     #Good version
    #
    #     # Evaluation of dLdV
    #     dL_dO = self.Y / self.O
    #     dO_dVS = self.O * (1.0 - self.O)
    #     dL_dV = np.tensordot(dL_dO*dO_dVS, self.S, axes=((0, 1), (0, 1)))
    #     c = (-self.eta) / (self.N * self.T) # Constant value including eta and 1/n
    #     # New matrix V
    #     Vnew = self.V - c * dL_dV
    #
    #     # Verifying the Good and the Third versions
    #     A = - self.Y + (self.Y * self.O)
    #     dL_dO = self.Y / self.O
    #     dO_dVS = self.O * (1.0 - self.O)
    #     B = dL_dO*dO_dVS
    #     print('Uguaglianza tra versione semplificata ed espansa: ', A == B)
    #     '''
    #
    #     # Third version - As good as the Good version
    #
    #     # Evaluation of dL_dV
    #     dLdVS = - self.Y + (self.Y * self.O)
    #     dLdV = np.tensordot(dLdVS, self.S, axes=((0, 1), (0, 1)))
    #     c = self.eta / (self.N * self.T)  # Constant value including eta and 1/n
    #     #c = self.eta
    #     # New matrix V
    #     Vnew = self.V - c * dLdV
    #
    #
    #     # Evaluation of dLdU
    #     S0 = np.zeros(self.S.shape) # S(t-1)
    #     S0[:, 1:, :] = self.S[:, :-1, :]
    #
    #
    #     # Second version of the second part - correct
    #     dtanh = (1 - np.power(self.S, 2))
    #     dLdS = np.tensordot(dLdVS, self.V, axes=(2, 0))
    #     dLdU = np.tensordot(dLdS * dtanh, self.X, axes=((0, 1), (0, 1)))  # returns an HxD matrix
    #     Unew = self.U - c * dLdU
    #
    #     # Evaluation of dLdW
    #     dL_dW = np.tensordot(dLdS * dtanh, S0, axes=((0, 1), (0, 1)))  # returns an HxH matrix
    #     Wnew = self.W - (c * dL_dW)
    #
    #
    #     '''
    #     # First version of the second part - not correct
    #     dS_dargTanh1 = 1 - self.S # Decomposition of dSdargTanh = tanh' = 1 - tanh^2 = (1 + tanh)(1 - tanh)
    #     dS_dargTanh2 = 1 + self.S # Decomposition of dSdargTanh = tanh' = 1 - tanh^2 = (1 + tanh)(1 - tanh)
    #     dL_dS = np.tensordot(dL_dVS, self.V, axes=(2, 0))  # returns an NxTxH matrix
    #     dL_dargTanh1 = np.tensordot(dL_dS, dS_dargTanh1, axes=((0, 1), (0, 1)))  # returns an HxH matrix
    #     dargTanh2_dU = np.tensordot(dS_dargTanh2, self.X, axes=((0, 1), (0, 1)))  # returns an HxD matrix
    #     dL_dU = dL_dargTanh1.dot(dargTanh2_dU)
    #     # New matrix U
    #     Unew = self.U - c * dL_dU
    #     # print('U aggiornato con dimensioni = ',Unew.shape)
    #     # Evaluation of dLdW
    #     dargTanh2_dW = np.tensordot(dS_dargTanh2, S0, axes=((0, 1), (0, 1)))  # returns an HxH matrix
    #     dL_dW = dL_dargTanh1.dot(dargTanh2_dW)  # returns an HxH matrix
    #     Wnew = self.W - c * dL_dW
    #     # print('W aggiornato con dimensione = ',Wnew.shape)
    #     '''
    #
    #     return (Vnew, Unew, Wnew)

    def training(self, K, mini_batch_size):
        loss_train, loss_val = [], []
        idx_train = list(range(self.n_train))
        self.N = mini_batch_size
        n_mini_batches = self.n_train // self.N

        print('Training set size: ', self.n_train)
        print('Mini-batch size: ', self.N)
        print('Number of mini-batches: ', n_mini_batches)

        for i in range(K):
            print('Epoch ', i, '/', K, ':')
            random.shuffle(idx_train)
            loss_t, loss_v = (0.0, 0.0)

            # forward and backprop steps
            for j in range(n_mini_batches):
                print('    Batch ', j + 1, '/', n_mini_batches)
                self.init_main_params([self.train[i] for i in idx_train[(j * self.N):((j + 1) * self.N)]])
                self.S, self.O = self.forward(self.X, self.S, self.O)
                self.V, self.U, self.W = self.backprop()
                loss_t += self.loss()
            loss_train.append(loss_t)
            print('    Loss: ', loss_t)
            # validation step
            print('Validation: ')
            for j in range(len(self.val) // self.N):
                print('    Batch ', j + 1, '/', len(self.val) // self.N)
                self.init_main_params(self.val[(j * self.N):((j + 1) * self.N)])
                self.forward(self.X, self.S, self.O)
                loss_v += self.loss()
            loss_val.append(loss_v)

        return loss_train, loss_val

    def testing(self):
        self.init_main_params(self.test)
        self.forward(self.X, self.S, self.O)
        print('N = ', self.N)
        loss_test = self.loss()
        acc_test = self.accuracy()
        return loss_test, acc_test

    # Function that implements the softmax computation
    def softmax(self, s):
        # Softmax over 2D-matrix if D dimension is on axis = 0
        s -= np.amax(s, axis=0)
        s = np.exp(s)
        return s / np.sum(s, axis=0)

    # Function that implements the activation of the hidden layer
    def out_HL(self, x, s_prev):
        return np.tanh(np.dot(self.U, x) + np.dot(self.W, s_prev))

    # Function that implements the loss function computation
    def loss(self):
        '''
        o = o.transpose()
        a = -y*np.log(o)
        return a.sum()
        '''
        O_ = np.log(self.O)
        c = -1 / (self.N * self.T)
        return c * np.tensordot(self.Y, O_, axes=((0, 1, 2), (0, 1, 2)))

    def accuracy(self):
        O_ = self.O.argmax(axis=2)
        Y_ = self.Y.argmax(axis=2)
        comp_res = Y_ == O_
        return comp_res.sum() / (self.N * self.T)
