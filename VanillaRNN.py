import random
import numpy as np
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

    # backward pass of the RNN
    def backprop(self):
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
        dLdVS = - self.Y + (self.Y * self.O)
        dLdV = np.tensordot(dLdVS, self.S, axes=((0, 1), (0, 1)))
        c = self.eta / (self.n_train * self.T)  # Constant value including eta and 1/n
        #c = self.eta
        # New matrix V
        Vnew = self.V - c * dLdV


        # Evaluation of dLdU
        S0 = np.zeros(self.S.shape) # S(t-1)
        S0[:, 1:, :] = self.S[:, :-1, :]


        # Second version of the second part - correct
        dtanh = (1 - np.power(self.S, 2))
        dLdS = np.tensordot(dLdVS, self.V, axes=(2, 0))
        dLdU = np.tensordot(dLdS * dtanh, self.X, axes=((0, 1), (0, 1)))  # returns an HxD matrix
        Unew = self.U - c * dLdU

        # Evaluation of dLdW
        dL_dW = np.tensordot(dLdS * dtanh, S0, axes=((0, 1), (0, 1)))  # returns an HxH matrix
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
