import random
import numpy as np
import numpy.matlib as mat
import tools


class RNN:

    def __init__(self, dictionary, train, val, test, H, eta, alpha, t_prev):
        self.dictionary, self.train, self.val, self.test = dictionary, train, val, test

        self.D = len(dictionary)
        #self.DU = self.D + 1
        self.T = max(len(s) for s in (train + val + test))
        self.H, self.eta, self.alpha = H, eta, alpha
        self.n_train = len(train)
        self.t_prev = t_prev
        # self.bias = 0.33
        # print('Total number of phrases: ', self.N)

        # Weight assignment with Glorot uniform
        wgtD = self.D ** (-0.5)
        #wgtDU = self.DU ** (-0.5) # for the bias in X and U
        wgtH = self.H ** (-0.5)
        wgtBias = random.uniform(-wgtH, wgtH)

        self.U = np.random.uniform(-wgtD, wgtD, (self.H, self.D))  # Hx(D+1) matrix
        self.W = np.random.uniform(-wgtH, wgtH, (self.H, self.H))  # HxH matrix
        self.V = np.random.uniform(-wgtH, wgtH, (self.D, self.H))  # DxH matrix

        # # weight assignment with simple weights
        # self.U = np.random.randn(self.H, self.D) * 0.01
        # self.W = np.random.randn(self.H, self.H) * 0.01
        # self.V = np.random.randn(self.D, self.H) * 0.01

        # self.U[-1,:] = wgtBias
        # self.V[:, -1] = wgtBias
        # self.W[:, -1] = wgtBias


    def init_main_params(self, data):
        # Set X (input)
        self.N = len(data)
        self.X = np.zeros((self.N, self.T, self.D))
        #self.X[:,:,-1] = 1 # bias
        for n, sent in enumerate(data):
            self.X[n, range(len(sent)), [self.dictionary.index(x) for x in sent]] = 1.0

        # Set Y (labels)
        self.Y = np.zeros((self.N, self.T, self.D))
        self.Y[:, :-1, :] = self.X[:, 1:, :] # X[:, 1:, :-1] for the bias
        # self.Y[:, -1:, 2] = 1

        # Set S and O (hidden output and output)
        self.S = np.zeros((self.N, self.T, self.H))
        #
        self.O = np.zeros((self.N, self.T, self.D))
        #self.S[:, :, -1] = 1.0
        # for c in data:
        #     self.S[:, len(c):,-1] = 0.0

    # forward step of the RNN
    def forward(self, X, U, S, O):
        # 1. s = tanh(Ux + Ws_prev)
        # 2. o = sigma(Vs)
        # 3. U,W,V = L(y,o)
        reLU = False

        for t in range(self.T):
            if reLU:
                S[:, t, :] = (self.U.dot(X[:, t, :].T) + self.W.dot(S[:, t - 1, :].T)).T
                S[:, t, :] = S[:, t, :] * (S[:, t, :] > 0)
            else:
                S[:, t, :] = self.out_HL(X[:, t, :].T, U, S[:, t - 1, :].T).T
            # O[:, t, :] = self.softmax(self.V.dot(S[:, t, :].T)).T
            O[:, t, :] = self.softmax(np.dot(self.V, S[:, t, :].T)).T
        return S, O

    # New version
    def dLdO(self, Y, O):
        #return (-self.Y)/self.O # returns a NxTxD matrix
        return -Y / O # returns a Dx1 vector

    def dOdV(self, dO_dVS, S):
        #return np.einsum('ntd,nth->dh',dO_dVS,self.S) # returns a DxH matrix
        #return np.einsum('ik,il->kl', dO_dVS, S)  # returns a DxH matrix
        return dO_dVS[:, None] * S[None, :] # Return a DxH matrix

    def dOdS(self, dO_dVS):
        #return np.einsum('ntd,dh->dh', dO_dVS, self.V) # returns a DxH matrix
        #return np.einsum('ik,kl->kl', dO_dVS, self.V)  # returns a DxH matrix
        return dO_dVS[:, None] * self.V # Returns a DxH matrix - Da verificare!!!

    def dSdU(self, S, X):
        S = (1 - S ** 2)
        #return np.einsum('nth,ntd->hd', S, self.X) # returns a HxD matrix
        #return np.einsum('ik,il->kl', S, X)  # returns a Hx(D+1) matrix
        return S[:, None] * X[None, :] # Returns an HxD matrix

    def dSdW(self, S, S0):
        #S = (1 - S ** 2)
        #return np.einsum('nth,nth->nth', S, S0) # returns an NxTxH matrix
        #return np.einsum('ik,ik->ik', S, S0)  # returns an NxH matrix
        return S[:, None] * S0[None, :] # Returns a HxH vector

    def dLdV(self, dL_dO, dO_dV):
        #return np.einsum('ntd,dh->dh', dL_dO, dO_dV) # returns a DxH matrix
        #return np.einsum('ik,kl->kl', dL_dO, dO_dV)  # returns a DxH matrix
        return dL_dO[:, None] * dO_dV # Returns a DxH matrix

    def dLdS(self, dL_dO, dO_dS):
        #return np.einsum('ntd,dh->dh', dL_dO, dO_dS) #returns a DxH matrix
        #return np.einsum('ik,kl->kl', dL_dO, dO_dS)  # returns a DxH matrix
        return dL_dO.dot(dO_dS) # Returns an Hx1 matrix

    def dLdU(self, dL_dS, dS_dU):
        #return np.einsum('dh,hd->hd', dL_dS, dS_dU) # returns an HxD matrix
        #return np.einsum('ij,jk->jk', dL_dS, dS_dU)  # returns an Hx(D+1) matrix
        return dL_dS[:, None] * dS_dU # Return HxD matrix

    def dLdW(self, dL_dS, dS_dW):
        #return np.einsum('dm,nth->mh', dL_dS, dS_dW) # returns an HxH matrix
        # dL_dW = np.zeros((self.H, self.H))
        # # for i in range(self.N):
        # #     for j in range(self.T):
        # #         for k in range(self.D):
        # #             dL_dW += dS_dW[i, j, :] * dL_dS[k, :]
        # dS_dW1 = dS_dW.repeat(self.H).reshape(self.N, self.H, self.H)
        # dS_dW1 = dS_dW1.sum(axis=0)
        # for i in range(self.D):
        #     dL_dW += dS_dW1 * dL_dS[i,:]
        # return dL_dW
        #return np.einsum('dm,nth->mh', dL_dS, dS_dW)  # returns an HxH matrix
        return dL_dS[:, None] * dS_dW # Returns an HxH matrix

    # backward pass of the RNN
    def backprop(self):
        reLU = False
        delta_V, delta_U, delta_W = np.zeros(self.V.shape), np.zeros(self.U.shape), np.zeros(self.W.shape)
        dL_dV, dL_dU, dL_dW = np.zeros(self.V.shape), np.zeros(self.U.shape), np.zeros(self.W.shape)
        S0 = np.zeros(self.S.shape)  # S(t-1)
        S0[:, 1:, :] = self.S[:, :-1, :]
        S2 = 1 - self.S**2
        c = self.eta
        #l = [len(a) for a in self.train]
        #c = self.eta/(self.N * sum(l))

        # Y1 = self.Y.nonzero()  # elements of Y different from zero
        # dL_dVS = self.Y
        # dL_dVS[Y1[0], Y1[1], Y1[2]] = self.O[Y1[0], Y1[1], Y1[2]] - 1
        dL_dVS = (self.O * self.Y) - self.Y

        for n in range(self.N):
            for t in range(self.T):
                # Versione del codice originale

                dL_dV += np.outer(dL_dVS[n, t, :], self.S[n, t, :].T)
                dL_dS = self.V.T.dot(dL_dVS[n, t, :])
                if reLU:
                    dL_dargTanh = (dL_dS > 0).astype(int)
                else:
                    dL_dargTanh = dL_dS * S2[n, t, :]
                for t_i in range(t, t - self.t_prev, -1):
                    dL_dU += np.outer(dL_dargTanh, self.X[n, t_i, :].T)
                    if t_i == 0:
                        h_prev = np.zeros((self.H))
                        dL_dW += np.outer(dL_dargTanh, h_prev.T)
                        break
                    else:
                        dL_dW += np.outer(dL_dargTanh, S0[n, t_i - 1, :].T)
                        dL_dargTanh = self.W.T.dot(dL_dargTanh) * (1 - S0[n, t_i - 1, :] ** 2)
                # Evaluation of dL/dV
                # print('Evaluation of dL/dV')
                # dL_dO = self.dLdO(self.Y[n, t, :], self.O[n, t, :]) # returns a NxD matrix
                # dO_dVS = self.O[n, t, :] * (1.0 - self.O[n, t, :]) # returns a NxD matrix
                # dO_dV = self.dOdV(dO_dVS, self.S[n, t, :]) # returns a DxH matrix
                # dL_dV += self.dLdV(dL_dO, dO_dV) # returns the final DxH matrix


                # dL_dV += self.dLdV(dL_dVS, dO_dV)  # returns the final DxH matrix
                #c = self.eta / (self.N * self.T)  # Constant value including eta and 1/n


                # print('V equality: ',np.array_equal(self.V, Vnew))

                # Evalutation of dL/dU
                # print('Evaluation of dL/dU')
                # dO_dS = self.dOdS(dO_dVS) # returns a DxH matrix
                # # dL_dS = dL_dO.dot(dO_dS) # returns a DxH matrix
                # dS_dU = self.dSdU(S2[n, t, :], self.X[n, t, :]) # returns a HxD matrix
                # dL_dS = self.dLdS(dL_dO, dO_dS)
                # dL_dU += self.dLdU(dL_dS,dS_dU)
                # dL_dU = dL_dS.T * dS_dU # returns the final HxD matrix


                # print('U equality: ', np.array_equal(self.U, Unew))

                # Evaluation of dL/dW
                # print('Evaluation of dL/dW')
                # dS_dW = self.dSdW(S2[n, t, :], S0[n, t, :]) # returns a HxD matrix
                # dL_dW += self.dLdW(dL_dS, dS_dW)


                #print('dL/dW dimensions: ', dL_dW.shape)

                # print('W equality: ', np.array_equal(self.W, Wnew))



        Vnew = self.V + (self.alpha * delta_V - c * dL_dV) # - c * dL_dV
        delta_V = self.alpha * delta_V - c * dL_dV
        Unew = self.U + (self.alpha * delta_U - c * dL_dU) # - c * dL_dU
        delta_U = self.alpha * delta_U - c * dL_dU
        Wnew = self.W + (self.alpha * delta_W - c * dL_dW) # - c * dL_dW
        delta_W = self.alpha * delta_W - c * dL_dW

        Vnew = np.clip(Vnew, -5, 5)
        Unew = np.clip(Unew, -5, 5)
        Wnew = np.clip(Wnew, -5, 5)

        return (Vnew, Unew, Wnew)

    def training(self, K, mini_batch_size):
        loss_train, loss_val = [], []
        acc_train, acc_val = [], []
        idx_train = list(range(self.n_train))
        self.N = mini_batch_size
        n_mini_batches = self.n_train // self.N

        print('Training set size: ', self.n_train)
        print('Mini-batch size: ', self.N)
        print('Number of mini-batches: ', n_mini_batches)

        # Bias introduction on X and U
        

        for i in range(K):
            print('Epoch ', i, '/', K, ':')
            random.shuffle(idx_train)
            loss_t, loss_v = 0.0, 0.0
            acc_t, acc_v = 0.0, 0.0

            # forward and backprop steps
            for j in range(n_mini_batches):
                print('    Batch ', j + 1, '/', n_mini_batches)
                self.init_main_params([self.train[i] for i in idx_train[(j * self.N):((j + 1) * self.N)]])
                self.S, self.O = self.forward(self.X, self.U, self.S, self.O)
                self.V, self.U, self.W = self.backprop()
                loss_t += self.loss(self.n_train)
                acc_t += self.accuracy()
                # print('Loss: ', self.loss())
            N = np.sum((len(y_i) for y_i in self.train))
            # print('Number of elements in the training set: ', N)
            loss = loss_t/ N # len(self.train)
            acc = acc_t/ N # n_mini_batches
            #print('Mean loss: ', loss)
            loss_train.append(loss)
            acc_train.append(acc)
            print('    Loss: ', loss)
            # validation step
            print('Validation: ')
            for j in range(len(self.val) // self.N):
                print('    Batch ', j + 1, '/', len(self.val) // self.N)
                self.init_main_params(self.val[(j * self.N):((j + 1) * self.N)])
                self.forward(self.X, self.U, self.S, self.O)
                l = [len(a) for a in self.val]
                #loss_v += self.loss()/(self.N * sum(l))
                loss_v += self.loss(len(self.val))
                acc_v += self.accuracy()
                # print('Validation loss: ', self.loss(len(self.val)))
                print('Validation accuracy: ', self.accuracy())
            l = []
            # loss_val.append(loss_v/(len(self.val) // self.N))
            N = np.sum((len(y_i) for y_i in self.val))
            loss = loss_v/ N # len(self.val) # (len(self.val) // self.N)
            acc = acc_v/ N # len(self.val) # (len(self.val) // self.N)
            print('Loss val: ', loss)
            loss_val.append(loss)
            acc_val.append(acc)

        return loss_train, loss_val, acc_train, acc_val

    def testing(self):
        self.init_main_params(self.test)
        self.forward(self.X, self.U, self.S, self.O)
        print('N = ', self.N)
        loss_test = self.loss(len(self.test))
        acc_test = self.accuracy()
        return loss_test, acc_test

    # Function that implements the softmax computation
    def softmax(self, s):
        # Softmax over 2D-matrix if D dimension is on axis = 0
        #s -= np.amax(s, axis=0)
        s = np.exp(s)
        return s / np.sum(s, axis=0)

    # Function that implements the activation of the hidden layer
    def out_HL(self, x, U, s_prev):
        # print('X shape: ', self.X.shape)
        # print('U shape: ', self.U.shape)
        return np.tanh(np.dot(U, x) + np.dot(self.W, s_prev)) # which verse of W am I using? which weights will be upgraded?

    # Function that implements the loss function computation
    def loss(self, n_phrases):
        '''
        o = o.transpose()
        a = -y*np.log(o)
        return a.sum()
        '''
        O_ = np.log(self.O)
        O_[ ~np.isfinite(O_)] = 0.0
        #c = -1 /(n_phrases * 10)
        #c = -1
        #return c * np.tensordot(self.Y, O_, axes=((0, 1, 2), (0, 1, 2)))
        # We only care about our prediction of the "correct" words
        correct_word_predictions = self.Y * O_
        # Add to the loss based on how off we were
        L = -1.0 * np.sum(correct_word_predictions)
        # res = -np.tensordot(self.Y, O_, axes=((0, 1, 2), (0, 1, 2)))
        #print('Equality of loss computations: ', L == res)
        return L #/(n_phrases * 10)

    def accuracy(self):
        O_ = self.O.argmax(axis=2)
        Y_ = self.Y.argmax(axis=2)
        acc = 0.0
        acc_tot = 0.0
        # print('O_ size: ', O_.size())
        N = np.shape(O_)[0]
        for i in range(N):
            for j in range(self.T):
                if Y_[i,j] == 0:
                    acc_tot += acc
                    acc = 0.0
                    break
                else:
                    acc += (Y_[i,j] == O_[i,j])


        #comp_res = Y_ == O_
        #return comp_res.sum() / (self.N * self.T)
        return acc_tot
