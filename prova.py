import matplotlib.pyplot as plt
from VanillaRNN import *
from Tools import *


def test_step2(aRnn):
    aRnn.init_mainParam(aRnn.test[0:500])
    aRnn.fwdRnn(aRnn.X, aRnn.S, aRnn.O)
    lossTest = aRnn.lossFunction()
    accTest = aRnn.accuracy()
    return lossTest, accTest


def prova_init_param(aRnn, data):
    aRnn.N = len(data)
    # print('N dimension: ', self.N)
    # Preparation of X
    X = np.zeros((aRnn.N, aRnn.T, aRnn.D))
    X[:, 0, 0] = 1
    #X[:, 1:, 2] = 1

    for s in data:
        i = data.index(s)  # Index of the phrase
        j, k = 0, 0
        print('Frase i: ', i, ':')
        for j in range(len(s)):
            #j = s.index(w) + 1  # Index of the word in the phrase +1 for the <startWD> token
            w = s[j]
            j += 1
            k = aRnn.dictionary.index(w)  # Index of the word in the dictionary
            print('Parola w: ', w, '; indice in frase j: ', j,'indice in dizionario k: ', k)
            X[i, j, k] = 1
            #X[i, j, 2] = 0

        #X[i, j + 1, 2] = 0
        #X[i, j + 1, 1] = 1
    aRnn.X = X
    aRnn.Y = np.zeros((aRnn.N, aRnn.T, aRnn.D))
    aRnn.Y[:, :-1, :] = aRnn.X[:, 1:, :]
    # self.Y[:, -1:, 2] = 1
    aRnn.S = np.zeros((aRnn.N, aRnn.T, aRnn.H))
    aRnn.O = np.zeros((aRnn.N, aRnn.T, aRnn.D))

def printResults():
    K, eta, H_size = 10, 0.9, 100

    print('Dataset creation')
    # Create dictionary
    dictionary, train, valid, test = build_dictionary('Dataset.txt')

    print('RNN initialization')
    # Initialize RNN
    myRnn = MyRNN(dictionary, train, valid, test, H_size, eta)

    V, U, W = importMatrix()

    #print(V)

    myRnn.V = V
    myRnn.U = U
    myRnn.W = W

    loss, acc = test_step2(myRnn)

    output = phrase_generator(myRnn, 'Time')
    print(output)

    '''
    Xidx = myRnn.X.argmax(axis=2) #Matrice X in versione NxT in cui in posizione X[i,j] è presente l'indice della parola corrispondente
    print(Xidx[10,:])
    phrase10 = ''
    for j in range(myRnn.T):
        phrase10 += myRnn.dictionary[Xidx[10, j]] + ' '
    print(phrase10)
    print(myRnn.test[10])
    
    
    input = prova_init_param(myRnn.test[9:12], myRnn.dictionary, myRnn.T, myRnn.D)
    Xidx = input.argmax(axis=2)
    print('Sequenza di input')
    print(Xidx[1,:])
    '''

    '''
    # Prova di stampa delle frasi predette col test set
    print('RNN Prediction')
    #Show some generated phrases
    Oidx = myRnn.O.argmax(axis=2)
    print(Oidx)
    strings, words = [], ''
    for i in range (myRnn.N):
        for j in range(myRnn.T):
            words += myRnn.dictionary[Oidx[i,j]] + ' '
        strings += [words]

    print(words)
    print(strings[0:20])
    '''

    '''
    train = import_file('lossTrain.txt')
    val = import_file('lossVal.txt')

    train = train.replace(',', '')
    val = val.replace(',', '')

    lossTrain = train.split(' ')
    lossTrain = lossTrain[:-1]
    lossTrain = [float(x)/100 for x in lossTrain] #It's no needed in the final version, except for float conversion
    #lossTrain = [float(x) for x in lossTrain] #Version needed in the final version
    print('LossTrain: ', lossTrain)
    lossVal = val.split(' ')
    lossVal = lossVal[:-1]
    lossVal = [float(x) for x in lossVal]
    print('LossVal: ', lossVal)

    t = list(range(K))
    plt.plot(t, lossTrain, 'ro', label='Training Set')
    plt.plot(t, lossVal, 'g^', label='Validation Set')
    plt.xlabel('time')
    plt.ylabel('loss value')
    plt.title('Loss function computation')
    plt.legend()
    plt.grid(True)

    plt.savefig("test.png")
    plt.show()
    '''

def phrase_generator(aRnn, startW):
    output, listWords = '', []
    iter = 0
    listWords += [startW]
    output += startW + ' '

    while iter < 10:
        print('Iter = ', iter)
        print('Prhase dimension: ', len(listWords))
        aRnn.init_mainParam([listWords])
        aRnn.S, aRnn.O = aRnn.fwdRnn(aRnn.X, aRnn.S, aRnn.O)
        print('Dimensione O: ',aRnn.O.shape)
        print('Nuovo indice: ', aRnn.O.argmax(axis=2)[0][iter])
        print('Nuova parola: ', aRnn.dictionary[aRnn.O.argmax(axis=2)[0][iter]])
        listWords += [aRnn.dictionary[aRnn.O.argmax(axis=2)[0][iter]]]
        output += aRnn.dictionary[aRnn.O.argmax(axis=2)[0][iter]] + ' '
        iter += 1

    return output


printResults()