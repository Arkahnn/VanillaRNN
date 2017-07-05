import matplotlib.pyplot as plt
from VanillaRNN import *
from Tools import *

def printResults():
    K, eta, H_size = 10, 0.9, 100

    '''
    print('Dataset creation')
    # Create dictionary
    dictionary, train, valid, test = build_dictionary('Dataset.txt')

    print('RNN initialization')
    # Initialize RNN
    myRnn = MyRNN(dictionary, train, valid, test, H_size, eta)

    V, U, W = importMatrix()

    print(V)

    myRnn.V = V
    myRnn.U = U
    myRnn.W = W

    loss, acc = myRnn.test_step()
    '''

    train = import_file('lossTrain.txt')
    val = import_file('lossVal.txt')

    train = train.replace(',', '')
    val = val.replace(',', '')

    lossTrain = train.split(' ')
    lossTrain = lossTrain[:-1]
    lossTrain = [float(x)/100 for x in lossTrain] #It's no needed in the final version
    print('LossTrain: ', lossTrain)
    lossVal = val.split(' ')
    lossVal = lossVal[:-1]
    lossVal = [float(x) for x in lossVal]
    print('LossVal: ', lossVal)
    num = list(range(K))

    plt.plot(num, lossTrain, 'ro', num, lossVal, 'g^')
    plt.xlabel('time')
    plt.ylabel('loss value')
    plt.title('Loss function computation')
    plt.grid(True)
    plt.savefig("test.png")
    plt.show()


printResults()