import matplotlib.pyplot as plt
from VanillaRNN import *
from Tools import *


# This is the correct way to create a main function in Python (simply run it)
if __name__ == "__main__":
    K, eta, H_size = 10, 0.9, 100

    print('Dataset creation')
    # Create dictionary
    dictionary, train, valid, test = build_dictionary('Dataset.txt')

    print('RNN initialization')
    # Initialize RNN
    myRnn = MyRNN(dictionary, train, valid, test, H_size, eta)

    print('Train the RNN')
    # Train the RNN
    lossTrain, lossVal = myRnn.training_step(K)

    print('Test the RNN')
    # Test the RNN
    lossTest = myRnn.test_step()

    # Save last weights in a file
    np.savetxt('Vmat.txt', myRnn.V, delimiter=',')
    np.savetxt('Umat.txt', myRnn.U, delimiter=',')
    np.savetxt('Wmat.txt', myRnn.W, delimiter=',')

    loss = ''
    for s in lossTrain:
        loss += str(s) + ', '
    file = open('lossTrain.txt', 'w')
    file.write(loss)
    file.close()

    loss = ''
    for s in lossVal:
        loss += str(s) + ', '
    file = open('lossVal.txt', 'w')
    file.write(loss)
    file.close()
    print('Loss value of the test set: ', lossTest)

    t = list(range(K))
    plt.plot(t, lossTrain, 'ro', label='Training Set')
    plt.plot(t, lossVal, 'g^', label='Validation Set')
    plt.xlabel('time')
    plt.ylabel('loss value')
    plt.title('Loss function computation')
    plt.grid(True)
    plt.legend()
    # plt.savefig("test.png")
    plt.show()

    # Multi-line comments are triggered with CTRL + /
    # for i in range(K):
    #     print('Epoch ', i, ':')
    #     myRnn.fwdRnn()
    #     Vnew, Unew, Wnew = myRnn.bwRnn()
    #     # V = myRnn.V
    #     myRnn.V, myRnn.U, myRnn.W = Vnew, Unew, Wnew
    #     # print('Vecchio valore di V: ', V)
    #     # print('Nuovo valore di V: ', (myRnn.V == Vnew))
    #     # print('Valore di O: ', myRnn.O.sum())
    #     loss_val += [myRnn.lossFunction(myRnn.Y, myRnn.O, myRnn.N)]
    #     myRnn.S = np.zeros((myRnn.N, myRnn.T, myRnn.H))
    #     myRnn.O = np.zeros((myRnn.N, myRnn.T, myRnn.D))
