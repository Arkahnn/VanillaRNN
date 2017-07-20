import matplotlib.pyplot as plt
import numpy as np
import random

import VanillaRNN
import tools


if __name__ == "__main__":
    # Set a seed to test the network. After having tested it, you can take it out
    np.random.seed(314)
    random.seed(314)
    K, eta, H_size, mini_batch_size = 100, 0.15, 100, 500

    print('Dataset creation')
    # Create dictionary
    dictionary, train, valid, test = tools.build_dictionary('Dataset.txt')

    print('RNN initialization')
    # Initialize RNN
    myRnn = VanillaRNN.RNN(dictionary, train, valid, test, H_size, eta)

    print('Train the RNN')
    # Train the RNN
    loss_train, loss_val = myRnn.training(K, mini_batch_size)

    print('Test the RNN')
    # Test the RNN
    loss_test = myRnn.testing()

    # Save last weights in a file
    np.savetxt('Vmat.txt', myRnn.V, delimiter=',')
    np.savetxt('Umat.txt', myRnn.U, delimiter=',')
    np.savetxt('Wmat.txt', myRnn.W, delimiter=',')

    loss = ''
    for s in loss_train:
        loss += str(s) + ', '
    file = open('lossTrain.txt', 'w')
    file.write(loss)
    file.close()

    loss = ''
    for s in loss_val:
        loss += str(s) + ', '

    with open('lossVal.txt', 'w') as f:
        f.write(loss)

    print('Loss value of the test set: ', loss_test)

    t = list(range(K))
    plt.plot(t, loss_train, 'ro', label='Training Set')
    plt.plot(t, loss_val, 'g^', label='Validation Set')
    plt.xlabel('time')
    plt.ylabel('loss value')
    plt.title('Loss function trend')
    plt.grid(True)
    plt.legend()
    # plt.savefig("test.png")
    plt.show()
