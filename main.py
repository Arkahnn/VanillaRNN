import matplotlib.pyplot as plt
import numpy as np
import random

import VanillaRNN
import tools


if __name__ == "__main__":
    # Set a seed to test the network. After having tested it, you can take it out
    np.random.seed(256)
    random.seed(256)
    K, eta, alpha, H_size, mini_batch_size, t_prev = 300, 0.1, 0.9, 100, 500, 4

    print('Dataset creation')
    # Create dictionary
    dictionary, train, valid, test = tools.build_dictionary('Dataset.txt')

    print('RNN initialization')
    # Initialize RNN
    myRnn = VanillaRNN.RNN(dictionary, train, valid, test, H_size, eta, alpha, t_prev)

    print('Train the RNN')
    # Train the RNN
    loss_train, loss_val, acc_train, acc_val = myRnn.training(K, mini_batch_size)

    # Save last weights in a file
    np.savetxt('Vmat.txt', myRnn.V, delimiter=',')
    np.savetxt('Umat.txt', myRnn.U, delimiter=',')
    np.savetxt('Wmat.txt', myRnn.W, delimiter=',')

    print('Test the RNN')
    # Test the RNN
    loss_test = myRnn.testing()

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
        f.close()

    acc = ''
    for s in acc_train:
        acc += str(s) + ', '
    file = open('accTrain.txt', 'w')
    file.write(acc)
    file.close()

    acc = ''
    for s in acc_val:
        acc += str(s) + ', '

    with open('accVal.txt', 'w') as f:
        f.write(acc)
        f.close()

    print('Loss value of the test set: ', loss_test)

    t = list(range(len(loss_train)))
    # loss_train = loss_train/K
    plt.plot(t, loss_train, 'ro', label='Training Set')
    plt.plot(t, loss_val, 'g^', label='Validation Set')
    plt.xlabel('time')
    plt.ylabel('loss value')
    plt.title('Loss function trend')
    plt.grid(True)
    plt.legend()
    # plt.savefig("test.png")
    plt.show()

    t = list(range(len(acc_train)))
    # loss_train = loss_train/K
    plt.plot(t, acc_train, 'ro', label='Training Set')
    plt.plot(t, acc_val, 'g^', label='Validation Set')
    plt.xlabel('time')
    plt.ylabel('accuracy value')
    plt.title('Accuracy function trend')
    plt.grid(True)
    plt.legend()
    # plt.savefig("test.png")
    plt.show()

    plt.plot(acc_val, loss_val, 'g^', label='Validation Set')
    plt.xlabel('accuracy value')
    plt.ylabel('loss value')
    plt.title('Loss VS Accuracy trend')
    plt.grid(True)
    plt.legend()
    # plt.savefig("test.png")
    plt.show()