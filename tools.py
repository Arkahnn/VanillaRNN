import numpy as np


def import_file(a_file):
    with open(a_file, "r") as f:
        body = f.read()
    return body


def build_dictionary(fnpath):
    len_dataset = 10000
    len_train = int(len_dataset * 0.5)
    len_val = int(len_dataset * 0.25)
    len_test = len_dataset - (len_train + len_val)
    body = import_file(fnpath)
    for char_ in [',', '.', ';', ':', '?']:
        body = body.replace(char_, '')
    strings = body.split('\n')
    strings = strings[:len_dataset]

    # words = [s.split(' ') for s in strings] # Non funziona, restituisce una lista di liste e non serve a niente!
    words = []
    for s in strings:
        words += s.split()
    dictionary = sorted(set(words))
    dictionary = ['<startW>', '<endW>'] + dictionary

    dataset = [# [dictionary[0]] +
               s.split()
               # + [dictionary[1]]
                for s in strings]
    print('dataset dimension: ', len(dataset))

    train = dataset[:len_train]
    val = dataset[len_train:(len_train + len_val)]
    test = dataset[-len_test:]

    return dictionary, train, val, test


def import_matrix():
    V = np.loadtxt('Vmat.txt', delimiter=',')
    U = np.loadtxt('Umat.txt', delimiter=',')
    W = np.loadtxt('Wmat.txt', delimiter=',')

    return V, U, W
