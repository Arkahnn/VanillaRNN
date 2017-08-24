import numpy as np


def import_file(a_file):
    with open(a_file, "r") as f:
        body = f.read()
    return body


def build_dictionary(fnpath):
    limitWords = False
    len_dataset = 10000
    len_train = int(len_dataset * 0.5)
    len_val = int(len_dataset * 0.25)
    len_test = len_dataset - (len_train + len_val)
    body = import_file(fnpath)
    for char_ in [',', '.', ';', ':', '?', '\'', ')', '-', '_']:
        body = body.replace(char_, '')
    strings = body.split('\n')
    strings = strings[:len_dataset]
    dataset = []
    for s in strings:
        dataset += [[x.lower() for x in s.split()]]    # makes a list of str for each phrase; the str are all in lowercase

    if limitWords:
        minLen = 10 #len(min(dataset, key = len))
        dataset = [x[:minLen - 1] for x in dataset]

    dataset = [['<startW>'] + x + ['<endW>'] for x in dataset]

    dictionary = sorted(set([x for sub in dataset for x in sub]))   # flattens the list of lists, removes duplicates and makes a dictionary

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
