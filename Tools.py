import numpy as np
import re


# Function that import a file
def import_file(aFile):
    fileTrain = open(aFile, "r")
    body = fileTrain.read()
    fileTrain.close()
    return body


def build_list_strings(aList):
    list_string_words = []
    for s in aList:
        list_string_words += [s.split(' ')]
    #return [s for s in list_string_words if len(s) < dimString]
    return list_string_words


def build_dictionary(dataset):
    body = import_file(dataset)
    # body = re.sub(r'[^\w]', ' ', body)
    body = body.replace(',', '')
    body = body.replace('.', '')
    body = body.replace(';', '')
    body = body.replace(':', '')
    body = body.replace('?', '')
    strings = body.split('\n')
    #n_phrases = len(strings)
    strings = strings[0:10000]
    n_phrases = len(strings)

    #words = [s.split(' ') for s in strings] # Non funziona, restituisce una lista di liste e non serve a niente!
    words = []
    for s in strings:
        words += s.split(' ')
    dictionary = sorted(set(words))
    dictionary = dictionary[1:] #per rimuovere la stringa vuota
    dictionary = ['<startW>', '<endW>'] + dictionary

    # stringTrain = strings[:len(strings)//2]
    # stringValid = strings[len(strings)//2:(len(strings)//2 + len(strings)//4)]
    # stringTest = strings[(len(strings)//2 + len(strings)//4):]
    print('Strings dimension: ', n_phrases)
    #lim = 50000
    lim = 5000
    stringTrain = build_list_strings(strings[:lim])
    stringTrain = list(filter(lambda a: a != [''], stringTrain))
    stringValid = build_list_strings(strings[lim:(lim + (n_phrases - lim) // 2)])
    stringValid = list(filter(lambda a: a != [''], stringValid))
    stringTest = build_list_strings(strings[(lim + (n_phrases - lim) // 2):])
    stringTest = list(filter(lambda a: a != [''], stringTest))

    return dictionary, stringTrain, stringValid, stringTest

def importMatrix():
    V = np.loadtxt('Vmat.txt', delimiter=',')
    U = np.loadtxt('Umat.txt', delimiter=',')
    W = np.loadtxt('Wmat.txt', delimiter=',')

    return V, U, W

# #Function that build the dictionary ad the data set
# def build_Dictionary(train_set,test_set):
#     body1 = import_file(train_set)
#     body2 = import_file(test_set)
#     body1 = re.sub(r'[^\w]', ' ', body1)
#     body2 = re.sub(r'[^\w]', ' ', body2)
#     stringTrain = body1.split('\n')
#     stringTest = body2.split('\n')
#     words = []
#     for s in stringTrain:
#         words += s.split(' ')
#     for s in stringTest:
#         words += s.split(' ')
#     dictionary = sorted(set(words))
#     dictionary = ['<startW>','<endW>','<NaW>'] + dictionary
#
#     return (dictionary, stringTrain, stringTest)

