import numpy as np
import re

#Function that import a file
def import_file(aFile):
    fileTrain = open(aFile,"r")
    body = fileTrain.read()
    fileTrain.close()
    return body
    
def build_list_strings(aList):
    list_string_words = []
    for s in aList:
        list_string_words += [s.split(' ')]
    return list_string_words

'''
#Function that build the dictionary ad the data set
def build_Dictionary(train_set,test_set):
    body1 = import_file(train_set)
    body2 = import_file(test_set)
    body1 = re.sub(r'[^\w]', ' ', body1)
    body2 = re.sub(r'[^\w]', ' ', body2)
    stringTrain = body1.split('\n')
    stringTest = body2.split('\n')
    words = []
    for s in stringTrain:
        words += s.split(' ')
    for s in stringTest:
        words += s.split(' ')
    dictionary = sorted(set(words))
    dictionary = ['<startW>','<endW>','<NaW>'] + dictionary
    
    return (dictionary, stringTrain, stringTest)
'''

def build_Dictionary(data_set):
    body = import_file(data_set)
    #body = re.sub(r'[^\w]', ' ', body)
    body = body.replace(',', '')
    body = body.replace('.', '')
    body = body.replace('?', '')
    strings = body.split('\n')
    words = []
    for s in strings:
        words += s.split(' ')
    dictionary = sorted(set(words))
    dictionary = ['<startW>','<endW>','<NaW>'] + dictionary
    
    #stringTrain = strings[:len(strings)//2]
    #stringValid = strings[len(strings)//2:(len(strings)//2 + len(strings)//4)]
    #stringTest = strings[(len(strings)//2 + len(strings)//4):]
    print('Strings dimension: ',len(strings))
    lim = 50000
    stringTrain = build_list_strings(strings[:lim])
    stringValid = build_list_strings(strings[lim:(lim + (len(strings)-lim)//2)])
    stringTest = build_list_strings(strings[(lim + (len(strings)-lim)//2):])
    
    return (dictionary, stringTrain, stringValid, stringTest)