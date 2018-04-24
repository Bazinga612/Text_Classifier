import sys,os
import glob
import numpy as np
import pandas as pd
import json
import csv as csv
import pandas as pd
from sklearn.cross_validation import train_test_split
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import multi_class_svm
import multi_class_neural_network
import multi_label_neural_network

#reading filename
def read_filename(inputFileName):
    abs_data_path = os.getcwd()
    df = pd.read_csv(abs_data_path + "/" + inputFileName, encoding='utf-8')
    return df

#pre-processing the file, in this function, I create the test and the training set
def preprocessing(df,multi_label):

    formatted_list = []
    for i, row in df.iterrows():
        if row['Case Type']:
            #splitting the case-type to identify the cases for each titles
            cases = row['Case Type'].replace(', ', ':').replace(' | ', ':').replace('/', ':').split(':')
            for values in cases:
                if values.strip() not in formatted_list:
                    formatted_list.append(values.strip())

    #the 8 CaseTypes
    #title dictionary would contain titles for each of the 8 cases
    title = {'business competition': [], 'competitive response': [], 'industry analysis': [], 'market sizing': [],
             'new business': [], 'organizational behavior': [], 'increase sales': [], 'mergers & acquisitions': []}

    for i, row in df.iterrows():
        if row['Case Type']:
            cases = row['Case Type'].replace(', ', ':').replace(' | ', ':').replace('/', ':').split(':')
            for values in cases:
                if values.strip() in title.keys():
                    title[values.strip()].append(row['title'])


    print("------------------------preprocessing done-------------------------------------------------------------")

    # forming test and training data for multi-label
    if multi_label==1:
        x_data = []
        y_data = []
        i = 0
        for values in title:
            for value in title[values]:
                if value in x_data:
                    index = x_data.index(value)
                    y_data[index].append(i)
                else:
                    x_data.append(value)
                    y_data.append([i])
            i = i + 1

        i = 0
        y2_data = []
        for i in range(len(y_data)):
            y2_data.append([0, 0, 0, 0, 0, 0, 0, 0])
            i = i + 1
        i = 0
        for i in range(len(y_data)):
            for value in y_data[i]:
                y2_data[i][value] = 1
            i = i + 1

        X_train, X_test, y_train, y_test = train_test_split(x_data, y2_data, test_size=0.2, random_state=1)

    # forming test and training data for multi-class
    else:
        x_data = []
        y_data = []
        i = 0
        for values in title:

            for value in title[values]:
                if value not in x_data:
                    x_data.append(value)
                    y_data.append(i)
            i = i + 1
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1)
    print("------------------------test and train set ready-------------------")
    return X_train,X_test,y_train,y_test

#creating word-embeddings for each sample manually
def form_word_embeddings_samples(X_train,X_test):

    #creating vocabulary for training
    X_train = [word_tokenize(x.lower()) for x in X_train]
    X_test = [word_tokenize(x.lower()) for x in X_test]

    x_distr = FreqDist(np.concatenate(X_train + X_test))
    x_vocab = x_distr.most_common(min(len(x_distr), 10000))


    x_idx2word = [word[0] for word in x_vocab]
    x_idx2word.insert(0, '<PADDING>')
    x_idx2word.append('<NA>')

    x_word2idx = {word: idx for idx, word in enumerate(x_idx2word)}

    x_train_seq = np.zeros((len(X_train), 20), dtype=np.int32)# padding implicitly present, as the index of the padding token is 0

    #using an embedding for samples training data
    for i, da in enumerate(X_train):
        for j, token in enumerate(da):
            # truncate long Titles
            if j >= 20:
                break

            # represent each token with the corresponding index
            if token in x_word2idx:
                x_train_seq[i][j] = x_word2idx[token]
            else:
                x_train_seq[i][j] = x_word2idx['<NA>']

    x_test_seq = np.zeros((len(X_test), 20),dtype=np.int32)  # padding implicitly present, as the index of the padding token is 0

    # form embeddings for samples testing data
    for i, da in enumerate(X_test):
        for j, token in enumerate(da):
            # truncate long Titles
            if j >= 20:
                break

            # represent each token with the corresponding index
            if token in x_word2idx:
                x_test_seq[i][j] = x_word2idx[token]
            else:
                x_test_seq[i][j] = x_word2idx['<NA>']

    print("---------------------------formed word-embeddings for samples-----------------------------------")
    return x_train_seq,x_test_seq

#forming one-hot representations for labels: train and test both
def form_word_embeddings_labels(y_train,y_test):

    #one hot representation for training labels
    y_distr = FreqDist(y_train + y_test)
    y_idx2word = [word[0] for word in sorted(y_distr.items())]
    y_word2idx = {token: idx for idx, token in enumerate(y_idx2word)}
    y_train_one_hot = np.zeros((len(y_train), len(y_word2idx)), dtype=np.int8)

    # one hot representation for testing labels
    for i, class_name in enumerate(y_train):
        y_train_one_hot[i][y_word2idx[class_name]] = 1
    y_test_one_hot = np.zeros((len(y_test), len(y_word2idx)), dtype=np.int8)
    for i, class_name in enumerate(y_test):
        y_test_one_hot[i][y_word2idx[class_name]] = 1

    print("---------------------------formed word-embeddings for labels-----------------------------------")
    return y_train_one_hot,y_test_one_hot

#call the respective classifier
def invoke_classifier(file_name,multi_label,svm):
    df=read_filename(file_name)
    X_train, X_test, y_train, y_test=preprocessing(df,multi_label)

    #this is a neural network and requires word_embeddings to be created before classifying
    if multi_label == 1:
        x_train_seq, x_test_seq=form_word_embeddings_samples(X_train, X_test)
        multi_label_neural_network.run_classifier(x_train_seq, x_test_seq,y_train,y_test);

    else:
        if svm==1:
            multi_class_svm.run_classifier(X_train, X_test, y_train, y_test);

        # this is a neural network and requires word_embeddings to be created before classifying
        else:
            x_train_seq, x_test_seq = form_word_embeddings_samples(X_train, X_test)
            y_train_one_hot,y_test_one_hot=form_word_embeddings_labels(y_train, y_test)
            multi_class_neural_network.run_classifier(x_train_seq, x_test_seq,y_train_one_hot,y_test_one_hot);

