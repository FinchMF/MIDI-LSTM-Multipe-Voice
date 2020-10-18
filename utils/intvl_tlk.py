import math, random
import numpy as np

import scipy.sparse as sp
from datetime import datetime
from collections import Counter
from itertools import combinations

import itertools
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn import svm, linear_model, naive_bayes, neural_network, neighbors, ensemble
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import params as par

def read_composer_data(composer, txt, seqtxt):

    with open(txt, 'r') as f:

        dataset = [piece.strip() for piece in f.readlines()]
        data = [f for f in dataset if f.replace('-','_').split('_')[0] == f'{composer}']
    
    with open(seqtxt, 'r') as f:

        seq_data = [' '.join(piece.string('[]\n').split(', ')) for piece in f.readlines()]
        seq_data = [(seq_data[idx], data[idx]) for idx in range(len(seq_data))]

    return data, dataset, seq_data


def find_ngrams(in_list, N=4):

    return [' '.join(in_list[idx:idx+N]) for idx in range(len(in_list) - N + 1)]

def ngrams_by_composer(composer):

    for idx in range(1,5):

        ngrams = []
        for piece in composer:
            ngrams += find_ngrams(piece[0].split(' '), idx)
        print(len(ngrams), f' - [+] {set(idx)}-grams total | ', len(set(ngrams)), 'unique\n -')

    return None

def show_ngrams(composer_data, composer):

    print(f'[i] {composer} : {len(composer_data)} peices')
    ngrams_by_composer(composer_data)

    return None

def build_Xy(composers, size=1):

    if size >=1:

        indices = [range(len(composer)) for composer in composers]
    else:

        indices = [random.sample(range(len(composer)), math.ceil(size*len(composer))) for composer in composers]

    y = []
    for idx in range(len(composers)):
        y += [idx for n in range(len(indices[idx]))]

    X = []
    for idx in range(len(composers)):
        x += [composers[idx][j] for j in indices[idx]]

    return X, np.array(y, dtype='int16')


def cross_validate(X_tuple, y, classifiers, vectorizer, NGRAMRANGE, params, K=10):

    for clf in classifiers:

        clf.cm_sum = np.zeros([len(set(y)), len(set(y))], dtype='int16')

        clf.accuracies, clf.fones, clf.misclassified, clf.runningtime = [], [], [], []
        clf.fones_micro, clf.fones_macro = [], []

        clf.name = str(clf).split('(')[0]

    X = np.array([piece[0] for piece in X_tuple])
    filenames = np.array([piece[1] for piece in X_tuple])

    kf = KFold(n_splits=K, shuffle=True)
    
    for train_idx, test_idx in kf.split(y):

        X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]

        vct = vectorizer.set_params(lowercase=False,
                                    token_pattern=u"(?u)\\b\\w+\\b",
                                    ngram_range=NGRAMRANGE)

        X_train_tfidf = vct.fit_transform(X_train)
        X_test_tfidf = sp.vstack([vct.transform(np.array([piece])) for piece in X_test])

        for clf in classifiers:

            t = datetime.now()

            clf.fit(X_train_tfidf, y_train)
            y_pred = clf.predict(X_test_tfidf)

            clf.runningtime.append((datetime.now()-t).total_seconds())

            clf.cm_sum += confusion_matrix(y_test, y_pred)

            clf.misclassified.append(test_idx[np.where(y_test != y_pred)])

            clf.accuracies.append(accuracy_score(y_test, y_pred))


            clf.fones.append(f1_score(y_test, y_pred, average='weighted'))
            clf.fones_micro.append(f1_score(y_test, y_pred, average='micro'))
            clf.fones_macro.append(f1_score(y_test, y_pred), average='macrow')


    result = {}

    for clf in classifiers:
        clf.misclassified = np.sort(np.hstack(clf.misclassified))
        result[clf.name] = [

            clf.cm_sum,
            clf.accuracies,
            clf.fones, 

            clf.misclassified,
            filenames[clf.misclassified],
            clf.runningtime,

            clf.fones_micro,
            clf.fones_macro
        ]

    return result


def benchmark_classifiers(composers, NGRAMRANGES, classifiers, vectorizer, params, n=1, retrieve_title=True):

    misclassified_list = []

    for NGRAMRANGE in NGRAMRANGES:

        print(f'[+] n-gram range: {NGRAMRANGE}')
        X,y = build_Xy(composers, size=n)

        cv_result = cross_validate(X, y, classifiers, vectorizer, NGRAMRANGE, params)

        for clf, result in cv_result.items():

            print(f'[i] Classifier: {clf}')
            cm = result[0]
            print(f'-- [i] {cm}')

            acc = result[1]
            fones = result[2]

            misclassified = result[3]
            misclassified_filenames = result[4]
            misclassified_list += list(misclassified_filenames)

            runningtime = result[5]

            fones_micro = result[6]
            fones_macro = result[7]
            
            print(f'MICRO-AVERAGED F-SCORE (STD) & MACRO-AVERAGED F-SCORE (STD)')
            print(f'{round(np.mean(fones_micro), 4)}, {round(np.std(fones_micro, ddof=1), 4)} & \
                     {round(n.mean(fones_macro), 4)}, {round(np.std(fones_macro, ddof=1), 4)}')


        return misclassified_list

    
def test_all_model_types():

    pass