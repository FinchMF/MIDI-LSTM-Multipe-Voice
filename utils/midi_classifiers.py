

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn import svm, linear_model, naive_bayes, neural_network, neighbors, ensemble
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
import random, math
import numpy as np
import scipy.sparse as sp
from datetime import datetime
from collections import Counter
from itertools import combinations

import itertools


def set_classifier_params():

    classification_models = {}

    classification_models['NGRAMRANGES'] = [(1,1), (2,2), (3,3), (4,4), (1,2), (3,4), (1,4)]

    classification_models['SVC'] = svm.LinearSVC(penalty='l2', C=5, loss='hinge')
    classification_models['Logistic_Regression'] = linear_model.LogisticRegression(penalty='l2', C=100, tol=1, multi_class='multinomial', solver='sag')
    classification_models['KNeighbors'] = neighbors.KNeighborsClassifier(weights='distance')
    classification_models['Naive_bayes'] = naive_bayes.MultinomialNB(alpha=0.00001, fit_prior=False)
    classification_models['MLP'] = neural_network.MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10,))

    classification_models['TFIDF_VECT'] = TfidfVectorizer(sublinear_tf=True)
    classification_models['COUNT_VECT'] = CountVectorizer(binary=True)

    return classification_models


def get_typelength_dict(COMPOSERS):

    flatten = lambda l: [ele for sub in l for ele in sub]
    n_composers = range(len(COMPOSERS))
    
    def get_duration_ngrams(composer):

        with open(composer, 'r') as f:
            data = [line.strip() for line in f.readlins()]
        return data

    comp_data = {}

    for composer in n_composers:

        data = get_duration_ngrams(composer)
        type_length = [piece.split(';') for piece in data]

        comp_data[str(composer)]= type_length

    return comp_data

    def combine_typelengths(comp_data):

        flatten = lambda l: [ele for sub in l for ele in sub]
        typelengths = []
        
        for composer, data in comp_data.items():  
            typelengths.append(data)

        combined = list(set(flatten(typelengths)))
        ty_dict = {combined[idx]: str(idx+300) for idx in range(len(combined))}

        typelengths = {}

        for composer, data in comp_data.items():

            x = [(' '.join([ty_dict[dur] for dur in piece]),'temp') for piece in data]
            typelengths[f'{composer}'] = x

        return typelengths

        

        