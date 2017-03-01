#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("./packages")

import matplotlib
import datetime
import numpy as np
import dpt
import features_generation_tool as fgt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.ensemble import GradientBoostingClassifier


# path used to save temporary doc2vec files
temp_doc2vec_file = r"./output/temp_doc2vec.txt"
# path to text file that contains background sentences used in doc2vec
background_samples_file_path = r"./output/background_samples.txt"

doc2vec_func = lambda x_train,x_test : fgt.get_doc2vec_features(x_train, x_test, temp_doc2vec_file, background_samples_file_path)
bow_func = lambda x_train,x_test : fgt.get_bow_features(x_train, x_test, (1,3))

# evaluate different features
gen_features_methods = [
    fgt.GenFeaturesMethod("bow_1_gram", lambda x_train, x_test : fgt.get_bow_features(x_train, x_test, (1,1))),
    fgt.GenFeaturesMethod("bow_2_gram", lambda x_train, x_test : fgt.get_bow_features(x_train, x_test, (2,2))),
    fgt.GenFeaturesMethod("bow_3_gram", lambda x_train, x_test : fgt.get_bow_features(x_train, x_test, (3,3))),
    fgt.GenFeaturesMethod("bow_1_3_gram", lambda x_train, x_test : fgt.get_bow_features(x_train, x_test, (1,3))),
    fgt.GenFeaturesMethod("doc2vec", lambda x_train, x_test : fgt.get_doc2vec_features(x_train, x_test, temp_doc2vec_file, background_samples_file_path)),
    fgt.GenFeaturesMethod("pos_3_3", lambda x_train, x_test : fgt.to_pos_bow(x_train, x_test, (3,3))),
    fgt.GenFeaturesMethod("bow_1_3_pos_3_3", lambda x_train, x_test : fgt.get_bow_and_pos_features(x_train, x_test, (1,3), (3,3))),
    fgt.GenFeaturesMethod("bow_1_3_doc2vec", lambda x_train, x_test : fgt.get_compound_features(x_train, x_test, [bow_func, doc2vec_func]))
]

#Cs= [0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.8] + np.linspace(1,5, 9).tolist()
Cs = np.linspace(0.005,0.25,10)

# evaluates different classifiers
evaluation_methods = [
    fgt.EvaluationMethod("logistic regression l1", lambda: LogisticRegression(C=0.1, penalty='l1', solver='liblinear')),
    fgt.EvaluationMethod("lr l1 cv", lambda: LogisticRegressionCV(penalty='l1', cv=5, scoring=make_scorer(f1_score), solver='liblinear', Cs=Cs, refit=True)),
    fgt.EvaluationMethod("lr l2 cv", lambda: LogisticRegressionCV(penalty='l2', cv=5, scoring=make_scorer(f1_score), solver='liblinear', Cs=Cs, refit=True)),
    #fgt.EvaluationMethod("GBC", lambda: GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=10, random_state=0))
]

# path to input dir
input_dir = r"./output"
startTime = datetime.datetime.now()

models = fgt.run_gen_features_pipeline(input_dir, gen_features_methods, evaluation_methods)

runTime = datetime.datetime.now() - startTime
print("Finished generating features, took:%s"%runTime)
