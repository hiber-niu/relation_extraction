#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
#              Generating text features for relation extraction.               #
#
# author: hiber.niu@gmail.com
# date: 2017-02-28
################################################################################


import data_preparation_tool as dpt
import fnmatch
from os import listdir
from os.path import isfile, join

from gensim.models.doc2vec import *
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
from spacy.en import English
import gensim
import multiprocessing
import numpy as np
import sklearn.metrics as metrics


DEFAULT_BOW_NGRAM_RANGE = (1, 1)
DEFAULT_BOW_MAX_FEATURES = None
DEFAULT_BOW_BINARY = True

BINARY_LABELS_TO_CLASSES_TABLE = {
    'NO_RELATION' : 0,
    'RELATION' : 1
}


class GenFeaturesMethod:
    def __init__(self, name, func, no_entities=False):
        self.name = name
        self.func = func
        self.no_entities = no_entities


class TrainTestData:
    def __init__(self, train_data, train_labels, test_data, test_labels, feature_gen_model=None):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.feature_gen_model = feature_gen_model


class EvaluationResult:
    def __init__(self, model, features_gen_model, test_data, scores):
        self.model = model
        self.test_data = test_data
        self.scores = scores
        self.features_gen_model = features_gen_model


def run_gen_features_pipeline(input_dir, gen_features_methods, evaluation_methods):
    data_dict, data_dict_no_entities = read_train_and_test_data_from_path(input_dir)
    results = []
    for name in data_dict:
        for gfm in gen_features_methods:
            print("generating %s features for %s"%(gfm.name, name))
            # whether or not remove entities before generating features.
            if (gfm.no_entities):
                data = data_dict_no_entities[name]
            else:
                data = data_dict[name]

            train_test_data = gen_features_and_classes(data, gfm.func)

            for em in evaluation_methods:
                print("model evaluation for: %s, %s, %s" % (name, gfm.name,
                                                            em.name))
                result = evaluate_model(train_test_data, em.func)
                results.append(result)
    return results


def read_train_and_test_data_from_path(path):
    only_files = [f for f in listdir(path) if (isfile(join(path, f)))]
    train_files = [f for f in only_files if fnmatch.fnmatch(f, '*_train.tsv')]
    data_names = ["_".join(f.split("_")[:-1]) for f in train_files]
    data_table = {}
    data_table_no_entities = {}

    for name in data_names:
        train_data, train_labels = read_data_from_file(join(path, name + "_train.tsv"))
        test_data, test_labels = read_data_from_file(join(path, name + "_test.tsv"))

        # without entities as well:
        train_data_no_entities = remove_entities_from_text(train_data)
        train_labels_no_entities = train_labels
        test_data_no_entities = remove_entities_from_text(test_data)
        test_labels_no_entities = test_labels

        data_table[name] = TrainTestData(train_data, train_labels, test_data, test_labels)
        data_table_no_entities[name] = TrainTestData(train_data_no_entities, train_labels_no_entities,
                                                     test_data_no_entities, test_labels_no_entities)

    return data_table, data_table_no_entities


def read_data_from_file(file_path):
    with open(file_path) as handle:
        labels = []
        data = []
        for l in handle:
            parts = l.rstrip().split("\t")
            if (len(parts) < 2):
                continue
            labels.append(parts[0])
            data.append(parts[1])
        return data, labels


def remove_entities_from_text(sentences):
    fixed_sentences = []
    indices_to_remove = []
    for s in sentences:
        new_sent = []
        for t in s.split('\t'):
            if not t.startswith('ENTITY'):
                new_sent.append(t)
        fixed_sentences.append("\t".join(new_sent))
    return fixed_sentences


def gen_features_and_classes(train_test_data, gen_features_func):
    train_classes = labels_to_classes(train_test_data.train_labels)
    test_classes = labels_to_classes(train_test_data.test_labels)
    train_features, test_features, model = gen_features_func(train_test_data.train_data, train_test_data.test_data)

    return TrainTestData(train_features, train_classes, test_features, test_classes, model)


def labels_to_classes(labels):
    classes = []
    for label in labels:
        classes.append(BINARY_LABELS_TO_CLASSES_TABLE[label])
    return classes


###################################
#  features generation functions  #
###################################
# get_bow_features
def to_bag_of_words(train_samples, test_samples, ngram_range=DEFAULT_BOW_NGRAM_RANGE,
                    max_features=DEFAULT_BOW_MAX_FEATURES, binary=DEFAULT_BOW_BINARY):
    #Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=max_features,
                                 binary=binary,
                                 ngram_range=ngram_range)

    train_data_features = vectorizer.fit_transform(train_samples)
    test_data_features = vectorizer.transform(test_samples)

    return train_data_features, test_data_features, vectorizer

def get_bow_features(train_samples, test_samples, ngram_range):
    return to_bag_of_words(train_samples, test_samples, ngram_range=ngram_range)


##########################
#  get_doc2vec_features  #
##########################
def build_doc2vec_model(data, temp_doc2vec_input_file_path, background_samples_file_path=None,
                      model_file_path=None, should_log=False):
    if (should_log):
        reload(logging)
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        logger = logging.getLogger()

    # merge the data into one file and then let gensim TaggedLineDocument class take care of the rest
    # this can be further optimized by creating a custom doc2vec iterator that will read the files in sequence
    print("creating temp file...")
    with open(temp_doc2vec_input_file_path, "w") as output:
        merge_into_file(data, output)
        merge_into_file(background_samples_file_path, output)

    with open(temp_doc2vec_input_file_path) as handle:
        print("creating model...")
        # TODO: add min_count = 5, but deal with empty sentences..
        ncpus = multiprocessing.cpu_count()
        model = Doc2Vec(TaggedLineDocument(handle), size=200, window=8, min_count=5, workers=ncpus)
        print("model built")
        if (model_file_path != None):
            model.save(model_file_path)

    return model

# get the doc2vec feature vectors
def get_doc2vec_features(train_data, test_data,
                         temp_doc2vec_input_file_path, background_samples_file_path=None):
    input_data = train_data + test_data
    model = build_doc2vec_model(input_data, temp_doc2vec_input_file_path, background_samples_file_path, should_log=True)

    # extract the vectors according to their class
    train_embeddings = [model.docvecs[index] for index in xrange(len(train_data))]
    test_embeddings = [model.docvecs[index] for index in xrange(len(train_data), len(train_data) + len(test_data))]
    #background_embeddings = [model.docvecs[index] for index in xrange(len(train_data) + len(test_data), model.docvecs.count)]

    return train_embeddings, test_embeddings, model


def merge_into_file(input_path_or_data, output):
    if (input_path_or_data == None):
        return

    # if it's data and not path
    if (dpt.is_sequence(input_path_or_data)):
        for l in input_path_or_data:
            output.write(l + "\n")
        return len(input_path_or_data)

    count = 0;
    with open(input_path_or_data) as input:
        for l in input:
            output.write(l)
            count = count + 1
        return count

################
#  to_pos_bow  #
################
nlp_parser = None
def to_nlp_objs(sentences):
    global nlp_parser
    # init once
    if (nlp_parser == None):
        nlp_parser = English()

    nlp_objs = []
    for s in sentences:
        nlp_objs.append(nlp_parser(s.decode('unicode-escape'), entity=False))
    return nlp_objs

def get_nlp_features(sentences):
    parsed = to_nlp_objs(sentences)
    pos_tags = []
    for p in parsed:
        pos_tags.append([s.pos_ for s in p])
    return pos_tags


def to_pos_bow(train_samples, test_samples, ngram_range=DEFAULT_BOW_NGRAM_RANGE, binary=DEFAULT_BOW_BINARY):
    #TODO: can do this more efficiently, this is a workaround for now
    pos_tags_train = [" ".join(s) for s in get_nlp_features(train_samples)]
    pos_tags_test = [" ".join(s) for s in get_nlp_features(test_samples)]
    return to_bag_of_words(pos_tags_train, pos_tags_test, ngram_range=ngram_range, binary=binary, max_features=None)

##############################
#  get_bow_and_pos_features  #
##############################
def get_bow_and_pos_features(train_samples, test_samples, ngram_range, pos_ngram_range):
    bow_train_features, bow_test_features, vectorizer = get_bow_features(train_samples, test_samples, ngram_range)
    pos_train_features, pos_test_features, vectorizer = to_pos_bow(train_samples, test_samples, ngram_range=pos_ngram_range)

    train_features = hstack((bow_train_features, pos_train_features))
    test_features = hstack((bow_test_features, pos_test_features))

    # The vectorizers of bow and pos_features is not returned.
    return train_features, test_features, None

###########################
#  get_compound_features  #
###########################
def get_compound_features(train_data, test_data, feature_gen_methods):
    train_features_list = []
    test_features_list = []

    for m in feature_gen_methods:
        # model is not used.
        train_features, test_features, model = m(train_data, test_data)
        train_features_list.append(train_features)
        test_features_list.append(test_features)

    train_features = train_features_list[0]
    test_features = test_features_list[0]

    for i in xrange(1, len(feature_gen_methods)):
        train_features = hstack((train_features, train_features_list[i]))
        test_features = hstack((test_features, test_features_list[i]))

    return train_features, test_features, None

################
#  Evaluation  #
################
class EvaluationMethod:
    def __init__(self, name, func):
        self.name = name
        self.func = func


def evaluate_model(train_test_data, model_initializer):
    clf = model_initializer()
    clf = clf.fit(train_test_data.train_data, train_test_data.train_labels)

    labels_predicted = clf.predict(train_test_data.test_data)
    scores_predicted = clf.predict_proba(train_test_data.test_data)
    print(metrics.classification_report(train_test_data.test_labels,
                                        labels_predicted))
    return EvaluationResult(clf, train_test_data.feature_gen_model, train_test_data.test_data, scores_predicted)
