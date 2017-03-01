#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Data preparation for realtion extraction, including tokenization, entity     #
# replacement, trimming, normalization, stopwords/digital removel, stemming    #
#
# author: hiber.niu@gmail.com
# date: 2017-02-28
################################################################################


import numpy as np
import nltk
import math
from sklearn.cross_validation import train_test_split
from os import path

DEFAULT_SENTENCE_COLUMN = 5
DEFAULT_LABEL_COLUMN = 0
DEFAULT_ENTITY_A_COLUMNS = [1, 2]
DEFAULT_ENTITY_B_COLUMNS = [3, 4]
DEFAULT_MIN_SENTENCE_LENGTH = 8
DEFAULT_EXTRA_WORDS_COUNT = 2
DEFAULT_TEST_SIZE = 0.25


def run_data_preparation_pipeline(positive_samples_file_path, negative_samples_file_path,
                                  output_files_prefix, output_dir):
    # two classes case:
    pos_sentences, pos_labels, pos_contexts = extract_sentences(positive_samples_file_path, label_tag='RELATION')
    neg_sentences, neg_labels, neg_contexts = extract_sentences(negative_samples_file_path, label_tag='NO_RELATION', sample_size = len(pos_sentences))
    contexts = pos_contexts + neg_contexts
    sentences = pos_sentences + neg_sentences
    labels = pos_labels + neg_labels

    print("producing inputs for binary case...")
    run_transformations_on_data(sentences, labels, contexts, output_files_prefix + "_binary", output_dir)


# extract data from csv file.
def extract_sentences(input_file_path,
                      sentence_column=DEFAULT_SENTENCE_COLUMN,
                      entity_a_column=DEFAULT_ENTITY_A_COLUMNS,
                      entity_b_column=DEFAULT_ENTITY_B_COLUMNS,
                      label_tag=None, sample_size=-1):
    sentences = []
    labels = []
    contexts = []
    all_sentences = {}
    with open(input_file_path) as input:
        for line in input:
            splitted_line = line.rstrip().split("\t")

            # remove duplicate lines.
            if (all_sentences.has_key(line.rstrip())):
                continue
            all_sentences[line.rstrip()] = 1

            # range of entity in the sentence(add 5 for header infos).
            contexts.append({"entity_a": '\t'.join(splitted_line[entity_a_column[0]+5:entity_a_column[1]+5]),
                             "entity_a_range": [splitted_line[entity_a_column[0]], splitted_line[entity_a_column[1]]],
                             "entity_b": '\t'.join(splitted_line[entity_b_column[0]+5:entity_b_column[1]+5]),
                             "entity_b_range": [splitted_line[entity_b_column[0]], splitted_line[entity_b_column[1]]]})

            sentence = '\t'.join(splitted_line[sentence_column:])
            sentences.append(sentence)

    if (sample_size != -1):
        indices = np.random.choice(len(sentences), sample_size)
        sentences = [sentences[index] for index in indices]
        contexts = [contexts[index] for index in indices]

    if (label_tag != None):
        labels = [label_tag] * len(sentences)
    return (sentences, labels, contexts)


# Object to pass along and contains the']]put to the different transformations
class InputData:
    def __init__(self, data_train, data_test, contexts_train, contexts_test):
        self.data_train = data_train
        self.data_test = data_test
        self.contexts_train = contexts_train
        self.contexts_test = contexts_test


def extract_and_replace_entities(text, context):
    words = text.split('\t')
    a_begin = int(context['entity_a_range'][0])
    a_end = int(context['entity_a_range'][1])
    b_begin = int(context['entity_b_range'][0])
    b_end = int(context['entity_b_range'][1])

    # XXX A entity is not in the another entity.
    if a_end <= b_begin:
        words = words[:a_begin]+['ENTITYA']+words[a_end:b_begin]+['ENTITYB']+words[b_end:]
    elif b_end < a_begin:
        words = words[:b_begin]+['ENTITYB']+words[b_end:a_begin]+['ENTITYA']+words[a_end:]
    else:
        print("A entity is in the another entity! This should not happened.")

    replaced_text = '\t'.join(words)
    return replaced_text


def trim_sentence_around_entities(text, context=None, min_length=DEFAULT_MIN_SENTENCE_LENGTH, extra_words_count=DEFAULT_EXTRA_WORDS_COUNT):
    sentence_parts = text.split('\t')
    if (len(sentence_parts) < min_length):
        return text

    first_index = -1
    last_index = -1
    for part, i in zip(sentence_parts, xrange(len(sentence_parts))):
        if part.startswith('ENTITY'):
            if (first_index == -1):
                first_index = i
            last_index = i

    size = last_index - first_index + extra_words_count * 2
    # ensure
    if (size < min_length):
        extra_words_count = extra_words_count + int(math.ceil((min_length - size) / 2))

    first_index = max(0, first_index-extra_words_count)
    last_index = min(len(sentence_parts), last_index+extra_words_count + 1)
    trimmed_sentence_parts = sentence_parts[first_index:last_index]
    return "\t".join(trimmed_sentence_parts)


# langueage based sentence process
def normalize_text(sent, context=None):
    return sent.lower()


# digits removal
def remove_all_digit_tokens(sent, context=None):
    processed_tokens = []
    tokens = sent.split('\t')
    for t in tokens:
        # ignore stop words
        if (t.isdigit()):
            continue
        processed_tokens.append(t)

    return "\t".join(processed_tokens)


# stop words removal
def remove_stop_words(sent, context=None):
    processed_tokens = []
    tokens = sent.split('\t')
    for t in tokens:
        # ignore stop words
        try:
            if (t in nltk.corpus.stopwords.words('english') or len(t) < 2):
                continue
        except Exception as e:
            print(e)
            continue
        processed_tokens.append(t)

    return "\t".join(processed_tokens)


# run stemmer on the words
def stem_text(sent, context=None):
    processed_tokens = []
    tokens = sent.split('\t')
    porter = nltk.PorterStemmer()
    for t in tokens:
        try:
            t = porter.stem(t)
        except Exception as e:
            print(e)
            continue
        processed_tokens.append(t)

    return "\t".join(processed_tokens)


# extract data from CSV/TSV file, the 3rd part specify required step
TRANSFORMATION_STEPS = [('entities', extract_and_replace_entities),
                        ('trim', trim_sentence_around_entities, ['entities']),
                        ('normalize', normalize_text, ['entities']),
                        ('rmdigits', remove_all_digit_tokens, ['entities']),
                        ('rmstopwords', remove_stop_words, ['entities']),
                        ('stem', stem_text, ['entities'])
                        ]


def run_step(step_name, step_func, inputs_dict, required):
    temp_dict = {}
    for k in inputs_dict:
        if (required != None):
            found_all = True
            for r in required:
                if (k.find(r) == -1):
                    found_all = False
            if (not found_all):
                continue

        result_train = []
        result_test = []

        for l, c in zip(inputs_dict[k].data_train, inputs_dict[k].contexts_train):
            result_train.append(step_func(l, context=c))

        for l, c in zip(inputs_dict[k].data_test, inputs_dict[k].contexts_test):
            result_test.append(step_func(l, context=c))

        temp_dict[k + "_" + step_name] = InputData(result_train, result_test,
                                                   inputs_dict[k].contexts_train, inputs_dict[k].contexts_test)

    for k in temp_dict:
        inputs_dict[k] = temp_dict[k]


def run_transformations_on_data(sentences, labels, contexts, output_files_prefix, output_dir):
    # split to train and test:
    print("before data split")
    s_train, s_test, l_train, l_test, c_train, c_test = split_to_test_and_train(sentences, labels, contexts)
    inputs = {
        'data' : InputData(s_train, s_test, c_train, c_test)
    }

    print("splitted data")
    # run each step on all of the already existing ones

    for s in TRANSFORMATION_STEPS:
        print("running step:%s" % (s[0]))
        required = None
        if (len(s) > 2):
            required = s[2]
        run_step(s[0], s[1], inputs, required)

    # todo: write outputs to files
    for name in inputs:
        train_file_path = path.join(output_dir, output_files_prefix + "_" + name + "_train.tsv")
        test_file_path = path.join(output_dir, output_files_prefix + "_" + name + "_test.tsv")

        train_data = inputs[name].data_train
        test_data = inputs[name].data_test

        with open(train_file_path, "w") as handle:
            for text, label in zip(train_data, l_train):
                if (len(text.strip()) == 0):
                    continue
                handle.write("%s\t%s\n" % (label, text))

        with open(test_file_path, "w") as handle:
            for text,label in zip(test_data,l_test):
                if (len(text.strip()) == 0):
                    continue
                handle.write("%s\t%s\n" % (label, text))


# Split to train and test sample sets:
def split_to_test_and_train(data, labels, entities, test_size=DEFAULT_TEST_SIZE):
    d_train, d_test, l_train, l_test, c_train, c_test = train_test_split(data, labels, entities, test_size=test_size)
    d_test_2 = []
    l_test_2 = []
    c_test_2 = []

    # remove samples that in train set from test set.
    train_dict = {}
    for d in d_train:
        train_dict[d] = 1

    for d,l,c in zip(d_test, l_test, c_test):
        if (train_dict.has_key(d)):
            continue
        d_test_2.append(d)
        l_test_2.append(l)
        c_test_2.append(c)

    return (d_train, d_test_2, l_train, l_test_2, c_train, c_test_2)


def is_sequence(arg):
    return (not hasattr(arg, "strip") and hasattr(arg, "__getitem__") or hasattr(arg, "__iter__"))
