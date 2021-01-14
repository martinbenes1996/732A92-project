# -*- coding: utf-8 -*-
"""
Plots the outputs of the data.

@author: martin
"""

from gensim.models import Word2Vec
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from rapidfuzz import fuzz, process
import re
from sklearn.model_selection import train_test_split
import spacy
import sys
plt.rcParams.update({'font.size': 18})
plt.rcParams["figure.figsize"] = (12,12)

sys.path.append('src')
import config

def _path_to_drive(path):
    return 'drive/My Drive/Colab Notebooks/' + path

_words = None
def _read_words(path = 'data/words.csv', drop_longer = True):
    """"""
    global _words
    if _words is not None:
        return _words
    if config.from_drive:
        path = _path_to_drive(path)
    # read csv
    words = pd.read_csv(path, encoding='utf8', engine='python')
    words['text'] = words.text.apply(eval)
    K = words.text.apply(len)
    logging.info("loaded data: %d rows", words.shape[0])
    logging.info("loaded data: attributes %s", words.columns.to_list())
    
    # drop long sentences
    if drop_longer:
        d1 = words.shape[0]
        removed_words = K[K > config.seq_len].sum()
        total_words = K.sum()
        words = words[K <= config.seq_len]\
            .reset_index(drop = True)
        d2 = words.shape[0]
        logging.info("dropped %d of %d samples (%5.3f %%), %d of %d words (%5.3f %%)",
                     d1 - d2, d1, (1 - d2/d1) * 100,
                     removed_words, total_words, (1 - removed_words/total_words)*100)
    _words = words
    return words

_sentences = None
def _read_sentences(path = 'data/sentences.csv'):
    """"""
    # cache
    global _sentences
    if _sentences is not None:
        return _sentences
    if config.from_drive:
        path = _path_to_drive(path)
    # read csv
    sentences = pd.read_csv(path, encoding='utf8', engine='python')
    logging.info("loaded data: %d rows", sentences.shape[0])
    logging.info("loaded data: attributes %s", sentences.columns.to_list())
    # write back
    _sentences = sentences
    return sentences

# split train-test
def _read_train_words(*args, ratio = .8, **kw):
    words = _read_words(*args, **kw)
    words_tr,words_te = train_test_split(words, test_size=1-ratio, random_state=42)
    logging.warning('train set %s', words_tr.shape)
    return words_tr
def _read_test_words(*args, ratio = .8, **kw):
    words = _read_words(*args, **kw)
    words_tr,words_te = train_test_split(words, test_size=1-ratio, random_state=42)
    logging.warning('test set %s', words_te.shape)
    return words_te
def _read_train_sentences(*args, ratio = .8, **kw):
    sentences = _read_sentences(*args, **kw)
    sentences_tr,sentences_te = train_test_split(sentences, test_size=1-ratio, random_state=42)
    logging.warning('train set %s', sentences_tr.shape)
    return sentences_tr
def _read_test_sentences(*args, ratio = .8, **kw):
    sentences = _read_sentences(*args, **kw)
    sentences_tr,sentences_te = train_test_split(sentences, test_size=1-ratio, random_state=42)
    logging.warning('test set %s', sentences_te.shape)
    return sentences_te

def word_lens():
    """"""
    # data
    words = _read_words()
    # lengths
    return words.text.apply(len)

def plot_length_ratio(words = None, bins = 100, **kw):
    """"""
    # data
    words = words if words is not None else _read_words(**kw)
    
    # plot histogram
    pd.Series(K).plot(kind = "hist", bins = 100)
    plt.xlabel('word count')
    plt.ylabel('number of sentences')
    plt.show()
    logging.info("maximal sentence length: %d", K.max())

def train_data(seq_len = None, retrain = False):
    """"""
    # data
    words = _read_words(seq_len = seq_len)
    # model
    word2vec = get_model(words = words, retrain = retrain)
    
    # train data
    N = words.shape[0]
    K = words.text.apply(len).to_numpy()
    train_y = words.label.to_numpy()
    train_x = np.zeros([N, K.max()], dtype=np.int32)
    for i,row in words.iterrows():
        for j,word in enumerate(row.text):
            train_x[i,j] = word2vec.wv.vocab[word].index
    # remove empty lines
    train_x,train_y = train_x[K > 0,:],train_y[K > 0]
    K = K[K > 0]
    
    logging.info("Train data |%d x %d|", train_x.shape[0], train_x.shape[1])
    logging.info("Labels |%d x 1|", train_y.shape[0])
    
    return train_x,train_y

def markov(tokenize = True):
    # path
    path = 'output/markov.txt'
    if config.from_drive:
        path = _path_to_drive(path)
    # load
    df = pd.read_csv(path)
    df.columns = ['text']
    # tokenize
    if tokenize:
        nlp = spacy.load("en_core_web_sm",
                         disable=["tagger","ner","textcat"])#"parser",,]) # to speed up

        # implement tokenizer
        def any_alnum(x):
            return any([c.isalnum() for c in x])
        def preprocess_words(text):
            return [tok.text for tok in nlp(text) if any_alnum(tok.text)]
        # tokenization
        logging.info("tokenizing")   
        df['text'] = df.text.apply(preprocess_words)
    
    return df
