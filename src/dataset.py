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
import sys
plt.rcParams.update({'font.size': 18})
plt.rcParams["figure.figsize"] = (12,12)

sys.path.append('src')
import config

def _path_to_drive(path):
    return '/drive/My Drive/Colab Notebooks/' + path

_words = None
def _read_words(path = 'data/words.csv', drop_longer = True, from_drive = False):
    """"""
    global _words
    print(from_drive)
    if _words is not None:
        return _words
    # drive path
    if from_drive:
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
def _read_sentences(path = 'data/sentences.csv', from_drive = False):
    """"""
    # cache
    global _sentences
    if _sentences is not None:
        return _sentences
    # drive path
    if from_drive:
        path = _path_to_drive(path)
    # read csv
    sentences = pd.read_csv(path, encoding='utf8', engine='python')
    logging.info("loaded data: %d rows", sentences.shape[0])
    logging.info("loaded data: attributes %s", sentences.columns.to_list())
    # write back
    _sentences = sentences
    return sentences

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

