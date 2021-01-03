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
plt.rcParams.update({'font.size': 18})
plt.rcParams["figure.figsize"] = (12,12)

def _read_words(path = 'data/words.csv', seq_len = None):
    """"""
    # read csv
    words = pd.read_csv(path, encoding='utf8', engine='python')
    words['text'] = words.text.apply(eval)
    K = words.text.apply(len)
    logging.info("loaded data: %d rows", words.shape[0])
    logging.info("loaded data: attributes %s", words.columns.to_list())
    
    # drop long sentences
    if seq_len is not None:
        d1 = words.shape[0]
        removed_words = K[K > seq_len].sum()
        total_words = K.sum()
        words = words[K <= seq_len]\
            .reset_index(drop = True)
        d2 = words.shape[0]
        logging.info("dropped %d of %d samples (%5.3f %%), %d of %d words (%5.3f %%)",
                     d1 - d2, d1, (1 - d2/d1) * 100,
                     removed_words, total_words, (1 - removed_words/total_words)*100)
    return words
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

def get_model(model_path = 'models/word2vec.model', retrain = False, words = None):
    """"""
    if retrain:
        # data
        words = _read_words() if words is None else words
        # collect vocabulary
        vocab,i = {},1
        for sentence in words.text:
            for word in sentence:
                if word.lower() not in vocab:
                    vocab[word.lower()] = i
                    i += 1
        df = pd.DataFrame({'word': vocab.keys(), 'value': vocab.values()})
        # fit embedding
        word2vec = Word2Vec(sentences=words.text, size=12,
                            workers=4, window=5, iter=30)
        word2vec_words = pd.DataFrame({'vocab': word2vec.wv.vocab.keys()})
        # map word
        def map_word(i):
            # find similar word
            best_lex = process.extractOne(i, word2vec_words.vocab)
            word = best_lex[0]
            # map to vector
            vector = word2vec.wv[word]
            return vector
        
        X = np.zeros((df.shape[0],12))
        for i in range(X.shape[0]):
            if i % 100 == 0:
                print("%d/%d" % (i,X.shape[0]))
            X[i,:] = map_word(df.word.iloc[i])
        #else:
        #    print(i)
        #df['vector'] = df.word.apply(map_word)
        
        print(df)     
        with open(model_path, 'wb') as model_file:
            pickle.dump(word2vec, model_file)
    else:
        with open(model_path, 'rb') as model_file:
            word2vec = pickle.load(model_file)
    return word2vec
def vocab_size():
    """"""
    # load model
    word2vec = get_model()
    # vocabulary size
    return len(word2vec.wv.vocab)

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

if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)
    # retrain model
    word2vec = get_model(retrain = True)
    # create plots
    #plot_length_ratio(seq_len = None)
    #plot_length_ratio(seq_len = 1200)
    
    train_x,train_y = train_data(seq_len = 1200)
    print("Train data |%d x %d|" % (train_x.shape))
    print("Labels |%d x 1|" % (train_y.shape))

