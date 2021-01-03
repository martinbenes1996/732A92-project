# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 15:12:24 2021

@author: martin
"""

import sys
sys.path.append('src')
import logging
logging.basicConfig(level = logging.INFO)

import dataset
import markov

# read data
words = dataset._read_words()

# load model
word2vec = dataset.get_model(retrain = True, words = words)
with open('something.txt','w', encoding = 'utf8') as fp:
    for w in word2vec:
        fp.write(w + '\n')

# generate words with Markov chain
predictions = markov.generate_sentences(words = words, N = 1000, threshold = 0)
predictions.to_csv('output/markov.txt', index = False)

# map predictions to ids
def map_sentence(sentence):
    return [word2vec[word.lower()] for word in sentence]
predictions.apply(map_sentence)

    
