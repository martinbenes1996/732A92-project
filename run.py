# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 15:12:24 2021

@author: martin
"""

import sys
sys.path.append('src')
import logging
logging.basicConfig(level = logging.INFO)

import config
import dataset
import embeddings
import evaluate
import gan
import markov

# read data
words = dataset._read_words()

# generate words with Markov chain
predictions = markov.generate_sentences(words = words, N = 1000, threshold = 0)
predictions.to_csv('output/markov.txt', index = False)

_ = embeddings.train.ScalarIncremental()
_ = embeddings.train.ClosestWord2Vec()

# generate Scalar Incremental embedding
emb = embeddings.ScalarIncremental(words.text)

# map predictions to ids
#def map_sentence(sentence):
#    return [word2vec[word.lower()] for word in sentence]
#predictions.apply(map_sentence)

# train Bert
import train
config.set_from_drive(False)
config.batch_size = 1
model = train.Bert()

model = gan.load('models/nn/Bert.model')
evaluate.error_plot(model)

model = gan.load('models/nn/ClosestWord2Vec.model')
sentences = evaluate.generate.ClosestWord2Vec(model, N = 10)
sentences.to_csv('output/word2vec.txt', index = False)
#evaluate.error_plot(model)


train_score, test_score, markov_score = evaluate.confusion.Bert()
