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
import markov

# read data
words = dataset._read_words()

# generate words with Markov chain
predictions = markov.generate_sentences(words = words, N = 1000, threshold = 0)
predictions.to_csv('output/markov.txt', index = False)

_ = embeddings.train.ScalarIncremental()
_ = embeddings.train.ClosestWord2Vec()w

# generate Scalar Incremental embedding
emb = embeddings.ScalarIncremental(words.text)

# map predictions to ids
#def map_sentence(sentence):
#    return [word2vec[word.lower()] for word in sentence]
#predictions.apply(map_sentence)


# test working
import gan
model_g,model_d = gan.ClosestWord2Vec()
print(model_g)
print(model_d)
state_g = model_g.init_state()
state_d = model_d.init_state()
logging.info("State G[0]: %s [%s]", state_g[0][:3,0,0], state_g[0].shape)
logging.info("State G[1]: %s [%s]", state_g[1][:3,0,0], state_g[1].shape)
logging.info("State D[0]: %s [%s]", state_d[0][:3,0,0], state_d[0].shape)
logging.info("State D[1]: %s [%s]", state_d[1][:3,0,0], state_d[1].shape)
input_matrix_g = model_g.generate_input(config.batch_size)
logging.info("Generator input: %s [%s]", str(input_matrix_g[:3,0,0]), str(input_matrix_g.shape))
y_g,_ = model_g(input_matrix_g, state_g)
y_d,_ = model_d(y_g, state_d)
logging.info("generator output: %s [%s]", y_g[0,:2,0], y_g.shape)
logging.info("discriminator output: %s [%s]", y_d[0,:2], y_d.shape)
