# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 00:31:27 2021

@author: martin
"""

import logging

import sys
sys.path.append('src')
import dataset
import config

from . import train
from . import models

@train.LoadData(words = dataset._read_words)
def ScalarIncremental(words, from_drive = False):
    for i in range(0,words.shape[0],config.batch_size):
        batch = words.iloc[i:i + config.batch_size]
        sentence_matrix = models.ScalarIncremental(batch.text, from_drive=from_drive)
        yield sentence_matrix
    else:
        batch = words.iloc[i:words.shape[0]]
        print(list(range(i,words.shape[0])))
        sentence_matrix = models.ScalarIncremental(batch.text)
        print(sentence_matrix.shape)
        yield sentence_matrix

@train.LoadData(words = dataset._read_words)
def ClosestWord2Vec(words, from_drive = False):
    for i in range(0,words.shape[0],config.batch_size):
        batch = words.iloc[i:i + config.batch_size]
        sentence_matrix = models.ClosestWord2Vec(batch.text, from_drive=from_drive)
        yield sentence_matrix
    else:
        if i < config.batch_size-1:
            batch = words.iloc[i:words.shape[0]]
            sentence_matrix = models.ClosestWord2Vec(batch.text)
            yield sentence_matrix
        
@train.LoadData(sentences = dataset._read_sentences)
def Bert(sentences, from_drive = False):
    for i in range(0,sentences.shape[0],config.batch_size):
        logging.info("loading Bert batch - data %d/%d", i, sentences.shape[0])
        batch = sentences.iloc[i:i + config.batch_size]
        sentence_matrix = models.Bert(batch.text, from_drive=from_drive)
        yield sentence_matrix
    else:
        batch = sentences.iloc[i:sentences.shape[0]]
        print(list(range(i,sentences.shape[0])))
        sentence_matrix = models.Bert(batch.text)
        print(sentence_matrix.shape)
        yield sentence_matrix

