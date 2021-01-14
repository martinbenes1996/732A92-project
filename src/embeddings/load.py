# -*- coding: utf-8 -*-
"""
Py Generators of data batches.

@author: Martin Benes
"""

import logging
import sys
sys.path.append('src')
import dataset
import config

from . import train
from . import models

class trainset:
    """Generators of train set."""
    @staticmethod
    @train.LoadData(words_tr = dataset._read_train_words)
    def ScalarIncremental(words_tr):
        """ScalarIncremental trainset."""
        for i in range(0,words_tr.shape[0],config.batch_size):
            batch = words_tr.iloc[i:i + config.batch_size]
            sentence_matrix = models.ScalarIncremental(batch.text)
            yield sentence_matrix
        else:
            if i < config.batch_size-1:
                batch = words_tr.iloc[i:words_tr.shape[0]]
                sentence_matrix = models.ScalarIncremental(batch.text)
                yield sentence_matrix
                
    @staticmethod
    @train.LoadData(words_tr = dataset._read_train_words)
    def ClosestWord2Vec(words_tr):
        """ClosestWord2Vec trainset."""
        for i in range(0,words_tr.shape[0],config.batch_size):
            batch = words_tr.iloc[i:i + config.batch_size]
            sentence_matrix = models.ClosestWord2Vec(batch.text)
            yield sentence_matrix
        else:
            if i < config.batch_size-1:
                batch = words_tr.iloc[i:words_tr.shape[0]]
                sentence_matrix = models.ClosestWord2Vec(batch.text)
                yield sentence_matrix
                
    @staticmethod
    @train.LoadData(sentences_tr = dataset._read_train_sentences)
    def Bert(sentences_tr):
        """Bert trainset."""
        for i in range(0,sentences_tr.shape[0],config.batch_size):
            batch = sentences_tr.iloc[i:i + config.batch_size]
            sentence_matrix = models.Bert(batch.text)
            yield sentence_matrix
        else:
            if i < config.batch_size-1:
                batch = sentences_tr.iloc[i:sentences_tr.shape[0]]
                sentence_matrix = models.Bert(batch.text)
                yield sentence_matrix

class testset:
    """Generators of test set."""
    @staticmethod
    @train.LoadData(words_te = dataset._read_test_words)
    def ScalarIncremental(words_te):
        """ScalarIncremental testset."""
        for i in range(0,words_te.shape[0],config.batch_size):
            batch = words_te.iloc[i:i + config.batch_size]
            sentence_matrix = models.ScalarIncremental(batch.text)
            yield sentence_matrix
        else:
            if i < config.batch_size-1:
                batch = words_te.iloc[i:words_te.shape[0]]
                sentence_matrix = models.ScalarIncremental(batch.text)
                yield sentence_matrix
                
    @staticmethod
    @train.LoadData(words_te = dataset._read_test_words)
    def ClosestWord2Vec(words_te):
        """ClosestWord2Vec testset."""
        for i in range(0,words_te.shape[0],config.batch_size):
            batch = words_te.iloc[i:i + config.batch_size]
            sentence_matrix = models.ClosestWord2Vec(batch.text)
            yield sentence_matrix
        else:
            if i < config.batch_size-1:
                batch = words_te.iloc[i:words_te.shape[0]]
                sentence_matrix = models.ClosestWord2Vec(batch.text)
                yield sentence_matrix
                
    @staticmethod
    @train.LoadData(sentences_te = dataset._read_test_sentences)
    def Bert(sentences_te):
        """Bert testset."""
        for i in range(0,sentences_te.shape[0],config.batch_size):
            batch = sentences_te.iloc[i:i + config.batch_size]
            sentence_matrix = models.Bert(batch.text)
            yield sentence_matrix
        else:
            if i < config.batch_size-1:
                batch = sentences_te.iloc[i:sentences_te.shape[0]]
                sentence_matrix = models.Bert(batch.text)
                yield sentence_matrix

class markov:
    """Generators of Markov dataset."""
    @staticmethod
    @train.LoadData(markov = dataset.markov)
    def ScalarIncremental(markov = None):
        """ScalarIncremental Markov."""
        markov = markov if markov is not None else dataset.markov()
        for i in range(0,markov.shape[0],config.batch_size):
            batch = markov.iloc[i:i + config.batch_size]
            batch.text = batch.text.apply(lambda s: s[:min(1200,len(s))])
            sentence_matrix = models.ScalarIncremental(batch.text)
            yield sentence_matrix
        else:
            if i < config.batch_size-1:
                batch = markov.iloc[i:markov.shape[0]]
                batch.text = batch.text.apply(lambda s: s[:min(1200,len(s))])
                sentence_matrix = models.ScalarIncremental(batch.text)
                yield sentence_matrix
    @staticmethod
    @train.LoadData(markov = dataset.markov)
    def ClosestWord2Vec(markov = None):
        """ClosestWord2Vec Markov."""
        markov = markov if markov is not None else dataset.markov()
        for i in range(0,markov.shape[0],config.batch_size):
            batch = markov.iloc[i:i + config.batch_size]
            batch.text = batch.text.apply(lambda s: s[:min(1200,len(s))])
            sentence_matrix = models.ClosestWord2Vec(batch.text)
            yield sentence_matrix
        else:
            if i < config.batch_size-1:
                batch = markov.iloc[i:markov.shape[0]]
                batch.text = batch.text.apply(lambda s: s[:min(1200,len(s))])
                sentence_matrix = models.ClosestWord2Vec(batch.text)
                yield sentence_matrix
    @staticmethod
    @train.LoadData(markov = dataset.markov)
    def Bert(markov = None):
        """Bert Markov."""
        markov = dataset.markov(tokenize = False)
        for i in range(0,markov.shape[0],config.batch_size):
            batch = markov.iloc[i:i + config.batch_size]
            batch.text = batch.text.apply(lambda s: s[:min(1200,len(s))])
            sentence_matrix = models.Bert(batch.text)
            yield sentence_matrix
        else:
            if i < config.batch_size-1:
                batch = markov.iloc[i:markov.shape[0]]
                batch.text = batch.text.apply(lambda s: s[:min(1200,len(s))])
                sentence_matrix = models.Bert(batch.text)
                yield sentence_matrix

class generator:
    """Generators of Generator output."""
    @staticmethod
    def ScalarIncremental(model_g, N = 1):
        """ScalarIncremental Generator output.
        
        Args:
            model_g (): Generator model.
            N (int): Number of samples to generate.
        """
        for i in range(0,N,config.batch_size):
            yield model_g.generate_input(min(config.batch_size,N))
        else:
            yield model_g.generate_input(N - i)
    @staticmethod
    def ClosestWord2Vec(model_g, N = 1):
        """ClosestWord2Vec Generator output.
        
        Args:
            model_g (): Generator model.
            N (int): Number of samples to generate.
        """
        for i in range(0,N,config.batch_size):
            yield model_g.generate_input(min(config.batch_size,N))
        else:
            yield model_g.generate_input(N - i)
    @staticmethod
    def Bert(model_g, N = 1):
        """Bert Generator output.
        
        Args:
            model_g (): Generator model.
            N (int): Number of samples to generate.
        """
        for i in range(0,N,config.batch_size):
            yield model_g.generate_input(min(config.batch_size,N))
        else:
            yield model_g.generate_input(N - i)