# -*- coding: utf-8 -*-
"""
Implements the operations with the GAN model: initialization, loading, saving.

@author: Martin Benes.
"""

import logging
import pickle
import sys
import torch
sys.path.append('src')

import config
import embeddings
from generator import *
from discriminator import *

class empty:
    """Initializes empty models to be trained."""
    @staticmethod
    def ScalarIncremental():
        """Initializes empty ScalarIncremental model."""
        # Generator
        G = Generator(
            input_size = 1,
            output_size = 1,
            seq_len = config.seq_len,
            hidden_size = 5,
            num_layers = 2,
            dropout_prob = config.dropout_prob
        ).to(config.device)
        # Discriminator
        D = Discriminator(
            input_size = 1,
            seq_len = config.seq_len,
            hidden_size = 5,
            num_layers = 2,
            dropout_prob = config.dropout_prob
        ).to(config.device)
        # return models
        return G,D
    @staticmethod
    def ClosestWord2Vec():
        """Initializes empty Closest Word2Vec model."""
        # Generator
        G = Generator(
            input_size = config.input_size,
            output_size = config.input_size,
            seq_len = config.seq_len,
            hidden_size = 20,
            num_layers = 2,
            dropout_prob = config.dropout_prob
        ).to(config.device)
        # Discriminator
        D = Discriminator(
            input_size = config.input_size,
            seq_len = config.seq_len,
            hidden_size = 20,
            num_layers = 2,
            dropout_prob = config.dropout_prob
        ).to(config.device)
        # return models
        return G,D
    @staticmethod
    def Bert():
        """Initializes empty Bert model."""
        # Generator
        G = Generator(
            input_size = config.bert_input_size,
            output_size = config.bert_input_size,
            seq_len = config.bert_seq_len,
            hidden_size = 2,
            num_layers = 1,
            dropout_prob = config.dropout_prob
        ).to(config.device)
        # Discriminator
        D = Discriminator(
            input_size = config.bert_input_size,
            seq_len = config.bert_seq_len,
            hidden_size = 2,
            num_layers = 1,
            dropout_prob = config.dropout_prob
            ).to(config.device)
        # return models
        return G,D

def load(path):
    """Loads model from given path.
    
    Args:
        path (str): Path to read from. Can be redirected to gdrive.
    """
    # load
    if config.from_drive:
        path = 'drive/My Drive/Colab Notebooks/' + path
    with open(path, 'rb') as fp:
        model = pickle.load(fp)
    logging.info("model %s loaded", path)
    # return
    return model

def save(model, path):
    """Saves model to given path.
    
    Args:
        model (): Object to be saved.
        path (str): Path to save to. Can be redirected to gdrive.
    """
    # save
    if config.from_drive:
        path = 'drive/My Drive/Colab Notebooks/' + path
    with open(path, 'wb') as fp:
        pickle.dump(model, fp)
    logging.info("model %s written", path)

    

