# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 17:43:41 2021

@author: martin
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
    @staticmethod
    def ScalarIncremental():
        """"""
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
        """"""
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
        """"""
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
    """"""
    # load
    if config.from_drive:
        path = 'drive/My Drive/Colab Notebooks/' + path
    with open(path, 'rb') as fp:
        #model = torch.load(fp, map_location=config.device)
        model = pickle.load(fp)
    logging.info("model %s loaded", path)
    # return
    return model

def save(model, path):
    """"""
    # save
    if config.from_drive:
        path = 'drive/My Drive/Colab Notebooks/' + path
    with open(path, 'wb') as fp:
        pickle.dump(model, fp)
    logging.info("model %s written", path)

    

