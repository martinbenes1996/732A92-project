# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 17:43:41 2021

@author: martin
"""

import sys
sys.path.append('src')

import config
import embeddings
from generator import *
from discriminator import *

def ScalarIncremental():
    """"""
    # Generator
    G = Generator(
        input_size = 1,
        output_size = 1,
        seq_len = config.seq_len,
        hidden_size = 5,
        num_layers = 2,
        dropout_prob = .5
    ).to(config.device)
    # Discriminator
    D = Discriminator(
        input_size = 1,
        seq_len = config.seq_len,
        hidden_size = 5,
        num_layers = 2,
        dropout_prob = .5
    ).to(config.device)
    # return models
    return G,D

def ClosestWord2Vec():
    """"""
    # Generator
    G = Generator(
        input_size = config.input_size,
        output_size = config.input_size,
        seq_len = config.seq_len,
        hidden_size = 20,
        num_layers = 2,
        dropout_prob = .5
    ).to(config.device)
    # Discriminator
    D = Discriminator(
        input_size = config.input_size,
        seq_len = config.seq_len,
        hidden_size = 20,
        num_layers = 2,
        dropout_prob = .5
    ).to(config.device)
    # return models
    return G,D

def Bert():
    """"""
    # Generator
    G = Generator(
        input_size = config.bert_input_size,
        output_size = config.bert_input_size,
        seq_len = config.bert_seq_len,
        hidden_size = 5,
        num_layers = 2,
        dropout_prob = .5
    ).to(config.device)
    # Discriminator
    D = Discriminator(
        input_size = config.bert_input_size,
        seq_len = config.bert_seq_len,
        hidden_size = 5,
        num_layers = 2,
        dropout_prob = .5
    ).to(config.device)
    # return models
    return G,D
    
