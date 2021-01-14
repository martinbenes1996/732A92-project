# -*- coding: utf-8 -*-
"""
Implementation of reversed embeddings.

@author: martin
"""

import sys
import torch
sys.path.append('src')
import config
import dataset
import gan

from . import train
from . import models

_scalar_incremental = None
@models.RequiresModel(train.ScalarIncremental)
def ScalarIncremental(word_code, model = None):
    """Reversed ScalarIncremental embedding.
    
    Args:
        word_code (int): Word_code to convert to word.
    """
    # reverse model
    global _scalar_incremental
    if _scalar_incremental is None:
        _scalar_incremental = {v:k for k,v in model.items()}
    # get word
    word = _scalar_incremental.get(word_code, None)
    return word
     
@models.RequiresModel(train.ClosestWord2Vec)
def ClosestWord2Vec(word_code, model = None):
    """Reversed ClosestWord2Vec embedding.
    
    Args:
        word_code (torch.tensor): Tensor to convert to word.
    """
    vectors,words = model
    vectors_lengths = (vectors**2).sum(axis = 1).sqrt() * (word_code**2).sum().sqrt()
    distances = (vectors @ word_code) / vectors_lengths
    return words[int(distances.argmax())]

def train_bert_vocab(sentences):
    """Trainer for Bert vocabulary.
    
    Args:
        sentences (): Train data.
    """
    bert_vectors,bert_words = models.BertVocabulary(sentences_tr.text)
    bert_vectors = bert_vectors\
        .to(config.device)
    gan.save({'vectors': bert_vectors, 'words': bert_words},
             'models/BertVocabulary.model')
    return bert_vectors,bert_words
def load_bert_vocab():
    """Loads Bert vocabulary from file."""
    try:
        bert_vocab = gan.load('models/BertVocabulary.model')
        return bert_vocab['vectors'], bert_vocab['words']
    except:
        return None
    
_bert = None
@models.RequiresModel(train.Bert)
@train.LoadData(sentences_tr = dataset._read_train_sentences)
def Bert(word_code, sentences_tr = None, model = None):
    """Reversed Bert embedding.
    
    Args:
        word_code (torch.tensor): Tensor to convert to word.
    """
    # load vocabulary
    global _bert
    _bert = load_bert_vocab()
    if _bert is None:
        _bert = train_bert_vocab(sentences_tr.text)
    # transform
    vectors,words = _bert
    vectors_lengths = (vectors**2).sum(axis = 1).sqrt() * (word_code**2).sum().sqrt()
    distances = (vectors @ word_code) / vectors_lengths
    return words[int(distances.argmax())]
    
    