# -*- coding: utf-8 -*-
"""
Performs embedding transformations.

@author: martin
"""

import logging
import sys
from gensim import models
from rapidfuzz import fuzz, process
import pandas as pd
import pickle
import torch
from transformers import BertTokenizer, BertModel

logging.basicConfig(level = logging.INFO)
sys.path.append('src')
import config
import dataset


_model_cache = {}
def CachedModel(path, only_runtime = False):
    def ModelLoader(decoratedF):
        model = None
        def FReplacer(*args, retrain = False, **kw):
            nonlocal model
            global _model_cache
            if not retrain and path in _model_cache:
                logging.info("model %s loaded from cache" % (path))
                return _model_cache[path]
            if not only_runtime and not retrain and model is None:
                try:
                    model_file = open(path, 'rb')
                    model = pickle.load(model_file)
                    logging.info("model %s loaded from file" % (path))
                    model_file.close()
                    retrain = False
                except:
                    logging.warning("model %s not found" % (path))
                    retrain = True
            if retrain or model is None:
                logging.info("retraining model %s" % (path))
                model = decoratedF(*args, **kw)
                if not only_runtime:
                    with open(path, 'wb') as model_file:
                        pickle.dump(model, model_file)
                    logging.info("model %s written" % (path))
            _model_cache[path] = model
            return model
        return FReplacer
    return ModelLoader

_data_cache = {}
def LoadData(**data):
    def DataLoader(decoratedF):
        _data = None
        def FReplacer(*args, from_drive = False, **kw):
            nonlocal _data,data
            global _data_cache
            if _data is None:
                _data = {}
                for k,v in data.items():
                    if k in _data_cache:
                        _data[k] = _data_cache[k]
                    else:
                        df = v(from_drive = from_drive)
                        _data[k] = df
                        _data_cache[k] = df
            return decoratedF(*args, **_data, **kw)
        return FReplacer
    return DataLoader

@CachedModel(path = 'models/incremental.model')
@LoadData(words = dataset._read_words)
def ScalarIncremental(words, *args, **kw):
    logging.warning("training ScalarIncremental")
    # collect vocabulary
    model,i = {},1
    for sentence in words.text:
        for word in sentence:
            if word.lower() not in model:
                model[word.lower()] = i
                i += 1
    return model

@CachedModel(path = 'models/word2vec.model')
@LoadData(words = dataset._read_words, scalar_model = ScalarIncremental)
def Word2Vec(words, scalar_model, *args, **kw):
    logging.warning("training Word2Vec")
    # build model
    word2vec = gensim.models.Word2Vec(sentences=words.text,
                                      size=config.input_size,
                                      workers=4, window=5, iter=10)
    return word2vec

@CachedModel(path = 'models/closest_word2vec.model')
@LoadData(words = dataset._read_words,
          word2vec = Word2Vec,
          scalar_model = ScalarIncremental)
def ClosestWord2Vec(words, word2vec, scalar_model):
    logging.warning("training ClosestWord2Vec")
    # build vocabulary
    vocabulary = list(word2vec.wv.vocab.keys())
    # words
    vectors = torch.zeros([len(scalar_model),12])
    keys = []
    for word_idx,word in enumerate(scalar_model.keys()):
        # closest match
        matched_word = process.extractOne(word, vocabulary)[0]
        vector = torch.from_numpy(word2vec.wv[matched_word])
        # store 
        vectors[word_idx,:] = vector
        keys.append(word)
        if word_idx % 10000 == 0:
            logging.info("Word %d/%d" % (word_idx, len(scalar_model)))
    return (vectors, keys)  

@CachedModel(path = 'models/bert-base-uncased', only_runtime = True)
def Bert():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('./models/bert-base-uncased/')
    
    return (tokenizer,model)
            


__all__ = ["ScalarIncremental","Word2Vec","ClosestWord2Vec"]