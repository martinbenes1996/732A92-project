# -*- coding: utf-8 -*-
"""
Performs embedding transformations.

@author: martin
"""

import logging
import sys
import gensim
from rapidfuzz import fuzz, process
import pandas as pd
import pickle
import torch
from transformers import BertTokenizer, BertModel

sys.path.append('src')
import config
import dataset


_model_cache = {}
def CachedModel(path, only_runtime = False):
    if config.from_drive:
        path = 'drive/MyDrive/Colab Notebooks/' + path
    def ModelLoader(decoratedF):
        model = None
        def FReplacer(*args, retrain = False, **kw):
            nonlocal model
            global _model_cache
            if not retrain and path in _model_cache:
                #logging.info("model %s loaded from cache" % (path))
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
                    try:
                        with open(path, 'wb') as model_file:
                            pickle.dump(model, model_file)
                        logging.info("model %s written", path)
                    except:
                        logging.warning('failed saving model %s, ignored', path)
            _model_cache[path] = model
            return model
        return FReplacer
    return ModelLoader

_data_cache = {}
def LoadData(**data):
    def DataLoader(decoratedF):
        _data = None
        def FReplacer(*args, **kw):
            nonlocal _data,data
            global _data_cache
            if _data is None:
                _data = {}
                for k,v in data.items():
                    if k in _data_cache:
                        _data[k] = _data_cache[k]
                    else:
                        df = v()
                        _data[k] = df
                        _data_cache[k] = df
            return decoratedF(*args, **_data, **kw)
        return FReplacer
    return DataLoader

import matplotlib.pyplot as plt
def error_plot(model, output = None, log = False):
    losses_g,losses_d = model['losses_g'],model['losses_d']

    # xgrid
    lsp = np.linspace(0,len(losses_g)-1,num=len(losses_g))
    N = len(losses_g) / len(trainloader)
    xgrid = lsp / N

    # error
    if log_error_plot:
        plt.plot(xgrid, np.log(losses_g), c = 'r')
        plt.plot(xgrid, np.log(losses_d), c = 'g')

    # log-error
    else:
        plt.plot(xgrid, losses_g, c = 'r')
        plt.plot(xgrid, losses_d, c = 'g')

    plt.rcParams.update({'font.size': 16})
    plt.legend(['Generator', 'Discriminator'])
    plt.xlabel('Batches/Epochs')
    plt.ylabel('Loss')
    if output is not None:
        plt.savefig(output)
    else:
        plt.show()

@CachedModel(path = 'models/incremental.model')
@LoadData(words_tr = dataset._read_train_words)
def ScalarIncremental(words_tr):
    logging.warning("training ScalarIncremental")
    # collect vocabulary
    model,i = {},1
    for sentence in words_tr.text:
        for word in sentence:
            if word.lower() not in model:
                model[word.lower()] = i
                i += 1
    return model

@CachedModel(path = 'models/word2vec.model')
@LoadData(words_tr = dataset._read_train_words, scalar_model = ScalarIncremental)
def Word2Vec(words_tr, scalar_model):
    logging.warning("training Word2Vec")
    # build model
    word2vec = gensim.models.Word2Vec(sentences=words_tr.text,
                                      size=config.input_size,
                                       workers=4, window=5, iter=10)
    return word2vec

@CachedModel(path = 'models/closest_word2vec.model')
@LoadData(words_tr = dataset._read_train_words,
          word2vec = Word2Vec,
          scalar_model = ScalarIncremental)
def ClosestWord2Vec(words_tr, word2vec, scalar_model):
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
@LoadData(sentences_tr = dataset._read_train_sentences)
def Bert(sentences_tr):
    logging.warning("training Bert")
    # path
    path = './models/bert-base-uncased/'
    if config.from_drive:
        path = './drive/My Drive/Colab Notebooks/models/bert-base-uncased/'
    # load model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained(path)\
        .to(config.device)
    
    return (tokenizer,model)
            


__all__ = ["ScalarIncremental","Word2Vec","ClosestWord2Vec"]