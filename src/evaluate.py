# -*- coding: utf-8 -*-
"""
Evaluate functions for trained GAN: loss plot, performance and generating.

@author: martin
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.append('src')
import config
import dataset
import embeddings
import gan

def error_plot(model, output = None, log = False):
    """Plots losses of the model training.
    
    Args:
        model (): Trained model
        output (str): Path to save the plot. 
        log (bool): Show log-losses or losses?
    """
    losses_g,losses_d = model['losses_g'],model['losses_d']

    # xgrid
    lsp = np.linspace(0,len(losses_g)-1,num=len(losses_g))
    N = len(losses_g) / config.num_epochs
    xgrid = lsp / N
    # log-error
    if log:
        plt.plot(xgrid, np.log(losses_g), c = 'r')
        plt.plot(xgrid, np.log(losses_d), c = 'g')
    # error
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

class confusion:
    """Performance of discriminator predicting."""
    @staticmethod
    def _perform(model, dataloader, dataname):
        """The prediction implementation."""
        prediction_score = []
        logging.info("%s data scoring", dataname)
        for i,batch in enumerate(dataloader):
            logging.info("%s batch %d", dataname, i)
            score,_ = model(batch)
            for s in score[:,1]:
                prediction_score.append(float(s))
        return pd.Series(prediction_score)
    @staticmethod
    def _mse(score):
        """MSE of the score tensor given."""
        return (score**2).mean()
    @staticmethod
    def _miss(vector):
        """Missclassification of the vector given."""
        return 1 - vector.mean()
    @classmethod
    def _score(cls, score):
        """Compute MSE and missclassification of the score given.
        
        Args:
            score (torch.Tensor): Score to measure.
        """
        return {'mse': cls._mse(score), 'miss': cls._miss(score < .5)}
    @classmethod
    def ScalarIncremental(cls, train = True, test = True, markov = True, N = 5000):
        """Measures ScalarIncremental Discriminator performance.
        
        Args:
            train (bool): Use train data.
            test (bool): Use test data.
            markov (bool): Use Markov data.
            N (int): Number of Generator outputs.
        """
        # load model
        model = gan.load("models/nn/ScalarIncremental.model")
        model_d = model['discriminator']\
            .to(config.device)\
            .eval()
        state_d = [m.to(config.device) for m in model['state_d']]
        model_dis = lambda d: model_d(d, state_d)
        score = {}
        
        # train data
        if train:
            dataloader = embeddings.load.trainset.ScalarIncremental()
            train_score = cls._perform(model_dis, dataloader, "train")
            score['train'] = cls._score(1 - train_score)
        # test data
        if test:
            dataloader = embeddings.load.testset.ScalarIncremental()
            test_score = cls._perform(model_dis, dataloader, "test")
            score['test'] = cls._score(1 - test_score)
        # markov data
        if markov:
            dataloader = embeddings.load.markov.ScalarIncremental()
            markov_score = cls._perform(model_dis, dataloader, "markov")
            score['markov'] = cls._score(markov_score)
        # generated data
        if N > 0:
            model_g = model['generator']\
                .to(config.device)\
                .eval()
            state_g = [m.to(config.device) for m in model['state_g']]
            def model_gen(d):
                pred_g,_ = model_g(d,state_g)
                return model_dis(pred_g)
            dataloader = embeddings.load.generator.ScalarIncremental(model_g, N = N)
            generator_score = cls._perform(model_gen, dataloader, "generator")
            score['generator'] = cls._score(generator_score)
        return score
    @classmethod
    def ClosestWord2Vec(cls, train = True, test = True, markov = True, N = 5000):
        """Measures ClosestWord2Vec Discriminator performance.
        
        Args:
            train (bool): Use train data.
            test (bool): Use test data.
            markov (bool): Use Markov data.
            N (int): Number of Generator outputs.
        """
        # load model
        model = gan.load("models/nn/ClosestWord2Vec.model")
        model_d = model['discriminator']\
            .eval()\
            .to(config.device)
        state_d = [m.to(config.device) for m in model['state_d']]
        model_dis = lambda d: model_d(d, state_d)
        score = {}
        
        # train data
        if train:
            dataloader = embeddings.load.trainset.ClosestWord2Vec()
            train_score = cls._perform(model_dis, dataloader, "train")
            score['train'] = cls._score(1 - train_score)
        # test data
        if test:
            dataloader = embeddings.load.testset.ClosestWord2Vec()
            test_score = cls._perform(model_dis, dataloader, "test")
            score['test'] = cls._score(1 - test_score)
        # markov data
        if markov:
            dataloader = embeddings.load.markov.ClosestWord2Vec()
            markov_score = cls._perform(model_dis, dataloader, "markov")
            score['markov'] = cls._score(markov_score)
        # generated data
        if N > 0:
            model_g = model['generator']\
                .eval()\
                .to(config.device)
            state_g = [m.to(config.device) for m in model['state_g']]
            def model_gen(d):
                pred_g,_ = model_g(d,state_g)
                return model_dis(pred_g)
            dataloader = embeddings.load.generator.ClosestWord2Vec(model_g, N = N)
            generator_score = cls._perform(model_gen, dataloader, "generator")
            score['generator'] = cls._score(generator_score)
        return score
    @classmethod
    def Bert(cls, train = True, test = True, markov = True, N = 5000):
        """Measures Bert Discriminator performance.
        
        Args:
            train (bool): Use train data.
            test (bool): Use test data.
            markov (bool): Use Markov data.
            N (int): Number of Generator outputs.
        """
        # load model
        model = gan.load("models/nn/Bert.model")
        model_d = model['discriminator']\
            .eval()\
            .to(config.device)
        state_d = [m.to(config.device) for m in model['state_d']]
        model_dis = lambda d: model_d(d, state_d)
        score = {}
        
        # train data
        if train:
            dataloader = embeddings.load.trainset.Bert()
            train_score = cls._perform(model_dis, dataloader, "train")
            score['train'] = cls._score(1 - train_score)
        # test data
        if test:
            dataloader = embeddings.load.testset.Bert()
            test_score = cls._perform(model_dis, dataloader, "test")
            score['test'] = cls._score(1 - test_score)
        # markov data
        if markov:
            dataloader = embeddings.load.markov.Bert()
            markov_score = cls._perform(model_dis, dataloader, "markov")
            score['markov'] = cls._score(markov_score)
        # generated data
        if N > 0:
            model_g = model['generator']\
                .eval()\
                .to(config.device)
            state_g = [m.to(config.device) for m in model['state_g']]
            def model_gen(d):
                pred_g,_ = model_g(d,state_g)
                return model_dis(pred_g)
            dataloader = embeddings.load.generator.Bert(model_g, N = N)
            generator_score = cls._perform(model_gen, dataloader, "generator")
            score['generator'] = cls._score(generator_score)
        return score
    
class generate:
    """"""
    @staticmethod
    def ScalarIncremental(N = 1):
        """Generates text from ScalarIncremental.
        
        Args:
            N (int): Number of samples.
        """
        # load model
        model = gan.load("models/nn/ScalarIncremental.model")
        model_g = model['generator'].eval();
        state_g = model['state_g']
        # predict
        input_g = model_g.generate_input(N)
        pred_g,state_g = model_g(input_g.detach(), state_g)
        # reverse mapping
        sentences = []
        for sentence_idx in range(N):
            words = []
            #print(sentence_idx)
            for word_idx in range(config.seq_len):
                # get word
                word_code = pred_g[sentence_idx, word_idx, 0].round()
                word = embeddings.rev.ScalarIncremental(int(word_code))
                if word is not None:
                    words.append(word)
            sentences.append(' '.join(words))
        return pd.Series(sentences)
        
    @staticmethod
    def ClosestWord2Vec(N = 1):
        """Generates text from ClosestWord2Vec.
        
        Args:
            N (int): Number of samples.
        """
        # load model
        model = gan.load("models/nn/ClosestWord2Vec.model")
        model_g = model['generator'].eval();
        state_g = model['state_g']
        # predict
        input_g = model_g.generate_input(N)
        pred_g,state_g = model_g(input_g.detach(), state_g)
        # reverse mapping
        sentences = []
        for sentence_idx in range(N):
            logging.info("sentence %d", sentence_idx)
            words = []
            for word_idx in range(config.seq_len):
                # get word vector
                word_vector = pred_g[sentence_idx, word_idx]
                # map to word
                word = embeddings.rev.ClosestWord2Vec(word_vector)
                # add word to sentence
                #print('  ',word_code, word)
                if word is not None:
                    words.append(word)
            sentences.append(' '.join(words))
        return pd.Series(sentences)
    @staticmethod
    def Bert(N = 1):
        """Generates text from Bert.
        
        Args:
            N (int): Number of samples.
        """
        # load model
        model = gan.load("models/nn/Bert.model")
        model_g = model['generator']\
            .eval()\
            .to(config.device)
        state_g = [m.to(config.device) for m in model['state_g']]
        # predict
        input_g = model_g.generate_input(N)
        pred_g,state_g = model_g(input_g.detach(), state_g)
        # reverse mapping
        sentences = []
        for sentence_idx in range(N):
            logging.info("sentence %d", sentence_idx)
            words = []
            for word_idx in range(config.bert_seq_len):
                # get word vector
                word_vector = pred_g[sentence_idx, word_idx]
                # map to word
                word = embeddings.rev.Bert(word_vector)
                # add word to sentence
                if word is not None:
                    words.append(word)
            sentences.append(' '.join(words))
        return pd.Series(sentences)
        