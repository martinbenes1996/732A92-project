# -*- coding: utf-8 -*-
"""
The Markov Chain generator.

@author: Martin Benes
"""

import logging
import math
import numpy as np
import pandas as pd
import pickle
import sys
sys.path.append('src')

import dataset

class MarkovChain:
    """Markov Chain implementation.
    
    Attributes:
        _data (dict): Object fit.
        _words (dict): Word vocabulary.
    """
    def __init__(self):
        """Constructor"""
        self._data = {}
        self._words = {}
        self._lens = np.array([])
        self._vocab = []
    @staticmethod
    def _word_id(w):
        """Constructs word to word id.
        
        Args:
            w (str): Word.
        """
        if w is not None:
            return w.lower()
        else:
            return None
    def _add_word(self, current_word, next_word = None):
        """Adds current_word - next_word to the object.
        
        Args:
            current_word (str): Current word.
            next_word (str): Next word.
        """
        # get word ids
        current_word_id = self._word_id(current_word)
        next_word_id = self._word_id(next_word)
        # new current word
        if current_word_id not in self._data:
            self._data[current_word_id] = {'next': {}}
        if next_word is None:
            return
        # new next word
        if next_word_id not in self._data[current_word_id]['next']:
            self._data[current_word_id]['next'][next_word_id] = 1
        # increment
        else:
            self._data[current_word_id]['next'][next_word_id] += 1
        # add current word to vocabulary
        self._add_word_to_vocabulary(current_word_id, current_word)
        
    def _add_word_to_vocabulary(self, word_id, word):
        """Adds word to the vocabulary.
        
        Args:
            word_id (str): Word id.
            word (str): Word.
        """
        # add word to vocabulary
        if word_id not in self._words:
            self._words[word_id] = {}
        if word not in self._words[word_id]:
            self._words[word_id][word] = 1
        else:
            self._words[word_id][word] += 1
    def _fit_sentence(self, sentence):
        """Adds sentence to the object.
        
        Args:
            sentence (iterable of strings): Input sentence.
        """
        # iterate words
        for i in range(len(sentence)-1):
            current_word,next_word = sentence[i],sentence[i+1]
            # add word pair to the model
            self._add_word(current_word, next_word)
        # add last word to vocabulary
        if len(sentence) > 0:
            last_word = sentence[-1]
            last_word_id = self._word_id(last_word)
            self._add_word(last_word)
            self._add_word_to_vocabulary(last_word_id, last_word)
        
    def fit(self, X):
        """Fits the object to the training dataset.
        
        Args:
            X (iterable of sentences): Training dataset.
        """
        # lengths
        self._lens = np.array([len(i) for i in X])
        # next-word
        for sentence in X:
            self._fit_sentence(sentence)
        # vocabulary word ids
        self._vocab = list(self._data.keys())
    def _prior(self, threshold = 0):
        """Uniform prior."""
        # choose index
        idx_space = np.array(range(len(self._data)))
        idx = np.random.choice(idx_space) # Uniform() prior
        # map index to word
        word_0_id = self._vocab[idx]
        if sum(self._words[word_0_id].values()) < threshold:
            raise RuntimeError
        return word_0_id
    def _predict_next_word(self, current_word, threshold = 0):
        """Generate next word from the current word.
        
        Args:
            current_word (str): Current word.
        Returns:
            (str): Next word.
        """
        # get distribution
        word_distribution = self._data[self._word_id(current_word)]['next']
        next_words = list(word_distribution.keys())
        probs = np.array([p for p in word_distribution.values()])
        probs = probs[probs >= threshold]
        if len(probs) == 0:
            raise RuntimeError
        probs = probs / np.sum(probs) # normalize probabilities
        # generate next word
        idx_space = np.arange(len(probs))
        idx = np.random.choice(idx_space, p = probs)
        next_word = next_words[idx]
        #logging.info('next word [%d]: %s', idx, next_word)
        return next_word
        
    def predict(self, N = 1, threshold = 0):
        """Generate N sentences from the fitted model.
        
        Args:
            N (int): Number of sentences.
            threshold (int): Threshold for occurences to count.
        """
        # sentence
        fx = []
        for i in range(N):
            # len
            D = np.random.choice(self._lens)
            # simulate words
            while True:
                try: word_0 = self._prior(threshold = threshold)
                except: pass
                else: break    
            
            sentence = [word_0]
            for d in range(D - 1):
                # next word
                try:
                    next_word_id = self._predict_next_word(sentence[-1])
                except:
                    break
                next_word = max(self._words[next_word_id], key=self._words[next_word_id].get)
                # append
                sentence.append(next_word)
            if i % 100:
                logging.info('generated sentence %d of length %d', i, len(sentence))
            fx.append(sentence)
        return pd.Series(fx)
    
    @classmethod
    def generate_sentences(cls, words = None, N = 1, threshold = 0):
        """Trains the model and generates N sentences.
        
        Args:
            words (pd.Series): Data input.
            N (int): Number of sentences.
            threshold (int): Minimum occurrences to use.
        """
        # read data
        words = dataset._read_words() if words is None else words
        # train model
        model = cls()
        logging.info('training %s model', __class__.__name__)
        model.fit(words.text)
        # predict
        logging.info('generating %d predictions', N)
        words = model.predict(N = N, threshold = 0)
        sentences = words.apply(lambda l: ' '.join(l))
        return sentences
    
    def _likelihood(self, current_word, next_word, eps = 1e-12):
        """"""
        # get ids
        current_word_id = self._word_id(current_word)
        next_word_id = self._word_id(next_word)
        # get count
        if current_word_id not in self._data:
            return eps
        bigram_count = self._data[current_word_id]['next'].get(next_word_id, 0)
        if bigram_count == 0:
            return eps
        bigram_prob = bigram_count / sum(self._data[current_word_id]['next'].values())
        #print((current_word,next_word), bigram_prob)
        return bigram_prob

    @classmethod
    def _perplexity(cls, test_words = None):
        """Trains Markov Chain and computes perplexity on test data."""
        # try loading
        try:
            logging.info("loading model")
            fp = open('models/train_mc.pickle','rb')
            model = pickle.load(fp)
            fp.close()
        # train and save
        except:
            logging.info('model not found, retraining')
            # read data
            train_words = dataset._read_train_words()\
                .reset_index(drop=True)\
                .text
            # train model
            model = cls()
            logging.info('training %s model', __class__.__name__)
            model.fit(train_words)
            # save
            with open('models/train_mc.pickle','wb') as fp:
                pickle.dump(model, fp)
        # load test data
        else:
            if test_words is None:
                test_words = dataset._read_test_words()\
                    .reset_index(drop=True)\
                    .text
        # iterate sentences
        N = 0
        llik = 0
        for test_word in test_words:
            N += len(test_word) - 1
            for j in range(len(test_word)-1):
                current_word,next_word = test_word[j],test_word[j+1]
                # get log likelihood of the bigram
                lik = model._likelihood(current_word, next_word)
                llik += math.log(lik)
        perplexity = math.exp(-1/N * llik)
        return perplexity
def generate_sentences(words = None, N = 1, threshold = 0):
    """Trains the model and generates N sentences.
        
    Args:
        words (pd.Series): Data input.
        N (int): Number of sentences.
        threshold (int): Minimum occurrences to use.
    """
    # distribute to the class
    return MarkovChain.generate_sentences(words = words, N = N, threshold = 0)

def test_perplexity():
    """Trains Markov Chain and computes perplexity on test data."""
    # distribute to the class
    test_words = dataset._read_test_words()
    #print(test_words)
    return MarkovChain._perplexity(test_words)
def markov_perplexity():
    """Trains Markov Chain and computes perplexity on test data."""
    # distribute to the class
    markov_words = dataset.markov()
    #print(markov_words)
    return MarkovChain._perplexity(markov_words)
def word2vec_perplexity():
    """Trains Markov Chain and computes perplexity on test data."""
    # distribute to the class
    word2vec_sentences = pd.read_csv('output/word2vec.txt', header=0)['0']
    word2vec_words = word2vec_sentences.apply(lambda sentence: [word for word in sentence.split(' ')])
    #print(word2vec_words)
    return MarkovChain._perplexity(word2vec_words)
def bert_perplexity():
    """Trains Markov Chain and computes perplexity on test data."""
    # distribute to the class
    bert_sentences = pd.read_csv('output/bert.txt', header=0)['0']
    bert_words = bert_sentences.apply(lambda sentence: [word for word in sentence.split(' ')])
    #print(bert_words)
    return MarkovChain._perplexity(bert_words)


test_ppx = test_perplexity()
print("Test:", test_ppx)
markov_ppx = markov_perplexity()
print("Markov:", markov_ppx)
word2vec_ppx = word2vec_perplexity()
print("Word2Vec:", word2vec_ppx)
bert_ppx = bert_perplexity()
print("Bert:", bert_ppx)