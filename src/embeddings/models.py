# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 23:48:56 2021

@author: martin
"""

import torch

import logging
import sys

logging.basicConfig(level = logging.INFO)
sys.path.append('src')
import config
import dataset
from . import train

def RequiresModel(loadModel):
    def ModelLoader(decoratedF):
        model = None
        def FReplacer(*args, from_drive = False, **kw):
            nonlocal model
            if model is None:
                model = loadModel(from_drive = from_drive)
            return decoratedF(*args, model = model, **kw)
        return FReplacer
    return ModelLoader

@RequiresModel(train.ScalarIncremental)
def ScalarIncremental(samples, model = None):
    """"""
    N = samples.shape[0]
    sentence_matrix = torch.zeros((N, config.seq_len))
    missing_words = 0
    for sentence_idx,sentence in enumerate(samples):
        for word_idx,word in enumerate(sentence):
            try:
                sentence_matrix[sentence_idx,word_idx] = model[word.lower()]
            except:
                sentence_matrix[sentence_idx,word_idx] = -1
                missing_words += 1
    if missing_words > 0:
        logging.warning("%d missing words", missing_words)
    sentence_matrix = sentence_matrix\
        .reshape([*sentence_matrix.shape, 1])\
        .to(config.device)
    return sentence_matrix

@RequiresModel(train.ClosestWord2Vec)
def ClosestWord2Vec(samples, model = None):
    """"""
    (vectors, keys) = model
    #print(vectors.shape)
    map_idx = {k:i for i,k in enumerate(keys)}
    N = samples.shape[0]
    sentence_matrix = torch.zeros((N, config.seq_len, config.input_size))
    missing_words = 0
    for sentence_idx,sentence in enumerate(samples):
        for word_idx,word in enumerate(sentence):
            idx = map_idx[word.lower()]
            #try:
            sentence_matrix[sentence_idx,word_idx,:] = vectors[idx,:]
            #except:
            #    sentence_matrix[sentence_idx,word_idx] = -1
            #    missing_words += 1
    if missing_words > 0:
        logging.warning("%d missing words", missing_words)
    sentence_matrix = sentence_matrix\
        .reshape([*sentence_matrix, 1])\
        .to(config.device)
    return sentence_matrix

@RequiresModel(train.Bert)
def _Bert(x, model):
    
    # initialize model
    tokenizer, model = model
    model.eval();
    
    # tokenize
    tokenized_text = tokenizer.tokenize("[CLS] " + x + " [SEP]")
    if len(tokenized_text) > config.bert_seq_len:
        tokenized_text = tokenized_text[:config.bert_seq_len - 1] + ["SEP"]
    #print(tokenized_text)
    # tokens to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    # PyTorch inputs
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    # Run the text through BERT, and collect all of the hidden states produced (12 layers). 
    with torch.no_grad(): 
        #print(model)
        outputs = model(tokens_tensor, segments_tensors)

        # Evaluating the model will return a different number of objects based on 
        # how it's  configured in the `from_pretrained` call earlier. In this case, 
        # becase we set `output_hidden_states = True`, the third item will be the 
        # hidden states from all layers. See the documentation for more details:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        hidden_states = outputs[2]

    # Concatenate the tensors for all layers. We use `stack` here to
    # create a new dimension in the tensor.
    token_embeddings = torch.stack(hidden_states, dim=0)
    #print(token_embeddings.size())
    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    #print(token_embeddings.size())
    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(1,0,2)
    #print(token_embeddings.size())
    
    # Stores the token vectors, with shape [22 x 768]
    token_vecs_sum = []
    # For each token in the sentence...
    for token in token_embeddings:
        # Sum the vectors from the last four layers.
        sum_vec = torch.sum(token[-4:], dim=0)
        # Use `sum_vec` to represent `token`.
        token_vecs_sum.append(sum_vec)
    
    new_text = []
    new_vectors = []
    for word_idx,word in enumerate(tokenized_text):
        # special tokens
        if word in {'[CLS]','[SEP]'}:
            continue
        # merge
        if len(word) > 2 and word[:2] == "##":
            new_text[-1] += word[2:]
            new_vectors[-1] += token_vecs_sum[word_idx]
        # append
        else:
            new_text.append(word)
            new_vectors.append(token_vecs_sum[word_idx])
    return new_text, new_vectors

def Bert(samples):
    # allocate sentence matrix
    N = samples.shape[0]
    sentence_matrix = torch.zeros((N, config.seq_len, config.bert_input_size))
    # iterate sentences
    for sentence_idx,sentence in enumerate(samples):
        # parse sentence
        words,vectors = _Bert(sentence)
        # sentence vector to sentence matrix
        for word_idx,word_vector in enumerate(vectors):
            sentence_matrix[sentence_idx,word_idx,:] = word_vector

    return sentence_matrix
    
__all__ = ["ScalarIncremental","ClosestWord2Vec","Bert"]
    