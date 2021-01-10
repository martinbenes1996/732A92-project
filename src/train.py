# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 16:32:22 2021

@author: martin
"""

import sys
import math
import torch
from torch import nn, cuda, optim, autograd
sys.path.append('src')
import logging
logging.basicConfig(level = logging.INFO)

import config
import gan
import embeddings

def create_training_step(create_models):
    # create models
    model_g,model_d = create_models()
    model_g.train()
    model_d.train()
    logging.info("empty models %s created", create_models)
    # generate states
    state_g = model_g.init_state()
    state_d = model_d.init_state()
    states = (state_g,state_d)
    # models and losses
    optim_g,optim_d = optimizers(model_g, model_d)
    optim_g.zero_grad()
    optim_d.zero_grad()
    crit_g,crit_d,true_label,postprocess_d = criterions_labels()
    
    def _train_step(epoch, batch_idx, batch, states):
        """"""
        # process input
        nonlocal model_g,model_d,optim_g,optim_d,crit_g,crit_d
        batch_size = batch.shape[0]
        state_g,state_d = states
        state_g = [s.detach() for s in state_g]
        state_d = [s.detach() for s in state_d]
        # generator input
        input_g = model_g.generate_input(batch_size)
        label_g = torch.zeros([batch_size]).to(config.device)
        # generator forward propagation
        #logging.info("generator forward propagation")
        pred_g,state_g = model_g(input_g, state_g)
        input_d = torch.cat([pred_g, batch], 0)
        label_d = torch.cat([torch.zeros([pred_g.shape[0]]),
                             torch.ones([batch.shape[0]])], 0)\
            .to(config.device)
        # descriminator forward propagation
        #logging.info("descriminator forward propagation")
        pred_d,_ = model_d(input_d, state_d)
        pred_d = postprocess_d(pred_d)
        # generator backward propagation
        #logging.info("generator backward propagation")
        score_dg = pred_d[:batch_size] # generator score
        loss_g = crit_g(score_dg, label_g)
        optim_g.zero_grad()
        loss_g.backward()
        nn.utils.clip_grad_norm_(model_g.parameters(), 100)
        optim_g.step()
        # discriminator forward propagation
        #logging.info("discriminator forward propagation 2")
        pred_d,state_d = model_d(input_d.detach(), state_d)
        pred_d = postprocess_d(pred_d)
        # discriminator backward propagation
        #logging.info("discriminator backward propagation")
        loss_d = crit_d(pred_d, label_d)
        optim_d.zero_grad()
        loss_d.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(model_d.parameters(), 100)
        optim_d.step()
        state_g = [s.detach() for s in state_g]
        state_d = [s.detach() for s in state_d]
        
        losses = (loss_g.detach().item(),loss_d.detach().item())
        
        if batch_idx % 10 == 0:
            print("  Batch %d: D log-error %.6f, G log-error %.6f" %
                  (batch_idx, math.log(losses[1]), math.log(losses[0])))
            
        return (state_g,state_d), losses
    
        # log
        #losses_d.append(loss_d.detach().item())
        #losses_g.append(loss_g.detach().item())
        
    return _train_step, states

def optimizers(model_g, model_d):
    """"""
    def _optimizer(params):
        """"""
        return optim.SGD(params, lr = config.learning_rate, momentum=.9)
    # optimizers
    torch.autograd.set_detect_anomaly(True)
    return _optimizer(model_g.parameters()), _optimizer(model_d.parameters())
def criterions_labels(crit_d = 'bce'):
    """"""
    assert(crit_d in {'crossentropy', 'bce', 'mse'})
    # true labels
    true_label = torch.ones([config.batch_size], dtype=torch.long)\
        .to(config.device)
    postprocess_d = lambda x:x
    # cross entropy
    if crit_d == 'crossentropy':
        criterion_d = nn.CrossEntropyLoss()
        true_label = true_label
    elif crit_d == 'bce': # BCE
        criterion_d = nn.BCELoss()
        true_label = true_label\
            .type(torch.float32)\
            .reshape((*true_label.shape, 1))
        postprocess_d = lambda x:x[:,1]
        true_label = torch.cat((true_label, 1 - true_label), 1)
    elif crit_d == 'mse': # MSE
        criterion_d = nn.MSELoss()
        postprocess_d = lambda x:x[:,1]
        true_label = true_label\
            .type(torch.float32)
    crit_g = nn.MSELoss()

    return crit_g,criterion_d,true_label,postprocess_d

def ScalarIncremental(from_drive = False):
    """"""
    train_step,states = create_training_step(gan.ScalarIncremental)
    logging.info("initializing ScalarIncremental model training")
    # iterate batches
    losses_g,losses_d = [],[]
    for epoch in range(config.num_epochs):
        for i,batch in enumerate(embeddings.load.ScalarIncremental(from_drive=from_drive)):
            logging.info("received batch %d: %s", i, batch.shape)
            states,losses = train_step(epoch, i, batch, states)
            losses_g.append(losses[0])
            losses_d.append(losses[1])
        print("Epoch %d/%d: D error %.6f, G error %.6f" %
          (epoch + 1, config.num_epochs, losses_d[-1], losses_g[-1]))

config.batch_size = 5
ScalarIncremental(from_drive = True)

def ClosestWord2Vec():
    optimizer_g = optim.SGD(model_g.parameters(), lr=learning_rate, momentum=0.9)
    optimizer_d = optim.SGD(model_d.parameters(), lr=learning_rate, momentum=0.9)
    vector_x = torch.zeros((config.batch_size,config.seq_len,config.input_size))\
        .to(config.device)
    for i,batch in enumerate(embeddings.load.ClosestWord2Vec()):
        # transform batch
        for sample in range(batch_size):
            for word in range(D.seq_len):
                vector_x[sample,word,:] = X_vectors[int(data_x[sample,word]),:]
        data_x = vector_x[:batch_size]
        print("Received batch %d: %s" % (i,batch.shape))

def Bert():
    optimizer_g = optim.SGD(model_g.parameters(), lr=learning_rate, momentum=0.9)
    optimizer_d = optim.SGD(model_d.parameters(), lr=learning_rate, momentum=0.9)
    for i,batch in enumerate(embeddings.load.Bert()):
        print("Received batch %d: %s" % (i,batch.shape))
        #break
ScalarIncremental()