# -*- coding: utf-8 -*-
"""
This module implements the Discriminator model of GAN.

@author: Martin Benes
"""

import sys
import torch
from torch import nn, cuda, optim, autograd

sys.path.append('src')
import config

class Discriminator(nn.Module):
  def __init__(self, seq_len, input_size,
               hidden_size = 5, num_layers = 1, dropout_prob = .5):
    """Generator constructor.
    
    Args:
      input_size (int): Size of input.
      hidden_size (int): Latent (hidden) layers size.
      num_layers (int): Number of latent (hidden) layers.
      dropout_prob (float): Probability parameter for dropout.
    """
    super(Discriminator, self).__init__()

    self.num_layers = num_layers
    self.hidden_size = hidden_size
    self.seq_len = seq_len
    self.input_size = self.embedding_dim = input_size

    # LSTM layer
    self.lstm = nn.LSTM(input_size = input_size,   # input size
                        hidden_size = hidden_size, # hidden size
                        num_layers = num_layers,   # number of layers
                        batch_first = False)
    self.lstm_dropout = nn.Dropout(p = dropout_prob)

    # linear layer (1)
    self.lin1 = nn.Linear(hidden_size * seq_len, hidden_size)
    self.lin1_dropout = nn.Dropout(p = dropout_prob)
    self.lin1_act = nn.LeakyReLU(0.2)

    # linear layer (2)
    self.lin2 = nn.Linear(hidden_size, 1)
    self.lin2_act = nn.Sigmoid()

  def forward(self, input, prev_state = None):
    """Forward propagation.
    
    Args:
        input (torch.Tensor): Input data batch.
        prev_state (torch.Tensor): Optional LSTM state.
    """
    #logging.info("discriminator: data [seq_len %d, batch_size %d]", input.shape[1], input.shape[0])

    h1, state = self.lstm(input, prev_state) # lstm
    h1d = self.lstm_dropout(h1)
    h1_ = h1.view(h1d.size(0), -1)
    
    l1 = self.lin1(h1_) # linear 1
    l1d = self.lin1_dropout(l1)
    a1 = self.lin1_act(l1d)

    l2 = self.lin2(a1) # linear 2
    a2 = self.lin2_act(l2)

    # output
    o = torch.cat((1 - a2, a2), 1)

    return o, state

  def init_state(self):
    """Generates initial (zero) LSTM state tensor."""
    return (
      autograd.Variable(
          torch.zeros(self.num_layers,self.seq_len,self.hidden_size,
                      dtype=torch.float32)\
              .to(config.device)
      ),
      autograd.Variable(
        torch.zeros(self.num_layers,self.seq_len,self.hidden_size,
                    dtype=torch.float32)\
            .to(config.device)
      )
    )
