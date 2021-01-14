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
import embeddings

class Generator(nn.Module):
  """Generator."""
  def __init__(self, output_size, seq_len, input_size,
               hidden_size = 2, num_layers = 1, dropout_prob = .5):
    """Generator constructor.
    
    Args:
      output_size (int): Size of output.
      input_size (int): Size of input.
      hidden_size (int): Latent (hidden) layers size.
      num_layers (int): Number of latent (hidden) layers.
    """
    super(Generator, self).__init__()
    
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    self.input_size = input_size
    self.seq_len = seq_len
    self.output_size = output_size

    # LSTM layer
    self.lstm = nn.LSTM(input_size = input_size,   # input size
                        hidden_size = hidden_size, # hidden size
                        num_layers = num_layers,   # number of layers
                        batch_first = False)   

    # linear layer
    self.linear = nn.Linear(hidden_size * seq_len, output_size * seq_len)

  def forward(self, input, prev_state = None):
    """Forward propagation.
    
    Args:
        input (torch.Tensor): Input data batch.
        prev_state (torch.Tensor): Optional LSTM state.
    """
    #logging.info("generator: data [seq_len %d, batch_size %d]", input.shape[1], input.shape[0])

    h1, state = self.lstm(input, prev_state) # lstm
    h1_ = h1.view(h1.size(0), -1)

    o = self.linear(h1_) # dense layer
    o = o.reshape((o.shape[0], self.seq_len, self.output_size))
    return o, state
  
  def init_state(self):
    """Generates initial (zero) LSTM state tensor."""
    return (
      torch.zeros(self.num_layers, self.seq_len, self.hidden_size)\
          .to(config.device),
      torch.zeros(self.num_layers, self.seq_len, self.hidden_size)\
          .to(config.device)
    )

  def generate_input(self, N):
    """Generates random input for generator.
    
    Args:
        N (int): Size of input (batch).
    """
    # generate
    return torch.randint(embeddings.vocab_size(), size = (N,self.seq_len,self.input_size),
                       dtype = torch.float32)\
        .to(config.device)
        