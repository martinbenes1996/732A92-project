# -*- coding: utf-8 -*-

from . import load
from . import train
from . import models

from .models import *


def vocab_size():
    """"""
    return len(train.ScalarIncremental())
