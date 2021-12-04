# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 18:21:25 2021

@author: user
"""

import numpy as np

def SGD(nn,alpha) :

    for layer in nn.p_.keys() :
        layer_ = nn.p_[layer]
        p_ = layer_.p_

        for w in p_.keys() :
            dw = layer_.dp_[w]
            p_[w] -= alpha*dw