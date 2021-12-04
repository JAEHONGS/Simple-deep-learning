# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 18:13:08 2021

@author: user
"""

import numpy as np

eps = 1e-8
class CrossEntropyWithSoftmax :
    
    def __init__(self,reduction=True) :
        self.p_ = {}
        self.T = None
        self.Y = None
        self.reduction = reduction

    def forward(self,T,X) :
        self.T = T
        expX = np.exp(X-np.max(X))
        Y = expX/(eps+np.sum(expX,axis=1,keepdims=True))
        self.Y = Y
        loss = T*np.log(Y)
        
        if self.reduction :
            loss = -np.sum(loss)

        else :
            loss = -np.sum(loss,axis=1)

        return (loss, Y)

    def backward(self,dY) :
        T = self.T; Y = self.Y
        
        return Y - T