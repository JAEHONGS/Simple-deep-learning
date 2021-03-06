# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 18:12:39 2021

@author: user
"""

import numpy as np

class ReLU() :
    
    def __init__(self) :
        self.p_ = {}
        self.mask = None

    def forward(self,X) :
        Y = X.copy()
        mask = (X<0)
        self.mask = mask
        Y[mask] = 0.

        return Y

    def backward(self,dY) :
        mask = self.mask
        dX = dY.copy()
        dX[mask] = 0.

        return dX

class Sigmoid() :
    
    def __init__(self) :
        self.Y = None
        
    def forward(self, x):
        Y = 1 / (1 + np.exp(-x))
        self.Y = Y
        
        return Y
    
    def backward(self, dY):
        dx = dY * (1.0 - self.Y) * self.Y
        
        return dx
