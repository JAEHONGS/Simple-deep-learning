# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 18:24:17 2021

@author: user
"""

from Layers import Linear, Conv2D, Pool2D, Flatten
from Activations import ReLU

import collections

class SimpleCNN :

    def __init__(self) :
        self.p_ = collections.OrderedDict()    # dictionary엔 보통 order가 없지만, order가 존재하는 dictionary

        n_ih = n_iw = 28                       # 28 x 28  h : height, w : width
        n_ic = 1; n_oc = 30; n_fh = n_fw = 5     
        p = 0; s = 1 
        self.p_['conv1'] = Conv2D(n_ic,n_oc,n_fh,n_fw,p,s)
        self.p_['relu1'] = ReLU()

        n_ih = int((n_ih + 2*p - n_fh)/s) + 1    # 위 Layer를 거치고 나온 output의 h, w
        n_iw = int((n_iw + 2*p - n_fw)/s) + 1    # 아래 pool1의 인풋으로 사용됨

        n_ic = n_oc; n_ph = n_pw = 2 
        p = 0; s = 2 
        self.p_['pool1'] = Pool2D(n_ic,n_ph,n_pw,p,s)

        n_ih = int((n_ih + 2*p - n_ph)/s) + 1 
        n_iw = int((n_iw + 2*p - n_pw)/s) + 1 

        self.p_['flatten'] = Flatten()
        
        self.p_['linear1'] = Linear(n_ic*n_ih*n_iw,100,'xavier')
        self.p_['relu2'] = ReLU()

        self.p_['linear2'] = Linear(100,10,'he')
        
    def forward(self,x) :
        p_ = self.p_
        
        for layer in p_.keys() :
            x = p_[layer].forward(x)
            
        return x

    def backward(self,dy) :
        p_ = self.p_
        
        for layer in reversed(p_.keys()) :
            dy = p_[layer].backward(dy)
        
        return dy

