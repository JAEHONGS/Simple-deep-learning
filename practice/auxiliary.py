# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 17:54:32 2021

@author: user
"""

import numpy as np
import pickle

def choose(n,batch_size,x_data,t_data) :
    x_batch = x_data[n:(n+batch_size)]
    t_batch = t_data[n:(n+batch_size)]
    
    return (x_batch,t_batch)

def accuracy(y_batch,t_batch) :
    ylb = np.argmax(y_batch,axis=1)
    tlb = np.argmax(t_batch,axis=1)
    
    return np.mean(ylb == tlb) * 100

def numerical_gradient(f, x):
    
    h = 1e-4                                     #유한차분법을 이용해서 수치적으로 도함수를 구함
    grad = np.zeros_like(x)                      #(배열의 iterator를 이용했기 때문에 x가 몇 차원 벡터인가에 관계없이 작동)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]

        x[idx] = float(tmp_val) + h; fxh1 = f(x) # f(x+h)
        x[idx] = tmp_val - h; fxh2 = f(x) # f(x-h)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val
        it.iternext() 
    
    return grad

def weights_save(nn,file_name) :
    
    data = {}
    
    for layer in nn.p_.keys() :
        layer_ = nn.p_[layer]
        p_ = layer_.p_
        
        weights = {}

        for w in p_.keys() :
            weights[w] = p_[w]

        data[layer] = weights
        
    with open(file_name,'wb') as f :
        pickle.dump(data,f)
        
def weights_load(nn,file_name) :

    with open(file_name,'rb') as f :
        data = pickle.load(f)

    for layer in nn.p_.keys() :
        layer_ = nn.p_[layer]
        p_ = layer_.p_

        for w in p_.keys() :
            p_[w] = data[layer][w]
        

