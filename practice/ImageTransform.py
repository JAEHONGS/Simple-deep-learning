# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 17:50:08 2021

@author: user
"""

import numpy as np

def im2col(X,N,n_ic,n_fh,n_fw,n_oh,n_ow,p,s,layer_type) :
    X = np.pad(X,((0,0),(0,0),(p,p),(p,p)),constant_values=(0,))
    X_col = np.empty([N,n_ic,n_fh,n_fw,n_oh,n_ow])

    for i in range(n_fh) :
        i1 = i; i2 = i1 + s*n_oh
        for j in range(n_fw) :
            j1 = j; j2 = j1 + s*n_ow
            X_col[:,:,i,j,:,:] = X[:,:,i1:i2:s,j1:j2:s]

    if layer_type == 'conv' :
        X_col = np.transpose(X_col,[0,4,5,1,2,3]) 
        X_col = np.reshape(X_col,[N*n_oh*n_ow,n_ic*n_fh*n_fw])
    
    elif layer_type == 'pool' :
        X_col = np.transpose(X_col,[0,1,4,5,2,3]) 
        X_col = np.reshape(X_col,[N*n_oh*n_ow*n_ic,n_fh*n_fw])

    return X_col

def col2im(X_col,N,n_ih,n_iw,n_ic,n_fh,n_fw,n_oh,n_ow,p,s,layer_type) :
    
    if layer_type == 'conv' :
        X_col = np.reshape(X_col,[N,n_oh,n_ow,n_ic,n_fh,n_fw])
        X_col = np.transpose(X_col,[0,3,4,5,1,2])

    elif layer_type == 'pool' :
        X_col = np.reshape(X_col,[N,n_ic,n_oh,n_ow,n_fh,n_fw])
        X_col = np.transpose(X_col,[0,1,4,5,2,3])

    X = np.zeros([N,n_ic,n_ih+2*p,n_iw+2*p])
    for i in range(n_fh) :
        i1 = i; i2 = i1 + s*n_oh
        for j in range(n_fw) :
            j1 = j; j2 = j1 + s*n_ow
            X[:,:,i1:i2:s,j1:j2:s] = X_col[:,:,i,j,:,:]
    
    X = X[:,:,p:n_ih+1,p:n_iw+1]

    return X