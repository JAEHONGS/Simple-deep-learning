# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 17:59:19 2021

@author: user
"""

from ImageTransform import im2col, col2im

import numpy as np

class Linear() :
    
    def __init__(self, n, m, init_method='he') :
        self.p_ = {}

        if init_method == 'xavier' : 
            self.p_['W'], self.p_['b'] = \
                self.init_xavier_weights(n, m)

        elif init_method == 'he' :
            self.p_['W'], self.p_['b'] = \
                self.init_he_weights(n, m) 

        self.dp_ = {}
        self.dp_['W'], self.dp_['b'] = (None, None)
        self.X = None
        
    def init_xavier_weights(self, n, m) :
        W = np.sqrt(1./n)*np.random.randn(n, m)
        b = np.zeros(m)

        return (W, b)

    def init_he_weights(self, n, m) :
        W = np.sqrt(2./n) * np.random.randn(n, m)
        b = np.zeros(m)

        return (W, b)
    
    def forward(self, X) :
        W,b = self.load()
        self.X = X 
        Y = np.dot(X, W) + b

        return Y
    
    def backward(self, dY) :
        W,b = self.load()
        X = self.X
        self.dp_['W'] = np.dot(X.T, dY)
        self.dp_['b'] = np.sum(dY, axis = 0)
        dX = np.dot(dY, W.T)
        
        return dX

    def load(self) :
        
        return (self.p_['W'], self.p_['b'])
    
class Conv2D() : 

    def __init__(self, n_ic, n_oc, n_fh, n_fw, p, s, init_method = 'he') :
        self.p_ = {} 
        
        if init_method == 'xavier' : 
            W,self.p_['b'] = self.init_xavier_weights(n_oc, n_ic, n_fh, n_fw)
        
        elif init_method == 'he' : 
            W,self.p_['b'] = self.init_he_weights(n_oc, n_ic, n_fh, n_fw) 
        
        self.p_['Wcol'] = W.reshape([n_oc, n_ic * n_fh * n_fw]).T

        self.dp_ = {}
        self.dp_['Wcol'],self.dp_['b'] = (None, None)

        self.conv_shape = (n_ic, n_oc, n_fh, n_fw, p, s)
        self.imag_shape = None; self.Xcol = None;

    def init_xavier_weights(self, n_oc, n_ic, n_fh, n_fw) :
        W = np.sqrt(1./n_fh * n_fw) * np.random.randn(n_oc, n_ic, n_fh, n_fw)
        b = np.zeros(n_oc)
        
        return (W, b)
    
    def init_he_weights(self, n_oc, n_ic, n_fh, n_fw) :
        W = np.sqrt(2./n_fh * n_fw) * np.random.randn(n_oc, n_ic, n_fh, n_fw)
        b = np.zeros(n_oc)

        return (W, b)

    def load(self) :
        
        return (self.p_['Wcol'], self.p_['b'], self.conv_shape, self.imag_shape)
    
    def forward(self, X) :
        Wcol, b, conv_shape, imag_shape = self.load()
        n_ic, n_oc, n_fh, n_fw, p, s = conv_shape
        N, _ , n_ih, n_iw = self.imag_shape = X.shape
        n_oh = int((n_ih + 2*p - n_fh) / s) + 1; n_ow = int((n_iw + 2*p - n_fw) / s) + 1

        Xcol = im2col(X, N, n_ic, n_fh, n_fw, n_oh, n_ow, p, s,'conv'); self.Xcol = Xcol

        Y = np.dot(Xcol, Wcol) + b
        
        Y = Y.reshape([N, n_oh, n_ow, n_oc]).transpose([0, 3, 1, 2])
        
        return Y

    def backward(self, dY) :
        Wcol,b,conv_shape,imag_shape = self.load(); Xcol = self.Xcol
        n_ic, n_oc, n_fh, n_fw, p, s = conv_shape
        N, _, n_ih, n_iw = imag_shape
        _, _, n_oh, n_ow = dY.shape

        dY = dY.transpose([0, 2, 3, 1]).reshape([N * n_oh * n_ow, n_oc])
        
        self.dp_['Wcol'] = np.dot(Xcol.T, dY)
        self.dp_['b'] = np.sum(dY,axis = 0)
        dXcol = np.dot(dY, Wcol.T)

        dX = col2im(dXcol, N, n_ih, n_iw, n_ic, n_fh, n_fw, n_oh, n_ow, p, s,'conv')
        
        return dX
    
class Pool2D() :

    def __init__(self, n_ic, n_ph, n_pw, p, s) :
        self.p_ = {}
        self.pool_shape =(n_ic, n_ph, n_pw, p, s)
        self.Xcol = None; self.iY = None

    def forward(self, X) :
        n_ic, n_ph, n_pw, p, s = self.pool_shape
        N, _, n_ih, n_iw = self.imag_shape = X.shape
        n_oh = int((n_ih + 2*p - n_ph) / s) + 1; n_ow = int((n_iw + 2*p - n_pw)/s) + 1

        Xcol = im2col(X, N, n_ic, n_ph, n_pw, n_oh, n_ow, p, s,'pool'); self.Xcol = Xcol

        Y = np.max(Xcol, axis = 1)
        self.iY = np.argmax(Xcol, axis = 1)

        Y = Y.reshape([N, n_ic, n_oh, n_ow])
        
        return Y

    def backward(self, dY) :
        n_ic, n_ph, n_pw, p, s = self.pool_shape
        _, _, n_ih, n_iw = self.imag_shape
        Xcol = self.Xcol
        N, n_oc, n_oh, n_ow = dY.shape

        dY = dY.reshape([N * n_ic * n_oh * n_ow, 1])
        
        dXcol = np.zeros([N * n_oh * n_ow * n_ic, n_ph * n_pw])
        dXcol[np.arange(len(self.iY)), self.iY] = dY.flatten()

        dX = col2im(dXcol, N, n_ih, n_iw, n_ic, n_ph, n_pw, n_oh, n_ow, p, s,'pool')

        return dX
    
class Flatten() :

    def __init__(self) :
        self.p_ = {}
        self.X_shape = None

    def forward(self,X) :
        self.X_shape = X.shape
        
        return np.reshape(X, [len(X), -1])

    def backward(self, dY) :
        
        return np.reshape(dY, self.X_shape)

class Max() :
    
    def __init__(self) :
        self.p_ = {}
        self.iX, self.Xshape = (None, None)

    def forward(self, X) :
        self.Xshape = X.shape
        self.iX = np.argmax(X, axis = 1)
        Y = np.max(X, axis = 1)

        return Y

    def backward(self, dY) :
        X_shape = self.X_shape; N = X_shape[0]
        dX = np.zeros(self.X_shape)
        dX[1:N+1, self.iX] = dY.flatten()

        return dX
    
class Dropout() : 

    def __init__(self, ratio=0.5) : 
        self.ratio = ratio
        self.mask = None
        self.mode = 'train'

    def set_mode(self, mode) :
        self.mode = mode

    def forward(self, x) :
        if self.mode == 'train' : 
            self.mask = (np.random.rand(*x.shape) > self.ratio)
        
            return x * self.mask

        elif self.mode == 'test' : 
            
            return (1.-self.ratio)*x

    def backward(self, dout) :

        return dout * self.mask

class BatchNorm2D() :

    def __init__(self, n_ic, momentum=0.9):
        self.mode = 'train'

        self.p_ = {}
        self.p_['gamma'] = np.ones([1,n_ic,1,1])
        self.p_['beta'] = np.zeros([1,n_ic,1,1])
        self.dp_ = {}
        self.dp_['gamma'], self.dp_['beta'] = (None,None)
        self.running_mean = np.zeros([1,n_ic,1,1])
        self.running_var = np.ones([1,n_ic,1,1])
        self.momentum = momentum
        self.xc, self.std = (None,None)

    def set_mode(self,mode) :
        self.mode = mode

    def load(self) :
        
        return (self.p_['gamma'],self.p_['beta'],self.running_mean,self.running_var)
    
    def forward(self, x):
        N, n_ic, n_ih, n_iw = x.shape
        gamma, beta, running_mean, running_var = self.load()

        if self.mode == 'train' :
            mu = np.sum(x,axis=(0,2,3),keepdims=True) / (N*n_ih*n_iw)
            xc = x - mu; self.xc = xc
            
            var = np.sum(xc**2,axis=(0,2,3),keepdims=True) / (N*n_ih*n_iw)
            std = np.sqrt(var + 1e-7); self.std = std
            xn = xc / std; self.xn = xn

            self.running_mean = self.momentum * self.running_mean + (1.-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1.-self.momentum) * var

        elif self.mode == 'test' :
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var+ 1e-7)))
        
        y = gamma * xn + beta
            
        return y

    def backward(self, dy):
        N,n_ic,n_ih,n_iw = dy.shape
        gamma,beta,running_mean,running_var = self.load()

        self.dp_['beta'] = np.sum(dy,axis=(0,2,3),keepdims=True)
        self.dp_['gamma'] = np.sum(self.xn * dy, axis=(0,2,3),keepdims=True)
        dxn = gamma * dy

        dxc = dxn / self.std
        dstd = -np.sum( (dxn * self.xc) / self.std**2, axis=(0,2,3),keepdims=True)
        dvar = 0.5 * dstd / self.std
        dxc += (2. / (N*n_ih*n_iw)) * self.xc * dvar

        dmu = np.sum(dxc, axis=(0,2,3),keepdims=True)
        dx = dxc - dmu / (N*n_ih*n_iw)
        
        return dx
