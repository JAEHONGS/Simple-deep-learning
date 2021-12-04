# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 19:41:11 2021

@author: user
"""
import matplotlib.pyplot as plt

from Auxiliary import weights_load, accuracy
from SimpleCNN import SimpleCNN
from Losses import CrossEntropyWithSoftmax

import numpy as np
import pickle

def forward(nn, loss_layer, x_batch, t_batch) :
    y_batch = nn.forward(x_batch) 
    loss, _ = loss_layer.forward(t_batch,y_batch)

    loss = loss/len(x_batch)
    acc = accuracy(y_batch,t_batch)

    return (loss,acc)

with open('mnist_data.pkl','rb') as f : 
    xx,tt = pickle.load(f) 
    train_x,validate_x,test_x = xx
    train_t,validate_t,test_t = tt

weights_file = 'weights.pkl'
nn= SimpleCNN()
loss_layer = CrossEntropyWithSoftmax()
weights_load(nn,weights_file)

test_loss,test_acc = forward(nn,loss_layer,test_x,test_t)
print('test set -> loss: %f, acc: %5.2f%%'%(test_loss,test_acc))

W = nn.p_['conv1'].p_['Wcol']
W = W.T.reshape([-1,5,5])
W = 255*(W - np.min(W))/(np.max(W)-np.min(W))   # 0~255 사이 수로 정규화
print(W[0])

plt.figure(figsize=(16,10))

for i in range(5) :
    for j in range(6) :
        idx = 6*i+j
        plt.subplot(5,6,idx+1)
        plt.imshow(W[idx],cmap='gray',vmin=0, vmax=255)

plt.show()