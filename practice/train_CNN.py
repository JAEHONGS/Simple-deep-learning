# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 18:14:53 2021

@author: user
"""

from Losses import CrossEntropyWithSoftmax
from Optimizers import SGD
from Auxiliary import choose, accuracy, weights_save
from SimpleCNN import SimpleCNN

import numpy as np
import numpy.random as rand
import pickle

def forward(nn, loss_layer, x_batch, t_batch) :
    y_batch = nn.forward(x_batch) 
    loss, _ = loss_layer.forward(t_batch,y_batch)

    loss = loss/len(x_batch)
    acc = accuracy(y_batch,t_batch)

    return (loss,acc)

def backward(nn, loss_layer) :
    dY = loss_layer.backward(1)
    nn.backward(dY)

def iterates(nn,batch_size,x_data,t_data,learning=True,shuffle=True) :          # 데이터 크기가 매우 커쳐서 배치가 엄청 커지면 
    x_data_ = np.array(x_data); t_data_ = np.array(t_data)                      # RAM 이 견디질 못 하고 컴퓨터가 다운될 수 있어서
                                                                                # iterates 함수를 만들어서 사용하게 되면 메모리를 효율적으로 사용
    order = np.arange(len(x_data_)) 
    if shuffle : np.random.shuffle(order)     #epoch마다 데이터를 섞을지 결정

    batch_size = 100; iter_tnum = int(len(x_data_)/batch_size)

    loss_vec = np.empty(iter_tnum); acc_vec = np.empty(iter_tnum)

    for j in range(iter_tnum) :

        if j%(iter_tnum/10) == 0 : 
            print('%d%%'%(j/iter_tnum*100),end=' ',flush=True)

        iter_idx = batch_size*j 
        
        if len(x_data_[iter_idx:]) < batch_size : break  #batch_size 크기만큼 미니배치를 못 만들면 끝냄
    
        x_batch, t_batch = choose(iter_idx,batch_size,x_data_,t_data_,order) #미니배치 생성

        loss_vec[j],acc_vec[j] = forward(nn,loss_layer,x_batch,t_batch) #순전파

        if learning :
            backward(nn,loss_layer)     #역전파 후 학습
            SGD(nn,alpha)

    print('100%')

    return (np.mean(loss_vec),np.mean(acc_vec))


rand.seed(0)

with open('mnist_data.pkl','rb') as f : 
    xx,tt = pickle.load(f) 
    train_x,validate_x,test_x = xx
    train_t,validate_t,test_t = tt

nn= SimpleCNN()
loss_layer = CrossEntropyWithSoftmax()

alpha = 1e-4; batch_size = 100; epoch_tnum = 100;

pre_validate_loss = np.full(5, 10000.)

for epoch in range(epoch_tnum) :
    train_loss,train_acc = iterates(nn,batch_size,train_x,train_t,True,False)
    validate_loss,validate_acc = iterates(nn,batch_size,validate_x,validate_t,False,False)

    print('*** %2dth epoch'%epoch)
    print('train set -> loss: %f, acc: %5.2f%%'%(train_loss,train_acc))
    print('validation set -> loss: %f, acc: %5.2f%%'%(validate_loss,validate_acc))

    if np.mean(pre_validate_loss) < validate_loss : break

    pre_validate_loss[:-1] = pre_validate_loss[1:]; pre_validate_loss[-1] = validate_loss

test_loss,test_acc = iterates(nn,batch_size,test_x,test_t,False,False)

print('test set -> loss: %f, acc: %5.2f%%'%(test_loss,test_acc))

file_name = 'weights.pkl'
weights_save(nn,file_name)