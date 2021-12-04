# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 19:27:29 2021

@author: user
"""

import torchvision
import torchvision.transforms as transforms
import numpy as np
import pickle

mnist_train = torchvision.datasets.MNIST('.',train=True,download=True,transform=transforms.ToTensor())
mnist_test = torchvision.datasets.MNIST('.',train=False,download=True,transform=transforms.ToTensor())
print( mnist_train ); print( mnist_test )

train_x = []; train_t = []; test_x = []; test_t = []
j = 0

for e in [mnist_train,mnist_test] :
    n = len(e)

    if j == 0 :
        e_x = train_x; e_t = train_t

    elif j == 1 :
        e_x = test_x; e_t = test_t

    j += 1

    for i in range(n) :
        val_x = e[i][0].numpy(); e_x.append( val_x )
        val_t = np.zeros(10,dtype='int'); val_t[e[i][1]] = 1; e_t.append( val_t )

        if i%int(n/10) == 0 : print('%d%%'%((i+1)/n*100),end=' ')
    
    print('100%')

validate_x = test_x[:5000]; validate_t = test_t[:5000]
test_x = test_x[5000:]; test_t = test_t[5000:]

data = ((train_x,validate_x,test_x),(train_t,validate_t,test_t))

with open('mnist_data.pkl','wb') as f : 
    pickle.dump(data,f)