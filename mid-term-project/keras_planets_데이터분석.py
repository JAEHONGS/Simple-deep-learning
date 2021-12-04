# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 19:02:01 2021

@author: user
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf
from tensorflow import keras
import autokeras as ak
from tensorflow.keras.utils import plot_model

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

from sklearn.metrics import confusion_matrix, plot_confusion_matrix

sns.set(style = 'darkgrid')
planets = sns.load_dataset('planets')

x_ = planets.iloc[:, 1:6]
y_ = planets.iloc[:, 0]

x_ = x_.interpolate()
y_ = y_.interpolate()

x_train_res, y_train_res = RandomOverSampler().fit_resample(x_, y_)

x_train, x_test, y_train, y_test = train_test_split(x_train_res, y_train_res, test_size = 0.25, shuffle = True)


from sklearn.preprocessing import StandardScaler, LabelEncoder

x_train_norm = StandardScaler().fit_transform(x_train)
x_test_norm = StandardScaler().fit_transform(x_test)

y_train_en = LabelEncoder().fit_transform(y_train)
y_train_oh = pd.get_dummies(y_train_en).values

y_test_en = LabelEncoder().fit_transform(y_test)
y_test_oh = pd.get_dummies(y_test_en).values


keras_model = Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(32, activation = 'relu'),
    layers.Dense(32, activation = 'relu'),
    layers.Dense(10, activation = 'softmax')
])


keras_model.compile( loss = 'categorical_crossentropy', optimizer = Adam(0.001), metrics = ['acc'])

early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

keras_model.fit(
    x_train_norm, y_train_oh,
    epochs=200,
    callbacks=[early_stopping, reduce_lr],
    validation_split=0.15
)

y_pred_keras = keras_model.predict(x_test_norm)

labels = list(planets['method'].unique())
pred_y = []
true_y = []

for i in range(len(y_pred_keras)):
    temp = labels[y_pred_keras[i].argmax()]
    temp2 = labels[y_test_oh[i].argmax()]
    pred_y.append(temp)
    true_y.append(temp2)

pred_y = np.array(pred_y)
true_y = np.array(true_y)

confusion_matrix = confusion_matrix(pred_y, true_y, labels = labels)

plt.figure(figsize=(10,10))
index = planets['method'].unique()
heatmap = sns.heatmap(confusion_matrix, annot = True)
heatmap.set_xticklabels(index,rotation=50)
heatmap.set_yticklabels(index,rotation=50)
    
    