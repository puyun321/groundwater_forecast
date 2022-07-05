# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:47:35 2022

@author: steve
"""

import math
import numpy as np
import pandas as pd

P_path = r"D:\important\research\groundwater_forecast\daily_data\rainfall(processed).csv"
P = pd.read_csv(P_path,index_col=0)
P = np.array(P)

G_path = r"D:\important\research\groundwater_forecast\daily_data\groundwater_l0(processed).csv"
G = pd.read_csv(G_path,index_col=0)
G_date = np.array(G.index)
G = np.array(G)

T = np.array([25 for i in range(0, P.shape[0]*P.shape[1])]) # set temperature to constant 25 degree celsius
T = np.reshape(T,(P.shape[0],P.shape[1]))

#%%
import tensorflow as tf
from keras.models import Sequential,load_model
from keras import backend as K
from keras.layers import Dense,LSTM,Conv1D,Flatten
from keras.layers.core import Activation
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

def Dense_model(output_dimension):
    model = Sequential()
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(output_dimension), activation='softmax'))
    print(model.summary())
    return model 