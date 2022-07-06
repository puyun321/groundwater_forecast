# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:47:35 2022

@author: steve
"""

import math
import numpy as np
import pandas as pd

G_path = r"D:\important\research\groundwater_forecast\daily_data\groundwater_l0.csv"
G = pd.read_csv(G_path,index_col=0)
G_date = np.array(G.index)
G_station = G.columns
G = np.array(G)

P_path = r"D:\important\research\groundwater_forecast\daily_data\rainfall.csv"
P = pd.read_csv(P_path,index_col=0)
P_station = P.columns
P = np.array(P)

T_path = r"D:\important\research\groundwater_forecast\daily_data\temperature.csv"
T = pd.read_csv(T_path,index_col=0)
T_station = T.columns
T = np.array(T)

ETpot_path = r"D:\important\research\groundwater_forecast\daily_data\evaporation_rate.csv"
ETpot = pd.read_csv(ETpot_path,index_col=0)
ETpot_station = ETpot.columns
ETpot = np.array(ETpot)

#%%
# Kriging


#%%
# Deep learning model
import tensorflow as tf
from keras import Model
from keras.engine.input_layer import Input
from keras.models import Sequential,load_model
from keras import backend as K
from keras.layers import Dense,LSTM,Conv1D,Flatten
from keras.layers.core import Activation
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.layers import BatchNormalization

def DNN_model(timestep,G_obs,P_obs,T_obs,output_dimension):
    inputs1 = Input(shape=(timestep,G_obs.shape[1]))
    output1=LSTM(36,stateful=False,return_sequences=True)(inputs1)
    output1=LSTM(36,stateful=False,return_sequences=False)(output1)
    output1=Flatten()(output1)
    output1=Dense(16, activation='linear')(output1)
    
    inputs2 = Input(shape=(P_obs.shape[1]))
    output2 = Dense(P_obs.shape[1], activation='linear')(inputs2)

    inputs3 = Input(shape=(G_obs.shape[1]))
    output3 = Dense(P_obs.shape[1], activation='linear')(inputs3)    
    
    inputs4 = Input(shape=(T_obs.shape[1]))
    output4 = Dense(T_obs.shape[1], activation='linear')(inputs4)  
    
    model = Model(inputs=[inputs1,inputs2,inputs3,inputs4], outputs=[output1,output2,output3,output4])

    print(model.summary())
    return model 

def HBV_error_function(k_pred,P_obs,T_obs,G_obs):
    
    parameters=k_pred
    parBETA = parameters[0]
    parCET = parameters[1]
    parFC = parameters[2]
    parK0 = parameters[3]
    parK1 = parameters[4]
    parK2 = parameters[5]
    parLP = parameters[6]
    parMAXBAS = parameters[7]
    parPERC = parameters[8]
    parUZL = parameters[9]
    parPCORR = parameters[10]
    parTT = parameters[11]
    parCFMAX = parameters[12]
    parSFCF = parameters[13]
    parCFR = parameters[14]
    parCWH = parameters[15]
    
    # Initialize time series of model variables
    SNOWPACK = np.zeros(parBETA.shape, dtype=np.float32) + 0.001
    MELTWATER = np.zeros(parBETA.shape, dtype=np.float32) + 0.001
    SM = np.zeros(parBETA.shape, dtype=np.float32) + 0.001
    SUZ = np.zeros(parBETA.shape, dtype=np.float32) + 0.001
    SLZ = np.zeros(parBETA.shape, dtype=np.float32) + 0.001
    ETact = np.zeros(parBETA.shape, dtype=np.float32) + 0.001
    Qsim = np.zeros(P_obs.shape, dtype=np.float32) * np.NaN

    # Apply correction factor to precipitation
    # P = np.tile(parPCORR, (len(P[:, 0]), 1)) * P
    
    # Separate precipitation into liquid and solid components
    PRECIP = P_obs * parPCORR
    RAIN = np.multiply(PRECIP, T_obs >= parTT)
    SNOW = np.multiply(PRECIP, T_obs < parTT)
    SNOW = SNOW * parSFCF

    # Snow
    SNOWPACK = SNOWPACK + SNOW
    melt = parCFMAX * (T_obs - parTT)
    melt = melt.clip(0.0, SNOWPACK)
    MELTWATER = MELTWATER + melt
    SNOWPACK = SNOWPACK - melt
    refreezing = parCFR * parCFMAX * (parTT - T_obs)
    refreezing = refreezing.clip(0.0, MELTWATER)
    SNOWPACK = SNOWPACK + refreezing
    MELTWATER = MELTWATER - refreezing
    tosoil = MELTWATER - (parCWH * SNOWPACK)
    tosoil = tosoil.clip(0.0, None)
    MELTWATER = MELTWATER - tosoil

    ### Caution !!! ###
    tosoil=0 # Taiwan did not melt snow
    
    # Soil and evaporation
    soil_wetness = (SM / parFC) ** parBETA
    soil_wetness = soil_wetness.clip(0.0, 1.0)
    recharge = (RAIN + tosoil) * soil_wetness
    SM = SM + RAIN + tosoil - recharge
    excess = SM - parFC
    excess = excess.clip(0.0, None)
    SM = SM - excess
    evapfactor = SM / (parLP * parFC)
    evapfactor = evapfactor.clip(0.0, 1.0)
    ETact = ETpot * evapfactor
    ETact = np.minimum(SM, ETact)
    SM = SM - ETact

    # Groundwater boxes
    SUZ = SUZ + recharge + excess
    PERC = np.minimum(SUZ, parPERC)
    SUZ = SUZ - PERC
    Q0 = parK0 * np.maximum(SUZ - parUZL, 0.0)
    SUZ = SUZ - Q0
    Q1 = parK1 * SUZ
    SUZ = SUZ - Q1
    SLZ = SLZ + PERC
    Q2 = parK2 * SLZ
    SLZ = SLZ - Q2
    Qsim = Q0 + Q1 + Q2
    
    return Qsim
    
#%%
import os
os.chdir(r"D:\important\research\research_use_function")
from multidimensional_reshape import multi_input_output

if __name__ == '__main__':
    
    timestep=3
    preprocessing_module=multi_input_output(G,input_timestep=timestep,output_timestep=timestep)
    G_multi_input=preprocessing_module.generate_input()
    G_multi_output=preprocessing_module.generate_output()
    
    model=DNN_model(timestep,G,P,T,output_dimension)
    
    
    
    
    