# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:47:35 2022

@author: steve
"""

import math
import numpy as np
import pandas as pd

G_path = r"D:\important\research\groundwater_forecast\daily_data\groundwater_l0.csv"
# G = pd.read_csv(G_path,index_col=0)
G = pd.read_csv(G_path);G['date'] = pd.to_datetime(G['date']);
# G = np.array(G);
G = G.resample('10D',on='date',base=0,loffset='9D').mean()
G_date = np.array(G.index)
G_station = G.columns
G_station = pd.Series([G_station[i][8:10] for i in range(0,len(G_station))])
G_station = G_station.drop_duplicates (keep='first').reset_index(drop=True)

#merge stations with same name
station_index=[[]*1 for i in range(0,len(G_station.duplicated(keep=False)))];record=0
for i in range(0,len(G_station)):
    if i==0:
        station_index[record].append(i)
        record2=G_station.iloc[i]
    else:
        if G_station.iloc[i]==record2:
            station_index[record].append(i)   
        else:
            station_index[record+1].append(i)               
            record2=G_station.iloc[i]
            record=record+1
station_index=list(filter(None,station_index))
G = np.array([G.iloc[:,station_index[i]].mean(axis=1) for i in range(0,len(station_index))]).T

P_path = r"D:\important\research\groundwater_forecast\daily_data\rainfall.csv"
# P = pd.read_csv(P_path,index_col=0)
P = pd.read_csv(P_path);P['date'] = pd.to_datetime(P['date']);
P = P.resample('10D',on='date',base=0,loffset='9D').mean()
P_station = P.columns
P = np.array(P)

T_path = r"D:\important\research\groundwater_forecast\daily_data\temperature.csv"
T = pd.read_csv(T_path,index_col=0)
T = pd.read_csv(T_path);T['date'] = pd.to_datetime(T['date']);
T = T.resample('10D',on='date',base=0,loffset='9D').mean()
T_station = T.columns
T = np.array(T)

ETpot_path = r"D:\important\research\groundwater_forecast\daily_data\evaporation_rate.csv"
ETpot = pd.read_csv(ETpot_path,index_col=0)
ETpot = pd.read_csv(ETpot_path);ETpot['date'] = pd.to_datetime(ETpot['date']);
ETpot = ETpot.resample('10D',on='date',base=0,loffset='9D').mean()
ETpot_station = ETpot.columns
ETpot = np.array(ETpot)

#%%
import os 
os.chdir(r"D:\important\research\research_use_function")
from multidimensional_reshape import multi_input_output
G_multi_module=multi_input_output(G,input_timestep=3,output_timestep=3)
G_input=G_multi_module.generate_input()
G_multi_output=G_multi_module.generate_output()
forecast_timestep=3
G_output=G_multi_output[:,forecast_timestep-1,:]
G_input_date=G_date[forecast_timestep-1:-forecast_timestep-1]

#%%
# Kriging
G_station_info = pd.read_csv(r"D:\important\research\groundwater_forecast\station_info\groundwater_station.csv")
G0_station_info = pd.DataFrame(np.squeeze(np.array([G_station_info.iloc[G_station_info[G_station_info.iloc[:,0]==G_station[i]].index,:] for i in range(0,len(G_station))])))
P_station_info = pd.read_csv(r"D:\important\research\groundwater_forecast\station_info\rainfall_station.csv")
T_station_info = pd.DataFrame(np.squeeze(np.array([P_station_info.iloc[P_station_info[P_station_info.iloc[:,0]==T_station[i]].index,:] for i in range(0,len(T_station))])))
T_station_info.columns = G_station_info.columns
ETpot_station_info = pd.DataFrame(np.squeeze(np.array([P_station_info.iloc[P_station_info[P_station_info.iloc[:,0]==ETpot_station[i]].index,:] for i in range(0,len(ETpot_station))])))
ETpot_station_info.columns = G_station_info.columns            

#%%
import math
import copy
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from scipy.interpolate import griddata

def Ordinary_Kriging(data,station_info,grid_lon,grid_lat):
    OK = OrdinaryKriging(station_info.loc[:, 'X'],station_info.loc[:, 'Y'], data, variogram_model="gaussian") 
    # Execute on grid:
    z1, ss1 = OK.execute("grid", grid_lon, grid_lat)
    return z1,ss1

def Universal_Kriging(data,station_info,grid_lon,grid_lat):
    OK = UniversalKriging(station_info.loc[:, 'X'],station_info.loc[:, 'Y'], data, variogram_model="exponential",drift_terms=["regional_linear"],) 
    # Execute on grid:
    z1, ss1 = OK.execute("grid", grid_lon, grid_lat)
    return z1,ss1

def interpolate2grid(data,station_info,grid_lon,grid_lat,interpolate_method='linear'):
    X,Y = np.meshgrid(grid_lon,grid_lat)
    Z = np.squeeze(griddata([(x,y) for x,y in zip(station_info.loc[:, 'X'],station_info.loc[:, 'Y'])], data, (X, Y), method=interpolate_method))
    return Z


P_grid_lon = np.linspace(math.floor(min(P_station_info.loc[:, 'X'])), math.ceil(max(P_station_info.loc[:, 'X'])), 30)
P_grid_lat = np.linspace(math.floor(min(P_station_info.loc[:, 'Y'])), math.ceil(max(P_station_info.loc[:, 'Y'])), 30)

# z1, ss1 = Ordinary_Kriging(pd.DataFrame(P[23, :]),P_station_info,P_grid_lon,P_grid_lat)
# z1, ss1 = Universal_Kriging(pd.DataFrame(P[17, :]),P_station_info,P_grid_lon,P_grid_lat)

P_z = [interpolate2grid(pd.DataFrame(P[i, :]),P_station_info,P_grid_lon,P_grid_lat) for i in range(0,len(P))]
P_z = np.nan_to_num(P_z)
T_z = np.array([interpolate2grid(pd.DataFrame(T[i, :]),T_station_info,P_grid_lon,P_grid_lat,interpolate_method='nearest') for i in range(0,len(T))])
ETpot_z = np.array([interpolate2grid(pd.DataFrame(ETpot[i, :]),ETpot_station_info,P_grid_lon,P_grid_lat,interpolate_method='nearest') for i in range(0,len(ETpot))])
X,Y = np.meshgrid(P_grid_lon,P_grid_lat)

#%%
def get_specific_coordinate(x1,y1,x2,y2):
    distance=[[]*1 for i in range(0,x2.shape[0])]
    for i in range(0,x2.shape[0]):
        for j in range(0,x2.shape[1]):
            # distance[i].append(math.dist([x1,y1],[x2[i,j],y2[i,j]]))
            distance[i].append(((((x2[i,j] - x1 )**2) + ((y2[i,j]-y1)**2) )**0.5))

    distance=np.array(distance)
    min_distance=np.min(distance)
    min_index=np.where(distance==min_distance)
    coordinate=np.array([x2[min_index[0],min_index[1]],y2[min_index[0],min_index[1]]])

    return coordinate,min_index

all_info=[];
for i in range(0,len(G0_station_info)):
    all_info.append(get_specific_coordinate(G0_station_info.iloc[i,1],G0_station_info.iloc[i,2],X,Y))

all_coordinate=[];min_index=[];
all_coordinate=np.squeeze(np.array([all_info[i][0] for i in range(0,len(all_info))]))
min_index=np.squeeze(np.array([all_info[i][1] for i in range(0,len(all_info))]))

#%%
#select specific grid
P_input=P_z[forecast_timestep-1:-forecast_timestep-1,min_index[:,0],min_index[:,1]]
T_input=T_z[forecast_timestep-1:-forecast_timestep-1,min_index[:,0],min_index[:,1]]
ETpot_input=ETpot_z[forecast_timestep-1:-forecast_timestep-1,min_index[:,0],min_index[:,1]]

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
    
    
    
    
    