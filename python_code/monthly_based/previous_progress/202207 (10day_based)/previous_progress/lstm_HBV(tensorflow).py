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
G = G.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
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
P = P.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
P_station = P.columns
P = np.array(P)

T_path = r"D:\important\research\groundwater_forecast\daily_data\temperature.csv"
# T = pd.read_csv(T_path,index_col=0)
T = pd.read_csv(T_path);T['date'] = pd.to_datetime(T['date']);
T = T.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
T_station = T.columns
T = np.array(T)

ETpot_path = r"D:\important\research\groundwater_forecast\daily_data\evaporation_rate.csv"
# ETpot = pd.read_csv(ETpot_path,index_col=0)
ETpot = pd.read_csv(ETpot_path);ETpot['date'] = pd.to_datetime(ETpot['date']);
ETpot = ETpot.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
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

""" Since the Kriging fail to provide a good interpolation, we applied the IDW interpolation method to grid our data"""
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
from keras.layers import Dense,LSTM,Conv1D,Flatten,Concatenate
from keras.layers.core import Activation
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.layers import BatchNormalization

def DNN_model(timestep,G_obs,P_obs,T_obs,ETpot_obs):
    inputs1 = Input(shape=(timestep,G_obs.shape[2]))
    output1=LSTM(36,stateful=False,return_sequences=True)(inputs1)
    output1=LSTM(36,stateful=False,return_sequences=False)(output1)
    output1=Flatten()(output1)
    output_1=Dense(G_obs.shape[2]*3, activation='relu')(output1) # 還有兩個參數還未被用到
    # output_2=Dense(G_obs.shape[2], activation='linear')(output1) # 還有兩個參數還未被用到
    # output_3=Dense(G_obs.shape[2], activation='linear')(output1) # 還有兩個參數還未被用到
  
    
    inputs2 = Input(shape=(P_obs.shape[1]))
    # HBV_input2 = Dense(P_obs.shape[1], activation='linear')(inputs2)

    inputs3 = Input(shape=(T_obs.shape[1]))
    # HBV_input3 = Dense(T_obs.shape[1], activation='linear')(inputs3)    
    
    inputs4 = Input(shape=(ETpot_obs.shape[1]))
    # HBV_input4 = Dense(ETpot_obs.shape[1], activation='linear')(inputs4)  
    
    output = Concatenate(axis=-1)([output_1,inputs2,inputs3,inputs4]);
    # output = Concatenate(axis=-1)([output_1,HBV_input2,HBV_input3,HBV_input4]);

    model = Model(inputs=[inputs1,inputs2,inputs3,inputs4], outputs=output)

    print(model.summary())
    return model 

def HBV_error_function(G_obs,model_output):
    
    (parFC, parPERC, parPCORR, P_obs, T_obs, ETpot_obs) = tf.split(model_output, num_or_size_splits=[tf.shape(G_obs)[1],tf.shape(G_obs)[1],tf.shape(G_obs)[1],8,8,8], axis=1)
    
    tf.random.set_seed(111)
    parameters = tf.random.uniform(shape=[11, tf.shape(G_obs)[1]],minval=0.0,maxval=1.0)
    (parBETA, parK0, parK1, parK2, parLP, parUZL, parTT, parCFMAX, parSFCF, parCFR, parCWH) = tf.split(parameters,num_or_size_splits=11,axis=0)
    """ Initialize time series of model variables """

    SNOWPACK = tf.add(tf.zeros(tf.shape(parBETA), dtype=tf.float32),0.001)
    MELTWATER = tf.add(tf.zeros(tf.shape(parBETA), dtype=tf.float32),0.001)
    SM = tf.add(tf.zeros(tf.shape(parBETA), dtype=tf.float32),0.001)
    SUZ = tf.add(tf.zeros(tf.shape(parBETA), dtype=tf.float32),0.001)
    SLZ = tf.add(tf.zeros(tf.shape(parBETA), dtype=tf.float32),0.001)
    ETact = tf.add(tf.zeros(tf.shape(parBETA), dtype=tf.float32),0.001)
    Qsim = tf.zeros(tf.shape(P_obs), dtype=tf.float32)
    # Apply correction factor to precipitation
    P_obs =  tf.multiply(parPCORR, P_obs)
    
    """ Separate precipitation into liquid and solid components """

    PRECIP = tf.multiply(P_obs, parPCORR)
    RAIN = tf.where(tf.math.greater_equal(T_obs,parTT),tf.multiply(PRECIP,T_obs), tf.multiply(PRECIP,0))
    SNOW = tf.where(tf.math.less(T_obs, parTT),tf.multiply(PRECIP,T_obs),tf.multiply(PRECIP,0))
    SNOW = tf.multiply(SNOW,parSFCF)
    
    # RAIN_SNOW = tf.cond(tf.math.greater_equal(T_obs,parTT),PRECIP,tf.multiply(PRECIP,parSFCF))
    """ Snow """

    SNOWPACK = tf.add(SNOWPACK,SNOW)
    melt = tf.multiply(parCFMAX,tf.subtract(T_obs,parTT))
    melt = tf.clip_by_value(melt,0.0, SNOWPACK)
    MELTWATER = tf.add(MELTWATER,melt)
    SNOWPACK = tf.subtract(SNOWPACK,melt)
    refreezing =  tf.multiply(parCFR, tf.multiply(parCFMAX ,tf.subtract(parTT,T_obs)))
    refreezing =  tf.clip_by_value(refreezing,0.0, MELTWATER)
    SNOWPACK = tf.add(SNOWPACK,refreezing)
    MELTWATER = tf.subtract(MELTWATER,refreezing)
    tosoil = tf.subtract(MELTWATER, tf.multiply(parCWH,SNOWPACK))
    tosoil = tf.clip_by_value(tosoil,0.0,tosoil)
    MELTWATER = tf.subtract(MELTWATER,tosoil)

    ### Caution !!! ###
    tosoil=0 # Taiwan did not melt snow
    
    """ Soil and evaporation """
        
    soil_wetness = tf.pow(tf.divide(SM, parFC), parBETA)
    soil_wetness = tf.clip_by_value(soil_wetness,0.0,1.0)
    recharge = tf.multiply(tf.add(RAIN,tosoil),soil_wetness)
    SM = tf.subtract(tf.add(tf.add(SM , RAIN),tosoil),recharge)
    excess = tf.subtract(SM,parFC)
    excess = tf.clip_by_value(excess,0.0,excess)
    SM = tf.subtract(SM,excess)
    evapfactor = tf.divide(SM, tf.multiply(parLP, parFC))
    evapfactor = tf.clip_by_value(evapfactor,0.0,1.0)
    ETact = tf.multiply(ETpot_obs, evapfactor)
    ETact = tf.minimum(SM, ETact)
    SM = tf.subtract(SM, ETact)

    """ Groundwater boxes """
   
    SUZ = tf.add(SUZ,tf.add(recharge,excess))
    PERC  = tf.minimum(SUZ, parPERC)
    SUZ = tf.subtract(SUZ, PERC)
    Q0 = tf.multiply(parK0,tf.maximum(tf.subtract(SUZ,parUZL), 0.0))
    SUZ = tf.subtract(SUZ,Q0)
    Q1 = tf.multiply(parK1, SUZ)
    SUZ = tf.subtract(SUZ, Q1)
    SLZ = tf.add(SLZ, PERC)
    Q2 = tf.multiply(parK2, SLZ)
    SLZ =tf.subtract(SLZ, Q2)
    Qsim = tf.add(Q0, tf.add(Q1,Q2))
    
    # print(Qsim)
    loss = K.mean(K.abs(tf.subtract(G_obs,Qsim)),axis=-1)
    return loss
    
#%%
import os
os.chdir(r"D:\important\research\research_use_function")
from multidimensional_reshape import multi_input_output

if __name__ == '__main__':
    
    timestep=3
    preprocessing_module=multi_input_output(G,input_timestep=timestep,output_timestep=timestep)
    G_multi_input=preprocessing_module.generate_input()
    G_multi_output=preprocessing_module.generate_output()
    
    model=DNN_model(forecast_timestep,G_input,P_input,T_input,ETpot_input)
    learning_rate=1e-4
    adam = Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam,loss=HBV_error_function)
    earlystopper = EarlyStopping(monitor='val_loss', patience=15, verbose=0)        
    save_path=r"D:\important\research\groundwater_forecast\python_code\puyun\model\dlstm.hdf5"
    checkpoint =ModelCheckpoint(save_path,save_best_only=True)
    callback_list=[earlystopper,checkpoint]        
    model.fit([G_input,P_input,T_input,ETpot_input], G_output, epochs=100, batch_size=1,validation_split=0.2,callbacks=callback_list,shuffle=True)
    
    
    
    