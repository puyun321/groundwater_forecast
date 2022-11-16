# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 15:52:33 2022

@author: steve
"""

import pandas as pd
import numpy as np

#%%
P_path = r"D:\important\research\groundwater_forecast\daily_data\rainfall.csv"
# P = pd.read_csv(P_path,index_col=0)
P = pd.read_csv(P_path);P['date'] = pd.to_datetime(P['date']);
P = P.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
P_station = P.columns
P = np.array(P)
P_station_info = pd.read_csv(r"D:\important\research\groundwater_forecast\station_info\rainfall_station.csv")

T_path = r"D:\important\research\groundwater_forecast\daily_data\temperature.csv"
# T = pd.read_csv(T_path,index_col=0)
T = pd.read_csv(T_path);T['date'] = pd.to_datetime(T['date']);
T = T.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
T_station = T.columns
T = np.array(T)
T_station_info = pd.DataFrame(np.squeeze(np.array([P_station_info.iloc[P_station_info[P_station_info.iloc[:,0]==T_station[i]].index,:] for i in range(0,len(T_station))])))
T_station_info.columns = P_station_info.columns

ETpot_path = r"D:\important\research\groundwater_forecast\daily_data\evaporation_rate.csv"
# ETpot = pd.read_csv(ETpot_path,index_col=0)
ETpot = pd.read_csv(ETpot_path);ETpot['date'] = pd.to_datetime(ETpot['date']);
ETpot = ETpot.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
ETpot_station = ETpot.columns
ETpot = np.array(ETpot)
ETpot_station_info = pd.DataFrame(np.squeeze(np.array([P_station_info.iloc[P_station_info[P_station_info.iloc[:,0]==ETpot_station[i]].index,:] for i in range(0,len(ETpot_station))])))
ETpot_station_info.columns = P_station_info.columns  

#%%
#merge stations with same name
def merge_samename(G_station,G):
    G_station=pd.Series(G_station)
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
    G = np.array([G.iloc[:,station_index[i]].max(axis=1) for i in range(0,len(station_index))]).T
    return G

def remain_onlyname(G_station):
    G_station_name=[]
    for i in range(0,len(G_station)):
        G_station_name.append(G_station[i][0:2])
    G_station_name=np.array(G_station_name)
    return G_station_name

def get_station_info(all_station_info,specific_station):
    specific_station_info = pd.DataFrame(np.squeeze(np.array([all_station_info.iloc[all_station_info[all_station_info.iloc[:,0]==specific_station[i]].index,:] for i in range(0,len(specific_station))])))
    specific_station_info.columns = all_station_info.columns
    return specific_station_info

#%%
well_info= pd.read_csv(r"D:\important\research\groundwater_forecast\station_info\well_info.txt",encoding='utf-16',delimiter='\t')
G_station_info = pd.read_csv(r"D:\important\research\groundwater_forecast\station_info\groundwater_station.csv")

G0 = pd.read_csv(r"D:\important\research\groundwater_forecast\daily_data\groundwater_l0.csv");G0['date'] = pd.to_datetime(G0['date']);
G0 = G0.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
G0_station = G0.columns
G0_station = pd.Series([G0_station[i][8:] for i in range(0,len(G0_station))])
G0_station_name = remain_onlyname(G0_station)
G0_station_info = get_station_info(G_station_info,G0_station_name)

G0_new = merge_samename(G0_station_name,G0)
G0_station_info_new = G0_station_info.drop_duplicates(keep="first").reset_index(drop=True)

G1 = pd.read_csv(r"D:\important\research\groundwater_forecast\daily_data\groundwater_l1.csv");G1['date'] = pd.to_datetime(G1['date']);
G1 = G1.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
G1_station = G1.columns
G1_station = pd.Series([G1_station[i][8:] for i in range(0,len(G1_station))])
G1_station_name = remain_onlyname(G1_station)
G1_station_info = get_station_info(G_station_info,G1_station_name)

G1_new = merge_samename(G1_station_name,G1)
G1_station_info_new = G1_station_info.drop_duplicates(keep="first").reset_index(drop=True)

G2 = pd.read_csv(r"D:\important\research\groundwater_forecast\daily_data\groundwater_l2.csv");G2['date'] = pd.to_datetime(G2['date']);
G2 = G2.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
G2_station = G2.columns
G2_station = pd.Series([G2_station[i][8:] for i in range(0,len(G2_station))])
G2_station_name = remain_onlyname(G2_station)
G2_station_info = get_station_info(G_station_info,G2_station_name)

G2_new = merge_samename(G2_station_name,G2)
G2_station_info_new = G2_station_info.drop_duplicates(keep="first").reset_index(drop=True)

G3 = pd.read_csv(r"D:\important\research\groundwater_forecast\daily_data\groundwater_l3.csv");G3['date'] = pd.to_datetime(G3['date']);
G3 = G3.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
G3_station = G3.columns
G3_station = pd.Series([G3_station[i][8:] for i in range(0,len(G3_station))])
G3_station_name = remain_onlyname(G3_station)
G3_station_info = get_station_info(G_station_info,G3_station_name)

G3_new = merge_samename(G3_station_name,G3)
G3_station_info_new = G3_station_info.drop_duplicates(keep="first").reset_index(drop=True)


G4 = pd.read_csv(r"D:\important\research\groundwater_forecast\daily_data\groundwater_l4.csv");G4['date'] = pd.to_datetime(G4['date']);
G4 = G4.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
G4_station = G4.columns
G4_station = pd.Series([G4_station[i][8:] for i in range(0,len(G4_station))])
G4_station_name = remain_onlyname(G4_station)
G4_station_info = get_station_info(G_station_info,G4_station_name)

G4_new = merge_samename(G4_station_name,G4)
G4_station_info_new = G4_station_info.drop_duplicates(keep="first").reset_index(drop=True)
        
G01_new = np.concatenate([G0_new,G1_new],axis=1)

#%%
import math


"""IDW interpolation"""
def simple_idw(x, y, z, xi, yi, power=1):
    """ Simple inverse distance weighted (IDW) interpolation 
    Weights are proportional to the inverse of the distance, so as the distance
    increases, the weights decrease rapidly.
    The rate at which the weights decrease is dependent on the value of power.
    As power increases, the weights for distant points decrease rapidly.
    """
    def distance_matrix(x0, y0, x1, y1):
        """ Make a distance matrix between pairwise observations.
        Note: from <http://stackoverflow.com/questions/1871536> 
        """
        x1, y1 = x1.flatten(), y1.flatten()
        obs = np.vstack((x0, y0)).T
        interp = np.vstack((x1, y1)).T

        d0 = np.subtract.outer(obs[:,0], interp[:,0])
        d1 = np.subtract.outer(obs[:,1], interp[:,1])
        
        # calculate hypotenuse
        # result = np.hypot(d0, d1)
        result = np.sqrt(((d0 * d0) + (d1 * d1)).astype(float))
        return result
    
    dist = distance_matrix(x,y, xi,yi)

    # In IDW, weights are 1 / distance
    weights = 1.0/(dist+1e-12)**power

    # Make weights sum to one
    weights /= weights.sum(axis=0)

    # Multiply the weights for each interpolated point by all observed Z-values
    return np.dot(weights.T, z)

def IDW_interpolation(data,station_info,grid_lon,grid_lat):
    xi, yi = np.meshgrid(grid_lon,grid_lat)
    Z = simple_idw(np.array(station_info.loc[:, 'X']),np.array(station_info.loc[:, 'Y']), data, xi, yi, power=15)
    Z = Z.reshape((xi.shape[0]),(yi.shape[0]))
    return Z


G_grid_lon = np.linspace(math.floor(min(P_station_info.loc[:, 'X'])), math.ceil(max(P_station_info.loc[:, 'X'])), 100)
G_grid_lat = np.linspace(math.floor(min(P_station_info.loc[:, 'Y'])), math.ceil(max(P_station_info.loc[:, 'Y'])), 100)
X,Y = np.meshgrid(G_grid_lon,G_grid_lat)
P_z = [IDW_interpolation(np.squeeze(P[i, :]),P_station_info,G_grid_lon,G_grid_lat) for i in range(0,len(P))]
P_z = np.nan_to_num(P_z)
T_z = np.array([IDW_interpolation(np.squeeze(T[i, :]),T_station_info,G_grid_lon,G_grid_lat) for i in range(0,len(T))])
ETpot_z = np.array([IDW_interpolation(np.squeeze(ETpot[i, :]),ETpot_station_info,G_grid_lon,G_grid_lat) for i in range(0,len(ETpot))])

#%%

G_name = pd.Series(np.concatenate([G0_station_name, G1_station_name, G2_station_name, G3_station_name, G4_station_name]))
G_unique = G_name.drop_duplicates(keep="first").reset_index(drop=True)
G_unique_station_info = pd.DataFrame(np.squeeze(np.array([G_station_info.iloc[G_station_info[G_station_info.iloc[:,0]==G_unique[i]].index,:] for i in range(0,len(G_unique))])))
G_unique_station_info.columns = G_station_info.columns 

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
for i in range(0,len(G0_station_info_new)+len(G1_station_info_new)):
# for i in range(0,len(G_unique_station_info)):

    all_info.append(get_specific_coordinate(G_unique_station_info.iloc[i,1],G_unique_station_info.iloc[i,2],X,Y))

all_coordinate=[];min_index=[];
all_coordinate=np.squeeze(np.array([all_info[i][0] for i in range(0,len(all_info))]))
min_index=np.squeeze(np.array([all_info[i][1] for i in range(0,len(all_info))]))

#%%
import tensorflow as tf
from keras import Model
from keras.engine.input_layer import Input
from keras.models import load_model
# from keras import backend as K
from keras.layers import Dense,LSTM,Conv1D,Flatten,Concatenate,Lambda,Layer
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
# from keras.layers import BatchNormalization

""" Define HBV layer """

fixed_parameter=((pd.read_csv(r"D:\important\research\groundwater_forecast\python_code\puyun\result\optima_parameter.csv",index_col=0).iloc[:,:-1]).T).astype(np.float32)

class HBV_layer(Layer):
    def __init__(self,**kwargs):
        super(HBV_layer, self).__init__(**kwargs)

    def call(self,tensor):
        def replacenan(t):
            return tf.where(tf.math.is_nan(t), tf.zeros_like(t), t)
        
        (P_obs, T_obs, ETpot_obs, G_obs_in) = tf.split(tensor, num_or_size_splits=[1,1,1,1], axis=1)

        parameters = tf.convert_to_tensor(fixed_parameter,dtype=tf.float32)
        (parPCORR,parLP,parBETA,parFC) = tf.split(parameters,num_or_size_splits=[1,1,1,1],axis=0)
        (parTT,parCFMAX,parSFCF,parCFR,parCWH) =tf.split(tf.add(tf.zeros(5, dtype=tf.float32),0.001),num_or_size_splits=5,axis=0)

        """ Initialize time series of model variables """
    
        SNOWPACK = tf.add(tf.zeros(tf.shape(parBETA), dtype=tf.float32),0.001)
        MELTWATER = tf.add(tf.zeros(tf.shape(parBETA), dtype=tf.float32),0.001)
        SM = G_obs_in
        ETact = tf.add(tf.zeros(tf.shape(parBETA), dtype=tf.float32),0.001)
        # Apply correction factor to precipitation
        P_obs =  tf.multiply(parPCORR, P_obs)
        
        """ Separate precipitation into liquid and solid components """
    
        PRECIP = tf.multiply(P_obs, parPCORR)
        RAIN = tf.where(tf.math.greater_equal(T_obs,parTT),tf.multiply(PRECIP,1), tf.multiply(PRECIP,0))
        SNOW = tf.where(tf.math.less(T_obs, parTT),tf.multiply(PRECIP,1),tf.multiply(PRECIP,0))
        SNOW = tf.multiply(SNOW,parSFCF)
        
        """ Snow """
    
        self.SNOWPACK = tf.add(self.SNOWPACK,self.SNOW)
        self.melt = tf.multiply(parCFMAX,tf.subtract(T_obs,parTT))
        self.melt = replacenan(tf.clip_by_value(self.melt,0.0, self.SNOWPACK))
        self.MELTWATER = tf.add(self.MELTWATER,self.melt)
        self.SNOWPACK = tf.subtract(self.SNOWPACK,self.melt)
        self.refreezing =  tf.multiply(parCFR, tf.multiply(parCFMAX ,tf.subtract(parTT,T_obs)))
        self.refreezing =  replacenan(tf.clip_by_value(self.refreezing,0.0, self.MELTWATER))
        self.SNOWPACK = tf.add(self.SNOWPACK,self.refreezing)
        self.MELTWATER = tf.subtract(self.MELTWATER,self.refreezing)
        self.tosoil = tf.subtract(self.MELTWATER, tf.multiply(parCWH,self.SNOWPACK))
        self.tosoil = replacenan(tf.clip_by_value(self.tosoil,0.0,self.tosoil))
        self.MELTWATER = tf.subtract(self.MELTWATER,self.tosoil)
    
        ### Caution !!! ###
        self.tosoil=0 # Taiwan did not melt snow
        
        """ Soil and evaporation """
            
        self.soil_wetness = tf.pow(tf.divide(self.SM, self.parFC), self.parBETA)
        self.soil_wetness = replacenan(tf.clip_by_value(self.soil_wetness,0.0,1.0))
        self.recharge = tf.multiply(tf.add(self.RAIN,self.tosoil),self.soil_wetness)
        self.SM = tf.subtract(tf.add(tf.add(self.SM , self.RAIN),self.tosoil),self.recharge)
        self.excess = tf.subtract(self.SM,self.parFC)
        self.excess = tf.clip_by_value(self.excess,0.0,self.excess)
        self.SM = tf.subtract(self.SM,self.excess)
        self.evapfactor = tf.divide(self.SM, tf.multiply(self.parLP, self.parFC))
        self.evapfactor = replacenan(tf.clip_by_value(self.evapfactor,0.0,1.0))
        self.ETact = tf.multiply(ETpot_obs, self.evapfactor)
        self.ETact = tf.minimum(self.SM, self.ETact)
        self.SM = tf.subtract(self.SM, self.ETact)
    
        return self.SM
    
#%%
""" Define DNN model """
            
def DNN_model_1(timestep,P_obs,T_obs,ETpot_obs,G_obs):
    inputs1 = Input(shape=(timestep,))
    output1 = Lambda(lambda x: tf.expand_dims(x, -1))(inputs1)
    output1=LSTM(36,stateful=False,return_sequences=True)(output1)
    output1=LSTM(36,stateful=False,return_sequences=False)(output1)
    output1=Flatten()(output1)
    # output_1=Dense(1, activation='linear')(output1) # 還有兩個參數還未被用到
    
    inputs2 = Input(shape=(1,))
    inputs3 = Input(shape=(1,))    
    inputs4 = Input(shape=(1,))
    inputs5 = Input(shape=(1,))
        
    output = Concatenate(axis=-1)([inputs2,inputs3,inputs4,inputs5]);
    simulate_layer = HBV_layer()(output)
    final_output = Concatenate(axis=-1)([output1,simulate_layer]);
    # final_output = Dense(32, activation='linear')(final_output)
    final_output = Dense(1, activation='linear')(final_output)

    model = Model(inputs=[inputs1,inputs2,inputs3,inputs4,inputs5], outputs=final_output)

    print(model.summary())
    return model 

#%%
P_input=P_z[:,min_index[:,0],min_index[:,1]]
T_input=T_z[:,min_index[:,0],min_index[:,1]]
ETpot_input=ETpot_z[:,min_index[:,0],min_index[:,1]]


import os
os.chdir(r"D:\important\research\research_use_function")
from multidimensional_reshape import multi_input_output

if __name__ == '__main__':
    
    # layer 0
    timestep=3
    preprocessing_module=multi_input_output(G01_new,input_timestep=timestep,output_timestep=timestep)
    G_multi_input=preprocessing_module.generate_input()
    G_multi_output=preprocessing_module.generate_output()
    
    G_train_input = G_multi_input[:int(len(G_multi_input)*0.8),:]
    G_train_output = G_multi_output[:int(len(G_multi_output)*0.8),:]
    G_train_input = np.concatenate([G_train_input[:,:,i] for i in range(0,len(G_train_input[0,0,:]))])
    G_train_output =  np.concatenate([G_train_output[:,:,i] for i in range(0,len(G_train_output[0,0,:]))])
    G_train_obs = G_train_input[:,2]
    
    P_input = P_input[timestep:-timestep]
    T_input = T_input[timestep:-timestep]    
    ETpot_input = ETpot_input[timestep:-timestep]        
    
    P_train_obs = P_input[:int(len(P_input)*0.8),:];P_train_obs = np.concatenate([P_train_obs[:,i] for i in range(0,len(P_train_obs[0,:]))])
    T_train_obs = T_input[:int(len(T_input)*0.8),:];T_train_obs = np.concatenate([T_train_obs[:,i] for i in range(0,len(T_train_obs[0,:]))])
    ETpot_train_obs = ETpot_input[:int(len(ETpot_input)*0.8),:];ETpot_train_obs = np.concatenate([ETpot_train_obs[:,i] for i in range(0,len(ETpot_train_obs[0,:]))])

    
    model=DNN_model_1(timestep,P_train_obs,T_train_obs,ETpot_train_obs,G_train_obs)
    learning_rate=1e-2
    adam = Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam,loss='mse')
    earlystopper = EarlyStopping(monitor='val_loss', patience=15, verbose=0)        
    save_path=r"D:\important\research\groundwater_forecast\python_code\puyun\model\dlstm.hdf5"
    checkpoint =ModelCheckpoint(save_path,save_best_only=True)
    callback_list=[earlystopper,checkpoint]        
    model.fit([G_train_input,P_train_obs,T_train_obs,ETpot_train_obs,G_train_obs], G_train_output, epochs=100, batch_size=1,validation_split=0.2,callbacks=callback_list,shuffle=True)
    model=load_model(save_path, custom_objects={'HBV_1st_layer': HBV_1st_layer}) 
    
    pred_train=model.predict([G_train_input,P_train_obs,T_train_obs,ETpot_train_obs,G_train_obs], batch_size=1)
    # pred_train_dernom = G_norm_module.denorm(pred_train)
    
    # pred_test=model.predict([G_test_input,P_test_input,T_test_input,ETpot_test_input], batch_size=1)
    # pred_test_dernom = G_norm_module.denorm(pred_test)    
    
