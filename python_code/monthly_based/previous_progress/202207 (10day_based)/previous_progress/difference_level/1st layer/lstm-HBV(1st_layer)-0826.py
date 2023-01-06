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
"""" output difference """
G01_difference = G01_new[1:]-G01_new[:-1]
G01_base = G01_new[:-1]

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
from keras import Model,regularizers
from keras.engine.input_layer import Input
from keras.models import load_model
from keras import backend as K
from keras.layers import Dense,LSTM,Conv1D,Flatten,Concatenate,Lambda,Layer,Average
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
# from keras.layers import BatchNormalization

""" Define HBV layer """

class HBV_layer(Layer):
    def __init__(self,**kwargs):
        super(HBV_layer, self).__init__(**kwargs)

    def call(self,tensor):
        def replacenan(t):
            return tf.where(tf.math.is_nan(t), tf.zeros_like(t), t)
                
        (adjust_parameter,P_obs, T_obs, ETpot_obs, G_obs_in, fixed_parameter1,fixed_parameter2,fixed_parameter3) = tf.split(tensor, num_or_size_splits=[3,1,1,1,1,4,4,4], axis=1)
        (adjust_parameter1, adjust_parameter2, adjust_parameter3) = tf.split(adjust_parameter, num_or_size_splits=[1,1,1], axis=1)
        (parPCORR_1,parLP_1,parBETA_1,parFC_1) = tf.split(fixed_parameter1, num_or_size_splits=[1,1,1,1], axis=1)
        (parPCORR_2,parLP_2,parBETA_2,parFC_2) = tf.split(fixed_parameter2, num_or_size_splits=[1,1,1,1], axis=1)       
        (parPCORR_3,parLP_3,parBETA_3,parFC_3) = tf.split(fixed_parameter3, num_or_size_splits=[1,1,1,1], axis=1)
        # (adjust_parameter,P_obs, T_obs, ETpot_obs, G_obs_in, parPCORR,parLP,parBETA,parFC) = tf.split(tensor, num_or_size_splits=[3,1,1,1,1,1,1,1,1], axis=1)
        
        def HBV_model(parPCORR,parLP,parBETA,parFC):
            (parTT,parCFMAX,parSFCF,parCFR,parCWH) =tf.split(tf.add(tf.zeros(shape=5, dtype=tf.float32),0.001),num_or_size_splits=5,axis=0)
            parTT = tf.cast(parTT,tf.float32); parCFMAX = tf.cast(parCFMAX,tf.float32); parSFCF = tf.cast(parSFCF,tf.float32);
            parCFR = tf.cast(parCFR,tf.float32); parCWH = tf.cast(parCWH,tf.float32)
            """ Initialize time series of model variables """
        
            SNOWPACK = tf.add(tf.zeros(tf.shape(parBETA), dtype=tf.float32),0.001)
            MELTWATER = tf.add(tf.zeros(tf.shape(parBETA), dtype=tf.float32),0.001)
            # SM_ = tf.cast(G_obs_in, dtype=tf.float32)
            SM = tf.add(tf.zeros(tf.shape(parBETA), dtype=tf.float32),0.001)
            ETact = tf.add(tf.zeros(tf.shape(parBETA), dtype=tf.float32),0.001)
            # Apply correction factor to precipitation
            P_obs_new =  tf.multiply(parPCORR, P_obs)
            # adjust_parameter = tf.unstack(adjust_parameter)
            
            """ Separate precipitation into liquid and solid components """
        
            PRECIP = tf.multiply(P_obs_new, parPCORR)
            RAIN = tf.where(tf.math.greater_equal(T_obs,parTT),tf.multiply(PRECIP,1), tf.multiply(PRECIP,0))
            SNOW = tf.where(tf.math.less(T_obs, parTT),tf.multiply(PRECIP,1),tf.multiply(PRECIP,0))
            SNOW = tf.multiply(SNOW,parSFCF)
            
            """ Snow """
        
            SNOWPACK = tf.add(SNOWPACK,SNOW)
            melt = tf.multiply(parCFMAX,tf.subtract(T_obs,parTT))
            tf.shape(melt);tf.shape(SNOWPACK)
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
            soil_wetness = replacenan(tf.clip_by_value(soil_wetness,0.0,1.0))
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
            # SM = tf.multiply(SM,adjust_parameter)
            # SM_forecast = tf.add(SM,SM_)     
            return SM

        # GW_forecast_1 = HBV_model(parPCORR_1,parLP_1,parBETA_1,parFC_1)
        # GW_forecast_2 = HBV_model(parPCORR_2,parLP_2,parBETA_2,parFC_2)
        # GW_forecast_3 = HBV_model(parPCORR_3,parLP_3,parBETA_3,parFC_3)
        GW_forecast_1 = tf.multiply(HBV_model(parPCORR_1,parLP_1,parBETA_1,parFC_1),adjust_parameter1)
        GW_forecast_2 = tf.multiply(HBV_model(parPCORR_2,parLP_2,parBETA_2,parFC_2),adjust_parameter2)
        GW_forecast_3 = tf.multiply(HBV_model(parPCORR_3,parLP_3,parBETA_3,parFC_3),adjust_parameter3)
        
        GW_forecast = tf.concat([GW_forecast_1,GW_forecast_2,GW_forecast_3],axis=1)

        return GW_forecast
    
#%%
""" Define DNN model """
            
def DNN_model(timestep):
    
    inputs1 = Input(shape=(timestep,))    
    output1 = Lambda(lambda x: tf.expand_dims(x, -1))(inputs1)  
    output1 =LSTM(6,stateful=False,return_sequences=False,kernel_regularizer=regularizers.L2(l2=1e-1))(output1)
    # output1 = Flatten()(output1)
    
    inputs2 = Input(shape=(timestep,))    
    output2 = Lambda(lambda x: tf.expand_dims(x, -1))(inputs2)  
    output2 =LSTM(6,stateful=False,return_sequences=False,kernel_regularizer=regularizers.L2(l2=1e-1))(output2)
    # output2 = Flatten()(output2)
    
    inputs3 = Input(shape=(timestep,))    
    output3 = Lambda(lambda x: tf.expand_dims(x, -1))(inputs3)  
    output3 =LSTM(6,stateful=False,return_sequences=False,kernel_regularizer=regularizers.L2(l2=1e-1))(output3)
    # output3 = Flatten()(output3)
    
    concat_output = Concatenate()([output1, output2, output3])
    concat_output = Dense(3, activation='linear')(concat_output)
    
    inputs4 = Input(shape=(16,))
    concat_input = Concatenate(axis=-1)([concat_output,inputs4]);

    simulate_layer = HBV_layer()(concat_input)
    
    final_output = Concatenate(axis=-1)([concat_output,simulate_layer])
    final_output = Dense(3, activation='linear')(final_output)

    model = Model(inputs=[inputs1,inputs2,inputs3,inputs4], outputs=final_output)
    # model = Model(inputs=[inputs1,inputs2,inputs3,inputs4], outputs=simulate_layer)

    print(model.summary())
    return model 

#%%
import os
os.chdir(r"D:\important\research\research_use_function")
from multidimensional_reshape import multi_input_output
from rescale import normalization
from error_indicator import error_indicator

def convert_result(pred,station_num):
    converted_pred=[[]*1 for i in range(0,station_num)]
    ratio=int(len(pred)/station_num)
    for i in range(0,station_num):
        for j in range(0,ratio):
            converted_pred[i].append(pred[j+ratio*i])
    converted_pred=np.transpose(np.squeeze(np.array(converted_pred)))
    return converted_pred
    

if __name__ == '__main__':
    
    
    fixed_parameter_1=(pd.read_csv(r"D:\important\research\groundwater_forecast\python_code\puyun\difference_level\result\1st_layer\optima_parameter(t+1)-difference.csv",index_col=0)).astype(np.float32)
    fixed_parameter_2=(pd.read_csv(r"D:\important\research\groundwater_forecast\python_code\puyun\difference_level\result\1st_layer\optima_parameter(t+2)-difference.csv",index_col=0)).astype(np.float32)
    fixed_parameter_3=(pd.read_csv(r"D:\important\research\groundwater_forecast\python_code\puyun\difference_level\result\1st_layer\optima_parameter(t+3)-difference.csv",index_col=0)).astype(np.float32)
    fixed_parameter = pd.concat([fixed_parameter_1,fixed_parameter_2,fixed_parameter_3],axis=1)
    timestep=9
    """ data preprocessing """ 
    P_input=P_z[:,min_index[:,0],min_index[:,1]]
    T_input=T_z[:,min_index[:,0],min_index[:,1]]
    ETpot_input=ETpot_z[:,min_index[:,0],min_index[:,1]]
    T_input = T_input[1:-timestep]; T_input = np.array([np.sum(T_input[i-timestep:i],axis=0) for i in range(timestep,len(T_input)+1)])
    
    preprocessing_module=multi_input_output(P_input[1:,:],input_timestep=timestep,output_timestep=timestep)
    P_multi_input=preprocessing_module.generate_input()
    
    preprocessing_module=multi_input_output(ETpot_input[1:,:],input_timestep=timestep,output_timestep=timestep)
    ETpot_multi_input=preprocessing_module.generate_input()
    
    preprocessing_module=multi_input_output(G01_new[1:,:],input_timestep=timestep,output_timestep=timestep)
    G_multi_input_origin=preprocessing_module.generate_input()
    G_multi_output_origin=preprocessing_module.generate_output()
    
    preprocessing_module=multi_input_output(G01_difference,input_timestep=timestep,output_timestep=timestep)
    G_multi_input=preprocessing_module.generate_input()    
    G_multi_output=preprocessing_module.generate_output()
               
    """"split dataset into train and test"""
    def split_data(data):
        train = data[:int(len(data)*0.8),:]
        train = np.concatenate([train[:,:,i] for i in range(0,len(train[0,0,:]))])
        test = data[int(len(data)*0.8):,:]
        test = np.concatenate([test[:,:,i] for i in range(0,len(test[0,0,:]))])
        return train,test
    
    G_train_input, G_test_input = split_data(G_multi_input)
    G_train_obs, G_test_obs = split_data(G_multi_input_origin)
    G_train_obs, G_test_obs = G_train_obs[:,timestep-1], G_test_obs[:,timestep-1]
    P_train_input, P_test_input = split_data(P_multi_input)
    P_train_obs,P_test_obs = P_train_input[:,timestep-1],P_test_input[:,timestep-1]
    ETpot_train_input, ETpot_test_input = split_data(ETpot_multi_input)      
    ETpot_train_obs,ETpot_test_obs = ETpot_train_input[:,timestep-1],ETpot_test_input[:,timestep-1]    
    G_train_output, G_test_output = split_data(G_multi_output)
    T_train_obs = T_input[:int(len(T_input)*0.8),:];T_train_obs = np.concatenate([T_train_obs[:,i] for i in range(0,len(T_train_obs[0,:]))]) 
    T_test_obs = T_input[int(len(T_input)*0.8):,:];T_test_obs = np.concatenate([T_test_obs[:,i] for i in range(0,len(T_test_obs[0,:]))]) 
   
     
    """declare model input"""
    parameter=np.array((pd.concat([fixed_parameter.iloc[i,:] for i in range(0,len(fixed_parameter)) for j in range(0,int(len(G_multi_input)*0.8))],axis=1).T).reset_index(drop=True))
    forecast_timestep=np.array([timestep/3,timestep*2/3,timestep]).astype(int)-1
    model_train_input =np.concatenate([np.expand_dims(P_train_obs,axis=1),np.expand_dims(T_train_obs,axis=1),np.expand_dims(ETpot_train_obs,axis=1),
                                  np.expand_dims(G_train_obs,axis=1),np.array(parameter)],axis=1)
    model_train_input=[G_train_input,P_train_input,ETpot_train_input, model_train_input]
    forecast_timestep=np.array([timestep/3,timestep*2/3,timestep]).astype(int)-1
    model_train_output=G_train_output[:,forecast_timestep]
    
    """ normalization """
    # model_train_output_module=normalization(model_train_output)
    # model_train_output_norm = model_train_output_module.norm()
    
    """ model construction """ 
    model=DNN_model(timestep)
    learning_rate=1e-4
    adam = Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam,loss='mse')
    earlystopper = EarlyStopping(monitor='val_loss', patience=100, verbose=0)        
    save_path=r"D:\important\research\groundwater_forecast\python_code\puyun\model\1st_layer\dlstm-hbv.hdf5"
    checkpoint =ModelCheckpoint(save_path,save_best_only=True)
    callback_list=[earlystopper,checkpoint]        
    model.fit(model_train_input, model_train_output, epochs=100, batch_size=16,validation_split=0.2,callbacks=callback_list,shuffle=True)
    model=load_model(save_path, custom_objects={'HBV_layer': HBV_layer}) 
    
    #%%
    """ model forecasting """ 
    pred_train= model.predict(model_train_input, batch_size=16)
    pred_train= np.array([pred_train[:,i]+G_train_obs for i in range(0,len(forecast_timestep))]).T
    
    test_parameter=np.array((pd.concat([fixed_parameter.iloc[i,:] for i in range(0,len(fixed_parameter)) for j in range(int(len(G_multi_input)*0.8),len(G_multi_input))],axis=1).T).reset_index(drop=True))
    model_test_input=[G_test_input, P_test_input, ETpot_test_input, np.concatenate([np.expand_dims(P_test_obs,axis=1),np.expand_dims(T_test_obs,axis=1),np.expand_dims(ETpot_test_obs,axis=1),
                                  np.expand_dims(G_test_obs,axis=1),np.array(test_parameter)],axis=1)]

    model_test_output=G_test_output[:,forecast_timestep]
    
    """ convert output """ 
    model_train_output_new = np.array([model_train_output[:,i]+G_train_obs for i in range(0,len(forecast_timestep))]).T
    model_test_output_new = np.array([model_test_output[:,i]+G_test_obs for i in range(0,len(forecast_timestep))]).T

    pred_test=model.predict(model_test_input, batch_size=16)
    pred_test=np.array([pred_test[:,i]+G_test_obs for i in range(0,len(forecast_timestep))]).T
    
    """ model performance evaluation """ 
    
    train_R2=[];train_rmse=[]; test_R2=[];test_rmse=[]
    for i in range(0,pred_train.shape[1]):
        train_R2.append(error_indicator.np_R2(model_train_output_new[:,i],pred_train[:,i]))
        train_rmse.append(error_indicator.np_RMSE(model_train_output_new[:,i],pred_train[:,i]))
        test_R2.append(error_indicator.np_R2(model_test_output_new[:,i],pred_test[:,i]))
        test_rmse.append(error_indicator.np_RMSE(model_test_output_new[:,i],pred_test[:,i]))

    train_R2 = pd.DataFrame(train_R2); train_rmse = pd.DataFrame(train_rmse); train=pd.concat([train_R2,train_rmse],axis=1)
    train.index = ['t+1','t+2','t+3']; train.columns = ['train_R2','train_rmse']
    test_R2 = pd.DataFrame(test_R2); test_rmse = pd.DataFrame(test_rmse); test=pd.concat([test_R2,test_rmse],axis=1)
    test.index = ['t+1','t+2','t+3']; test.columns = ['test_R2','test_rmse']
    performance = pd.concat([train,test],axis=1)

    g1_station_num=G01_new.shape[1]
    pred_train_new=[];pred_test_new=[]
    for i in range(0,3):
        pred_train_new.append(convert_result(pred_train[:,i],g1_station_num))
        pred_test_new.append(convert_result(pred_test[:,i],g1_station_num))
    
    train_new_R2=[[]*1 for i in range(0,3)]; train_new_rmse=[[]*1 for i in range(0,3)]; 
    test_new_R2=[[]*1 for i in range(0,3)]; test_new_rmse=[[]*1 for i in range(0,3)]
    for i in range(0,3):
        for station in range(0,g1_station_num):
            train_new_R2[i].append(error_indicator.np_R2(np.squeeze(G_multi_output_origin[:int(len(G_multi_output_origin)*0.8),forecast_timestep[i],station]),pred_train_new[i][:,station]))
            train_new_rmse[i].append(error_indicator.np_RMSE(np.squeeze(G_multi_output_origin[:int(len(G_multi_output_origin)*0.8),forecast_timestep[i],station]),pred_train_new[i][:,station]))
            test_new_R2[i].append(error_indicator.np_R2(np.squeeze(G_multi_output_origin[int(len(G_multi_output_origin)*0.8):,forecast_timestep[i],station]),pred_test_new[i][:,station]))
            test_new_rmse[i].append(error_indicator.np_RMSE(np.squeeze(G_multi_output_origin[int(len(G_multi_output_origin)*0.8):,forecast_timestep[i],station]),pred_test_new[i][:,station]))
    
    
    for i in range(0,3):
        writer = pd.ExcelWriter(r"D:\important\research\groundwater_forecast\python_code\puyun\result\1st_layer\lstmhbv-forecast(t+%s).xlsx"%(i+1),engine='xlsxwriter')
        pd.DataFrame(performance.iloc[i,:]).to_excel(writer,sheet_name="all_performance")
        pd.DataFrame(pred_train_new[i]).to_excel(writer,sheet_name="train_eachstation")
        pd.DataFrame(pred_test_new[i]).to_excel(writer,sheet_name="test_eachstation")
        pd.DataFrame(train_new_rmse[i]).to_excel(writer,sheet_name="train_rmse_eachstation")
        pd.DataFrame(test_new_rmse[i]).to_excel(writer,sheet_name="test_rmse_eachstation")                
        pd.DataFrame(train_new_R2[i]).to_excel(writer,sheet_name="train_R2_eachstation")
        pd.DataFrame(test_new_R2[i]).to_excel(writer,sheet_name="test_R2_eachstation")
        writer.save()
    
    