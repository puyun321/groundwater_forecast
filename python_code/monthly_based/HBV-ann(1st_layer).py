# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 16:54:06 2022

@author: steve
"""

"""change woking directory to current directory"""
import os

working= os.path.dirname(os.path.abspath('__file__')) #try run this if work

path=r'D:\important\research\groundwater_forecast\python_code\monthly_based' #if the code above cant work, please key in your parent directory
os.chdir(path)

#%%
""" Read Center Weather Bureau(CWB) Dataset """
import pandas as pd
import numpy as np

## Precipitation Dataset
P_path = r"daily_data\rainfall.csv"
P = pd.read_csv(P_path);P['date'] = pd.to_datetime(P['date']);
P = P.resample('1M',on='date',base=0,loffset='1M').mean() #convert daily data to monthly based
P_station = P.columns
P = np.array(P)
P_station_info = pd.read_csv(r"station_info\rainfall_station.csv") #get rainfall station info

## Temperature Dataset
T_path = r"daily_data\temperature.csv"
T = pd.read_csv(T_path);T['date'] = pd.to_datetime(T['date']);
T = T.resample('1M',on='date',base=0,loffset='1M').mean() #convert daily data to monthly based
T_station = T.columns
T = np.array(T)
T_station_info = pd.concat([P_station_info.iloc[P_station_info[P_station_info.iloc[:,0]==T_station[i]].index,:] for i in range(0,len(T_station))])
T_station_info.columns = P_station_info.columns #get rainfall station info

## Evaporation Dataset
ETpot_path = r"daily_data\evaporation_rate.csv"
ETpot = pd.read_csv(ETpot_path);ETpot['date'] = pd.to_datetime(ETpot['date']);
ETpot = ETpot.resample('1M',on='date',base=0,loffset='1M').mean() #convert daily data to monthly based
ETpot_station = ETpot.columns
ETpot = np.array(ETpot)
ETpot_station_info = pd.concat([P_station_info.iloc[P_station_info[P_station_info.iloc[:,0]==ETpot_station[i]].index,:] for i in range(0,len(ETpot_station))])
ETpot_station_info.columns = P_station_info.columns  

#%%
""" define function """
#merge stations with same name since there is some station has 3 wells with different height
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
    specific_station_info = pd.concat([all_station_info.iloc[all_station_info[all_station_info.iloc[:,0]==specific_station[i]].index,:] for i in range(0,len(specific_station))])
    specific_station_info.columns = all_station_info.columns
    return specific_station_info

#%%
"""Read Water Resource Agency(WRA) Groundwater Dataset"""

## Groundwater well info
G_detail_info= pd.read_csv(r"station_info\well_info.txt",encoding='utf-16',delimiter='\t')
G_station_info = pd.read_csv(r"station_info\groundwater_station.csv",delimiter='\t')

## Groundwater level 0 data
G0 = pd.read_csv(r"daily_data\groundwater_l0.csv",delimiter=',');G0['date'] = pd.to_datetime(G0['date']);
G0 = G0.resample('1M',on='date',base=0,loffset='1M').mean() #convert daily data to 10day based
G0_station = G0.columns
G0_station = pd.Series([G0_station[i][8:] for i in range(0,len(G0_station))])
G0_station_name = remain_onlyname(G0_station)
G0_station_info = get_station_info(G_station_info,G0_station_name)

G0_new = merge_samename(G0_station_name,G0)
G0_station_info_new = G0_station_info.drop_duplicates(keep="first").reset_index(drop=True)

## Groundwater level 1 data
G1 = pd.read_csv(r"daily_data\groundwater_l1.csv",delimiter=',');G1['date'] = pd.to_datetime(G1['date']);
G1 = G1.resample('1M',on='date',base=0,loffset='1M').mean() #convert daily data to 10day based
G1_station = G1.columns
G1_station = pd.Series([G1_station[i][8:] for i in range(0,len(G1_station))])
G1_station_name = remain_onlyname(G1_station)
G1_station_info = get_station_info(G_station_info,G1_station_name)

G1_new = merge_samename(G1_station_name,G1)
G1_station_info_new = G1_station_info.drop_duplicates(keep="first").reset_index(drop=True)

## Combine level 0 and 1 data
G01_new = np.concatenate([G0_new,G1_new],axis=1)

#%%
"""Other levels groundwater station (currently unused)"""
## Groundwater level 2 to 4 data 
G2 = pd.read_csv(r"daily_data\groundwater_l2.csv",delimiter=',');G2['date'] = pd.to_datetime(G2['date']);
G2 = G2.resample('1M',on='date',base=0,loffset='1M').mean() #convert daily data to 10day based
G2_station = G2.columns
G2_station = pd.Series([G2_station[i][8:] for i in range(0,len(G2_station))])
G2_station_name = remain_onlyname(G2_station)
G2_station_info = get_station_info(G_station_info,G2_station_name)

G2_new = merge_samename(G2_station_name,G2)
G2_station_info_new = G2_station_info.drop_duplicates(keep="first").reset_index(drop=True)

G3 = pd.read_csv(r"daily_data\groundwater_l3.csv",delimiter=',');G3['date'] = pd.to_datetime(G3['date']);
G3 = G3.resample('1M',on='date',base=0,loffset='1M').mean() #convert daily data to 10day based
G3_station = G3.columns
G3_station = pd.Series([G3_station[i][8:] for i in range(0,len(G3_station))])
G3_station_name = remain_onlyname(G3_station)
G3_station_info = get_station_info(G_station_info,G3_station_name)

G3_new = merge_samename(G3_station_name,G3)
G3_station_info_new = G3_station_info.drop_duplicates(keep="first").reset_index(drop=True)


G4 = pd.read_csv(r"daily_data\groundwater_l4.csv",delimiter=',');G4['date'] = pd.to_datetime(G4['date']);
G4 = G4.resample('1M',on='date',base=0,loffset='1M').mean() #convert daily data to 10day based
G4_station = G4.columns
G4_station = pd.Series([G4_station[i][8:] for i in range(0,len(G4_station))])
G4_station_name = remain_onlyname(G4_station)
G4_station_info = get_station_info(G_station_info,G4_station_name)

G4_new = merge_samename(G4_station_name,G4)
G4_station_info_new = G4_station_info.drop_duplicates(keep="first").reset_index(drop=True)

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

"""Interpolated as regional dataset"""
## gridenize data
G_grid_lon = np.linspace(math.floor(min(P_station_info.loc[:, 'X'])), math.ceil(max(P_station_info.loc[:, 'X'])), 100)
G_grid_lat = np.linspace(math.floor(min(P_station_info.loc[:, 'Y'])), math.ceil(max(P_station_info.loc[:, 'Y'])), 100)
X,Y = np.meshgrid(G_grid_lon,G_grid_lat)

## Regional Rainfall 
P_z = [IDW_interpolation(np.squeeze(P[i, :]),P_station_info,G_grid_lon,G_grid_lat) for i in range(0,len(P))]
P_z = np.nan_to_num(P_z)
## Regional Temperature
T_z = np.array([IDW_interpolation(np.squeeze(T[i, :]),T_station_info,G_grid_lon,G_grid_lat) for i in range(0,len(T))])
## Regional Evaporation 
ETpot_z = np.array([IDW_interpolation(np.squeeze(ETpot[i, :]),ETpot_station_info,G_grid_lon,G_grid_lat) for i in range(0,len(ETpot))])
    
#%%
"""Find grid location of gw well at level 0 and 1 """
G_name = pd.Series(np.concatenate([G0_station_name, G1_station_name, G2_station_name, G3_station_name, G4_station_name]))
G_unique = G_name.drop_duplicates(keep="first").reset_index(drop=True)
G_unique_station_info = pd.concat([G_station_info.iloc[G_station_info[G_station_info.iloc[:,0]==G_unique[i]].index,:] for i in range(0,len(G_unique))])
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
    all_info.append(get_specific_coordinate(G_unique_station_info.iloc[i,1],G_unique_station_info.iloc[i,2],X,Y))

all_coordinate=[];min_index=[];
all_coordinate=np.squeeze(np.array([all_info[i][0] for i in range(0,len(all_info))]))
## grid location of each well
min_index=np.squeeze(np.array([all_info[i][1] for i in range(0,len(all_info))])) 

#%%
import tensorflow as tf
from keras import regularizers
from keras import Model
from keras.engine.input_layer import Input
from keras.models import load_model
from keras import backend as K
from keras.layers import Dense,LSTM,Conv1D,Flatten,Concatenate,Lambda,Layer,Average
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

""" Define DNN model """
            
def DNN_model(timestep):
    
    inputs1 = Input(shape=(timestep,))
    output1 = Lambda(lambda x: tf.expand_dims(x, -1))(inputs1)
    output1 = LSTM(6,stateful=False,return_sequences=True)(output1)
    output1 = LSTM(6,stateful=False,return_sequences=False)(output1)
    
    inputs2 = Input(shape=(3,))
    output2 = Lambda(lambda x: tf.expand_dims(x, -1))(inputs2)
    output2 = LSTM(3,stateful=False,return_sequences=True,kernel_regularizer=regularizers.L2(l2=1e-2))(output2);
    output2 = LSTM(3,stateful=False,return_sequences=False,kernel_regularizer=regularizers.L2(l2=1e-2))(output2);
    
    final_output = Concatenate(axis=-1)([output1,output2])
    final_output = Dense(6, activation='linear')(final_output)
    final_output = Dense(3, activation='linear')(final_output)


    model = Model(inputs=[inputs1,inputs2], outputs=final_output)

    print(model.summary())
    return model 

def DNN_model2(timestep):
    
    inputs1 = Input(shape=(timestep-1,))
    inputs2 = Input(shape=(3,))
    final_output = Concatenate(axis=-1)([inputs1,inputs2])

    
    final_output = Dense(18, activation='linear',kernel_regularizer=regularizers.L2(l2=1e-2))(final_output)
    final_output = Dense(9, activation='linear',kernel_regularizer=regularizers.L2(l2=1e-2))(final_output)
    final_output = Dense(3, activation='linear',kernel_regularizer=regularizers.L2(l2=1e-2))(final_output)


    model = Model(inputs=[inputs1,inputs2], outputs=final_output)

    print(model.summary())
    return model 

#%%
import os
os.chdir(r"D:\important\research\research_use_function")
from multidimensional_reshape import multi_input_output
from error_indicator import error_indicator
os.chdir(path)

def convert_result(pred,station_num):
    converted_pred=[[]*1 for i in range(0,station_num)]
    ratio=int(len(pred)/station_num)
    for i in range(0,station_num):
        for j in range(0,ratio):
            converted_pred[i].append(pred[j+ratio*i])
    converted_pred=np.transpose(np.squeeze(np.array(converted_pred)))
    return converted_pred

def convert_shuffle_result(pred,shuffle_index,station_num):
    converted_pred=[[]*1 for i in range(0,station_num)]
    ratio=int(len(pred)/station_num)
    for i in range(0,station_num):
        for j in range(0,ratio):
            converted_pred[i].append(pred[j+ratio*i])
    converted_pred=np.transpose(np.squeeze(np.array(converted_pred)))
    return converted_pred
        
#%%    
""" Read HBV model parameters """ 
fixed_parameter_1=(pd.read_csv(r"result\1st_layer\optima_parameter(t+1).csv",index_col=0)).astype(np.float32)
fixed_parameter_2=(pd.read_csv(r"result\1st_layer\optima_parameter(t+2).csv",index_col=0)).astype(np.float32)
fixed_parameter_3=(pd.read_csv(r"result\1st_layer\optima_parameter(t+3).csv",index_col=0)).astype(np.float32)
fixed_parameter = pd.concat([fixed_parameter_1,fixed_parameter_2,fixed_parameter_3],axis=1)
timestep=3

#%%
""" data preprocessing """ 
P_input=P_z[:,min_index[:,0],min_index[:,1]]
T_input=T_z[:,min_index[:,0],min_index[:,1]]
ETpot_input=ETpot_z[:,min_index[:,0],min_index[:,1]]

P_input = P_input[:-timestep]; P_input = np.array([np.sum(P_input[i-timestep:i,:],axis=0) for i in range(timestep,len(P_input)+1)])
T_input = T_input[:-timestep]; T_input = np.array([np.sum(T_input[i-timestep:i],axis=0) for i in range(timestep,len(T_input)+1)])
ETpot_input = ETpot_input[:-timestep]; ETpot_input = np.array([np.sum(ETpot_input[i-timestep:i],axis=0) for i in range(timestep,len(ETpot_input)+1)])

preprocessing_module=multi_input_output(G01_new[1:,:],input_timestep=timestep,output_timestep=timestep)
G_multi_input_origin=preprocessing_module.generate_input()
G_multi_output_origin=preprocessing_module.generate_output()

# preprocessing_module=multi_input_output(G01_difference,input_timestep=timestep,output_timestep=timestep)
# G_multi_input=preprocessing_module.generate_input()    
# G_multi_output=preprocessing_module.generate_output()

## split train test
forecast_timestep=np.array([timestep/3,timestep*2/3,timestep]).astype(int)
date=pd.DataFrame(G0.iloc[timestep-1+forecast_timestep[0]:-(timestep-forecast_timestep[0])].index)
train_index = date[date.iloc[:,0]<"2017-01-01"].index; test_index = date[date.iloc[:,0]>="2017-01-01"].index

G_train_input = G_multi_input_origin[train_index[1:] ,1:,:] # select t-1, t timestep
# G_train_output = G_multi_output[train_index[1:] ,:]
G_train_output = G_multi_output_origin[train_index[1:] ,:]


## shuffle dataset
shuffle_index = np.array(pd.read_excel(r"result\1st_layer\sorted_predict.xlsx",sheet_name="shuffle_index",index_col=0))
shuffle_split_index = int(len(shuffle_index)*0.8)
shuffle_train_index=[]; shuffle_test_index=[]; shuffle_index_=[]
for i in range(0,len(shuffle_index)):
    if shuffle_index[i]>0:
        if i<shuffle_split_index:
            shuffle_train_index.append(shuffle_index[i]-1)
        else:
            shuffle_test_index.append(shuffle_index[i]-1)
        shuffle_index_.append(shuffle_index[i]-1)
    else:
        skip_index=i
shuffle_train_index=np.squeeze(np.array(shuffle_train_index)); shuffle_test_index=np.squeeze(np.array(shuffle_test_index)); shuffle_index_=np.squeeze(np.array(shuffle_index_))

G_shuffle_train_input = G_train_input[shuffle_train_index,:,:]
G_shuffle_train_input = np.concatenate([G_shuffle_train_input[:,:,i] for i in range(0,len(G_shuffle_train_input[0,0,:]))])
G_shuffle_train_output = G_train_output[shuffle_train_index,:,:]
G_shuffle_train_output =  np.concatenate([G_shuffle_train_output[:,:,i] for i in range(0,len(G_shuffle_train_output[0,0,:]))])

G_shuffle_test_input = G_train_input[shuffle_test_index,:,:]
G_shuffle_test_input = np.concatenate([G_shuffle_test_input[:,:,i] for i in range(0,len(G_shuffle_test_input[0,0,:]))])
G_shuffle_test_output = G_train_output[shuffle_test_index,:,:]
G_shuffle_test_output =  np.concatenate([G_shuffle_test_output[:,:,i] for i in range(0,len(G_shuffle_test_output[0,0,:]))])

#%%
"""read simulate data"""

simulate_train=[]
simulate_train.append(np.array(pd.read_excel(r"result\1st_layer\sorted_predict.xlsx",sheet_name="t+1(shuffle)",index_col=0)))
simulate_train.append(np.array(pd.read_excel(r"result\1st_layer\sorted_predict.xlsx",sheet_name="t+2(shuffle)",index_col=0)))
simulate_train.append(np.array(pd.read_excel(r"result\1st_layer\sorted_predict.xlsx",sheet_name="t+3(shuffle)",index_col=0)))

simulate_test=[]
simulate_test.append(np.array(pd.read_excel(r"result\1st_layer\sorted_predict.xlsx",sheet_name="t+1(test)",index_col=0)))
simulate_test.append(np.array(pd.read_excel(r"result\1st_layer\sorted_predict.xlsx",sheet_name="t+2(test)",index_col=0)))
simulate_test.append(np.array(pd.read_excel(r"result\1st_layer\sorted_predict.xlsx",sheet_name="t+3(test)",index_col=0)))


def simulate_merge(simulate):
    simulate_merge=[[]*1 for i in range(0,len(simulate[0]))]
    for i in range(0,len(simulate[0])):
        for j in range(0,3):
            simulate_merge[i].append(simulate[j][i,:])
        simulate_merge[i]=np.array(simulate_merge[i])
    simulate_merge=np.array(simulate_merge)
    return simulate_merge

simulate_merge_train=simulate_merge(simulate_train)

simulate_shuffle_train = simulate_merge_train[shuffle_train_index,:]
simulate_shuffle_train = np.concatenate([simulate_shuffle_train[:,:,i] for i in range(0,len(simulate_shuffle_train[0,0,:]))])

simulate_shuffle_test = simulate_merge_train[shuffle_test_index,:,:]
simulate_shuffle_test = np.concatenate([simulate_shuffle_test[:,:,i] for i in range(0,len(simulate_shuffle_test[0,0,:]))])

simulate_merge_test=simulate_merge(simulate_test)
simulate_test = np.concatenate([simulate_merge_test[:,:,i] for i in range(0,len(simulate_merge_test[0,0,:]))])

#%%
os.chdir(path)
model_train_input_1 = G_shuffle_train_input
model_train_input_2 = simulate_shuffle_train
forecast_timestep = np.array([timestep/3,timestep*2/3,timestep]).astype(int)-1
model_train_output = G_shuffle_train_output[:,forecast_timestep]

""" model construction """ 
model=DNN_model2(timestep)
learning_rate=1e-3
adam = Adam(learning_rate=learning_rate)
model.compile(optimizer=adam,loss='mse')
earlystopper = EarlyStopping(monitor='val_loss', patience=100, verbose=0)        
save_path=r"model\1st_layer\hbv-ann.hdf5"
checkpoint =ModelCheckpoint(save_path,save_best_only=True)
callback_list=[earlystopper,checkpoint]        
model.fit([model_train_input_1,model_train_input_2], model_train_output, epochs=200, batch_size=32,validation_split=0.2,callbacks=callback_list)
model=load_model(save_path) 

#%%
"""get grondwater origin"""
G_obs = G_multi_input_origin[:,timestep-1,:]
G_shuffle_train_obs, G_shuffle_test_obs = G_obs[shuffle_train_index,:], G_obs[shuffle_test_index,:]
G_test_obs = G_obs[len(shuffle_train_index)+len(shuffle_test_index):,:]
G_shuffle_train_obs =  np.concatenate([G_shuffle_train_obs[:,i] for i in range(0,len(G_shuffle_train_obs[0,:]))])
G_shuffle_test_obs =  np.concatenate([G_shuffle_test_obs[:,i] for i in range(0,len(G_shuffle_test_obs[0,:]))])
G_test_obs =  np.concatenate([G_test_obs[:,i] for i in range(0,len(G_test_obs[0,:]))])

#%%
""" model forecasting """ 
#shuffle train
pred_train=model.predict([model_train_input_1,model_train_input_2], batch_size=32)
# pred_train= np.array([pred_train[:,i]+G_shuffle_train_obs for i in range(0,len(forecast_timestep))]).T

#shuffle test
model_test_input_1 = G_shuffle_test_input
model_test_input_2 = simulate_shuffle_test
forecast_timestep = np.array([timestep/3,timestep*2/3,timestep]).astype(int)-1
model_test_output = G_shuffle_test_output[:,forecast_timestep]
pred_test=model.predict([model_test_input_1,model_test_input_2], batch_size=4)
# pred_test = np.array([pred_test[:,i]+G_shuffle_test_obs for i in range(0,len(forecast_timestep))]).T

#%%
""" convert output """  ## if output is difference only use this code
# model_train_output = np.array([model_train_output[:,i]+G_shuffle_train_obs for i in range(0,len(forecast_timestep))]).T
# model_test_output = np.array([model_test_output[:,i]+G_shuffle_test_obs for i in range(0,len(forecast_timestep))]).T

#%%
""" model performance evaluation """ 

train_R2=[];train_rmse=[]; test_R2=[];test_rmse=[]
for i in range(0,pred_train.shape[1]):
    train_R2.append(error_indicator.np_R2(model_train_output[:,i],pred_train[:,i]))
    train_rmse.append(error_indicator.np_RMSE(model_train_output[:,i],pred_train[:,i]))
    test_R2.append(error_indicator.np_R2(model_test_output[:,i],pred_test[:,i]))
    test_rmse.append(error_indicator.np_RMSE(model_test_output[:,i],pred_test[:,i]))

train_R2 = pd.DataFrame(train_R2); train_rmse = pd.DataFrame(train_rmse); train=pd.concat([train_R2,train_rmse],axis=1)
train.index = ['t+1','t+2','t+3']; train.columns = ['train_R2','train_rmse']
test_R2 = pd.DataFrame(test_R2); test_rmse = pd.DataFrame(test_rmse); test=pd.concat([test_R2,test_rmse],axis=1)
test.index = ['t+1','t+2','t+3']; test.columns = ['test_R2','test_rmse']
performance = pd.concat([train,test],axis=1)

## arrange series data into parallel data (each column is one station data)
split_index=int(0.8*len(G_train_output))
g1_station_num=G01_new.shape[1]
pred_train_new=[];pred_test_new=[];sorted_train_new=[]
for i in range(0,3):
    train=convert_result(pred_train[:,i],g1_station_num)
    test=convert_result(pred_test[:,i],g1_station_num)
    train_test=np.concatenate([train,test])
    sorted_train=[[]*1 for i in range(0,len(train)+len(test))]
    for j in range(0,len(shuffle_index_)):
        sorted_train[shuffle_index_[j]].append(train_test[j,:])
    sorted_train=np.squeeze(np.array(sorted_train))

    pred_train_new.append(convert_result(pred_train[:,i],g1_station_num))
    pred_test_new.append(convert_result(pred_test[:,i],g1_station_num))
    sorted_train_new.append(sorted_train) 

## calculate each station performance
train_new_R2=[[]*1 for i in range(0,3)]; train_new_rmse=[[]*1 for i in range(0,3)]; 
test_new_R2=[[]*1 for i in range(0,3)]; test_new_rmse=[[]*1 for i in range(0,3)]
for i in range(0,3):
    for station in range(0,g1_station_num):
    
        train_new_R2[i].append(error_indicator.np_R2(np.squeeze(G_multi_output_origin[shuffle_train_index,forecast_timestep[i],station]),pred_train_new[i][:,station]))
        train_new_rmse[i].append(error_indicator.np_RMSE(np.squeeze(G_multi_output_origin[shuffle_train_index,forecast_timestep[i],station]),pred_train_new[i][:,station]))
        test_new_R2[i].append(error_indicator.np_R2(np.squeeze(G_multi_output_origin[shuffle_test_index,forecast_timestep[i],station]),pred_test_new[i][:,station]))
        test_new_rmse[i].append(error_indicator.np_RMSE(np.squeeze(G_multi_output_origin[shuffle_test_index,forecast_timestep[i],station]),pred_test_new[i][:,station]))


for i in range(0,3):
    writer = pd.ExcelWriter(r"result\1st_layer\hbvann-forecast(t+%s).xlsx"%(i+1),engine='xlsxwriter')
    pd.DataFrame(performance.iloc[i,:]).to_excel(writer,sheet_name="all_performance")
    pd.DataFrame(pred_train_new[i]).to_excel(writer,sheet_name="train_eachstation")
    pd.DataFrame(pred_test_new[i]).to_excel(writer,sheet_name="test_eachstation")
    pd.DataFrame(train_new_rmse[i]).to_excel(writer,sheet_name="train_rmse_eachstation")
    pd.DataFrame(test_new_rmse[i]).to_excel(writer,sheet_name="test_rmse_eachstation")                
    pd.DataFrame(train_new_R2[i]).to_excel(writer,sheet_name="train_R2_eachstation")
    pd.DataFrame(test_new_R2[i]).to_excel(writer,sheet_name="test_R2_eachstation")
    writer.save()

#%%
"""Real test (Real world dataset)"""
test_index=test_index-1
G_test_input = G_multi_input_origin[test_index,1:]
G_test_output = G_multi_output_origin[test_index,:]
G_test_input = np.concatenate([G_test_input[:,:,i] for i in range(0,len(G_test_input[0,0,:]))])
G_test_output =  np.concatenate([G_test_output[:,:,i] for i in range(0,len(G_test_output[0,0,:]))])

test_parameter=np.array((pd.concat([fixed_parameter.iloc[i,:] for i in range(0,len(fixed_parameter)) for j in range(int(len(G_multi_input_origin)*0.8),len(G_multi_input_origin))],axis=1).T).reset_index(drop=True))
model_test_input_1=G_test_input
model_test_input_2=simulate_test

model_test_output=G_test_output[:,forecast_timestep]
model_test_output = np.array([model_test_output[:,i]+G_test_obs for i in range(0,len(forecast_timestep))]).T

pred_test_real=model.predict([model_test_input_1,model_test_input_2], batch_size=32)
# pred_test_real=np.array([pred_test_real[:,i]+G_test_obs for i in range(0,len(forecast_timestep))]).T

#%%
""" model performance evaluation""" 

test_R2=[];test_rmse=[]
for i in range(0,pred_test.shape[1]):
    test_R2.append(error_indicator.np_R2(model_test_output[:,i],pred_test_real[:,i]))
    test_rmse.append(error_indicator.np_RMSE(model_test_output[:,i],pred_test_real[:,i]))

test_R2 = pd.DataFrame(test_R2); test_rmse = pd.DataFrame(test_rmse); test=pd.concat([test_R2,test_rmse],axis=1)
test.index = ['t+1','t+2','t+3']; test.columns = ['test_R2','test_rmse']
performance = test

## arrange series data into parallel data (each column is one station data)
pred_test_new=[]
for i in range(0,3):
    pred_test_new.append(convert_result(pred_test_real[:,i],g1_station_num))
    
## calculate each station performance
test_new_R2=[[]*1 for i in range(0,3)]; test_new_rmse=[[]*1 for i in range(0,3)]
for i in range(0,3):
    for station in range(0,g1_station_num):
        test_new_R2[i].append(error_indicator.np_R2(np.squeeze(G_multi_output_origin[test_index,forecast_timestep[i],station]),pred_test_new[i][:,station]))
        test_new_rmse[i].append(error_indicator.np_RMSE(np.squeeze(G_multi_output_origin[test_index,forecast_timestep[i],station]),pred_test_new[i][:,station]))
    

for i in range(0,3):
    writer = pd.ExcelWriter(r"result\1st_layer\hbvann-test(t+%s).xlsx"%(i+1),engine='xlsxwriter')
    pd.DataFrame(performance.iloc[i,:]).to_excel(writer,sheet_name="all_performance")
    pd.DataFrame(pred_test_new[i]).to_excel(writer,sheet_name="test_eachstation")
    pd.DataFrame(test_new_rmse[i]).to_excel(writer,sheet_name="test_rmse_eachstation")                
    pd.DataFrame(test_new_R2[i]).to_excel(writer,sheet_name="test_R2_eachstation")
    writer.save()
    
#%%
"""sort shuffled result"""
## sort predict result
split_index=int(0.8*len(G_train_output))
g1_station_num=G01_new.shape[1]
writer = pd.ExcelWriter(r"result\1st_layer\sorted_predict(hbv-ann).xlsx",engine='xlsxwriter')
sorted_train_new=[]
for i in range(0,3):
    train=convert_result(pred_train[:,i],g1_station_num)
    test=convert_result(pred_test[:,i],g1_station_num)
    train_test=np.concatenate([train,test])
    sorted_train=[[]*1 for i in range(0,len(train)+len(test))]
    for j in range(0,len(shuffle_index_)):
        sorted_train[shuffle_index_[j]].append(train_test[j,:])
    sorted_train=np.squeeze(np.array(sorted_train))
    pd.DataFrame(sorted_train).to_excel(writer,sheet_name="t+%s(shuffle)"%(i+1))
    sorted_train_new.append(sorted_train)
    
    real_test = convert_result(pred_test_real[:,i],g1_station_num)
    pd.DataFrame(real_test).to_excel(writer,sheet_name="t+%s(test)"%(i+1))
writer.save()   

## sort observation result
writer = pd.ExcelWriter(r"result\1st_layer\sorted_observation.xlsx",engine='xlsxwriter')
# sorted_obs_new=[]
for i in range(0,3):

    sorted_obs=np.squeeze(G_train_output[:,i,:])
    pd.DataFrame(sorted_obs).to_excel(writer,sheet_name="t+%s(train)"%(i+1))
    
    real_test = convert_result(G_test_output[:,i],g1_station_num)
    pd.DataFrame(real_test).to_excel(writer,sheet_name="t+%s(test)"%(i+1))
writer.save()   