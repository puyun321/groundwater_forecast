# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 16:54:06 2022

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
"""read simulate data"""

simulate_train=[]
simulate_train.append(np.array(pd.read_excel(r"D:\important\research\groundwater_forecast\python_code\puyun\original_level\result\1st_layer\simulate(train).xlsx",sheet_name="simulate_result(t+1)",index_col=0)))
simulate_train.append(np.array(pd.read_excel(r"D:\important\research\groundwater_forecast\python_code\puyun\original_level\result\1st_layer\simulate(train).xlsx",sheet_name="simulate_result(t+2)",index_col=0)))
simulate_train.append(np.array(pd.read_excel(r"D:\important\research\groundwater_forecast\python_code\puyun\original_level\result\1st_layer\simulate(train).xlsx",sheet_name="simulate_result(t+3)",index_col=0)))

simulate_test=[]
simulate_test.append(np.array(pd.read_excel(r"D:\important\research\groundwater_forecast\python_code\puyun\original_level\result\1st_layer\simulate(test).xlsx",sheet_name="simulate_result(t+1)",index_col=0)))
simulate_test.append(np.array(pd.read_excel(r"D:\important\research\groundwater_forecast\python_code\puyun\original_level\result\1st_layer\simulate(test).xlsx",sheet_name="simulate_result(t+2)",index_col=0)))
simulate_test.append(np.array(pd.read_excel(r"D:\important\research\groundwater_forecast\python_code\puyun\original_level\result\1st_layer\simulate(test).xlsx",sheet_name="simulate_result(t+3)",index_col=0)))


def simulate_merge(simulate):
    simulate_merge=[[]*1 for i in range(0,len(simulate[0]))]
    for i in range(0,len(simulate[0])):
        for j in range(0,3):
            simulate_merge[i].append(simulate[j][i,:])
        simulate_merge[i]=np.array(simulate_merge[i])
    simulate_merge=np.array(simulate_merge)
    return simulate_merge

simulate_merge_train=simulate_merge(simulate_train)
simulate_train = np.concatenate([simulate_merge_train[:,:,i] for i in range(0,len(simulate_merge_train[0,0,:]))])
simulate_merge_test=simulate_merge(simulate_test)
simulate_test = np.concatenate([simulate_merge_test[:,:,i] for i in range(0,len(simulate_merge_test[0,0,:]))])

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
    # output1 = Dense(3, activation='linear')(inputs1)
    # output1 = Dense(3, activation='linear')(output1)
    # output1 = Conv1D(6,kernel_size=2,padding='same')(output1)
    # output1=Flatten()(output1)  

    
    inputs2 = Input(shape=(3,))
    # output2 = Dense(3, activation='linear')(inputs2)
    # output2 = Dense(3, activation='linear')(output2)
    
    output2 = Lambda(lambda x: tf.expand_dims(x, -1))(inputs2)
    output2 = LSTM(3,stateful=False,return_sequences=True,kernel_regularizer=regularizers.L2(l2=1e-2))(output2);
    output2 = LSTM(3,stateful=False,return_sequences=False,kernel_regularizer=regularizers.L2(l2=1e-2))(output2);
    
    final_output = Concatenate(axis=-1)([output1,output2])
    final_output = Dense(6, activation='linear')(final_output)
    final_output = Dense(3, activation='linear')(final_output)


    model = Model(inputs=[inputs1,inputs2], outputs=final_output)
    # model = Model(inputs=[inputs1,inputs2], outputs=simulate_layer)

    print(model.summary())
    return model 

def DNN_model2(timestep):
    
    inputs1 = Input(shape=(timestep,))
    inputs2 = Input(shape=(3,))
    final_output = Concatenate(axis=-1)([inputs1,inputs2])
    
    final_output = Dense(6, activation='linear')(final_output)
    final_output = Dense(3, activation='linear')(final_output)


    model = Model(inputs=[inputs1,inputs2], outputs=final_output)
    # model = Model(inputs=[inputs1,inputs2], outputs=simulate_layer)

    print(model.summary())
    return model 


#%%
import os
os.chdir(r"D:\important\research\research_use_function")
from multidimensional_reshape import multi_input_output
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
    
    
    fixed_parameter_1=(pd.read_csv(r"D:\important\research\groundwater_forecast\python_code\puyun\original_level\result\1st_layer\optima_parameter(t+1).csv",index_col=0)).astype(np.float32)
    fixed_parameter_2=(pd.read_csv(r"D:\important\research\groundwater_forecast\python_code\puyun\original_level\result\1st_layer\optima_parameter(t+2).csv",index_col=0)).astype(np.float32)
    fixed_parameter_3=(pd.read_csv(r"D:\important\research\groundwater_forecast\python_code\puyun\original_level\result\1st_layer\optima_parameter(t+3).csv",index_col=0)).astype(np.float32)
    fixed_parameter = pd.concat([fixed_parameter_1,fixed_parameter_2,fixed_parameter_3],axis=1)
    timestep=9
    """ data preprocessing """ 
    P_input=P_z[:,min_index[:,0],min_index[:,1]]
    T_input=T_z[:,min_index[:,0],min_index[:,1]]
    ETpot_input=ETpot_z[:,min_index[:,0],min_index[:,1]]
    
    P_input = P_input[:-timestep]; P_input = np.array([np.sum(P_input[i-timestep:i,:],axis=0) for i in range(timestep,len(P_input)+1)])
    T_input = T_input[:-timestep]; T_input = np.array([np.sum(T_input[i-timestep:i],axis=0) for i in range(timestep,len(T_input)+1)])
    ETpot_input = ETpot_input[:-timestep]; ETpot_input = np.array([np.sum(ETpot_input[i-timestep:i],axis=0) for i in range(timestep,len(ETpot_input)+1)])
    
    preprocessing_module=multi_input_output(G01_new,input_timestep=timestep,output_timestep=timestep)
    G_multi_input=preprocessing_module.generate_input()
    G_multi_output=preprocessing_module.generate_output()
    
    G_train_input = G_multi_input[:int(len(G_multi_input)*0.8),:]
    G_train_output = G_multi_output[:int(len(G_multi_output)*0.8),:]
    
    
    shuffle_index =np.arange(len(G_train_input))
    np.random.shuffle(shuffle_index)
    
    G_train_input = np.concatenate([G_train_input[:,:,i] for i in range(0,len(G_train_input[0,0,:]))])
    G_train_output =  np.concatenate([G_train_output[:,:,i] for i in range(0,len(G_train_output[0,0,:]))])
    G_train_obs = G_train_input[:,timestep-1]

    P_train_obs = P_input[:int(len(P_input)*0.8),:];P_train_obs = np.concatenate([P_train_obs[:,i] for i in range(0,len(P_train_obs[0,:]))])
    T_train_obs = T_input[:int(len(T_input)*0.8),:];T_train_obs = np.concatenate([T_train_obs[:,i] for i in range(0,len(T_train_obs[0,:]))])
    ETpot_train_obs = ETpot_input[:int(len(ETpot_input)*0.8),:];ETpot_train_obs = np.concatenate([ETpot_train_obs[:,i] for i in range(0,len(ETpot_train_obs[0,:]))])
        
    parameter=np.array((pd.concat([fixed_parameter.iloc[i,:] for i in range(0,len(fixed_parameter)) for j in range(0,int(len(G_multi_input)*0.8))],axis=1).T).reset_index(drop=True))
    
    
    model_train_input_1=G_train_input
    model_train_input_2=simulate_train
    forecast_timestep=np.array([timestep/3,timestep*2/3,timestep]).astype(int)-1
    model_train_output=G_train_output[:,forecast_timestep]
    
    """ model construction """ 
    model=DNN_model2(timestep)
    learning_rate=1e-3
    adam = Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam,loss='mse')
    earlystopper = EarlyStopping(monitor='val_loss', patience=100, verbose=0)        
    save_path=r"D:\important\research\groundwater_forecast\python_code\puyun\original_level\model\1st_layer\hbv-lstm.hdf5"
    checkpoint =ModelCheckpoint(save_path,save_best_only=True)
    callback_list=[earlystopper,checkpoint]        
    model.fit([model_train_input_1,model_train_input_2], model_train_output, epochs=200, batch_size=32,validation_split=0.2,callbacks=callback_list,shuffle=True)
    model=load_model(save_path) 
    
    #%%
    """ model forecasting """ 
    pred_train=model.predict([model_train_input_1,model_train_input_2], batch_size=32)
    
    G_test_input = G_multi_input[int(len(G_multi_input)*0.8):,:]
    G_test_output = G_multi_output[int(len(G_multi_output)*0.8):,:]
    G_test_input = np.concatenate([G_test_input[:,:,i] for i in range(0,len(G_test_input[0,0,:]))])
    G_test_output =  np.concatenate([G_test_output[:,:,i] for i in range(0,len(G_test_output[0,0,:]))])
    G_test_obs = G_test_input[:,2]  
    
    P_test_obs = P_input[int(len(P_input)*0.8):,:];P_test_obs = np.concatenate([P_test_obs[:,i] for i in range(0,len(P_test_obs[0,:]))])
    T_test_obs = T_input[int(len(T_input)*0.8):,:];T_test_obs = np.concatenate([T_test_obs[:,i] for i in range(0,len(T_test_obs[0,:]))])
    ETpot_test_obs = ETpot_input[int(len(ETpot_input)*0.8):,:];ETpot_test_obs = np.concatenate([ETpot_test_obs[:,i] for i in range(0,len(ETpot_test_obs[0,:]))])
    test_parameter=np.array((pd.concat([fixed_parameter.iloc[i,:] for i in range(0,len(fixed_parameter)) for j in range(int(len(G_multi_input)*0.8),len(G_multi_input))],axis=1).T).reset_index(drop=True))
    model_test_input_1=G_test_input
    model_test_input_2=simulate_test

    model_test_output=G_test_output[:,forecast_timestep]
    pred_test=model.predict([model_test_input_1,model_test_input_2], batch_size=1)
    
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

    g1_station_num=G01_new.shape[1]
    pred_train_new=[];pred_test_new=[]
    for i in range(0,3):
        pred_train_new.append(convert_result(pred_train[:,i],g1_station_num))
        pred_test_new.append(convert_result(pred_test[:,i],g1_station_num))
        
#%%   
    
    train_new_R2=[[]*1 for i in range(0,3)]; train_new_rmse=[[]*1 for i in range(0,3)]; 
    test_new_R2=[[]*1 for i in range(0,3)]; test_new_rmse=[[]*1 for i in range(0,3)]
    for i in range(0,3):
        for station in range(0,g1_station_num):
            train_new_R2[i].append(error_indicator.np_R2(np.squeeze(G_multi_output[:int(len(G_multi_output)*0.8),forecast_timestep[i],station]),pred_train_new[i][:,station]))
            train_new_rmse[i].append(error_indicator.np_RMSE(np.squeeze(G_multi_output[:int(len(G_multi_output)*0.8),forecast_timestep[i],station]),pred_train_new[i][:,station]))
            test_new_R2[i].append(error_indicator.np_R2(np.squeeze(G_multi_output[int(len(G_multi_output)*0.8):,forecast_timestep[i],station]),pred_test_new[i][:,station]))
            test_new_rmse[i].append(error_indicator.np_RMSE(np.squeeze(G_multi_output[int(len(G_multi_output)*0.8):,forecast_timestep[i],station]),pred_test_new[i][:,station]))
        
    
    for i in range(0,3):
        writer = pd.ExcelWriter(r"D:\important\research\groundwater_forecast\python_code\puyun\original_level\result\1st_layer\hbvlstm-forecast(t+%s).xlsx"%(i+1),engine='xlsxwriter')
        pd.DataFrame(performance.iloc[i,:]).to_excel(writer,sheet_name="all_performance")
        pd.DataFrame(pred_train_new[i]).to_excel(writer,sheet_name="train_eachstation")
        pd.DataFrame(pred_test_new[i]).to_excel(writer,sheet_name="test_eachstation")
        pd.DataFrame(train_new_rmse[i]).to_excel(writer,sheet_name="train_rmse_eachstation")
        pd.DataFrame(test_new_rmse[i]).to_excel(writer,sheet_name="test_rmse_eachstation")                
        pd.DataFrame(train_new_R2[i]).to_excel(writer,sheet_name="train_R2_eachstation")
        pd.DataFrame(test_new_R2[i]).to_excel(writer,sheet_name="test_R2_eachstation")
        writer.save()
    
#%%
    train_new_R2=[[]*1 for i in range(0,3)]; train_new_rmse=[[]*1 for i in range(0,3)]; 
    test_new_R2=[[]*1 for i in range(0,3)]; test_new_rmse=[[]*1 for i in range(0,3)]
    for i in range(0,3):
        for station in range(0,g1_station_num):
            train_new_R2[i].append(error_indicator.np_R2(np.squeeze(G_multi_output[:int(len(G_multi_output)*0.8),forecast_timestep[i],station]),simulate_merge_train[:,i,station]))
            train_new_rmse[i].append(error_indicator.np_RMSE(np.squeeze(G_multi_output[:int(len(G_multi_output)*0.8),forecast_timestep[i],station]),simulate_merge_train[:,i,station]))
            test_new_R2[i].append(error_indicator.np_R2(np.squeeze(G_multi_output[int(len(G_multi_output)*0.8):,forecast_timestep[i],station]),simulate_merge_test[:,i,station]))
            test_new_rmse[i].append(error_indicator.np_RMSE(np.squeeze(G_multi_output[int(len(G_multi_output)*0.8):,forecast_timestep[i],station]),simulate_merge_test[:,i,station]))
        
    
    for i in range(0,3):
        writer = pd.ExcelWriter(r"D:\important\research\groundwater_forecast\python_code\puyun\original_level\result\1st_layer\simulate(t+%s).xlsx"%(i+1),engine='xlsxwriter')
        pd.DataFrame(performance.iloc[i,:]).to_excel(writer,sheet_name="all_performance")
        pd.DataFrame(G_multi_output[:int(len(G_multi_output)*0.8),forecast_timestep[i],:]).to_excel(writer,sheet_name="train_obs")
        pd.DataFrame(G_multi_output[int(len(G_multi_output)*0.8),forecast_timestep[i]:,:]).to_excel(writer,sheet_name="test_obs")
        pd.DataFrame(simulate_merge_train[:,i,:]).to_excel(writer,sheet_name="train_eachstation")
        pd.DataFrame(simulate_merge_test[:,i,:]).to_excel(writer,sheet_name="test_eachstation")
        pd.DataFrame(train_new_rmse[i]).to_excel(writer,sheet_name="train_rmse_eachstation")
        pd.DataFrame(test_new_rmse[i]).to_excel(writer,sheet_name="test_rmse_eachstation")                
        pd.DataFrame(train_new_R2[i]).to_excel(writer,sheet_name="train_R2_eachstation")
        pd.DataFrame(test_new_R2[i]).to_excel(writer,sheet_name="test_R2_eachstation")
        writer.save()
    