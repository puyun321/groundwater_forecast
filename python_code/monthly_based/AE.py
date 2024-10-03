# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 17:53:22 2023

@author: Steve
"""

"""change woking directory to current directory"""
import os

working= os.path.dirname(os.path.abspath('__file__')) #try run this if work

path=r'D:\lab\research\groundwater_estimation\github\python_code\monthly_based' #if the code above cant work, please key in your parent directory
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
G01_station_info = pd.DataFrame(np.squeeze(np.array(np.concatenate([G0_station_info_new,G1_station_info_new]))))
# G01_station_info = pd.DataFrame(np.squeeze(np.array([G_station_info.iloc[G_station_info[G_station_info.iloc[:,0]==G01_station[i,0]].index,:] for i in range(0,len(G01_station))])))
G01_station_info.columns = ['station_name','X','Y']

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

## Regional Groundwater
G_z = [IDW_interpolation(np.squeeze(G01_new[i, :]),G01_station_info,G_grid_lon,G_grid_lat) for i in range(0,len(G01_new))]

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
    all_info.append(get_specific_coordinate(G01_station_info.iloc[i,1],G01_station_info.iloc[i,2],X,Y))

all_coordinate=[];min_index=[];
all_coordinate=np.squeeze(np.array([all_info[i][0] for i in range(0,len(all_info))]))
## grid location of each well
min_index=np.squeeze(np.array([all_info[i][1] for i in range(0,len(all_info))])) 

#%%
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense,LSTM,Conv2D,Flatten,Concatenate,Lambda,Layer,MaxPooling2D,UpSampling2D,Reshape
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

""" Define DNN model """
            
def AE_model(input_shape):
    
    inputs = Input(shape=(input_shape[1],input_shape[2]))
    encoder = Lambda(lambda x: tf.expand_dims(x, -1))(inputs)
    encoder = Conv2D(filters=32, kernel_size=3, activation='relu', padding="same")(encoder)
    encoder = MaxPooling2D((2, 2), padding="same")(encoder)
    encoder = Conv2D(filters=64, kernel_size=3, activation='relu', padding="same")(encoder)
    encoder = MaxPooling2D((2, 2), padding="same")(encoder)
    encoder = Flatten()(encoder)
    encoder = Dense(10)(encoder)

    
    decoder = Dense(units=25*25*32, activation='relu')(encoder)
    decoder = Reshape(target_shape=(25,25,32))(decoder)
    decoder= Conv2D(filters=64, kernel_size=3, activation='relu', padding="same")(decoder)
    decoder = UpSampling2D((2, 2))(decoder)
    decoder= Conv2D(filters=32, kernel_size=3, activation='relu', padding="same")(decoder)
    decoder = UpSampling2D((2, 2))(decoder)
    decoder = Conv2D(1, (2, 2), padding='same')(decoder)
    model = Model(inputs=inputs, outputs=decoder)

    print(model.summary())
    return model 

def DNN_model(timestep):
    
    inputs1 = Input(shape=(timestep-1,))
    inputs2 = Input(shape=(3,))
    final_output = Concatenate(axis=-1)([inputs1,inputs2])

    
    final_output = Dense(18, activation='linear',kernel_regularizer=regularizers.L2(l2=1e-2))(final_output)
    final_output = Dense(9, activation='linear',kernel_regularizer=regularizers.L2(l2=1e-2))(final_output)
    final_output = Dense(3, activation='linear',kernel_regularizer=regularizers.L2(l2=1e-2))(final_output)

    model = Model(inputs=[inputs1,inputs2], outputs=final_output)

    print(model.summary())
    return model 

def DLSTM_model(timestep):
    
    inputs1 = Input(batch_input_shape=(1,timestep-1,))
    inputs2 = Input(batch_input_shape=(1,3,))
    inputs3 = Input(batch_input_shape=(1,timestep-1,10))
    inputs4 = Input(batch_input_shape=(1,3,10))
    
    output = Concatenate(axis=-1)([inputs1,inputs2])
    output = Lambda(lambda x: tf.expand_dims(x, -1))(output)
    output = LSTM(12,stateful=True,return_sequences=True)(output)
    output = LSTM(12,stateful=True,return_sequences=False)(output)
    
    output2 = Concatenate(axis=1)([inputs3,inputs4])
    output2 = LSTM(12,stateful=True,return_sequences=True)(output2)
    output2 = LSTM(12,stateful=True,return_sequences=False)(output2)    
    
    final_output = Concatenate(axis=-1)([output,output2])
    # final_output = Dense(6, activation='linear',kernel_regularizer=regularizers.L2(l2=1e-2))(final_output)
    final_output = Dense(3, activation='linear',kernel_regularizer=regularizers.L2(l2=1e-2))(final_output)
    # final_output = Dense(3, activation='linear')(final_output)    

    model = Model(inputs=[inputs1,inputs2,inputs3,inputs4], outputs=final_output)

    print(model.summary())
    return model 


#%%
import os
os.chdir(r"D:\lab\research\research_use_function")
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
preprocessing_module=multi_input_output(pd.DataFrame(G01_new),input_timestep=timestep,output_timestep=timestep)
G_multi_input_origin=preprocessing_module.generate_input()
G_multi_output_origin=preprocessing_module.generate_output()

## split train test
forecast_timestep=np.array([timestep/3,timestep*2/3,timestep]).astype(int)
date=pd.DataFrame(G0.iloc[timestep+forecast_timestep[0]:-(timestep-forecast_timestep[0])].index)
train_index = date[date.iloc[:,0]<"2017-01-01"].index; test_index = date[date.iloc[:,0]>="2017-01-01"].index

G_train_input = G_multi_input_origin[train_index,1:,:] # select t-1, t timestep
G_train_output = G_multi_output_origin[train_index,:]

G_test_input = G_multi_input_origin[test_index,:,:] # select t-1, t timestep
G_test_output = G_multi_output_origin[test_index,:]

# Regional groundwater
G_z_train = [G_z[train_index[i]+timestep] for i in range(0,len(train_index))]
G_z_test = [G_z[test_index[i]-1] for i in range(0,len(test_index))]

G_z1_train = [G_z[train_index[i]+timestep-1] for i in range(0,len(train_index))]
G_z1_test = [G_z[test_index[i]-2] for i in range(0,len(test_index))]

#%%
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
simulate_train.append(np.array(pd.read_excel(r"result\1st_layer\sorted_predict.xlsx",sheet_name="t+1(shuffle)",index_col=0))[1:])
simulate_train.append(np.array(pd.read_excel(r"result\1st_layer\sorted_predict.xlsx",sheet_name="t+2(shuffle)",index_col=0))[1:])
simulate_train.append(np.array(pd.read_excel(r"result\1st_layer\sorted_predict.xlsx",sheet_name="t+3(shuffle)",index_col=0))[1:])

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

## Regional Groundwater Forecast
simulate_train_z = [[IDW_interpolation(np.squeeze(simulate_merge_train[i,j,:]),G01_station_info,G_grid_lon,G_grid_lat) for j in range(0,3)] for i in range(0,len(simulate_merge_train))]

simulate_merge_test=simulate_merge(simulate_test)
simulate_test = np.concatenate([simulate_merge_test[:,:,i] for i in range(0,len(simulate_merge_test[0,0,:]))])

## Regional Groundwater Forecast
simulate_test_z = [[IDW_interpolation(np.squeeze(simulate_merge_test[i,j,:]),G01_station_info,G_grid_lon,G_grid_lat) for j in range(0,3)] for i in range(0,len(simulate_merge_test))]

#%%
""" autoencoder input""" 
os.chdir(path)

model_ae_input_1 = G_z_train
model_ae_input_2 = [simulate_train_z[i][j] for i in range(0,len(simulate_train_z)) for j in range(0,3)]
model_ae_train_input = np.concatenate([model_ae_input_1,model_ae_input_2])

model_ae_input_3 = G_z_test
model_ae_input_4 = [simulate_test_z[i][j] for i in range(0,len(simulate_test_z)) for j in range(0,3)]
model_ae_test_input = np.concatenate([model_ae_input_3,model_ae_input_4])

#%%
save_path=r"model\1st_layer\autoencoder.hdf5"
ae_model=load_model(save_path) 

""" autoencoder model estimation """ 
#training#
pred_ae_train=np.squeeze(ae_model.predict(model_ae_train_input, batch_size=128))
pred_ae_train_flatten=pred_ae_train.reshape((pred_ae_train.shape[0]*pred_ae_train.shape[1]*pred_ae_train.shape[2]))
model_ae_train_input_flatten=model_ae_train_input.reshape((model_ae_train_input.shape[0]*model_ae_train_input.shape[1]*model_ae_train_input.shape[2]))

#testing#
pred_ae_test=np.squeeze(ae_model.predict(model_ae_test_input, batch_size=128))
pred_ae_test_flatten=pred_ae_test.reshape((pred_ae_test.shape[0]*pred_ae_test.shape[1]*pred_ae_test.shape[2]))
model_ae_test_input_flatten=model_ae_test_input.reshape((model_ae_test_input.shape[0]*model_ae_test_input.shape[1]*model_ae_test_input.shape[2]))

#%%
""" model performance evaluation """ 
train_ae_R2=error_indicator.np_R2(model_ae_train_input_flatten,pred_ae_train_flatten)
train_ae_rmse=error_indicator.np_RMSE(model_ae_train_input_flatten,pred_ae_train_flatten)
test_ae_R2=error_indicator.np_R2(model_ae_test_input_flatten,pred_ae_test_flatten)
test_ae_rmse=error_indicator.np_RMSE(model_ae_test_input_flatten,pred_ae_test_flatten)

train=pd.DataFrame([train_ae_R2,train_ae_rmse])
train.index = ['train_ae_R2','train_rmse']
test=pd.DataFrame([test_ae_R2,test_ae_rmse])
test.index = ['test_ae_R2','test_rmse']
ae_performance = pd.concat([train,test],axis=0)

#%%
""" model performance for each station """ 
station_ae_train_output=pd.DataFrame(np.squeeze(np.array([[model_ae_train_input[:,min_index[i,0],min_index[i,1]]] for i in range(len(min_index))]))).T
station_ae_train_pred=pd.DataFrame(np.squeeze(np.array([[pred_ae_train[:,min_index[i,0],min_index[i,1]]] for i in range(len(min_index))]))).T
station_ae_test_output=pd.DataFrame(np.squeeze(np.array([[model_ae_test_input[:,min_index[i,0],min_index[i,1]]] for i in range(len(min_index))]))).T
station_ae_test_pred=pd.DataFrame(np.squeeze(np.array([[pred_ae_test[:,min_index[i,0],min_index[i,1]]] for i in range(len(min_index))]))).T

overall_train_ae_R2=[error_indicator.np_R2(station_ae_train_output.iloc[:,i],station_ae_train_pred.iloc[:,i]) for i in range(len(min_index))]
overall_train_ae_rmse=[error_indicator.np_RMSE(station_ae_train_output.iloc[:,i],station_ae_train_pred.iloc[:,i]) for i in range(len(min_index))]
overall_test_ae_R2=[error_indicator.np_R2(station_ae_test_output.iloc[:,i],station_ae_test_pred.iloc[:,i]) for i in range(len(min_index))]
overall_test_ae_rmse=[error_indicator.np_RMSE(station_ae_test_output.iloc[:,i],station_ae_test_pred.iloc[:,i]) for i in range(len(min_index))]

overall_train_ae_performance=pd.DataFrame([overall_train_ae_R2,overall_train_ae_rmse]).T
overall_test_ae_performance=pd.DataFrame([overall_test_ae_R2,overall_test_ae_rmse]).T


train_length=len(station_ae_train_output); test_length=len(station_ae_test_output)
all_horizon_performance=[]
for timestep in range(0,4):
    train_initial=int(train_length*timestep/4); train_end=int(train_length*(timestep+1)/4)
    test_initial=int(test_length*timestep/4); test_end=int(test_length*(timestep+1)/4)
   
    station_train_ae_R2=[error_indicator.np_R2(station_ae_train_output.iloc[train_initial:train_end,i],station_ae_train_pred.iloc[train_initial:train_end,i]) for i in range(len(min_index))]
    station_train_ae_rmse=[error_indicator.np_RMSE(station_ae_train_output.iloc[train_initial:train_end,i],station_ae_train_pred.iloc[train_initial:train_end,i]) for i in range(len(min_index))]
    station_test_ae_R2=[error_indicator.np_R2(station_ae_test_output.iloc[test_initial:test_end,i],station_ae_test_pred.iloc[test_initial:test_end,i]) for i in range(len(min_index))]
    station_test_ae_rmse=[error_indicator.np_RMSE(station_ae_test_output.iloc[test_initial:test_end,i],station_ae_test_pred.iloc[test_initial:test_end,i]) for i in range(len(min_index))]
    
    station_train_ae_performance=pd.DataFrame([station_train_ae_R2,station_train_ae_rmse]).T
    station_test_ae_performance=pd.DataFrame([station_test_ae_R2,station_test_ae_rmse]).T
    
    performance=pd.concat([station_train_ae_performance,station_test_ae_performance],axis=1)
    performance.columns=['train_R2','train_RMSE','test_R2','test_RMSE']
    all_horizon_performance.append(performance)

#%%
train_date=date.iloc[:int(train_length/4),0]; 
test_date=date.iloc[int(train_length/4):,0];

train_name=[]; test_name=[]
for timestep in range(0,4):
    train_name.append(['T+%s_%s'%(timestep,train_date.iloc[j]) for j in range(len(train_date))])
    test_name.append(['T+%s_%s'%(timestep,test_date.iloc[j]) for j in range(len(test_date))])

train_name=np.concatenate(train_name);
test_name=np.concatenate(test_name)

#%% Plot
""" Kriging plot """ 
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import shapefile
import os
import gc

def shp_clip(originfig, ax, shpfile):
    sf = shapefile.Reader(shpfile)
    vertices = []
    codes = []
    for shape_rec in sf.shapeRecords():
        pts = shape_rec.shape.points
        prt = list(shape_rec.shape.parts) + [len(pts)]
        for i in range(len(prt) - 1):
            for j in range(prt[i], prt[i + 1]):
                vertices.append((pts[j][0], pts[j][1]))
            codes += [Path.MOVETO]                               # MOVETO 表示多邊形的開始點
            codes += [Path.LINETO] * (prt[i + 1] - prt[i] - 2)   # LINETO 表示創建每一段直線
            codes += [Path.CLOSEPOLY]                            # CLOSEPOLY 表示閉合多邊形
        clip = Path(vertices, codes)
        clip = PathPatch(clip, transform=ax.transData)
    for contour in originfig.collections:
        contour.set_clip_path(clip)
    return contour

# min_value = -10
# max_value = 200
min_value = -30; max_value = 200
# 定義顏色級距和顏色映射
split_ratio = 1
bounds = [i/split_ratio for i in range(min_value*split_ratio,max_value*split_ratio+1,1)]  # 色階級距的邊界值
# bounds = np.concatenate([bounds,[i for i in range(max_value+10,200,10)]])
# 創建顏色映射
cmap = get_cmap("Blues")

# 創建邊界規範
norm = BoundaryNorm(bounds, cmap.N)
plot_index=0
for plot_index in range(len(pred_ae_test)):
    predictions_array = pred_ae_test[plot_index,:,:]
    obs_array = model_ae_test_input[plot_index,:,:]
    predictions_array_flatten=predictions_array.reshape((predictions_array.shape[0]*predictions_array.shape[1]))
    obs_array_flatten=obs_array.reshape((obs_array.shape[0]*obs_array.shape[1]))
    
    filename= test_name[plot_index]
    
    fig = plt.figure(figsize=(16, 9),dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([120.05, 121.4, 23.4, 24.25], crs=ccrs.PlateCarree())
    
    
    shp_file = shpreader.Reader(r'zhuoshui(shp)\zhuoshui.shp')
    ax.add_geometries(shp_file.geometries(), crs=ccrs.PlateCarree(), linewidths=1.2, edgecolor='k', facecolor='none') # edgecolor=調整邊框顏色
    cf = ax.contourf(X, Y, predictions_array, levels=bounds, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    
    shp_clip(cf, ax, r'zhuoshui(shp)\zhuoshui.shp')
    
    plt.rcParams['font.sans-serif'] =['Taipei Sans TC Beta']  #匯入中文字體
    cb = plt.colorbar(cf)
    lons = np.array(G_station_info.loc[:,'X']); lats = G_station_info.loc[:,'Y']; station_name = G_station_info.iloc[:,0]
    # plt.scatter(lons, lats, color='navy', s=35)  # s是調整座標點大小
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    for i in range(len(lons)):        
        # plt.text(lons[i]+0.05,lats[i]+0.02,station_name[i], color='k', fontsize=12)  # 將測站名稱標示在座標點上
        plt.axis('off')  # 去除圖片外框
        
    ae_R2=error_indicator.np_R2(obs_array_flatten,predictions_array_flatten)
    ae_rmse=error_indicator.np_RMSE(obs_array_flatten,predictions_array_flatten)
    
    
    plt.title('prediction time %s (RMSE=%s , R2=%s)'%(filename,ae_rmse,ae_R2), fontsize=20)
        # output_path=r'important\research\groundwater_forecast\python_code'
    output_path=r'result\1st_layer\kriging ae figure(origin)\test\pred'
    isExist = os.path.exists(output_path)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(output_path)
      print("The new directory is created!")
    plt.savefig(os.path.join(output_path+'\\%s.png'%str(filename)[:14]), bbox_inches='tight')  # bbox_inches是修改成窄邊框
    plt.close()
    gc.collect()  # 清理站存記
    
    
    fig = plt.figure(figsize=(16, 9),dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([120.05, 121.4, 23.4, 24.25], crs=ccrs.PlateCarree())
    
    
    shp_file = shpreader.Reader(r'zhuoshui(shp)\zhuoshui.shp')
    ax.add_geometries(shp_file.geometries(), crs=ccrs.PlateCarree(), linewidths=1.2, edgecolor='k', facecolor='none') # edgecolor=調整邊框顏色
    cf = ax.contourf(X, Y, obs_array, levels=bounds, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    
    shp_clip(cf, ax, r'zhuoshui(shp)\zhuoshui.shp')
    
    plt.rcParams['font.sans-serif'] =['Taipei Sans TC Beta']  #匯入中文字體
    cb = plt.colorbar(cf)
    lons = np.array(G_station_info.loc[:,'X']); lats = G_station_info.loc[:,'Y']; station_name = G_station_info.iloc[:,0]
    # plt.scatter(lons, lats, color='navy', s=35)  # s是調整座標點大小
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    for i in range(len(lons)):        
        # plt.text(lons[i]+0.05,lats[i]+0.02,station_name[i], color='k', fontsize=12)  # 將測站名稱標示在座標點上
        plt.axis('off')  # 去除圖片外框
        
    ae_R2=error_indicator.np_R2(obs_array_flatten,predictions_array_flatten)
    ae_rmse=error_indicator.np_RMSE(obs_array_flatten,predictions_array_flatten)
    
    
    plt.title('prediction time %s (RMSE=%s , R2=%s)'%(filename,ae_rmse,ae_R2), fontsize=20)
        # output_path=r'important\research\groundwater_forecast\python_code'
    output_path=r'result\1st_layer\kriging ae figure(origin)\test\obs'
    isExist = os.path.exists(output_path)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(output_path)
      print("The new directory is created!")
    plt.savefig(os.path.join(output_path+'\\%s.png'%str(filename)[:14]), bbox_inches='tight')  # bbox_inches是修改成窄邊框
    plt.close()
    gc.collect()  # 清理站存記

#%%

train_date=date.iloc[:int(train_length/4),0]; train_date=pd.concat([train_date for i in range(4)]).reset_index(drop=True)
test_date=date.iloc[int(train_length/4):,0]; test_date=pd.concat([test_date for i in range(4)]).reset_index(drop=True)

""" month to season"""

season_datetime=['2017-04-30','2017-07-31','2017-10-31','2018-01-31','2018-04-30','2018-07-31','2018-10-31','2019-01-31','2019-04-30','2019-07-31','2019-10-31']
season_name=['2017 spring','2017 summer','2017 autumn','2018 winter','2018 spring','2018 summer','2018 autmn','2019 winter','2019 spring','2019 summer','2019 autumn']

""" plot """ 
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import shapefile
import os
import gc

def shp_clip(originfig, ax, shpfile):
    sf = shapefile.Reader(shpfile)
    vertices = []
    codes = []
    for shape_rec in sf.shapeRecords():
        pts = shape_rec.shape.points
        prt = list(shape_rec.shape.parts) + [len(pts)]
        for i in range(len(prt) - 1):
            for j in range(prt[i], prt[i + 1]):
                vertices.append((pts[j][0], pts[j][1]))
            codes += [Path.MOVETO]                               # MOVETO 表示多邊形的開始點
            codes += [Path.LINETO] * (prt[i + 1] - prt[i] - 2)   # LINETO 表示創建每一段直線
            codes += [Path.CLOSEPOLY]                            # CLOSEPOLY 表示閉合多邊形
        clip = Path(vertices, codes)
        clip = PathPatch(clip, transform=ax.transData)
    for contour in originfig.collections:
        contour.set_clip_path(clip)
    return contour


shp_file = shpreader.Reader(r'zhuoshui(shp)\zhuoshui.shp')

for season in range(0,len(season_datetime)):
    if season==0:
        choose_index=test_date[test_date<=season_datetime[season]].index
    else:
        choose_index=test_date[(test_date<=season_datetime[season]) & (test_date<=season_datetime[season-1])].index
        
    predictions_array = np.mean(pred_ae_test[choose_index,:,:],axis=0)
    obs_array = np.mean(model_ae_test_input[choose_index,:,:],axis=0)
    predictions_array_flatten=predictions_array.reshape((predictions_array.shape[0]*predictions_array.shape[1]))
    obs_array_flatten=obs_array.reshape((obs_array.shape[0]*obs_array.shape[1]))

    filename= season_name[season]
    
    """ plot obs""" 
    min_value = -30; max_value = 30
    # 定義顏色級距和顏色映射
    split_ratio = 3
    bounds = [i/split_ratio for i in range(min_value*split_ratio,max_value*split_ratio+1,1)]  # 色階級距的邊界值
    bounds = np.concatenate([bounds,[i for i in range(max_value+10,200,10)]])
    # 創建顏色映射
    cmap = get_cmap("jet_r")
    
    # 創建邊界規範
    norm = BoundaryNorm(bounds, cmap.N)

    
    fig = plt.figure(figsize=(16, 9),dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([120.05, 121.4, 23.4, 24.25], crs=ccrs.PlateCarree())
    
    ax.add_geometries(shp_file.geometries(), crs=ccrs.PlateCarree(), linewidths=1.2, edgecolor='k', facecolor='none') # edgecolor=調整邊框顏色
    # cf = ax.contourf(X, Y, obs_array, levels=np.linspace(min_value,max_value,100), cmap='jet_r', transform=ccrs.PlateCarree())
    cf = ax.contourf(X, Y, obs_array, levels=bounds, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

    shp_clip(cf, ax, r'zhuoshui(shp)\zhuoshui.shp')

    plt.rcParams['font.sans-serif'] =['Taipei Sans TC Beta']  #匯入中文字體
    cb = plt.colorbar(cf)
    lons = np.array(G_station_info.loc[:,'X']); lats = G_station_info.loc[:,'Y']; station_name = G_station_info.iloc[:,0]
    # plt.scatter(lons, lats, color='navy', s=35)  # s是調整座標點大小
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    for i in range(len(lons)):        
        # plt.text(lons[i]+0.05,lats[i]+0.02,station_name[i], color='k', fontsize=12)  # 將測站名稱標示在座標點上
        plt.axis('off')  # 去除圖片外框

    ae_R2=error_indicator.np_R2(obs_array_flatten,predictions_array_flatten)
    ae_rmse=error_indicator.np_RMSE(obs_array_flatten,predictions_array_flatten)
    
    
    plt.title('obs time %s'%(filename), fontsize=20)
        # output_path=r'important\research\groundwater_forecast\python_code'
    output_path=r'result\1st_layer\kriging ae figure(seasonal)\test\obs'
    isExist = os.path.exists(output_path)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(output_path)
      print("The new directory is created!")
    plt.savefig(os.path.join(output_path+'\\%s.png'%filename), bbox_inches='tight')  # bbox_inches是修改成窄邊框
    plt.close()
    gc.collect()  # 清理站存記
    
    """ plot pred""" 
    fig = plt.figure(figsize=(16, 9),dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([120.05, 121.4, 23.4, 24.25], crs=ccrs.PlateCarree())
    ax.add_geometries(shp_file.geometries(), crs=ccrs.PlateCarree(), linewidths=1.2, edgecolor='k', facecolor='none') # edgecolor=調整邊框顏色

    # cf = ax.contourf(X, Y, predictions_array, levels=np.linspace(min_value,max_value,100), cmap='jet_r', transform=ccrs.PlateCarree())
    cf = ax.contourf(X, Y, predictions_array, levels=bounds, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

    shp_clip(cf, ax, r'zhuoshui(shp)\zhuoshui.shp')

    plt.rcParams['font.sans-serif'] =['Taipei Sans TC Beta']  #匯入中文字體
    cb = plt.colorbar(cf)
    lons = np.array(G_station_info.loc[:,'X']); lats = G_station_info.loc[:,'Y']; station_name = G_station_info.iloc[:,0]
    # plt.scatter(lons, lats, color='navy', s=35)  # s是調整座標點大小
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    for i in range(len(lons)):        
        # plt.text(lons[i]+0.05,lats[i]+0.02,station_name[i], color='k', fontsize=12)  # 將測站名稱標示在座標點上
        plt.axis('off')  # 去除圖片外框

    ae_R2=error_indicator.np_R2(obs_array_flatten,predictions_array_flatten)
    ae_rmse=error_indicator.np_RMSE(obs_array_flatten,predictions_array_flatten)
    
    
    plt.title('prediction time %s (RMSE=%s , R2=%s)'%(filename,ae_rmse,ae_R2), fontsize=20)
        # output_path=r'important\research\groundwater_forecast\python_code'
    output_path=r'result\1st_layer\kriging ae figure(seasonal)\test\pred'
    isExist = os.path.exists(output_path)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(output_path)
      print("The new directory is created!")
    plt.savefig(os.path.join(output_path+'\\%s.png'%filename), bbox_inches='tight')  # bbox_inches是修改成窄邊框
    plt.close()
    gc.collect()  # 清理站存記
    
    """ plot difference""" 
    min_value = 0; max_value = 3
    # 定義顏色級距和顏色映射
    split_ratio = 10
    bounds = [i/split_ratio for i in range(min_value,max_value*split_ratio+1,1)]  # 色階級距的邊界值
    bounds = np.concatenate([bounds,[i for i in range(max_value+1,5)]])
    
    # 創建顏色映射
    cmap = get_cmap("Reds")
    # cmap = get_cmap("nipy_spectral")
    
    # 創建邊界規範
    norm = BoundaryNorm(bounds, cmap.N)
    # difference_array = np.mean(model_ae_test_input[choose_index,:,:]-pred_ae_test[choose_index,:,:],axis=0)
    difference_array = error_indicator.np_RMSE(model_ae_test_input[choose_index,:,:],pred_ae_test[choose_index,:,:])
    difference_array = difference_array/np.abs(np.mean(model_ae_test_input[choose_index,:,:],axis=0))
    
    fig = plt.figure(figsize=(16, 9),dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([120.05, 121.4, 23.4, 24.25], crs=ccrs.PlateCarree())
    ax.add_geometries(shp_file.geometries(), crs=ccrs.PlateCarree(), linewidths=1.2, edgecolor='k', facecolor='none') # edgecolor=調整邊框顏色

    cf = ax.contourf(X, Y, difference_array, levels=bounds, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    # Add color bar

    shp_clip(cf, ax, r'zhuoshui(shp)\zhuoshui.shp')

    plt.rcParams['font.sans-serif'] =['Taipei Sans TC Beta']  #匯入中文字體
    cb = plt.colorbar(cf)
    lons = np.array(G_station_info.loc[:,'X']); lats = G_station_info.loc[:,'Y']; station_name = G_station_info.iloc[:,0]
    # plt.scatter(lons, lats, color='navy', s=35)  # s是調整座標點大小
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    for i in range(len(lons)):        
        # plt.text(lons[i]+0.05,lats[i]+0.02,station_name[i], color='k', fontsize=12)  # 將測站名稱標示在座標點上
        plt.axis('off')  # 去除圖片外框

    ae_R2=error_indicator.np_R2(obs_array_flatten,predictions_array_flatten)
    ae_rmse=error_indicator.np_RMSE(obs_array_flatten,predictions_array_flatten)
    
    
    plt.title('prediction time %s'%(filename), fontsize=20)
        # output_path=r'important\research\groundwater_forecast\python_code'
    output_path=r'result\1st_layer\kriging ae figure(seasonal)\test\relative rmse'
    isExist = os.path.exists(output_path)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(output_path)
      print("The new directory is created!")
    plt.savefig(os.path.join(output_path+'\\%s.png'%filename), bbox_inches='tight')  # bbox_inches是修改成窄邊框
    plt.close()
    gc.collect()  # 清理站存記
    
    """ plot RMSE""" 
    min_value = 0; max_value = 15
    # 定義顏色級距和顏色映射
    split_ratio = 10
    bounds = [i/(split_ratio) for i in range(int(min_value),int(max_value*split_ratio)+1,1)]  # 色階級距的邊界值
    bounds = np.concatenate([bounds,[i for i in range(max_value+10,60,10)]])
    
    # 創建顏色映射
    cmap = get_cmap("Reds")
    # cmap = get_cmap("nipy_spectral")
    
    # 創建邊界規範
    norm = BoundaryNorm(bounds, cmap.N)
    # difference_array = np.mean(model_ae_test_input[choose_index,:,:]-pred_ae_test[choose_index,:,:],axis=0)
    difference_array = error_indicator.np_RMSE(model_ae_test_input[choose_index,:,:],pred_ae_test[choose_index,:,:])
    # difference_array = difference_array/np.abs(np.mean(model_ae_test_input[choose_index,:,:],axis=0))
    
    fig = plt.figure(figsize=(16, 9),dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([120.05, 121.4, 23.4, 24.25], crs=ccrs.PlateCarree())
    ax.add_geometries(shp_file.geometries(), crs=ccrs.PlateCarree(), linewidths=1.2, edgecolor='k', facecolor='none') # edgecolor=調整邊框顏色

    cf = ax.contourf(X, Y, difference_array, levels=bounds, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    # Add color bar

    shp_clip(cf, ax, r'zhuoshui(shp)\zhuoshui.shp')

    plt.rcParams['font.sans-serif'] =['Taipei Sans TC Beta']  #匯入中文字體
    cb = plt.colorbar(cf)
    lons = np.array(G_station_info.loc[:,'X']); lats = G_station_info.loc[:,'Y']; station_name = G_station_info.iloc[:,0]
    # plt.scatter(lons, lats, color='navy', s=35)  # s是調整座標點大小
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    for i in range(len(lons)):        
        # plt.text(lons[i]+0.05,lats[i]+0.02,station_name[i], color='k', fontsize=12)  # 將測站名稱標示在座標點上
        plt.axis('off')  # 去除圖片外框

    ae_R2=error_indicator.np_R2(obs_array_flatten,predictions_array_flatten)
    ae_rmse=error_indicator.np_RMSE(obs_array_flatten,predictions_array_flatten)
    
    
    plt.title('prediction time %s'%(filename), fontsize=20)
        # output_path=r'important\research\groundwater_forecast\python_code'
    output_path=r'result\1st_layer\kriging ae figure(seasonal)\test\RMSE'
    isExist = os.path.exists(output_path)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(output_path)
      print("The new directory is created!")
    plt.savefig(os.path.join(output_path+'\\%s.png'%filename), bbox_inches='tight')  # bbox_inches是修改成窄邊框
    plt.close()
    gc.collect()  # 清理站存記