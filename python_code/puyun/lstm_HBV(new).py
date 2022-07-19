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

#%%
import math
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from scipy.interpolate import griddata

def Ordinary_Kriging(data,station_info,grid_lon,grid_lat):
    OK = OrdinaryKriging(station_info.loc[:, 'X'],station_info.loc[:, 'Y'], data, variogram_model="linear",exact_values=True) 
    # Execute on grid:
    z1, ss1 = OK.execute("grid", grid_lon, grid_lat)
    return np.array(z1)

def Universal_Kriging(data,station_info,grid_lon,grid_lat):
    OK = UniversalKriging(station_info.loc[:, 'X'],station_info.loc[:, 'Y'], data, variogram_model="exponential",drift_terms=["regional_linear"],) 
    # Execute on grid:
    z1, ss1 = OK.execute("grid", grid_lon, grid_lat)
    return np.array(z1)

def interpolate2grid(data,station_info,grid_lon,grid_lat,interpolate_method='nearest'):
    X,Y = np.meshgrid(grid_lon,grid_lat)
    Z = np.squeeze(griddata([(x,y) for x,y in zip(station_info.loc[:, 'X'],station_info.loc[:, 'Y'])], data, (X, Y), method=interpolate_method))
    return Z

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

G0_z = np.array([IDW_interpolation(np.squeeze(G0_new[i, :]),G0_station_info_new,G_grid_lon,G_grid_lat) for i in range(0,len(G0_new))])
G1_z = np.array([IDW_interpolation(np.squeeze(G1_new[i, :]),G1_station_info_new,G_grid_lon,G_grid_lat) for i in range(0,len(G1_new))])
G2_z = np.array([IDW_interpolation(np.squeeze(G2_new[i, :]),G2_station_info_new,G_grid_lon,G_grid_lat) for i in range(0,len(G2_new))])
G3_z = np.array([IDW_interpolation(np.squeeze(G3_new[i, :]),G3_station_info_new,G_grid_lon,G_grid_lat) for i in range(0,len(G3_new))])
G4_z = np.array([IDW_interpolation(np.squeeze(G4_new[i, :]),G4_station_info_new,G_grid_lon,G_grid_lat) for i in range(0,len(G4_new))])

#%%

""" Since the Kriging fail to provide a good interpolation, we applied the IDW interpolation method to grid our data"""

P_z = [IDW_interpolation(np.squeeze(P[i, :]),P_station_info,G_grid_lon,G_grid_lat) for i in range(0,len(P))]
P_z = np.nan_to_num(P_z)
T_z = np.array([IDW_interpolation(np.squeeze(T[i, :]),T_station_info,G_grid_lon,G_grid_lat) for i in range(0,len(T))])
ETpot_z = np.array([IDW_interpolation(np.squeeze(ETpot[i, :]),ETpot_station_info,G_grid_lon,G_grid_lat) for i in range(0,len(ETpot))])
    
X,Y = np.meshgrid(G_grid_lon,G_grid_lat)

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
for i in range(0,len(G_unique_station_info)):
    all_info.append(get_specific_coordinate(G_unique_station_info.iloc[i,1],G_unique_station_info.iloc[i,2],X,Y))

all_coordinate=[];min_index=[];
all_coordinate=np.squeeze(np.array([all_info[i][0] for i in range(0,len(all_info))]))
min_index=np.squeeze(np.array([all_info[i][1] for i in range(0,len(all_info))]))

#%%
#select 49 stations dataset
G_grid=[[]*1 for i in range(0,len(min_index))]
for i in range(0,len(min_index)):
    G_grid[i].append(G0_z[:,min_index[i,0],min_index[i,1]])
    G_grid[i].append(G1_z[:,min_index[i,0],min_index[i,1]])
    G_grid[i].append(G2_z[:,min_index[i,0],min_index[i,1]])
    G_grid[i].append(G3_z[:,min_index[i,0],min_index[i,1]])
    G_grid[i].append(G4_z[:,min_index[i,0],min_index[i,1]])

G_grid=[np.array(G_grid[i]).T for i in range(0,len(G_grid))]
P_grid=P_z[:,min_index[:,0],min_index[:,1]]
T_grid=T_z[:,min_index[:,0],min_index[:,1]]
ETpot_z_grid=ETpot_z[:,min_index[:,0],min_index[:,1]]
# exclude error station
exclude_index=[]
G_mean=np.array([G_grid[i].mean(axis=0) for i in range(0,len(G_grid))])
for i in range(0,G_mean.shape[0]):
    for j in range(1,G_mean.shape[1]):
        if G_mean[i,j]<G_mean[i,j-1]:
            continue
        else:
            exclude_index.append(i)
            break

forecast_timestep=3
#select specific grid
G_select_input=[];G_select_output=[];P_select_input=[];T_select_input=[];ETpot_select_input=[];
for i in range(0,len(G_grid)):
    if i not in exclude_index:
        G_select_input.append(G_grid[i][forecast_timestep-1:-forecast_timestep-1:,:])        
        G_select_output.append(G_grid[i][forecast_timestep+forecast_timestep:,:])
        P_select_input.append(P_grid[forecast_timestep-1:-forecast_timestep-1:,i])        
        T_select_input.append(T_grid[forecast_timestep-1:-forecast_timestep-1:,i])        
        ETpot_select_input.append(ETpot_z_grid[forecast_timestep-1:-forecast_timestep-1,i])
        
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
        
        (parPCORR, P_obs, T_obs, ETpot_obs, G_obs_in) = tf.split(tensor, num_or_size_splits=[9,9,9,9,9*5], axis=1)
        
        parameters = tf.convert_to_tensor(fixed_parameter,dtype=tf.float32)
        (parBETA, parFC, parK0, parK1, parK2, parK3, parLP, parPERC0, parPERC1, parPERC2, parUZL) = tf.split(parameters,num_or_size_splits=13,axis=0)
        (parTT,parCFMAX,parSFCF,parCFR,parCWH) = tf.split(tf.add(tf.zeros(tf.shape(parBETA)*5, dtype=tf.float32),0.001),num_or_size_splits=5,axis=0)
        """ Initialize time series of model variables """
    
        SNOWPACK = tf.add(tf.zeros(tf.shape(parBETA), dtype=tf.float32),0.001)
        MELTWATER = tf.add(tf.zeros(tf.shape(parBETA), dtype=tf.float32),0.001)
        SM,SUZ,SLZ1,SLZ2,SLZ3 = tf.split(G_obs_in,num_or_size_splits=5,axis=0)
        ETact = tf.add(tf.zeros(tf.shape(parBETA), dtype=tf.float32),0.001)
        Qsim = tf.zeros(tf.shape(P_obs), dtype=tf.float32)
        # Apply correction factor to precipitation
        P_obs =  tf.multiply(parPCORR, P_obs)
        
        """ Separate precipitation into liquid and solid components """
    
        PRECIP = tf.multiply(P_obs, parPCORR)
        RAIN = tf.where(tf.math.greater_equal(T_obs,parTT),tf.multiply(PRECIP,T_obs), tf.multiply(PRECIP,0))
        SNOW = tf.where(tf.math.less(T_obs, parTT),tf.multiply(PRECIP,T_obs),tf.multiply(PRECIP,0))
        SNOW = tf.multiply(SNOW,parSFCF)
        
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
    
        """ Groundwater boxes """
       
        SUZ = tf.add(SUZ,tf.add(recharge,excess))
        PERC0  = tf.minimum(SUZ, parPERC0)
        SUZ = tf.subtract(SUZ, PERC0)
        Q0 = tf.multiply(parK0,tf.maximum(tf.subtract(SUZ,parUZL), 0.0))
        SUZ = tf.subtract(SUZ,Q0)
        
        SLZ1 = tf.add(SLZ1,PERC0)
        PERC1 = tf.minimum(SLZ1,parPERC1)
        Q1 = tf.multiply(parK1,SLZ1)
        SLZ1 =  tf.subtract(SLZ1,Q1)     
        
        SLZ2 = tf.add(SLZ2,PERC1)
        PERC2 = tf.minimum(SLZ2,parPERC2)
        Q2 = tf.multiply(parK2,SLZ2)
        SLZ2 =  tf.subtract(SLZ2,Q2)       
        
        SLZ3 = tf.add(SLZ3,PERC2)
        Q3 = tf.multiply(parK3,SLZ3)
        SLZ3 = tf.subtract(SLZ3,Q3)

        Qsim = tf.add(Q3, tf.add(tf.add(Q0,Q1),Q2))

        return SM,SUZ,SLZ1,SLZ2,SLZ3
    