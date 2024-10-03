# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:47:28 2023

@author: Steve
"""

import os
path=r'D:\lab\research\groundwater_estimation\github\python_code\monthly_based' #if the code above cant work, please key in your parent directory
os.chdir(path)

import pandas as pd
import numpy as np


## Precipitation Dataset
P_path = r"daily_data\rainfall.csv"
P = pd.read_csv(P_path);P['date'] = pd.to_datetime(P['date']);
P = P.resample('1M',on='date',base=0,loffset='1M').sum() #convert daily data to monthly based
date=pd.DataFrame(P.index)
P_station = P.columns
P = np.array(P)
P_station_info = pd.read_csv(r"station_info\rainfall_station.csv") #get rainfall station info

#%%
path2='result\\1st_layer\\performance_comparison\\train'
# path2='result\\1st_layer\\performance_comparison\\test'
obs_train=pd.read_excel(path2+"\\T+1.xlsx",sheet_name="obs",index_col=0)
pred_train=pd.read_excel(path2+"\\T+1.xlsx",sheet_name="HBV-AE-LSTM",index_col=0)
simulate_train=(pd.read_excel(path2+"\\T+1.xlsx",sheet_name="HBV",index_col=0))
train_time=obs_train.iloc[:,0]
G_station_info = pd.read_excel(r"station_info\station_fullinfo.xlsx",sheet_name="G1",index_col=0) 

#%%
"""choose date"""
# start_date='2000-01-01';end_date='2007-12-31'
start_date='2007-01-01';end_date='2007-12-31' 
choose_date = (date[(date['date']>=start_date) & (date['date']<=end_date)].iloc[:,0]).reset_index(drop=True)
select_index= date[(date['date']>=start_date) & (date['date']<=end_date)].index

"""choose rainfall event"""
P_event = np.mean(P[select_index,:],axis=0)

"""choose groundwater recharge event"""
G_pred_event = pred_train.iloc[select_index[0],1:]-pred_train.iloc[select_index[-1],1:]
G_simulate_event = simulate_train.iloc[select_index[0],1:]-simulate_train.iloc[select_index[-1],1:]
G_obs_event = obs_train.iloc[select_index[0],1:]-obs_train.iloc[select_index[-1],1:]

#%%

import math

G_grid_lon = np.linspace(math.floor(min(P_station_info.loc[:, 'X'])), math.ceil(max(P_station_info.loc[:, 'X'])), 30)
G_grid_lat = np.linspace(math.floor(min(P_station_info.loc[:, 'Y'])), math.ceil(max(P_station_info.loc[:, 'Y'])), 30)

X,Y = np.meshgrid(G_grid_lon,G_grid_lat)


#%%
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
    Z = simple_idw(np.array(station_info.loc[:, 'X']),np.array(station_info.loc[:, 'Y']), data, xi, yi, power=5)
    Z = Z.reshape((xi.shape[0]),(yi.shape[0]))
    return Z

#%%
"""Regional rainfall event"""
P_z = IDW_interpolation(np.squeeze(P_event),P_station_info,G_grid_lon,G_grid_lat)
P_z  = np.nan_to_num(P_z )

"""Regional gw event"""
G_obs_train_z = IDW_interpolation(np.squeeze(G_obs_event),G_station_info,G_grid_lon,G_grid_lat)
G_obs_train_z = np.nan_to_num(G_obs_train_z)

G_pred_train_z = IDW_interpolation(np.squeeze(G_pred_event),G_station_info,G_grid_lon,G_grid_lat)
G_pred_train_z = np.nan_to_num(G_pred_train_z)

G_simulate_train_z = IDW_interpolation(np.squeeze(G_simulate_event),G_station_info,G_grid_lon,G_grid_lat)
G_simulate_train_z = np.nan_to_num(G_simulate_train_z)

#%%
"""Read landsubsidence event"""
# year = [2001, 2007, 2012, 2015, 2017, 2018, 2019]
year= 2007
    
"""landsubsidence"""
filename = 'landsubsidence_%s'%year
# landsubsidence_file = pd.read_csv(r"landsubsidence(shp)\%s\%s.csv"%(year,filename))
landsubsidence_file = pd.read_csv(r"landsubsidence(shp)\%s\%s.csv"%(year,filename), encoding='gbk')
landsubsidence_file.columns=['ID','land_subsidence','distance','','X','Y']
land_x=landsubsidence_file.loc[:,'X']; land_y=landsubsidence_file.loc[:,'Y']

select_index=[]
for i in range(0,len(landsubsidence_file)):
    if land_x.iloc[i]>=min(G_station_info.loc[:, 'X']) and land_x.iloc[i]<=max(G_station_info.loc[:, 'X']):
        
        if land_y.iloc[i]>=min(G_station_info.loc[:, 'Y']) and land_y.iloc[i]<=max(G_station_info.loc[:, 'Y']):
            select_index.append(i)
        
select_index=np.array(select_index)
landsubsidence_file=landsubsidence_file.iloc[select_index]
land_x=land_x.iloc[select_index]; land_y=land_y.iloc[select_index]

G_grid_lon = np.linspace(math.floor(min(P_station_info.loc[:, 'X'])), math.ceil(max(P_station_info.loc[:, 'X'])), 30)
G_grid_lat = np.linspace(math.floor(min(P_station_info.loc[:, 'Y'])), math.ceil(max(P_station_info.loc[:, 'Y'])), 30)
X,Y = np.meshgrid(G_grid_lon,G_grid_lat)

#注意xy順序
boundary = []
for j in range(1,7):
    for i in range(int(np.min(land_x)*100000/(1+j/100000)),int(np.max(land_x)*100000*(1+j/100000)),5000):
        boundary.append([0,0,0,0,0,i/100000,np.min(land_y)/(1+j/1000)])
    for i in range(int(np.min(land_x)*100000/(1+j/100000)),int(np.max(land_x)*100000*(1+j/100000)),5000):
        boundary.append([0,0,0,0,0,i/100000,np.max(land_y)*(1+j/1000)])
    for i in range(int(np.min(land_y)*100000/(1+j/100000)),int(np.max(land_y)*100000*(1+j/100000)),5000):
        boundary.append([0,0,0,0,0,np.min(land_x)/(1+j*0.5/1000),i/100000])
    for i in range(int(np.min(land_y)*100000/(1+j/100000)),int(np.max(land_y)*100000*(1+j/100000)),5000):
        boundary.append([0,0,0,0,0,np.max(land_x)*(1+j*0.5/1000),i/100000])

    
boundary=pd.DataFrame(np.array(boundary))
boundary=boundary.iloc[:,1:]
boundary.columns = landsubsidence_file.columns

landsubsidence_file = pd.concat([landsubsidence_file,boundary],axis=0)
landsubsidence_file = landsubsidence_file.reset_index(drop=True)
landsubsidence = landsubsidence_file.iloc[:,1] 
# min_value=np.min(landsubsidence); max_value=np.max(landsubsidence)
min_value=-10; max_value = 0

# land_x=landsubsidence_file.loc[:,'x']; land_y=landsubsidence_file.loc[:,'y']

landsubsidence_z = IDW_interpolation(landsubsidence_file.iloc[:,1],landsubsidence_file,G_grid_lon,G_grid_lat)

#%%
import os
os.chdir(r'D:\lab\research\research_use_function')
from error_indicator import error_indicator
os.chdir(path)
obs_flatten=np.array((landsubsidence_z).flatten()).astype(float)

P_flatten=np.array((P_z).flatten()).astype(float)
P_rmse = round(error_indicator.np_RMSE(obs_flatten,P_flatten),2)
P_r2 = round(error_indicator.np_R2(obs_flatten,P_flatten),2)

G_obs_flatten=np.array((G_obs_train_z).flatten()).astype(float)
G_obs_rmse = round(error_indicator.np_RMSE(obs_flatten,G_obs_flatten),2)
G_obs_r2 = round(error_indicator.np_R2(obs_flatten,G_obs_flatten),2)

G_simulate_flatten=np.array((G_simulate_train_z).flatten()).astype(float)
G_simulate_rmse = round(error_indicator.np_RMSE(obs_flatten,G_simulate_flatten),2)
G_simulate_r2 = round(error_indicator.np_R2(obs_flatten,G_simulate_flatten),2)

G_pred_flatten=np.array((G_pred_train_z).flatten()).astype(float)
G_pred_rmse = round(error_indicator.np_RMSE(obs_flatten,G_pred_flatten),2)
G_pred_r2 = round(error_indicator.np_R2(obs_flatten,G_pred_flatten),2)

obs_mean=round(np.mean(landsubsidence_z),2); obs_std=round(np.std(landsubsidence_z),2)


