# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 18:04:01 2022

@author: steve
"""

import pandas as pd
import numpy as np
import os
path=r'D:\important\research\groundwater_forecast(monthly)'
os.chdir(path)

forecast_timestep='t+3'
obs_train=pd.read_excel(r"D:\important\research\groundwater_forecast(monthly)\result\1st_layer\sorted_observation.xlsx",sheet_name="%s(train)"%forecast_timestep,index_col=0)
pred_train=pd.read_excel(r"D:\important\research\groundwater_forecast(monthly)\result\1st_layer\sorted_predict(hbv-ann).xlsx",sheet_name="%s(shuffle)"%forecast_timestep,index_col=0)
simulate_train=(pd.read_excel(r"D:\important\research\groundwater_forecast(monthly)\result\1st_layer\sorted_predict.xlsx",sheet_name="%s(shuffle)"%forecast_timestep,index_col=0)).iloc[1:,:]

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
    specific_station_info = pd.concat([all_station_info.iloc[all_station_info[all_station_info.iloc[:,0]==specific_station[i]].index,:] for i in range(0,len(specific_station))])
    specific_station_info.columns = all_station_info.columns
    return specific_station_info

#%%
G_station_info = pd.read_csv(r"D:\important\research\groundwater_forecast\station_info\groundwater_station.csv",delimiter='\t')

G0 = pd.read_csv(r"D:\important\research\groundwater_forecast\daily_data\groundwater_l0.csv",delimiter=',');G0['date'] = pd.to_datetime(G0['date']);
G0_ = G0.resample('1M',on='date',base=0,loffset='1M').mean()
# G0_ = G0.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
G0_station = G0.columns
G0_station = pd.Series([G0_station[i][8:10] for i in range(1,len(G0_station))])
G0_station = G0_station.drop_duplicates (keep='first').reset_index(drop=True)

G1 = pd.read_csv(r"D:\important\research\groundwater_forecast\daily_data\groundwater_l1.csv",delimiter=',');G1['date'] = pd.to_datetime(G1['date']);
G1_ = G1.resample('1M',on='date',base=0,loffset='1M').mean()
G1_station = G1.columns
G1_station = pd.Series([G1_station[i][8:10] for i in range(1,len(G1_station))])
G1_station = G1_station.drop_duplicates (keep='first').reset_index(drop=True)

G01_station = np.concatenate([G0_station,G1_station])

G_station_info = pd.read_csv(r"D:\important\research\groundwater_forecast\station_info\groundwater_station.csv",delimiter='\t')
G01_station_info = pd.DataFrame(np.squeeze(np.array([G_station_info.iloc[G_station_info[G_station_info.iloc[:,0]==G01_station[i]].index[0],:] for i in range(0,len(G01_station))])))
G01_station_info.columns = ['station_name','X','Y']

time = G0_.index; train_time = pd.DataFrame(time[:len(obs_train)])

#%%
P_path = r"daily_data\rainfall.csv"
# P = pd.read_csv(P_path,index_col=0)
P = pd.read_csv(P_path);P['date'] = pd.to_datetime(P['date']);
P = P.resample('1M',on='date',base=0,loffset='1M').mean() #convert daily data to 10day based
P_station = P.columns
P = np.array(P)
P_station_info = pd.read_csv(r"D:\important\research\groundwater_forecast\station_info\rainfall_station.csv")

T_path = r"daily_data\temperature.csv"
# T = pd.read_csv(T_path,index_col=0)
T = pd.read_csv(T_path);T['date'] = pd.to_datetime(T['date']);
T = T.resample('1M',on='date',base=0,loffset='1M').mean() #convert daily data to 10day based
T_station = T.columns
T = np.array(T)
T_station_info = pd.concat([P_station_info.iloc[P_station_info[P_station_info.iloc[:,0]==T_station[i]].index,:] for i in range(0,len(T_station))])
T_station_info.columns = P_station_info.columns

ETpot_path = r"daily_data\evaporation_rate.csv"
# ETpot = pd.read_csv(ETpot_path,index_col=0)
ETpot = pd.read_csv(ETpot_path);ETpot['date'] = pd.to_datetime(ETpot['date']);
ETpot = ETpot.resample('1M',on='date',base=0,loffset='1M').mean() #convert daily data to 10day based
ETpot_station = ETpot.columns
ETpot = np.array(ETpot)
ETpot_station_info = pd.concat([P_station_info.iloc[P_station_info[P_station_info.iloc[:,0]==ETpot_station[i]].index,:] for i in range(0,len(ETpot_station))])
ETpot_station_info.columns = P_station_info.columns 

#%%
"""choose date"""

#%%
import math

G_grid_lon = np.linspace(math.floor(min(P_station_info.loc[:, 'X'])), math.ceil(max(P_station_info.loc[:, 'X'])), 30)
G_grid_lat = np.linspace(math.floor(min(P_station_info.loc[:, 'Y'])), math.ceil(max(P_station_info.loc[:, 'Y'])), 30)

X,Y = np.meshgrid(G_grid_lon,G_grid_lat)

#%%
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import shapefile
import os
import gc

shp_file = shpreader.Reader(r'D:\important\research\groundwater_forecast\zhuoshui(shp)\zhuoshui.shp')
fig = plt.figure(figsize=(16, 9),dpi=300)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([120.05, 121.4, 23.4, 24.25], crs=ccrs.PlateCarree())

ax.add_geometries(shp_file.geometries(), crs=ccrs.PlateCarree(), linewidths=1.2, edgecolor='k', facecolor='none') # edgecolor=調整邊框顏色
ax.scatter(X,Y)

