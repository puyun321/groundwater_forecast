# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 22:19:52 2023

@author: Steve
"""

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
P_station = pd.DataFrame(P.columns)
date=pd.DataFrame(P.index)
P = np.array(P)
P_station_info = pd.read_csv(r"station_info\rainfall_station.csv") #get rainfall station info

#%%
""" split rainfall station into different region """
p_r1 = pd.read_excel(r"station_info\rainfall_station.xlsx",sheet_name="R1") #get rainfall station info
p_r2 = pd.read_excel(r"station_info\rainfall_station.xlsx",sheet_name="R2") #get rainfall station info
p_r3 = pd.read_excel(r"station_info\rainfall_station.xlsx",sheet_name="R3") #get rainfall station info

r1_index=[P_station[P_station.iloc[:,0]==i].index[0] for i in np.array(p_r1.iloc[:,0])]
r2_index=[P_station[P_station.iloc[:,0]==i].index[0] for i in np.array(p_r2.iloc[:,0])]
r3_index=[P_station[P_station.iloc[:,0]==i].index[0] for i in np.array(p_r3.iloc[:,0])]

#%%

season_datetime=['2017-04-30','2017-07-31','2017-10-31','2018-01-31','2018-04-30','2018-07-31','2018-10-31','2019-01-31','2019-04-30','2019-07-31','2019-10-31']
season_name=['2017 spring','2017 summer','2017 autumn','2018 winter','2018 spring','2018 summer','2018 autmn','2019 winter','2019 spring','2019 summer','2019 autumn']
obs_all=[]; stats_mean=[]; stats_std=[];
stats_r1_mean=[]; stats_r1_std=[];stats_r2_mean=[]; stats_r2_std=[];stats_r3_mean=[]; stats_r3_std=[];
train_obs=pd.read_excel(r"result\1st_layer\performance_comparison\train\T+1.xlsx",sheet_name="obs")
first_winter_date=train_obs.iloc[-1,1]
choose_index=date[date.iloc[:,0]==first_winter_date].index
select_obs= np.array(P[choose_index,:])
    
obs_all.append(np.mean(np.array(select_obs),axis=0))
stats_mean.append(np.mean(np.mean(select_obs,axis=0)))
stats_std.append(np.mean(np.std(select_obs,axis=0)))
stats_r1_mean.append(np.mean(np.mean(select_obs[:,r1_index],axis=0)))
stats_r1_std.append(np.mean(np.std(select_obs[:,r1_index],axis=0)))
stats_r2_mean.append(np.mean(np.mean(select_obs[:,r2_index],axis=0)))
stats_r2_std.append(np.mean(np.std(select_obs[:,r2_index],axis=0)))
stats_r3_mean.append(np.mean(np.mean(select_obs[:,r3_index],axis=0)))
stats_r3_std.append(np.mean(np.std(select_obs[:,r3_index],axis=0)))



for i in range(0,len(season_datetime)):
    if i==0:
        choose_index=date[date.iloc[:,0]<=season_datetime[i]].index
        select_obs= np.array(P[choose_index[0]:choose_index[0]+2,:])
            
        obs_all.append(np.mean(np.array(select_obs),axis=0))
        stats_mean.append(np.mean(np.mean(select_obs,axis=0)))
        stats_std.append(np.mean(np.std(select_obs,axis=0)))
        stats_r1_mean.append(np.mean(np.mean(select_obs[:,r1_index],axis=0)))
        stats_r1_std.append(np.mean(np.std(select_obs[:,r1_index],axis=0)))
        stats_r2_mean.append(np.mean(np.mean(select_obs[:,r2_index],axis=0)))
        stats_r2_std.append(np.mean(np.std(select_obs[:,r2_index],axis=0)))
        stats_r3_mean.append(np.mean(np.mean(select_obs[:,r3_index],axis=0)))
        stats_r3_std.append(np.mean(np.std(select_obs[:,r3_index],axis=0)))
        
        
    else:
        choose_index=date[(date.iloc[:,0]<=season_datetime[i]) & (date.iloc[:,0]>season_datetime[i-1])].index
        select_obs= P[choose_index,:]
        
        obs_all.append(np.mean(np.array(select_obs),axis=0))
        stats_mean.append(np.mean(np.mean(select_obs,axis=0)))
        stats_std.append(np.mean(np.std(select_obs,axis=0)))
        stats_r1_mean.append(np.mean(np.mean(select_obs[:,r1_index],axis=0)))
        stats_r1_std.append(np.mean(np.std(select_obs[:,r1_index],axis=0)))
        stats_r2_mean.append(np.mean(np.mean(select_obs[:,r2_index],axis=0)))
        stats_r2_std.append(np.mean(np.std(select_obs[:,r2_index],axis=0)))
        stats_r3_mean.append(np.mean(np.mean(select_obs[:,r3_index],axis=0)))
        stats_r3_std.append(np.mean(np.std(select_obs[:,r3_index],axis=0)))
        
#%%

