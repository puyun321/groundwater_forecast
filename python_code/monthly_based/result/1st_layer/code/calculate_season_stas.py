# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:03:25 2023

@author: Steve
"""

"""change woking directory to current directory"""
import os

working= os.path.dirname(os.path.abspath('__file__')) #try run this if work

path=r'D:\lab\research\groundwater_estimation\github\python_code\monthly_based' #if the code above cant work, please key in your parent directory
os.chdir(path)

#%%
""" Read Forecast Result """
import pandas as pd
import numpy as np


"""test"""
obs1=pd.read_excel(r"result\1st_layer\performance_comparison\test\T+1.xlsx",sheet_name="obs")
hbvlstm1_o=pd.read_excel(r"result\1st_layer\performance_comparison\test\T+1.xlsx",sheet_name="HBV-AE-LSTM")
hbv1=pd.read_excel(r"result\1st_layer\performance_comparison\test\T+1.xlsx",sheet_name="HBV")

obs2=pd.read_excel(r"result\1st_layer\performance_comparison\test\T+2.xlsx",sheet_name="obs")
hbvlstm2_o=pd.read_excel(r"result\1st_layer\performance_comparison\test\T+2.xlsx",sheet_name="HBV-AE-LSTM")
hbv2=pd.read_excel(r"result\1st_layer\performance_comparison\test\T+2.xlsx",sheet_name="HBV")

obs3=pd.read_excel(r"result\1st_layer\performance_comparison\test\T+3.xlsx",sheet_name="obs")
hbvlstm3_o=pd.read_excel(r"result\1st_layer\performance_comparison\test\T+3.xlsx",sheet_name="HBV-AE-LSTM")
hbv3=pd.read_excel(r"result\1st_layer\performance_comparison\test\T+3.xlsx",sheet_name="HBV")

obs=obs1.iloc[2:,1:].reset_index(drop=True)
hbvlstm1=hbvlstm1_o.iloc[2:,1:].reset_index(drop=True)
hbvlstm2=hbvlstm2_o.iloc[1:-1,1:].reset_index(drop=True)
hbvlstm3=hbvlstm3_o.iloc[:-2,1:].reset_index(drop=True)

choose_date=obs1.iloc[2:,0].reset_index(drop=True)

#%%
""" month to season"""

season_datetime=['2017-04-30','2017-07-31','2017-10-31','2018-01-31','2018-04-30','2018-07-31','2018-10-31','2019-01-31','2019-04-30','2019-07-31','2019-10-31']
season_name=['2017 spring','2017 summer','2017 autumn','2018 winter','2018 spring','2018 summer','2018 autmn','2019 winter','2019 spring','2019 summer','2019 autumn']

train_obs=pd.read_excel(r"result\1st_layer\performance_comparison\train\T+3.xlsx",sheet_name="obs")
winter_obs=pd.DataFrame(train_obs.iloc[-2:,2:])

"""Read station info"""
station_info = pd.read_excel(r"station_info\station_fullinfo.xlsx",sheet_name="G1",index_col=0)
sorted_info=pd.read_excel('data_statistic(sort_std).xlsx',sheet_name='l1')
sorted_winter_obs=pd.concat([winter_obs.iloc[:,station_info[station_info.iloc[:,0]==sorted_info.iloc[i,2]].index] for i in range(len(sorted_info))],axis=1).reset_index(drop=True)

import os
os.chdir(r"D:\lab\research\research_use_function")
from error_indicator import error_indicator
os.chdir(path)


stats_mean=[]; stats_std=[];
obs_all=[]
winter_mean=np.array(np.mean(sorted_winter_obs,axis=0)).astype('float64')
obs_all.append(winter_mean)
stats_mean.append(np.mean(winter_mean))
stats_std.append(np.mean(np.std(sorted_winter_obs,axis=0)))

for i in range(0,len(season_datetime)):
    if i==0:
        choose_index=choose_date[choose_date<=season_datetime[i]].index
        select_obs= np.array(obs.iloc[choose_index[0]:choose_index[0]+2,:])

        # select_obs= np.array(obs.iloc[choose_index,:])
        # select_obs= np.concatenate([np.expand_dims(np.array(sorted_winter_obs.iloc[1,:]),axis=0),select_obs],axis=0)
            
        obs_all.append(np.mean(np.array(select_obs),axis=0))
        stats_mean.append(np.mean(np.mean(select_obs,axis=0)))
        stats_std.append(np.mean(np.std(select_obs,axis=0)))
        
    else:
        choose_index=choose_date[(choose_date<=season_datetime[i]) & (choose_date>season_datetime[i-1])].index
        select_obs= obs.iloc[choose_index,:]
    
    
        obs_1=[]; obs_2=[]; obs_3=[]
        for j in range(0,select_obs.shape[1]):
            obs_1.append(select_obs.iloc[0,j])
            obs_2.append(select_obs.iloc[1,j])
            obs_3.append(select_obs.iloc[2,j])
            
        obs_all.append(np.mean([obs_1,obs_2,obs_3],axis=0))
        stats_mean.append(np.mean(np.mean([obs_1,obs_2,obs_3],axis=0)))
        stats_std.append(np.mean(np.std([obs_1,obs_2,obs_3],axis=0)))




