# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 13:58:16 2022

@author: steve
"""

import os
path=r'D:\important\research\groundwater_forecast(monthly)'
os.chdir(path)
import pandas as pd
import numpy as np

#%%
#level 1
g1_train=pd.read_excel(r"result\1st_layer\simulate-shuffle(train).xlsx",sheet_name="groundwater_observation(t+1)",index_col=0)
g1_test=pd.read_excel(r"result\1st_layer\simulate-shuffle(test).xlsx",sheet_name="groundwater_observation(t+1)",index_col=0)
station_info = pd.read_excel(r"station_info\station_fullinfo.xlsx",sheet_name="G1",index_col=0)

g1_obs=pd.concat([g1_train,g1_test],axis=0).reset_index(drop=True)

g1_std=np.std(g1_obs)
g1_std=pd.concat([g1_std,station_info.iloc[:,[0,-1]]],axis=1).reset_index()
g1_std.columns=['index','std','測站','區域']
g1_std=g1_std.sort_values(by=['區域','index']).reset_index(drop=True)

g1_mean=np.mean(g1_obs)
g1_mean=pd.concat([g1_mean,station_info.iloc[:,[0,-1]]],axis=1).reset_index()
g1_mean.columns=['index','mean','測站','區域']
g1_mean=g1_mean.sort_values(by=['區域','index']).reset_index(drop=True)

station_info_=station_info.reset_index()
station_info_=station_info_.sort_values(by=[0,'index']).reset_index(drop=True)


#%%

#level 2
g2_train=pd.read_excel(r"result\2nd_layer\simulate-shuffle(train).xlsx",sheet_name="groundwater_observation(t+1)",index_col=0)
g2_test=pd.read_excel(r"result\2nd_layer\simulate-shuffle(test).xlsx",sheet_name="groundwater_observation(t+1)",index_col=0)
station_info = pd.read_excel(r"station_info\station_fullinfo.xlsx",sheet_name="G2",index_col=0)

g2_obs=pd.concat([g2_train,g2_test],axis=0).reset_index(drop=True)

g2_std=np.std(g2_obs)
g2_std=pd.concat([g2_std,station_info.iloc[:,[0,-1]]],axis=1).reset_index()
g2_std.columns=['index','std','測站','區域']
g2_std=g2_std.sort_values(by=['區域','index']).reset_index(drop=True)

g2_mean=np.mean(g2_obs)
g2_mean=pd.concat([g2_mean,station_info.iloc[:,[0,-1]]],axis=1).reset_index()
g2_mean.columns=['index','mean','測站','區域']
g2_mean=g2_mean.sort_values(by=['區域','index']).reset_index(drop=True)

station_info_=station_info.reset_index()
station_info_=station_info_.sort_values(by=[0,'index']).reset_index(drop=True)