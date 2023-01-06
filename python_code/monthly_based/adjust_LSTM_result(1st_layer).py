# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 23:24:12 2022

@author: Steve
"""
import os
import pandas as pd
import numpy as np

path=r'D:\lab\research\groundwater_estimation\github\python_code\monthly_based' #if the code above cant work, please key in your parent directory
os.chdir(path)

#%%
pred_test_1=(pd.read_excel(r"result\1st_layer\predict(hbv-ae-ann)2.xlsx",sheet_name="pred_test_1").iloc[1:,1:]).reset_index(drop=True)
simulate_test_1=(pd.read_excel(r"result\1st_layer\sorted_predict.xlsx",sheet_name="t+1(test)").iloc[:-1,1:]).reset_index(drop=True)
obs_test_1=(pd.read_excel(r"result\1st_layer\predict(hbv-ae-ann)2.xlsx",sheet_name="obs_test_1").iloc[1:,1:]).reset_index(drop=True)

#%%

adjust_ratio = pred_test_1.iloc[:-1,:]/obs_test_1.iloc[:-1,:]
pred_test_adjust_1 = (pred_test_1.iloc[1:,:].reset_index(drop=True))/adjust_ratio
simulate_test_1=simulate_test_1.iloc[1:,:].reset_index(drop=True)
obs_test_1=obs_test_1.iloc[1:,:].reset_index(drop=True)

#%%
#time series compare
from matplotlib import pyplot as plt
"""choose date"""
timestep=3
G0 = pd.read_csv(r"daily_data\groundwater_l0.csv",delimiter=',');G0['date'] = pd.to_datetime(G0['date']);
G0 = G0.resample('1M',on='date',base=0,loffset='1M').mean() #convert daily data to 10day based
date=pd.DataFrame(G0.iloc[timestep:-(timestep-1)].index)
start_date='2017-01-01'
choose_date = (date[(date['date']>=start_date)]).reset_index(drop=True)
# select_index= date[(date['date']>=start_date)].index

path1='result\\1st_layer\\time series figure\\%s'%start_date
isExist = os.path.exists(path1)
if not isExist:
  # Create a new directory because it does not exist 
  os.makedirs(path1)
  print("The new directory is created!")


#%%
path2=r'D:\lab\research\research_use_function'
os.chdir(path2)

from error_indicator import error_indicator 

pred_R2=[];simulate_R2=[]
for i in range(0,pred_test_adjust_1.shape[1]):
    pred_R2.append(error_indicator.np_R2(obs_test_1.iloc[:,i],pred_test_adjust_1.iloc[:,i]))
    simulate_R2.append(error_indicator.np_R2(obs_test_1.iloc[:,i],simulate_test_1.iloc[:,i]))
    
pred_R2=np.array(pred_R2)
simulate_R2=np.array(simulate_R2)