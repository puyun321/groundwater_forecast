# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 18:16:52 2022

@author: steve
"""

import pandas as pd

t=1
lstmhbv=pd.read_excel(r"D:\important\research\groundwater_forecast\python_code\puyun\difference_level\result\1st_layer\lstmhbv-forecast(t+%s).xlsx"%t,sheet_name="test_eachstation",index_col=0)
lstm=pd.read_excel(r"D:\important\research\groundwater_forecast\python_code\puyun\difference_level\result\1st_layer\lstm-forecast(t+%s).xlsx"%t,sheet_name="test_eachstation",index_col=0)
obs=pd.read_excel(r"D:\important\research\groundwater_forecast\python_code\puyun\difference_level\result\1st_layer\simulate(test).xlsx",sheet_name="groundwater_observation(t+%s)"%t,index_col=0)

#%%
import numpy as np
import os
os.chdir(r"D:\important\research\research_use_function")
from error_indicator import error_indicator

r2=[];rmse=[]
for i in range(0,obs.shape[1]):
    r2.append(error_indicator.np_R2(obs.iloc[:,i],lstmhbv.iloc[:,i]))
    rmse.append(error_indicator.np_RMSE(obs.iloc[:,i],lstmhbv.iloc[:,i]))
    
r2=np.array(r2)
rmse=np.array(rmse)