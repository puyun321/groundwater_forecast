# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 10:41:59 2022

@author: steve
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


#%%
"""layer 1 performance comparison"""

writer = pd.ExcelWriter(r"D:\important\research\groundwater_forecast\python_code\puyun\difference_level\result\1st_layer\performance(layer1).xlsx",engine='xlsxwriter')
station_info=pd.read_excel(r"D:\important\research\groundwater_forecast\G01_station.xlsx")

for t in range(1,4):

    station_name=station_info.iloc[:,1]
    lstm_R2=pd.read_excel(r"D:\important\research\groundwater_forecast\python_code\puyun\difference_level\result\1st_layer\lstm-forecast(t+%s).xlsx"%t,sheet_name="test_R2_eachstation",index_col=0)
    lstmhbv_R2=pd.read_excel(r"D:\important\research\groundwater_forecast\python_code\puyun\difference_level\result\1st_layer\lstmhbv-forecast(t+%s).xlsx"%t,sheet_name="test_R2_eachstation",index_col=0)
    simulate_R2=pd.read_excel(r"D:\important\research\groundwater_forecast\python_code\puyun\difference_level\result\1st_layer\simulate(test).xlsx",sheet_name="testing_performance").iloc[:,1:4]
    
    # x=np.array(lstm_R2.index)
    y1=np.array(simulate_R2)[:,t-1]; y2=np.array(lstm_R2)[:,0]; y3=np.array(lstmhbv_R2)[:,0]
    
    """merge all results"""
    combined_R2=pd.DataFrame(np.asarray([station_name,y1,y2,y3]).T)
    combined_R2.columns=['測站','simulate','lstm','lstmhbv']
    combined_R2.to_excel(writer,sheet_name="t+%s_R2"%t)
    
    lstm_rmse=pd.read_excel(r"D:\important\research\groundwater_forecast\python_code\puyun\difference_level\result\1st_layer\lstm-forecast(t+1).xlsx",sheet_name="test_rmse_eachstation",index_col=0)
    lstmhbv_rmse=pd.read_excel(r"D:\important\research\groundwater_forecast\python_code\puyun\difference_level\result\1st_layer\lstmhbv-forecast(t+1).xlsx",sheet_name="test_rmse_eachstation",index_col=0)
    simulate_rmse=pd.read_excel(r"D:\important\research\groundwater_forecast\python_code\puyun\difference_level\result\1st_layer\simulate(test).xlsx",sheet_name="testing_performance").iloc[:,4:]
    
    y1=np.array(simulate_rmse)[:,t-1]; y2=np.array(lstm_rmse)[:,0]; y3=np.array(lstmhbv_rmse)[:,0]
    
    """merge all results"""
    combined_rmse=pd.DataFrame(np.asarray([station_name,y1,y2,y3]).T)
    combined_rmse.columns=['測站','simulate','lstm','lstmhbv']
    combined_rmse.to_excel(writer,sheet_name="t+%s_rmse"%t)

writer.save()
