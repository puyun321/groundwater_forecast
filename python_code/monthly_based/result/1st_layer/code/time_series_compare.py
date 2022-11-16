# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 17:32:03 2022

@author: steve
"""
import os
path=r'D:\important\research\groundwater_forecast(monthly)'
os.chdir(path)
import pandas as pd

hbv_lstm=pd.read_excel(r"result\1st_layer\sorted_predict(hbv-ann).xlsx",sheet_name="t+1(shuffle)",index_col=0)
hbv=pd.read_excel(r"result\1st_layer\sorted_predict.xlsx",sheet_name="t+1(shuffle)",index_col=0).iloc[1:,:]

#%%
import numpy as np
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

G_station_info = pd.read_csv(r"station_info\groundwater_station.csv",delimiter='\t')

G0 = pd.read_csv(r"daily_data\groundwater_l0.csv",delimiter=',');G0['date'] = pd.to_datetime(G0['date']);
G0 = G0.resample('1M',on='date',base=0,loffset='1M').mean() #convert daily data to 10day based
G0_station = G0.columns
G0_station = pd.Series([G0_station[i][8:] for i in range(0,len(G0_station))])
G0_station_name = remain_onlyname(G0_station)
G0_station_info = get_station_info(G_station_info,G0_station_name)

G0_new = merge_samename(G0_station_name,G0)
G0_station_info_new = G0_station_info.drop_duplicates(keep="first").reset_index(drop=True)

G1 = pd.read_csv(r"daily_data\groundwater_l1.csv",delimiter=',');G1['date'] = pd.to_datetime(G1['date']);
G1 = G1.resample('1M',on='date',base=0,loffset='1M').mean() #convert daily data to 10day based
G1_station = G1.columns
G1_station = pd.Series([G1_station[i][8:] for i in range(0,len(G1_station))])
G1_station_name = remain_onlyname(G1_station)
G1_station_info = get_station_info(G_station_info,G1_station_name)

G1_new = merge_samename(G1_station_name,G1)
G1_station_info_new = G1_station_info.drop_duplicates(keep="first").reset_index(drop=True)


G01_new = np.concatenate([G0_new,G1_new],axis=1)

station_info = np.concatenate([G0_station_info_new,G1_station_info_new])

#%%
import os
os.chdir(r"D:\important\research\research_use_function")
from multidimensional_reshape import multi_input_output
os.chdir(path)
timestep=3
preprocessing_module=multi_input_output(G01_new[1:,:],input_timestep=timestep,output_timestep=timestep)
G_multi_input_origin=preprocessing_module.generate_input()
G_multi_output_origin=preprocessing_module.generate_output()

forecast_timestep = np.array([timestep/3,timestep*2/3,timestep]).astype(int)-1
model_train_output = G_multi_output_origin[:len(hbv_lstm),forecast_timestep[0],:]

# writer = pd.ExcelWriter(r"D:\important\research\groundwater_forecast\python_code\puyun\result\1st_layer\groundwater_obs_data.xlsx",engine='xlsxwriter')
# for i in range(0,len(forecast_timestep)):
#     model_train_output = G_multi_output_origin[:len(hbv_lstm),forecast_timestep[i],:]
#     pd.DataFrame(model_train_output).to_excel(writer,sheet_name="t+%s"%(i+1))
# writer.save()

#%%
import os
os.chdir(r"D:\important\research\research_use_function")
from multidimensional_reshape import multi_input_output
from error_indicator import error_indicator
os.chdir(path)

"""choose date"""
date=pd.DataFrame(G0.iloc[timestep-1+forecast_timestep[0]:-(timestep-forecast_timestep[0])].index)
train_date=date.iloc[1:len(hbv)+1]

start_date='2000-01-01';end_date='2007-12-31'
# start_date='2008-01-01';end_date='2016-12-31'
# start_date='2000-08-01';end_date='2000-10-01'
# start_date='2001-06-01';end_date='2001-10-01'
# start_date='2005-07-01';end_date='2001-11-01'
# start_date='2006-07-01';end_date='2006-08-31'
# start_date='2007-08-01';end_date='2007-10-01'
# start_date='2008-01-01';end_date='2016-12-31'
# start_date='2008-07-01';end_date='2008-09-01'
# start_date='2010-09-01';end_date='2010-10-31'
# start_date='2011-05-01';end_date='2011-09-01'
# start_date='2012-06-01';end_date='2012-10-01'
# start_date='2013-07-01';end_date='2013-10-31'
# start_date='2014-06-01';end_date='2014-10-01'
# start_date='2015-05-01';end_date='2015-09-01'


# start_date='2016-07-01';end_date='2016-09-30'
choose_date = (train_date[(train_date['date']>=start_date) & (train_date['date']<=end_date)].iloc[:,0]).reset_index(drop=True)
select_index= train_date[(train_date['date']>=start_date) & (train_date['date']<=end_date)].index-1

#%%

# Check whether the specified path exists or not
path1='result\\1st_layer\\time series figure\\%s'%start_date
isExist = os.path.exists(path1)
if not isExist:
  # Create a new directory because it does not exist 
  os.makedirs(path1)
  print("The new directory is created!")

import seaborn as sns
from matplotlib import pyplot as plt

for station in range(0,33):
    hbv_ann_R2=round(error_indicator.np_R2(model_train_output[select_index,station],hbv_lstm.iloc[select_index,station]),2)
    hbv_ann_RMSE=round(error_indicator.np_RMSE(model_train_output[select_index,station],hbv_lstm.iloc[select_index,station]),2)
    
    hbv_R2=round(error_indicator.np_R2(model_train_output[select_index,station],hbv.iloc[select_index,station]),2)
    hbv_RMSE=round(error_indicator.np_RMSE(model_train_output[select_index,station],hbv.iloc[select_index,station]),2)
    
    fig = plt.figure(figsize=(10,6))
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']   
    plt.rcParams['axes.unicode_minus']=False  
    plt.style.use('ggplot')
    plt.title('%s Station'%station_info[station,0])
    plt.plot(choose_date,model_train_output[select_index,station], color='black', linewidth=1.5)
    plt.plot(choose_date,hbv_lstm.iloc[select_index,station], color='red', linewidth=0.8)

    plt.plot(choose_date,hbv.iloc[select_index,station], color='blue', linewidth=0.8)
    # plt.legend(labels=['Observed data','HBV-ANN','HBV'],loc='upper right')
    # plt.legend(labels=['Observed data $R^2$(RMSE)','HBV-ANN %s(%s)'%(hbv_ann_R2,hbv_ann_RMSE),'HBV %s(%s)'%(hbv_R2,hbv_RMSE)],loc='upper right')
    plt.ylabel('Groundwater Level'); plt.xlabel('Groundwater Level')
    # plt.show()
    plt.savefig(path1+'\\%s.png'%station_info[station,0])