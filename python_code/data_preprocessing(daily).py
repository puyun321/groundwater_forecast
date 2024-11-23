# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 19:40:20 2022

@author: steve
"""

import os
os.chdir(os.path.dirname(__file__))
#%%
import pandas as pd
from datetime import datetime
import pickle

riverflow=pd.read_csv(r"raw_data\riverflow.csv")
rainfall=pd.read_csv(r"raw_data\rainfall.csv")
date=pd.DataFrame([datetime.strptime(rainfall.iloc[i,0],'%Y/%m/%d') for i in range(0,len(rainfall))])
date.columns=['date']
select_index=[index for index,element in enumerate(date.iloc[:,0]) if element<datetime(2020,1,1)]
rainfall.loc[:,'date']=date.iloc[:,0]

with open(r'raw_data\gw_dict20_l0_daily','rb') as fp:
   gw_dict20_l0_daily = pickle.load(fp)
with open(r'raw_data\gw_dict20_l1_daily','rb') as fp:
   gw_dict20_l1_daily = pickle.load(fp)
with open(r'raw_data\gw_dict20_l2_daily','rb') as fp:
   gw_dict20_l2_daily = pickle.load(fp)
with open(r'raw_data\gw_dict20_l3_daily','rb') as fp:
   gw_dict20_l3_daily = pickle.load(fp)
with open(r'raw_data\gw_dict20_l4_daily','rb') as fp:
   gw_dict20_l4_daily = pickle.load(fp)
   
#%%
groundwater_l0=pd.concat([date.iloc[select_index],pd.concat([value.iloc[:,1] for key,value in gw_dict20_l0_daily.items()],axis=1)],axis=1)
groundwater_l0.columns=[date.columns[0]]+[key for key,value in gw_dict20_l0_daily.items()]
groundwater_l1=pd.concat([date.iloc[select_index],pd.concat([value.iloc[:,1] for key,value in gw_dict20_l1_daily.items()],axis=1)],axis=1)
groundwater_l1.columns=[date.columns[0]]+[key for key,value in gw_dict20_l1_daily.items()]
groundwater_l2=pd.concat([date.iloc[select_index],pd.concat([value.iloc[:,1] for key,value in gw_dict20_l2_daily.items()],axis=1)],axis=1)
groundwater_l2.columns=[date.columns[0]]+[key for key,value in gw_dict20_l2_daily.items()]
groundwater_l3=pd.concat([date.iloc[select_index],pd.concat([value.iloc[:,1] for key,value in gw_dict20_l3_daily.items()],axis=1)],axis=1)
groundwater_l3.columns=[date.columns[0]]+[key for key,value in gw_dict20_l3_daily.items()]
groundwater_l4=pd.concat([date.iloc[select_index],pd.concat([value.iloc[:,1] for key,value in gw_dict20_l4_daily.items()],axis=1)],axis=1)
groundwater_l4.columns=[date.columns[0]]+[key for key,value in gw_dict20_l4_daily.items()]

groundwater_l0=groundwater_l0.set_index('date');groundwater_l1=groundwater_l1.set_index('date');
groundwater_l2=groundwater_l2.set_index('date');groundwater_l3=groundwater_l3.set_index('date');groundwater_l4=groundwater_l4.set_index('date');
selected_riverflow=riverflow.iloc[select_index,:];selected_riverflow=selected_riverflow.set_index('date');
selected_rainfall=rainfall.iloc[select_index,:];selected_rainfall=selected_rainfall.set_index('date');

#%% data interpolation
import os
os.chdir(r"D:\important\research\research_use_function")
from preprocessing import preprocessing
preprocessing_=preprocessing(selected_riverflow)
riverflow_new=preprocessing_.interpolate(specific_factor=[],remove_negative=True)

preprocessing_=preprocessing(selected_rainfall)
rainfall_new=preprocessing_.interpolate(specific_factor=[],remove_negative=True)

preprocessing_=preprocessing(groundwater_l0)
groundwater_l0_new=preprocessing_.interpolate(specific_factor=[],remove_negative=False)
preprocessing_=preprocessing(groundwater_l1)
groundwater_l1_new=preprocessing_.interpolate(specific_factor=[],remove_negative=False)
preprocessing_=preprocessing(groundwater_l2)
groundwater_l2_new=preprocessing_.interpolate(specific_factor=[],remove_negative=False)
preprocessing_=preprocessing(groundwater_l3)
groundwater_l3_new=preprocessing_.interpolate(specific_factor=[],remove_negative=False)
preprocessing_=preprocessing(groundwater_l4)
groundwater_l4_new=preprocessing_.interpolate(specific_factor=[],remove_negative=False)

#%% save as csv
riverflow_new.to_csv(r"monthly_based\daily_data\riverflow")
rainfall_new.to_csv(r"monthly_based\daily_data\rainfall")
groundwater_l0_new.to_csv(r"monthly_based\daily_data\groundwater_l0.csv")
groundwater_l1_new.to_csv(r"monthly_based\daily_data\groundwater_l1.csv")
groundwater_l2_new.to_csv(r"monthly_based\daily_data\groundwater_l2.csv")
groundwater_l3_new.to_csv(r"monthly_based\daily_data\groundwater_l3.csv")
groundwater_l4_new.to_csv(r"monthly_based\daily_data\groundwater_l4.csv")
