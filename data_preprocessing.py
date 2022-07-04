# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 21:09:52 2022

@author: steve
"""

import pandas as pd
from datetime import datetime
import pickle

riverflow=pd.read_excel(r"D:\research\groundwater_estimation\daily_data\流量.xlsx",sheet_name="2000-",engine="openpyxl")
rainfall=pd.read_csv(r"D:\research\groundwater_estimation\daily_data\雨量.csv")
date=pd.DataFrame([datetime.strptime(rainfall.iloc[i,0],'%Y/%m/%d') for i in range(0,len(rainfall))])
date.columns=['date']
select_index=[index for index,element in enumerate(date.iloc[:,0]) if element<datetime(2020,1,1)]
rainfall.loc[:,'date']=date.iloc[:,0]

with open(r'D:\research\groundwater_estimation\daily_data\地下水\gw_dict20_l0_daily','rb') as fp:
   gw_dict20_l0_daily = pickle.load(fp)
with open(r'D:\research\groundwater_estimation\daily_data\地下水\gw_dict20_l1_daily','rb') as fp:
   gw_dict20_l1_daily = pickle.load(fp)
with open(r'D:\research\groundwater_estimation\daily_data\地下水\gw_dict20_l2_daily','rb') as fp:
   gw_dict20_l2_daily = pickle.load(fp)
with open(r'D:\research\groundwater_estimation\daily_data\地下水\gw_dict20_l3_daily','rb') as fp:
   gw_dict20_l3_daily = pickle.load(fp)
with open(r'D:\research\groundwater_estimation\daily_data\地下水\gw_dict20_l4_daily','rb') as fp:
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

selected_riverflow=riverflow.iloc[select_index,:]
selected_rainfall=rainfall.iloc[select_index,:]  


#%%
riverflow_10days=(selected_riverflow.resample('10D',on='date',base=0,loffset='9D').mean())
rainfall_10days=(selected_rainfall.resample('10D',on='date',base=0,loffset='9D').mean())
groundwater_l0_10days=(groundwater_l0.resample('10D',on='date',base=0,loffset='9D').mean())
groundwater_l1_10days=(groundwater_l1.resample('10D',on='date',base=0,loffset='9D').mean())
groundwater_l2_10days=(groundwater_l2.resample('10D',on='date',base=0,loffset='9D').mean())
groundwater_l3_10days=(groundwater_l3.resample('10D',on='date',base=0,loffset='9D').mean())
groundwater_l4_10days=(groundwater_l4.resample('10D',on='date',base=0,loffset='9D').mean())

#%%
#interpolation
import os
os.chdir(r"D:\research\research_use_function")
from preprocessing import preprocessing
preprocessing_=preprocessing(riverflow_10days)
riverflow_10days=preprocessing_.interpolate(specific_factor=[],remove_negative=True)

preprocessing_=preprocessing(rainfall_10days)
rainfall_10days=preprocessing_.interpolate(specific_factor=[],remove_negative=True)

preprocessing_=preprocessing(groundwater_l0_10days)
groundwater_l0_10days=preprocessing_.interpolate(specific_factor=[],remove_negative=False)
preprocessing_=preprocessing(groundwater_l1_10days)
groundwater_l1_10days=preprocessing_.interpolate(specific_factor=[],remove_negative=False)
preprocessing_=preprocessing(groundwater_l2_10days)
groundwater_l2_10days=preprocessing_.interpolate(specific_factor=[],remove_negative=False)
preprocessing_=preprocessing(groundwater_l3_10days)
groundwater_l3_10days=preprocessing_.interpolate(specific_factor=[],remove_negative=False)
preprocessing_=preprocessing(groundwater_l4_10days)
groundwater_l4_10days=preprocessing_.interpolate(specific_factor=[],remove_negative=False)

#%%
writer = pd.ExcelWriter(r'D:\research\groundwater_estimation\daily_data\underground_dataset(10days-based).xlsx', engine='xlsxwriter')
riverflow_10days.to_excel(writer,sheet_name="riverflow")
rainfall_10days.to_excel(writer,sheet_name="rainfall")
groundwater_l0_10days.to_excel(writer,sheet_name="groundwater_l0")
groundwater_l1_10days.to_excel(writer,sheet_name="groundwater_l1")
groundwater_l2_10days.to_excel(writer,sheet_name="groundwater_l2")
groundwater_l3_10days.to_excel(writer,sheet_name="groundwater_l3")
groundwater_l4_10days.to_excel(writer,sheet_name="groundwater_l4")
writer.save()

#%%
writer = pd.ExcelWriter(r'D:\research\groundwater_estimation\daily_data\underground_dataset(10days-based)_without_interpolate.xlsx', engine='xlsxwriter')
riverflow_10days.to_excel(writer,sheet_name="riverflow")
rainfall_10days.to_excel(writer,sheet_name="rainfall")
groundwater_l0_10days.to_excel(writer,sheet_name="groundwater_l0")
groundwater_l1_10days.to_excel(writer,sheet_name="groundwater_l1")
groundwater_l2_10days.to_excel(writer,sheet_name="groundwater_l2")
groundwater_l3_10days.to_excel(writer,sheet_name="groundwater_l3")
groundwater_l4_10days.to_excel(writer,sheet_name="groundwater_l4")
writer.save()