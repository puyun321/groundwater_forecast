# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 17:04:11 2023

@author: Steve
"""

import os
import pandas as pd
import numpy as np

path=r'D:\lab\research\groundwater_estimation\github\python_code\monthly_based' #if the code above cant work, please key in your parent directory
os.chdir(path)

#%%
"""Read station info"""
station_info = pd.read_excel(r"station_info\station_fullinfo.xlsx",sheet_name="G1",index_col=0)

#%%
"""test"""
obs1=pd.read_excel(r"result\1st_layer\performance_comparison\test\T+1.xlsx",sheet_name="obs",index_col=0)
hbvlstm1=pd.read_excel(r"result\1st_layer\performance_comparison\test\T+1.xlsx",sheet_name="HBV-AE-LSTM",index_col=0)
hbv1=pd.read_excel(r"result\1st_layer\performance_comparison\test\T+1.xlsx",sheet_name="HBV",index_col=0)

obs2=pd.read_excel(r"result\1st_layer\performance_comparison\test\T+2.xlsx",sheet_name="obs",index_col=0)
hbvlstm2=pd.read_excel(r"result\1st_layer\performance_comparison\test\T+2.xlsx",sheet_name="HBV-AE-LSTM",index_col=0)
hbv2=pd.read_excel(r"result\1st_layer\performance_comparison\test\T+2.xlsx",sheet_name="HBV",index_col=0)

obs3=pd.read_excel(r"result\1st_layer\performance_comparison\test\T+3.xlsx",sheet_name="obs",index_col=0)
hbvlstm3=pd.read_excel(r"result\1st_layer\performance_comparison\test\T+3.xlsx",sheet_name="HBV-AE-LSTM",index_col=0)
hbv3=pd.read_excel(r"result\1st_layer\performance_comparison\test\T+3.xlsx",sheet_name="HBV",index_col=0)

obs=obs1.iloc[2:,:].reset_index(drop=True)
hbvlstm1=hbvlstm1.iloc[2:,:].reset_index(drop=True)
hbvlstm2=hbvlstm2.iloc[1:-1,:].reset_index(drop=True)
hbvlstm3=hbvlstm3.iloc[:-2,:].reset_index(drop=True)

choose_date=obs.iloc[:,0].reset_index(drop=True)

#%%
import matplotlib as mpl
from matplotlib import pyplot as plt
path2='result\\1st_layer\\time series figure\\test\\compare'
isExist = os.path.exists(path2)
if not isExist:
  # Create a new directory because it does not exist 
  os.makedirs(path2)
  print("The new directory is created!")

for station in range(0,33):
    station=0
    fig = plt.figure(figsize=(6,6),dpi=300)
    ax = fig.add_subplot(111)
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']   
    plt.rcParams['axes.unicode_minus']=False  
    plt.style.use('ggplot')
    plt.title('%s Station'%station_info.iloc[station,0])
    plt.plot(choose_date,obs.iloc[:,station+1], color='black', linewidth=2)
    plt.plot(choose_date,hbvlstm1.iloc[:,station+1], color='red', linewidth=1.5)
    plt.plot(choose_date,hbvlstm2.iloc[:,station+1], color='green', linewidth=1)
    plt.plot(choose_date,hbvlstm3.iloc[:,station+1], color='purple', linewidth=0.8)
    plt.xticks(rotation = 45) 
    plt.ylabel('Groundwater Level', fontsize=24); plt.xlabel('time', fontsize=18)
    ax.tick_params(labelsize=14)
    mpl.rcParams['text.color'] = 'black'
    mpl.rcParams['axes.labelcolor'] = 'black'
    mpl.rcParams['xtick.color'] = 'black'
    mpl.rcParams['ytick.color'] = 'black'
    # plt.show()
    plt.savefig(path2+'\\%s.png'%station_info.iloc[station,0])
    plt.clf()
