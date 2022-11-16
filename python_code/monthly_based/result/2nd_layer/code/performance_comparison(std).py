# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 11:00:38 2022

@author: steve
"""

import os
path=r'D:\important\research\groundwater_forecast(monthly)'
os.chdir(path)
import pandas as pd

station_info = pd.read_excel(r"station_info\station_fullinfo.xlsx",sheet_name="G2",index_col=0)

#%%
"""R2"""

lstmhbv_t1_R2=pd.read_excel(r"result\2nd_layer\hbvann-forecast(t+1).xlsx",sheet_name="test_R2_eachstation",index_col=0)
lstmhbv_t2_R2=pd.read_excel(r"result\2nd_layer\hbvann-forecast(t+2).xlsx",sheet_name="test_R2_eachstation",index_col=0)
lstmhbv_t3_R2=pd.read_excel(r"result\2nd_layer\hbvann-forecast(t+3).xlsx",sheet_name="test_R2_eachstation",index_col=0)

simulate_R2=pd.read_excel(r"result\2nd_layer\simulate-shuffle(test).xlsx",sheet_name="testing_performance",index_col=0).iloc[:,0:3]

#%%

def merge_data(simulate,lstmhbv,station_info):
    merged_data=pd.concat([station_info.iloc[:,[0,3]],simulate,lstmhbv],axis=1).reset_index()
    merged_data.columns=['index','測站','區域','simulate','lstm-hbv']
    merged_data=merged_data.sort_values(by=['區域','index']).reset_index(drop=True)
    merged_data=merged_data.iloc[:,1:]
    return merged_data

t1_R2=merge_data(simulate_R2.iloc[:,0],lstmhbv_t1_R2,station_info)
t2_R2=merge_data(simulate_R2.iloc[:,1],lstmhbv_t2_R2,station_info)
t3_R2=merge_data(simulate_R2.iloc[:,2],lstmhbv_t3_R2,station_info)

#%%
#sort again
data_statistic=pd.read_excel(r"data_statistic(sort_std).xlsx",sheet_name="l2")
shuffle_index=data_statistic.loc[:,'index']-34

t1_R2=t1_R2.iloc[shuffle_index,:].reset_index(drop=True)
t2_R2=t2_R2.iloc[shuffle_index,:].reset_index(drop=True)
t3_R2=t3_R2.iloc[shuffle_index,:].reset_index(drop=True)

#%%
"""RMSE"""

lstmhbv_t1_rmse=pd.read_excel(r"result\2nd_layer\hbvann-forecast(t+1).xlsx",sheet_name="test_rmse_eachstation",index_col=0)
lstmhbv_t2_rmse=pd.read_excel(r"result\2nd_layer\hbvann-forecast(t+2).xlsx",sheet_name="test_rmse_eachstation",index_col=0)
lstmhbv_t3_rmse=pd.read_excel(r"result\2nd_layer\hbvann-forecast(t+3).xlsx",sheet_name="test_rmse_eachstation",index_col=0)

simulate_rmse=pd.read_excel(r"result\2nd_layer\simulate-shuffle(test).xlsx",sheet_name="testing_performance",index_col=0).iloc[:,3:]

#%%
t1_rmse=merge_data(simulate_rmse.iloc[:,0],lstmhbv_t1_rmse,station_info)
t2_rmse=merge_data(simulate_rmse.iloc[:,1],lstmhbv_t2_rmse,station_info)
t3_rmse=merge_data(simulate_rmse.iloc[:,2],lstmhbv_t3_rmse,station_info)

#%%
t1_rmse=t1_rmse.iloc[shuffle_index,:].reset_index(drop=True)
t2_rmse=t2_rmse.iloc[shuffle_index,:].reset_index(drop=True)
t3_rmse=t3_rmse.iloc[shuffle_index,:].reset_index(drop=True)

#%%
import numpy as np
from matplotlib import pyplot as plt

barWidth = 0.25
br1 = np.arange(len(t1_R2))
br2 = [x + barWidth for x in br1]
# br3 = [x + barWidth for x in br2]

def plot_figure(performance,name):
    # csfont = {'fontname':'Times New Roman'}
    plt.style.use('ggplot')
    # Figure Size
    fig, ax =plt.subplots(figsize =(15, 4),dpi=300)
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']   
    plt.rcParams['axes.unicode_minus']=False 
    # Horizontal Bar Plot
    plt.bar(br1, performance.iloc[:,2],color='b',width = barWidth,edgecolor ='grey',label='HBV')
    plt.bar(br2, performance.iloc[:,3],color='r',width = barWidth,edgecolor ='grey',label='HBV-LSTM')
    plt.xticks([r + barWidth for r in range(len(performance))],performance.iloc[:,0])
    plt.title('%s'%name,color="black")
    # plt.title('%s'%name,color="black",**csfont)
    ax.tick_params(axis='y', colors='black') ;ax.tick_params(axis='x', colors='black') 
    plt.legend('')
    if 'R2' in name:
        plt.ylim([0, 1.1])
    else:
        plt.ylim([0, 11])       
    plt.savefig(r"result\2nd_layer\performance figure\%s.jpeg"%name)
    # Show Plot
    plt.show()

plot_figure(t1_R2,'Testing R2(T+1)')
plot_figure(t2_R2,'Testing R2(T+2)')
plot_figure(t3_R2,'Testing R2(T+3)')
    
plot_figure(t1_rmse,'Testing rmse(T+1)')
plot_figure(t2_rmse,'Testing rmse(T+2)')
plot_figure(t3_rmse,'Testing rmse(T+3)')
