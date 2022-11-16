# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 11:00:38 2022

@author: steve
"""

import os
path=r'D:\important\research\groundwater_forecast(monthly)'
os.chdir(path)
import pandas as pd

station_info = pd.read_excel(r"station_info\station_fullinfo.xlsx",sheet_name="G1",index_col=0)

#%%
"""R2"""

lstmhbv_t1_R2=pd.read_excel(r"result\1st_layer\hbvann-forecast(t+1).xlsx",sheet_name="test_R2_eachstation",index_col=0)
lstmhbv_t2_R2=pd.read_excel(r"result\1st_layer\hbvann-forecast(t+2).xlsx",sheet_name="test_R2_eachstation",index_col=0)
lstmhbv_t3_R2=pd.read_excel(r"result\1st_layer\hbvann-forecast(t+3).xlsx",sheet_name="test_R2_eachstation",index_col=0)

simulate_R2=pd.read_excel(r"result\1st_layer\simulate-shuffle(test).xlsx",sheet_name="testing_performance",index_col=0).iloc[:,0:3]

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
data_statistic=pd.read_excel(r"data_statistic(sort_std).xlsx",sheet_name="l1")
shuffle_index=data_statistic.loc[:,'index']-1

t1_R2=t1_R2.iloc[shuffle_index,:].reset_index(drop=True)
t2_R2=t2_R2.iloc[shuffle_index,:].reset_index(drop=True)
t3_R2=t3_R2.iloc[shuffle_index,:].reset_index(drop=True)

#%%
"""RMSE"""

lstmhbv_t1_rmse=pd.read_excel(r"result\1st_layer\hbvann-forecast(t+1).xlsx",sheet_name="test_rmse_eachstation",index_col=0)
lstmhbv_t2_rmse=pd.read_excel(r"result\1st_layer\hbvann-forecast(t+2).xlsx",sheet_name="test_rmse_eachstation",index_col=0)
lstmhbv_t3_rmse=pd.read_excel(r"result\1st_layer\hbvann-forecast(t+3).xlsx",sheet_name="test_rmse_eachstation",index_col=0)

simulate_rmse=pd.read_excel(r"result\1st_layer\simulate-shuffle(test).xlsx",sheet_name="testing_performance",index_col=0).iloc[:,3:]

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
# br1 = np.arange(len(t1_R2))
# br2 = [x + barWidth for x in br1]

def plot_figure(performance,name):
    # Figure Size
    plt.style.use('bmh')
    # plt.style.use('ggplot')
    # plt.style.use('classic')
    fig  =plt.figure(figsize=(2,2),dpi=300)
    ax = fig.add_subplot(111, polar=True)# polar引數！！代表畫圓形！！！！
    ax.grid(color='k', linestyle='-', linewidth=0.1, alpha=0.5)
    plt.rcParams['axes.unicode_minus']=False 
    # radar Plot
    angles = np.linspace(0, 2*np.pi, len(performance), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    
    data1= np.concatenate((performance.iloc[:,2],[performance.iloc[0,2]]))
    data2= np.concatenate((performance.iloc[:,3],[performance.iloc[0,3]]))
    labels = np.concatenate((performance.index+1,[performance.index[0]+1]))
    
    ax.plot(angles, data1, 'bo-', linewidth=0.5, markersize=1, label='HBV')# 畫線四個引數為x,y,標記和顏色，閒的寬度
    ax.plot(angles, data2, 'ro-', linewidth=0.5, markersize=1,label='HBV-LSTM')# 畫線四個引數為x,y,標記和顏色，閒的寬度
    ax.set_thetagrids(angles * 180/np.pi, labels, fontproperties="Times New Roman")
    ax.tick_params(labelsize=8)
    plt.yticks(fontname = "Times New Roman") 
    ax.set_title('%s'%name,color="black", fontsize=10, fontproperties="Times New Roman")

    # plt.legend()
    if 'R2' in name:
        ax.set_rlim([0, 1.1])
    else:
        ax.set_rlim([0, 7])    
    ax.set_xlabel('Station no.', fontsize=12, fontproperties="Times New Roman")
    ax.yaxis.label.set_size(10)  
    
    plt.savefig(r"result\1st_layer\performance figure\%s(radar).jpeg"%name)
    # Show Plot
    plt.show()

plot_figure(t1_R2,'Testing R2(T+1)')
plot_figure(t2_R2,'Testing R2(T+2)')
plot_figure(t3_R2,'Testing R2(T+3)')
    
plot_figure(t1_rmse,'Testing rmse(T+1)')
plot_figure(t2_rmse,'Testing rmse(T+2)')
plot_figure(t3_rmse,'Testing rmse(T+3)')
