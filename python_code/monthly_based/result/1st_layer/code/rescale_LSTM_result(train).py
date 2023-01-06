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
"""Read station info"""
station_info = pd.read_excel(r"station_info\station_fullinfo.xlsx",sheet_name="G1",index_col=0)

#%%
forecast_timestep=1
pred_train=(pd.read_excel(r"result\1st_layer\predict(hbv-ae-ann)2.xlsx",sheet_name="pred_train_%s"%(forecast_timestep)).iloc[2:,1:]).reset_index(drop=True)
simulate_train=(pd.read_excel(r"result\1st_layer\sorted_predict.xlsx",sheet_name="t+%s(shuffle)"%(forecast_timestep)).iloc[:-3,1:]).reset_index(drop=True)
obs_train=(pd.read_excel(r"result\1st_layer\predict(hbv-ae-ann)2.xlsx",sheet_name="obs_train_%s"%(forecast_timestep)).iloc[2:,1:]).reset_index(drop=True)

#%%
"""Adjust forecast"""
adjust_ratio = pred_train.iloc[:-1,:].reset_index(drop=True)/obs_train.iloc[:-1,:].reset_index(drop=True)
pred_train_adjust = (pred_train.iloc[1:,:].reset_index(drop=True))/adjust_ratio
pred_mean = np.mean(pred_train_adjust,axis=0)
pred_std = np.std(pred_train_adjust,axis=0)

for j in range(0,pred_train_adjust.shape[1]):
    for i in range(0,len(pred_train_adjust)):
        if pred_train_adjust.iloc[i,j]>pred_mean[j]+3*pred_std[j] or pred_train_adjust.iloc[i,j]<pred_mean[j]-3*pred_std[j]:
            pred_train_adjust.iloc[i,j]=np.nan
    
def nan_helper(data_array):
    return np.isnan(data_array), lambda z: z.nonzero()[0]

def interpolate(data):                    
    c_array=np.array(data)
    nans, x=nan_helper(c_array)
    c_array[nans]= np.interp(x(nans), x(~nans), c_array[~nans].astype(np.float))    
    return c_array

pred_train_adjust=pd.DataFrame(interpolate(pred_train_adjust))
simulate_train=simulate_train.iloc[1:,:].reset_index(drop=True)
obs_train=obs_train.iloc[1:,:].reset_index(drop=True)

#%%
"""time series compare"""

path1=r'D:\lab\research\research_use_function'
os.chdir(path1)
from error_indicator import error_indicator 
os.chdir(path)

"""convert to monthly"""
G0 = pd.read_csv(r"daily_data\groundwater_l0.csv",delimiter=',');G0['date'] = pd.to_datetime(G0['date']);
G0 = G0.resample('1M',on='date',base=0,loffset='1M').mean() #convert daily data to 10day based
"""choose date"""
date=pd.DataFrame(G0.index)
end_date='2017-01-01'
train_date=(date[(date['date']<end_date)]).reset_index(drop=True)
choose_date = (train_date.iloc[3+forecast_timestep:-4+forecast_timestep]).reset_index(drop=True)

#%%
"""performance compare"""

pred_R2=[];simulate_R2=[];pred_RMSE=[];simulate_RMSE=[]
for i in range(0,pred_train_adjust.shape[1]):
    pred_R2.append(error_indicator.np_R2(obs_train.iloc[:,i],pred_train_adjust.iloc[:,i]))
    simulate_R2.append(error_indicator.np_R2(obs_train.iloc[:,i],simulate_train.iloc[:,i]))
    pred_RMSE.append(error_indicator.np_RMSE(obs_train.iloc[:,i],pred_train_adjust.iloc[:,i]))
    simulate_RMSE.append(error_indicator.np_RMSE(obs_train.iloc[:,i],simulate_train.iloc[:,i])) 
    
pred_R2=np.array(pred_R2)
simulate_R2=np.array(simulate_R2)
pred_RMSE=np.array(pred_RMSE)
simulate_RMSE=np.array(simulate_RMSE)

all_performance=pd.DataFrame(np.transpose(np.stack([pred_R2, pred_RMSE, simulate_R2, simulate_RMSE],axis=0)))
all_performance.columns = ['pred_R2','pred_RMSE','simulate_R2','simulate_RMSE']

#%%
def merge_data(data,station_info):
    merged_data=pd.concat([station_info.iloc[:,[0,3]],data],axis=1).reset_index()
    merged_data.columns=np.concatenate([['index','測站','區域'],[i for i in data.columns]])
    merged_data=merged_data.sort_values(by=['區域','index']).reset_index(drop=True)
    merged_data=merged_data.iloc[:,1:]
    return merged_data

all_performance_=merge_data(all_performance,station_info)

#%%
""" save as result"""
path4='result\\1st_layer\\performance_comparison\\train'
isExist = os.path.exists(path4)
if not isExist:
  # Create a new directory because it does not exist 
  os.makedirs(path4)
  print("The new directory is created!")
writer = pd.ExcelWriter(path4+"\\T+%s.xlsx"%forecast_timestep,engine='xlsxwriter')

all_performance_.to_excel(writer,sheet_name="all_performance")
obs_train_=pd.concat([choose_date,obs_train],axis=1);obs_train_.columns=np.concatenate([['date'],[i for i in station_info.iloc[:,0]]])
obs_train_.to_excel(writer,sheet_name="obs")
pred_train_adjust_=pd.concat([choose_date,pred_train_adjust],axis=1); pred_train_adjust_.columns=np.concatenate([['date'],[i for i in station_info.iloc[:,0]]])
pred_train_adjust_.to_excel(writer,sheet_name="HBV-AE-LSTM")
simulate_train_=pd.concat([choose_date,simulate_train],axis=1); simulate_train_.columns=np.concatenate([['date'],[i for i in station_info.iloc[:,0]]])
simulate_train_.to_excel(writer,sheet_name="HBV")
writer.save()

#%%

from matplotlib import pyplot as plt
path2='result\\1st_layer\\time series figure\\train\\T+%s\\2000-01-01'%(forecast_timestep)
isExist = os.path.exists(path2)
if not isExist:
  # Create a new directory because it does not exist 
  os.makedirs(path2)
  print("The new directory is created!")

for station in range(0,33):
    hbv_ann_R2=round(error_indicator.np_R2(obs_train.iloc[:,station],pred_train_adjust.iloc[:,station]),2)
    hbv_ann_RMSE=round(error_indicator.np_RMSE(obs_train.iloc[:,station],pred_train_adjust.iloc[:,station]),2)
    
    hbv_R2=round(error_indicator.np_R2(obs_train.iloc[:,station],simulate_train.iloc[:,station]),2)
    hbv_RMSE=round(error_indicator.np_RMSE(obs_train.iloc[:,station],simulate_train.iloc[:,station]),2)
    
    fig = plt.figure(figsize=(10,6))
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']   
    plt.rcParams['axes.unicode_minus']=False  
    plt.style.use('ggplot')
    plt.title('%s Station'%station_info.iloc[station,0])
    plt.plot(choose_date,obs_train.iloc[:,station], color='black', linewidth=1.5)
    plt.plot(choose_date,pred_train_adjust.iloc[:,station], color='red', linewidth=0.8)

    plt.plot(choose_date,simulate_train.iloc[:,station], color='blue', linewidth=0.8)
    plt.ylabel('Groundwater Level'); plt.xlabel('Groundwater Level')
    # plt.show()
    plt.savefig(path2+'\\%s.png'%station_info.iloc[station,0])
    plt.clf()



#%%
import seaborn as sns
from matplotlib import pyplot as plt

barWidth = 0.25

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
    
    data1= np.concatenate((performance.iloc[:,0],[performance.iloc[0,0]]))
    data2= np.concatenate((performance.iloc[:,1],[performance.iloc[0,1]]))
    labels = np.concatenate((performance.index+1,[performance.index[0]+1]))
    
    ax.plot(angles, data1, 'ro-', linewidth=0.5, markersize=1, label='HBV-LSTM')# 畫線四個引數為x,y,標記和顏色，閒的寬度
    ax.plot(angles, data2, 'bo-', linewidth=0.5, markersize=1,label='HBV')# 畫線四個引數為x,y,標記和顏色，閒的寬度
    ax.set_thetagrids(angles * 180/np.pi, labels, fontproperties="Times New Roman")
    ax.tick_params(labelsize=8)
    plt.yticks(fontname = "Times New Roman") 
    ax.set_title('%s'%name,color="black", fontsize=10, fontproperties="Times New Roman")

    # plt.legend()
    if 'R2' in name:
        ax.set_rlim([0, 1.1])
    else:
        ax.set_rlim([0, 7])    
    ax.set_xlabel('Station no.', fontsize=8, fontproperties="Times New Roman")
    ax.yaxis.label.set_size(8)  
    
    path3 ='result\\1st_layer\\performance figure\\train\\T+%s'%(forecast_timestep)
    isExist = os.path.exists(path3)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(path3)
      print("The new directory is created!")
    
    plt.savefig(path3+"\\%s(radar).jpeg"%name)
    # Show Plot
    plt.show()
    plt.clf()

plot_figure(all_performance.iloc[:,[0,2]],'training R2(T+%s)'%forecast_timestep)
plot_figure(all_performance.iloc[:,[1,3]],'training RMSE(T+%s)'%forecast_timestep)

    
# plot_figure(all_performance.iloc[:,1],'Pred training rmse(T+1)')
# plot_figure(all_performance.iloc[:,3],'Simulate training rmse(T+1)')
