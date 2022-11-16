# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 16:41:17 2022

@author: steve
"""


import os
path=r'D:\important\research\groundwater_forecast(monthly)'
os.chdir(path)
import pandas as pd
import numpy as np


#%%
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

#%%
well_info= pd.read_csv(r"station_info\well_info.txt",encoding='utf-16',delimiter='\t')
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
G01_station_info_new = pd.concat([G0_station_info_new,G1_station_info_new],axis=0).reset_index(drop=True)

G2 = pd.read_csv(r"daily_data\groundwater_l2.csv",delimiter=',');G2['date'] = pd.to_datetime(G2['date']);
G2 = G2.resample('1M',on='date',base=0,loffset='1M').mean() #convert daily data to 10day based
G2_station = G2.columns
G2_station = pd.Series([G2_station[i][8:] for i in range(0,len(G2_station))])
G2_station_name = remain_onlyname(G2_station)
G2_station_info = get_station_info(G_station_info,G2_station_name)

G2_new = merge_samename(G2_station_name,G2)
G2_station_info_new = G2_station_info.drop_duplicates(keep="first").reset_index(drop=True)

G3 = pd.read_csv(r"daily_data\groundwater_l3.csv",delimiter=',');G3['date'] = pd.to_datetime(G3['date']);
G3 = G3.resample('1M',on='date',base=0,loffset='1M').mean() #convert daily data to 10day based
G3_station = G3.columns
G3_station = pd.Series([G3_station[i][8:] for i in range(0,len(G3_station))])
G3_station_name = remain_onlyname(G3_station)
G3_station_info = get_station_info(G_station_info,G3_station_name)

G3_new = merge_samename(G3_station_name,G3)
G3_station_info_new = G3_station_info.drop_duplicates(keep="first").reset_index(drop=True)


G4 = pd.read_csv(r"daily_data\groundwater_l4.csv",delimiter=',');G4['date'] = pd.to_datetime(G4['date']);
G4 = G4.resample('1M',on='date',base=0,loffset='1M').mean() #convert daily data to 10day based
G4_station = G4.columns
G4_station = pd.Series([G4_station[i][8:] for i in range(0,len(G4_station))])
G4_station_name = remain_onlyname(G4_station)
G4_station_info = get_station_info(G_station_info,G4_station_name)

G4_new = merge_samename(G4_station_name,G4)
G4_station_info_new = G4_station_info.drop_duplicates(keep="first").reset_index(drop=True)

#%%
G_name = pd.Series(np.concatenate([G0_station_name, G1_station_name, G2_station_name, G3_station_name, G4_station_name]))
G_unique = G_name.drop_duplicates(keep="first").reset_index(drop=True)
G_unique_station_info = pd.concat([G_station_info.iloc[G_station_info[G_station_info.iloc[:,0]==G_unique[i]].index,:] for i in range(0,len(G_unique))])
G_unique_station_info.columns = G_station_info.columns 

#%%
top=['田中','六合','烏塗','新光','新民','坪頂','社寮','竹山']
central=['國聖','東芳','花壇','好修','合興','田尾','港後','九隆','田洋','芳草','虎尾','虎溪','東和','宏菕','嘉興','溫厝','古坑','東光','舊莊','三和','崁脚','東榮','安和']
tail=['線西','顔厝','洛津','文昌','趙甲','豐榮','海豐','興化','海園','明德','箔子','大溝','金湖','蔡厝','水林','宜梧','瓊埔','東石頭']

#%%
def find_region(station_info):
    station_info=np.array(station_info)
    location=[]
    for i in range(0,len(station_info)):
        if station_info[i,0] in top:
            location.append('r1')
        elif station_info[i,0] in central:
            location.append('r2')
        else:
            location.append('r3')        
    location = pd.Series(location)

    return location

G01_location = find_region(G01_station_info_new)
G2_location = find_region(G2_station_info_new)
G3_location = find_region(G3_station_info_new)
G4_location = find_region(G4_station_info_new)

G01_station_info_new = pd.concat([G01_station_info_new,G01_location],axis=1)
G2_station_info_new = pd.concat([G2_station_info_new,G2_location],axis=1)
G3_station_info_new = pd.concat([G3_station_info_new,G3_location],axis=1)
G4_station_info_new = pd.concat([G4_station_info_new,G4_location],axis=1)


#%%

writer = pd.ExcelWriter(r"station_info\station_fullinfo.xlsx",engine='xlsxwriter')
G01_station_info_new.to_excel(writer,sheet_name='G1')
G2_station_info_new.to_excel(writer,sheet_name='G2')
G3_station_info_new.to_excel(writer,sheet_name='G3')
G4_station_info_new.to_excel(writer,sheet_name='G4')
writer.save()