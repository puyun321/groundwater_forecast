# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 13:58:25 2022

@author: steve
"""

import pandas as pd
import numpy as np

#%%
P_path = r"D:\important\research\groundwater_forecast\daily_data\rainfall.csv"
# P = pd.read_csv(P_path,index_col=0)
P = pd.read_csv(P_path);P['date'] = pd.to_datetime(P['date']);
P = P.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
P_station = P.columns
P = np.array(P)
P_station_info = pd.read_csv(r"D:\important\research\groundwater_forecast\station_info\rainfall_station.csv")

T_path = r"D:\important\research\groundwater_forecast\daily_data\temperature.csv"
# T = pd.read_csv(T_path,index_col=0)
T = pd.read_csv(T_path);T['date'] = pd.to_datetime(T['date']);
T = T.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
T_station = T.columns
T = np.array(T)
T_station_info = pd.DataFrame(np.squeeze(np.array([P_station_info.iloc[P_station_info[P_station_info.iloc[:,0]==T_station[i]].index,:] for i in range(0,len(T_station))])))
T_station_info.columns = P_station_info.columns

ETpot_path = r"D:\important\research\groundwater_forecast\daily_data\evaporation_rate.csv"
# ETpot = pd.read_csv(ETpot_path,index_col=0)
ETpot = pd.read_csv(ETpot_path);ETpot['date'] = pd.to_datetime(ETpot['date']);
ETpot = ETpot.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
ETpot_station = ETpot.columns
ETpot = np.array(ETpot)
ETpot_station_info = pd.DataFrame(np.squeeze(np.array([P_station_info.iloc[P_station_info[P_station_info.iloc[:,0]==ETpot_station[i]].index,:] for i in range(0,len(ETpot_station))])))
ETpot_station_info.columns = P_station_info.columns  

#%%
#merge stations with same name
def merge_samename(G_station,G):
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
    G = np.array([G.iloc[:,station_index[i]].mean(axis=1) for i in range(0,len(station_index))]).T
    return G

def remain_onlyname(G_station):
    G_station_name=[]
    for i in range(0,len(G_station)):
        G_station_name.append(G_station[i][0:2])
    G_station_name=np.array(G_station_name)
    return G_station_name

def get_station_info(all_station_info,specific_station):
    specific_station_info = pd.DataFrame(np.squeeze(np.array([all_station_info.iloc[all_station_info[all_station_info.iloc[:,0]==specific_station[i]].index,:] for i in range(0,len(specific_station))])))
    specific_station_info.columns = all_station_info.columns
    return specific_station_info

#%%
well_info= pd.read_csv(r"D:\important\research\groundwater_forecast\station_info\well_info.txt",encoding='utf-16',delimiter='\t')
G_station_info = pd.read_csv(r"D:\important\research\groundwater_forecast\station_info\groundwater_station.csv")

G0 = pd.read_csv(r"D:\important\research\groundwater_forecast\daily_data\groundwater_l0.csv");G0['date'] = pd.to_datetime(G0['date']);
G0 = G0.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
G0_station = G0.columns
G0_station = pd.Series([G0_station[i][8:] for i in range(0,len(G0_station))])
G0_station_name = remain_onlyname(G0_station)
G0_station_info = get_station_info(G_station_info,G0_station_name)

G1 = pd.read_csv(r"D:\important\research\groundwater_forecast\daily_data\groundwater_l1.csv");G1['date'] = pd.to_datetime(G1['date']);
G1 = G1.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
G1_station = G1.columns
G1_station = pd.Series([G1_station[i][8:] for i in range(0,len(G1_station))])
G1_station_name = remain_onlyname(G1_station)
G1_station_info = get_station_info(G_station_info,G1_station_name)

G2 = pd.read_csv(r"D:\important\research\groundwater_forecast\daily_data\groundwater_l2.csv");G2['date'] = pd.to_datetime(G2['date']);
G2 = G2.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
G2_station = G2.columns
G2_station = pd.Series([G2_station[i][8:] for i in range(0,len(G2_station))])
G2_station_name = remain_onlyname(G2_station)
G2_station_info = get_station_info(G_station_info,G2_station_name)

G3 = pd.read_csv(r"D:\important\research\groundwater_forecast\daily_data\groundwater_l3.csv");G3['date'] = pd.to_datetime(G3['date']);
G3 = G3.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
G3_station = G3.columns
G3_station = pd.Series([G3_station[i][8:] for i in range(0,len(G3_station))])
G3_station_name = remain_onlyname(G3_station)
G3_station_info = get_station_info(G_station_info,G3_station_name)

G4 = pd.read_csv(r"D:\important\research\groundwater_forecast\daily_data\groundwater_l4.csv");G4['date'] = pd.to_datetime(G4['date']);
G4 = G4.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
G4_station = G4.columns
G4_station = pd.Series([G4_station[i][8:] for i in range(0,len(G4_station))])
G4_station_name = remain_onlyname(G4_station)
G4_station_info = get_station_info(G_station_info,G4_station_name)

G_name = pd.Series(np.concatenate([G0_station_name, G1_station_name, G2_station_name, G3_station_name, G4_station_name]))
G_merge_info = pd.Series.reset_index(pd.concat([G0_station, G1_station, G2_station, G3_station, G4_station]),drop=True)
level_no = pd.concat([pd.Series([0 for i in range(0,len(G0_station))]), pd.Series([1 for i in range(0,len(G1_station))]), pd.Series([2 for i in range(0,len(G2_station))]),
                      pd.Series([3 for i in range(0,len(G3_station))]), pd.Series([4 for i in range(0,len(G4_station))])]).reset_index(drop=True)

G = pd.concat([G0,G1,G2,G3,G4],axis=1)
G.columns = G_merge_info

#%%
G_info = pd.concat([G_merge_info,G_name,level_no],axis=1)
G_info = G_info.sort_values(by=[0])
G_unique = G_name.drop_duplicates(keep="first").reset_index(drop=True)
G_unique_station_info = pd.DataFrame(np.squeeze(np.array([G_station_info.iloc[G_station_info[G_station_info.iloc[:,0]==G_unique[i]].index,:] for i in range(0,len(G_unique))])))
G_unique_station_info.columns = G_station_info.columns 

G_unique_index = [[]*1 for i in range(0,len(G_unique))]
for i in range(0,len(G_unique)):
    G_unique_index[i] = G_info.iloc[G_info[G_info.iloc[:,1]==G_unique[i]].index, 2]


#%%
import math
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from scipy.interpolate import griddata

def Ordinary_Kriging(data,station_info,grid_lon,grid_lat):
    OK = OrdinaryKriging(station_info.loc[:, 'X'],station_info.loc[:, 'Y'], data, variogram_model="gaussian") 
    # Execute on grid:
    z1, ss1 = OK.execute("grid", grid_lon, grid_lat)
    return z1,ss1

def Universal_Kriging(data,station_info,grid_lon,grid_lat):
    OK = UniversalKriging(station_info.loc[:, 'X'],station_info.loc[:, 'Y'], data, variogram_model="exponential",drift_terms=["regional_linear"],) 
    # Execute on grid:
    z1, ss1 = OK.execute("grid", grid_lon, grid_lat)
    return z1,ss1

def interpolate2grid(data,station_info,grid_lon,grid_lat,interpolate_method='linear'):
    X,Y = np.meshgrid(grid_lon,grid_lat)
    Z = np.squeeze(griddata([(x,y) for x,y in zip(station_info.loc[:, 'X'],station_info.loc[:, 'Y'])], data, (X, Y), method=interpolate_method))
    return Z

P_grid_lon = np.linspace(math.floor(min(G_unique_station_info.loc[:, 'X'])), math.ceil(max(G_unique_station_info.loc[:, 'X'])), 30)
P_grid_lat = np.linspace(math.floor(min(G_unique_station_info.loc[:, 'Y'])), math.ceil(max(G_unique_station_info.loc[:, 'Y'])), 30)

""" Since the Kriging fail to provide a good interpolation, we applied the IDW interpolation method to grid our data"""

P_z = [interpolate2grid(pd.DataFrame(P[i, :]),P_station_info,P_grid_lon,P_grid_lat) for i in range(0,len(P))]
P_z = np.nan_to_num(P_z)
T_z = np.array([interpolate2grid(pd.DataFrame(T[i, :]),T_station_info,P_grid_lon,P_grid_lat,interpolate_method='nearest') for i in range(0,len(T))])
ETpot_z = np.array([interpolate2grid(pd.DataFrame(ETpot[i, :]),ETpot_station_info,P_grid_lon,P_grid_lat,interpolate_method='nearest') for i in range(0,len(ETpot))])
X,Y = np.meshgrid(P_grid_lon,P_grid_lat)

#%%
def get_specific_coordinate(x1,y1,x2,y2):
    distance=[[]*1 for i in range(0,x2.shape[0])]
    for i in range(0,x2.shape[0]):
        for j in range(0,x2.shape[1]):
            # distance[i].append(math.dist([x1,y1],[x2[i,j],y2[i,j]]))
            distance[i].append(((((x2[i,j] - x1 )**2) + ((y2[i,j]-y1)**2) )**0.5))

    distance=np.array(distance)
    min_distance=np.min(distance)
    min_index=np.where(distance==min_distance)
    coordinate=np.array([x2[min_index[0],min_index[1]],y2[min_index[0],min_index[1]]])

    return coordinate,min_index

all_info=[];
for i in range(0,len(G_unique_station_info)):
    all_info.append(get_specific_coordinate(G_unique_station_info.iloc[i,1],G_unique_station_info.iloc[i,2],X,Y))

all_coordinate=[];min_index=[];
all_coordinate=np.squeeze(np.array([all_info[i][0] for i in range(0,len(all_info))]))
min_index=np.squeeze(np.array([all_info[i][1] for i in range(0,len(all_info))]))