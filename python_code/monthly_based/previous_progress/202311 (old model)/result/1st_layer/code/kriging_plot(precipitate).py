# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 17:40:36 2022

@author: Steve
"""

import pandas as pd
import numpy as np
import os
path=r'D:\lab\research\groundwater_estimation\github\python_code\monthly_based'
os.chdir(path)

""" Read Center Weather Bureau(CWB) Dataset """
import pandas as pd
import numpy as np

## Precipitation Dataset
P_path = r"daily_data\rainfall.csv"
P = pd.read_csv(P_path);P['date'] = pd.to_datetime(P['date']);
P = P.resample('1M',on='date',base=0,loffset='1M').sum() #convert daily data to monthly based
P_station = P.columns
P = np.array(P)
P_station_info = pd.read_csv(r"station_info\rainfall_station.csv") #get rainfall station info

## Temperature Dataset
T_path = r"daily_data\temperature.csv"
T = pd.read_csv(T_path);T['date'] = pd.to_datetime(T['date']);
T = T.resample('1M',on='date',base=0,loffset='1M').mean() #convert daily data to monthly based
T_station = T.columns
T = np.array(T)
T_station_info = pd.concat([P_station_info.iloc[P_station_info[P_station_info.iloc[:,0]==T_station[i]].index,:] for i in range(0,len(T_station))])
T_station_info.columns = P_station_info.columns #get rainfall station info

## Evaporation Dataset
ETpot_path = r"daily_data\evaporation_rate.csv"
ETpot = pd.read_csv(ETpot_path);ETpot['date'] = pd.to_datetime(ETpot['date']);
ETpot = ETpot.resample('1M',on='date',base=0,loffset='1M').mean() #convert daily data to monthly based
ETpot_station = ETpot.columns
ETpot = np.array(ETpot)
ETpot_station_info = pd.concat([P_station_info.iloc[P_station_info[P_station_info.iloc[:,0]==ETpot_station[i]].index,:] for i in range(0,len(ETpot_station))])
ETpot_station_info.columns = P_station_info.columns  

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
"""Read Water Resource Agency(WRA) Groundwater Dataset"""

## Groundwater well info
G_detail_info= pd.read_csv(r"station_info\well_info.txt",encoding='utf-16',delimiter='\t')
G_station_info = pd.read_csv(r"station_info\groundwater_station.csv",delimiter='\t')

## Groundwater level 0 data
G0 = pd.read_csv(r"daily_data\groundwater_l0.csv",delimiter=',');G0['date'] = pd.to_datetime(G0['date']);
G0 = G0.resample('1M',on='date',base=0,loffset='1M').mean() #convert daily data to 10day based
G0_station = G0.columns
G0_station = pd.Series([G0_station[i][8:] for i in range(0,len(G0_station))])
G0_station_name = remain_onlyname(G0_station)
G0_station_info = get_station_info(G_station_info,G0_station_name)

G0_new = merge_samename(G0_station_name,G0)
G0_station_info_new = G0_station_info.drop_duplicates(keep="first").reset_index(drop=True)

## Groundwater level 1 data
G1 = pd.read_csv(r"daily_data\groundwater_l1.csv",delimiter=',');G1['date'] = pd.to_datetime(G1['date']);
G1 = G1.resample('1M',on='date',base=0,loffset='1M').mean() #convert daily data to 10day based
G1_station = G1.columns
G1_station = pd.Series([G1_station[i][8:] for i in range(0,len(G1_station))])
G1_station_name = remain_onlyname(G1_station)
G1_station_info = get_station_info(G_station_info,G1_station_name)

G1_new = merge_samename(G1_station_name,G1)
G1_station_info_new = G1_station_info.drop_duplicates(keep="first").reset_index(drop=True)

## Combine level 0 and 1 data
G01_new = np.concatenate([G0_new,G1_new],axis=1)
G01_station_info_new = pd.DataFrame(np.concatenate([G0_station_info_new,G1_station_info_new],axis=0))
G01_station_info_new.columns = ['station_name','X','Y']

G_station_info = pd.read_csv(r"station_info\groundwater_station.csv",delimiter='\t')

time = G0.index;
time = pd.DataFrame(time)

#%%
"""Other levels groundwater station (currently unused)"""
## Groundwater level 2 to 4 data 
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

P_station_info = pd.read_csv(r"station_info\rainfall_station.csv")

#%%
"""choose date"""

start_date='2000-01-01';end_date='2016-12-31' 

choose_date = (time[(time['date']>=start_date) & (time['date']<=end_date)].iloc[:,0]).reset_index(drop=True)
select_index= time[(time['date']>=start_date) & (time['date']<=end_date)].index

#%%
"""choose prediction event"""

obs_event = P[select_index,:]

#%%
import math
from pykrige.ok import OrdinaryKriging

G_grid_lon = np.linspace(math.floor(min(P_station_info.loc[:, 'X'])), math.ceil(max(P_station_info.loc[:, 'X'])), 30)
G_grid_lat = np.linspace(math.floor(min(P_station_info.loc[:, 'Y'])), math.ceil(max(P_station_info.loc[:, 'Y'])), 30)

# G_grid_lon = np.linspace(math.floor(min(G_station_info.loc[:, 'X'])), math.ceil(max(G_station_info.loc[:, 'X'])), 30)
# G_grid_lat = np.linspace(math.floor(min(G_station_info.loc[:, 'Y'])), math.ceil(max(G_station_info.loc[:, 'Y'])), 30)

X,Y = np.meshgrid(G_grid_lon,G_grid_lat)

#%%
"""Find grid location of gw well at level 0 and 1 """
G_name = pd.Series(np.concatenate([G0_station_name, G1_station_name, G2_station_name, G3_station_name, G4_station_name]))
G_unique = G_name.drop_duplicates(keep="first").reset_index(drop=True)
G_unique_station_info = pd.concat([G_station_info.iloc[G_station_info[G_station_info.iloc[:,0]==G_unique[i]].index,:] for i in range(0,len(G_unique))])
G_unique_station_info.columns = G_station_info.columns 

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
for i in range(0,len(G0_station_info_new)+len(G1_station_info_new)):
    all_info.append(get_specific_coordinate(G_unique_station_info.iloc[i,1],G_unique_station_info.iloc[i,2],X,Y))

all_coordinate=[];min_index=[];
all_coordinate=np.squeeze(np.array([all_info[i][0] for i in range(0,len(all_info))]))
## grid location of each well
min_index=np.squeeze(np.array([all_info[i][1] for i in range(0,len(all_info))])) 

#%%

"""IDW interpolation"""
def simple_idw(x, y, z, xi, yi, power=1):
    """ Simple inverse distance weighted (IDW) interpolation 
    Weights are proportional to the inverse of the distance, so as the distance
    increases, the weights decrease rapidly.
    The rate at which the weights decrease is dependent on the value of power.
    As power increases, the weights for distant points decrease rapidly.
    """
    def distance_matrix(x0, y0, x1, y1):
        """ Make a distance matrix between pairwise observations.
        Note: from <http://stackoverflow.com/questions/1871536> 
        """
        x1, y1 = x1.flatten(), y1.flatten()
        obs = np.vstack((x0, y0)).T
        interp = np.vstack((x1, y1)).T

        d0 = np.subtract.outer(obs[:,0], interp[:,0])
        d1 = np.subtract.outer(obs[:,1], interp[:,1])
        
        # calculate hypotenuse
        # result = np.hypot(d0, d1)
        result = np.sqrt(((d0 * d0) + (d1 * d1)).astype(float))
        return result
    
    dist = distance_matrix(x,y, xi,yi)

    # In IDW, weights are 1 / distance
    weights = 1.0/(dist+1e-12)**power

    # Make weights sum to one
    weights /= weights.sum(axis=0)

    # Multiply the weights for each interpolated point by all observed Z-values
    return np.dot(weights.T, z)

def IDW_interpolation(data,station_info,grid_lon,grid_lat):
    xi, yi = np.meshgrid(grid_lon,grid_lat)
    Z = simple_idw(np.array(station_info.loc[:, 'X']),np.array(station_info.loc[:, 'Y']), data, xi, yi, power=5)
    Z = Z.reshape((xi.shape[0]),(yi.shape[0]))
    return Z

def Ordinary_Kriging(data,station_info,grid_lon,grid_lat):
    OK = OrdinaryKriging(station_info.loc[:, 'X'],station_info.loc[:, 'Y'], data, variogram_model="gaussian") 
    # Execute on grid:
    z1, ss1 = OK.execute("grid", grid_lon, grid_lat)
    return z1,ss1

selected_P=[]
for i in range(0,len(select_index)):
    obs_train_z = IDW_interpolation(obs_event[i,:],P_station_info,G_grid_lon,G_grid_lat)
    obs_train_z = np.nan_to_num(obs_train_z)
    selected_P.append(obs_train_z[min_index[:,0],min_index[:,1]])

selected_P=np.array(selected_P)

#%%
import matplotlib.pyplot as plt
import os
import gc

sorted_station=pd.read_excel("data_statistic(sort_std).xlsx",sheet_name="l1")
sort_index=[G01_station_info_new[G01_station_info_new.iloc[:,0]==sorted_station.iloc[i,2]].index[0] for i in range(0,len(sorted_station))]

for i in range(0,len(sorted_station)):
    data=selected_P[:,sort_index[i]]
    plt.plot(choose_date,data)
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('%s'%sorted_station.iloc[i,2])
    plt.savefig(r'result\rainfall\train\time_series\%s.png'%sorted_station.iloc[i,2])
    plt.close()
    plt.cla()
    plt.clf()
    gc.collect()  # 清理站存記

