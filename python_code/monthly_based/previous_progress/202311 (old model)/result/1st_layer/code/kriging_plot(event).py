# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 17:06:40 2022

@author: steve
"""


import pandas as pd
import numpy as np
import os
path=r'D:\important\research\groundwater_forecast(monthly)'
os.chdir(path)

forecast_timestep='t+1'
obs_train=pd.read_excel(r"D:\important\research\groundwater_forecast(monthly)\result\1st_layer\sorted_observation.xlsx",sheet_name="%s(train)"%forecast_timestep,index_col=0)
pred_train=pd.read_excel(r"D:\important\research\groundwater_forecast(monthly)\result\1st_layer\sorted_predict(hbv-ann).xlsx",sheet_name="%s(shuffle)"%forecast_timestep,index_col=0)
simulate_train=(pd.read_excel(r"D:\important\research\groundwater_forecast(monthly)\result\1st_layer\sorted_predict.xlsx",sheet_name="%s(shuffle)"%forecast_timestep,index_col=0)).iloc[1:,:]

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
G_station_info = pd.read_csv(r"D:\important\research\groundwater_forecast\station_info\groundwater_station.csv",delimiter='\t')

G0 = pd.read_csv(r"D:\important\research\groundwater_forecast\daily_data\groundwater_l0.csv",delimiter=',');G0['date'] = pd.to_datetime(G0['date']);
G0_ = G0.resample('1M',on='date',base=0,loffset='1M').mean()
# G0_ = G0.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
G0_station = G0.columns
G0_station = pd.Series([G0_station[i][8:10] for i in range(1,len(G0_station))])
G0_station = G0_station.drop_duplicates (keep='first').reset_index(drop=True)

G1 = pd.read_csv(r"D:\important\research\groundwater_forecast\daily_data\groundwater_l1.csv",delimiter=',');G1['date'] = pd.to_datetime(G1['date']);
G1_ = G1.resample('1M',on='date',base=0,loffset='1M').mean()
G1_station = G1.columns
G1_station = pd.Series([G1_station[i][8:10] for i in range(1,len(G1_station))])
G1_station = G1_station.drop_duplicates (keep='first').reset_index(drop=True)

G01_station = np.concatenate([G0_station,G1_station])

G_station_info = pd.read_csv(r"D:\important\research\groundwater_forecast\station_info\groundwater_station.csv",delimiter='\t')
G01_station_info = pd.DataFrame(np.squeeze(np.array([G_station_info.iloc[G_station_info[G_station_info.iloc[:,0]==G01_station[i]].index[0],:] for i in range(0,len(G01_station))])))
G01_station_info.columns = ['station_name','X','Y']

time = G0_.index; train_time = pd.DataFrame(time[:len(obs_train)])

#%%

P_station_info = pd.read_csv(r"D:\important\research\groundwater_forecast\station_info\rainfall_station.csv")

#%%
"""choose date"""

# start_date='2000-01-01';end_date='2007-12-31'
# start_date='2008-01-01';end_date='2016-12-31' 

# start_date='2000-07-31';end_date='2000-08-31' #碧利斯
# start_date='2001-08-31';end_date='2001-10-01' #納莉颱風
# start_date='2003-07-31';end_date='2003-08-31' #莫拉克
# start_date='2007-07-31';end_date='2007-08-31' #海園
# start_date='2008-06-30';end_date='2008-07-31' #鳳凰
# start_date='2011-04-30';end_date='2011-05-31' #艾利、桑達
# start_date='2012-07-31';end_date='2012-08-31' #鳳凰
# start_date='2013-07-31';end_date='2013-08-31' #潭美都、康芮
# start_date='2014-06-30';end_date='2014-07-31' #蘇力、麥德姆
start_date='2015-07-31';end_date='2015-08-31' #蘇迪勒
# start_date='2016-06-30';end_date='2016-07-31' #尼伯特

choose_date = (train_time[(train_time['date']>=start_date) & (train_time['date']<=end_date)].iloc[:,0]).reset_index(drop=True)
select_index= train_time[(train_time['date']>=start_date) & (train_time['date']<=end_date)].index

#%%
"""choose prediction event"""
pred_event = pred_train.iloc[select_index[0],:]-pred_train.iloc[select_index[-1],:]
simulate_event = simulate_train.iloc[select_index[0],:]-simulate_train.iloc[select_index[-1],:]
obs_event = obs_train.iloc[select_index[0],:]-obs_train.iloc[select_index[-1],:]

#%%
import math
from pykrige.ok import OrdinaryKriging

G_grid_lon = np.linspace(math.floor(min(P_station_info.loc[:, 'X'])), math.ceil(max(P_station_info.loc[:, 'X'])), 30)
G_grid_lat = np.linspace(math.floor(min(P_station_info.loc[:, 'Y'])), math.ceil(max(P_station_info.loc[:, 'Y'])), 30)

# G_grid_lon = np.linspace(math.floor(min(G_station_info.loc[:, 'X'])), math.ceil(max(G_station_info.loc[:, 'X'])), 30)
# G_grid_lat = np.linspace(math.floor(min(G_station_info.loc[:, 'Y'])), math.ceil(max(G_station_info.loc[:, 'Y'])), 30)

X,Y = np.meshgrid(G_grid_lon,G_grid_lat)

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

pred_train_z = IDW_interpolation(pred_event,G01_station_info,G_grid_lon,G_grid_lat)
pred_train_z = np.nan_to_num(pred_train_z)

# pred_train_z = Ordinary_Kriging(pred_event,G01_station_info,G_grid_lon,G_grid_lat)[0]
# pred_train_z = np.nan_to_num(pred_train_z)

obs_train_z = IDW_interpolation(obs_event,G01_station_info,G_grid_lon,G_grid_lat)
obs_train_z = np.nan_to_num(obs_train_z)

# obs_train_z = Ordinary_Kriging(obs_event,G01_station_info,G_grid_lon,G_grid_lat)[0]
# obs_train_z = np.nan_to_num(obs_train_z)

simulate_train_z = IDW_interpolation(simulate_event,G01_station_info,G_grid_lon,G_grid_lat)
simulate_train_z = np.nan_to_num(simulate_train_z)

# simulate_train_z = Ordinary_Kriging(simulate_event,G01_station_info,G_grid_lon,G_grid_lat)[0]
# simulate_train_z = np.nan_to_num(simulate_train_z)

#%%
selected_grid=[(6,2),
               (5,3),(6,3),(7,3),
               (5,4),(6,4),(7,4),
               (4,5),(5,5),(6,5),(7,5),(8,5),
               (4,6),(5,6),(6,6),(7,6),(8,6),
               (2,7),(3,7),(4,7),(5,7),(6,7),(7,7),(8,7),(9,7),(10,7),(11,7),(12,7),(13,7),(14,7),(15,7),(16,7),(17,7),
               (2,8),(3,8),(4,8),(5,8),(6,8),(7,8),(8,8),(9,8),(10,8),(11,8),(12,8),(13,8),(14,8),(15,8),(16,8),(17,8),
               (2,9),(3,9),(4,9),(5,9),(6,9),(7,9),(8,9),(9,9),(10,9),(11,9),(12,9),(13,9),(14,9),(15,9),(16,9),(17,9),
               (1,10),(2,10),(3,10),(4,10),(5,10),(6,10),(7,10),(8,10),(9,10),(10,10),(11,10),(12,10),(13,10),(14,10),(15,10),(16,10),(17,10),
               (1,11),(2,11),(3,11),(4,11),(5,11),(6,11),(7,11),(8,11),(9,11),(11,11),(11,11),(12,11),(13,11),(14,11),(15,11),(16,11),
               (11,12),(12,12),(13,12),(14,12)]
selected_grid = [tuple(np.array(i)-1) for i in selected_grid]

pred_sum = np.sum([pred_train_z[i] for i in selected_grid])
obs_sum = np.sum([obs_train_z[i] for i in selected_grid])

#%%
import os
os.chdir(r"D:\important\research\research_use_function")
from error_indicator import error_indicator
path=r'D:\important\research\groundwater_forecast(monthly)'
os.chdir(path)
pred_train_RMSE = round(error_indicator.np_RMSE((obs_train_z).flatten(),(pred_train_z).flatten()),2)
simulate_train_RMSE = round(error_indicator.np_RMSE((obs_train_z).flatten(),(simulate_train_z).flatten()),2)
pred_train_R2 = round(error_indicator.np_R2((obs_train_z).flatten(),(pred_train_z).flatten()),2)
simulate_train_R2 = round(error_indicator.np_R2((obs_train_z).flatten(),(simulate_train_z).flatten()),2)
obs_mean=round(np.mean(obs_train_z),2); obs_std=round(np.std(obs_train_z),2)

#%%
"""plot kriging"""

# Plot
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import shapefile
import os
import gc

def shp_clip(originfig, ax, shpfile):
    sf = shapefile.Reader(shpfile)
    vertices = []
    codes = []
    for shape_rec in sf.shapeRecords():
        pts = shape_rec.shape.points
        prt = list(shape_rec.shape.parts) + [len(pts)]
        for i in range(len(prt) - 1):
            for j in range(prt[i], prt[i + 1]):
                vertices.append((pts[j][0], pts[j][1]))
            codes += [Path.MOVETO]                               # MOVETO 表示多邊形的開始點
            codes += [Path.LINETO] * (prt[i + 1] - prt[i] - 2)   # LINETO 表示創建每一段直線
            codes += [Path.CLOSEPOLY]                            # CLOSEPOLY 表示閉合多邊形
        clip = Path(vertices, codes)
        clip = PathPatch(clip, transform=ax.transData)
    for contour in originfig.collections:
        contour.set_clip_path(clip)
    return contour


# min_value= np.min([np.min(obs_train_z),np.min(pred_train_z),np.min(simulate_train_z)])
# max_value= np.max([np.max(obs_train_z),np.max(pred_train_z),np.max(simulate_train_z)])

# if np.absolute(min_value)>np.absolute(max_value):
#     max_value = -min_value
# else:
#     min_value = -max_value
    
    
min_value = -10
max_value = 10

#%%
predictions_array = pred_train_z
simulate_array= simulate_train_z
obs_array = obs_train_z
filename=str(np.array(train_time.iloc[select_index[0]])[0])[0:10]

fig = plt.figure(figsize=(16, 9),dpi=300)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([120.05, 121.4, 23.4, 24.25], crs=ccrs.PlateCarree())


shp_file = shpreader.Reader(r'D:\important\research\groundwater_forecast\zhuoshui(shp)\zhuoshui.shp')
ax.add_geometries(shp_file.geometries(), crs=ccrs.PlateCarree(), linewidths=1.2, edgecolor='k', facecolor='none') # edgecolor=調整邊框顏色
cf = ax.contourf(X, Y, predictions_array, levels=np.linspace(min_value,max_value,40), cmap='seismic_r', transform=ccrs.PlateCarree())

shp_clip(cf, ax, r'D:\important\research\groundwater_forecast\zhuoshui(shp)\zhuoshui.shp')

plt.rcParams['font.sans-serif'] =['Taipei Sans TC Beta']  #匯入中文字體
cb = plt.colorbar(cf)
lons = np.array(G01_station_info.loc[:,'X']); lats = G01_station_info.loc[:,'Y']; station_name = G01_station
plt.scatter(lons, lats, color='navy', s=15)  # s是調整座標點大小
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
# for i in range(len(lons)):        
#     plt.text(lons[i]+0.05,lats[i]+0.02,station_name[i], color='k', fontsize=12)  # 將測站名稱標示在座標點上
#     plt.axis('off')  # 去除圖片外框
plt.axis('off')  # 去除圖片外框
plt.title('prediction time %s (RMSE=%s , R2=%s)'%(filename,pred_train_RMSE,pred_train_R2), fontsize=20)
    # output_path=r'important\research\groundwater_forecast\python_code'
output_path=r'D:\important\research\groundwater_forecast(monthly)\result\1st_layer\kriging event figure\%s\train\regional_predict\%s'%(forecast_timestep,filename)
plt.savefig(os.path.join(output_path+'.png'), bbox_inches='tight')  # bbox_inches是修改成窄邊框
plt.close()
plt.cla()
plt.clf()
gc.collect()  # 清理站存記

fig = plt.figure(figsize=(16, 9),dpi=300)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([120.05, 121.4, 23.4, 24.25], crs=ccrs.PlateCarree())

shp_file = shpreader.Reader(r'D:\important\research\groundwater_forecast\zhuoshui(shp)\zhuoshui.shp')
ax.add_geometries(shp_file.geometries(), crs=ccrs.PlateCarree(), linewidths=1.2, edgecolor='k', facecolor='none') # edgecolor=調整邊框顏色
cf = ax.contourf(X, Y, simulate_array, levels=np.linspace(min_value,max_value,40), cmap='seismic_r', transform=ccrs.PlateCarree())

shp_clip(cf, ax, r'D:\important\research\groundwater_forecast\zhuoshui(shp)\zhuoshui.shp')

plt.rcParams['font.sans-serif'] =['Taipei Sans TC Beta']  #匯入中文字體
cb = plt.colorbar(cf)
lons = np.array(G01_station_info.loc[:,'X']); lats = G01_station_info.loc[:,'Y']; station_name = G01_station
plt.scatter(lons, lats, color='navy', s=15)  # s是調整座標點大小
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
# for i in range(len(lons)):        
#     plt.text(lons[i]+0.05,lats[i]+0.02,station_name[i], color='k', fontsize=12)  # 將測站名稱標示在座標點上
#     plt.axis('off')  # 去除圖片外框
plt.axis('off')  # 去除圖片外框
plt.title('simulate time %s (RMSE=%s , R2=%s)'%(filename,simulate_train_RMSE,simulate_train_R2), fontsize=20)
    # output_path=r'important\research\groundwater_forecast\python_code'
output_path=r'D:\important\research\groundwater_forecast(monthly)\result\1st_layer\kriging event figure\%s\train\regional_simulate\%s'%(forecast_timestep,filename)
plt.savefig(os.path.join(output_path+'.png'), bbox_inches='tight')  # bbox_inches是修改成窄邊框
plt.close()
plt.cla()
plt.clf()
gc.collect()  # 清理站存記

fig = plt.figure(figsize=(16, 9),dpi=300)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([120.05, 121.4, 23.4, 24.25], crs=ccrs.PlateCarree())

ax.add_geometries(shp_file.geometries(), crs=ccrs.PlateCarree(), linewidths=1.2, edgecolor='k', facecolor='none') # edgecolor=調整邊框顏色
cf = ax.contourf(X, Y, obs_array, levels=np.linspace(min_value,max_value,40), cmap='seismic_r', transform=ccrs.PlateCarree())
shp_clip(cf, ax, r'D:\important\research\groundwater_forecast\zhuoshui(shp)\zhuoshui.shp')

plt.rcParams['font.sans-serif'] =['Taipei Sans TC Beta']  #匯入中文字體
cb = plt.colorbar(cf)
lons = np.array(G01_station_info.loc[:,'X']); lats = G01_station_info.loc[:,'Y']; station_name = G01_station
plt.scatter(lons, lats, color='navy', s=15)  # s是調整座標點大小
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
# for i in range(len(lons)):        
#     plt.text(lons[i]+0.05,lats[i]+0.02,station_name[i], color='k', fontsize=12)  # 將測站名稱標示在座標點上
#     plt.axis('off')  # 去除圖片外框
plt.axis('off')  # 去除圖片外框
plt.title('obs time %s (%s,%s)'%(filename,obs_mean,obs_std), fontsize=20)
    # output_path=r'important\research\groundwater_forecast\python_code'
output_path=r'D:\important\research\groundwater_forecast(monthly)\result\1st_layer\kriging event figure\%s\train\regional_obs\%s'%(forecast_timestep,filename)
plt.savefig(os.path.join(output_path+'.png'), bbox_inches='tight')  # bbox_inches是修改成窄邊框
plt.close()
plt.cla()
plt.clf()
gc.collect()  # 清理站存記

