# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 00:08:53 2023

@author: Steve
"""

"""change woking directory to current directory"""
import os

working= os.path.dirname(os.path.abspath('__file__')) #try run this if work

path=r'D:\lab\research\groundwater_estimation\github\python_code\monthly_based' #if the code above cant work, please key in your parent directory
os.chdir(path)

#%%
""" Read Forecast Result """
import pandas as pd
import numpy as np


"""test"""
obs1=pd.read_excel(r"result\1st_layer\performance_comparison\test\T+1.xlsx",sheet_name="obs")
hbvlstm1_o=pd.read_excel(r"result\1st_layer\performance_comparison\test\T+1.xlsx",sheet_name="HBV-AE-LSTM")
hbv1=pd.read_excel(r"result\1st_layer\performance_comparison\test\T+1.xlsx",sheet_name="HBV")

obs2=pd.read_excel(r"result\1st_layer\performance_comparison\test\T+2.xlsx",sheet_name="obs")
hbvlstm2_o=pd.read_excel(r"result\1st_layer\performance_comparison\test\T+2.xlsx",sheet_name="HBV-AE-LSTM")
hbv2=pd.read_excel(r"result\1st_layer\performance_comparison\test\T+2.xlsx",sheet_name="HBV")

obs3=pd.read_excel(r"result\1st_layer\performance_comparison\test\T+3.xlsx",sheet_name="obs")
hbvlstm3_o=pd.read_excel(r"result\1st_layer\performance_comparison\test\T+3.xlsx",sheet_name="HBV-AE-LSTM")
hbv3=pd.read_excel(r"result\1st_layer\performance_comparison\test\T+3.xlsx",sheet_name="HBV")

obs=obs1.iloc[2:,1:].reset_index(drop=True)
hbvlstm1=hbvlstm1_o.iloc[2:,1:].reset_index(drop=True)
hbvlstm2=hbvlstm2_o.iloc[1:-1,1:].reset_index(drop=True)
hbvlstm3=hbvlstm3_o.iloc[:-2,1:].reset_index(drop=True)

choose_date=obs1.iloc[2:,0].reset_index(drop=True)

#%%
""" month to season"""

season_datetime=['2017-04-30','2017-07-31','2017-10-31','2018-01-31','2018-04-30','2018-07-31','2018-10-31','2019-01-31','2019-04-30','2019-07-31','2019-10-31']
season_name=['2017 spring','2017 summer','2017 autumn','2018 winter','2018 spring','2018 summer','2018 autmn','2019 winter','2019 spring','2019 summer','2019 autumn']

import os
os.chdir(r"D:\lab\research\research_use_function")
from error_indicator import error_indicator
os.chdir(path)

performance_1=[];performance_2=[];performance_3=[]
performance=[[]*1 for k in range(len(season_datetime))]
for i in range(0,len(season_datetime)):
    if i==0:
        choose_index=choose_date[choose_date<=season_datetime[i]].index
    else:
        choose_index=choose_date[(choose_date<=season_datetime[i]) & (choose_date>season_datetime[i-1])].index
        
    select_obs= obs.iloc[choose_index,:]
    select_hbvlstm1= hbvlstm1.iloc[choose_index,:]
    select_hbvlstm2= hbvlstm2.iloc[choose_index,:]
    select_hbvlstm3= hbvlstm3.iloc[choose_index,:]

    performance1=[]; performance2=[]; performance3=[];
    for j in range(0,select_obs.shape[1]):
        performance1.append(error_indicator.np_RMSE(select_obs.iloc[:,j],select_hbvlstm1.iloc[:,j]))
        performance2.append(error_indicator.np_RMSE(select_obs.iloc[:,j],select_hbvlstm2.iloc[:,j]))
        performance3.append(error_indicator.np_RMSE(select_obs.iloc[:,j],select_hbvlstm3.iloc[:,j]))
        
    performance_mean=np.mean([performance1,performance2,performance3],axis=0)
    
    performance_1.append(performance1)
    performance_2.append(performance2)
    performance_3.append(performance3)
    performance[i].append(performance_mean)

performance_1=pd.DataFrame(np.array(performance_1)); performance_1.index=season_name
performance_2=pd.DataFrame(np.array(performance_2)); performance_2.index=season_name
performance_3=pd.DataFrame(np.array(performance_3)); performance_3.index=season_name
performance=pd.DataFrame(np.squeeze(np.array(performance))); performance.index=season_name

#%%
performance_1_R2=[];performance_2_R2=[];performance_3_R2=[]
performance_R2=[[]*1 for k in range(len(season_datetime))]
for i in range(0,len(season_datetime)):
    i=0
    if i==0:
        choose_index=choose_date[choose_date<=season_datetime[i]].index
        select_obs= obs.iloc[choose_index,:]
        select_hbvlstm1= hbvlstm1.iloc[choose_index,:]

        
    else:
        choose_index=choose_date[(choose_date<=season_datetime[i]) & (choose_date>season_datetime[i-1])].index
        
    select_obs= obs.iloc[choose_index,:]
    select_hbvlstm1= hbvlstm1.iloc[choose_index,:]
    select_hbvlstm2= hbvlstm2.iloc[choose_index,:]
    select_hbvlstm3= hbvlstm3.iloc[choose_index,:]

    performance1=[]; performance2=[]; performance3=[];
    for j in range(0,select_obs.shape[1]):
        performance1.append(error_indicator.np_R2(select_obs.iloc[:,j],select_hbvlstm1.iloc[:,j]))
        performance2.append(error_indicator.np_R2(select_obs.iloc[:,j],select_hbvlstm2.iloc[:,j]))
        performance3.append(error_indicator.np_R2(select_obs.iloc[:,j],select_hbvlstm3.iloc[:,j]))
        
    performance_mean=np.mean([performance1,performance2,performance3],axis=0)
    
    performance_1_R2.append(performance1)
    performance_2_R2.append(performance2)
    performance_3_R2.append(performance3)
    performance_R2[i].append(performance_mean)

performance_1_R2=pd.DataFrame(np.array(performance_1_R2)); performance_1_R2.index=season_name
performance_2_R2=pd.DataFrame(np.array(performance_2_R2)); performance_2_R2.index=season_name
performance_3_R2=pd.DataFrame(np.array(performance_3_R2)); performance_3_R2.index=season_name
performance_R2=pd.DataFrame(np.squeeze(np.array(performance_R2))); performance_R2.index=season_name

#%%
obs_mean=[[]*1 for k in range(len(season_datetime))]
obs_std=[[]*1 for k in range(len(season_datetime))]

for i in range(0,len(season_datetime)):
    if i==0:
        choose_index=choose_date[choose_date<=season_datetime[i]].index
    else:
        choose_index=choose_date[(choose_date<=season_datetime[i]) & (choose_date>season_datetime[i-1])].index
        
    select_obs= obs.iloc[choose_index,1:]
    obs_mean[i].append(np.mean(select_obs))
    obs_std[i].append(np.std(select_obs))   

obs_mean=pd.DataFrame(np.squeeze(np.array(obs_mean)));obs_mean.index=season_name
obs_std=pd.DataFrame(np.squeeze(np.array(obs_std)));obs_std.index=season_name
obs_std_mean=np.mean(obs_std,axis=1)


#%%
writer = pd.ExcelWriter(r"result\1st_layer\performance_comparison\test\performance(season).xlsx",engine='xlsxwriter')
performance_1.to_excel(writer,sheet_name="T+1")
performance_2.to_excel(writer,sheet_name="T+2")
performance_3.to_excel(writer,sheet_name="T+3")
performance.to_excel(writer,sheet_name="mean performance")
writer.save()

#%%
G_station_info = pd.read_excel(r"station_info\station_fullinfo.xlsx",sheet_name="G1",index_col=0) 
P_station_info = pd.read_csv(r"station_info\rainfall_station.csv")

#%%
import math
from pykrige.ok import OrdinaryKriging

G_grid_lon = np.linspace(math.floor(min(G_station_info.loc[:, 'X'])), math.ceil(max(G_station_info.loc[:, 'X'])), 30)
G_grid_lat = np.linspace(math.floor(min(G_station_info.loc[:, 'Y'])), math.ceil(max(G_station_info.loc[:, 'Y'])), 30)

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

performance_z = [IDW_interpolation(performance.iloc[i, :],G_station_info,G_grid_lon,G_grid_lat) for i in range(0,len(performance))]
performance_z = np.nan_to_num(performance_z)


performance_1_z = [IDW_interpolation(performance_1.iloc[i, :],G_station_info,G_grid_lon,G_grid_lat) for i in range(0,len(performance_1))]
performance_1_z = np.nan_to_num(performance_1_z)


performance_2_z = [IDW_interpolation(performance_2.iloc[i, :],G_station_info,G_grid_lon,G_grid_lat) for i in range(0,len(performance_2))]
performance_2_z = np.nan_to_num(performance_2_z)

performance_3_z = [IDW_interpolation(performance_3.iloc[i, :],G_station_info,G_grid_lon,G_grid_lat) for i in range(0,len(performance_3))]
performance_3_z = np.nan_to_num(performance_3_z)


#%%
"""plot kriging"""

# Plot
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import get_cmap
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

#%%

min_value= 0
max_value= np.quantile([np.max(performance),np.max(performance_1),np.max(performance_2),np.max(performance_3)],0.95)

for time in range(0,len(performance_z)):
    one_z = performance_1_z[time,:,:]
   
    filename=str(season_name[time])

    fig = plt.figure(figsize=(16, 9),dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([120.05, 121.4, 23.4, 24.25], crs=ccrs.PlateCarree())    
    shp_file = shpreader.Reader(r'zhuoshui(shp)\zhuoshui.shp')
    ax.add_geometries(shp_file.geometries(), crs=ccrs.PlateCarree(), linewidths=1.2, edgecolor='k', facecolor='none') # edgecolor=調整邊框顏色
    
    min_value = 0; max_value = 2.5
    # 定義顏色級距和顏色映射
    split_ratio = 20
    bounds = [i/split_ratio for i in range(min_value*split_ratio,int(max_value*split_ratio)+1,1)]  # 色階級距的邊界值
    bounds = np.concatenate([bounds,[i for i in range(int(max_value)+1,6)]])
    cmap = get_cmap("autumn_r")
    norm = BoundaryNorm(bounds, cmap.N)

    cf = ax.contourf(X, Y, one_z, levels=bounds, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    
    shp_clip(cf, ax, r'zhuoshui(shp)\zhuoshui.shp')
    plt.rcParams['font.sans-serif'] =['Taipei Sans TC Beta']  #匯入中文字體
    cb = plt.colorbar(cf)
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('%s'%filename, fontsize=20)
        # output_path=r'important\research\groundwater_forecast\python_code'
    output_path=r'result\1st_layer\kriging season figure\one(0614)'
    isExist = os.path.exists(output_path)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(output_path)
      print("The new directory is created!")
    plt.savefig(os.path.join(output_path+'\\%s.png'%filename), bbox_inches='tight')  # bbox_inches是修改成窄邊框
    plt.close()
    gc.collect()  # 清理站存記

    
#%%
""" 2017 winter"""
os.chdir(path)
train_obs=pd.read_excel(r"result\1st_layer\performance_comparison\train\T+1.xlsx",sheet_name="obs")

train_hbvlstm3=pd.read_excel(r"result\1st_layer\performance_comparison\train\T+1.xlsx",sheet_name="HBV-AE-LSTM")
winter_obs=pd.DataFrame(train_obs.iloc[-1,2:])
winter_hbvlstm3=pd.DataFrame(train_hbvlstm3.iloc[-1,2:])
winter_performance=[]
for j in range(0,len(winter_obs)):
    winter_performance.append(error_indicator.np_RMSE(winter_obs.iloc[j],winter_hbvlstm3.iloc[j]))
    
winter_z = IDW_interpolation(winter_performance ,G_station_info,G_grid_lon,G_grid_lat)
winter_z = np.nan_to_num(winter_z)

filename=str('2017 winter(0614)')

fig = plt.figure(figsize=(16, 9),dpi=300)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([120.05, 121.4, 23.4, 24.25], crs=ccrs.PlateCarree())    
shp_file = shpreader.Reader(r'zhuoshui(shp)\zhuoshui.shp')
ax.add_geometries(shp_file.geometries(), crs=ccrs.PlateCarree(), linewidths=1.2, edgecolor='k', facecolor='none') # edgecolor=調整邊框顏色
# cf = ax.contourf(X, Y, winter_z , levels=np.linspace(min_value,max_value,40), cmap='autumn_r', transform=ccrs.PlateCarree())
cf = ax.contourf(X, Y, winter_z, levels=bounds, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

shp_clip(cf, ax, r'zhuoshui(shp)\zhuoshui.shp')
plt.rcParams['font.sans-serif'] =['Taipei Sans TC Beta']  #匯入中文字體
cb = plt.colorbar(cf)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
plt.title('%s'%filename, fontsize=20)
    # output_path=r'important\research\groundwater_forecast\python_code'
output_path=r'result\1st_layer\kriging season figure\2017_winter'
isExist = os.path.exists(output_path)
if not isExist:
  # Create a new directory because it does not exist 
  os.makedirs(output_path)
  print("The new directory is created!")
plt.savefig(os.path.join(output_path+'\\%s.png'%filename), bbox_inches='tight')  # bbox_inches是修改成窄邊框
plt.close()
gc.collect()  # 清理站存記
    