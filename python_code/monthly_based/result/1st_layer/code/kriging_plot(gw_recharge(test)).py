# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 13:31:39 2022

@author: steve
"""
import os
path=r'D:\lab\research\groundwater_estimation\github\python_code\monthly_based' #if the code above cant work, please key in your parent directory
os.chdir(path)

#%%
import pandas as pd
import numpy as np

path2='result\\1st_layer\\performance_comparison\\test'
obs_test=pd.read_excel(path2+"\\T+1.xlsx",sheet_name="obs",index_col=0)
pred_test=pd.read_excel(path2+"\\T+1.xlsx",sheet_name="HBV-AE-LSTM",index_col=0)
simulate_test=(pd.read_excel(path2+"\\T+1.xlsx",sheet_name="HBV",index_col=0))
test_time=obs_test.iloc[:,0]
G_station_info = pd.read_excel(r"station_info\station_fullinfo.xlsx",sheet_name="G1",index_col=0) 
P_station_info = pd.read_csv(r"station_info\rainfall_station.csv")

#%%
"""calculate recharge"""
#颱風季節
start_date='2017-07-31';end_date='2017-08-31' 
# start_date='2018-07-31';end_date='2018-08-31' 
# start_date='2019-07-31';end_date='2019-08-31' 

# #非颱風季節
# start_date='2017-01-01';end_date='2017-12-31'
# start_date='2018-01-01';end_date='2018-12-31' 
# start_date='2019-01-01';end_date='2019-12-31' 

choose_date = (test_time[(test_time>=start_date) & (test_time<=end_date)]).reset_index(drop=True)
select_index= test_time[(test_time>=start_date) & (test_time<=end_date)].index

pred_event = pred_test.iloc[select_index[0],1:]-pred_test.iloc[select_index[-1],1:]
simulate_event = simulate_test.iloc[select_index[0],1:]-simulate_test.iloc[select_index[-1],1:]
obs_event = obs_test.iloc[select_index[0],1:]-obs_test.iloc[select_index[-1],1:]

#%%
import math
from pykrige.ok import OrdinaryKriging

G_grid_lon = np.linspace(math.floor(min(P_station_info.loc[:, 'X'])), math.ceil(max(P_station_info.loc[:, 'X'])), 30)
G_grid_lat = np.linspace(math.floor(min(P_station_info.loc[:, 'Y'])), math.ceil(max(P_station_info.loc[:, 'Y'])), 30)

X,Y = np.meshgrid(G_grid_lon,G_grid_lat)

def Ordinary_Kriging(data,station_info,grid_lon,grid_lat):
    OK = OrdinaryKriging(station_info.loc[:, 'X'],station_info.loc[:, 'Y'], data, variogram_model="gaussian") 
    # Execute on grid:
    z1, ss1 = OK.execute("grid", grid_lon, grid_lat)
    return z1,ss1

pred_test_z = Ordinary_Kriging(np.squeeze(pred_event),G_station_info,G_grid_lon,G_grid_lat)[0]
pred_test_z = np.nan_to_num(pred_test_z)

obs_test_z = Ordinary_Kriging(np.squeeze(obs_event),G_station_info,G_grid_lon,G_grid_lat)[0]
obs_test_z = np.nan_to_num(obs_test_z)

simulate_test_z = Ordinary_Kriging(np.squeeze(simulate_event),G_station_info,G_grid_lon,G_grid_lat)[0]
simulate_test_z = np.nan_to_num(simulate_test_z)

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

pred_sum = round(np.sum([pred_test_z[i] for i in selected_grid]),2)
obs_sum = round(np.sum([obs_test_z[i] for i in selected_grid]),2)
simulate_sum = round(np.sum([simulate_test_z[i] for i in selected_grid]),2)

#%%
import os
os.chdir(r'D:\lab\research\research_use_function')
from error_indicator import error_indicator
os.chdir(path)
pred_test_RMSE = round(error_indicator.np_RMSE((obs_test_z).flatten(),(pred_test_z).flatten()),2)
simulate_test_RMSE = round(error_indicator.np_RMSE((obs_test_z).flatten(),(simulate_test_z).flatten()),2)
pred_test_R2 = round(error_indicator.np_R2((obs_test_z).flatten(),(pred_test_z).flatten()),2)
simulate_test_R2 = round(error_indicator.np_R2((obs_test_z).flatten(),(simulate_test_z).flatten()),2)
obs_mean=round(np.mean(obs_test_z),2); obs_std=round(np.std(obs_test_z),2)

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


# min_value= np.min([np.min(obs_test_z),np.min(pred_test_z)])
# max_value= np.max([np.max(obs_test_z),np.max(pred_test_z)])
min_value = -10
max_value = 10

predictions_array = pred_test_z
simulate_array = simulate_test_z
obs_array = obs_test_z
filename=start_date

fig = plt.figure(figsize=(16, 9),dpi=300)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([120.05, 121.4, 23.4, 24.25], crs=ccrs.PlateCarree())


shp_file = shpreader.Reader(r'zhuoshui(shp)\zhuoshui.shp')
ax.add_geometries(shp_file.geometries(), crs=ccrs.PlateCarree(), linewidths=1.2, edgecolor='k', facecolor='none') # edgecolor=調整邊框顏色
cf = ax.contourf(X, Y, predictions_array, levels=np.linspace(min_value,max_value,40), cmap='seismic_r', transform=ccrs.PlateCarree())

shp_clip(cf, ax, r'zhuoshui(shp)\zhuoshui.shp')

plt.rcParams['font.sans-serif'] =['Taipei Sans TC Beta']  #匯入中文字體
cb = plt.colorbar(cf)
lons = np.array(G_station_info.loc[:,'X']); lats = G_station_info.loc[:,'Y']; station_name = G_station_info.iloc[:,0]
plt.scatter(lons, lats, color='navy', s=35)  # s是調整座標點大小
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
for i in range(len(lons)):        
    # plt.text(lons[i]+0.05,lats[i]+0.02,station_name[i], color='k', fontsize=12)  # 將測站名稱標示在座標點上
    plt.axis('off')  # 去除圖片外框
plt.title('prediction time %s (RMSE=%s , R2=%s, pred_sum=%s)'%(filename,pred_test_RMSE,pred_test_R2,pred_sum), fontsize=20)
    # output_path=r'important\research\groundwater_forecast\python_code'
output_path=r'result\1st_layer\kriging recharge event figure\test\pred'
isExist = os.path.exists(output_path)
if not isExist:
  # Create a new directory because it does not exist 
  os.makedirs(output_path)
  print("The new directory is created!")
plt.savefig(os.path.join(output_path+'\\%s.png'%filename), bbox_inches='tight')  # bbox_inches是修改成窄邊框
plt.close()
gc.collect()  # 清理站存記


fig = plt.figure(figsize=(16, 9),dpi=300)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([120.05, 121.4, 23.4, 24.25], crs=ccrs.PlateCarree())

ax.add_geometries(shp_file.geometries(), crs=ccrs.PlateCarree(), linewidths=1.2, edgecolor='k', facecolor='none') # edgecolor=調整邊框顏色
cf = ax.contourf(X, Y, simulate_array, levels=np.linspace(min_value,max_value,40), cmap='seismic_r', transform=ccrs.PlateCarree())
shp_clip(cf, ax, r'zhuoshui(shp)\zhuoshui.shp')

plt.rcParams['font.sans-serif'] =['Taipei Sans TC Beta']  #匯入中文字體
cb = plt.colorbar(cf)
lons = np.array(G_station_info.loc[:,'X']); lats = G_station_info.loc[:,'Y']; station_name = G_station_info.iloc[:,0]
plt.scatter(lons, lats, color='navy', s=35)  # s是調整座標點大小
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
for i in range(len(lons)):        
    # plt.text(lons[i]+0.05,lats[i]+0.02,station_name[i], color='k', fontsize=12)  # 將測站名稱標示在座標點上
    plt.axis('off')  # 去除圖片外框
plt.title('simulate time %s (RMSE=%s , R2=%s, simulate_sum=%s)'%(filename,simulate_test_RMSE,simulate_test_R2,simulate_sum ), fontsize=20)
    # output_path=r'important\research\groundwater_forecast\python_code'
output_path=r'result\1st_layer\kriging recharge event figure\test\simulate'
isExist = os.path.exists(output_path)
if not isExist:
  # Create a new directory because it does not exist 
  os.makedirs(output_path)
  print("The new directory is created!")
plt.savefig(os.path.join(output_path+'\\%s.png'%filename), bbox_inches='tight')  # bbox_inches是修改成窄邊框
plt.close()
gc.collect()  # 清理站存記

fig = plt.figure(figsize=(16, 9),dpi=300)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([120.05, 121.4, 23.4, 24.25], crs=ccrs.PlateCarree())

ax.add_geometries(shp_file.geometries(), crs=ccrs.PlateCarree(), linewidths=1.2, edgecolor='k', facecolor='none') # edgecolor=調整邊框顏色
cf = ax.contourf(X, Y, obs_array, levels=np.linspace(min_value,max_value,40), cmap='seismic_r', transform=ccrs.PlateCarree())
shp_clip(cf, ax, r'zhuoshui(shp)\zhuoshui.shp')

plt.rcParams['font.sans-serif'] =['Taipei Sans TC Beta']  #匯入中文字體
cb = plt.colorbar(cf)
lons = np.array(G_station_info.loc[:,'X']); lats = G_station_info.loc[:,'Y']; station_name = G_station_info.iloc[:,0]
plt.scatter(lons, lats, color='navy', s=35)  # s是調整座標點大小
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
for i in range(len(lons)):        
    # plt.text(lons[i]+0.05,lats[i]+0.02,station_name[i], color='k', fontsize=12)  # 將測站名稱標示在座標點上
    plt.axis('off')  # 去除圖片外框
plt.title('obs time %s (obs_sum=%s)'%(filename,obs_sum), fontsize=20)
    # output_path=r'important\research\groundwater_forecast\python_code'
output_path=r'result\1st_layer\kriging recharge event figure\test\obs'
isExist = os.path.exists(output_path)
if not isExist:
  # Create a new directory because it does not exist 
  os.makedirs(output_path)
  print("The new directory is created!")
plt.savefig(os.path.join(output_path+'\\%s.png'%filename), bbox_inches='tight')  # bbox_inches是修改成窄邊框
plt.close()
gc.collect()  # 清理站存記