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
import math
from pykrige.ok import OrdinaryKriging

# G_grid_lon = np.linspace(math.floor(min(G_station_info.loc[:, 'X'])), math.ceil(max(G_station_info.loc[:, 'X'])), 30)
# G_grid_lat = np.linspace(math.floor(min(G_station_info.loc[:, 'Y'])), math.ceil(max(G_station_info.loc[:, 'Y'])), 30)
G_grid_lon = np.linspace(math.floor(min(P_station_info.loc[:, 'X'])), math.ceil(max(P_station_info.loc[:, 'X'])), 30)
G_grid_lat = np.linspace(math.floor(min(P_station_info.loc[:, 'Y'])), math.ceil(max(P_station_info.loc[:, 'Y'])), 30)

X,Y = np.meshgrid(G_grid_lon,G_grid_lat)

def Ordinary_Kriging(data,station_info,grid_lon,grid_lat):
    OK = OrdinaryKriging(station_info.loc[:, 'X'],station_info.loc[:, 'Y'], data, variogram_model="gaussian") 
    # Execute on grid:
    z1, ss1 = OK.execute("grid", grid_lon, grid_lat)
    return z1,ss1

pred_test_z = [Ordinary_Kriging(np.squeeze(pred_test.iloc[i, 1:]),G_station_info,G_grid_lon,G_grid_lat)[0] for i in range(0,len(pred_test))]
pred_test_z = np.nan_to_num(pred_test_z)

obs_test_z = [Ordinary_Kriging(np.squeeze(obs_test.iloc[i, 1:]),G_station_info,G_grid_lon,G_grid_lat)[0] for i in range(0,len(obs_test))]
obs_test_z = np.nan_to_num(obs_test_z)

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


min_value= np.min([np.min(obs_test_z),np.min(pred_test_z)])
max_value= np.max([np.max(obs_test_z),np.max(pred_test_z)])
# time=0
for time in range(0,len(pred_test)):
    predictions_array = pred_test_z[time,:,:]
    obs_array = obs_test_z[time,:,:]
    filename=str(test_time[time])[0:10]
    # min_value= np.min([np.min(obs_array),np.min(predictions_array)])
    # max_value= np.max([np.max(obs_array),np.max(predictions_array)])
    
    fig = plt.figure(figsize=(16, 9),dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([120.05, 121.4, 23.4, 24.25], crs=ccrs.PlateCarree())
    
    
    shp_file = shpreader.Reader(r'zhuoshui(shp)\zhuoshui.shp')
    ax.add_geometries(shp_file.geometries(), crs=ccrs.PlateCarree(), linewidths=1.2, edgecolor='k', facecolor='none') # edgecolor=調整邊框顏色
    cf = ax.contourf(X, Y, predictions_array, levels=np.linspace(min_value,max_value,40), cmap='YlOrRd', transform=ccrs.PlateCarree())
    
    shp_clip(cf, ax, r'zhuoshui(shp)\zhuoshui.shp')
    
    plt.rcParams['font.sans-serif'] =['Taipei Sans TC Beta']  #匯入中文字體
    cb = plt.colorbar(cf)
    lons = np.array(G_station_info.loc[:,'X']); lats = G_station_info.loc[:,'Y']; station_name = G_station_info.iloc[:,0]
    plt.scatter(lons, lats, color='navy', s=35)  # s是調整座標點大小
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    for i in range(len(lons)):        
        plt.text(lons[i]+0.05,lats[i]+0.02,station_name[i], color='k', fontsize=12)  # 將測站名稱標示在座標點上
        plt.axis('off')  # 去除圖片外框
    plt.title('prediction time %s'%filename, fontsize=20)
        # output_path=r'important\research\groundwater_forecast\python_code'
    output_path=r'result\1st_layer\kriging event figure\test\pred'
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
    cf = ax.contourf(X, Y, obs_array, levels=np.linspace(min_value,max_value,40), cmap='YlOrRd', transform=ccrs.PlateCarree())
    shp_clip(cf, ax, r'zhuoshui(shp)\zhuoshui.shp')
    
    plt.rcParams['font.sans-serif'] =['Taipei Sans TC Beta']  #匯入中文字體
    cb = plt.colorbar(cf)
    lons = np.array(G_station_info.loc[:,'X']); lats = G_station_info.loc[:,'Y']; station_name = G_station_info.iloc[:,0]
    plt.scatter(lons, lats, color='navy', s=35)  # s是調整座標點大小
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    for i in range(len(lons)):        
        plt.text(lons[i]+0.05,lats[i]+0.02,station_name[i], color='k', fontsize=12)  # 將測站名稱標示在座標點上
        plt.axis('off')  # 去除圖片外框
    plt.title('obs time %s'%filename, fontsize=20)
        # output_path=r'important\research\groundwater_forecast\python_code'
    output_path=r'result\1st_layer\kriging event figure\test\obs'
    isExist = os.path.exists(output_path)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(output_path)
      print("The new directory is created!")
    plt.savefig(os.path.join(output_path+'\\%s.png'%filename), bbox_inches='tight')  # bbox_inches是修改成窄邊框
    plt.close()
    gc.collect()  # 清理站存記