# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 10:42:55 2022

@author: steve
"""

import pandas as pd
import numpy as np

pred_test=pd.read_excel(r"D:\important\research\groundwater_forecast\python_code\puyun\difference_level\result\1st_layer\hbvlstm(shuffle)-test(t+1).xlsx",sheet_name="test_eachstation",index_col=0)
obs_test=pd.read_excel(r"D:\important\research\groundwater_forecast\python_code\puyun\difference_level\result\1st_layer\simulate(test).xlsx",sheet_name="groundwater_observation(t+1)",index_col=0)
simulate_test=pd.read_excel(r"D:\important\research\groundwater_forecast\python_code\puyun\difference_level\result\1st_layer\simulate(test).xlsx",sheet_name="simulate_result(t+1)",index_col=0)

#%%
G0_path = r"D:\important\research\groundwater_forecast\daily_data\groundwater_l0.csv"
G0 = pd.read_csv(G0_path,header=0);G0['date'] = pd.to_datetime(G0['date']);
G0_ = G0.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
G0_station = G0.columns
G0_station = pd.Series([G0_station[i][8:10] for i in range(1,len(G0_station))])
G0_station = G0_station.drop_duplicates (keep='first').reset_index(drop=True)

G1_path = r"D:\important\research\groundwater_forecast\daily_data\groundwater_l1.csv"
G1 = pd.read_csv(G1_path,header=0);G1['date'] = pd.to_datetime(G1['date']);
G1_ = G1.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
G1_station = G1.columns
G1_station = pd.Series([G1_station[i][8:10] for i in range(1,len(G1_station))])
G1_station = G1_station.drop_duplicates (keep='first').reset_index(drop=True)

G01_station = np.concatenate([G0_station,G1_station])

G_station_info = pd.read_csv(r"D:\important\research\groundwater_forecast\station_info\groundwater_station.csv",delimiter='\t')
G01_station_info = pd.DataFrame(np.squeeze(np.array([G_station_info.iloc[G_station_info[G_station_info.iloc[:,0]==G01_station[i]].index,:] for i in range(0,len(G01_station))])))
G01_station_info.columns = ['station_name','X','Y']

time = G0_.index; test_time = time[-143:]

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
    Z = simple_idw(np.array(station_info.loc[:, 'X']),np.array(station_info.loc[:, 'Y']), data, xi, yi, power=10)
    Z = Z.reshape((xi.shape[0]),(yi.shape[0]))
    return Z

def Ordinary_Kriging(data,station_info,grid_lon,grid_lat):
    OK = OrdinaryKriging(station_info.loc[:, 'X'],station_info.loc[:, 'Y'], data, variogram_model="gaussian") 
    # Execute on grid:
    z1, ss1 = OK.execute("grid", grid_lon, grid_lat)
    return z1,ss1

# pred_test_z = [IDW_interpolation(np.squeeze(pred_test.iloc[i, :]),G01_station_info,G_grid_lon,G_grid_lat) for i in range(0,len(pred_test))]
# pred_test_z = np.nan_to_num(pred_test_z)

pred_test_z = [Ordinary_Kriging(np.squeeze(pred_test.iloc[i, :]),G01_station_info,G_grid_lon,G_grid_lat)[0] for i in range(0,len(pred_test))]
pred_test_z = np.nan_to_num(pred_test_z)

# obs_test_z = [IDW_interpolation(np.squeeze(obs_test.iloc[i, :]),G01_station_info,G_grid_lon,G_grid_lat) for i in range(0,len(obs_test))]
# obs_test_z = np.nan_to_num(obs_test_z)

obs_test_z = [Ordinary_Kriging(np.squeeze(obs_test.iloc[i, :]),G01_station_info,G_grid_lon,G_grid_lat)[0] for i in range(0,len(obs_test))]
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

# time=0
for time in range(0,len(pred_test)):
    predictions_array = pred_test_z[time,:,:]
    obs_array = obs_test_z[time,:,:]
    min_value= np.min([np.min(obs_array),np.min(predictions_array)])
    max_value= np.max([np.max(obs_array),np.max(predictions_array)])
    
    fig = plt.figure(figsize=(16, 9),dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([120.05, 121.4, 23.4, 24.25], crs=ccrs.PlateCarree())
    
    
    shp_file = shpreader.Reader(r'D:\important\research\groundwater_forecast\zhuoshui(shp)\zhuoshui.shp')
    ax.add_geometries(shp_file.geometries(), crs=ccrs.PlateCarree(), linewidths=1.2, edgecolor='k', facecolor='none') # edgecolor=調整邊框顏色
    cf = ax.contourf(X, Y, predictions_array, levels=np.linspace(min_value,max_value,40), cmap='YlOrRd', transform=ccrs.PlateCarree())
    
    shp_clip(cf, ax, r'D:\important\research\groundwater_forecast\zhuoshui(shp)\zhuoshui.shp')
    
    plt.rcParams['font.sans-serif'] =['Taipei Sans TC Beta']  #匯入中文字體
    cb = plt.colorbar(cf)
    lons = np.array(G01_station_info.loc[:,'X']); lats = G01_station_info.loc[:,'Y']; station_name = G01_station
    plt.scatter(lons, lats, color='navy', s=35)  # s是調整座標點大小
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    for i in range(len(lons)):        
        plt.text(lons[i]+0.05,lats[i]+0.02,station_name[i], color='k', fontsize=12)  # 將測站名稱標示在座標點上
        plt.axis('off')  # 去除圖片外框
    plt.title('prediction time %s'%test_time[time], fontsize=20)
        # output_path=r'important\research\groundwater_forecast\python_code'
    output_path=r'D:\important\research\groundwater_forecast\python_code\puyun\difference_level\result\1st_layer\regional_predict\%s'%time
    plt.savefig(os.path.join(output_path+'.png'), bbox_inches='tight')  # bbox_inches是修改成窄邊框
    plt.close()
    gc.collect()  # 清理站存記
    
    
    fig = plt.figure(figsize=(16, 9),dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([120.05, 121.4, 23.4, 24.25], crs=ccrs.PlateCarree())
    
    ax.add_geometries(shp_file.geometries(), crs=ccrs.PlateCarree(), linewidths=1.2, edgecolor='k', facecolor='none') # edgecolor=調整邊框顏色
    cf = ax.contourf(X, Y, obs_array, levels=np.linspace(min_value,max_value,40), cmap='YlOrRd', transform=ccrs.PlateCarree())
    shp_clip(cf, ax, r'D:\important\research\groundwater_forecast\zhuoshui(shp)\zhuoshui.shp')
    
    plt.rcParams['font.sans-serif'] =['Taipei Sans TC Beta']  #匯入中文字體
    cb = plt.colorbar(cf)
    lons = np.array(G01_station_info.loc[:,'X']); lats = G01_station_info.loc[:,'Y']; station_name = G01_station
    plt.scatter(lons, lats, color='navy', s=35)  # s是調整座標點大小
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    for i in range(len(lons)):        
        plt.text(lons[i]+0.05,lats[i]+0.02,station_name[i], color='k', fontsize=12)  # 將測站名稱標示在座標點上
        plt.axis('off')  # 去除圖片外框
    plt.title('obs time %s'%test_time[time], fontsize=20)
        # output_path=r'important\research\groundwater_forecast\python_code'
    output_path=r'D:\important\research\groundwater_forecast\python_code\puyun\difference_level\result\1st_layer\regional_obs\%s'%time
    plt.savefig(os.path.join(output_path+'.png'), bbox_inches='tight')  # bbox_inches是修改成窄邊框
    plt.close()
    gc.collect()  # 清理站存記