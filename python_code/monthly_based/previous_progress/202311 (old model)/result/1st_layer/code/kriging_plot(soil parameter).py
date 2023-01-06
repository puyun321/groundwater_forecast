# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 22:20:23 2022

@author: Steve
"""

import os
path=r'D:\lab\research\groundwater_estimation\github\python_code\monthly_based'
os.chdir(path)
import pandas as pd
import numpy as np

#%%
G_station_info = pd.read_csv(r"station_info\groundwater_station.csv",delimiter='\t')

G0 = pd.read_csv(r"daily_data\groundwater_l0.csv",delimiter=',');G0['date'] = pd.to_datetime(G0['date']);
G0_ = G0.resample('1M',on='date',base=0,loffset='1M').mean()
# G0_ = G0.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
G0_station = G0.columns
G0_station = pd.Series([G0_station[i][8:10] for i in range(1,len(G0_station))])
G0_station = G0_station.drop_duplicates (keep='first').reset_index(drop=True)

G1 = pd.read_csv(r"daily_data\groundwater_l1.csv",delimiter=',');G1['date'] = pd.to_datetime(G1['date']);
G1_ = G1.resample('1M',on='date',base=0,loffset='1M').mean()
G1_station = G1.columns
G1_station = pd.Series([G1_station[i][8:10] for i in range(1,len(G1_station))])
G1_station = G1_station.drop_duplicates (keep='first').reset_index(drop=True)

G01_station = np.concatenate([G0_station,G1_station])

G_station_info = pd.read_csv(r"station_info\groundwater_station.csv",delimiter='\t')
G01_station_info = pd.DataFrame(np.squeeze(np.array([G_station_info.iloc[G_station_info[G_station_info.iloc[:,0]==G01_station[i]].index[0],:] for i in range(0,len(G01_station))])))
G01_station_info.columns = ['station_name','X','Y']

#%%

P_station_info = pd.read_csv(r"station_info\rainfall_station.csv")

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


#%%
import math
from pykrige.ok import OrdinaryKriging
"""plot kriging"""
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

G_grid_lon = np.linspace(math.floor(min(P_station_info.loc[:, 'X'])), math.ceil(max(P_station_info.loc[:, 'X'])), 30)
G_grid_lat = np.linspace(math.floor(min(P_station_info.loc[:, 'Y'])), math.ceil(max(P_station_info.loc[:, 'Y'])), 30)
X,Y = np.meshgrid(G_grid_lon,G_grid_lat)
cmaps=['Oranges','Wistia','Purples','Greys']
index=0 #index=0,1,2 (T+1-T+3)
optima = pd.read_csv(r"result\1st_layer\optima_parameter(t+%s).csv"%(index+1),index_col=0)
for j in range(0,optima.shape[1]):
    predictions_array = IDW_interpolation(optima.iloc[:,j],G01_station_info,G_grid_lon,G_grid_lat)
    min_value= np.min(predictions_array)
    max_value= np.max(predictions_array)

    fig = plt.figure(figsize=(16, 9),dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([120.05, 121.4, 23.4, 24.25], crs=ccrs.PlateCarree())


    shp_file = shpreader.Reader(r'zhuoshui(shp)\zhuoshui.shp')
    ax.add_geometries(shp_file.geometries(), crs=ccrs.PlateCarree(), linewidths=1.2, edgecolor='k', facecolor='none') # edgecolor=調整邊框顏色
    cf = ax.contourf(X, Y, predictions_array, levels=np.linspace(min_value,max_value,40), cmap=cmaps[j], transform=ccrs.PlateCarree())

    shp_clip(cf, ax, r'zhuoshui(shp)\zhuoshui.shp')

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
    plt.title('parameter %s'%(optima.columns[j]), fontsize=20)
        # output_path=r'important\research\groundwater_forecast\python_code'
    output_path=r'result\1st_layer\kriging parameter\%s'%(optima.columns[j])
    plt.savefig(os.path.join(output_path+'.png'), bbox_inches='tight')  # bbox_inches是修改成窄邊框
    plt.close()
    plt.cla()
    plt.clf()
    gc.collect()  # 清理站存記