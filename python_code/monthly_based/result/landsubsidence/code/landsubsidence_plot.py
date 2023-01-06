# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 17:14:44 2022

@author: steve
"""

import os
path=r'D:\lab\research\groundwater_estimation\github\python_code\monthly_based' #if the code above cant work, please key in your parent directory
os.chdir(path)

import pandas as pd
import numpy as np

# path2='result\\1st_layer\\performance_comparison\\train'
path2='result\\1st_layer\\performance_comparison\\test'
obs_test=pd.read_excel(path2+"\\T+1.xlsx",sheet_name="obs",index_col=0)
pred_test=pd.read_excel(path2+"\\T+1.xlsx",sheet_name="HBV-AE-LSTM",index_col=0)
simulate_test=(pd.read_excel(path2+"\\T+1.xlsx",sheet_name="HBV",index_col=0))
test_time=obs_test.iloc[:,0]
G_station_info = pd.read_excel(r"station_info\station_fullinfo.xlsx",sheet_name="G1",index_col=0) 
P_station_info = pd.read_csv(r"station_info\rainfall_station.csv")

#%%

import math

G_grid_lon = np.linspace(math.floor(min(P_station_info.loc[:, 'X'])), math.ceil(max(P_station_info.loc[:, 'X'])), 30)
G_grid_lat = np.linspace(math.floor(min(P_station_info.loc[:, 'Y'])), math.ceil(max(P_station_info.loc[:, 'Y'])), 30)

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

#%%

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

#%%
# year = [2001, 2007, 2012, 2015, 2017, 2018, 2019]
year= 2019

filename = 'landsubsidence_%s'%year
landsubsidence_file = pd.read_csv(r"landsubsidence(shp)\%s\%s.csv"%(year,filename), encoding='gbk')
landsubsidence_file.columns=['ID','land_subsidence','distance','','x','y']
land_x=landsubsidence_file.loc[:,'x']; land_y=landsubsidence_file.loc[:,'y']

select_index=[]
for i in range(0,len(landsubsidence_file)):
    if land_x.iloc[i]>=min(G_station_info.loc[:, 'X']) and land_x.iloc[i]<=max(G_station_info.loc[:, 'X']):
        
        if land_y.iloc[i]>=min(G_station_info.loc[:, 'Y']) and land_y.iloc[i]<=max(G_station_info.loc[:, 'Y']):
            select_index.append(i)
        
select_index=np.array(select_index)
landsubsidence_file=landsubsidence_file.iloc[select_index]
land_x=land_x.iloc[select_index]; land_y=land_y.iloc[select_index]

G_grid_lon = np.linspace(math.floor(min(P_station_info.loc[:, 'X'])), math.ceil(max(P_station_info.loc[:, 'X'])), 30)
G_grid_lat = np.linspace(math.floor(min(P_station_info.loc[:, 'Y'])), math.ceil(max(P_station_info.loc[:, 'Y'])), 30)
X,Y = np.meshgrid(G_grid_lon,G_grid_lat)

#注意xy順序
boundary = []
for j in range(1,7):
    for i in range(int(np.min(land_x)*100000/(1+j/100000)),int(np.max(land_x)*100000*(1+j/100000)),5000):
        boundary.append([0,0,0,0,0,i/100000,np.min(land_y)/(1+j/1000)])
    for i in range(int(np.min(land_x)*100000/(1+j/100000)),int(np.max(land_x)*100000*(1+j/100000)),5000):
        boundary.append([0,0,0,0,0,i/100000,np.max(land_y)*(1+j/1000)])
    for i in range(int(np.min(land_y)*100000/(1+j/100000)),int(np.max(land_y)*100000*(1+j/100000)),5000):
        boundary.append([0,0,0,0,0,np.min(land_x)/(1+j*0.5/1000),i/100000])
    for i in range(int(np.min(land_y)*100000/(1+j/100000)),int(np.max(land_y)*100000*(1+j/100000)),5000):
        boundary.append([0,0,0,0,0,np.max(land_x)*(1+j*0.5/1000),i/100000])

    
boundary=pd.DataFrame(np.array(boundary))
boundary=boundary.iloc[:,1:]
boundary.columns = landsubsidence_file.columns

landsubsidence_file = pd.concat([landsubsidence_file,boundary],axis=0)
landsubsidence_file = landsubsidence_file.reset_index(drop=True)
landsubsidence = landsubsidence_file.iloc[:,1] 
# min_value=np.min(landsubsidence); max_value=np.max(landsubsidence)
min_value=-10; max_value = 0

land_x=landsubsidence_file.loc[:,'x']; land_y=landsubsidence_file.loc[:,'y']

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
    Z = simple_idw(np.array(station_info.loc[:, 'x']),np.array(station_info.loc[:, 'y']), data, xi, yi, power=30)
    Z = Z.reshape((xi.shape[0]),(yi.shape[0]))
    return Z

pred_train_z = IDW_interpolation(landsubsidence_file.iloc[:,1],landsubsidence_file,G_grid_lon,G_grid_lat)
pred_train_z = np.nan_to_num(pred_train_z)

#%%

fig = plt.figure(figsize=(16, 9),dpi=300)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([120.05, 121.4, 23.4, 24.25], crs=ccrs.PlateCarree())

shp_file = shpreader.Reader(r'zhuoshui(shp)\zhuoshui.shp')
ax.add_geometries(shp_file.geometries(), crs=ccrs.PlateCarree(), linewidths=1.2, edgecolor='k', facecolor='none') # edgecolor=調整邊框顏色
cf = ax.contourf(X, Y, pred_train_z, levels=np.linspace(min_value,max_value,40), cmap='Reds_r', transform=ccrs.PlateCarree())

shp_clip(cf, ax, r'zhuoshui(shp)\zhuoshui.shp')

plt.rcParams['font.sans-serif'] =['Taipei Sans TC Beta']  #匯入中文字體
cb = plt.colorbar(cf)
# plt.scatter(land_x, land_y, color='navy', s=10)  # s是調整座標點大小
# plt.scatter(boundary.loc[:,'x'], boundary.loc[:,'y'], color='navy', s=10)  # s是調整座標點大小

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
plt.axis('off')  # 去除圖片外框
plt.title(filename, fontsize=20)
    # output_path=r'important\research\groundwater_forecast\python_code'
output_path=r'result\landsubsidence\result\%s'%filename
plt.savefig(os.path.join(output_path+'.png'), bbox_inches='tight')  # bbox_inches是修改成窄邊框
plt.close()
plt.cla()
plt.clf()
gc.collect()  # 清理站存記