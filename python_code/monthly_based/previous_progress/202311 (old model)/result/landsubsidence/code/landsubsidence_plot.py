# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 17:14:44 2022

@author: steve
"""

import pandas as pd
import numpy as np

obs_train=pd.read_excel(r"D:\important\research\groundwater_forecast(monthly)\result\1st_layer\sorted_observation.xlsx",sheet_name="t+1(train)",index_col=0)
pred_train=pd.read_excel(r"D:\important\research\groundwater_forecast(monthly)\result\1st_layer\sorted_predict(hbv-ann).xlsx",sheet_name="t+1(shuffle)",index_col=0)
simulate_train=(pd.read_excel(r"D:\important\research\groundwater_forecast(monthly)\result\1st_layer\sorted_predict.xlsx",sheet_name="t+1(shuffle)",index_col=0)).iloc[1:,:]

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
# year = [2001, 2007, 2012, 2015]
year= 2015

filename = 'landsubsidence_%s'%year
landsubsidence_file = pd.read_csv(r"D:\important\research\groundwater_forecast(monthly)\landsubsidence(shp)\%s\%s.csv"%(year,filename))
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

shp_file = shpreader.Reader(r'D:\important\research\groundwater_forecast\zhuoshui(shp)\zhuoshui.shp')
ax.add_geometries(shp_file.geometries(), crs=ccrs.PlateCarree(), linewidths=1.2, edgecolor='k', facecolor='none') # edgecolor=調整邊框顏色
cf = ax.contourf(X, Y, pred_train_z, levels=np.linspace(min_value,max_value,40), cmap='Reds_r', transform=ccrs.PlateCarree())

shp_clip(cf, ax, r'D:\important\research\groundwater_forecast\zhuoshui(shp)\zhuoshui.shp')

plt.rcParams['font.sans-serif'] =['Taipei Sans TC Beta']  #匯入中文字體
cb = plt.colorbar(cf)
# plt.scatter(land_x, land_y, color='navy', s=10)  # s是調整座標點大小
# plt.scatter(boundary.loc[:,'x'], boundary.loc[:,'y'], color='navy', s=10)  # s是調整座標點大小

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
plt.axis('off')  # 去除圖片外框
plt.title(filename, fontsize=20)
    # output_path=r'important\research\groundwater_forecast\python_code'
output_path=r'D:\important\research\groundwater_forecast(monthly)\result\landsubsidence\result\%s'%filename
plt.savefig(os.path.join(output_path+'.png'), bbox_inches='tight')  # bbox_inches是修改成窄邊框
plt.close()
plt.cla()
plt.clf()
gc.collect()  # 清理站存記