# -*- coding: utf-8 -*-
"""
Created on Wed May 25 12:13:57 2022

@author: user
"""

import pandas as pd
import os
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import shapefile
import gc

#%%  

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
"""
Load data
"""

input_file_and_path = r'D:\研究室\濁水溪\\地下水測站對應雨量.csv'
groundwater_input_file_and_path=r'D:\研究室\濁水溪\\地下水測站.csv'

rainfall_df=pd.read_csv(input_file_and_path, encoding='big5')
groundwater_df=pd.read_csv(groundwater_input_file_and_path, encoding='big5')


time_list=list(rainfall_df.keys())
title_name=[]
for i in range(len(time_list)):
    if i>= len(time_list):
        break
    #tem_name=time_list[i].replace('1月','')
    title_name.append(time_list[i])#名稱
lons=rainfall_df['經度']
lats=rainfall_df['緯度']
station_name=rainfall_df['站名']
gr_lons=groundwater_df['X']
gr_lats=groundwater_df['Y']
gr_station_name=groundwater_df['測站']
gr_index=groundwater_df['參數']
gr_grid=pd.DataFrame(dict(long_gr=gr_lons,lat_gr=gr_lats))

#%%
list_grid_lon=[]
list_grid_lon=[]

d={}
for data_index in range(3,len(time_list)):
    
    data=rainfall_df[time_list[data_index]]
    d[time_list[data_index]]={}
    zero=[]
    data_fin=[]
    if (data == 0).astype(int).sum(axis=0)==20:
        zero_data=0
        for value in range(0,len(gr_index)):
            zero.append(zero_data)
        d[time_list[data_index]]=zero
    else:
        grid_lon=np.linspace(120, 121.5,1000)
        grid_lat=np.linspace(23, 24.5, 1000)
        list_grid_lon=grid_lon.tolist()
        list_grid_lat=grid_lat.tolist()

        OK = OrdinaryKriging(lons, lats, data, variogram_model='linear', nlags=6)
        z1, ss1 = OK.execute('grid', grid_lon, grid_lat)
        z1.shape
        a=[]
        xgrid, ygrid=np.meshgrid(grid_lon, grid_lat)
        a.extend(z1.flatten())
        df_grid=pd.DataFrame(dict(long=xgrid.flatten(), lat=ygrid.flatten() ,Krig_linear=z1.flatten()))
        for value in range(0,len(gr_index)):
            data_fin.append(df_grid.at[gr_index[value],"Krig_linear"])
        d[time_list[data_index]]=data_fin
    
#%%視覺化


        fig = plt.figure(figsize=(16, 9),dpi=300)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent([120.05, 121.4, 23.4, 24.25], crs=ccrs.PlateCarree())


        shp_file = shpreader.Reader('D:\\研究室\\濁水溪\\濁水溪.shp')
        ax.add_geometries(shp_file.geometries(), crs=ccrs.PlateCarree(), linewidths=1.2, edgecolor='k', facecolor='none') # edgecolor=調整邊框顏色
        cf = ax.contourf(xgrid, ygrid, z1, levels=np.linspace(0,6000,40), cmap='YlOrRd', transform=ccrs.PlateCarree())

        shp_clip(cf, ax,'D:\\研究室\\濁水溪\\濁水溪.shp')

        plt.rcParams['font.sans-serif'] =['Taipei Sans TC Beta']  #匯入中文字體
        cb = plt.colorbar(cf)    
        plt.scatter(lons, lats, color='navy', s=35)  # s是調整座標點大小
        for i in range(len(lons)):        
            plt.text(lons[i]+0.05,lats[i]+0.02,station_name[i], color='k', fontsize=12)  # 將測站名稱標示在座標點上
            plt.axis('off')  # 去除圖片外框
            plt.title(str(title_name[data_index]), fontsize=20)
            output_path=r'D:\研究室\濁水溪\克利金圖'
            plt.savefig(os.path.join(output_path, str(title_name[data_index])+'.png'), bbox_inches='tight')  # bbox_inches是修改成窄邊框

        plt.close()
        gc.collect()  # 清理站存記

