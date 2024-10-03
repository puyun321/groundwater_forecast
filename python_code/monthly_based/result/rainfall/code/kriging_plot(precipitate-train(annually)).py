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


"""Read station info"""
station_info = pd.read_excel(r"station_info\station_fullinfo.xlsx",sheet_name="G1",index_col=0)

time = G0.index;
time = pd.DataFrame(time)

#%%

P_station_info = pd.read_csv(r"station_info\rainfall_station.csv")

#%%
"""choose date"""
forecast_timestep=1
obs_train=pd.read_excel(r"result\1st_layer\sorted_observation.xlsx",sheet_name="t+%s(train)"%forecast_timestep,index_col=0)

date=pd.DataFrame(G0.index)

train_date = pd.DataFrame(date[:len(obs_train)])

# start_date='2000-01-01';end_date='2007-12-31'
start_date='2007-01-01';end_date='2007-12-31' 


choose_date = (train_date[(train_date['date']>=start_date) & (train_date['date']<=end_date)].iloc[:,0]).reset_index(drop=True)
select_index= train_date[(train_date['date']>=start_date) & (train_date['date']<=end_date)].index

"""choose rainfall event"""
obs_event = np.mean(P[select_index,:],axis=0)

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


all_coordinate=[];min_index=[];all_info=[];
for i in range(0,len(station_info)):
    all_info.append(get_specific_coordinate(station_info.iloc[:,1],station_info.iloc[:,2],X,Y))
    
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

# selected_P=[]
# for i in range(0,len(select_index)):
#     obs_train_z = IDW_interpolation(obs_event[i,:],P_station_info,G_grid_lon,G_grid_lat)
#     obs_train_z = np.nan_to_num(obs_train_z)
#     selected_P.append(obs_train_z[min_index[:,0],min_index[:,1]])

# selected_P=np.array(selected_P)

obs_event_z = IDW_interpolation(obs_event,P_station_info,G_grid_lon,G_grid_lat)
obs_event_z = np.nan_to_num(obs_event_z)

#%%
"""plot kriging"""

# Plot
import matplotlib
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
   
    
min_value = 0
max_value = 10

#%%
cdict = {'red': [(0.000, 0.76078, 0.76078),
                 (0.0033, 0.76078, 0.61177),
                 (0.0067, 0.61177, 0.01176),
                 (0.0167, 0.01176, 0.01961),
                 (0.0333, 0.01961, 0.01176),
                 (0.05, 0.01176, 0.01961),
                 (0.0667, 0.01961, 0.22352),
                 (0.1, 0.22352, 1),
                 (0.1333, 1, 1),
                 (0.1667, 1, 1),
                 (0.2333, 1, 1),
                 (0.3, 1, 0.8),
                 (0.3667, 0.8, 0.6),
                 (0.4333, 0.6, 0.58823),
                 (0.5, 0.58823, 0.79216),
                 (0.6667, 0.79216, 0.98431),
                 (1, 0.98431, 0.99216)],
         
         'green':[(0.000, 0.76078, 0.76078),
                  (0.0033, 0.76078, 0.98824),
                  (0.0067, 0.98824, 0.78431),
                  (0.0167, 0.78431, 0.60784),
                  (0.0333, 0.60784, 0.38824),
                  (0.05, 0.38824, 0.6),
                  (0.0667, 0.6, 1),
                  (0.1, 1, 0.98431),
                  (0.1333, 0.98431, 0.78431),
                  (0.1667, 0.78431, 0.58431),
                  (0.2333, 0.58431, 0),
                  (0.3, 0, 0),
                  (0.3667, 0, 0),
                  (0.4333, 0, 0),
                  (0.5, 0, 0),
                  (0.6667, 0, 0),
                  (1, 0, 0.78824)],
         
         'blue':[(0.000, 0.76078, 0.76078),
                  (0.0033, 0.76078, 1),
                  (0.0067, 1, 1),
                  (0.0167, 1, 1),
                  (0.0333, 1, 1),
                  (0.05, 1, 0.00784),
                  (0.0667, 0.00784, 0.01176),
                  (0.1, 0.01176, 0.01176),
                  (0.1333, 0.01176, 0),
                  (0.1667, 0, 0),
                  (0.2333, 0, 0),
                  (0.3, 0, 0),
                  (0.3667, 0, 0),
                  (0.4333, 0, 0.6),
                  (0.5, 0.6, 0.80392),
                  (0.6667, 0.80392, 1),
                  (1, 1, 1)]}

def add_mid_column(cdict_element):
    cdict_=np.concatenate(cdict_element,axis=1)
    cdict_1 =[]
    for i in range(0,len(cdict_[1,:])):
        if i==0:
            cdict_1.append(cdict_[1,i])
        else:
            cdict_1.append(cdict_[1,i-1])
    cdict_1=np.asarray(cdict_1)
    cdict_=np.asarray([cdict_[0,:],cdict_1,cdict_[1,:]])
    cdict_=np.transpose(cdict_)
    return cdict_

def cdict_gradient(cdict):
    cdict_red=[[]*1 for i in range(0,len(cdict['red'])-1)]
    cdict_green=[[]*1 for i in range(0,len(cdict['green'])-1)]
    cdict_blue=[[]*1 for i in range(0,len(cdict['blue'])-1)]
    for i in cdict:
        if i=='red':
            
            for j in range(0,len(cdict[i])-1): 
                interpolate_down=np.asarray(cdict[i][j]);interpolate_up=np.asarray(cdict[i][j+1]);
                zero=np.linspace(interpolate_down[0],interpolate_up[0],24)
                two=np.linspace(interpolate_down[2],interpolate_up[2],24)
                if j<len(cdict[i])-2:
                    cdict_red[j].append([zero[i] for i in range(0,len(zero)-1)])
                    cdict_red[j].append([two[i] for i in range(0,len(two)-1)])
                else:
                    cdict_red[j].append([zero[i] for i in range(0,len(zero))])
                    cdict_red[j].append([two[i] for i in range(0,len(two))])                    
                
        elif i=='green':
            
            for j in range(0,len(cdict[i])-1): 
                interpolate_down=np.asarray(cdict[i][j]);interpolate_up=np.asarray(cdict[i][j+1]);
                zero=np.linspace(interpolate_down[0],interpolate_up[0],24)
                two=np.linspace(interpolate_down[2],interpolate_up[2],24)
                if j<len(cdict[i])-2:
                    cdict_green[j].append([zero[i] for i in range(0,len(zero)-1)])      
                    cdict_green[j].append([two[i] for i in range(0,len(two)-1)])
                else:
                    cdict_green[j].append([zero[i] for i in range(0,len(zero))])
                    cdict_green[j].append([two[i] for i in range(0,len(two))])         
                
        else:
            
            for j in range(0,len(cdict[i])-1): 
                interpolate_down=np.asarray(cdict[i][j]);interpolate_up=np.asarray(cdict[i][j+1]);
                zero=np.linspace(interpolate_down[0],interpolate_up[0],24)
                two=np.linspace(interpolate_down[2],interpolate_up[2],24)                
                if j<len(cdict[i])-2:
                    cdict_blue[j].append([zero[i] for i in range(0,len(zero)-1)])            
                    cdict_blue[j].append([two[i] for i in range(0,len(two)-1)])       
                else:
                    cdict_blue[j].append([zero[i] for i in range(0,len(zero))])
                    cdict_blue[j].append([two[i] for i in range(0,len(two))])                         
    
    cdict_red_=add_mid_column(cdict_red)
    cdict_green_=add_mid_column(cdict_green)
    cdict_blue_=add_mid_column(cdict_blue)
    cdict_gradient={'red':cdict_red_,
                    'green':cdict_green_,
                    'blue':cdict_blue_
                    }
    return cdict_gradient

cdict_gradient=cdict_gradient(cdict)

cmp_rain= matplotlib.colors.LinearSegmentedColormap('name',cdict_gradient)
levels=[0,10,20,60,100,150,200,300,400,500,600,700,800,900,1000,1200,1500]
# levels=[0,1,2,5,10,15,20,30,40,50,70,90,110,130,150,200,300]


level_gradient=[]
for i in range(1,len(levels)):
    gradient=np.linspace(levels[i-1],levels[i],24)
    if i<len(levels)-1:
        level_gradient.append([gradient[i] for i in range(0,len(gradient)-1)])
    else:
        level_gradient.append([gradient[i] for i in range(0,len(gradient))])
level_gradient=np.concatenate(level_gradient)

#%%

obs_array = obs_event_z
total_rain = np.sum(obs_event_z)
filename=start_date

shp_file = shpreader.Reader(r'zhuoshui(shp)\zhuoshui.shp')
fig = plt.figure(figsize=(16, 9),dpi=300)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([120.05, 121.4, 23.4, 24.25], crs=ccrs.PlateCarree())

ax.add_geometries(shp_file.geometries(), crs=ccrs.PlateCarree(), linewidths=1.2, edgecolor='k', facecolor='none') # edgecolor=調整邊框顏色
cf = ax.contourf(X, Y, obs_array, levels=level_gradient, cmap= cmp_rain, transform=ccrs.PlateCarree())
shp_clip(cf, ax, r'zhuoshui(shp)\zhuoshui.shp')

plt.rcParams['font.sans-serif'] =['Taipei Sans TC Beta']  #匯入中文字體
cb = plt.colorbar(cf)
lons = np.array(station_info.loc[:,'X']); lats = station_info.loc[:,'Y']; station_name = station_info.iloc[:,0]
# plt.scatter(lons, lats, color='navy', s=15)  # s是調整座標點大小
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
# for i in range(len(lons)):        
#     plt.text(lons[i]+0.05,lats[i]+0.02,station_name[i], color='k', fontsize=12)  # 將測站名稱標示在座標點上
#     plt.axis('off')  # 去除圖片外框
plt.axis('off')  # 去除圖片外框
plt.title('obs time %s (%s)'%(filename,total_rain), fontsize=20)
    # output_path=r'important\research\groundwater_forecast\python_code'
output_path=r'result\rainfall\train'
isExist = os.path.exists(output_path)
if not isExist:
  # Create a new directory because it does not exist 
  os.makedirs(output_path)
  print("The new directory is created!")
plt.savefig(os.path.join(output_path+'\\%s.png'%(filename)), bbox_inches='tight')  # bbox_inches是修改成窄邊框
plt.close()
plt.cla()
plt.clf()
gc.collect()  # 清理站存記
