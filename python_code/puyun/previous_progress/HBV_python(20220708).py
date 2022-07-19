import math
import numpy as np
import pdb
from scipy.linalg import kron
import pandas as pd

def hargreaves(tmin, tmax, tmean, lat, day_of_year):
    # Loop to reduce memory usage
    pet = np.zeros(tmin.shape, dtype=np.float32) * np.NaN
    for ii in np.arange(len(pet[:, 0])):
        trange = tmax[ii, :] - tmin[ii, :]
        trange[trange < 0] = 0

        latitude = np.deg2rad(lat[ii, :])

        SOLAR_CONSTANT = 0.0820

        sol_dec = 0.409 * np.sin(((2.0 * np.pi / 365.0) * day_of_year[ii, :] - 1.39))

        sha = np.arccos(np.clip(-np.tan(latitude) * np.tan(sol_dec), -1, 1))

        ird = 1 + (0.033 * np.cos((2.0 * np.pi / 365.0) * day_of_year[ii, :]))

        tmp1 = (24.0 * 60.0) / np.pi
        tmp2 = sha * np.sin(latitude) * np.sin(sol_dec)
        tmp3 = np.cos(latitude) * np.cos(sol_dec) * np.sin(sha)
        et_rad = tmp1 * SOLAR_CONSTANT * ird * (tmp2 + tmp3)

        pet[ii, :] = 0.0023 * (tmean[ii, :] + 17.8) * trange ** 0.5 * 0.408 * et_rad

    pet[pet < 0] = 0

    return pet

#%%

G_path = r"D:\important\research\groundwater_forecast\daily_data\groundwater_l0.csv"
# G = pd.read_csv(G_path,index_col=0)
G = pd.read_csv(G_path);G['date'] = pd.to_datetime(G['date']);
# G = np.array(G);
G = G.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
G_date = np.array(G.index)
G_station = G.columns
G_station = pd.Series([G_station[i][8:10] for i in range(0,len(G_station))])
G_station = G_station.drop_duplicates (keep='first').reset_index(drop=True)

#merge stations with same name
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
G = np.array([G.iloc[:,station_index[i]].mean(axis=1) for i in range(0,len(station_index))]).T

P_path = r"D:\important\research\groundwater_forecast\daily_data\rainfall.csv"
# P = pd.read_csv(P_path,index_col=0)
P = pd.read_csv(P_path);P['date'] = pd.to_datetime(P['date']);
P = P.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
P_station = P.columns
P = np.array(P)

T_path = r"D:\important\research\groundwater_forecast\daily_data\temperature.csv"
# T = pd.read_csv(T_path,index_col=0)
T = pd.read_csv(T_path);T['date'] = pd.to_datetime(T['date']);
T = T.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
T_station = T.columns
T = np.array(T)

ETpot_path = r"D:\important\research\groundwater_forecast\daily_data\evaporation_rate.csv"
# ETpot = pd.read_csv(ETpot_path,index_col=0)
ETpot = pd.read_csv(ETpot_path);ETpot['date'] = pd.to_datetime(ETpot['date']);
ETpot = ETpot.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
ETpot_station = ETpot.columns
ETpot = np.array(ETpot)

#%%
import os 
os.chdir(r"D:\important\research\research_use_function")
from rescale import normalization
G_norm_module = normalization(G)
G_norm = G_norm_module.norm()

P_norm_module = normalization(P)
P_norm = P_norm_module.norm()

T_norm_module = normalization(T)
T_norm = T_norm_module.norm()

ETpot_norm_module = normalization(ETpot)
ETpot_norm = ETpot_norm_module .norm()

#%%
from multidimensional_reshape import multi_input_output
G_multi_module=multi_input_output(G,input_timestep=3,output_timestep=3)
G_input=G_multi_module.generate_input()
G_multi_output=G_multi_module.generate_output()
forecast_timestep=3
G_output=G_multi_output[:,forecast_timestep-1,:]
G_input_date=G_date[forecast_timestep-1:-forecast_timestep-1]

#%%
# Kriging
G_station_info = pd.read_csv(r"D:\important\research\groundwater_forecast\station_info\groundwater_station.csv")
G0_station_info = pd.DataFrame(np.squeeze(np.array([G_station_info.iloc[G_station_info[G_station_info.iloc[:,0]==G_station[i]].index,:] for i in range(0,len(G_station))])))
P_station_info = pd.read_csv(r"D:\important\research\groundwater_forecast\station_info\rainfall_station.csv")
T_station_info = pd.DataFrame(np.squeeze(np.array([P_station_info.iloc[P_station_info[P_station_info.iloc[:,0]==T_station[i]].index,:] for i in range(0,len(T_station))])))
T_station_info.columns = G_station_info.columns
ETpot_station_info = pd.DataFrame(np.squeeze(np.array([P_station_info.iloc[P_station_info[P_station_info.iloc[:,0]==ETpot_station[i]].index,:] for i in range(0,len(ETpot_station))])))
ETpot_station_info.columns = G_station_info.columns            

#%%
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from scipy.interpolate import griddata

def Ordinary_Kriging(data,station_info,grid_lon,grid_lat):
    OK = OrdinaryKriging(station_info.loc[:, 'X'],station_info.loc[:, 'Y'], data, variogram_model="gaussian") 
    # Execute on grid:
    z1, ss1 = OK.execute("grid", grid_lon, grid_lat)
    return z1,ss1

def Universal_Kriging(data,station_info,grid_lon,grid_lat):
    OK = UniversalKriging(station_info.loc[:, 'X'],station_info.loc[:, 'Y'], data, variogram_model="exponential",drift_terms=["regional_linear"],) 
    # Execute on grid:
    z1, ss1 = OK.execute("grid", grid_lon, grid_lat)
    return z1,ss1

def interpolate2grid(data,station_info,grid_lon,grid_lat,interpolate_method='linear'):
    X,Y = np.meshgrid(grid_lon,grid_lat)
    Z = np.squeeze(griddata([(x,y) for x,y in zip(station_info.loc[:, 'X'],station_info.loc[:, 'Y'])], data, (X, Y), method=interpolate_method))
    return Z


P_grid_lon = np.linspace(math.floor(min(P_station_info.loc[:, 'X'])), math.ceil(max(P_station_info.loc[:, 'X'])), 30)
P_grid_lat = np.linspace(math.floor(min(P_station_info.loc[:, 'Y'])), math.ceil(max(P_station_info.loc[:, 'Y'])), 30)

""" Since the Kriging fail to provide a good interpolation, we applied the IDW interpolation method to grid our data"""
# z1, ss1 = Ordinary_Kriging(pd.DataFrame(P[23, :]),P_station_info,P_grid_lon,P_grid_lat) 
# z1, ss1 = Universal_Kriging(pd.DataFrame(P[17, :]),P_station_info,P_grid_lon,P_grid_lat)

P_z = [interpolate2grid(pd.DataFrame(P[i, :]),P_station_info,P_grid_lon,P_grid_lat) for i in range(0,len(P))]
P_z = np.nan_to_num(P_z)
T_z = np.array([interpolate2grid(pd.DataFrame(T[i, :]),T_station_info,P_grid_lon,P_grid_lat,interpolate_method='nearest') for i in range(0,len(T))])
ETpot_z = np.array([interpolate2grid(pd.DataFrame(ETpot[i, :]),ETpot_station_info,P_grid_lon,P_grid_lat,interpolate_method='nearest') for i in range(0,len(ETpot))])
X,Y = np.meshgrid(P_grid_lon,P_grid_lat)

#%%
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

all_info=[];
for i in range(0,len(G0_station_info)):
    all_info.append(get_specific_coordinate(G0_station_info.iloc[i,1],G0_station_info.iloc[i,2],X,Y))

all_coordinate=[];min_index=[];
all_coordinate=np.squeeze(np.array([all_info[i][0] for i in range(0,len(all_info))]))
min_index=np.squeeze(np.array([all_info[i][1] for i in range(0,len(all_info))]))

#%%
#select specific grid
P_input=P_z[forecast_timestep-1:-forecast_timestep-1,min_index[:,0],min_index[:,1]]
T_input=T_z[forecast_timestep-1:-forecast_timestep-1,min_index[:,0],min_index[:,1]]
ETpot_input=ETpot_z[forecast_timestep-1:-forecast_timestep-1,min_index[:,0],min_index[:,1]]

#%%
from scipy.optimize import differential_evolution

bounds = [(0, 1), (0, 1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1)]
optima=[[]*1 for i in range(0,G_output.shape[1])]
for station in range(0,G_output.shape[1]):
    P,T,ETpot,G=P_input[:,station],T_input[:,station],ETpot_input[:,station],G_output[:,station]
    
    def HBV_model(parameter):
        parBETA = parameter[0]
        parFC = parameter[1]
        parK0 = parameter[2]
        parK1 = parameter[3]
        parK2 = parameter[4]
        parLP = parameter[5]
        parPERC = parameter[6]
        parUZL = parameter[7]
        parPCORR = parameter[8]
        parTT = parameter[9]
        parCFMAX = parameter[10]
        parSFCF = parameter[11]
        parCFR = parameter[12]
        parCWH = parameter[13]
        # parCET = parameter[14]
        # parMAXBAS = parameter[15]
        
        # Initialize time series of model variables
        SNOWPACK = np.zeros(parBETA.shape, dtype=np.float32) + 0.001
        MELTWATER = np.zeros(parBETA.shape, dtype=np.float32) + 0.001
        SM = np.zeros(parBETA.shape, dtype=np.float32) + 0.001
        SUZ = np.zeros(parBETA.shape, dtype=np.float32) + 0.001
        SLZ = np.zeros(parBETA.shape, dtype=np.float32) + 0.001
        ETact = np.zeros(parBETA.shape, dtype=np.float32) + 0.001
        Qsim = np.zeros(P.shape, dtype=np.float32) * np.NaN
        # Qsim[0, :] = 0.001
        
        for t in range(0, len(P)):
            # Separate precipitation into liquid and solid components
            PRECIP = P[t] * parPCORR
            RAIN = np.multiply(PRECIP, T[t] >= parTT)
            SNOW = np.multiply(PRECIP, T[t] < parTT)
            SNOW = SNOW * parSFCF
        
            # Snow
            SNOWPACK = SNOWPACK + SNOW
            melt = parCFMAX * (T[t] - parTT)
            melt = melt.clip(0.0, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK = SNOWPACK - melt
            refreezing = parCFR * parCFMAX * (parTT - T[t])
            refreezing = refreezing.clip(0.0, MELTWATER)
            SNOWPACK = SNOWPACK + refreezing
            MELTWATER = MELTWATER - refreezing
            tosoil = MELTWATER - (parCWH * SNOWPACK)
            tosoil = tosoil.clip(0.0, None)
            MELTWATER = MELTWATER - tosoil
            
            tosoil=0        
            # Soil and evaporation
            soil_wetness = (SM / parFC) ** parBETA
            soil_wetness = np.nan_to_num(soil_wetness.clip(0.0, 1.0))
            recharge = (RAIN + tosoil) * soil_wetness
            SM = SM + RAIN + tosoil - recharge
            excess = SM - parFC
            excess = excess.clip(0.0, None)
            SM = SM - excess
            evapfactor = SM / (parLP * parFC)
            evapfactor = evapfactor.clip(0.0, 1.0)
            ETact = ETpot[t] * evapfactor
            ETact = np.minimum(SM, ETact)
            SM = SM - ETact
        
            # Groundwater boxes
            SUZ = SUZ + recharge + excess
            PERC = np.minimum(SUZ, parPERC)
            SUZ = SUZ - PERC
            Q0 = parK0 * np.maximum(SUZ - parUZL, 0.0)
            SUZ = SUZ - Q0
            Q1 = parK1 * SUZ
            SUZ = SUZ - Q1
            SLZ = SLZ + PERC
            Q2 = parK2 * SLZ
            SLZ = SLZ - Q2
            Qsim[t] = Q0 + Q1 + Q2
    
        loss=np.mean(np.absolute(G-Q2))
        return loss
    
    # average_loss=HBV_model(parameter,P_input,T_input,ETpot_input,G_output)
    res=differential_evolution(HBV_model,bounds,tol=1e-8, disp=True)
    print(res.fun)
    print(res.success)
    print(res.x)
    optima[station].append(res.x)

#%%
optima = np.squeeze(np.array(optima))
optima = pd.DataFrame(optima)

optima.columns = ['parBETA','parFC','parK0','parK1','parK2','parLP','parPERC','parUZL','parPCORR','parTT','parCFMAX','parSFCF','parCFR','parCWH']
optima.to_csv(r"D:\important\research\groundwater_forecast\python_code\puyun\result\optima_parameter.csv")