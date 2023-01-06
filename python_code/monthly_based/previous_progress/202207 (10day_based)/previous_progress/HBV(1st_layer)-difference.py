# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 13:58:25 2022

@author: steve
"""

import pandas as pd
import numpy as np

#%%
P_path = r"D:\important\research\groundwater_forecast\daily_data\rainfall.csv"
# P = pd.read_csv(P_path,index_col=0)
P = pd.read_csv(P_path);P['date'] = pd.to_datetime(P['date']);
P = P.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
P_station = P.columns
P = np.array(P)
P_station_info = pd.read_csv(r"D:\important\research\groundwater_forecast\station_info\rainfall_station.csv")

T_path = r"D:\important\research\groundwater_forecast\daily_data\temperature.csv"
# T = pd.read_csv(T_path,index_col=0)
T = pd.read_csv(T_path);T['date'] = pd.to_datetime(T['date']);
T = T.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
T_station = T.columns
T = np.array(T)
T_station_info = pd.DataFrame(np.squeeze(np.array([P_station_info.iloc[P_station_info[P_station_info.iloc[:,0]==T_station[i]].index,:] for i in range(0,len(T_station))])))
T_station_info.columns = P_station_info.columns

ETpot_path = r"D:\important\research\groundwater_forecast\daily_data\evaporation_rate.csv"
# ETpot = pd.read_csv(ETpot_path,index_col=0)
ETpot = pd.read_csv(ETpot_path);ETpot['date'] = pd.to_datetime(ETpot['date']);
ETpot = ETpot.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
ETpot_station = ETpot.columns
ETpot = np.array(ETpot)
ETpot_station_info = pd.DataFrame(np.squeeze(np.array([P_station_info.iloc[P_station_info[P_station_info.iloc[:,0]==ETpot_station[i]].index,:] for i in range(0,len(ETpot_station))])))
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
    specific_station_info = pd.DataFrame(np.squeeze(np.array([all_station_info.iloc[all_station_info[all_station_info.iloc[:,0]==specific_station[i]].index,:] for i in range(0,len(specific_station))])))
    specific_station_info.columns = all_station_info.columns
    return specific_station_info

#%%
well_info= pd.read_csv(r"D:\important\research\groundwater_forecast\station_info\well_info.txt",encoding='utf-16',delimiter='\t')
G_station_info = pd.read_csv(r"D:\important\research\groundwater_forecast\station_info\groundwater_station.csv")

G0 = pd.read_csv(r"D:\important\research\groundwater_forecast\daily_data\groundwater_l0.csv");G0['date'] = pd.to_datetime(G0['date']);
G0 = G0.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
G0_station = G0.columns
G0_station = pd.Series([G0_station[i][8:] for i in range(0,len(G0_station))])
G0_station_name = remain_onlyname(G0_station)
G0_station_info = get_station_info(G_station_info,G0_station_name)

G0_new = merge_samename(G0_station_name,G0)
G0_station_info_new = G0_station_info.drop_duplicates(keep="first").reset_index(drop=True)

G1 = pd.read_csv(r"D:\important\research\groundwater_forecast\daily_data\groundwater_l1.csv");G1['date'] = pd.to_datetime(G1['date']);
G1 = G1.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
G1_station = G1.columns
G1_station = pd.Series([G1_station[i][8:] for i in range(0,len(G1_station))])
G1_station_name = remain_onlyname(G1_station)
G1_station_info = get_station_info(G_station_info,G1_station_name)

G1_new = merge_samename(G1_station_name,G1)
G1_station_info_new = G1_station_info.drop_duplicates(keep="first").reset_index(drop=True)

G2 = pd.read_csv(r"D:\important\research\groundwater_forecast\daily_data\groundwater_l2.csv");G2['date'] = pd.to_datetime(G2['date']);
G2 = G2.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
G2_station = G2.columns
G2_station = pd.Series([G2_station[i][8:] for i in range(0,len(G2_station))])
G2_station_name = remain_onlyname(G2_station)
G2_station_info = get_station_info(G_station_info,G2_station_name)

G2_new = merge_samename(G2_station_name,G2)
G2_station_info_new = G2_station_info.drop_duplicates(keep="first").reset_index(drop=True)

G3 = pd.read_csv(r"D:\important\research\groundwater_forecast\daily_data\groundwater_l3.csv");G3['date'] = pd.to_datetime(G3['date']);
G3 = G3.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
G3_station = G3.columns
G3_station = pd.Series([G3_station[i][8:] for i in range(0,len(G3_station))])
G3_station_name = remain_onlyname(G3_station)
G3_station_info = get_station_info(G_station_info,G3_station_name)

G3_new = merge_samename(G3_station_name,G3)
G3_station_info_new = G3_station_info.drop_duplicates(keep="first").reset_index(drop=True)


G4 = pd.read_csv(r"D:\important\research\groundwater_forecast\daily_data\groundwater_l4.csv");G4['date'] = pd.to_datetime(G4['date']);
G4 = G4.resample('10D',on='date',base=0,loffset='9D').mean() #convert daily data to 10day based
G4_station = G4.columns
G4_station = pd.Series([G4_station[i][8:] for i in range(0,len(G4_station))])
G4_station_name = remain_onlyname(G4_station)
G4_station_info = get_station_info(G_station_info,G4_station_name)

G4_new = merge_samename(G4_station_name,G4)
G4_station_info_new = G4_station_info.drop_duplicates(keep="first").reset_index(drop=True)

G01_new = np.concatenate([G0_new,G1_new],axis=1)

#%%
import math

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
    Z = simple_idw(np.array(station_info.loc[:, 'X']),np.array(station_info.loc[:, 'Y']), data, xi, yi, power=15)
    Z = Z.reshape((xi.shape[0]),(yi.shape[0]))
    return Z

""" Since the Kriging fail to provide a good interpolation, we applied the IDW interpolation method to grid our data"""

G_grid_lon = np.linspace(math.floor(min(P_station_info.loc[:, 'X'])), math.ceil(max(P_station_info.loc[:, 'X'])), 100)
G_grid_lat = np.linspace(math.floor(min(P_station_info.loc[:, 'Y'])), math.ceil(max(P_station_info.loc[:, 'Y'])), 100)
X,Y = np.meshgrid(G_grid_lon,G_grid_lat)
P_z = [IDW_interpolation(np.squeeze(P[i, :]),P_station_info,G_grid_lon,G_grid_lat) for i in range(0,len(P))]
P_z = np.nan_to_num(P_z)
T_z = np.array([IDW_interpolation(np.squeeze(T[i, :]),T_station_info,G_grid_lon,G_grid_lat) for i in range(0,len(T))])
ETpot_z = np.array([IDW_interpolation(np.squeeze(ETpot[i, :]),ETpot_station_info,G_grid_lon,G_grid_lat) for i in range(0,len(ETpot))])
    
#%%

G_name = pd.Series(np.concatenate([G0_station_name, G1_station_name, G2_station_name, G3_station_name, G4_station_name]))
G_unique = G_name.drop_duplicates(keep="first").reset_index(drop=True)
G_unique_station_info = pd.DataFrame(np.squeeze(np.array([G_station_info.iloc[G_station_info[G_station_info.iloc[:,0]==G_unique[i]].index,:] for i in range(0,len(G_unique))])))
G_unique_station_info.columns = G_station_info.columns 

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
for i in range(0,len(G0_station_info_new)+len(G1_station_info_new)):
# for i in range(0,len(G_unique_station_info)):

    all_info.append(get_specific_coordinate(G_unique_station_info.iloc[i,1],G_unique_station_info.iloc[i,2],X,Y))

all_coordinate=[];min_index=[];
all_coordinate=np.squeeze(np.array([all_info[i][0] for i in range(0,len(all_info))]))
min_index=np.squeeze(np.array([all_info[i][1] for i in range(0,len(all_info))]))

#%%
# 過去9個小時的雨量、蒸發散加總會影響未來三個月的地下水儲水量
timestep=9
P_input=P_z[:,min_index[:,0],min_index[:,1]]
T_input=T_z[:,min_index[:,0],min_index[:,1]]
ETpot_input=ETpot_z[:,min_index[:,0],min_index[:,1]]

P_input = P_input[:-timestep]; P_input = np.array([np.sum(P_input[i-timestep:i,:],axis=0) for i in range(timestep,len(P_input)+1)])
T_input = T_input[:-timestep]; T_input = np.array([np.sum(T_input[i-timestep:i],axis=0) for i in range(timestep,len(T_input)+1)])
ETpot_input = ETpot_input[:-timestep]; ETpot_input = np.array([np.sum(ETpot_input[i-timestep:i],axis=0) for i in range(timestep,len(ETpot_input)+1)])
G01_new_ = G01_new[timestep-1:-timestep]; G01_base = G01_new[timestep-2:-timestep-1]
forecast_timestep=np.array([timestep/3,timestep*2/3,timestep]).astype(int)
G01_out1 = G01_new[timestep-2+forecast_timestep[0]:-(timestep-forecast_timestep[0])]; G01_out1=G01_out1[1:,:]-G01_out1[0:-1,:]
G01_out2 = G01_new[timestep-2+forecast_timestep[1]:-(timestep-forecast_timestep[1])]; G01_out2=G01_out2[1:,:]-G01_out2[0:-1,:]
G01_out3 = G01_new[timestep-2+forecast_timestep[2]:]; G01_out3 =G01_out3[1:,:]-G01_out3[0:-1,:]

P_train_obs = P_input[:int(len(P_input)*0.8),:]
T_train_obs = T_input[:int(len(T_input)*0.8),:]
ETpot_train_obs = ETpot_input[:int(len(ETpot_input)*0.8),:]
G_train_obs = G01_new_[:int(len(G01_new_)*0.8),:]

G_train_out1 = G01_out1[:int(len(G01_out1)*0.8),:]
G_train_out2 = G01_out2[:int(len(G01_out2)*0.8),:]
G_train_out3 = G01_out3[:int(len(G01_out3)*0.8),:]
G_train_out = [G_train_out1, G_train_out2, G_train_out3]

P_test_obs = P_input[int(len(P_input)*0.8):,:]
T_test_obs = T_input[int(len(T_input)*0.8):,:]
ETpot_test_obs = ETpot_input[int(len(ETpot_input)*0.8):,:]
G_test_obs = G01_new_[int(len(G01_new_)*0.8):,:]

G_test_out1 = G01_out1[int(len(G01_out1)*0.8):,:]
G_test_out2 = G01_out2[int(len(G01_out2)*0.8):,:]
G_test_out3 = G01_out3[int(len(G01_out3)*0.8):,:]
G_test_out = [G_test_out1, G_test_out2, G_test_out3]

#%%
from scipy.optimize import differential_evolution

bounds = [(1,6), (50, 1000), (0.2,1), (0,0.2)]
optima_alltimestep=[]
for j in range(0,3):
    optima=[[]*1 for i in range(0,len(G_train_obs[0]))]
    for station in range(0,len(G_train_obs[0])):
        # G_in=G_train_obs[:,station]    
        P_in,T_in,ETpot_in=P_input[:,station],T_input[:,station],ETpot_input[:,station]
        G_out=G_train_out[j][:,station]
        
        def HBV_model(parameter,location=1):      
            parBETA = parameter[0]
            parFC = parameter[1]
            parLP = parameter[2]
            parPCORR = parameter[3]
            
            if location==1:
                parTT,parCFMAX,parSFCF,parCFR,parCWH = 0,0,0,0,0
           
            else:
                parTT = parameter[4]
                parCFMAX = parameter[5]
                parSFCF = parameter[6]
                parCFR = parameter[7]
                parCWH = parameter[8]
        
            # Initialize time series of model variables
            SNOWPACK = np.zeros(parBETA.shape, dtype=np.float32) + 0.001
            MELTWATER = np.zeros(parBETA.shape, dtype=np.float32) + 0.001
            # SM_ = G_in
            ETact = np.zeros(parBETA.shape, dtype=np.float32) + 0.001
            
            loss=[];
            for t in range(0, len(G_out)):
                SM = 0.001
                # Separate precipitation into liquid and solid components
                PRECIP = P_in[t] * parPCORR
                RAIN = np.multiply(PRECIP, T_in[t] >= parTT)
                SNOW = np.multiply(PRECIP, T_in[t] < parTT)
                SNOW = SNOW * parSFCF
            
                # Snow
                SNOWPACK = SNOWPACK + SNOW
                melt = parCFMAX * (T_in[t] - parTT)
                melt = melt.clip(0.0, SNOWPACK)
                MELTWATER = MELTWATER + melt
                SNOWPACK = SNOWPACK - melt
                refreezing = parCFR * parCFMAX * (parTT - T_in[t])
                refreezing = refreezing.clip(0.0, MELTWATER)
                SNOWPACK = SNOWPACK + refreezing
                MELTWATER = MELTWATER - refreezing
                tosoil = MELTWATER - (parCWH * SNOWPACK)
                tosoil = tosoil.clip(0.0, None)
                MELTWATER = MELTWATER - tosoil
                
                if location==1:
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
                ETact = ETpot_in[t] * evapfactor
                ETact = np.minimum(SM, ETact)
                SM = SM - ETact
                loss0 = np.absolute(G_out[t]- SM)                       
                loss.append(loss0)
            loss=np.mean(loss)

            return loss

        res=differential_evolution(HBV_model,bounds,tol=1e-8, disp=True)
        print(res.fun)
        print(res.success)
        print(res.x)
        optima[station].append(res.x)
    optima_alltimestep.append(optima)

#%%
for index,optima in enumerate(optima_alltimestep):
    
    optima = np.squeeze(np.array(optima))
    optima = pd.DataFrame(optima)
    
    optima.columns = ['parBETA','parFC','parLP','parPCORR']
    optima.to_csv(r"D:\important\research\groundwater_forecast\python_code\puyun\result\1st_layer\optima_parameter(t+%s)-difference.csv"%(index+1))

#%%
#verify the trained model
all_simulate_train_result=[]
all_simulate_test_result=[]

for index in range(0,3):
    optima = pd.read_csv(r"D:\important\research\groundwater_forecast\python_code\puyun\result\1st_layer\optima_parameter(t+%s)-difference.csv"%(index+1),index_col=0)
    location=1
    
    def HBV_trained_model(P_input,T_input,ETpot_input):
        parTT,parCFMAX,parSFCF,parCFR,parCWH = 0,0,0,0,0
        simulate_result=[[]*1 for i in range(0,len(optima))]
        for station in range(0,len(optima)):            
            P_in,T_in,ETpot_in=P_input[:,station],T_input[:,station],ETpot_input[:,station]
            train_len=len(P_in)
            # Initialize time series of model variables
            SNOWPACK = np.zeros(1, dtype=np.float32) + 0.001
            MELTWATER = np.zeros(1, dtype=np.float32) + 0.001
            ETact = np.zeros(1, dtype=np.float32) + 0.001
            
            for t in range(0, train_len):      
                
                SM = 0.001
                parBETA = optima.iloc[station,0]
                parFC = optima.iloc[station,1]
                parLP = optima.iloc[station,2]
                parPCORR = optima.iloc[station,3]
                # Separate precipitation into liquid and solid components
                PRECIP = P_in[t] * parPCORR
                RAIN = np.multiply(PRECIP, T_in[t] >= parTT)
                SNOW = np.multiply(PRECIP, T_in[t] < parTT)
                SNOW = SNOW * parSFCF
            
                # Snow
                SNOWPACK = SNOWPACK + SNOW
                melt = parCFMAX * (T_in[t] - parTT)
                melt = melt.clip(0.0, SNOWPACK)
                MELTWATER = MELTWATER + melt
                SNOWPACK = SNOWPACK - melt
                refreezing = parCFR * parCFMAX * (parTT - T_in[t])
                refreezing = refreezing.clip(0.0, MELTWATER)
                SNOWPACK = SNOWPACK + refreezing
                MELTWATER = MELTWATER - refreezing
                tosoil = MELTWATER - (parCWH * SNOWPACK)
                tosoil = tosoil.clip(0.0, None)
                MELTWATER = MELTWATER - tosoil
                
                if location==1:
                    tosoil=0        
                # Soil and evaporation           
                soil_wetness = (SM / parFC) ** parBETA
                soil_wetness = np.nan_to_num(soil_wetness.clip(0.0, 1.0))
                recharge = (RAIN + tosoil) * soil_wetness
                SM = SM + RAIN + tosoil - recharge
                excess = SM - parFC
                excess = excess.clip(0.0, None)
                # print(excess)
                SM = SM - excess               
                evapfactor = SM / (parLP * parFC)
                evapfactor = evapfactor.clip(0.0, 1.0)
                ETact = ETpot_in[t] * evapfactor
                ETact = np.minimum(SM, ETact)
                SM = SM - ETact
               
    
                simulate_result[station].append(SM)        
        simulate_result=np.transpose(np.array(simulate_result))
        return simulate_result
        
    simulate_train_result=HBV_trained_model(P_train_obs,T_train_obs,ETpot_train_obs)
    simulate_test_result=HBV_trained_model(P_test_obs,T_test_obs,ETpot_test_obs)
    all_simulate_train_result.append(simulate_train_result)
    all_simulate_test_result.append(simulate_test_result)


#%%
import os
os.chdir(r"D:\important\research\research_use_function")
from error_indicator import error_indicator
# save result

writer = pd.ExcelWriter(r"D:\important\research\groundwater_forecast\python_code\puyun\result\1st_layer\simulate(train)-difference.xlsx",engine='xlsxwriter')
for i,simulate_train_result in enumerate(all_simulate_train_result):
    pd.DataFrame(simulate_train_result).to_excel(writer,sheet_name="simulate_result(t+%s)"%(i+1))
    pd.DataFrame(G_train_out[i]).to_excel(writer,sheet_name="groundwater_observation(t+%s)"%(i+1))

train_R2=[[]*1 for i in range(0,3)];train_rmse=[[]*1 for i in range(0,3)]
for i in range(0,3):
    for j in range(0,G_train_out[i].shape[1]):
        pred=pd.DataFrame(all_simulate_train_result[i][:,j])
        obs=pd.DataFrame(G_train_out[i][:,j])
        train_R2[i].append(error_indicator.np_R2(obs,pred))
        train_rmse[i].append(error_indicator.np_RMSE(obs,pred))
    
train_R2=pd.DataFrame(np.transpose(np.array(train_R2)))
train_R2.columns=['T+1','T+2','T+3']
train_rmse=pd.DataFrame(np.transpose(np.squeeze(np.array(train_rmse))))
train_rmse.columns=['T+1','T+2','T+3']
train=pd.concat([train_R2,train_rmse],axis=1)
train.to_excel(writer,sheet_name="training_performance")
writer.save()


writer = pd.ExcelWriter(r"D:\important\research\groundwater_forecast\python_code\puyun\result\1st_layer\simulate(test)-difference.xlsx",engine='xlsxwriter')
for i,simulate_train_result in enumerate(all_simulate_train_result):
    pd.DataFrame(simulate_train_result).to_excel(writer,sheet_name="simulate_result(t+%s)"%(i+1))
    pd.DataFrame(G_test_out[i]).to_excel(writer,sheet_name="groundwater_observation(t+%s)"%(i+1))

test_R2=[[]*1 for i in range(0,3)];test_rmse=[[]*1 for i in range(0,3)]
for i in range(0,3):
    for j in range(0,G_test_out[i].shape[1]):
        pred=pd.DataFrame(all_simulate_test_result[i][:,j])
        obs=pd.DataFrame(G_test_out[i][:,j])
        test_R2[i].append(error_indicator.np_R2(obs,pred))
        test_rmse[i].append(error_indicator.np_RMSE(obs,pred))
    
test_R2=pd.DataFrame(np.transpose(np.array(test_R2)))
test_R2.columns=['T+1','T+2','T+3']
test_rmse=pd.DataFrame(np.transpose(np.squeeze(np.array(test_rmse))))
test_rmse.columns=['T+1','T+2','T+3']
test=pd.concat([test_R2,test_rmse],axis=1)
test.to_excel(writer,sheet_name="testing_performance")
writer.save()

#%%
