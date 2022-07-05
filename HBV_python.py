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
# def HBV(P, ETpot, T, parameters):
    
np.random.seed(111)

P_path=[]
while len(P_path)==0:
    P_path = input("!!!The first column of the data should be index!!! \n Please key in your precipitation data path(csv file):\n ")

P = pd.read_csv(P_path,index_col=0)
# P_date = np.array(P.index)
P = np.array(P)
# P = np.random.uniform(low=10.0, high=30, size=(20, 10))

G_path=[]
while len(G_path)==0:
    G_path = input("!!!The first column of the data should be index!!! \n Please key in your groundwater level data path(csv file):\n ")
    # P_data_structure = int(input(" 1. row: sample number; column: site number (Factor) \n 2. row: site number (Factor); column: sample number\n "
    #                          +"Please key in your data structure type (key in num 1 or 2):"))

G = pd.read_csv(G_path,index_col=0)
G_date = np.array(G.index)
G= np.array(G)

location=[]
while type(location)!=int:
    location = int(input(" 1. Area without melting snow (i.e. Taiwan, SEA countries and etc.) \n 2. Area with melting snow (i.e. U.S., Europe and etc.)  \n Please select your study area (key in number 1 or 2): "))
if location==1:
    const_T = int(input("Set your temperature: "))
    T = np.array([25 for i in range(0, P.shape[0]*P.shape[1])]) # set temperature to constant 25 degree celsius
    T = np.reshape(T,(P.shape[0],P.shape[1]))
else:
    T = np.random.uniform(low=10.0, high=20, size=(P.shape[0], P.shape[1]))

watershed_size=[]
while type(watershed_size)!=int:
    watershed_size = int(input(" 1. small \n 2. large  \n How is your watershed size? (key in number 1 or 2): "))
if watershed_size==1:
    const_ETpot = int(input("Set your evaporation rate: "))
    ETpot = np.array([const_ETpot for i in range(0, P.shape[0]*P.shape[1])]) # set temperature to constant 25 degree celsius
    ETpot = np.reshape(ETpot,(P.shape[0],P.shape[1]))
else:
    ETpot = np.random.uniform(low=10.0, high=20, size=(P.shape[0], P.shape[1]))


parameters = np.random.uniform(low=0.0, high=1.0, size=(16, P.shape[1]))


# HBV(P, ETpot, T, parameters)
#
# Runs the HBV-light hydrological model (Seibert, 2005). NaN values have to be
# removed from the inputs.
#
# Input:
#     P = array with daily values of precipitation (mm/d)
#     ETpot = array with daily values of potential evapotranspiration (mm/d)
#     T = array with daily values of air temperature (deg C)
#     parameters = array with parameter values having the following structure:
#         [BETA; CET; FC; K0; K1; K2; LP; MAXBAS; PERC; UZL; PCORR; TT; CFMAX; SFCF; CFR; CWH]
#
# Output, all in mm:
#     Qsim = daily values of simulated streamflow
#     SM = soil storage
#     SUZ = upper zone storage
#     SLZ = lower zone storage
#     SNOWPACK = snow depth
#     ETact = actual evaporation
#
#     Python implementation of HBV by Hylke Beck
#     (hylke.beck@gmail.com).
#
#     Last modified 4 November 2014

parBETA = parameters[0]
parCET = parameters[1]
parFC = parameters[2]
parK0 = parameters[3]
parK1 = parameters[4]
parK2 = parameters[5]
parLP = parameters[6]
parMAXBAS = parameters[7]
parPERC = parameters[8]
parUZL = parameters[9]
parPCORR = parameters[10]
parTT = parameters[11]
parCFMAX = parameters[12]
parSFCF = parameters[13]
parCFR = parameters[14]
parCWH = parameters[15]

# Apply correction factor to precipitation
P = np.tile(parPCORR, (len(P[:, 0]), 1)) * P

# Add initialization period to the model input time series
# if len(P) < int(10*365.25):
#    repeats = np.ceil((10*365.25) / len(P))
#    init_period = int(repeats*len(P))
#    P = kron(np.ones((repeats+1, 1)), P)
#    ETpot = kron(np.ones((repeats+1, 1)), ETpot)
#    T = kron(np.ones((repeats+1, 1)), T)
# else:
#    init_period = int(10*365.25)
#    P = np.concatenate([P[:init_period], P])
#    ETpot = np.concatenate([ETpot[:init_period], ETpot])
#    T = np.concatenate([T[:init_period], T])

# Initialize time series of model variables
SNOWPACK = np.zeros(parBETA.shape, dtype=np.float32) + 0.001
MELTWATER = np.zeros(parBETA.shape, dtype=np.float32) + 0.001
SM = np.zeros(parBETA.shape, dtype=np.float32) + 0.001
SUZ = np.zeros(parBETA.shape, dtype=np.float32) + 0.001
SLZ = np.zeros(parBETA.shape, dtype=np.float32) + 0.001
ETact = np.zeros(parBETA.shape, dtype=np.float32) + 0.001
Qsim = np.zeros(P.shape, dtype=np.float32) * np.NaN
Qsim[0, :] = 0.001

#%%

for t in range(1, len(P[:, 0])):
    # Separate precipitation into liquid and solid components
    PRECIP = P[t, :] * parPCORR
    RAIN = np.multiply(PRECIP, T[t, :] >= parTT)
    SNOW = np.multiply(PRECIP, T[t, :] < parTT)
    SNOW = SNOW * parSFCF

    # Snow
    SNOWPACK = SNOWPACK + SNOW
    melt = parCFMAX * (T[t, :] - parTT)
    melt = melt.clip(0.0, SNOWPACK)
    MELTWATER = MELTWATER + melt
    SNOWPACK = SNOWPACK - melt
    refreezing = parCFR * parCFMAX * (parTT - T[t, :])
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
    soil_wetness = soil_wetness.clip(0.0, 1.0)
    recharge = (RAIN + tosoil) * soil_wetness
    SM = SM + RAIN + tosoil - recharge
    excess = SM - parFC
    excess = excess.clip(0.0, None)
    SM = SM - excess
    evapfactor = SM / (parLP * parFC)
    evapfactor = evapfactor.clip(0.0, 1.0)
    ETact = ETpot[t, :] * evapfactor
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
    Qsim[t, :] = Q0 + Q1 + Q2

    # print str(t)+" "+str(Qsim[t,0])

## Check water balance closure
# storage_diff = SM[-1]-SM[0]+SUZ[-1]-SUZ[0]+SLZ[-1]-SLZ[0]+SNOWPACK[-1]-SNOWPACK[0]+MELTWATER[-1]-MELTWATER[0]
# error = np.mean(P*365.25) - np.mean(Qsim*365.25) - np.mean(ETact*365.25) - (365.25*storage_diff/len(P))
# if error > 1:
#    print "Warning: Water balance error of "+str(round(error*1000) / 1000)+" mm/yr"

# Add routing delay to simulated runoff, uses MAXBAS parameter
''' 
parMAXBAS = np.round(parMAXBAS*100) / 100
window = parMAXBAS*100
w = np.empty(int(window))
for x in range(0, int(window)):
    w[x] = window/2 - abs(window/2-x-0.5)
w = np.concatenate([w, [0.0]*200])
w_small = [0.0]*int(np.ceil(parMAXBAS)) 
#w_small = np.arange(2, 10, dtype=np.float)
for x in range(0, int(np.ceil(parMAXBAS))):
    w_small[x] = np.sum(w[x*100:x*100+100])
w_small = w_small/np.sum(w_small)
Qsim_smoothed = np.array([0.0]*len(Qsim))
for ii in range(len(w_small)):
    Qsim_smoothed[ii:len(Qsim_smoothed)] = Qsim_smoothed[ii:len(Qsim_smoothed)] \
        + Qsim[0:len(Qsim_smoothed)-ii]*w_small[ii]
'''

# Remove initialization period
'''
Qsim = Qsim_smoothed[init_period:]
SM = SM[init_period:]
SUZ = SUZ[init_period:]
SLZ = SLZ[init_period:]
SNOWPACK = SNOWPACK[init_period:]
ETact = ETact[init_period:]
'''

# return Qsim  # Return Qsim_smoothed in future!!!!!!!!