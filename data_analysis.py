# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 20:04:36 2022

@author: steve
"""

import pandas as pd
import numpy as np
import os
os.chdir(r"D:\research\research_use_function")
from multidimensional_reshape import multi_input_output

groundwater_l0=pd.read_excel(r"D:\research\groundwater_estimation\daily_data\underground_dataset(10days-based).xlsx",engine="openpyxl",sheet_name="groundwater_l0",index_col=0)
rainfall=pd.read_excel(r"D:\research\groundwater_estimation\daily_data\underground_dataset(10days-based).xlsx",engine="openpyxl",sheet_name="rainfall",index_col=0)
riverflow=pd.read_excel(r"D:\research\groundwater_estimation\daily_data\underground_dataset(10days-based).xlsx",engine="openpyxl",sheet_name="riverflow",index_col=0)
#%%

multi_io=multi_input_output(groundwater_l0,input_timestep=3,output_timestep=3)
l0_input=multi_io.generate_input();
l0_output=multi_io.generate_output()

multi_io=multi_input_output(rainfall,input_timestep=3,output_timestep=3)
rainfall_input=multi_io.generate_input()

multi_io=multi_input_output(riverflow,input_timestep=3,output_timestep=3)
riverflow_input=multi_io.generate_input()

#%%
l0_model_input=np.concatenate([l0_input,rainfall_input,riverflow_input],axis=2)
l0_model_output=l0_output

#%%
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.multioutput import MultiOutputRegressor
import tensorflow as tf