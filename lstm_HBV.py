# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:47:35 2022

@author: steve
"""

import math
import numpy as np
import pandas as pd

P_path = r"D:\important\research\groundwater_forecast\daily_data\rainfall(processed).csv"
P = pd.read_csv(P_path,index_col=0)
P = np.array(P)

G_path = r"D:\important\research\groundwater_forecast\daily_data\groundwater_l0(processed).csv"
G = pd.read_csv(G_path,index_col=0)
G_date = np.array(G.index)
G = np.array(G)
