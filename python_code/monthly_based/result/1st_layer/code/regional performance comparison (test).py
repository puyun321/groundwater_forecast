# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:23:00 2023

@author: Steve
"""

import os
import pandas as pd
import numpy as np

path=r'D:\lab\research\groundwater_estimation\github\python_code\monthly_based' #if the code above cant work, please key in your parent directory
os.chdir(path)
#%%

performance_1=pd.read_excel(r"result\1st_layer\performance_comparison\test\T+1.xlsx",sheet_name="all_performance",index_col=0)
regional_1_mean=performance_1.groupby(by='區域').mean()
performance_2=pd.read_excel(r"result\1st_layer\performance_comparison\test\T+2.xlsx",sheet_name="all_performance",index_col=0)
regional_2_mean=performance_2.groupby(by='區域').mean()
performance_3=pd.read_excel(r"result\1st_layer\performance_comparison\test\T+3.xlsx",sheet_name="all_performance",index_col=0)
regional_3_mean=performance_3.groupby(by='區域').mean()

#%%

writer = pd.ExcelWriter(r"result\1st_layer\performance_comparison\test\performance(regional).xlsx",engine='xlsxwriter')
regional_1_mean.to_excel(writer,sheet_name="T+1")
regional_2_mean.to_excel(writer,sheet_name="T+2")
regional_3_mean.to_excel(writer,sheet_name="T+3")

writer.save()