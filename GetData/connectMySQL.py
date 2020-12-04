# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 16:44:47 2020

@author: cy
"""

import pyodbc
# import numpy as np
# from pandas import DataFrame,Timestamp,Series
import pandas as pd
# import csv
# from WindPy import w
import time 
#%% 数据库参数
driver='MySQL ODBC 8.0 Unicode Driver'
host = '192.168.41.56'
port = '3306'
user = 'inforesdep01'
passwd = 'tfyfInfo@1602'
db = 'wind'    
#%% 需要的資料表
require_data = {
    'AShareEODPrices':[
        'S_INFO_WINDCODE',
        'TRADE_DT',
        'S_DQ_OPEN',
        'S_DQ_HIGH',
        'S_DQ_LOW',
        'S_DQ_CLOSE',
        'S_DQ_CHANGE',
        'S_DQ_PCTCHANGE',
        'S_DQ_VOLUME',
        'S_DQ_AMOUNT',
        'S_DQ_ADJPRECLOSE',
        'S_DQ_ADJOPEN',
        'S_DQ_ADJHIGH',
        'S_DQ_ADJLOW',
        'S_DQ_ADJCLOSE',
        'S_DQ_ADJFACTOR',
        'S_DQ_AVGPRICE',
        ]
    
    }

#%% 数据提取
cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+host+';DATABASE='+db+';UID='+user+';PWD='+passwd)
columns_list = require_data['AShareEODPrices']
columns_string = ", a.".join(["{}"]*len(columns_list)).format(*columns_list)
sql=('''
      select a.{} 
      from AShareEODPrices as a
      '''.format(columns_string)
      ) 

# aTable = 'AShareEODPrices'
# aColumn = require_data[aTable][0]
# sql=('''
#       select a.S_INFO_WINDCODE, a.TRADE_DT, a.{} 
#       from {} as a
#       '''.format(aColumn, aTable)
#       ) 
#%%
start_time = time.time()
factor=pd.read_sql(sql, cnxn)
print(time.time() - start_time)

