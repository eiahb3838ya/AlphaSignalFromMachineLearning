import os
import pandas as pd
import numpy as np

pkl_path = r'D:\AlphaSignalFromMachineLearning\GetData\tables\windDBData'
temp_data = pd.read_csv(r'D:\AlphaSignalFromMachineLearning\GetData\tables\materialData\S_DQ_ADJOPEN.csv',index_col=0)
temp_cols = temp_data.columns
indicators = pd.read_pickle(pkl_path+r'\ASHAREL2INDICATORS')
print(indicators.head())
ind_cols = indicators.columns
initiative_buy_rate = indicators.pivot(index='TRADE_DT',columns='S_INFO_WINDCODE',values=ind_cols[2])
initiative_sell_rate = indicators.pivot(index='TRADE_DT',columns='S_INFO_WINDCODE',values=ind_cols[3])
large_buy_rate = indicators.pivot(index='TRADE_DT',columns='S_INFO_WINDCODE',values=ind_cols[4])
large_sell_rate = indicators.pivot(index='TRADE_DT',columns='S_INFO_WINDCODE',values=ind_cols[5])

initiative_buy_rate.to_pickle(r'D:\AlphaSignalFromMachineLearning\GetData\tables\materialData\S_LI_INITIATIVEBUYRATE')
initiative_sell_rate.to_pickle(r'D:\AlphaSignalFromMachineLearning\GetData\tables\materialData\S_LI_INITIATIVESELLRATE')
large_buy_rate.to_pickle(r'D:\AlphaSignalFromMachineLearning\GetData\tables\materialData\S_LI_LARGEBUYRATE')
large_sell_rate.to_pickle(r'D:\AlphaSignalFromMachineLearning\GetData\tables\materialData\S_LI_LARGESELLRATE')