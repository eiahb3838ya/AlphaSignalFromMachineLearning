# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 19:48:12 2021

@author: Lantian
"""

import os
import pandas as pd
import numpy as np
'''
中国A股日行情数据估算，包括原来相应量价数据和交易状态，涨跌停一字板限制的推测
'''
#这一段到24行都是更新原来的量价数据，之后用的话肯定有区别，因为又多了几个用这个表更新不出来的文件
filePath = r'D:\ml\AlphaSignalFromMachineLearning\GetData\tables\materialData'
a=os.listdir(filePath)
data = pd.read_pickle('./data0106/AShareDailyQuery')
keep = ['S_DQ_TRADESTATUS','S_DQ_TRADESTATUSCODE','S_DQ_LIMIT','S_DQ_STOPPING','S_DQ_HIGH','S_DQ_LOW','TRADE_DT']
names = []
for i in a:
    names.append(i.split('.')[0])
for name in names:
    temp=data[[name,'S_INFO_WINDCODE','TRADE_DT']]
    temp = temp.pivot(index='TRADE_DT',columns='S_INFO_WINDCODE',values=name)['20170101':]
    temp.to_csv('./materialdata/'+name+'.csv')
    if name not in keep:
        data.drop(columns=name,inplace=True)
#是否正常交易
temp = data[['S_DQ_TRADESTATUS','S_INFO_WINDCODE','TRADE_DT']]
temp['S_DQ_TRADE']=temp['S_DQ_TRADESTATUS'].apply(lambda x: 1 if x=='交易' else 0)
temp = temp.pivot(index='TRADE_DT',columns='S_INFO_WINDCODE',values='S_DQ_TRADE')['20170101':]
temp.to_csv('./materialdata/S_DQ_TRADE.csv')
#是否涨停买不了
temp = data[['S_DQ_LIMIT','S_INFO_WINDCODE','TRADE_DT','S_DQ_LOW']]
temp['S_DQ_BUYLIMIT']=(temp['S_DQ_LIMIT']-temp['S_DQ_LOW']).apply(lambda x: 1 if x<=0.1 else 0)
temp = temp.pivot(index='TRADE_DT',columns='S_INFO_WINDCODE',values='S_DQ_BUYLIMIT')['20170101':]
temp.to_csv('./materialdata/S_DQ_BUYLIMIT.csv')
#是否跌停卖不出
temp = data[['S_DQ_STOPPING','S_INFO_WINDCODE','TRADE_DT','S_DQ_HIGH']]
temp['S_DQ_SELLLIMIT']=(-temp['S_DQ_STOPPING']+temp['S_DQ_HIGH']).apply(lambda x: 1 if x<=0.1 else 0)
temp = temp.pivot(index='TRADE_DT',columns='S_INFO_WINDCODE',values='S_DQ_SELLLIMIT')['20170101':]
temp.to_csv('./materialdata/S_DQ_SELLLIMIT.csv')
del data,temp

'''
中国A股基本信息，从上市日期推算到上市日止的交易日
目前这个写法是因为我联不上聚宽，后续应该可以优化
'''
#计算上市起的交易日
keep = ['S_INFO_WINDCODE','OPDATE']
df = pd.read_pickle(r'd:/ml/data0106/AShareListDate')[keep]
# data['OPDATE'] = pd.to_datetime(data['OPDATE'])
opdays = {key:values for key , values in zip(df[keep[0]],df[keep[1]])} 
dates = list(set(pd.read_pickle(r'd:/ml/data0106/AShareDailyQuery')['TRADE_DT']))
dates.sort()
all_trade_days = pd.DatetimeIndex(np.load('all_trade_days.npy',allow_pickle=True))
stocks = list(df[keep[0]])
temp = pd.DataFrame({key:values for key, values in zip(stocks,[dates]*len(stocks))},index=dates)
for stock in stocks:
    temp[stock] = temp[stock].apply(lambda x: len(all_trade_days[(all_trade_days>=opdays[stock])&(all_trade_days<=x)]))

temp['20170101':].to_csv('./materialdata/TRADE_DAYS.csv') 
del temp
'''
中国A股行情衍生数据，估算每支股票当日自由流通市值，生成自由流通市值表，总市值表，总流通市值表
'''
#估算自由市值
keep=['S_INFO_WINDCODE','TRADE_DT','S_VAL_MV','S_DQ_MV','FLOAT_A_SHR_TODAY','TOT_SHR_TODAY']
data = pd.read_pickle('./data0106/derivativeIndicator')[keep]
data['S_FREE_MV']=data['FLOAT_A_SHR_TODAY']/data['TOT_SHR_TODAY']*data['S_DQ_MV']
demands = ['S_FREE_MV','S_VAL_MV','S_DQ_MV']
for demand in demands:
    temp=data[[demand,'S_INFO_WINDCODE','TRADE_DT']]
    temp = temp.pivot(index='TRADE_DT',columns='S_INFO_WINDCODE',values=demand)['20170101':]
    temp.to_csv('./materialdata/'+demand+'.csv') 
