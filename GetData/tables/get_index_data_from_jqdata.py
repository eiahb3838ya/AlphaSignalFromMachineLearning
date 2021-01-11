# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 15:30:10 2021

@author: Ye Donggua
"""
import os
import pandas as pd
from GetData.tables.config import JQDataConfig
from jqdatasdk import *
# 登录Joinquant
auth(JQDataConfig.username, JQDataConfig.password)


INDEX_DATA_DIR = './indexData'
INDEX_QUOTE_DATA_DIR = os.path.join(INDEX_DATA_DIR, 'indexQuote')
INDEX_WEIGHT_DATA_DIR = os.path.join(INDEX_DATA_DIR, 'indexWeight')
for path in [INDEX_DATA_DIR, INDEX_QUOTE_DATA_DIR, INDEX_WEIGHT_DATA_DIR]:
    if not os.path.exists(path):
        os.makedirs(path)


def wind2jq(code):
    if code.endswith('.SH'):
        return code.split('.')[0] + ".XSHG"
    elif code.endswith(".SZ"):
        return code.split('.')[0] + ".XSHE"
    else:
        print(code+"代码格式错误")
        return None


def jq2wind(code):
    if code.endswith('.XSHG'):
        return code.split('.')[0] + ".SH"
    elif code.endswith(".XSHE"):
        return code.split('.')[0] + ".SZ"
    else:
        print(code+"代码格式错误")
        return None


index_code_list = ['000300.SH', '000905.SH']
start_date = pd.to_datetime('2016-01-01')
end_date = pd.to_datetime('2021-01-05')

all_trade_dates = pd.DatetimeIndex(get_all_trade_days())
to_get_date_list = all_trade_dates[(all_trade_dates >= start_date) & (all_trade_dates <= end_date)]


# %% get index quote data
for index_code in index_code_list:
    df = get_price(wind2jq(index_code), to_get_date_list[0], to_get_date_list[-1],
                   fields=['open', 'high', 'low', 'close', 'volume', 'money', 'pre_close'])
    df['ret'] = df['close']/df['pre_close'] - 1
    df['code'] = index_code
    df['datetime'] = pd.DatetimeIndex(df.index)
    df.reset_index(inplace=True, drop=True)
    df.rename(columns={'money': 'amount', 'pre_close': 'preclose', 'ret': 'pctChange'}, inplace=True)
    df.to_pickle(os.path.join(INDEX_QUOTE_DATA_DIR, index_code))


# %% get index weight data
end_of_month_list = []
for i in range(len(to_get_date_list)):
    if i > 0 and to_get_date_list.month[i] != to_get_date_list.month[i-1]:
        end_of_month_list.append(to_get_date_list[i-1])
if all_trade_dates[all_trade_dates > to_get_date_list[-1]][0].month != to_get_date_list[-1].month:
    end_of_month_list.append(to_get_date_list[-1])

for index_code in index_code_list:
    l = []
    df = get_index_weights(wind2jq(index_code), date=to_get_date_list[0])
    df.rename(columns={'date': 'update_date'}, inplace=True)
    df['index_code'] = index_code
    df.reset_index(inplace=True)
    df['code'] = df['code'].apply(func=jq2wind)
    df['datetime'] = to_get_date_list[0]
    for date in to_get_date_list:
        if date in end_of_month_list and date != to_get_date_list[0]:
            df = get_index_weights(wind2jq(index_code), date=date)
            assert df['date'].iat[0] == date
            df.rename(columns={'date': 'update_date'}, inplace=True)
            df['index_code'] = index_code
            df.reset_index(inplace=True)
            df['code'] = df['code'].apply(func=jq2wind)
        t_df = df.copy(deep=True)
        t_df['datetime'] = date
        l.append(t_df)
    result = pd.concat(l)
    result['weight'] /= 100.
    result.to_pickle(os.path.join(INDEX_WEIGHT_DATA_DIR, index_code))
    print(result.head())
