import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from GetData.tables.config import JQDataConfig
from jqdatasdk import auth, get_all_trade_days, get_all_securities, get_industry, get_query_count
# 登录Joinquant
auth(JQDataConfig.username, JQDataConfig.password)


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


start_date = pd.to_datetime('2017-01-01')
end_date = pd.to_datetime('2021-01-05')

all_trade_dates = pd.DatetimeIndex(get_all_trade_days())
to_get_date_list = all_trade_dates[(all_trade_dates >= start_date) & (all_trade_dates <= end_date)]

ref_df = pd.read_csv('./materialData/S_DQ_CLOSE.csv', index_col=0)
ref_df.index = pd.DatetimeIndex(ref_df.index.astype(str))

result_df = pd.DataFrame(index=ref_df.index, columns=ref_df.columns)
df_n_days = result_df.copy(deep=True)
df_existing = result_df.copy(deep=True)
df_sw_l1 = result_df.copy(deep=True)
df_sw_l2 = result_df.copy(deep=True)
df_sw_l3 = result_df.copy(deep=True)
df_zjw = result_df.copy(deep=True)
for date in tqdm(to_get_date_list):
    df1 = get_all_securities(types=["stock"], date=date)
    n_days = (date - pd.DatetimeIndex(df1['start_date'])).days + 1
    d_n_days = dict(zip([jq2wind(x) for x in df1.index], n_days))
    existing = (df1['end_date'] > date) & (df1['start_date'] <= date)
    d_existing = dict(zip([jq2wind(x) for x in df1.index], df1))

    d = get_industry(security=list(df1.index), date=date)
    l_sw_l1 = []
    l_sw_l2 = []
    l_sw_l3 = []
    l_zjw = []
    for jq_code, industry_d in d.items():
        if 'sw_l1' in industry_d:
            l_sw_l1.append(industry_d['sw_l1']['industry_name'])
            l_sw_l2.append(industry_d['sw_l2']['industry_name'])
            l_sw_l3.append(industry_d['sw_l3']['industry_name'])
        else:
            l_sw_l1.append(np.nan)
            l_sw_l2.append(np.nan)
            l_sw_l3.append(np.nan)
        if 'zjw' in industry_d:
            l_zjw.append(industry_d['zjw']['industry_name'])
        else:
            l_zjw.append(np.nan)
    d_sw_l1 = dict(zip([jq2wind(x) for x in df1.index], l_sw_l1))
    d_sw_l2 = dict(zip([jq2wind(x) for x in df1.index], l_sw_l2))
    d_sw_l3 = dict(zip([jq2wind(x) for x in df1.index], l_sw_l3))
    d_zjw = dict(zip([jq2wind(x) for x in df1.index], l_zjw))

    df_n_days.loc[date, :] = d_n_days
    df_existing.loc[date, :] = d_existing
    df_sw_l1.loc[date, :] = d_sw_l1
    df_sw_l2.loc[date, :] = d_sw_l2
    df_sw_l3.loc[date, :] = d_sw_l3
    df_zjw.loc[date, :] = d_zjw
    print(get_query_count()['spare'])

df_n_days.to_csv('./materialData/LISTED_DAYS.csv')
print(df_n_days.head())
df_existing.fillna(0).to_csv('./materialData/EXISTING.csv')
print(df_existing.head())
df_sw_l1.to_csv('./materialData/SW_L1.csv')
print(df_sw_l1.head())
df_sw_l2.to_csv('./materialData/SW_L2.csv')
print(df_sw_l2.head())
df_sw_l3.to_csv('./materialData/SW_L3.csv')
print(df_sw_l3.head())
df_zjw.to_csv('./materialData/ZJW.csv')
print(df_zjw.head())
