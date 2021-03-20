# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 21:25:50 2020

@author: 国欣然
"""
from BackTesting.Signal

CrossSectionalModels.
import pandas as pd
import json
import sys
import matplotlib.pyplot as plt
sys.path.append("../../")


def GetSignals(model,jsonPath = None, paraDict = {},code_list,factor_list,control_factor_list,start_date,end_date):

    Factor_date_dict = get_daily_factor_preprocessed(code_list, factor_list, control_factor_list, start_date, end_date)
    date_list = list(Factor_date_dict.keys())
    '''
    daily_return = daily_quote.pivot('code','datetime','return')
    '''
    daily_quote = DatabaseReader.get_daily_quote(code_list, start_date, end_date)
    daily_close = daily_quote.pivot('code', 'datetime', 'close')
    daily_return = daily_close.pct_change(axis=1)

    FactorReturn_df = pd.DataFrame(None, index=date_list, columns=factor_list)
    for i in range(len(date_list) - 1):
        date0 = date_list[i]
        date1 = date_list[i + 1]
        factor_date0 = Factor_date_dict[date0]
        return_date1 = daily_return[date1]
        Model = model(jsonPath,paraDict = {}).fit()
        FactorReturn_df.loc[date0] = Model.params
    return FactorReturn_df.dropna()
    
    
    
    
    