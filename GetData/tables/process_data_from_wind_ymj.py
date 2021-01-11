import os
import pandas as pd
import numpy as np

pkl_path = r'D:\AlphaSignalFromMachineLearning\GetData\tables\windDBData'
temp_data = pd.read_csv(r'D:\AlphaSignalFromMachineLearning\GetData\tables\materialData\S_DQ_CLOSE.csv', index_col=0)
indicators = pd.read_pickle(pkl_path + r'\ASHAREL2INDICATORS')
indicators.TRADE_DT = indicators.TRADE_DT.astype('int')
print(indicators.head())
ind_cols = indicators.columns
initiative_buy_rate = indicators.pivot(index='TRADE_DT', columns='S_INFO_WINDCODE', values='S_LI_INITIATIVEBUYRATE')
initiative_sell_rate = indicators.pivot(index='TRADE_DT', columns='S_INFO_WINDCODE', values='S_LI_INITIATIVESELLRATE')
large_buy_rate = indicators.pivot(index='TRADE_DT', columns='S_INFO_WINDCODE', values='S_LI_LARGEBUYRATE')
large_sell_rate = indicators.pivot(index='TRADE_DT', columns='S_INFO_WINDCODE', values='S_LI_LARGESELLRATE')

# initiative_buy_rate.to_pickle(r'D:\AlphaSignalFromMachineLearning\GetData\tables\materialData\S_LI_INITIATIVEBUYRATE')
# initiative_sell_rate.to_pickle(r'D:\AlphaSignalFromMachineLearning\GetData\tables\materialData\S_LI_INITIATIVESELLRATE')
# large_buy_rate.to_pickle(r'D:\AlphaSignalFromMachineLearning\GetData\tables\materialData\S_LI_LARGEBUYRATE')
# large_sell_rate.to_pickle(r'D:\AlphaSignalFromMachineLearning\GetData\tables\materialData\S_LI_LARGESELLRATE')
temp_cols = temp_data.columns
temp_inds = temp_data.index
factor_cols = initiative_buy_rate.columns
factor_inds = initiative_buy_rate.index
cols_to_add = list(set(temp_cols) - set(factor_cols))
# cols_to_drop = list(set(factor_cols)-set(temp_cols))
inds_to_add = list(set(temp_inds) - set(factor_inds))
# inds_to_drop = list(set(factor_inds)-set(temp_inds))
df_to_append = pd.DataFrame(index=inds_to_add, columns=temp_cols)


def get_df_as_template(df):
    '''
    把df转化成和template一样的格式
    index：datetime要对齐
    columns：stock_code要对齐
    按照20170103-20210105这段时间市场上的所有股票来做
    如果时间上没有这么早，就直接append一堆空的df到前面，
    已经有股票退市了，那就把stock_code加上，也是nan
    '''
    for col in cols_to_add:
        df[col] = np.nan
    # df.drop(columns=cols_to_drop,inplace=True)
    # df.drop(index=inds_to_drop,inplace=True)
    df = df.append(df_to_append)
    df = df.loc[temp_inds, temp_cols]
    return df


initiative_buy_rate = get_df_as_template(initiative_buy_rate)
initiative_sell_rate = get_df_as_template(initiative_sell_rate)
large_buy_rate = get_df_as_template(large_buy_rate)
large_sell_rate = get_df_as_template(large_sell_rate)

initiative_buy_rate.to_csv(r'D:\AlphaSignalFromMachineLearning\GetData\tables\materialData\S_LI_INITIATIVEBUYRATE.csv')
initiative_sell_rate.to_csv(r'D:\AlphaSignalFromMachineLearning\GetData\tables\materialData\S_LI_INITIATIVESELLRATE.csv')
large_buy_rate.to_csv(r'D:\AlphaSignalFromMachineLearning\GetData\tables\materialData\S_LI_LARGEBUYRATE.csv')
large_sell_rate.to_csv(r'D:\AlphaSignalFromMachineLearning\GetData\tables\materialData\S_LI_LARGESELLRATE.csv')
# for df in [initiative_buy_rate, initiative_sell_rate, large_buy_rate, large_sell_rate]:
#     for col in cols_to_add:
#         df[col] = np.nan
#     # df.drop(columns=cols_to_drop,inplace=True)
#     # df.drop(index=inds_to_drop,inplace=True)
#     df = df.append(df_to_append)
#     df = df.loc[temp_inds, temp_cols]
