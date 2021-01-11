from WindPy import w
import numpy as np
import pandas as pd
w.start()
# ,delist_date,industry_citic"
tmp_table = pd.read_csv('./tables/materialData/S_DQ_ADJCLOSE.csv', index_col=0)
stock_code = list(tmp_table.columns)
datetime = list(tmp_table.index)
data = w.wsd(stock_code, "ipo_date","2021-01-06", "industryType=1",usedf=True)
data = w.wss(stock_code, "ipo_date,delist_date",usedf=True)[1]
data.to_csv('./tables/materialData/IPO_DELIST_DATE.csv')
error, industry_zx1 = w.wsd(stock_code, "industry_citic", "2021-01-03", "2021-01-05", "industryType=1", usedf=True)
