from WindPy import w
import pandas as pd
import numpy as np
from datetime import datetime
w.start()

fileds = "sec_name,ipo_date,delist_date,industry_sw,industry_citic".split(',')
ref_df = pd.read_csv('./materialData/S_DQ_CLOSE.csv', index_col=0)
stock_list = list(ref_df.columns)
for filed in fileds:
    wdata = w.wss(stock_list, filed, "industryType=1;tradeDate=20210105")
    if filed == 'delist_date':
        dts = wdata.Data[0]
        for i, dt in enumerate(dts):
            if dt.year == 1899:
                dts[i] = datetime(2099, 12, 30, 0, 0)

        wdata.Data[0] = dts
    d = dict(zip(stock_list, wdata.Data[0]))
    print(d)
    np.save(f'./tempData/{filed}.npy', d)

