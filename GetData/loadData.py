# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:43:53 2020

@author: Evan Hu (Yi Fan Hu)

"""
#%%
import os
import pandas as pd
from copy import deepcopy
try:
    from Tool import globalVars
    from Tool import GeneralData
    from Tool.Factor import Factor
except :
    PROJECT_ROOT = 'C:\\Users\\eiahb\\Documents\\MyFiles\\WorkThing\\tf\\01task\\GeneticProgrammingProject\\AlphaSignalFromMachineLearning\\'
    os.chdir(PROJECT_ROOT)
    print("change wd to {}".format(PROJECT_ROOT))
    from Tool import globalVars
    from Tool import GeneralData
    from Tool.Factor import Factor


try:
    logger = globalVars.logger
except :
    import logging
    logger = logging.getLogger()
#%% DataFileDict

#####################################
#materialData index format 為 "%Y%m%d"
#####################################
materialDataFileDict = {
        # 'hx1': 'hx1.csv',
        # 'hx2': 'hx2.csv',
        # 'hx3': 'hx3.csv',
        # 'alpha3': 'alpha3.csv',
        # 'alpha13': 'alpha13.csv',
        # 'alpha14': 'alpha14.csv',
        # 'alpha15': 'alpha15.csv',
        # 'alpha16': 'alpha16.csv',
        # 'alpha17': 'alpha17.csv',
        'close': 'S_DQ_ADJCLOSE.csv',
        'high': 'S_DQ_ADJHIGH.csv',
        'low': 'S_DQ_ADJLOW.csv',
        'open': 'S_DQ_ADJOPEN.csv',
        # 'preclose': 'S_DQ_ADJPRECLOSE.csv',
        'amount': 'S_DQ_AMOUNT.csv',
        'volume': 'S_DQ_VOLUME.csv',
        'pctChange': 'S_DQ_PCTCHANGE.csv',

        'close_moneyflow_pct_value':'CLOSE_MONEYFLOW_PCT_VALUE.csv',
        'close_moneyflow_pct_volume': 'CLOSE_MONEYFLOW_PCT_VOLUME.csv',
        'close_net_inflow_rate_value': 'CLOSE_NET_INFLOW_RATE_VALUE.csv',
        'close_net_inflow_rate_volume': 'CLOSE_NET_INFLOW_RATE_VOLUME.csv',
        'moneyflow_pct_value': 'MONEYFLOW_PCT_VALUE.csv',
        'moneyflow_pct_volume':'MONEYFLOW_PCT_VOLUME.csv',
        'net_inflow_rate_value':'NET_INFLOW_RATE_VALUE.csv',
        'net_inflow_rate_volume':'NET_INFLOW_RATE_VOLUME.csv',
        'open_moneyflow_pct_value':'OPEN_MONEYFLOW_PCT_VALUE.csv',
        'open_moneyflow_pct_volume':'OPEN_MONEYFLOW_PCT_VOLUME.csv',
        'open_net_inflow_rate_value':'OPEN_NET_INFLOW_RATE_VALUE.csv',
        'open_net_inflow_rate_volume':'OPEN_NET_INFLOW_RATE_VOLUME.csv',
        's_mfd_inflow':'S_MFD_INFLOW.csv',
        's_mfd_inflowvolume':'S_MFD_INFLOWVOLUME.csv',
        's_mfd_inflow_closevolume':'S_MFD_INFLOW_CLOSEVOLUME.csv',
        's_mfd_inflow_openvolume':'S_MFD_INFLOW_OPENVOLUME.csv'


        # 'raw_close': 'S_DQ_ADJCLOSE.csv',
        # 'raw_high': 'S_DQ_ADJHIGH.csv',
        # 'raw_low': 'S_DQ_ADJLOW.csv',
        # 'raw_open': 'S_DQ_ADJOPEN.csv',
        # 'is_buy_limit': 'S_DQ_BUYLIMIT.csv',  # 是否涨停
        # 'is_sell_limit': 'S_DQ_SELLLIMIT.csv',  # 是否跌停
        # 'is_trading': 'S_DQ_TRADE.csv',
        # 'market_cap': 'S_VAL_MV.csv',  # 总市值
        # 'circulating_market_cap': 'S_DQ_MV.csv',  # 流通市值
        # 'free_circulating_market_cap': 'S_FREE_MV.csv',  # 自由流通市值
        # 'large_sell_rate': 'S_LI_LARGESELLRATE.csv',  # 大卖比率
        # 'large_buy_rate': 'S_LI_LARGEBUYRATE.csv',  # 大买比率
        # 'initiative_sell_rate': 'S_LI_INITIATIVESELLRATE.csv',  # 主卖比率
        # 'initiative_buy_rate': 'S_LI_INITIATIVEBUYRATE.csv',  # 主买比率
        # 'ipo_date': ''  # 上市日期
        # 'is_exist': ''  # 是否存续中
        # 'is_st': ''  # 是否st
        # 'industry_zx1_name': ''  # 中信一级行业名称
        # 'industry_sw1_name': ''  # 申万一级行业名称
        # 'name': ''   # 股票简称

    }


#####################################
#barra index format 為 "%Y-%m-%d"
#####################################
barraFileDict = {
        'beta':'beta.csv',
        'blev':'BLEV.csv',
        'bp':'BP.csv',
        'cetop':'CETOP.csv',
        # 'cmra':'CMRA.csv',
        'dastd':'DASTD.csv',
        # 'dtoa':'DTOA.csv',
        # 'egrlf':'EGRLF.csv',
        # 'egro':'EGRO.csv',
        # 'egrsf':'EGRSF.csv',
        # 'epfwd':'EPFWD.csv',
        'etop':'ETOP.csv',
        # 'hsigma':'HSIGMA.csv',
        # 'mlev':'MLEV.csv',
        'mom':'momentum.csv',
        'nonlinear_size':'Non_linear_size.csv',
        # 'report_period':'REPORT_PERIOD.csv',
        # 'sgro':'SGRO.csv',
        'size':'size.csv',
        'stoa':'STOA.csv',
        'stom':'STOM.csv',
        'stoq':'STOQ.csv',
        'beta': 'beta.csv',
        'blev': 'BLEV.csv'
    }

#%% load data functions
def load_data_csv(dataFileDict, TABLE_PATH, dictName = None, **kwargs):
    toReturnList = []
    # add dictionary to globalVars
    if dictName not in globalVars.varList:
        logger.info("The dict {} was not in globalVars".format(dictName))
        globalVars.register(dictName, {})
    for k, v in dataFileDict.items():
        if k not in globalVars.__getattribute__(dictName):
            data = GeneralData(name = k, filePath = os.path.join(TABLE_PATH, v), **kwargs)
            globalVars.__getattribute__(dictName)[k] = data
            # print('==================================================================\n\
            #       {} is now in globalVars.{}\n'.format(k, dictName), data)
            logger.info('{} is now in globalVars.{}'.format(k, dictName))
            toReturnList.append(k)
        else:
            # print('==================================================================\n\
            #       {} is already in globalVars.{}\n'.format(k, dictName))
            logger.info('{} is already in globalVars.{}'.format(k, dictName))
    return(toReturnList)

def load_data(name, filedir, filetype = 'h5', dataFileDict = None,**kwargs): 
    '''
    關於 globalVars:
    現在不支持直接在 globalVars 中建立非字典的變量(以前會看讀取的數據是不是字典，不是的話直接建立變量如 globalVars.變量)，
    但是仍然支持在讀數據時在 globalVars 中建立字典，會提示 "The dict {} was not in globalVars".format(name) 

    可以用兩種不同的方式讀取數據，如果沒有指定的話就是使用 h5 檔案
    h5:
    指定字典名稱 name 與 檔案位置可以使用與檔案名稱不同的字典名
    會先嘗試用 filedir 直接讀取檔案如: .\\AlphaSignalFromMachineLearning\\GetData\\h5\\materialData.h5
    那麼就會直接讀該檔案，並用 name 在 globalVars 中建立字典
    如果指定的 filedir 沒法讀取(他不是 h5 file 或是我們只指定了資料夾位置)，會嘗試讀取 filedir 資料夾底下的 name.h5 檔 (就是 "{}.h5".format(name))
    並且使用 name 在 globalVars 中建立字典
    csv:
    會先嘗試讀取 filedir 資料夾底下的 csv 檔案，如果找不到 csv 如: .\\GeneticProgrammingProject\\AlphaSignalFromMachineLearning\\GetData\\tables
    如果讀不到就會嘗試: os.path.join(filedir, name) 底下的資料夾如: .\\GeneticProgrammingProject\\AlphaSignalFromMachineLearning\\GetData\\tables\\barra
    思路與 h5 是一樣的
    注意使用 csv 模式時候需要傳入 dataFileDict 來指定 csv 的檔名，否則報錯
    此外可以使用 indexFormat 來指定不同的 datetime 的 format (此目的為適應不同 csv 檔案標註日期的格式) default 為 "%Y-%m-%d"


    '''   
    if filetype == 'h5':
        # 如果用的是 h5 file
        if name not in globalVars.varList:
            logger.info("The dict {} was not in globalVars".format(name))
            globalVars.register(name, {})
        try:
            assert os.path.exists(filedir), "FileNotFound, please check the file path is currect"
            hdf = pd.HDFStore(filedir)
        except Exception as e:
            print(e)
            print("try with {}".format(os.path.join(filedir, '{}.h5'.format(name))))
            assert os.path.exists(os.path.join(filedir, '{}.h5'.format(name))), "FileNotFound, please check the file path is currect"
            hdf = pd.HDFStore(os.path.join(filedir, '{}.h5'.format(name)))
        for rawk in hdf.keys():
            k = rawk.split('/')[1]
            v = GeneralData(k, hdf.get(k))
            globalVars.__getattribute__(name)[k] = v
        logger.info('{} is now in globalVars.{}'.format(list(hdf.keys()), name))
        return(list(hdf.keys()))            
    elif filetype == "csv":
        # 如果用的是 csv file
        indexFormat =  "%Y-%m-%d"
        if dataFileDict == None:
            print("dataFileDict is needed in csv mode")
            raise Exception
        if "indexFormat" in kwargs:
            indexFormat = kwargs['indexFormat']
        try:
            toReturnList = load_data_csv(dataFileDict = dataFileDict, TABLE_PATH = filedir, dictName=name, indexFormat = indexFormat)
        except FileNotFoundError as fnfe:
            filedir = os.path.join(filedir, name)
            print("try with {}".format(filedir))
            toReturnList = load_data_csv(dataFileDict = dataFileDict, TABLE_PATH = filedir, dictName=name, indexFormat = indexFormat)
        except Exception as e:
            print(e)
            raise e
        return(toReturnList)





def simple_load_factor(factorName):
    if 'factors' not in globalVars.varList:
        globalVars.register('factors', {})
    globalVars.factors['{}_factor'.format(factorName)] = Factor('{}_factor'.format(factorName), globalVars.materialData[factorName])
    print(factorName, 'is now in globalVars.factors')

 

def align_data(data, alignTo):
    data_df = pd.DataFrame(data.generalData, index=data.timestamp, columns=data.columnNames)
    reindexed = data_df.reindex(index=alignTo.timestamp, columns=alignTo.columnNames)
    toReturn = GeneralData(data.name, generalData=reindexed)
    return(toReturn)

def align_all_to(dict_, alignTo):
    dict__ = deepcopy(dict_)
    for k, v in dict__.items():
        dict__[k] = align_data(v, alignTo)
    return(dict__)

        
    
    
    
#%% main
if __name__ == '__main__':
    PROJECT_ROOT = "c:\\Users\\eiahb\\Documents\\MyFiles\\WorkThing\\tf\\01task\\GeneticProgrammingProject\\AlphaSignalFromMachineLearning\\"
    import os
    os.chdir(PROJECT_ROOT)
    from Tool import globalVars

    globalVars.initialize()
    # read h5
    # 用例 1 
    # load_data("barra",
    #     os.path.join(os.path.join(PROJECT_ROOT,"data"), "h5")
    # )

    # 用例 2
    load_data("materialData",
        os.path.join(os.path.join(os.path.join(PROJECT_ROOT,"data"), "h5"), "materialData_newData.h5")
    )

    # #read csv
    # # 用例 3 
    # load_data("barra",
    #     os.path.join(os.path.join(os.path.join(PROJECT_ROOT,"data"), "tables"), "barra"),
    #     filetype="csv",dataFileDict=barraFileDict
    # )

    # 用例 4
    # load_data("materialData",
    #     os.path.join(os.path.join(PROJECT_ROOT,"data"), "tables"),
    #     filetype="csv",dataFileDict=materialDataFileDict,indexFormat = "%Y%m%d"
    # )





# %%
# h5_path = "C:\\Users\\eiahb\\Documents\MyFiles\\WorkThing\\tf\\01task\\GeneticProgrammingProject\\AlphaSignalFromMachineLearning\\data\\h5"
# hdf = pd.HDFStore(os.path.join(h5_path, '{}.h5'.format("materialData_newData")))
# # %%
# for k, v in globalVars.materialData.items():
#     print(v)
#     hdf.put(k, v.to_DataFrame())
# hdf.close()

# %%
# %%
