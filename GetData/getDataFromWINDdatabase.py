import cx_Oracle
import pandas as pd

'''
改一下client的地址
'''
cx_Oracle.init_oracle_client(lib_dir=r"d:\oracle\instantclient_19_9")

conn = cx_Oracle.connect('student2001212409', 'QTA_ymj_2020', '219.223.208.202:1521/orcl')

cursor = conn.cursor()


def get_data_from_windDB(description_sql, name):
    cursor.execute(description_sql)
    result = cursor.fetchall()
    col_list = [i[0] for i in cursor.description]
    res_df = pd.DataFrame(result, columns=col_list)
    '''
    改一下存的位置
    '''
    res_df.to_pickle('D:/AlphaSignalFromMachineLearning/GetData/tables/windDBData/{}'.format(name))


l2_indicator_sql = "SELECT S_INFO_WINDCODE, TRADE_DT,\
S_LI_INITIATIVEBUYRATE, S_LI_INITIATIVESELLRATE,\
S_LI_LARGEBUYRATE, S_LI_LARGESELLRATE FROM FILESYNC.ASHAREL2INDICATORS ORDER BY TRADE_DT "
l2_indicator_name = 'ASHAREL2INDICATORS'
get_data_from_windDB(l2_indicator_sql, l2_indicator_name)
