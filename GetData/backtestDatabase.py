import os

import pandas as pd
import numpy as np

from functools import lru_cache

from Tool import globalVars

cur_path = os.path.abspath(os.path.dirname(__file__))

ALL_TRADING_DAYS_DATA_PATH = os.path.join(cur_path, 'tables/all_trade_days.npy')
INDEX_DATA_DIR = os.path.join(cur_path, 'tables/indexData')
INDEX_QUOTE_DATA_DIR = os.path.join(INDEX_DATA_DIR, 'indexQuote')
INDEX_WEIGHT_DATA_DIR = os.path.join(INDEX_DATA_DIR, 'indexWeight')


class BacktestDatabase:
    fields = ['industry_sw1_name', 'industry_zx1_name', 'name', 'listed_date', 'is_st',
              'is_exist', 'industry_zx1_name', 'is_trading', 'market_cap', 'circulating_market_cap',
              'free_circulating_market_cap', 'open', 'high', 'low', 'close', 'volume', 'amount']

    @classmethod
    @lru_cache(maxsize=1000)
    def get_all_trade_days(cls, start_date=None, end_date=None):
        """
        获取指定日期时间段的交易日期数据，如果没有输入，默认返回所有交易时间段的数据
        :param start_date: 开始日期，如果没有输入
        :param end_date:
        :return:
        """
        all_trade_dates = pd.DatetimeIndex(np.load(ALL_TRADING_DAYS_DATA_PATH, allow_pickle=True))

        start_date = pd.to_datetime(start_date) if start_date is not None else all_trade_dates[0]
        end_date = pd.to_datetime(end_date) if end_date is not None else all_trade_dates[-1]
        trading_days = all_trade_dates[(all_trade_dates >= start_date) & (all_trade_dates <= end_date)]
        if len(trading_days) > 0:
            return trading_days
        else:
            return None

    @classmethod
    def _parse_start_and_end_date(cls, start_date, end_date):
        start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
        trade_dates = cls.get_all_trade_days(start_date, end_date)
        return trade_dates[0], trade_dates[-1], trade_dates

    @classmethod
    def get_next_trade_date(cls, date, n=1):
        if n == 0:
            return date
        try:
            all_trade_days = cls.get_all_trade_days()
            return all_trade_days[all_trade_days > date][n-1]
        except IndexError:
            raise IndexError("The given {0} days after date {1} out of the upper bound {2}"
                             .format(n, date, all_trade_days[-1]))

    @classmethod
    # @lru_cache(maxsize=1000)
    def get_daily_factor(cls, code_list, factor_list, start_date, end_date):
        """
        从数据库中读取 某一段 时间切片上 特定股票的 日频因子数据
        :param start_date: 输入指定的开始日期
        :param end_date:  输入指定的结束日期
        :param factor_list: 需要请求的那些字段，默认返回所有字段，比如 factor_list = ["is_st", "market_cap"]
        :param code_list: 返回特定的几只股票的数据
        :return:
        """

        # 将输入的时间转换成 datetime 类型，这样就可以直接和数据库里面的日期 比较
        start_date, end_date, trade_dates = cls._parse_start_and_end_date(start_date, end_date)
        if isinstance(code_list, str):
            code_list = code_list.split(",")
        if factor_list is None:
            factor_list = cls.fields
        l = []
        for field in factor_list:
            general_data = globalVars.materialData[field]
            if start_date == end_date:
                data = general_data.get_data(at=start_date).reshape(1, -1)
            else:
                data = general_data.get_data(start=start_date, end=cls.get_next_trade_date(end_date))
            df = pd.DataFrame(data, columns=general_data.columnNames)
            df['datetime'] = trade_dates
            melted = df.melt(id_vars=['datetime'], value_vars=code_list, value_name=field, var_name='code')
            sr = melted[field]
            l.append(sr)
        db_data = pd.concat(l, axis=1)
        db_data['code'] = melted['code'].values
        db_data['datetime'] = melted['datetime']

        return db_data

    @classmethod
    def get_daily_quote(cls, code_list, start_date, end_date):
        """
        从数据库中读取 某一段 时间切片上 特定股票的 日频因子数据
        :param start_date: 输入指定的开始日期
        :param end_date:  输入指定的结束日期
        :param code_list: 返回特定的几只股票的数据
        :return:
        """
        index_code_list = []
        for code in ["000300.SH", "000905.SH"]:
            if code in code_list:
                index_code_list.append(code)
                code_list.remove(code)
        quote_list = ['open', 'high', 'low', 'close', 'volume', 'preclose', 'amount']
        stock_df = cls.get_daily_factor(code_list, quote_list, start_date, end_date)
        for index_code in index_code_list:
            index_df = cls._load_index_quote(index_code)
            index_df = index_df[(index_df['datetime'] >= start_date) & (index_df['datetime'] <= end_date)]
            stock_df = pd.concat([stock_df, index_df[stock_df.columns]])
        stock_df['pctChange'] = stock_df['close'] / stock_df['preclose'] - 1
        return stock_df

    @classmethod
    @lru_cache(maxsize=1000)
    def _load_index_quote(cls, index_code):
        return pd.read_pickle(os.path.join(INDEX_QUOTE_DATA_DIR, index_code))

    @classmethod
    @lru_cache(maxsize=1000)
    def _load_index_weight(cls, index_code):
        return pd.read_pickle(os.path.join(INDEX_WEIGHT_DATA_DIR, index_code))

    @classmethod
    @lru_cache(maxsize=1000)
    def get_index_weight(cls, index_code, start_date, end_date):
        """
        通过 wss 时间切片的方法，从数据库中读取 某一天 时间切片上所有股票的 相关因子数据
        :param start_date: 输入指定的开始日期
        :param end_date:  输入指定的结束日期
        :param index_code: 股票指数代码，目前只支持000300.XSHG 和 000905.XSHG
        :return:
        """
        if index_code in ["沪深300", "hs300", "300", "HS", "000300.SH", "000300.XSHG"]:
            index_code = "000300.SH"

        elif index_code in ["中证500", "zz500", "500", "000905.SH", "000905.XSHG"]:
            index_code = "000905.SH"
        assert index_code in ['000300.SH', '000905.SH']

        start_date, end_date, trade_dates = cls._parse_start_and_end_date(start_date, end_date)
        df = cls._load_index_weight(index_code)
        return df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]

    @classmethod
    def get_stock_info(cls, code_list, start_date, end_date, field_list=None):
        """
        通过 wss 时间切片的方法，从数据库中读取 某一天 时间切片上所有股票的 相关因子数据
        :param start_date: 输入指定的开始日期
        :param end_date:  输入指定的结束日期
        :param code_list: 返回特定的几只股票的数据
        :param field_list: 指定字段
        :return:
        """
        fixed_list = ['is_trading', 'market_cap', 'circulating_market_cap', 'free_circulating_market_cap']
        if field_list is not None:
            for field in field_list:
                assert field in cls.fields
            field_list = list(set(field_list) | set(fixed_list))
        else:
            field_list = fixed_list
        df = cls.get_daily_factor(code_list, field_list, start_date, end_date)

        temp_fileds = "sec_name,ipo_date,delist_date,industry_sw,industry_citic".split(',')
        rename_dict = {'sec_name': 'name',
                       'ipo_date': 'ipo_date',
                       'delist_date': 'delist_date',
                       'industry_sw': 'industry_sw1_name',
                       'industry_citic': 'industry_zx1_name'}
        for field in temp_fileds:
            d = np.load(f'{cur_path}/tables/tempData/{field}.npy', allow_pickle=True).item()
            df[rename_dict[field]] = df['code'].map(d)
        df['is_exist'] = (df['ipo_date'] <= df['datetime']) & (df['delist_date'] >= df['datetime'])
        # Todo: 'is_st'  # 是否st
        return df


if __name__ == '__main__':
    from GetData.loadData import load_material_data

    BacktestDatabase.get_all_trade_days()
    globalVars.initialize()
    load_material_data()
    BacktestDatabase.get_daily_quote(['000300.SH'], pd.to_datetime("2020-01-01"),
                                     pd.to_datetime("2020-02-28"))
    BacktestDatabase.get_daily_factor(['000001.SZ', '000002.SZ'], ["close", "open"], pd.to_datetime("2020-01-01"),
                                      pd.to_datetime("2020-02-28"))
    BacktestDatabase.get_index_weight('沪深300', pd.to_datetime("2020-01-22"),
                                      pd.to_datetime("2020-01-22"))

    # start_date = pd.to_datetime('2016-01-01')
    # end_date = pd.to_datetime('2021-01-05')
    # all_trade_days = BacktestDatabase.get_all_trade_days()
    # to_get_date_list = all_trade_days[(all_trade_days >= start_date) & (all_trade_days <= end_date)]
    # for index_code in ['000300.SH', '000905.SH']:
    #     df = pd.read_pickle(os.path.join(INDEX_WEIGHT_DATA_DIR, index_code))
    #     l = []
    #     for date in to_get_date_list:
    #         if date not in df['datetime']:
    #             sl = df['datetime'][df['datetime'] <= date]
    #             if len(sl) == 0:
    #                 print(date)
    #                 continue
    #             shift_date = sl.iat[-1]
    #             tmp_df = df[df['datetime'] == shift_date].copy(deep=True)
    #             tmp_df['datetime'] = date
    #             l.append(tmp_df)
    #     l.append(df)
    #     res_df = pd.concat(l)
    #     res_df.sort_values('datetime').to_pickle(os.path.join(INDEX_WEIGHT_DATA_DIR, index_code))



