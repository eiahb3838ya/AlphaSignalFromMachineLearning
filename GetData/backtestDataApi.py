import pandas as pd
import datetime

from .backtestDatabase import BacktestDatabase as DatabaseReader


class BacktestDataApi:
    @classmethod
    def get_universe(cls, bk, date, factor_list=None):
        """
        板块名称字符串，支持 “沪深300”、“中证500”
        :param bk: 版块名称，比如 沪深300
        :param date: 指定的日期， 比如 2018-11-30 目前只有每个月最后一个交易日的数据，其他时间返回的数据是None
        :param factor_list: 暂时不支持指定，默认返回全部的factor列表
        :return:
        """
        # 获取最近交易日的日期，自动过滤非交易日
        date = cls.get_nearest_trading_date(date)
        stock_info = DatabaseReader.get_stock_info(None, date, date, factor_list)
        # 根据 stock_code_list 获取指定日期的因子数据
        if bk in ["沪深300", "hs300", "300", "HS", "000300.SH"]:
            df = DatabaseReader.get_index_weight("000300.SH", start_date=date, end_date=date)
            selected_df = stock_info[stock_info['code'].isin(list(df['code']))]
        elif bk in ["中证500", "zz500", "500", "000905.SH"]:
            df = DatabaseReader.get_index_weight("000905.SH", start_date=date, end_date=date)
            selected_df = stock_info[stock_info['code'].isin(list(df['code']))]
        elif bk == '全A':
            selected_df = stock_info
        else:
            raise ValueError("输入的版块名称错误 ：{0}".format(bk))
        selected_df.set_index('code', inplace=True)
        return selected_df

    @classmethod
    def get_index_weight(cls, index_id, date, factor_list=None):
        """
        获取指数在某一天的权重信息
        :param date: 指定的日期，date 输入格式为 yyyy-mm-dd 的方式
        :param index_id: 对应的指数代码
        :param factor_list: 具体返回哪些因子列表 
        :return:
        """
        pd_data = DatabaseReader.get_index_weight(index_id, start_date=date, end_date=date)
        if pd_data is not None:
            pd_data.set_index("code", inplace=True)
            # 列名重命名 权重 isST 重命名为 ST 保持和股票因子 ST数据一致
            pd_data.rename(columns={"weight": "权重"}, inplace=True)

            weight_data = pd_data.loc[:, ["权重"]]
            # 然后获取因子数据
            stock_info = DatabaseReader.get_stock_info(weight_data.index.tolist(), date, date, factor_list)
            stock_info.set_index('code', inplace=True)
            pd_data = pd.concat([weight_data, stock_info], axis=1, sort=False)
            return pd_data
    
    @classmethod
    def get_index_weight_by_date_list(cls, index_id, date_list, factor_list=None):
        """
        获取指数在某个时间列表的的权重信息
        :param date_list: 指定日期组成的list,比如["2005-04-29", "2005-05-09"]
        :param index_id: 对应的指数代码
        :param factor_list: 具体返回哪些因子列表 
        :return:
        """
        result = dict()
        for date in date_list:
            pd_data = DatabaseReader.get_index_weight(index_id, start_date=date, end_date=date)
            if pd_data is not None:
                pd_data.set_index("code", inplace=True)
                pd_data.rename(columns={"weight": "权重"}, inplace=True)
                weight_data = pd_data.loc[:, ["权重"]]
                # 然后获取因子数据
                stock_info = DatabaseReader.get_stock_info(weight_data.index.tolist(), date, date, factor_list)
                stock_info.set_index('code', inplace=True)
                pd_data = pd.concat([weight_data, stock_info], axis=1, sort=False)
                result[pd.to_datetime(date).strftime("%Y-%m-%d")] = pd_data
            else:
                result[pd.to_datetime(date).strftime("%Y-%m-%d")] = None

        return result

    @classmethod
    def get_period_return(cls, code_list, start_date=None, end_date=None):
        """
        返回某个股票，或者某列股票，在某个时间区间段内的收益， 注意是区间段的收益，不是时间收益率序列

        股票或指数的字符串代码，或者代码的list，指数行情目前只需要 000300.SH，000016.SH 和 399905.SZ，以及申万一级行业指数
        如果传入的是单个代码，返回一个浮点数。如果传入的是一个list，返回一个series，index是股票代码，值是收益率。
        收益率= enddate_close/startdate_close-1
        （采用后复权的价格）
        :param stock_code: 股票代码列表
        :param start_date: 开始日期
        :param end_date: 结束日期
        :return:
        """
        # 对于时间的预处理，获取最近的交易日期 
        # 获取所有可 交易日期序列 
        start_date = cls.get_nearest_trading_date(start_date)
        end_date = cls.get_nearest_trading_date(end_date)
        if start_date > end_date:
            raise ValueError("输入的开始日期{0} 大于 结束日期{1}".format(start_date, end_date))
        # 开始日期
        start_date_data = DatabaseReader.get_daily_quote(code_list, start_date, start_date).set_index('code')
        # 结束日期
        end_date_data = DatabaseReader.get_daily_quote(code_list, end_date, end_date).set_index('code')
        # 两者拼接 
        data = pd.concat([start_date_data['close'], end_date_data['close']],
                         axis=1, sort=False)
        data = data.loc[code_list]
        data.columns = ["起始日", "结束日"]
        ret = data.loc[:, "结束日"] / data.loc[:, "起始日"] - 1
        ret.name = "收益率"
        return ret

    @classmethod
    def get_stock_return_timeseries(cls, stock_code, start_date, ndays=1):
        """
        基于某个日期，往前或者往后推，去计算某个时间段的收益率时间日期序列
        :param stock_code: 股票或指数字符串代码，或者是代码的list
        :param start_date: 指定某个日期
        :param ndays: 交易日天数，最大支持220，最小支持1. 正数表示往未来看nday，负数表示往历史看nday
        如果传入的stock_code是list，返回一个DataFrame，index是交易日期，columns是股票代码。
        值是每个股票在历史/未来ndays天的收益率序列，行数等于ndays。如果股票当天停牌，则收益率值为nan。
        在更新版的数据库中，会直接采用 涨跌幅2 这个字段，现在暂时使用其他方法来解决
        :return:
        """
        # 获取开始时间段
        start_date = cls.get_nearest_trading_date(start_date)
        # 获取所有可 交易日期序列
        trading_days = DatabaseReader.get_all_trade_days()
        # 开始日期
        # 寻找开始节点
        temp = trading_days <= start_date
        idx = temp.sum()
        trading_days_list = trading_days.tolist()
        # 从零开始，比如 idx是1，那么30个之后是30， 其实是 1+ndays-1 然后是从零开始，所以需要减去2
        end_index = max(idx, idx + ndays-2)
        end_date = trading_days_list[end_index]
        end_date = pd.to_datetime(end_date)
        # 从数据库获取对应的数据
        pd_data = DatabaseReader.get_daily_quote(stock_code, start_date=start_date, end_date=end_date)
        return pd_data

    @classmethod
    def get_period_quote_timeseries(cls, code_list, start_date, end_date):
        df = DatabaseReader.get_daily_quote(code_list, start_date, end_date)
        ret_df = df.pivot(index='datetime', columns='code', values='pctChange')
        return ret_df

    @classmethod
    def get_nearest_trading_date(cls, input_date):
        if input_date is None:
            input_date = datetime.datetime.today()
        trading_days = DatabaseReader.get_all_trade_days()
        # 获取距离今日最近交易日的数据 ，如果今天不是交易日，那么看上一个交易日是否是交易日
        input_date = pd.to_datetime(input_date)
        while input_date not in trading_days:
            input_date -= datetime.timedelta(days=1)
        return input_date

    @classmethod
    def get_all_trade_days(cls, start_date=None, end_date=None):
        return DatabaseReader.get_all_trade_days(start_date, end_date)


if __name__ == "__main__":
    import time

    s = time.time()
    print(BacktestDataApi.get_universe('沪深300', '2020-05-20', ['close', 'circulating_market_cap']))
    print(time.time() - s)

    s = time.time()
    print(BacktestDataApi.get_index_weight('000300.SH', '2020-05-20', ['close', 'circulating_market_cap']))
    print(time.time() - s)

    s = time.time()
    print(BacktestDataApi.get_index_weight_by_date_list('000300.SH', ['2020-05-20', '2020-05-21'],
                                                        ['close', 'circulating_market_cap']))
    print(time.time() - s)

    s = time.time()
    print(BacktestDataApi.get_period_return(['000001.SZ', '000002.SZ'], '2020-05-20', '2020-06-20'))
    print(time.time() - s)

    s = time.time()
    print(BacktestDataApi.get_stock_return_timeseries(['000001.SZ', '000002.SZ'], '2020-05-20', 30))
    print(time.time() - s)

    s = time.time()
    print(BacktestDataApi.get_nearest_trading_date('2020-06-20'))
    print(time.time() - s)

