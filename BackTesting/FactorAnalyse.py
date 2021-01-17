import sys
sys.path.append('..')

from copy import deepcopy
from collections import OrderedDict
from datetime import timedelta
import pandas as pd
import numpy as np
import scipy.stats as st
from scipy.optimize import curve_fit
from tqdm.notebook import tqdm
import statsmodels.api as sm
from GetData.backtestDataApi import BacktestDataApi as WXDBReader
from BackTesting.DataPreProcessing import GroupingMethod


class GroupingTestResultAnalyser:
    """
    这个类用于打包回测结果，以及进行一些回测结果的分析
    """
    benchmark_id = ''  # 基准指数的代码
    group_list = None  # 分组的序号，如Q1、Q2、Q3等。

    all_group_ret_df = None  # 记录每个分组的净值，index是datetime格式日期，columns是Q1、Q2....，values是净值
    ic_series = None  # 记录持有期IC的series
    turnover_rate_df = None  # 记录每个分组的换手率，index是datetime格式日期，columns是Q1、Q2...，values是换手率
    result_count_series_dict = None  # 记录每个持有期内的分组股数，key是refresh_date，values是series，详见notebook
    half_time_period_series = None  # 记录每次的持仓的半衰期，index是refresh_date，values是半衰期
    daily_ic_df = None  # 记录每次持仓IC随持有天数的变化，index是持有天数，columns是refresh_date，values是ic
    regression_result_df = None  # 记录每次持仓的回归结果，index是refresh_date， columns是回归结果。
    industry_alpha_df = None  # 记录每次持仓各行业的超额收益，index是refresh_date， columns行业

    ret_df = None  # 用于记录回测所得净值的收益率，columns有p、b、excess，分别代表持仓、基准、超额。
    nav_df = None  # 用于记录回测所得净值，columns有p、b、excess，分别代表持仓、基准、超额。

    # 年化收益率，每年按照365天计算, 参考的是万得计算标准
    DAILY_ANNUAL_FACTOR = 365
    # 年化 1.5%
    RISK_FREE_RATE = 3.0 / 100 / DAILY_ANNUAL_FACTOR

    def __init__(self, result_dict):
        for k, v in result_dict.items():
            self.__dict__[k] = v
        self.group_list.reverse()  # 原本的group_list是Qn...Q3、Q2、Q1这样的顺序，这里把他反过来，从1开始。
        self.ret_df = self.all_group_ret_df[[self.group_list[0], self.benchmark_id]]  # 取Q1作为'p'
        self.ret_df.columns = ('p', 'b')   # position 和 benchmark
        self.ret_df['excess'] = self.ret_df['p'] - self.ret_df['b']
        self.ret_df = self.ret_df.astype(float)  # 没有这一步的话后面计算beta、最大回撤等指标时会报错。

        self.nav_df = (self.ret_df + 1).cumprod()

    def get_annual_return_statistic(self):
        """
        生成回测净值的年度统计
        :return:年度统计表 dataframe
        """
        result_df = pd.DataFrame(columns=('累计收益(%)', '年化收益(%)', '最大回撤(%)',
                                          '年化超额收益(%)', '超额最大回撤(%)',
                                          '年化alpha(%)', 'Beta', '跟踪误差(%)', '信息比率',
                                          'Sharpe比率', '超额Sharpe比率', 'Calmar比率', '超额Calmar比率'))
        for year, annual_df in self.ret_df.groupby(self.ret_df.index.year):
            # 把净值按年切割，传入分析函数进行净值分析
            result_df.loc[year] = self._get_period_return_statistic(annual_df)
        # 最后算全区间的净值分析
        result_df.loc['全区间'] = self._get_period_return_statistic(self.ret_df)
        return result_df

    def _get_period_return_statistic(self, annual_df_):
        """
        :param annual_df_: 一段时间内的净值Dataframe，需要有'p'和'b'两列，分别是组合和基准的净值日收益率
        :return: d: 一个dict，包含了所有年度统计中需要的字段
        """
        annual_df = annual_df_.copy().astype(float)  # 没有astype(float)的话后面计算beta、最大回撤等指标时会报错。
        annual_df['exceed'] = annual_df['p'] - annual_df['b']
        ann_factor = self.DAILY_ANNUAL_FACTOR
        rf = self.RISK_FREE_RATE

        days_delta = annual_df.index[-1] - annual_df.index[0]
        num_days = days_delta.days  # 待分析的净值共持续多少自然日

        d = dict()
        d['累计收益(%)'] = (1 + annual_df['p']).cumprod()[-1] - 1
        d['年化收益(%)'] = (1. + d['累计收益(%)']) ** (ann_factor / num_days) - 1

        cov = annual_df[['p', 'b']].cov().values
        d['Beta'] = cov[0, 1] / cov[1, 1]

        adj_returns = annual_df['p'] - rf  # 减无风险收益率，下同
        adj_factor_returns = annual_df['b'] - rf
        adj_excess_returns = annual_df['exceed'] - rf
        alpha_series = adj_returns - (d['Beta'] * adj_factor_returns)
        # alpha_series 可能存在缺失值
        alpha_series = alpha_series.dropna()
        # 计算累计alpha收益率
        cum_returns_final = (alpha_series + 1).cumprod(axis=0)[-1] - 1
        d['年化alpha(%)'] = (1. + cum_returns_final) ** (ann_factor / num_days) - 1

        # 计算累计alpha收益率
        cum_returns_final = (annual_df['exceed'] + 1).cumprod()[-1] - 1
        d['年化超额收益(%)'] = (1. + cum_returns_final) ** (ann_factor / num_days) - 1

        nav_data = (1 + annual_df['p']).cumprod()
        max_return = np.fmax.accumulate(nav_data)
        try:
            d['最大回撤(%)'] = (np.nanmin((nav_data - max_return) / max_return)).min()
        except:
            d['最大回撤(%)'] = np.nan

        excess_nav_data = (1 + annual_df['exceed']).cumprod()
        max_return = np.fmax.accumulate(excess_nav_data)
        try:
            d['超额最大回撤(%)'] = (np.nanmin((excess_nav_data - max_return) / max_return)).min()
        except:
            d['超额最大回撤(%)'] = np.nan

        try:
            # ddof=1，使得 算std时的分母为n-1。
            d['Sharpe比率'] = np.mean(adj_returns) / np.std(annual_df['p'], ddof=1) * \
                            np.sqrt(ann_factor)
        except:
            d['Sharpe比率'] = np.nan

        try:
            d['超额Sharpe比率'] = np.mean(adj_excess_returns) / np.std(annual_df['p'], ddof=1) * \
                            np.sqrt(ann_factor)
        except:
            d['超额Sharpe比率'] = np.nan

        try:
            d['Calmar比率'] = - d['年化收益(%)'] / d['最大回撤(%)']
        except:
            d['Calmar比率'] = np.nan

        try:
            d['超额Calmar比率'] = - d['年化超额收益(%)'] / d['超额最大回撤(%)']
        except:
            d['超额Calmar比率'] = np.nan

        d['跟踪误差(%)'] = np.nanstd(annual_df['exceed'], ddof=1)

        try:
            d['信息比率'] = np.nanmean(annual_df['exceed']) / np.nanstd(annual_df['exceed'], ddof=1)
        except:
            d['信息比率'] = np.nan

        return d


class FactorAnalyserBase(object):
    start_date = None  # 回测开始日期
    end_date = None  # 回测结束日期
    benchmark_id = None  # 基准ID
    universe = None  # 选股股票池，可选'全A'、'沪深300'、'中证500'
    cost_rate = None  # 回测时单边交易费率
    change_date_method = None  # 换仓日期模式，月初换、月末换、自定义。
    customized_universe = None  # 自定义股票池，尚未支持，占坑。
    all_trade_days = None  # 中国A股交易日序列
    recalculate_date_list = None  # 重新计算下期持仓的日期序列
    refresh_date_list = None  # 重新换仓的日期序列， 应与recalculate_date_list等长且比其滞后至少1日。
    raw_universe_df_dict = None  # 用于储存在每个recalculate_date里universe的股票因子数据，不对这里的数据清洗、加工
    processed_universe_df_dict = None  # 用于储存在每个recalculate_date里universe的股票因子数据，所有的数据清洗、加工在此
    benchmark_weight_df_dict = None  # 用于储存在每个recalculate_date里的基准指数的股票因子数据及权重

    def __init__(self, director, start_date, end_date, benchmark_id, universe, props):
        """
        :param director: 计算信号的director
        :param start_date:回测开始日期
        :param end_date:回测结束日期
        :param benchmark_id:基准ID
        :param universe:选股股票池
        :param props:其余参数，目前可设置：单边交易费率、换仓日期模式、自定义换仓日期、自定义篮子。
        """
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark_id = benchmark_id
        self.universe = universe  # 如果要自定义universe，传None

        self.cost_rate = props.get('单边交易费率', 0.0015)  # 单边
        self.change_date_method = props.get('换仓日期模式', '月初换')  # 可选“月初换”， “月底换”，“自定义”
        # 如果change_date_method取了自定义，那么要把具体的日期序列传递给change_date_list
        self.change_date_list = props.get('自定义换仓日期', None)  # 数据类型是list
        self.customized_universe = props.get('自定义篮子', None)  # pd.DataFrame, columns=('wind_code','date')

        self.all_trade_days = pd.DatetimeIndex(WXDBReader.get_all_trade_days())

        self.recalculate_date_list = None
        self.refresh_date_list = None
        self.raw_universe_df_dict = dict()
        self.processed_universe_df_dict = dict()
        self.benchmark_weight_df_dict = dict()

        self.director = director
        general_data = self.director.run()["shiftedReturn"]
        self.signals = general_data.to_DataFrame()

    def _get_date_list(self):
        """重新算持仓日期和调仓日期"""
        all_trade_days = self.all_trade_days
        if self.change_date_method == '月初换':
            # 根据上个月月底因子数据计算篮子，月初第一天下午收盘换仓
            init_date = all_trade_days[all_trade_days <= self.start_date][-1]
            recalculate_date_list = [init_date]  # 重新计算篮子的日期
            refresh_date_list = [all_trade_days[all_trade_days > init_date][0]]  # 换仓的日期
            trade_days = all_trade_days[(all_trade_days > self.start_date) & (all_trade_days <= self.end_date)]
            for i, date in enumerate(trade_days):
                # -2，因为i得出recalculate_date，而refresh_date比它长一天，另外refresh_date如果在-1则毫无意义。
                # 因为这意味着在回测的最后一天收盘时才买入新篮子。
                if i >= len(trade_days) - 2:
                    continue
                else:
                    if trade_days[i].month != trade_days[i + 1].month:
                        recalculate_date_list.append(date)
                        refresh_date_list.append(trade_days[i + 1])
        elif self.change_date_method == '月末换':
            # 根据上个月倒数第二个交易日的因子数据计算篮子，倒数第一个交易日下午收盘换仓
            init_date = all_trade_days[all_trade_days <= self.start_date][-1]
            recalculate_date_list = [init_date]  # 重新计算篮子的日期
            refresh_date_list = [all_trade_days[all_trade_days > init_date][0]]  # 换仓的日期
            trade_days = all_trade_days[(all_trade_days > self.start_date) & (all_trade_days <= self.end_date)]
            for i, date in enumerate(trade_days):
                if i >= len(trade_days) - 2:
                    continue
                else:
                    if trade_days[i].month != trade_days[i + 1].month:
                        if i != 0:
                            recalculate_date_list.append(trade_days[i-1])
                            refresh_date_list.append(trade_days[i])
        elif self.change_date_method == '自定义':
            refresh_date_list = self.change_date_list
            # recalculate_date选在每个refresh_day的前一天
            recalculate_date_list = [all_trade_days[all_trade_days<x][-1] for x in refresh_date_list]
        elif self.change_date_method == '每日换':
            in_period_days = all_trade_days[(all_trade_days >= self.start_date) & (all_trade_days <= self.end_date)]
            refresh_date_list = list(in_period_days[1:])
            recalculate_date_list = list(in_period_days[:-1])
        else:
            self.log('参数设定错误，不存在这种换仓日期模式')
            return
        self.recalculate_date_list, self.refresh_date_list = recalculate_date_list, refresh_date_list

    def prepare_data(self):
        self._get_date_list()

        for date in self.recalculate_date_list:
            # 没有指定自定义的篮子
            if self.customized_universe is None:

                # 按日期记录因子数据到dict里
                self.raw_universe_df_dict[date] = WXDBReader.get_universe(self.universe, date)

                # 按日期记录指数权重数据到dict里
                self.benchmark_weight_df_dict[date] = WXDBReader.get_index_weight(self.benchmark_id, date)

            # 指定了自定义篮子
            else:
                pass
        # 复制一份出来，后面对这个进行数据清洗。raw_universe_df_dict里的df不去进行任何去极值、标准化、中性化等操作，这些
        # 操作全部都在processed_universe_df_dict里的df上操作。
        # 1.为的是回测完能够对选出来的组合观察因子原始值的分布，比如市值的分布、市盈率的分布等操作。
        # 2.如果选择分组配权方式里的group_by_benchmark模式，只能用原始值去进行选股。不然，假如把universe的因子处理了，
        # 再把benchmark的股票因子处理了，两者就无法拿到一起比较了。
        self.processed_universe_df_dict = deepcopy(self.raw_universe_df_dict)

    def filter(self):
        """
        初步过滤一些股票，如去除PE<0 的股票等，需要用户自己定义
        """
        pass

    def rate_stock(self):
        """
        因子清洗、合成逻辑,， 需要自己定义。关键是给self.processed_universe_df_dict 的每个df加一列 'score'，分数越高越好
        :return:
        """
        pass

    def grouping_test(self, group_num, control_dict, group_by_benchmark=False, weight_method='LVW', max_stock_num=30):
        """
        分组收益分析，基于Score分组，由大到小分别是Q1、Q2...Q group_num，快速回测
        :param group_num: 分组数量
        :param control_dict: 需要控制的因子，若不为空必须是OrderDict，需要控制的因子名称为key，分组数量为value，
                              非数字型因子，如行业分类，则value为空字符串 ： ""。将按顺序依次控制分组。
                              例如：OrderedDict([('industry_zx1_name', ''), ('circulating_market_cap', 3)])
        :param group_by_benchmark: 是否按基准的因子去划分
        :param weight_method:配权方法，目前只支持EW、LVW和VW，分别是等权、流通市值平方根加权，总市值平方根加权
        :param max_stock_num 每个分组网格内最大的持股数
        :return: result_analyser:一个GroupingTestResultAnalyser类的对象。
                                  包含8个结果。分组回测的净值、分组回测每个持有周期的IC、分组回测每次的换手率、
                                  控制变量下每组的股票数量。。。。等。详见GroupingTestResultAnalyser类的属性定义。
        """
        if weight_method not in ['EW', 'LVW', 'VW']:
            self.log('错误的权重方法:{0}，权重方法必须为EW、LVW或者VW'.format(weight_method))
            return
        result_count_series_dict = dict()
        # 分组
        group_list = ['Q' + str(x + 1) for x in range(group_num)]
        group_list.reverse()
        # 如果不需要控制变量
        if not control_dict:
            for date, df in self.processed_universe_df_dict.items():
                weight_sr_dict = GroupingMethod.Method_Blank_QCut(df, group_num, group_list,
                                                                  weight_method, max_stock_num)
                for group_label, weight_series in weight_sr_dict.items():
                    # 把权重添加到raw和processed的df里，列名为group_label：Q1、Q2.....
                    self.processed_universe_df_dict[date].loc[weight_series.index, group_label] = weight_series
                    self.raw_universe_df_dict[date].loc[weight_series.index, group_label] = weight_series
                result_count_series_dict[date] = df.reset_index().groupby('grouping')['code'].count().sort_index()

        # 控制变量
        else:
            # 根据control_dict提到的因子进行分组的组名list
            factor_group_list = ['group_' + x for x in control_dict.keys()]
            # 基于benchmark去确定分组区间
            if group_by_benchmark:
                # 这种控制变量方法下只能通过原始值去分组，而不能用处理过的因子数据。
                for date, df in self.raw_universe_df_dict.items():
                    benchmark_df = self.benchmark_weight_df_dict.get(date, None)
                    weight_sr_dict, result_count_df = GroupingMethod.Method_Group_By_Benchmark(
                        df, benchmark_df, control_dict, group_num, group_list, factor_group_list,
                        weight_method, max_stock_num)
                    for group_label, weight_series in weight_sr_dict.items():
                        # 把权重添加到raw和processed的df里，列名为group_label：Q1、Q2.....
                        self.raw_universe_df_dict[date].loc[weight_series.index, group_label] = weight_series
                        self.processed_universe_df_dict[date].loc[weight_series.index, group_label] = weight_series
                    result_count_series_dict[date] = result_count_df.groupby(factor_group_list + ['grouping']).sum()

            # 基于股票池确定因子分组区间
            else:
                for date, df in self.processed_universe_df_dict.items():
                    benchmark_df = self.benchmark_weight_df_dict.get(date, None)
                    weight_sr_dict, result_count_df = GroupingMethod.Method_Group_By_Universe(
                        df, benchmark_df, control_dict, group_num, group_list, factor_group_list,
                        weight_method, max_stock_num)
                    for group_label, weight_series in weight_sr_dict.items():
                        # 把权重添加到raw和processed的df里，列名为group_label：Q1、Q2.....
                        self.raw_universe_df_dict[date].loc[weight_series.index, group_label] = weight_series
                        self.processed_universe_df_dict[date].loc[weight_series.index, group_label] = weight_series
                    result_count_series_dict[date] = result_count_df.groupby(factor_group_list + ['grouping']).sum()

        # 通过以上步骤，得到了记录了每个分组的选股数量的result_count_series_dict。
        # 且processed_universe_df_dict和raw_universe_df_dict多了Q1、Q2..列，每列是每组的权重
        # 以下开始回测

        # 用来记录回测中各分组、以及benchmark的日收益率。从第一个建仓日开始，直到回测的结束日期。
        all_group_ret_df = pd.DataFrame(columns=group_list+[self.benchmark_id],
                                        index=self.all_trade_days[(self.all_trade_days >= self.refresh_date_list[0]) &
                                                                  (self.all_trade_days <= self.end_date)])
        # 下面这几个都是用来储存回测结果的空变量。具体意义可以看GroupingTestResultAnalyser的属性定义。
        ic_series = pd.Series(index=self.refresh_date_list)
        daily_ic_df = pd.DataFrame(index=self.refresh_date_list, columns=range(1, 21))
        half_time_period_series = pd.Series(index=self.refresh_date_list)
        last_period_weight_dict = dict()
        turnover_rate_df = pd.DataFrame(columns=group_list, index=self.refresh_date_list)
        regression_result_df = pd.DataFrame(columns=('score_系数', 'score_t值', 'score_p值',
                                                     'R-squared', 'R-squared_Adj', 'F值'),
                                            index=self.refresh_date_list)
        industry_alpha_list = []
        for i, recalculate_date in tqdm(enumerate(self.recalculate_date_list)):
            refresh_date = self.refresh_date_list[i]
            next_refresh_date = self.refresh_date_list[i+1] if i < len(self.refresh_date_list)-1 else self.end_date
            if refresh_date == next_refresh_date:
                break
            all_pos_df = self.processed_universe_df_dict[recalculate_date]
            benchmark_df = self.benchmark_weight_df_dict[recalculate_date]
            # 一次性取出来所有组里选到的股票和基准股票的收益率,假设在refresh_date的收盘时换仓,所以取ret数据时start_date要+1日
            # 此外为了计算因子半衰期，start_date~end_date要取满40天。
            all_stock_list = list(set(all_pos_df[group_list].dropna(how='all').index) | set(benchmark_df.index)) + \
                               [self.benchmark_id]
            trade_day_after_39 = self.all_trade_days[min(self.all_trade_days.get_loc(refresh_date)+39,
                                                         len(self.all_trade_days))]
            all_stock_ret_df = WXDBReader.get_period_quote_timeseries(all_stock_list, refresh_date+timedelta(days=1),
                                                                      max(next_refresh_date,
                                                                          trade_day_after_39))
            # 基准指数的各股票本期收益率序列
            benchmark_ret_df = all_stock_ret_df[benchmark_df.index]
            benchmark_ret_df = benchmark_ret_df[(benchmark_ret_df.index > refresh_date) &
                                                (benchmark_ret_df.index <= next_refresh_date)]
            if len(benchmark_ret_df) == 0:
                continue
            for group_label in group_list:
                # 算组合收益率时间序列
                pos_df = all_pos_df[all_pos_df[group_label] > 0]   # 取权重大于0的出来。
                # 有些退市股票可能被选中，而这些股票取不到行情数据，这里先通过交集过滤得出stock_list。今后因子数据库补全
                # '是否退市'这个字段后可以在self.filter里通过它过滤掉，到时候删掉下面这两行代码。并解除往下第三行的注释。
                stock_list = list(set(pos_df.index) & set(all_stock_ret_df.columns))
                ret_df = all_stock_ret_df[stock_list]

                # ret_df = all_stock_ret_df[pos_df.index]
                # 前面的收益率是取了至少40天的，这里我们要开始计算净值，所以只取两个持有期中间的日期。
                ret_df = ret_df[(ret_df.index > refresh_date) & (ret_df.index <= next_refresh_date)]
                pos_ret_series = (pos_df[group_label] * ret_df).sum(axis=1)  # 组合的每日收益率

                # 计算组合换手率
                if i > 0:
                    # 把上期和今期的组合权重合并到一张df里，并补0到nan处，计算换手率
                    aligned_df = pd.concat([last_period_weight_dict[group_label], pos_df[group_label]],
                                           axis=1, sort=True)
                    aligned_df.fillna(0, inplace=True)
                    delta_weight = aligned_df.iloc[:, 1] - aligned_df.iloc[:, 0]
                    # 换手率由两部分加总而得，
                    # 左边delta_weight.abs().sum() / 2 是在换仓前后总仓位不变的情况下算的换手率，除以2的原因是
                    # 保证这部分不超出100%，比如原组合持有20%的格力，新组合换为20%的美的，这里带来的换手率算为20而不是40
                    # 右边abs(delta_weight.sum()) 是总仓位的变化的绝对值。
                    turnover_rate = delta_weight.abs().sum() / 2 + abs(delta_weight.sum())
                    turnover_rate_df.loc[refresh_date, group_label] = turnover_rate
                # 如果i == 0，意味着没有上一期，换手率为nan。不需要往换手率df里填充任何数据。
                else:
                    delta_weight = pos_df[group_label]
                # 计算佣金，在换仓次日扣去。在整个回测的最后一天没有对平仓操作扣除佣金
                commission = delta_weight.abs().sum() * self.cost_rate
                pos_ret_series.iat[0] -= commission

                # 计算Q1组合各行业的超额收益率
                if group_label == 'Q1':
                    pos_df = all_pos_df[all_pos_df['Q1'] > 0].copy()

                    # 使得每个行业的总权重都为1，即假设每个行业各自独立满仓。
                    def func(series):
                        return series / series.sum()

                    pos_df['adj_weight'] = pos_df.groupby('industry_zx1_name')['Q1'].apply(func)
                    benchmark_df['adj_weight'] = benchmark_df.groupby('industry_zx1_name')['权重'].apply(func)
                    # 每只股票权重乘以日涨跌幅，并按持有期累计，得出持有期累计盈利率。
                    pos_df['pnl_rate'] = ((1+pos_df['adj_weight'] * ret_df).cumprod()-1).iloc[-1, :]
                    benchmark_df['pnl_rate'] = ((1+benchmark_df['adj_weight'] * benchmark_ret_df).cumprod()-1).iloc[-1, :]
                    # 按行业加总，放在一个dataframe里对齐
                    aligned_df = pd.concat([pos_df.groupby('industry_zx1_name')['pnl_rate'].sum(),
                                            benchmark_df.groupby('industry_zx1_name')['pnl_rate'].sum()],
                                           axis=1, sort=True)
                    aligned_df['超额收益率'] = aligned_df.iloc[:, 0] - aligned_df.iloc[:, 1]
                    industry_alpha_list.append(aligned_df['超额收益率'])

                all_group_ret_df.loc[pos_ret_series.index, group_label] = pos_ret_series
                last_period_weight_dict[group_label] = pos_df[group_label]
            # 记录基准指数的收益率序列
            all_group_ret_df.loc[ret_df.index, self.benchmark_id] = all_stock_ret_df[self.benchmark_id].loc[ret_df.index]

            # 算持有周期的IC和回归分析，不分组
            period_ret_series = ((1+all_stock_ret_df).cumprod()-1).iloc[-1, :]  # 所有股票区间收益率Series
            period_ret_series.rename('next_period_ret', inplace=True)
            aligned_df = pd.concat([all_pos_df[['circulating_market_cap', 'industry_zx1_name', 'score']], period_ret_series], axis=1, sort=True).dropna()
            # IC
            ic_series.loc[refresh_date] = st.spearmanr(aligned_df['score'], aligned_df['next_period_ret']).correlation
            # 回归
            dummied_X_df = pd.get_dummies(aligned_df[['score', 'industry_zx1_name']])  # 把申万一级行业展开为哑变量
            dummied_X_df['const'] = 1.  # 回归的常数项
            y = np.array(aligned_df['next_period_ret'])
            X = np.array(np.array(dummied_X_df))
            if len(X) > 0:
                try:
                    regression_result = sm.OLS(y, X).fit()
                    regression_result_df.loc[refresh_date, :] = [regression_result.params[0], regression_result.tvalues[0],
                                                                 regression_result.pvalues[0], regression_result.rsquared,
                                                                 regression_result.rsquared_adj, regression_result.fvalue]
                except:
                    regression_result_df.loc[refresh_date, :] = np.nan
            else:
                regression_result_df.loc[refresh_date, :] = np.nan

                # IC半衰期，不分组
            # for j in range(len(all_stock_ret_df)):
            #     aligned_df = pd.concat([all_pos_df['score'], all_stock_ret_df.iloc[j, :]], axis=1, sort=True).dropna()
            #     daily_ic_df.loc[refresh_date, j+1] = st.spearmanr(aligned_df.iloc[:, 0], aligned_df.iloc[:, 1]).correlation
            # half_time_period_series.loc[refresh_date] = FactorAnalyserBase.cal_half_time_period_fun(daily_ic_df.loc[refresh_date])
            # print(recalculate_date.strftime("%Y-%m-%d") + "Done!")

        all_group_ret_df.iloc[0, :] = 0.
        industry_alpha_df = pd.concat(industry_alpha_list, axis=1, sort=False)
        if len(industry_alpha_list) == len(self.recalculate_date_list):
            industry_alpha_df.columns = self.recalculate_date_list
        else:
            industry_alpha_df.columns = self.recalculate_date_list[:-1]
        result_dict = {'all_group_ret_df': all_group_ret_df, 'ic_series': ic_series,
                       'turnover_rate_df': turnover_rate_df, 'result_count_series_dict': result_count_series_dict,
                       'half_time_period_series': half_time_period_series, 'daily_ic_df': daily_ic_df,
                       'regression_result_df': regression_result_df, 'group_list': group_list,
                       'benchmark_id': self.benchmark_id, 'industry_alpha_df': industry_alpha_df}
        result_analyser = GroupingTestResultAnalyser(result_dict)
        return result_analyser

    @classmethod
    def cal_half_time_period_fun(cls, daily_ic_series):
        """
        计算半衰期，具体公式见notebook。
        :param daily_ic_series: pd.Series，index是持有天数，默认是1~60，values是IC。
        :return: half_time_period: float，拟合得出的半衰期
        """
        def func(x, a, b):
            return a * np.exp(-b * x)
        _daily_ic_series = daily_ic_series.dropna()
        X = np.linspace(1, len(_daily_ic_series), len(_daily_ic_series))
        y = np.array(_daily_ic_series, dtype=np.float32)
        # Fit for the parameters a, b, c of the function `func`
        popt1, pcov1 = curve_fit(func, X, y)
        half_time_period = np.log(2) / popt1[1]
        return half_time_period

    def set_filter(self, filter_series, date):
        self.raw_universe_df_dict[date]['filter'] = filter_series
        self.processed_universe_df_dict[date]['filter'] = filter_series

    def set_score(self, score_series, date):
        self.raw_universe_df_dict[date]['score'] = score_series
        self.processed_universe_df_dict[date]['score'] = score_series

    def log(self, context):
        print(context)


class RegressionMultipleFactorAnalyser(FactorAnalyserBase):
    pass


if __name__ == '__main__':
    import datetime
    import logging
    from sklearn.metrics import mean_squared_error
    from Tool.logger import Logger
    from Tool.DataPreProcessing import DeExtremeMethod, ImputeMethod, StandardizeMethod
    from BackTesting.Signal.SignalSynthesis import SignalSynthesis
    from BackTesting.systhesisDirector import SignalDirector

    np.warnings.filterwarnings('ignore')

    logger = Logger("SignalDirector")
    logger.setLevel(logging.INFO)

    params = {
        "startDate": pd.to_datetime('20200101'),
        "endDate": pd.to_datetime('20200301'),
        "panelSize": 3,
        "trainTestGap": 1,
        "maskList": None,
        "deExtremeMethod": DeExtremeMethod.MeanStd(),
        "imputeMethod": ImputeMethod.JustMask(),
        "standardizeMethod": StandardizeMethod.StandardScaler(),
        "pipeline": None,
        "factorNameList": ['close', 'amount', 'open', 'high', 'low'],
        # params for XGBoost
        "modelParams": {
            "jsonPath": None,
            "paraDict": {
                "n_estimators": 50,
                "random_state": 42,
                "max_depth": 2}
        },
        # metric function for machine learning models
        "metric_func": mean_squared_error,
        # smoothing params
        "smoothing_params": None
    }

    director = SignalDirector(SignalSynthesis, params=params, logger=logger)

    class SampleStrategy(FactorAnalyserBase):
        def __init__(self, start_date, end_date, benchmark_id, universe, props):
            super(SampleStrategy, self).__init__(director, start_date, end_date, benchmark_id,
                                                 universe, props)

        def filter(self):
            for date, df in self.raw_universe_df_dict.items():
                df['上市天数'] = (date-df['ipo_date']).dt.days + 1  # 自然日
                self.set_filter(df['is_trading'].astype(bool) & (df['上市天数'] > 180) & df['is_exist'], date)

        def rate_stock(self):
            """
            选股逻辑，去极值、中性化、标准化等。需要用户自己定义
            """
            for date, df in self.processed_universe_df_dict.items():
                score = self.signals.loc[date, df.index.to_list()]
                self.set_score(score, date)


    fab = SampleStrategy(params['startDate'], params['endDate'], '000300.SH', '全A',
                         {"换仓日期模式": "每日换"})
    fab.prepare_data()
    fab.filter()
    fab.rate_stock()

    start_ = datetime.datetime.now()
    result = fab.grouping_test(5, OrderedDict([('circulating_market_cap', 5)]),
                               group_by_benchmark=True, weight_method='LVW')
    print(datetime.datetime.now() - start_)
    # print(result.__dict__)
    result.get_annual_return_statistic()

