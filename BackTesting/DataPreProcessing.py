__all__ = ['FactorDeExtremeMethod', 'FactorStandardizeMethod', 'GroupingMethod', 'WeightMethod',
           'FactorNeutralizeMethod']
import pandas as pd
import numpy as np
import statsmodels.api as sm


class FactorDeExtremeMethod(object):
    """去极值方法"""
    def __init__(self, **kwargs):
        pass

    @classmethod
    def Method_Median(cls, factor_series, multiple=5.2):
        """
        中位数去极值法，参考天软文档
        参数 multiple 是用于计算上下轨的倍数, 默认值为5.2
        factor_series中可能有空值nan，计算均值、中位数等统计量时会跳过空值
        """
        median = factor_series.dropna().median()
        distance_to_median = (factor_series - median).abs()  # 每个数据点与中位数的距离
        median_of_distance = distance_to_median.dropna().median()     # 中位数距离的中位数

        upper_limit = median + multiple * median_of_distance  # 上轨
        lower_limit = median - multiple * median_of_distance  # 下轨
        # 替换数据
        result = factor_series.copy()
        result[result > upper_limit] = upper_limit
        result[result < lower_limit] = lower_limit
        return result

    @classmethod
    def Method_Mean_Std(cls, factor_series, multiple=3):
        """
        n倍标准差法，参考天软文档
        参数 multiple 是用于计算上下轨的倍数, 默认值为3
        factor_series中可能有空值nan，计算均值、中位数等统计量时会跳过空值
        """
        mean = factor_series.dropna().mean()
        std = factor_series.dropna().std()

        upper_limit = mean + multiple * std  # 上轨
        lower_limit = mean - multiple * std  # 下轨
        # 替换数据
        result = factor_series.copy()
        result[result > upper_limit] = upper_limit
        result[result < lower_limit] = lower_limit
        return result

    @classmethod
    def Method_Quantile(cls, factor_series, multiple=1.5):
        """
        n倍标准差法，参考天软文档
        参数 multiple 是用于计算上下轨的倍数, 默认值为3
        factor_series中可能有空值nan，计算均值、中位数等统计量时会跳过空值
        """
        quantile = factor_series.dropna().quantile([0.25, 0.5, 0.75])
        gap1 = quantile[0.75] - quantile[0.5]
        gap2 = quantile[0.5] - quantile[0.25]

        upper_limit = quantile[0.75] + multiple * gap1  # 上轨
        lower_limit = quantile[0.25] - multiple * gap2  # 下轨
        # 替换数据
        result = factor_series.copy()
        result[result > upper_limit] = upper_limit
        result[result < lower_limit] = lower_limit
        return result


class FactorStandardizeMethod(object):
    def __init__(self):
        pass

    @classmethod
    def Method_Z_Score(cls, factor_series):
        # z-score标准化
        mean = factor_series.dropna().mean()
        std = factor_series.dropna().std()
        return (factor_series - mean) / std

    @classmethod
    def Method_0_1(cls, factor_series):
        # [0, 1]正规化
        return (factor_series - factor_series.min()) / (factor_series.max() - factor_series.min())

    @classmethod
    def Method_1_1(cls, factor_series):
        # [-1, 1]标准化
        return 2 * (factor_series - factor_series.min()) / (factor_series.max() - factor_series.min()) - 1

    @classmethod
    def Method_Percentile(cls, factor_series):
        # 百分比打分法
        ss = factor_series.rank()
        ss /= ss.max()
        return ss


class FactorNeutralizeMethod(object):
    def __init__(self):
        pass

    @classmethod
    def Method_Residual(cls, factor_series, control_factor_df):
        # 线性回归不能出现NAN，此处仅取出y和X都没NaN的行
        dropna_df = pd.concat([factor_series, control_factor_df], axis=1).dropna()
        # 把行业等标签类因子变为哑变量
        dff = pd.get_dummies(dropna_df)
        # 回归
        result = sm.OLS(np.array(dropna_df.iloc[:, 0]), np.array(dff.iloc[:, 1:])).fit()
        # 把回归的残差放回原有的日期集合里
        result_sr = pd.Series(index=factor_series.index)
        result_sr.loc[dropna_df.index] = result.resid

        return result_sr


class GroupingMethod(object):
    def __init__(self):
        pass

    @classmethod
    def Method_Blank_QCut(cls, df, group_num, group_list, weight_method, max_stock_num):
        """
        空白Qcut，即不做任何因子的控制，直接对universe的股票按score进行QCut。
        :param df: universe df，index是股票wind代码，columns至少要有filter和score这两列。
        :param group_num: 分组数
        :param group_list: 分组列表 ['Qgroup_num', 'Qgroup_num-1', ... , 'Q2', 'Q1']
        :param weight_method: 配权方式
        :param max_stock_num: 细分小组内最大的选股数量
        :return: result_dict: keys是group_list里的分组名称，values是Series，Series的index是股票代码，values是权重
        """
        filter_df = df[df['filter']]
        filter_df['grouping'] = pd.qcut(filter_df['score'], group_num, group_list, duplicates='drop')
        df.loc[filter_df.index, 'grouping'] = filter_df['grouping']

        result_dict = dict()
        # 得出各组的权重
        for group_label in group_list:
            temp_df = filter_df[filter_df['grouping'] == group_label]
            temp_df = temp_df.sort_values('score', ascending=False).head(max_stock_num)  # 限制股票数量
            if weight_method == 'EW':
                temp_df['weight'] = 1 / len(temp_df)
                weight_series = temp_df['weight']
            elif weight_method == 'LVW':
                weight_series = np.sqrt(temp_df['circulating_market_cap']) / \
                                np.sqrt(temp_df['circulating_market_cap']).sum()
            elif weight_method == 'VW':
                weight_series = np.sqrt(temp_df['market_cap']) / \
                                np.sqrt(temp_df['market_cap']).sum()
            result_dict[group_label] = weight_series
        return result_dict

    @classmethod
    def Method_Group_By_Benchmark(cls, df, benchmark_df, control_dict, group_num, group_list, factor_group_list,
                                  weight_method, max_stock_num):
        """
        根据benchmark的因子分布来做中性配权。
        :param df: universe df，index是股票wind代码，columns至少要有filter和score这两列和control_dict里提到的因子。
        :param benchmark_df: benchmark股票池的因子df，与df的结构一致
        :param control_dict: 需要控制的因子，若不为空必须是OrderDict，需要控制的因子名称为key，分组数量为value，
                              非数字型因子，如行业分类，则value为空字符串 ： ""。将按顺序依次控制分组。
                              例如：OrderedDict([('申万一级行业', ''), ('流通市值', 3)])
        :param group_num:分组数
        :param group_list:分组列表 ['Qgroup_num', 'Qgroup_num-1', ... , 'Q2', 'Q1']
        :param factor_group_list:根据control_dict提到的因子进行分组的组名list
        :param weight_method:配权方式
        :param max_stock_num:细分小组内最大的选股数量
        :return:result_dict: keys是group_list里的分组名称，values是Series，Series的index是股票代码，values是权重
                 result_count_df:每个细分小组内的最大选股数量
        """
        # 只对在filter方法中过滤得出的股票进行打分和分组。
        filter_df = df[df['filter']]

        is_numeric_dict = {}
        count = 0  # 记录控制到第几层
        # 先对基准篮子分组
        for control_factor, control_group_num in control_dict.items():
            # 判断是数字型因子还是标签型因子，比如PE-TTM就是数字型，行业就是标签型。
            is_numeric = pd.api.types.is_numeric_dtype(benchmark_df[control_factor])
            is_numeric_dict[control_factor] = is_numeric
            # 第1层时，不需要先通过上一层控制后再分组。
            if count == 0:
                if is_numeric:
                    # 按指数成份股的因子划定分位区间
                    benchmark_df['group_' + control_factor] = pd.qcut(benchmark_df[control_factor],
                                                                      control_group_num, duplicates='drop',
                                                                      precision=10)
                else:
                    benchmark_df['group_' + control_factor] = benchmark_df[control_factor]
            # 非第1层时，需要先从第1层到上一层，层层控制下来再分组。所以要先groupby(factor_group_list[:count])
            else:
                if is_numeric:
                    def q_cut_func(x): return pd.qcut(x, min(control_group_num, len(x)), duplicates='drop', precision=10)
                    gb = benchmark_df.groupby(factor_group_list[:count])
                    benchmark_df['group_' + control_factor] = gb[control_factor].transform(q_cut_func)
            count += 1

        every_group_selected_stock_dict = dict()  # 记录各组选出的股票，及其权重。二重dict
        for group_label in group_list:
            every_group_selected_stock_dict[group_label] = dict()

        gb = benchmark_df.groupby(factor_group_list)
        result_count_list = []
        # 从股票池选出对应细分小组的股票
        for index, temp_df in gb:
            pos_df = filter_df.copy()
            # 按照基准股票池的分组标准取出股票池对应的股票到pos_df
            for i, sub_group in enumerate(index):
                control_factor_list = list(control_dict.keys())
                # 数字型
                if isinstance(sub_group, pd.Interval):
                    pos_df = pos_df[(sub_group.left < pos_df[control_factor_list[i]]) &
                                    (pos_df[control_factor_list[i]] <= sub_group.right)]
                # 标签型
                else:
                    pos_df = pos_df[pos_df[control_factor_list[i]] == sub_group]
                if len(pos_df) == 0:
                    break
            # length需要drop_duplicates，不然有些score重复值比较多的pos_df会不够分
            length = len(pos_df['score'].drop_duplicates())
            if 0 < length <= group_num:
                # 如果股票池在这个细分小组里面的股票数量不够分组数，那么全选。
                if weight_method == 'EW':
                    pos_df['weight'] = temp_df['权重'].sum() / length
                elif weight_method == 'LVW':
                    pos_df['weight'] = temp_df['权重'].sum() * np.sqrt(pos_df['circulating_market_cap']) / \
                                       np.sqrt(pos_df['circulating_market_cap']).sum()
                elif weight_method == 'VW':
                    pos_df['weight'] = temp_df['权重'].sum() * np.sqrt(pos_df['market_cap']) / \
                                       np.sqrt(pos_df['market_cap']).sum()

                for group_label in group_list:
                    # 记录选中的股票和权重
                    every_group_selected_stock_dict[group_label].update(pos_df['weight'].to_dict())
                    # 记录选股数量
                    result_count_list.append(list(index) + [group_label, len(pos_df)])

            elif group_num < length:
                # 如果股票池在这个细分小组里的股票数量大于分组数，那么可以按score去排序分层
                pos_df['grouping'] = pd.qcut(pos_df['score'], group_num, labels=group_list,
                                             duplicates='drop', precision=10)
                ggb = pos_df.groupby('grouping')
                for group_label, temp_df1 in ggb:
                    temp_df1 = temp_df1.sort_values('score', ascending=False).head(max_stock_num)
                    if weight_method == 'EW':
                        temp_df1['weight'] = temp_df['权重'].sum() / len(temp_df1)
                        weight_series = temp_df1['weight']
                    elif weight_method == 'LVW':
                        weight_series = temp_df['权重'].sum() * np.sqrt(temp_df1['circulating_market_cap']) / \
                                        np.sqrt(temp_df1['circulating_market_cap']).sum()
                    elif weight_method == 'VW':
                        weight_series = temp_df['权重'].sum() * np.sqrt(temp_df1['market_cap']) / \
                                        np.sqrt(temp_df1['market_cap']).sum()

                    every_group_selected_stock_dict[group_label].update(weight_series.to_dict())
                    result_count_list.append(list(index) + [group_label, len(temp_df1)])

            else:
                # 在股票备选池里找不到任何一个落在此分组区间的股票，直接把基准的股票拿来代替.
                for group_label in group_list:
                    every_group_selected_stock_dict[group_label].update(temp_df['权重'].to_dict())
                    result_count_list.append(list(index) + [group_label, len(temp_df)])

        result_count_df = pd.DataFrame(result_count_list, columns=factor_group_list + ['grouping', '数量'])
        result_dict = dict()
        for group_label in group_list:
            weight_series = pd.Series(every_group_selected_stock_dict[group_label])
            # weight_series /= weight_series.sum()   # 归一化
            result_dict[group_label] = weight_series
        return result_dict, result_count_df

    @classmethod
    def Method_Group_By_Universe(cls, df, benchmark_df, control_dict, group_num, group_list, factor_group_list,
                                 weight_method, max_stock_num):
        """
        根据universe的因子分布来分组。
        :param df: universe df，index是股票wind代码，columns至少要有filter和score这两列和control_dict里提到的因子。
        :param benchmark_df: benchmark股票池的因子df，与df的结构一致
        :param control_dict: 需要控制的因子，若不为空必须是OrderDict，需要控制的因子名称为key，分组数量为value，
                              非数字型因子，如行业分类，则value为空字符串 ： ""。将按顺序依次控制分组。
                              例如：OrderedDict([('申万一级行业', ''), ('流通市值', 3)])
        :param group_num:分组数
        :param group_list:分组列表 ['Qgroup_num', 'Qgroup_num-1', ... , 'Q2', 'Q1']
        :param factor_group_list:根据control_dict提到的因子进行分组的组名list
        :param weight_method:配权方式
        :param max_stock_num:细分小组内最大的选股数量
        :return:result_dict: keys是group_list里的分组名称，values是Series，Series的index是股票代码，values是权重
                 result_count_df:每个细分小组内的最大选股数量
        """
        # 只对在filter方法中过滤得出的股票进行打分和分组。
        filter_df = df[df['filter']].copy()
        # 1.分组
        is_numeric_dict = {}
        count = 0
        for control_factor, control_group_num in control_dict.items():
            # 判断是数字型因子还是标签型因子，比如PE-TTM就是数字型，行业就是标签型。
            is_numeric = pd.api.types.is_numeric_dtype(filter_df[control_factor])
            is_numeric_dict[control_factor] = is_numeric
            if count == 0:
                if is_numeric:
                    # 按指数成份股的因子划定分位区间
                    filter_df['group_' + control_factor] = pd.qcut(filter_df[control_factor],
                                                                   control_group_num, duplicates='drop')
                else:
                    filter_df['group_' + control_factor] = filter_df[control_factor]
            else:
                if is_numeric:
                    def q_cut_func(x): return pd.qcut(x, min(control_group_num, len(x)), duplicates='drop')
                    gb = filter_df.groupby(factor_group_list[:count])
                    filter_df['group_' + control_factor] = gb[control_factor].transform(q_cut_func)
            count += 1

        every_group_selected_stock_dict = dict()  # 各组选出的股票。key是分组Q1、Q2...value是list
        for group_label in group_list:
            every_group_selected_stock_dict[group_label] = []
        gb = filter_df.groupby(factor_group_list)
        result_count_list = []
        # 2.从股票池选出对应细分小组的股票
        for index, temp_df in gb:
            pos_df = temp_df.copy()
            length = len(pos_df)
            if 0 < length < group_num:
                # 如果股票池在这个细分小组里面的股票数量不够分组数，那么全选。
                for group_label in group_list:
                    every_group_selected_stock_dict[group_label] += list(pos_df.index)
                    result_count_list.append(list(index) + [group_label, len(pos_df)])
            elif group_num < length:
                # 如果股票池在这个细分小组里的股票数量大于分组数，那么可以按score去排序分层
                pos_df['grouping'] = pd.qcut(pos_df['score'], group_num, labels=group_list)
                ggb = pos_df.groupby('grouping')
                for group_label, temp_df1 in ggb:
                    temp_df1 = temp_df1.sort_values('score', ascending=False).head(max_stock_num)
                    every_group_selected_stock_dict[group_label] += list(temp_df1.index)
                    result_count_list.append(list(index) + [group_label, len(temp_df1)])
        result_count_df = pd.DataFrame(result_count_list, columns=factor_group_list + ['grouping', '数量'])
        # 3.配权
        result_dict = dict()
        for group_label in group_list:
            weight_series = WeightMethod.Method_Label_Neutral(
                filter_df.loc[every_group_selected_stock_dict[group_label], :], benchmark_df, method=weight_method)
            result_dict[group_label] = weight_series
        return result_dict, result_count_df


class WeightMethod(object):
    def __init__(self):
        pass

    @classmethod
    def Method_Label_Neutral(cls, universe_df, benchmark_df, label_name='industry_zx1_name', method='LVW'):
        # 此函数用于生成在某标签上的权重与基准一致（中性）的股票篮子series。同标签的默认总市值加权。常用于行业中性.
        result_series = pd.Series()
        #
        s = set(benchmark_df[label_name])
        for label in s:
            temp_df = universe_df[universe_df[label_name]==label]
            if len(temp_df) == 0:
                # 如果这个行业基准有，而股票池没有，直接用基准的权重。
                result_series = pd.concat([result_series, benchmark_df[benchmark_df[label_name]==label]['权重']])
            else:
                total_weight = benchmark_df[benchmark_df[label_name]==label]['权重'].sum()
                if method == 'EW':
                    # 等权
                    n = len(temp_df)
                    for code, _ in temp_df.iterrows():
                        result_series.loc[code] = total_weight / n
                elif method == 'VW':
                    # 总市值加权
                    total_market_cap = np.sqrt(temp_df['market_cap']).sum()
                    for code, row in temp_df.iterrows():
                        result_series.loc[code] = total_weight * np.sqrt(row['market_cap']) / total_market_cap
                elif method == 'LVW':
                    # 流通市值加权
                    total_market_cap = np.sqrt(temp_df['circulating_market_cap']).sum()
                    for code, row in temp_df.iterrows():
                        result_series.loc[code] = total_weight * np.sqrt(row['circulating_market_cap']) / total_market_cap

                else:
                    return None
        return result_series / result_series.sum()
