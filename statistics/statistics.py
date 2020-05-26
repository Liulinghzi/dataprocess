'''
@Author: your name
@Date: 2020-05-25 16:20:42
@LastEditTime: 2020-05-26 10:16:21
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /dataprocess/statistics/statistics.py
'''
from scipy.stats import entropy


def pre_period_statistics(self, x, groupby_cols, op_cols, period_col=['day'], pre_preiod=1, op_mapper=None):
    """
    Parameters
    ----------
    x: pd.DataFrame

    groupby_cols: str, 计算user的序列还是item的序列，直接传入列名

    op_cols: 计算哪些行为的序列，[店铺id， 品类id， 商品id]

    period_col: pd.DataFrame

    
    一直没有理清楚的apply和agg之间的区别
    apply vs agg
        apply对所有的行进行相同的操作
        agg对一组内的所有行进行相同的操作
    apply vs map
        apply对一行，或者一列进行相同的操作，属于向量化操作
        map只能对一列，进行相同的操作，不是向量化操作，所以对列进行处理时，尽量用apply，而不是map（仅限于内置函数, lambda函数无法向量化）,所以能用np绝对不用lambda
    """
    for c in groupby_cols:
        if x[c].dtype == 'object':
            print('='*32, 'WARNING', '='*32)
            print('%s是字符串类型，会导致groupby大幅度降低速度' % c)
            print('='*32, 'WARNING', '='*32)

    stat = x.groupby(groupby_cols, as_index=False)[op_cols].agg(op_mapper)

    pre_stat = stat[period_col] + pre_preiod
    # 原本统计的第1天的数据，现在变成了第2天， 按照天进行merge，那么pre_stat中的第2天，和stat中的第二天merge，实际上是stat的第一天和第二天进行了merge
    # 但是这样只能计算第几天的统计信息，而不能计算1至10天这段时间内的信息
    #   如果要计算时间段信息， 需要先进行sum

    x = x.merge(pre_stat, on=groupby_cols + period_col, how='left')

    keys = list(op_mapper.keys())
    x[keys] = x[keys].fillna(0)
    return x


def cross_statistics(self, cross_cols, ops=['nunique', 'entropy', 'count', 'count_ratio']):
    for left_col in cross_cols:
        for right_col in cross_cols:
            if left_col == right_col:
                continue
            print('------------------ %s %s ------------------' %
                    (left_col, right_col))
            x = x.merge(x[left_col, right_col].groupby(left_col, as_index=False)[right_col].agg(
                {
                    'cross_%s_%s_nunique' % (left_col, right_col): 'nunique',
                    'cross_%s_%s_ent' % (left_col, right_col): lambda x: entropy(x.value_counts() / x.shape[0]),
                }
            )
            )
            # nunique和entropy是顺序有关的，所以两个方向都要做，下面的count ratio是顺序无关的，所以正反只做一次

            if 'count' in ops:
                if 'cross_%s_%s_count' % (left_col, right_col) not in x.columns.values \
                        and 'cross_%s_%s_count' % (right_col, left_col) not in x.columns.values:
                    x = x.merge(
                        x[[left_col, right_col, 'id']].groupby([left_col, right_col], as_index=False)['id'].
                        agg({
                            # 共现次数
                            'cross_%s_%s_count' % (left_col, right_col): 'count'
                        }), 
                        on=[left_col, right_col], 
                        how='left')

            if 'count_ratio' in ops:
                if 'cross_%s_%s_count_ratio' % (right_col, left_col) not in x.columns.values:
                    x['cross_%s_%s_count_ratio' % (right_col, left_col)] = x['cross_%s_%s_count' % (
                        left_col, right_col)] / x[left_col + '_count']  # 比例偏好
                if 'cross_%s_%s_count_ratio' % (left_col, right_col) not in x.columns.values:
                    x['cross_%s_%s_count_ratio' % (left_col, right_col)] = x['cross_%s_%s_count' % (
                        left_col, right_col)] / x[right_col + '_count']  # 比例偏好
            # x['cross_%s_%s_nunique_ratio_%s_count' % (left_col, right_col, left_col)] = x['cross_%s_%s_nunique' % (
            #     left_col, right_col)] / x[left_col + '_count']

