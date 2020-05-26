'''
@Author: your name
@Date: 2020-05-25 18:18:20
@LastEditTime: 2020-05-26 10:15:23
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /dataprocess/sequence/seq.py
'''


import pandas as pd
import numpy as np


def make_seq(x, groupby_feature, behavior_features, time_feature, max_len):
    """
    Parameters
    ----------
    x: pd.DataFrame

    groupby_feature: str, 计算user的序列还是item的序列，直接传入列名

    behavior_features: 计算哪些行为的序列，[店铺id， 品类id， 商品id]

    time_feature: 时间特征列，需要先按照这里进行排序

    max_len: 返回的seq的长度, 不足的用0填充

    """
    # 在numpy层面进行计算，速度快
    if isinstance(behavior_features, list):
        features = [groupby_feature] + behavior_features
    else:
        features = [groupby_feature, behavior_features]

    key_value_list = x[features].sort_values(
        [groupby_feature, time_feature]).values.T

    keys = key_value_list[0]
    ukeys, index = np.unique(keys, True)
    # 以groupby_feature作为第一排序，然后直接找到每次groupby_feature变化时候的index

    arrays = [np.split(values, index[1:]) for values in key_value_list[1:]]
    hist_dict = {
        'hist_'+behavior_features[i]:
         [list(a[:max_len])  if len(a)>max_len else list(a) + [0]*(max_len - len(a)) for a in arrays[i]]
          for i, feat in enumerate(behavior_features)}
    hist_dict[groupby_feature] = ukeys
    
    hist_mapper = pd.DataFrame(hist_dict)

    x.set_index(groupby_feature, inplace=True)
    hist_mapper.set_index(groupby_feature, inplace=True)
    x = x.merge(hist_mapper, left_index=True, right_index=True)
    x = x.reset_index()
    # x = x.merge(hist_mapper, on=groupby_feature, how='left')
    return x


data = pd.DataFrame(
    {
        'a': list(range(1000000)), 
        'b': list(range(1000000)), 
        'c': list(range(1000000))})

print(make_seq(data, 'a', ['b', 'c'], 'a', 100))
