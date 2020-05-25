'''
@Author: your name
@Date: 2020-05-25 18:18:20
@LastEditTime: 2020-05-25 18:57:13
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /dataprocess/sequence/seq.py
'''


import pandas as pd
import numpy as np


def make_seq(x, groupby_feature, behavior_features, time_feature, max_len):
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
         [list(a[:max_len]) for a in arrays[i]]
          for i, feat in enumerate(behavior_features)}
    hist_dict[groupby_feature] = ukeys
    
    hist_mapper = pd.DataFrame(hist_dict)
    x = x.merge(hist_mapper, on=groupby_feature, how='left')
    return x


# data = pd.DataFrame({'a': [1, 1, 1, 2, 2, 2], 'b': [
#                     11, 12, 13, 21, 22, 23], 'c': [111, 112, 113, 221, 222, 223], })
# print(make_seq(data, 'a', ['b', 'c'], 'a', 100))


