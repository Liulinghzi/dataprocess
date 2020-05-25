'''
@Author: your name
@Date: 2020-05-25 14:47:38
@LastEditTime: 2020-05-25 16:11:21
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /dataprocess/transform/encoder.py
'''

from category_encoders import OrdinalEncoder, OneHotEncoder, CountEncoder, TargetEncoder
import pickle
import json
import time
import os

import pandas as pd


class Encoder():
    encode_methods = {
        'OrdinalEncoder': OrdinalEncoder,
        'OneHotEncoder': OneHotEncoder,
        'CountEncoder': CountEncoder,
        'TargetEncoder': TargetEncoder,
    }
    # spark_encode_methods = {
    #     'mean_encoder':,
    #     'target_encoder':,
    #     'label_encoder':,
    #     'onehot_encoder'
    # }
    # target_encoder，mean_encoder在编码时，不能够把训练集和验证机concat在一起进行编码
    # label_encoder,onehot_encoder可以

    def __init__(self, sparksess=None, logdir='/encoder', handle_unknown='-99999', save_encoder=False):
        self.spark = sparksess
        self.logdir = logdir
        self.save_encoder

        self.ordinal_encoder_features = []
        self.onehot_encoder_features = []
        self.count_encoder_features = []
        self.target_encoder_features = []
        self.ordinal_encoder = OrdinalEncoder(
            cols=self.ordinal_encoder_features, return_df=True, handle_unknown=handle_unknown)
        self.onehot_encoder = OneHotEncoder(
            cols=self.onehot_encoder_features, return_df=True, handle_unknown=handle_unknown)
        self.count_encoder = CountEncoder(
            cols=self.count_encoder_features, return_df=True, handle_unknown=handle_unknown)
        self.target_encoder = TargetEncoder(
            cols=self.target_encoder_features, return_df=True, handle_unknown=handle_unknown)

    def fit(self, X_train, X_val=None, y_train=None, y_val=None, method_mapper=None):
        for feat in method_mapper:
            if method_mapper[feat] == 'OrdinalEncoder':
                self.ordinal_encoder_features.append(feat)
            elif method_mapper[feat] == 'OneHotEncoder':
                self.onehot_encoder_features.append(feat)
            elif method_mapper[feat] == 'CountEncoder':
                self.count_encoder_features.append(feat)
            elif method_mapper[feat] == 'TargetEncoder':
                self.target_encoder_features.append(feat)
            else:
                raise ValueError(
                    '编码方式只支持[OrdinalEncoder, OneHotEncoder, CountEncoder, TargetEncoder], 接收到%s' % feat)

        if self.spark is None:
            if len(self.ordinal_encoder_features) != 0 or len(self.onehot_encoder_features) != 0:
                X_whole = X_train.append(X_val)
                y_whole = None
                if not y_train is None and not y_val is None:
                    y_whole = y_train.append(y_val)

                X_whole = self.ordinal_encoder.fit_transform(X_whole, y_whole)
                X_whole = self.onehot_encoder.fit_transform(X_whole, y_whole)
                X_train = X_whole[:len(X_train)]
                X_val = X_whole[len(X_train):]

            X_train = self.count_encoder.fit_transform(X_train, y_train)
            X_val = self.count_encoder.transform(X_val, y_val)
            X_train = self.target_encoder.fit_transform(X_train, y_train)
            X_val = self.target_encoder.transform(X_val, y_val)

            if self.save_encoder:
                self.save_encoder()
        return X_train, y_train, X_val, y_val

    def transform(self, X, y=None):
        X = self.ordinal_encoder.transform(X, y)
        X = self.onehot_encoder.transform(X, y)
        X = self.count_encoder.transform(X, y)
        X = self.target_encoder.transform(X, y)
        return X, y

    def fit_transform(self, X_train, X_val=None, y_train=None, y_val=None, method_mapper=None):
        self.fit(X_train, X_val, y_train, y_val, method_mapper)
        X_train, y_train = self.transform(X_train, y_train)
        if X_val is not None:
            X_val, y_val = self.transform(X_val, y_val)
        return X_train, y_train, X_val, y_val

    def save_encoder(self):
        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        os.makedirs(os.path.join(self.logdir, now))

        with open(os.path.join(self.logdir, now, 'OrdinalEncoder.pkl'), 'wb') as f:
            pickle.dump(self.ordinal_encoder, f)
        with open(os.path.join(self.logdir, now, 'OneHotEncoder.pkl'), 'wb') as f:
            pickle.dump(self.onehot_encoder, f)
        with open(os.path.join(self.logdir, now, 'CountEncoder.pkl'), 'wb') as f:
            pickle.dump(self.count_encoder, f)
        with open(os.path.join(self.logdir, now, 'TargetEncoder.pkl'), 'wb') as f:
            pickle.dump(self.target_encoder, f)

        with open(os.path.join(self.logdir, now, 'OrdinalEncoderFeatures.json'), 'w') as f:
            json.dump(self.ordinal_encoder_features, f)
        with open(os.path.join(self.logdir, now, 'OneHotEncoderFeatures.json'), 'w') as f:
            json.dump(self.onehot_encoder_features, f)
        with open(os.path.join(self.logdir, now, 'CountEncoderFeatures.json'), 'w') as f:
            json.dump(self.count_encoder_features, f)
        with open(os.path.join(self.logdir, now, 'TargetEncoderFeatures.json'), 'w') as f:
            json.dump(self.target_encoder_features, f)

    def load_encoder(self, logdir=None):
        with open(os.path.join(self.logdir, 'OrdinalEncoder.pkl'), 'wb') as f:
            pickle.dump(self.ordinal_encoder, f)
        with open(os.path.join(self.logdir, 'OneHotEncoder.pkl'), 'wb') as f:
            pickle.dump(self.onehot_encoder, f)
        with open(os.path.join(self.logdir, 'CountEncoder.pkl'), 'wb') as f:
            pickle.dump(self.count_encoder, f)
        with open(os.path.join(self.logdir, 'TargetEncoder.pkl'), 'wb') as f:
            pickle.dump(self.target_encoder, f)

        with open(os.path.join(self.logdir, 'OrdinalEncoderFeatures.json'), 'w') as f:
            json.dump(self.ordinal_encoder_features, f)
        with open(os.path.join(self.logdir, 'OneHotEncoderFeatures.json'), 'w') as f:
            json.dump(self.onehot_encoder_features, f)
        with open(os.path.join(self.logdir, 'CountEncoderFeatures.json'), 'w') as f:
            json.dump(self.count_encoder_features, f)
        with open(os.path.join(self.logdir, 'TargetEncoderFeatures.json'), 'w') as f:
            json.dump(self.target_encoder_features, f)


data = pd.DataFrame({'a':[1,2,3], 'b':[11,22,33], 'c':[111,222,333], 'd':[1111,2222,3333], 'e':[1111,2222,3333]})
encoder = Encoder(logdir='./encoders', save_encoder=True)
mapper = {
    'a':'OrdinalEncoder',
    'b':'OneHotEncoder',
    'c':'CountEncoder',
    'd':'TargetEncoder',
}
# encoder.fit(X_train=data[['a', 'b', 'c', 'd']],  y_train=data['e'], method_mapper=mapper)
# print(encoder.transform(X=data[['a', 'b', 'c', 'd']]))
print(encoder.fit_transform(X_train=data[['a', 'b', 'c', 'd']],  y_train=data['e'], method_mapper=mapper))