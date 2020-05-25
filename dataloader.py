'''
@Author: your name
@Date: 2020-05-25 14:31:10
@LastEditTime: 2020-05-25 14:45:03
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /dataprocess/dataLoader.py
'''
import numpy as np
import pandas as pd
import os
import sys

import pyspark

from pyspark.sql import SQLContext, SparkSession


os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
class SparkDataLoader():
    def __init__(self, sparksess):
        self.spark=sparksess

    def load(self, filepath, format='csv'):
        if format == 'csv':
            spark_df = self.spark.read.format('CSV').option('header', 'true').option('inferSchema', 'true').load(filepath)
            spark_df.write.save(filepath[:-4], format='parquet')
            spark_df = self.spark.read.load(filepath[:-4])
        else:
            spark_df = self.spark.read.load(filepath[:-4])

        return spark_df