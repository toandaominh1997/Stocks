import os
import scipy
import numpy as np
import pandas as pd
from pathlib import Path
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
from sklearn import set_config
set_config(display = 'diagram')
from pandas.api.types import infer_dtype


# Scikit Learn import
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.pipeline import FeatureUnion
# from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn.preprocessing import RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion
from sklearn import cluster
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor


# Category Encoder
import category_encoders as ce
# pyod
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA as PCA_pyod
from pyod.models.mcd import MCD
from pyod.models.sod import SOD
from pyod.models.sos import SOS
# model
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier, CatBoostRegressor
from imblearn.over_sampling import SMOTE

import re

import vnquant.DataLoader as dl
from datetime import datetime, timedelta
import pytz
from pathlib import Path


df = pd.read_parquet('./data.pq')


numeric_columns = [
    "high", 
    "low", 
    "open", 
    "close", 
    "volume",
    "mean_open_9s",
    "mean_open_26s",
    "mean_open_52s",
    
    "mean_close_9s",
    "mean_close_26s",
    "mean_close_52s",
    
    "std_close_9s",
    "std_close_26s",
    "std_close_52s",
    
    "mean_low_9s",
    "mean_low_26s",
    "mean_low_52s",
    
    "mean_high_9s",
    "mean_high_26s",
    "mean_high_52s",
    
    "mean_volume_9s",
    "mean_volume_26s",
    "mean_volume_52s",
    
    "min_low_9s",
    "min_low_26s",
    "min_low_52s",
    
    "max_high_9s",
    "max_high_26s",
    "max_high_52s",
    
    "money_flow",
    
    "mean_money_flow_9s",
    "mean_money_flow_26s",
    "mean_money_flow_52s",
    
    "std_money_flow_9s",
    "std_money_flow_26s",
    "std_money_flow_52s",
    
    "TR",
    "ATR_14s",
    
    "macd",
    "macd_signal",
    
    "RSI"
    
    
    
    
]
category_columns = ['symbol']
time_columns = ['date']
feature_columns = numeric_columns+category_columns+time_columns
target_columns = ['target_day_1', 'target_day_2', 'target_day_3', 'target_day_4', 'target_day_5']
# X = df.loc[df[target_columns].isnull().any(axis = 1)].drop(columns=target_columns)
# y = df.loc[df[target_columns].notnull().any(axis = 1)][target_columns]
# test = df.loc[df[target_columns].isnull().any(axis = 1)]

class simpleImputer(SimpleImputer):
    def fit(self, X, y = None):
        self.fn_ = X.columns.tolist()
        return super().fit(X, y)
    def transform(self, X):
        self.fn_ = X.columns.tolist()
        return super().transform(X)
        
    def get_feature_names_out(self, input_features=None):
        return self.fn_


class TimePreprocess(BaseEstimator, TransformerMixin):
    def __init__(self, list_of_features=["day", "dayofweek", "month"],):
        super().__init__()
        self.time_features = []
        self.list_of_features = list_of_features
    def fit(self, X, y = None):
        for col in X.columns:
            try:
                X[col] = pd.to_datetime(X[col])
                self.time_features.append(col)
            except:
                logger.error('Time columns cannot be convert to datetime')
        return self
    def transform(self, X):
        columns = []
        for col in self.time_features:
            X[col] = pd.to_datetime(X[col])
            if 'day' in self.list_of_features:
                columns.append(col+'_day')
                X[columns[-1]] = X[col].dt.day
            if 'month' in self.list_of_features:
                columns.append(col+'_month')
                X[columns[-1]] = X[col].dt.month
            if 'dayofweek' in self.list_of_features:
                columns.append(col+'_dayofweek')
                X[columns[-1]] = X[col].dt.dayofweek
        self.feature_names_in_ = columns
        return X[columns]
    def fit_transform(self, X, y = None):
        self.fit(X, y)
        return self.transform(X)
    def get_feature_names_out(self, input_features=None):
        print("ne: ", self.feature_names_in_)
        return self.feature_names_in_
class oneHotEncoder(ce.OneHotEncoder):
    def get_feature_names_out(self, input_features=None):
        print("ne: ", self.get_feature_names())
        return self.get_feature_names()
    
    
numeric_preprocess  = Pipeline(steps = [
    ('imputer', simpleImputer(strategy='mean', fill_value=0)),
    ('scaler_quantile', QuantileTransformer()),
    # ('scaler', MinMaxScaler())
])
category_preprocess = Pipeline(steps = [
    ('imputer', simpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder()),
    ('scaler_quantile', QuantileTransformer()),
])

time_preprocess = Pipeline(steps = [
    ('timer', TimePreprocess()),
])

preprocessing = ColumnTransformer(transformers=[
        # ('numeric_preprocess', numeric_preprocess, numeric_columns),
        # ('category_columns', category_preprocess, category_columns),
        ('time_preprocess', TimePreprocess(), time_columns),
        # ('hold_feature', HoldFeature(), hold_columns),
    ],
                                      n_jobs = 1,
                                  verbose_feature_names_out = True,
                                     )

preprocessing.fit_transform(df.loc[:, time_columns])
print(preprocessing.get_feature_names_out())