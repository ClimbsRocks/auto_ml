from collections import OrderedDict
import csv
import datetime
import itertools
import dateutil
import math
import os
import random

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import load_boston
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.feature_selection import GenericUnivariateSelect, RFECV, SelectFromModel
# from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
# from sklearn.linear_model import RandomizedLasso, RandomizedLogisticRegression, RANSACRegressor, LinearRegression, Ridge, Lasso, ElasticNet, LassoLars, OrthogonalMatchingPursuit, BayesianRidge, ARDRegression, SGDRegressor, PassiveAggressiveRegressor, LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.metrics import mean_squared_error, make_scorer, brier_score_loss, accuracy_score, explained_variance_score, mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import pandas as pd
import pathos
import scipy

# XGBoost can be a pain to install. It's also a super powerful and effective package.
# So we'll make it optional here. If a user wants to install XGBoost themselves, we fully support XGBoost!
# But, if they just want to get running out of the gate, without dealing with any installation other than what's done for them automatically, we won't force them to go through that.
# The same logic will apply to deep learning with Keras and TensorFlow
global xgb_installed
xgb_installed = False
try:
    import xgboost as xgb
    xgb_installed = True
except NameError:
    pass
except ImportError:
    pass

if xgb_installed:
    import xgboost as xgb


def write_gs_param_results_to_file(trained_gs, most_recent_filename):

    timestamp_time = datetime.datetime.now()
    write_most_recent_gs_result_to_file(trained_gs, most_recent_filename, timestamp_time)

    grid_scores = trained_gs.grid_scores_
    scorer = trained_gs.scorer_
    best_score = trained_gs.best_score_

    file_name = 'pipeline_grid_search_results.csv'
    write_header = False
    if not os.path.isfile(file_name):
        write_header = True

    with open(file_name, 'a') as results_file:
        writer = csv.writer(results_file, dialect='excel')
        if write_header:
            writer.writerow(['timestamp', 'scorer', 'best_score', 'all_grid_scores'])
        writer.writerow([timestamp_time, scorer, best_score, grid_scores])


def write_most_recent_gs_result_to_file(trained_gs, most_recent_filename, timestamp):

    timestamp_time = timestamp
    grid_scores = trained_gs.grid_scores_
    scorer = trained_gs.scorer_
    best_score = trained_gs.best_score_

    file_name = most_recent_filename

    write_header = False
    make_header = False
    if not os.path.isfile(most_recent_filename):
        header_row = ['timestamp', 'scorer', 'best_score', 'cv_mean', 'cv_all']
        write_header = True
        make_header = True

    rows_to_write = []

    for score in grid_scores:

        row = [timestamp_time, scorer, best_score, score[1], score[2]]

        for k, v in score[0].items():
            if make_header:
                header_row.append(k)
            row.append(v)
        rows_to_write.append(row)
        make_header = False


    with open(file_name, 'a') as results_file:
        writer = csv.writer(results_file, dialect='excel')
        if write_header:
            writer.writerow(header_row)
        for row in rows_to_write:
            writer.writerow(row)


def safely_drop_columns(df, cols_to_drop):
    safe_cols_to_drop = []
    for col in cols_to_drop:
        if col in df.columns:
            safe_cols_to_drop.append(col)

    df = df.drop(safe_cols_to_drop, axis=1)
    return df


def drop_duplicate_columns(df):
    cols = list(df.columns)
    for idx, item in enumerate(df.columns):
        if item in df.columns[:idx]:
            print('#####################################################')
            print('We found a duplicate column, and will be removing it')
            print('If you intended to send in two different pieces of information, please make sure they have different column names')
            print('Here is the duplicate column:')
            print(item)
            print('#####################################################')
            cols[idx] = "toDROP"
    df.columns = cols

    try:
        df = df.drop("toDROP", axis=1)
    except:
        pass
    return df


def get_boston_dataset():
    boston = load_boston()
    df_boston = pd.DataFrame(boston.data)
    df_boston.columns = boston.feature_names
    df_boston['MEDV'] = boston['target']
    df_boston_train, df_boston_test = train_test_split(df_boston, test_size=0.2, random_state=42)
    return df_boston_train, df_boston_test
