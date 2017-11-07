import os
import sys
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path

os.environ['is_test_suite'] = 'True'

from auto_ml import Predictor

import dill
from nose.tools import assert_equal, assert_not_equal, with_setup
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import utils_testing as utils



def test_predict_uncertainty_true():
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

    column_descriptions = {
        'MEDV': 'output'
        , 'CHAS': 'categorical'
    }

    df_boston_train, uncertainty_data = train_test_split(df_boston_train, test_size=0.5)

    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_boston_train, predict_intervals=True)

    intervals = ml_predictor.predict_intervals(df_boston_test)

    assert isinstance(intervals, pd.DataFrame)
    assert intervals.shape[0] == df_boston_test.shape[0]

    result_list = ml_predictor.predict_intervals(df_boston_test, return_type='list')

    assert isinstance(result_list, list)
    assert len(result_list) == df_boston_test.shape[0]
    for idx, row in enumerate(result_list):
        assert isinstance(row, list)
        assert len(row) == 3

    singles = df_boston_test.head().to_dict('records')

    for row in singles:
        result = ml_predictor.predict_intervals(row)
        assert isinstance(result, dict)
        assert 'prediction' in result
        assert 'interval_0.05' in result
        assert 'interval_0.95' in result

    for row in singles:
        result = ml_predictor.predict_intervals(row, return_type='list')
        assert isinstance(result, list)
        assert len(result) == 3

    df_intervals = ml_predictor.predict_intervals(df_boston_test, return_type='df')
    assert isinstance(df_intervals, pd.DataFrame)

    try:
        ml_predictor.predict_intervals(df_boston_test, return_type='this will not work')
        assert False
    except ValueError:
        assert True


def test_predict_intervals_takes_in_custom_intervals():
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

    column_descriptions = {
        'MEDV': 'output'
        , 'CHAS': 'categorical'
    }

    df_boston_train, uncertainty_data = train_test_split(df_boston_train, test_size=0.5)

    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_boston_train, predict_intervals=[0.4, 0.6])

    intervals = ml_predictor.predict_intervals(df_boston_test, return_type='list')

    assert isinstance(intervals, list)

    singles = df_boston_test.head().to_dict('records')

    acceptable_keys = set(['prediction', 'interval_0.4', 'interval_0.6'])
    for row in singles:
        result = ml_predictor.predict_intervals(row)
        assert isinstance(result, dict)
        assert 'prediction' in result
        assert 'interval_0.4' in result
        assert 'interval_0.6' in result
        # print(result)
        # print(type(result))
        # print(result.keys())
        for key in result.keys():
            assert key in acceptable_keys

    for row in singles:
        result = ml_predictor.predict_intervals(row, return_type='list')
        assert isinstance(result, list)
        assert len(result) == 3

    df_intervals = ml_predictor.predict_intervals(df_boston_test, return_type='df')
    assert df_intervals.shape[0] == df_boston_test.shape[0]
    assert isinstance(df_intervals, pd.DataFrame)


    # Now make sure that the interval values are actually different
    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_boston_train, predict_intervals=True)

    default_intervals = ml_predictor.predict_intervals(df_boston_test, return_type='list')

    # This is a super flaky test, because we've got such a small datasize, and we're trying to get distributions from it
    num_failures = 0
    for idx, row in enumerate(intervals):
        default_row = default_intervals[idx]

        if int(row[1]) <= int(default_row[1]):
            num_failures += 1
        if int(row[2]) >= int(default_row[2]):
            num_failures += 1

    len_intervals = len(intervals)
    assert num_failures < 0.25 * len_intervals


def test_prediction_intervals_actually_work():
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

    column_descriptions = {
        'MEDV': 'output'
        , 'CHAS': 'categorical'
    }

    df_boston_train, uncertainty_data = train_test_split(df_boston_train, test_size=0.5)

    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_boston_train, predict_intervals=True)

    intervals = ml_predictor.predict_intervals(df_boston_test)

    count_under = 0
    count_over = 0
    for row in intervals:
        if row[0] < row[1]:
            count_under += 1
        if row[0] > row[3]:
            count_over += 1

    len_intervals = len(intervals)

    pct_under = count_under * 1.0 / len_intervals
    pct_over = count_over * 1.0 / len_intervals
    # There's a decent bit of noise since this is such a small dataset
    assert pct_under < 0.1
    assert pct_over < 0.1


def test_prediction_intervals_lets_the_user_specify_number_of_intervals():
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

    column_descriptions = {
        'MEDV': 'output'
        , 'CHAS': 'categorical'
    }

    df_boston_train, uncertainty_data = train_test_split(df_boston_train, test_size=0.5)

    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_boston_train, predict_intervals=True, prediction_intervals=[.2])

    intervals = ml_predictor.predict_intervals(df_boston_test, return_type='list')

    assert len(intervals[0]) == 2


def test_predict_intervals_should_fail_if_not_trained():
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

    column_descriptions = {
        'MEDV': 'output'
        , 'CHAS': 'categorical'
    }

    df_boston_train, uncertainty_data = train_test_split(df_boston_train, test_size=0.5)

    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_boston_train)

    try:
        intervals = ml_predictor.predict_intervals(df_boston_test)
        assert False
    except ValueError:
        assert True


