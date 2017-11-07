# This file is just to test passing a bunch of different parameters into train to make sure that things work
# At first, it is not necessarily testing whether those things have the intended effect or not

import datetime
import os
import random
import sys
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path

from auto_ml import Predictor
from auto_ml.utils_models import load_ml_model

import dill
from nose.tools import assert_equal, assert_not_equal, with_setup
import numpy as np
from sklearn.model_selection import train_test_split

import utils_testing as utils


# Tests on regression models:

def test_perform_feature_selection_true_regression(model_name=None):
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

    column_descriptions = {
        'MEDV': 'output'
        , 'CHAS': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_boston_train, perform_feature_selection=True, model_names=model_name)

    test_score = ml_predictor.score(df_boston_test, df_boston_test.MEDV)

    print('test_score')
    print(test_score)

    # Bumping this up since without these features our score drops
    lower_bound = -4.0
    if model_name == 'DeepLearningRegressor':
        lower_bound = -14.5
    if model_name == 'LGBMRegressor':
        lower_bound = -4.95


    assert lower_bound < test_score < -2.8


def test_perform_feature_selection_false_regression(model_name=None):
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

    column_descriptions = {
        'MEDV': 'output'
        , 'CHAS': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_boston_train, perform_feature_selection=False, model_names=model_name)

    test_score = ml_predictor.score(df_boston_test, df_boston_test.MEDV)

    print('test_score')
    print(test_score)

    lower_bound = -3.0

    assert lower_bound < test_score < -2.7

def test_perform_feature_scaling_true_regression(model_name=None):
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

    column_descriptions = {
        'MEDV': 'output'
        , 'CHAS': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_boston_train, perform_feature_scaling=True)

    test_score = ml_predictor.score(df_boston_test, df_boston_test.MEDV)

    print('test_score')
    print(test_score)

    assert -3.0 < test_score < -2.7

def test_perform_feature_scaling_false_regression(model_name=None):
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

    column_descriptions = {
        'MEDV': 'output'
        , 'CHAS': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_boston_train, perform_feature_scaling=False, model_names=model_name)

    test_score = ml_predictor.score(df_boston_test, df_boston_test.MEDV)

    print('test_score')
    print(test_score)

    lower_bound = -3.0

    assert lower_bound < test_score < -2.7

def test_compare_all_models_regression():
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

    column_descriptions = {
        'MEDV': 'output'
        , 'CHAS': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_boston_train, compare_all_models=True)

    test_score = ml_predictor.score(df_boston_test, df_boston_test.MEDV)

    print('test_score')
    print(test_score)

    # ExtraTrees again throws this off
    assert -3.6 < test_score < -2.8


