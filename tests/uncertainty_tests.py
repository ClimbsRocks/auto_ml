import os
import sys
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path

from auto_ml import Predictor

import dill
from nose.tools import assert_equal, assert_not_equal, with_setup
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import utils_testing as utils


# Tests on regression models:

def test_predict_uncertainty_returns_pandas_DataFrame_for_more_than_one_value():
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

    column_descriptions = {
        'MEDV': 'output'
        , 'CHAS': 'categorical'
    }

    df_boston_train, uncertainty_data = train_test_split(df_boston_train, test_size=0.5)

    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_boston_train, perform_feature_selection=True, train_uncertainty_model=True, uncertainty_data=uncertainty_data)

    uncertainties = ml_predictor.predict_uncertainty(df_boston_test)

    assert isinstance(uncertainties, pd.DataFrame)


def test_predict_uncertainty_returns_dict_for_one_value():
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

    column_descriptions = {
        'MEDV': 'output'
        , 'CHAS': 'categorical'
    }

    df_boston_train, uncertainty_data = train_test_split(df_boston_train, test_size=0.5)

    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_boston_train, perform_feature_selection=True, train_uncertainty_model=True, uncertainty_data=uncertainty_data)

    test_list = df_boston_test.to_dict('records')

    for item in test_list:
        prediction = ml_predictor.predict_uncertainty(item)
        assert isinstance(prediction, dict)


def test_score_uncertainty():
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

    column_descriptions = {
        'MEDV': 'output'
        , 'CHAS': 'categorical'
    }

    df_boston_train, uncertainty_data = train_test_split(df_boston_train, test_size=0.5)

    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_boston_train, perform_feature_selection=True, train_uncertainty_model=True, uncertainty_data=uncertainty_data)

    uncertainty_score = ml_predictor.score_uncertainty(df_boston_test, df_boston_test.MEDV)

    print('uncertainty_score')
    print(uncertainty_score)

    assert uncertainty_score > -0.2



    # TODO:
    # Do we want to have a score_uncertainty function? all it would do is probably call .score on the underlying model (nested one level)
    # we definitely want to have a calibrate_uncertainty function



    # test_score = ml_predictor.score(df_boston_test, df_boston_test.MEDV)

    # print('test_score')
    # print(test_score)

    # # Bumping this up since without these features our score drops
    # lower_bound = -4.0
    # if model_name == 'DeepLearningRegressor':
    #     lower_bound = -14.5
    # if model_name == 'LGBMRegressor':
    #     lower_bound = -4.95


    # assert lower_bound < test_score < -2.8

    # ml_predictor.get_uncertainty_prediction(df_boston_test)

