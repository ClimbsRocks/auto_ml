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


def test_calibrate_uncertainty():
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

    column_descriptions = {
        'MEDV': 'output'
        , 'CHAS': 'categorical'
    }

    df_boston_train, uncertainty_data = train_test_split(df_boston_train, test_size=0.5)
    uncertainty_data, uncertainty_calibration_data = train_test_split(uncertainty_data, test_size=0.5)

    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    uncertainty_calibration_settings = {
        'num_buckets': 3
        , 'percentiles': [25, 50, 75]
    }
    ml_predictor.train(df_boston_train, perform_feature_selection=True, train_uncertainty_model=True, uncertainty_data=uncertainty_data, calibrate_uncertainty=True, uncertainty_calibration_settings=uncertainty_calibration_settings, uncertainty_calibration_data=uncertainty_calibration_data)

    uncertainty_score = ml_predictor.predict_uncertainty(df_boston_test)


    assert 'percentile_25_delta' in list(uncertainty_score.columns)
    assert 'percentile_50_delta' in list(uncertainty_score.columns)
    assert 'percentile_75_delta' in list(uncertainty_score.columns)
    assert 'bucket_num' in list(uncertainty_score.columns)


    # API:
    # calibrate_uncertainty=False
    # uncertainty_calibration_settings = {
    #    'num_buckets': 5
    #    , 'percentiles': [25, 50, 75]
    # }
    # uncertainty_calibration_data = None
    # methodology:
    # 1. get predictions on our uncertainty_calibration_data
    # 2. get actual deltas between true values, and predicted values for each row
        # we will need to make sure the uncertainty_calibration_data has a y column that contains the true values to our base regression problem
    # 3. divide our uc_data into "num_buckets" buckets based on the predicted_uncertainty percentages
        # "here is the group of deliveries we predicted 0-20% uncertainty for"
        # worth noting that the buckets will probably be based on the distribution of hte uc_data, rather than on fixed percentage intervals. so, for 5 buckets, we won't have 0-20, 20 - 40, etc., we will instead have "the lowest 20% of uncertainty predictions go from 0-1%, the next lowest 20% of buckets fall from 1-6%", etc.
    # 4. for each bucket, figure out what the actual deltas are at each percentile
        # so, for this lowest predicted 20% of the uc_data, their 13th percentile actual delta was...
        # maybe also include the rmse and std of this group
    # essentially, what we're getting is:
        # the model predicted a base value of A for the regression problem. THe uncertainty model predicted a probability of B that we will be off by some amount. For all rows where we predicted roughly B, what was the distribution by which we were actually off?

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

