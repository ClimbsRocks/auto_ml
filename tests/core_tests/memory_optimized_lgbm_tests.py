import datetime
import os
import random
import sys
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path
sys.path = [os.path.abspath(os.path.dirname(os.path.dirname(__file__)))] + sys.path

os.environ['is_test_suite'] = 'True'

from auto_ml import Predictor
from auto_ml.utils_models import load_ml_model

import dill
from nose.tools import assert_equal, assert_not_equal, with_setup
import numpy as np
from sklearn.model_selection import train_test_split

import utils_testing as utils


# Tests on regression models:

def test_memory_optimized_lgbm_regression(model_name=None):
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

    column_descriptions = {
        'MEDV': 'output'
        , 'CHAS': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_boston_train, perform_feature_selection=False, lgbm_memory_optimized=True, model_names='LGBMRegressor')

    test_score = ml_predictor.score(df_boston_test, df_boston_test.MEDV)

    print('test_score')
    print(test_score)

    # Bumping this up since without these features our score drops
    lower_bound = -3.5
    print('test_score')
    print(test_score)

    assert lower_bound < test_score < -2.8


def test_memory_optimized_lgbm_classification():
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output'
        , 'sex': 'categorical'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    ml_predictor.train(df_titanic_train, perform_feature_selection=False, lgbm_memory_optimized=True, model_names='LGBMClassifier')

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

    print('test_score')
    print(test_score)

    # We are strongly overfitting here. not sure why, as I thought we were using the same params as the sklearn model, but also not too worried about it, since lgbm's kinda known to overfit to small datasets like this
    assert -0.16 < test_score < -0.135




def test_getting_single_predictions_classification():
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output'
        , 'sex': 'categorical'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    ml_predictor.train(df_titanic_train, perform_feature_selection=False, lgbm_memory_optimized=True, model_names='LGBMClassifier')

    file_name = ml_predictor.save(str(random.random()))

    saved_ml_pipeline = load_ml_model(file_name)

    df_titanic_test_dictionaries = df_titanic_test.to_dict('records')

    # 1. make sure the accuracy is the same

    predictions = []
    for row in df_titanic_test_dictionaries:
        predictions.append(saved_ml_pipeline.predict_proba(row)[1])

    print('predictions')
    print(predictions)

    first_score = utils.calculate_brier_score_loss(df_titanic_test.survived, predictions)
    print('first_score')
    print(first_score)
    # Make sure our score is good, but not unreasonably good

    lower_bound = -0.6
    upper_bound = -0.135

    assert lower_bound < first_score < upper_bound

    # 2. make sure the speed is reasonable (do it a few extra times)
    data_length = len(df_titanic_test_dictionaries)
    start_time = datetime.datetime.now()
    for idx in range(1000):
        row_num = idx % data_length
        saved_ml_pipeline.predict(df_titanic_test_dictionaries[row_num])
    end_time = datetime.datetime.now()
    duration = end_time - start_time

    print('duration.total_seconds()')
    print(duration.total_seconds())

    # It's very difficult to set a benchmark for speed that will work across all machines.
    # On my 2013 bottom of the line 15" MacBook Pro, this runs in about 0.8 seconds for 1000 predictions
    # That's about 1 millisecond per prediction
    # Assuming we might be running on a test box that's pretty weak, multiply by 3
    # Also make sure we're not running unreasonably quickly
    assert 0.2 < duration.total_seconds() < 60


    # 3. make sure we're not modifying the dictionaries (the score is the same after running a few experiments as it is the first time)

    predictions = []
    for row in df_titanic_test_dictionaries:
        predictions.append(saved_ml_pipeline.predict_proba(row)[1])

    print('predictions')
    print(predictions)
    print('df_titanic_test_dictionaries')
    print(df_titanic_test_dictionaries)
    second_score = utils.calculate_brier_score_loss(df_titanic_test.survived, predictions)
    print('second_score')
    print(second_score)
    # Make sure our score is good, but not unreasonably good

    assert lower_bound < second_score < upper_bound


