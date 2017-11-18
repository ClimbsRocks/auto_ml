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

def optimize_final_model_regression(model_name=None):
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

    # We just want to make sure these run, not necessarily make sure that they're super accurate (which takes more time, and is dataset dependent)
    df_boston_train = df_boston_train.sample(frac=0.5)

    column_descriptions = {
        'MEDV': 'output'
        , 'CHAS': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_boston_train, optimize_final_model=True, model_names=model_name)

    test_score = ml_predictor.score(df_boston_test, df_boston_test.MEDV)

    print('test_score')
    print(test_score)

    # the random seed gets a score of -3.21 on python 3.5
    # There's a ton of noise here, due to small sample sizes
    lower_bound = -3.4
    if model_name == 'DeepLearningRegressor':
        lower_bound = -24
    if model_name == 'LGBMRegressor':
        lower_bound = -9.5
    if model_name == 'GradientBoostingRegressor':
        lower_bound = -5.1
    if model_name == 'CatBoostRegressor':
        lower_bound = -4.5
    if model_name == 'XGBRegressor':
        lower_bound = -4.8

    assert lower_bound < test_score < -2.75



def getting_single_predictions_regression(model_name=None):
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

    column_descriptions = {
        'MEDV': 'output'
        , 'CHAS': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_boston_train, model_names=model_name)

    file_name = ml_predictor.save(str(random.random()))

    saved_ml_pipeline = load_ml_model(file_name)

    os.remove(file_name)
    try:
        keras_file_name = file_name[:-5] + '_keras_deep_learning_model.h5'
        os.remove(keras_file_name)
    except:
        pass


    df_boston_test_dictionaries = df_boston_test.to_dict('records')

    # 1. make sure the accuracy is the same

    predictions = []
    for row in df_boston_test_dictionaries:
        predictions.append(saved_ml_pipeline.predict(row))

    print('predictions')
    print(predictions)
    print('predictions[0]')
    print(predictions[0])
    print('type(predictions)')
    print(type(predictions))
    first_score = utils.calculate_rmse(df_boston_test.MEDV, predictions)
    print('first_score')
    print(first_score)
    # Make sure our score is good, but not unreasonably good

    lower_bound = -2.9
    if model_name == 'DeepLearningRegressor':
        lower_bound = -7.8
    if model_name == 'LGBMRegressor':
        lower_bound = -4.95
    if model_name == 'XGBRegressor':
        lower_bound = -3.4
    if model_name == 'CatBoostRegressor':
        lower_bound = -3.7

    assert lower_bound < first_score < -2.7

    # 2. make sure the speed is reasonable (do it a few extra times)
    data_length = len(df_boston_test_dictionaries)
    start_time = datetime.datetime.now()
    for idx in range(1000):
        row_num = idx % data_length
        saved_ml_pipeline.predict(df_boston_test_dictionaries[row_num])
    end_time = datetime.datetime.now()
    duration = end_time - start_time

    print('duration.total_seconds()')
    print(duration.total_seconds())

    # It's very difficult to set a benchmark for speed that will work across all machines.
    # On my 2013 bottom of the line 15" MacBook Pro, this runs in about 0.8 seconds for 1000 predictions
    # That's about 1 millisecond per prediction
    # Assuming we might be running on a test box that's pretty weak, multiply by 3
    # Also make sure we're not running unreasonably quickly
    assert 0.1 < duration.total_seconds() / 1.0 < 60


    # 3. make sure we're not modifying the dictionaries (the score is the same after running a few experiments as it is the first time)

    predictions = []
    for row in df_boston_test_dictionaries:
        predictions.append(saved_ml_pipeline.predict(row))

    second_score = utils.calculate_rmse(df_boston_test.MEDV, predictions)
    print('second_score')
    print(second_score)
    # Make sure our score is good, but not unreasonably good

    assert lower_bound < second_score < -2.7

