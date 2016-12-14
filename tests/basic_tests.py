"""
To get standard out, run nosetests as follows:
  $ nosetests -s tests
"""
import datetime
import os
import sys
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path

from auto_ml import Predictor

from nose.tools import assert_equal, assert_not_equal, with_setup

import dill
import numpy as np
import utils_testing as utils

def test_binary_classification():
    np.random.seed(0)


    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()
    ml_predictor = utils.train_basic_binary_classifier(df_titanic_train)

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived, verbose=0)
    # Right now we're getting a score of -.205
    # Make sure our score is good, but not unreasonably good
    assert -0.215 < test_score < -0.17



def test_multilabel_classification():
    np.random.seed(0)

    df_twitter_train, df_twitter_test = utils.get_twitter_sentiment_multilabel_classification_dataset()
    ml_predictor = utils.train_basic_multilabel_classifier(df_twitter_train)

    test_score = ml_predictor.score(df_twitter_test, df_twitter_test.airline_sentiment, verbose=0)
    # Right now we're getting a score of -.205
    # Make sure our score is good, but not unreasonably good
    print('test_score')
    print(test_score)
    assert 0.67 < test_score < 0.79


def test_nlp_multilabel_classification():
    np.random.seed(0)

    df_twitter_train, df_twitter_test = utils.get_twitter_sentiment_multilabel_classification_dataset()

    column_descriptions = {
        'airline_sentiment': 'output'
        , 'airline': 'categorical'
        , 'text': 'nlp'
        , 'tweet_location': 'categorical'
        , 'user_timezone': 'categorical'
        , 'tweet_created': 'date'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)
    ml_predictor.train(df_twitter_train)

    test_score = ml_predictor.score(df_twitter_test, df_twitter_test.airline_sentiment, verbose=0)
    # Right now we're getting a score of -.205
    # Make sure our score is good, but not unreasonably good
    print('test_score')
    print(test_score)
    assert 0.67 < test_score < 0.75




def test_regression():
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()
    ml_predictor = utils.train_basic_regressor(df_boston_train)
    test_score = ml_predictor.score(df_boston_test, df_boston_test.MEDV, verbose=0)

    # Currently, we expect to get a score of -3.09
    # Make sure our score is good, but not unreasonably good
    assert -3.2 < test_score < -2.8


def test_saving_trained_pipeline_regression():
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()
    ml_predictor = utils.train_basic_regressor(df_boston_train)
    file_name = ml_predictor.save()

    with open(file_name, 'rb') as read_file:
        saved_ml_pipeline = dill.load(read_file)

    test_score = saved_ml_pipeline.score(df_boston_test, df_boston_test.MEDV)
    # Make sure our score is good, but not unreasonably good
    assert -3.2 < test_score < -2.8


def test_saving_trained_pipeline_binary_classification():
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()
    ml_predictor = utils.train_basic_binary_classifier(df_titanic_train)
    file_name = ml_predictor.save()

    with open(file_name, 'rb') as read_file:
        saved_ml_pipeline = dill.load(read_file)

    test_score = saved_ml_pipeline.score(df_titanic_test, df_titanic_test.survived)
    # Right now we're getting a score of -.205
    assert -0.215 < test_score < -0.17

def test_getting_single_predictions_regression():
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()
    ml_predictor = utils.train_basic_regressor(df_boston_train)
    file_name = ml_predictor.save()

    with open(file_name, 'rb') as read_file:
        saved_ml_pipeline = dill.load(read_file)

    df_boston_test_dictionaries = df_boston_test.to_dict('records')

    # 1. make sure the accuracy is the same

    predictions = []
    for row in df_boston_test_dictionaries:
        predictions.append(saved_ml_pipeline.predict(row))

    first_score = utils.calculate_rmse(df_boston_test.MEDV, predictions)
    print('first_score')
    print(first_score)
    # Make sure our score is good, but not unreasonably good
    assert -3.2 < first_score < -2.8

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
    assert 0.2 < duration.total_seconds() / 1.0 < 3


    # 3. make sure we're not modifying the dictionaries (the score is the same after running a few experiments as it is the first time)

    predictions = []
    for row in df_boston_test_dictionaries:
        predictions.append(saved_ml_pipeline.predict(row))

    second_score = utils.calculate_rmse(df_boston_test.MEDV, predictions)
    print('second_score')
    print(second_score)
    # Make sure our score is good, but not unreasonably good
    assert -3.2 < second_score < -2.8




def test_getting_single_predictions_classification():
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()
    ml_predictor = utils.train_basic_binary_classifier(df_titanic_train)
    file_name = ml_predictor.save()

    with open(file_name, 'rb') as read_file:
        saved_ml_pipeline = dill.load(read_file)

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
    assert -0.215 < first_score < -0.17

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
    assert 0.2 < duration.total_seconds() < 3


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
    assert -0.215 < second_score < -0.17




