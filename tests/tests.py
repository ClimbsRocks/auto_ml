"""
To get standard out, run nosetests as follows:
  $ nosetests -sv tests
"""
import sys, os
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path

from nose.tools import assert_equal, assert_not_equal, with_setup

import dill

import utils_testing as utils


def test_binary_classification():
    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()
    ml_predictor = utils.train_basic_binary_classifier(df_titanic_train)

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived, verbose=0)
    # Right now we're getting a score of -.205
    # Make sure our score is good, but not unreasonably good
    assert -0.215 < test_score < -0.17




def test_regression():
    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()
    ml_predictor = utils.train_basic_regressor(df_boston_train)
    test_score = ml_predictor.score(df_boston_test, df_boston_test.MEDV, verbose=0)

    # Currently, we expect to get a score of -3.09
    # Make sure our score is good, but not unreasonably good
    assert -3.2 < test_score < -2.8


def test_saving_trained_pipeline_regression():
    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()
    ml_predictor = utils.train_basic_regressor(df_boston_train)
    file_name = ml_predictor.save()

    with open(file_name, 'rb') as read_file:
        saved_ml_pipeline = dill.load(read_file)

    test_score = saved_ml_pipeline.score(df_boston_test, df_boston_test.MEDV)
    # Make sure our score is good, but not unreasonably good
    assert -3.2 < test_score < -2.8


def test_saving_trained_pipeline_binary_classification():
    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()
    ml_predictor = utils.train_basic_binary_classifier(df_titanic_train)
    file_name = ml_predictor.save()

    with open(file_name, 'rb') as read_file:
        saved_ml_pipeline = dill.load(read_file)

    test_score = saved_ml_pipeline.score(df_titanic_test, df_titanic_test.survived)
    # Right now we're getting a score of -.205
    # Make sure our score is good, but not unreasonably good
    assert -0.215 < test_score < -0.17


# def test_getting_single_predictions_regression():
